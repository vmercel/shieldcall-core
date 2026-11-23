"""FreeSWITCH mod_audio_fork adapter (TDD v2.0 section 4, Media Ingest Fabric).

Translates FreeSWITCH's mod_audio_fork WebSocket protocol into the
canonical 10 ms frame envelope. Pure and transport-agnostic: it
takes raw text frames and raw binary audio and returns frames. It
opens no socket, holds no credential, and knows nothing about
HTTP. The serving layer owns auth and the websocket; see
shieldcall.serve.http_app for the /v1/ingest/freeswitch route
that drives this class.

Protocol (mod_audio_fork README, andrewvmail/freeswitch-modules,
read 2026-11-23; corroborated by AMDY's FreeSWITCH integration
guide and a mod_audio_fork streaming guide, read the same day):
- The operator starts a fork with `uuid_audio_fork <uuid> start
  <wss-url> <mix-type> <sampling-rate> <streamID> <accID>
  <callSid> <track> <metadata>`. Mix types are `mono` (one
  channel, the caller's audio), `mixed` (one channel, both legs
  summed) and `stereo` (two channels, caller audio in one,
  callee audio in the other). Sampling rates are `8k` and
  `16k`. The module resamples the leg to the requested rate.
- The first WebSocket frame is a text frame carrying the
  operator's `metadata`, verbatim: arbitrary data the operator
  chose, JSON in practice (the README's example passes caller
  data as JSON; AMDY passes a config object). No other start
  message exists: there is no handshake declaring the format.
- Binary frames that follow are raw L16: 16-bit signed
  little-endian PCM, no header, no container, in 20 ms chunks.
  A stereo packet interleaves the two channels sample by
  sample.
- `uuid_audio_fork <uuid> send_text <metadata>` sends a text
  frame mid-call (for example a DTMF notification). The module
  defines no shape for these frames.
- `uuid_audio_fork <uuid> stop <metadata>` sends a final text
  frame and then closes the connection. There is no stop event
  distinct from any other text frame; the end of the stream is
  the socket closing.
- JSON text frames from the server to the module, with `event`
  set to `audio`, `killAudio`, `transcription`, `transfer`,
  `disconnect` or `error`, are control events. This adapter
  never sends them; the route sends nothing back at all
  (ADR-014 item 10).

What the protocol does NOT provide, and this adapter does not
pretend otherwise: no stream identifier, call identifier,
sequence number or timestamp defined by the module itself, and
no declared format except what the operator wrote into the
metadata. The format is therefore pinned from the first text
frame under the contract ADR-014 item 4 defines: the metadata
must be a JSON object declaring `sample_rate` and `mix_type`.
Without that declaration the adapter refuses the stream rather
than guessing, because guessing 8 kHz mono would silently
mis-decode every other configuration. A lost packet is
indistinguishable from unsent silence, so no gap accounting is
possible at all: `discontinuous` is always False, and
presentation time is counted from the samples received.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from .frame import FRAME_SAMPLES_8K, MediaFrame

__all__ = ["FreeSwitchError", "FreeSwitchMediaStream"]

SAMPLE_RATE = 8000
"""The only sample rate this adapter accepts (see ADR-014)."""

MIX_MONO = "mono"
"""One channel: the forked leg's audio, analysed as inbound."""

MIX_STEREO = "stereo"
"""Two interleaved channels; channel 0 is analysed, channel 1 counted."""

MIX_MIXED = "mixed"
"""One summed channel; refused, never scored (see ADR-014 item 5)."""

MIX_TYPES = (MIX_MONO, MIX_STEREO, MIX_MIXED)

TRACK_INBOUND = "inbound"
"""The analysed leg: the forked leg's audio, the leg ShieldCall scores."""

# The README says stereo carries caller audio in one channel and
# callee audio in the other without stating the order; a deployment
# write-up reports channel 0 as the caller. Channel 0 as inbound
# is taken from that report and flagged unverified in ADR-014
# item 6: the first live fork must confirm it.
STEREO_ANALYSED_CHANNEL = 0


class FreeSwitchError(ValueError):
    """A FreeSWITCH protocol violation: malformed metadata, wrong
    state, or an unsupported media format. The caller should close
    the stream; the adapter never guesses its way past one."""


class FreeSwitchMediaStream:
    """One mod_audio_fork WebSocket stream, from the metadata text
    frame to the socket closing.

    - call_id: the pre-opened ShieldCall call id. The operator
      opens the call with POST /v1/calls and puts the same id in
      the websocket URI as a query parameter, because the module
      sends no call identifier of its own. When the metadata
      carries `call_id` it is checked against this binding
      (ADR-014 item 9); when it does not, the binding rests on
      the URI alone, and the route owns it.
    """

    def __init__(self, call_id: str):
        if not call_id:
            raise FreeSwitchError("call_id is required")
        self.call_id = call_id
        self.started = False
        self.finished = False
        # The operator's metadata object, recorded for audit.
        # Extra fields are kept and ignored; only sample_rate,
        # mix_type, call_id and stream_id are interpreted.
        self.metadata: Dict[str, Any] = {}
        self.sample_rate: Optional[int] = None
        self.mix_type: Optional[str] = None
        self.stream_sid: str = ""
        # Text frames after the metadata frame, recorded verbatim.
        # They are operator data (DTMF notifications, the final
        # frame a stop command sends); the module defines no shape
        # for them, so they are never parsed as protocol.
        self.text_events: List[str] = []
        self._buffer = np.empty(0, dtype=np.float32)
        # Samples emitted as frames so far; the presentation clock,
        # counted from received samples because the module sends
        # no clock of its own.
        self._emitted_samples = 0
        self._frames = 0
        self._stats: Dict[str, int] = {
            "messages": 0,
            "text_events": 0,
            "audio_packets": 0,
            "audio_bytes": 0,
            "samples": 0,
            "samples_other_track": 0,
            "frames": 0,
            "discarded_on_close": 0,
        }

    # -- public ---------------------------------------------------------

    @property
    def is_started(self) -> bool:
        return self.started

    @property
    def is_finished(self) -> bool:
        return self.finished

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def handle_text(self, text: str) -> List[MediaFrame]:
        """Handle one raw text WebSocket frame. The first frame must
        be the metadata object that pins the format (ADR-014 item
        4); later frames are recorded verbatim and produce no
        frames. Raises FreeSwitchError on any protocol violation."""
        if self.finished:
            raise FreeSwitchError("text frame received after the stream finished")
        if not isinstance(text, str):
            raise FreeSwitchError("text frame must be a string")
        self._stats["messages"] += 1
        if not self.started:
            self._on_metadata(text)
            return []
        # A later text frame never re-pins the format, whatever
        # JSON it happens to contain.
        self.text_events.append(text)
        self._stats["text_events"] += 1
        return []

    def handle_audio(self, data: bytes) -> List[MediaFrame]:
        """Translate one binary WebSocket message (raw L16 PCM) into
        canonical frames. Raises FreeSwitchError on any protocol
        violation."""
        if self.finished:
            raise FreeSwitchError("audio received after the stream finished")
        if not self.started:
            raise FreeSwitchError("audio received before the metadata frame")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise FreeSwitchError("audio must be bytes")
        raw = bytes(data)
        if not raw:
            raise FreeSwitchError("empty audio packet")
        if self.mix_type == MIX_STEREO:
            # One interleaved pair is two 16-bit samples: a packet
            # whose byte count is not a multiple of four ends in a
            # truncated pair, not a whole instant of audio.
            if len(raw) % 4:
                raise FreeSwitchError(
                    f"stereo audio packet has byte count {len(raw)}, "
                    "not a multiple of 4: a truncated interleaved pair"
                )
            pairs = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
            analysed = pairs[:, STEREO_ANALYSED_CHANNEL]
            other = pairs[:, 1 - STEREO_ANALYSED_CHANNEL]
            self._stats["samples_other_track"] += int(other.size)
        else:
            if len(raw) % 2:
                raise FreeSwitchError(
                    f"audio packet has odd byte count {len(raw)}: "
                    "a truncated 16-bit sample"
                )
            analysed = np.frombuffer(raw, dtype="<i2")
        self._stats["audio_packets"] += 1
        self._stats["audio_bytes"] += len(raw)
        # 16-bit signed little-endian PCM, scaled to [-1, 1]. The
        # divisor is 32768 so that -32768 maps to exactly -1.0.
        samples = analysed.astype(np.float32) / 32768.0
        self._stats["samples"] += int(samples.size)
        self._buffer = (
            np.concatenate([self._buffer, samples])
            if self._buffer.size
            else samples
        )
        return self._drain_frames()

    def finish(self) -> None:
        """End the stream when the socket closes. The module sends
        no stop event, so this is route-side cleanup, and it is
        idempotent: a second call must not raise. A partial
        trailing frame is discarded, not padded: padding would
        invent audio the caller never sent. The count is kept in
        stats so a short final packet is visible, not silent."""
        if self.finished:
            return
        self._stats["discarded_on_close"] += int(self._buffer.size)
        self._buffer = np.empty(0, dtype=np.float32)
        self.finished = True

    # -- metadata -------------------------------------------------------

    def _on_metadata(self, text: str) -> None:
        if not text.strip():
            raise FreeSwitchError(
                "metadata frame is empty: the first text frame must "
                "be a JSON object declaring sample_rate and mix_type "
                "(see ADR-014 item 4)"
            )
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            raise FreeSwitchError(
                f"metadata frame is not JSON: {exc}; the first text "
                "frame must be a JSON object declaring sample_rate "
                "and mix_type (see ADR-014 item 4)"
            ) from exc
        if not isinstance(parsed, dict):
            raise FreeSwitchError(
                "metadata frame must be a JSON object declaring "
                "sample_rate and mix_type (see ADR-014 item 4)"
            )
        sample_rate = self._parse_sample_rate(parsed.get("sample_rate"))
        mix_type = self._parse_mix_type(parsed.get("mix_type"))
        declared_call_id = parsed.get("call_id")
        if declared_call_id is not None:
            if not isinstance(declared_call_id, str) or not declared_call_id:
                raise FreeSwitchError(
                    "metadata call_id must be a non-empty string"
                )
            if declared_call_id != self.call_id:
                raise FreeSwitchError(
                    "metadata call_id does not match the pre-opened "
                    "call: refusing a misrouted fork"
                )
        stream_id = parsed.get("stream_id")
        if stream_id is not None and not isinstance(stream_id, str):
            raise FreeSwitchError("metadata stream_id must be a string")
        self.metadata = parsed
        self.sample_rate = sample_rate
        self.mix_type = mix_type
        self.stream_sid = stream_id or ""
        self.started = True

    # -- helpers --------------------------------------------------------

    def _drain_frames(self) -> List[MediaFrame]:
        frames: List[MediaFrame] = []
        while self._buffer.size >= FRAME_SAMPLES_8K:
            chunk_samples = self._buffer[:FRAME_SAMPLES_8K]
            self._buffer = self._buffer[FRAME_SAMPLES_8K:]
            frames.append(
                MediaFrame(
                    call_id=self.call_id,
                    stream_sid=self.stream_sid,
                    track=TRACK_INBOUND,
                    seq=self._frames,
                    # No clock on the wire: discontinuous is always
                    # False (ADR-014 item 7). Loss is invisible to
                    # this adapter and must not be read into it.
                    t_sec=self._emitted_samples / SAMPLE_RATE,
                    samples=chunk_samples.copy(),
                    sample_rate=SAMPLE_RATE,
                    discontinuous=False,
                    source="freeswitch",
                )
            )
            self._frames += 1
            self._stats["frames"] += 1
            self._emitted_samples += FRAME_SAMPLES_8K
        return frames

    @staticmethod
    def _parse_sample_rate(value: Any) -> int:
        """The fork's sample rate, from the metadata declaration.
        Accepts 8000, "8000" and "8k", the spellings the start
        command and deployment configs use. 16 kHz is refused:
        resampling to 8 kHz needs a tested decimator (ADR-014
        follow-ups), not a silent slice of every second sample."""
        if value is None:
            raise FreeSwitchError(
                "metadata has no sample_rate: declare 8000 "
                "(see ADR-014 item 4)"
            )
        if isinstance(value, bool):
            raise FreeSwitchError(f"sample_rate {value!r} is not a sample rate")
        if isinstance(value, (int, float)):
            rate = value
        elif isinstance(value, str):
            text = value.strip().lower()
            if text.endswith("k"):
                try:
                    rate = int(text[:-1]) * 1000
                except ValueError as exc:
                    raise FreeSwitchError(
                        f"sample_rate {value!r} is not a sample rate"
                    ) from exc
            else:
                try:
                    rate = int(text)
                except ValueError as exc:
                    raise FreeSwitchError(
                        f"sample_rate {value!r} is not a sample rate"
                    ) from exc
        else:
            raise FreeSwitchError(f"sample_rate {value!r} is not a sample rate")
        if rate != int(rate):
            raise FreeSwitchError(f"sample_rate {value!r} is not a whole rate")
        rate = int(rate)
        if rate != SAMPLE_RATE:
            raise FreeSwitchError(
                f"unsupported FreeSWITCH sample rate {rate} "
                f"(supported: {SAMPLE_RATE}; start the fork at 8k)"
            )
        return rate

    @staticmethod
    def _parse_mix_type(value: Any) -> str:
        """The fork's mix type, from the metadata declaration.
        `mixed` is refused: a summed channel cannot be split into
        legs, and scoring it would score the protected user's own
        voice (ADR-014 item 5)."""
        if value is None:
            raise FreeSwitchError(
                "metadata has no mix_type: declare \"mono\" or "
                "\"stereo\" (see ADR-014 item 4)"
            )
        if not isinstance(value, str):
            raise FreeSwitchError(f"mix_type {value!r} is not a string")
        mix_type = value.strip().lower()
        if mix_type == MIX_MIXED:
            raise FreeSwitchError(
                "mixed forks are refused: one summed channel cannot "
                "be split into legs (start the fork mono or stereo; "
                "see ADR-014 item 5)"
            )
        if mix_type not in (MIX_MONO, MIX_STEREO):
            raise FreeSwitchError(
                f"unknown mix_type {value!r} (supported: mono, stereo; "
                "mixed is refused, see ADR-014 item 5)"
            )
        return mix_type
