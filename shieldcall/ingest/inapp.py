"""In-app VoIP/WebRTC adapter (TDD v2.0 section 4, Media Ingest Fabric).

Translates the app-originated media fork into the canonical 10 ms
frame envelope. Pure and transport-agnostic: it takes parsed JSON
messages and returns frames. It opens no socket, holds no
credential, and knows nothing about HTTP. The serving layer owns
auth and the websocket; see shieldcall.serve.http_app for the
/v1/ingest/inapp route that drives this class.

Protocol (defined by this ADR, ADR-015 item 3; there is no vendor
spec to follow, because the sender is ShieldCall's own app, and
no app client exists yet -- see ADR-015 item 11):

- hello, first, JSON: {"type": "hello", "codec": "opus",
    "sample_rate": 8000|16000|24000|48000, "channels": 1,
    "track": "far-end"|"near-end", "stream_id": "...",
    "call_id": "...", "consent_ref": "..."}
  The hello pins the format before any audio is decoded. The
  app's Encoded Transform sees RTCEncodedAudioFrame values
  (timestamp + data, W3C WebRTC Encoded Transform); the
  transform API exposes no RTP sequence number, so the fork
  assigns `seq` itself, in send order, and carries the frame
  timestamp as `timestamp` on the 48 kHz Opus RTP clock
  (RFC 7587: the Opus RTP clock is always 48 kHz, whatever
  rate was encoded).
- media, JSON: {"type": "media", "seq": 0..65535,
    "timestamp": 0..2**32-1, "payload": "<base64 Opus packet>"}
  One Opus packet per message. `seq` starts at 0 on every fork
  and increments per packet in send order; the first packet of
  a stream is seq 0. Binary WebSocket frames carry
  no seq or timestamp and are refused by the route, which
  reads JSON only.
- stop, JSON: {"type": "stop"}. The socket closing ends the
  stream the same way; finish() is idempotent.

What the protocol does NOT provide, and this adapter does not
pretend otherwise: no WebRTC peer connection, no RTP/SRTP/DTLS
(the fork is the app's own WebSocket channel, not a peer
connection); no consent-ledger validation (consent_ref is
recorded, never validated -- no ledger exists yet, ADR-015
item 8); no packet-loss concealment (a gap is flagged
discontinuous on the next real frame, never filled with
synthesised audio, ADR-015 item 7).
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .frame import FRAME_MS, FRAME_SAMPLES_8K, MediaFrame
from .opus import OpusDecoder, OpusDecoderUnavailable, OpusError

__all__ = ["InAppError", "InAppMediaStream"]

OUTPUT_RATE = 8000
"""Frames are emitted at 8 kHz: the decoder is created at 8 kHz,
so libopus does the rate conversion and no resampler runs here."""

RTP_CLOCK = 48000
"""Opus RTP timestamps always run at 48 kHz (RFC 7587)."""

REORDER_WINDOW_TICKS = 2880
"""60 ms of reorder window, in RTP ticks (TDD section 4.1)."""

ENCODED_RATES = (8000, 16000, 24000, 48000)
"""Encoded rates the hello may declare (TDD fabric table: Opus 8 to 48 kHz)."""

CODEC_OPUS = "opus"

TRACK_FAR_END = "far-end"
"""The remote party's voice: the only leg scored acoustically."""

TRACK_NEAR_END = "near-end"
"""The protected user's own voice: counted, never decoded or framed."""

TRACKS = (TRACK_FAR_END, TRACK_NEAR_END)

_MAX_SEQ_JUMP = 512
"""A forward sequence jump larger than this, with no hello in
between, is a new fork wearing an old stream's clothes: refuse
it rather than re-sync silently."""

_MAX_PACKET_BYTES = 4096
"""Generous cap on one Opus packet, so garbage cannot grow the
decoder's input without bound. A real Opus packet is far smaller."""

_SEQ_MODULUS = 65536
_TIMESTAMP_MODULUS = 1 << 32


class InAppError(ValueError):
    """An in-app fork protocol violation: malformed message, wrong
    state, or an unsupported media format. The caller should close
    the stream; the adapter never guesses its way past one."""


@dataclass
class _PendingPacket:
    seq: int
    timestamp: int  # unwrapped, 48 kHz ticks from the hello's base
    samples: np.ndarray


class InAppMediaStream:
    """One in-app media fork, from `hello` to `stop` / socket close.

    - call_id: the pre-opened ShieldCall call id. The app opens
      the call with POST /v1/calls and puts the same id in the
      websocket URI; a hello `call_id` that disagrees is a
      protocol error, not something to re-route.
    - decoder: an object with decode(bytes) -> float32 samples at
      8 kHz and an optional close(). Supplied by tests; when
      omitted, a real OpusDecoder is built at hello time, so a
      host with no libopus fails closed at hello, not mid-call.
      The stream takes ownership of the decoder and closes it
      on finish/stop: one decoder per stream, never shared.
    """

    def __init__(self, call_id: str, decoder: Any = None):
        if not call_id:
            raise InAppError("call_id is required")
        self.call_id = call_id
        self.stream_sid = ""
        self.started = False
        self.finished = False
        self.hello: Dict[str, Any] = {}
        self.consent_ref: Optional[str] = None
        self.track: Optional[str] = None
        self.sample_rate: Optional[int] = None
        self._decoder = decoder
        # Sequence numbers start at 0 on every fork (ADR-015 item 3):
        # the first packet of a stream is seq 0, so an out-of-order
        # first packet waits in the reorder buffer for it instead
        # of being mistaken for the start of the stream.
        self._expected_seq: int = 0
        self._unwrap_ref: Optional[int] = None
        self._base_unwrapped: Optional[int] = None
        self._pending: Dict[int, _PendingPacket] = {}
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_t_sec: Optional[float] = None
        self._last_frame_end_t_sec: Optional[float] = None
        self._gap_before_next_frame = False
        self._frames = 0
        self._emitted_samples = 0
        self._stats: Dict[str, int] = {
            "messages": 0,
            "media_messages": 0,
            "near_end_messages": 0,
            "packets": 0,
            "bytes": 0,
            "frames": 0,
            "samples": 0,
            "gaps": 0,
            "lost_packets": 0,
            "lost_samples": 0,
            "duplicates_dropped": 0,
            "late_dropped": 0,
            "timestamp_jumps": 0,
            "discarded_on_gap": 0,
            "discarded_on_stop": 0,
            "discarded_on_close": 0,
            "pending_discarded": 0,
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

    def handle_message(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        """Route one parsed JSON message. Returns the canonical
        frames it produced (none for control messages). Raises
        InAppError on any protocol violation."""
        if self.finished:
            raise InAppError("stream is finished: no messages after stop/close")
        if not isinstance(msg, dict):
            raise InAppError("message must be a JSON object")
        self._stats["messages"] += 1
        kind = msg.get("type")
        if kind == "hello":
            self._on_hello(msg)
            return []
        if not self.started:
            raise InAppError(
                "first message must be hello: media before hello "
                "cannot be decoded, because the format is pinned "
                "by the hello (see ADR-015 item 3)"
            )
        if kind == "media":
            return self._on_media(msg)
        if kind == "stop":
            self._on_stop()
            return []
        raise InAppError(f"unknown message type {kind!r} (expected hello/media/stop)")

    def finish(self) -> None:
        """End the stream at socket close. Idempotent: discards a
        partial trailing frame rather than padding it, drops any
        reordered packets still held, and closes the decoder."""
        if self.finished:
            return
        self.finished = True
        self._stats["discarded_on_close"] += int(self._buffer.size)
        self._buffer = np.empty(0, dtype=np.float32)
        self._stats["pending_discarded"] += len(self._pending)
        self._pending.clear()
        self._close_decoder()

    # -- hello ----------------------------------------------------------

    def _on_hello(self, msg: Dict[str, Any]) -> None:
        if self.started:
            raise InAppError(
                "second hello: a reconnect is a new connection with "
                "a fresh hello, never a re-pin mid-stream"
            )
        codec = msg.get("codec")
        if not isinstance(codec, str) or codec.strip().lower() != CODEC_OPUS:
            raise InAppError(
                f"unsupported codec {codec!r} (supported: {CODEC_OPUS!r}; "
                "the in-app fork carries Opus packets, see ADR-015 item 3)"
            )
        rate = msg.get("sample_rate")
        if isinstance(rate, bool) or not isinstance(rate, int) or rate not in ENCODED_RATES:
            raise InAppError(
                f"unsupported sample_rate {rate!r} "
                f"(supported: {', '.join(map(str, ENCODED_RATES))})"
            )
        channels = msg.get("channels")
        if isinstance(channels, bool) or not isinstance(channels, int) or channels != 1:
            raise InAppError(
                f"unsupported channels {channels!r}: the in-app fork is mono"
            )
        track = msg.get("track")
        if not isinstance(track, str) or track.strip().lower() not in TRACKS:
            raise InAppError(
                f"unknown track {track!r} (expected {TRACK_FAR_END!r} or "
                f"{TRACK_NEAR_END!r}; a mixed track cannot be split into "
                "legs and is refused, see ADR-015 item 6)"
            )
        track = track.strip().lower()
        stream_id = msg.get("stream_id")
        if stream_id is not None and (not isinstance(stream_id, str) or not stream_id):
            raise InAppError("stream_id must be a non-empty string when present")
        declared_call_id = msg.get("call_id")
        if declared_call_id is not None:
            if not isinstance(declared_call_id, str) or not declared_call_id:
                raise InAppError("hello call_id must be a non-empty string")
            if declared_call_id != self.call_id:
                raise InAppError(
                    "hello call_id does not match the pre-opened call: "
                    "refusing a misrouted fork"
                )
        consent_ref = msg.get("consent_ref")
        if consent_ref is not None and not isinstance(consent_ref, str):
            raise InAppError("consent_ref must be a string when present")
        self.hello = dict(msg)
        self.sample_rate = rate
        self.track = track
        self.stream_sid = stream_id or ""
        # Recorded for audit only: no consent ledger exists to
        # validate it against (ADR-015 item 8). Recording it is
        # not a claim that it was checked.
        self.consent_ref = consent_ref
        if track == TRACK_FAR_END and self._decoder is None:
            try:
                self._decoder = OpusDecoder(output_rate=OUTPUT_RATE, channels=1)
            except OpusDecoderUnavailable as exc:
                raise InAppError(str(exc)) from exc
            except OpusError as exc:
                raise InAppError(str(exc)) from exc
        self.started = True

    # -- media ----------------------------------------------------------

    def _on_media(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        seq = self._parse_seq(msg.get("seq"))
        timestamp = self._parse_timestamp(msg.get("timestamp"))
        if self.track == TRACK_NEAR_END:
            self._stats["near_end_messages"] += 1
            self._account_near_end(seq)
            return []
        payload = self._parse_payload(msg.get("payload"))
        self._stats["media_messages"] += 1
        self._stats["bytes"] += len(payload)
        unwrapped = self._unwrap(timestamp)
        if seq in self._pending:
            # Already held in the reorder buffer: a retransmit,
            # not new audio. Drop it without decoding it again.
            self._stats["duplicates_dropped"] += 1
            return []
        forward = (seq - self._expected_seq) % _SEQ_MODULUS
        if forward == 0:
            samples = self._decode(payload)
            self._stats["packets"] += 1
            self._append_decoded(samples, unwrapped)
            self._expected_seq = (seq + 1) % _SEQ_MODULUS
            self._drain_pending()
            return self._drain_frames()
        if 1 <= forward <= _MAX_SEQ_JUMP:
            samples = self._decode(payload)
            self._stats["packets"] += 1
            self._pending[seq] = _PendingPacket(seq, unwrapped, samples)
            self._flush_stalled(unwrapped)
            return self._drain_frames()
        if forward > _SEQ_MODULUS // 2:
            # Behind the expected sequence: already emitted, or
            # already held. A late packet is dropped, never
            # re-emitted out of order.
            if seq in self._pending:
                self._stats["duplicates_dropped"] += 1
            else:
                self._stats["late_dropped"] += 1
            return []
        raise InAppError(
            f"sequence jump from {self._expected_seq} to {seq}: a fork "
            "that restarts its sequence reconnects with a fresh hello "
            "(see ADR-015 item 3)"
        )

    def _account_near_end(self, seq: int) -> None:
        """Sequence accounting for the near-end leg only. Its audio
        is never decoded and never framed, so there is nothing to
        reorder: a later sequence simply records the gap."""
        forward = (seq - self._expected_seq) % _SEQ_MODULUS
        if forward == 0:
            self._expected_seq = (seq + 1) % _SEQ_MODULUS
        elif 1 <= forward <= _MAX_SEQ_JUMP:
            self._stats["gaps"] += 1
            self._stats["lost_packets"] += forward
            self._expected_seq = (seq + 1) % _SEQ_MODULUS
        elif forward > _SEQ_MODULUS // 2:
            self._stats["late_dropped"] += 1
        else:
            raise InAppError(
                f"sequence jump from {self._expected_seq} to {seq}: a fork "
                "that restarts its sequence reconnects with a fresh hello "
                "(see ADR-015 item 3)"
            )

    def _drain_pending(self) -> None:
        """Emit the buffered run that is now contiguous, in seq order."""
        while self._expected_seq in self._pending:
            packet = self._pending.pop(self._expected_seq)
            self._append_decoded(packet.samples, packet.timestamp)
            self._expected_seq = (packet.seq + 1) % _SEQ_MODULUS

    def _flush_stalled(self, newest_unwrapped: int) -> None:
        """Flush buffered packets whose expected predecessor has not
        arrived within the 60 ms reorder window (TDD section 4.1).
        The window is measured in RTP ticks from the timeline
        anchor (the end of the audio emitted or buffered so far,
        or the oldest held packet when nothing has been emitted
        yet) to the newest packet to arrive: once more than
        60 ms of audio past the missing packet has arrived, the
        missing sequences are counted lost, and the first frame
        after the gap is flagged discontinuous. Tick space is
        used so no presentation base is set before the first
        packet is emitted in sequence order."""
        while self._pending:
            smallest = min(self._pending, key=lambda s: (s - self._expected_seq) % _SEQ_MODULUS)
            if smallest == self._expected_seq:
                self._drain_pending()
                return
            anchor = self._anchor_ticks()
            if anchor is None:
                return
            if newest_unwrapped - anchor <= REORDER_WINDOW_TICKS:
                return
            packet = self._pending[smallest]
            distance = (smallest - self._expected_seq) % _SEQ_MODULUS
            self._stats["gaps"] += 1
            self._stats["lost_packets"] += distance
            self._stats["lost_samples"] += max(
                0, round((packet.timestamp - anchor) / (RTP_CLOCK / OUTPUT_RATE))
            )
            self._discard_partial_for_gap()
            self._gap_before_next_frame = True
            self._pending.pop(smallest)
            self._append_decoded(packet.samples, packet.timestamp)
            self._expected_seq = (packet.seq + 1) % _SEQ_MODULUS
            self._drain_pending()

    def _anchor_ticks(self) -> Optional[int]:
        """Unwrapped RTP timestamp of the end of the audio emitted
        or buffered so far, or of the oldest held packet when
        nothing has been emitted yet."""
        if self._base_unwrapped is not None:
            if self._buffer.size:
                end_sec = (self._buffer_t_sec or 0.0) + self._buffer.size / OUTPUT_RATE
                return self._base_unwrapped + round(end_sec * RTP_CLOCK)
            if self._last_frame_end_t_sec is not None:
                return self._base_unwrapped + round(
                    self._last_frame_end_t_sec * RTP_CLOCK
                )
        if self._pending:
            oldest = min(
                self._pending.values(),
                key=lambda p: (p.seq - self._expected_seq) % _SEQ_MODULUS,
            )
            return oldest.timestamp
        return None

    def _discard_partial_for_gap(self) -> None:
        """A partial frame buffered before a gap is discarded, not
        spliced across the gap: the samples on either side are not
        contiguous in time. The emitted timeline is forgotten too,
        so the packet after the gap starts a fresh timeline
        instead of being judged against the old one."""
        self._stats["discarded_on_gap"] += int(self._buffer.size)
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_t_sec = None
        self._last_frame_end_t_sec = None

    def _append_decoded(self, samples: np.ndarray, unwrapped: int) -> None:
        """Append decoded samples, flagging a sender-clock jump as a
        gap even when the buffer is empty: a jump between the end
        of the last emitted frame and this packet's timestamp is
        lost time, and the next frame must say so."""
        packet_t_sec = self._t_sec(unwrapped)
        if self._buffer.size:
            expected_end = (self._buffer_t_sec or 0.0) + self._buffer.size / OUTPUT_RATE
        else:
            expected_end = self._last_frame_end_t_sec
        if expected_end is not None and abs(packet_t_sec - expected_end) > 0.0015:
            self._stats["timestamp_jumps"] += 1
            self._stats["gaps"] += 1
            self._stats["lost_samples"] += max(
                0, round((packet_t_sec - expected_end) * OUTPUT_RATE)
            )
            self._discard_partial_for_gap()
            self._gap_before_next_frame = True
        if self._buffer.size == 0:
            self._buffer_t_sec = packet_t_sec
        self._buffer = np.concatenate([self._buffer, samples])

    def _drain_frames(self) -> List[MediaFrame]:
        frames: List[MediaFrame] = []
        first = True
        while self._buffer.size >= FRAME_SAMPLES_8K:
            chunk = self._buffer[:FRAME_SAMPLES_8K]
            self._buffer = self._buffer[FRAME_SAMPLES_8K:]
            t_sec = self._buffer_t_sec or 0.0
            frames.append(
                MediaFrame(
                    call_id=self.call_id,
                    stream_sid=self.stream_sid,
                    track="inbound",
                    seq=self._frames,
                    t_sec=t_sec,
                    samples=chunk.copy(),
                    sample_rate=OUTPUT_RATE,
                    discontinuous=self._gap_before_next_frame and first,
                    source="inapp",
                    source_codec=CODEC_OPUS,
                    source_rate=self.sample_rate or 0,
                )
            )
            self._frames += 1
            self._stats["frames"] += 1
            self._emitted_samples += FRAME_SAMPLES_8K
            self._stats["samples"] += FRAME_SAMPLES_8K
            self._buffer_t_sec = t_sec + FRAME_MS / 1000.0
            self._last_frame_end_t_sec = self._buffer_t_sec
            first = False
        if frames:
            self._gap_before_next_frame = False
        return frames

    # -- stop / helpers ---------------------------------------------------

    def _on_stop(self) -> None:
        self.finished = True
        self._stats["discarded_on_stop"] += int(self._buffer.size)
        self._buffer = np.empty(0, dtype=np.float32)
        self._stats["pending_discarded"] += len(self._pending)
        self._pending.clear()
        self._close_decoder()

    def _close_decoder(self) -> None:
        """Close the decoder. The stream takes ownership of the
        decoder it was given or built: one decoder per stream,
        never shared, so closing it here cannot strand a caller."""
        if self._decoder is not None:
            close = getattr(self._decoder, "close", None)
            if callable(close):
                close()
        self._decoder = None

    def _decode(self, payload: bytes) -> np.ndarray:
        assert self._decoder is not None
        try:
            samples = self._decoder.decode(payload)
        except OpusError as exc:
            raise InAppError(f"Opus decode failed: {exc}") from exc
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise InAppError("Opus packet decoded to zero samples")
        return samples

    def _unwrap(self, timestamp: int) -> int:
        """Unwrap a 32-bit RTP timestamp against the first raw value
        seen, so wraparound mid-call cannot move the clock
        backwards. Call lengths are far below one wrap period
        (2**32 ticks is over a day at 48 kHz). The unwrap
        reference is the first arrival, not the first emission:
        the presentation base is set separately, by the first
        packet actually emitted, in _t_sec."""
        if self._unwrap_ref is None:
            self._unwrap_ref = timestamp
            return timestamp
        ref = self._unwrap_ref
        delta = ((timestamp - ref + (1 << 31)) % _TIMESTAMP_MODULUS) - (1 << 31)
        return ref + delta

    def _t_sec(self, unwrapped: int) -> float:
        """Presentation time of an unwrapped timestamp, in seconds
        from the first emitted packet. The base is the first
        packet appended to the frame buffer in sequence order,
        so a first arrival that is out of order cannot shift
        the whole stream's clock: seq 0 still lands at 0."""
        if self._base_unwrapped is None:
            self._base_unwrapped = unwrapped
            return 0.0
        return (unwrapped - self._base_unwrapped) / RTP_CLOCK

    @staticmethod
    def _parse_seq(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise InAppError(f"seq {value!r} is not an integer (expected 0..65535)")
        if not 0 <= value < _SEQ_MODULUS:
            raise InAppError(f"seq {value!r} is out of range (expected 0..65535)")
        return value

    @staticmethod
    def _parse_timestamp(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise InAppError(
                f"timestamp {value!r} is not an integer "
                "(expected an RTP timestamp on the 48 kHz clock)"
            )
        if not 0 <= value < _TIMESTAMP_MODULUS:
            raise InAppError(
                f"timestamp {value!r} is out of range "
                "(expected 0..4294967295)"
            )
        return value

    @staticmethod
    def _parse_payload(value: Any) -> bytes:
        if not isinstance(value, str):
            raise InAppError("payload must be a base64 string")
        try:
            payload = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InAppError(f"payload is not valid base64: {exc}") from exc
        if not payload:
            raise InAppError("payload is empty: a lost packet is a gap, not a packet")
        if len(payload) > _MAX_PACKET_BYTES:
            raise InAppError(
                f"payload is {len(payload)} bytes, over the "
                f"{_MAX_PACKET_BYTES}-byte cap for one Opus packet"
            )
        return payload
