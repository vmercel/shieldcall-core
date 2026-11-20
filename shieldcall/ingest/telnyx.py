"""Telnyx Media Streaming adapter (TDD v2.0 §4, Media Ingest Fabric).

Translates Telnyx's WebSocket JSON protocol into the canonical 10 ms
frame envelope. Pure and transport-agnostic: it takes parsed JSON
messages and returns frames. It opens no socket, holds no credential,
and knows nothing about HTTP. The serving layer owns auth and the
websocket; see shieldcall.serve.http_app for the /v1/ingest/telnyx
route that drives this class.

Protocol (Telnyx docs, "Media Streaming over Websockets", read
2026-11-20):
- connected: {"event": "connected", "version": "1.0.0"}
  No protocol field, no stream_id, no sequence_number.
- start: {"event": "start", "sequence_number": "1",
    "start": {"user_id", "call_control_id", "call_session_id",
              "from", "to", "tags", "client_state",
              "media_format": {"encoding": "PCMU",
                               "sample_rate": 8000, "channels": 1}},
    "stream_id": "..."}
  The stream_id is top level, after the start object. There is no
  tracks list and no custom-parameters object: track selection is a
  stream-request parameter (stream_track: inbound_track (default),
  outbound_track, both_tracks), and the per-message track is
  media.track.
- media: {"event": "media", "sequence_number": "4",
    "media": {"track": "inbound"|"outbound", "chunk": "2",
              "timestamp": "5", "payload": "<base64>"},
    "stream_id": "..."}
  The payload is a base64-encoded RTP payload (no headers).
  timestamp is the RTP timestamp of the media packet, in the codec's
  RTP clock units: it is not wall-clock time and not milliseconds.
  At 8 kHz one timestamp unit is one sample.
- stop: {"event": "stop", "sequence_number": "5",
    "stop": {"user_id", "call_control_id"}, "stream_id": "..."}
- dtmf: {"event": "dtmf", "stream_id": "...",
    "occurred_at": "<ISO-8601>", "sequence_number": "5",
    "dtmf": {"digit": "1"}}
  DTMF may arrive out of order; occurred_at is the
  telephony-engine-side timestamp.
- mark (received): {"event": "mark", "stream_id": "...",
    "sequence_number": "5", "mark": {"name": "..."}}
- error: {"event": "error",
    "payload": {"code": <int>, "title": <str>, "detail": <str>},
    "stream_id": "..."}
  No sequence_number. Codes: 100002 unknown_error, 100003
  malformed_frame, 100004 invalid_media, 100005 rate_limit_reached.
- clear: {"event": "clear"} is sent only; this adapter never sends.

Continuity is judged from media.timestamp, not from chunk: chunk is
ONE counter shared by both tracks, so a jump in it means media was
lost somewhere on the stream but cannot say on which leg. A
timestamp that jumps past the end of the previous payload is a gap;
a timestamp that rewinds, or a chunk repeated on the same track, is
a duplicate and is dropped. The order of events is not guaranteed
by Telnyx; this adapter emits frames in arrival order and drops
rewinds rather than buffering for a reorder window (see ADR-011).
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .frame import FRAME_MS, FRAME_SAMPLES_8K, MediaFrame
from .g711 import alaw_decode, mulaw_decode

__all__ = ["TelnyxError", "TelnyxMediaStream"]

SAMPLE_RATE = 8000
"""Telnyx media is 8 kHz (per the docs above)."""

SAMPLES_PER_MS = SAMPLE_RATE / 1000.0
TRACK_INBOUND = "inbound"
TRACK_OUTBOUND = "outbound"
TRACKS = (TRACK_INBOUND, TRACK_OUTBOUND)

# media_format.encoding values the adapter accepts, with their decoder.
# PCMU is G.711 mu-law, PCMA is G.711 A-law; both are 8 kHz telephony
# codecs decoded by the shared g711 module. G722, OPUS, AMR-WB and L16
# are refused: the adapter never guesses what a codec it was not told
# to expect sounds like.
ENCODINGS = {"PCMU": mulaw_decode, "PCMA": alaw_decode}

# Telnyx error-frame codes (docs table, "Error message").
ERROR_TITLES = {
    100002: "unknown_error",
    100003: "malformed_frame",
    100004: "invalid_media",
    100005: "rate_limit_reached",
}


class TelnyxError(ValueError):
    """A Telnyx protocol violation: malformed message, wrong state, or
    an unsupported media format. The caller should close the stream;
    the adapter never guesses its way past one."""


@dataclass
class _TrackState:
    """Per-track continuity state. Only the analyzed track buffers
    audio; the other track's messages are counted and validated but
    never decoded, so the legs can never be mixed."""

    last_chunk: Optional[int] = None
    # RTP timestamp (codec clock units: one unit is one sample at
    # 8 kHz) of the previous payload's first sample.
    last_timestamp: Optional[int] = None
    last_duration_samples: Optional[int] = None
    # RTP timestamp of the first buffered sample not yet emitted as a
    # frame.
    buffer_t: Optional[int] = None
    buffer: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    frames: int = 0
    messages: int = 0


class TelnyxMediaStream:
    """One Telnyx media stream, from `connected` to `stop`.

    - call_id: the pre-opened ShieldCall call id. The operator opens
      the call with the Telnyx call_control_id (a `v2:...` id), and
      `start.call_control_id` must match it; a mismatch is a
      protocol error, not something to re-route.
    - analyzed_track: the leg that becomes frames. Default "inbound"
      (the remote caller, the leg ShieldCall scores). The other leg is
      counted but never decoded. Telnyx declares no track list in
      `start`, so the operator must request the right leg with
      `stream_track` when starting the stream; the per-message
      filter here is the backstop.
    """

    def __init__(self, call_id: str, analyzed_track: str = TRACK_INBOUND):
        if not call_id:
            raise TelnyxError("call_id is required")
        if analyzed_track not in TRACKS:
            raise TelnyxError(f"unknown track {analyzed_track!r}")
        self.call_id = call_id
        self.analyzed_track = analyzed_track
        self.stream_id: Optional[str] = None
        self.call_control_id: Optional[str] = None
        self.call_session_id: Optional[str] = None
        self.from_number: Optional[str] = None
        self.to_number: Optional[str] = None
        self.tags: List[str] = []
        self.client_state: Optional[str] = None
        self.encoding: Optional[str] = None
        self.last_sequence_number: Optional[int] = None
        self.started = False
        self.stopped = False
        self.dtmf_events: List[Dict[str, str]] = []
        self.marks: List[str] = []
        self._state: Dict[str, _TrackState] = {t: _TrackState() for t in TRACKS}
        # chunk is ONE counter shared by both tracks: a jump past the
        # highest chunk seen means Telnyx lost media somewhere on the
        # stream, on either leg. Tracked globally, never per track.
        self._max_chunk: Optional[int] = None
        self._stats: Dict[str, int] = {
            "messages": 0,
            "media_messages": 0,
            "media_messages_other_track": 0,
            "frames": 0,
            "samples": 0,
            "gaps": 0,
            "lost_samples": 0,
            "chunk_gaps": 0,
            "duplicates_dropped": 0,
            "out_of_order_dropped": 0,
            "discarded_on_stop": 0,
            "dtmf": 0,
            "marks": 0,
            "errors": 0,
        }

    # -- public ---------------------------------------------------------

    @property
    def is_started(self) -> bool:
        return self.started

    @property
    def is_stopped(self) -> bool:
        return self.stopped

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def handle_message(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        """Route one parsed Telnyx JSON message. Returns the canonical
        frames it produced (empty for control messages). Raises
        TelnyxError on any protocol violation."""
        if not isinstance(msg, dict):
            raise TelnyxError("message must be a JSON object")
        self._stats["messages"] += 1
        self._record_sequence_number(msg.get("sequence_number"))
        event = msg.get("event")
        if event == "connected":
            return self._on_connected(msg)
        if event == "start":
            return self._on_start(msg)
        if event == "media":
            return self._on_media(msg)
        if event == "stop":
            return self._on_stop(msg)
        if event == "dtmf":
            return self._on_dtmf(msg)
        if event == "mark":
            return self._on_mark(msg)
        if event == "error":
            return self._on_error(msg)
        raise TelnyxError(f"unknown Telnyx event {event!r}")

    # -- events ---------------------------------------------------------

    def _on_connected(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if self.started or self.stopped:
            raise TelnyxError("connected received after the stream started")
        # Telnyx sends no protocol field on connected (unlike Twilio);
        # the version is recorded, not enforced, so a future 1.x must
        # not break a deployed adapter.
        self._version = str(msg.get("version") or "")
        return []

    def _on_start(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if self.stopped:
            raise TelnyxError("start received after stop")
        if self.started:
            raise TelnyxError("second start message")
        start = msg.get("start")
        if not isinstance(start, dict):
            raise TelnyxError("start message has no start object")
        # stream_id is top level, after the start object, not inside it.
        stream_id = msg.get("stream_id") or start.get("stream_id") or ""
        if not stream_id:
            raise TelnyxError("start message has no stream_id")
        call_control_id = start.get("call_control_id") or ""
        if not call_control_id:
            raise TelnyxError("start message has no call_control_id")
        if call_control_id != self.call_id:
            raise TelnyxError(
                f"start call_control_id {call_control_id!r} "
                f"does not match call {self.call_id!r}"
            )
        fmt = start.get("media_format")
        if not isinstance(fmt, dict):
            raise TelnyxError("start message has no media_format")
        encoding = fmt.get("encoding")
        if encoding not in ENCODINGS:
            raise TelnyxError(
                f"unsupported Telnyx encoding {encoding!r} "
                f"(supported: {sorted(ENCODINGS)})"
            )
        if fmt.get("sample_rate") != SAMPLE_RATE:
            raise TelnyxError(
                f"unsupported Telnyx sample_rate {fmt.get('sample_rate')!r}"
            )
        if fmt.get("channels") != 1:
            raise TelnyxError(
                f"unsupported Telnyx channels {fmt.get('channels')!r}"
            )
        # No tracks list is declared in start (unlike Twilio): the
        # operator selects the leg with stream_track at stream start,
        # and the adapter filters per message below.
        self.stream_id = str(stream_id)
        self.call_control_id = str(call_control_id)
        self.call_session_id = _as_str(start.get("call_session_id"))
        self.from_number = _as_str(start.get("from"))
        self.to_number = _as_str(start.get("to"))
        tags = start.get("tags")
        self.tags = [str(t) for t in tags] if isinstance(tags, list) else []
        self.client_state = _as_str(start.get("client_state"))
        self.encoding = str(encoding)
        self.started = True
        return []

    def _on_media(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if not self.started:
            raise TelnyxError("media received before start")
        if self.stopped:
            raise TelnyxError("media received after stop")
        if (msg.get("stream_id") or "") != self.stream_id:
            raise TelnyxError("media stream_id does not match the started stream")
        media = msg.get("media")
        if not isinstance(media, dict):
            raise TelnyxError("media message has no media object")
        track = media.get("track")
        if track not in TRACKS:
            raise TelnyxError(f"unknown media track {track!r}")
        state = self._state[track]
        state.messages += 1
        self._stats["media_messages"] += 1
        chunk = self._parse_int(media.get("chunk"), "media.chunk")
        timestamp = self._parse_int(media.get("timestamp"), "media.timestamp")
        # chunk is one counter for both tracks: a jump past the highest
        # chunk seen means Telnyx lost media somewhere on the stream.
        # It is counted for visibility only; it cannot say which leg
        # lost it, so it never flags this track discontinuous.
        if self._max_chunk is not None and chunk > self._max_chunk + 1:
            self._stats["chunk_gaps"] += 1
        if self._max_chunk is None or chunk > self._max_chunk:
            self._max_chunk = chunk
        if track != self.analyzed_track:
            # Counted and validated, never decoded: the legs stay separate.
            self._stats["media_messages_other_track"] += 1
            return []
        payload = media.get("payload")
        if not isinstance(payload, str) or not payload:
            raise TelnyxError("media message has no payload")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise TelnyxError(f"media payload is not valid base64: {exc}") from exc
        if not raw:
            raise TelnyxError("media payload decodes to zero bytes")
        # Duplicate: same chunk on this track at the same or an earlier
        # RTP timestamp. Out of order: timestamp behind the previous
        # payload's end. Both are dropped, never replayed into the
        # pipeline.
        if (
            state.last_chunk is not None
            and chunk <= state.last_chunk
            and timestamp <= (state.last_timestamp or 0)
        ):
            self._stats["duplicates_dropped"] += 1
            return []
        duration_samples = len(raw)
        discontinuous = False
        if state.last_timestamp is not None:
            # The next timestamp should land exactly at the end of the
            # previous payload: one RTP timestamp unit is one sample at
            # 8 kHz, so the expected advance is the previous payload's
            # sample count, not its duration in milliseconds.
            expected = state.last_timestamp + (state.last_duration_samples or duration_samples)
            if timestamp > expected:
                gap_samples = timestamp - expected
                self._stats["gaps"] += 1
                self._stats["lost_samples"] += gap_samples
                discontinuous = True
                # A partial buffered frame straddles the gap: its
                # samples end at expected while the new audio starts
                # later. Discard it rather than splicing two timelines
                # into one frame, and restart the buffer clock at the
                # new message.
                if state.buffer.size:
                    self._stats["lost_samples"] += int(state.buffer.size)
                    state.buffer = np.empty(0, dtype=np.float32)
                state.buffer_t = timestamp
            elif timestamp < state.last_timestamp:
                self._stats["out_of_order_dropped"] += 1
                return []
        state.last_chunk = chunk
        state.last_timestamp = timestamp
        state.last_duration_samples = duration_samples
        if state.buffer_t is None:
            state.buffer_t = timestamp
        decode = ENCODINGS[self.encoding or "PCMU"]
        samples = decode(raw)
        state.buffer = np.concatenate([state.buffer, samples]) if state.buffer.size else samples
        self._stats["samples"] += int(samples.size)
        return self._drain_frames(state, track, discontinuous)

    def _on_stop(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if not self.started:
            raise TelnyxError("stop received before start")
        if self.stopped:
            raise TelnyxError("second stop message")
        if (msg.get("stream_id") or "") != self.stream_id:
            raise TelnyxError("stop stream_id does not match the started stream")
        stop = msg.get("stop")
        if isinstance(stop, dict) and stop.get("call_control_id"):
            if stop["call_control_id"] != self.call_id:
                raise TelnyxError(
                    f"stop call_control_id {stop['call_control_id']!r} "
                    f"does not match call {self.call_id!r}"
                )
        # A partial trailing frame is discarded, not padded: padding
        # would invent audio the caller never sent. The count is kept
        # in stats so a short final chunk is visible, not silent.
        for state in self._state.values():
            self._stats["discarded_on_stop"] += int(state.buffer.size)
            state.buffer = np.empty(0, dtype=np.float32)
            state.buffer_t = None
        self.stopped = True
        return []

    def _on_dtmf(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if not self.started:
            raise TelnyxError("dtmf received before start")
        dtmf = msg.get("dtmf")
        if not isinstance(dtmf, dict) or not dtmf.get("digit"):
            raise TelnyxError("dtmf message has no digit")
        # DTMF may arrive out of order; occurred_at is the
        # telephony-engine-side timestamp and is kept with the digit.
        self.dtmf_events.append(
            {
                "digit": str(dtmf["digit"]),
                "occurred_at": str(msg.get("occurred_at") or ""),
            }
        )
        self._stats["dtmf"] += 1
        return []

    def _on_mark(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if not self.started:
            raise TelnyxError("mark received before start")
        if (msg.get("stream_id") or "") != self.stream_id:
            raise TelnyxError("mark stream_id does not match the started stream")
        mark = msg.get("mark")
        if not isinstance(mark, dict) or not mark.get("name"):
            raise TelnyxError("mark message has no name")
        self.marks.append(str(mark["name"]))
        self._stats["marks"] += 1
        return []

    def _on_error(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        # Telnyx reports stream errors as a frame; the adapter turns
        # it into a protocol error so the caller closes the stream
        # rather than carrying on with a broken feed.
        self._stats["errors"] += 1
        payload = msg.get("payload")
        if isinstance(payload, dict):
            code = payload.get("code")
            title = payload.get("title") or ERROR_TITLES.get(code, "unknown_error")
            detail = payload.get("detail") or ""
            raise TelnyxError(f"telnyx error {code} {title}: {detail}".strip())
        raise TelnyxError("telnyx error frame with no payload")

    # -- helpers --------------------------------------------------------

    def _record_sequence_number(self, value: Any) -> None:
        """sequence_number counts the frames Telnyx sends; it can be
        skipped and two frames can share a number, so it orders but
        does not detect loss. Recorded for diagnostics only, parsed
        leniently: connected and error frames do not carry one, and a
        malformed value must not break the stream."""
        if value is None:
            return
        try:
            self.last_sequence_number = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass

    def _drain_frames(
        self, state: _TrackState, track: str, discontinuous: bool
    ) -> List[MediaFrame]:
        frames: List[MediaFrame] = []
        first = True
        while state.buffer.size >= FRAME_SAMPLES_8K:
            chunk_samples = state.buffer[:FRAME_SAMPLES_8K]
            state.buffer = state.buffer[FRAME_SAMPLES_8K:]
            # One RTP timestamp unit is one sample at 8 kHz, so the
            # presentation time is the buffer clock over the rate.
            t_sec = (state.buffer_t or 0) / SAMPLE_RATE
            frames.append(
                MediaFrame(
                    call_id=self.call_id,
                    stream_sid=self.stream_id or "",
                    track=track,
                    seq=state.frames,
                    t_sec=t_sec,
                    samples=chunk_samples.copy(),
                    sample_rate=SAMPLE_RATE,
                    discontinuous=discontinuous and first,
                    source="telnyx",
                )
            )
            state.frames += 1
            self._stats["frames"] += 1
            state.buffer_t = (state.buffer_t or 0) + FRAME_SAMPLES_8K
            first = False
        return frames

    @staticmethod
    def _parse_int(value: Any, name: str) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise TelnyxError(f"{name} is not an integer: {value!r}") from exc


def _as_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)
