"""Twilio Media Streams adapter (TDD v2.0 §4, Media Ingest Fabric).

Translates Twilio's WebSocket JSON protocol into the canonical 10 ms
frame envelope. Pure and transport-agnostic: it takes parsed JSON
messages and returns frames. It opens no socket, holds no credential,
and knows nothing about HTTP. The serving layer owns auth and the
websocket; see shieldcall.serve.http_app for the /v1/ingest/twilio
route that drives this class.

Protocol (Twilio docs, "Media Streams - WebSocket Messages"):
- connected: {"event": "connected", "protocol": "Call", "version": "1.0.0"}
- start: {"event": "start", "sequenceNumber": "1",
    "start": {"streamSid", "accountSid", "callSid",
              "tracks": ["inbound", ...],
              "mediaFormat": {"encoding": "audio/x-mulaw",
                              "sampleRate": 8000, "channels": 1},
              "customParameters": {...}},
    "streamSid": ...}
- media: {"event": "media", "sequenceNumber": "3",
    "media": {"track": "inbound"|"outbound", "chunk": "1",
              "timestamp": "5", "payload": "<base64 mu-law>"},
    "streamSid": ...}
  chunk starts at "1" and increments per message; timestamp is the
  presentation time in milliseconds from the start of the stream;
  payloads are 20 ms (160 bytes) of G.711 mu-law at 8 kHz.
- stop: {"event": "stop", "stop": {"accountSid", "callSid"}, "streamSid"}
- dtmf: {"event": "dtmf", "dtmf": {"track": "inbound", "digit": "1"}}
- mark: {"event": "mark", "mark": {"name": "..."}} (bidirectional only)

Continuity is judged from media.timestamp, not from chunk: the chunk
counter's scope across tracks is not pinned down by the docs (their
example numbers it globally across tracks), so a chunk jump across
tracks is not a gap. A timestamp that jumps past the end of the
previous payload is a gap; a timestamp that goes backwards, or a
chunk repeated on the same track, is a duplicate and is dropped.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .frame import FRAME_MS, FRAME_SAMPLES_8K, MediaFrame
from .g711 import _MULAW_TABLE, mulaw_decode

__all__ = ["TwilioError", "TwilioMediaStream", "mulaw_decode", "_MULAW_TABLE"]

SAMPLE_RATE = 8000
"""Twilio media is always 8 kHz mu-law (per the docs above)."""

SAMPLES_PER_MS = SAMPLE_RATE / 1000.0
TRACK_INBOUND = "inbound"
TRACK_OUTBOUND = "outbound"
TRACKS = (TRACK_INBOUND, TRACK_OUTBOUND)


class TwilioError(ValueError):
    """A Twilio protocol violation: malformed message, wrong state, or
    an unsupported media format. The caller should close the stream;
    the adapter never guesses its way past one."""


@dataclass
class _TrackState:
    """Per-track continuity state. Only the analyzed track buffers
    audio; the other track's messages are counted and validated but
    never decoded, so the legs can never be mixed."""

    last_chunk: Optional[int] = None
    last_timestamp_ms: Optional[float] = None
    last_duration_ms: Optional[float] = None
    # Presentation time (ms from stream start) of the first buffered
    # sample not yet emitted as a frame.
    buffer_t_ms: Optional[float] = None
    buffer: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    frames: int = 0
    messages: int = 0


class TwilioMediaStream:
    """One Twilio Media Stream, from `connected` to `stop`.

    - call_id: the pre-opened ShieldCall call id. The operator opens
      the call with the Twilio Call SID, and `start.callSid` must match
      it; a mismatch is a protocol error, not something to re-route.
    - analyzed_track: the leg that becomes frames. Default "inbound"
      (the remote caller, the leg ShieldCall scores). The other leg is
      counted but never decoded.
    """

    def __init__(self, call_id: str, analyzed_track: str = TRACK_INBOUND):
        if not call_id:
            raise TwilioError("call_id is required")
        if analyzed_track not in TRACKS:
            raise TwilioError(f"unknown track {analyzed_track!r}")
        self.call_id = call_id
        self.analyzed_track = analyzed_track
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None
        self.tracks: List[str] = []
        self.started = False
        self.stopped = False
        self.dtmf_digits: List[str] = []
        self.marks: List[str] = []
        self._state: Dict[str, _TrackState] = {t: _TrackState() for t in TRACKS}
        self._stats: Dict[str, int] = {
            "messages": 0,
            "media_messages": 0,
            "media_messages_other_track": 0,
            "frames": 0,
            "samples": 0,
            "gaps": 0,
            "lost_samples": 0,
            "duplicates_dropped": 0,
            "out_of_order_dropped": 0,
            "discarded_on_stop": 0,
            "dtmf": 0,
            "marks": 0,
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
        """Route one parsed Twilio JSON message. Returns the canonical
        frames it produced (empty for control messages). Raises
        TwilioError on any protocol violation."""
        if not isinstance(msg, dict):
            raise TwilioError("message must be a JSON object")
        self._stats["messages"] += 1
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
        raise TwilioError(f"unknown Twilio event {event!r}")

    # -- events ---------------------------------------------------------

    def _on_connected(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if self.started or self.stopped:
            raise TwilioError("connected received after the stream started")
        protocol = msg.get("protocol")
        if protocol != "Call":
            raise TwilioError(f"unsupported Twilio protocol {protocol!r}")
        # The version is recorded, not enforced: a future 1.x must not
        # break a deployed adapter, and the message shapes are what we
        # validate, field by field, below.
        self._version = str(msg.get("version") or "")
        return []

    def _on_start(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if self.stopped:
            raise TwilioError("start received after stop")
        if self.started:
            raise TwilioError("second start message")
        start = msg.get("start")
        if not isinstance(start, dict):
            raise TwilioError("start message has no start object")
        stream_sid = start.get("streamSid") or msg.get("streamSid") or ""
        if not stream_sid:
            raise TwilioError("start message has no streamSid")
        call_sid = start.get("callSid") or ""
        if not call_sid:
            raise TwilioError("start message has no callSid")
        if call_sid != self.call_id:
            raise TwilioError(
                f"start callSid {call_sid!r} does not match call {self.call_id!r}"
            )
        fmt = start.get("mediaFormat")
        if not isinstance(fmt, dict):
            raise TwilioError("start message has no mediaFormat")
        if fmt.get("encoding") != "audio/x-mulaw":
            raise TwilioError(
                f"unsupported Twilio encoding {fmt.get('encoding')!r}"
            )
        if fmt.get("sampleRate") != SAMPLE_RATE:
            raise TwilioError(
                f"unsupported Twilio sampleRate {fmt.get('sampleRate')!r}"
            )
        if fmt.get("channels") != 1:
            raise TwilioError(f"unsupported Twilio channels {fmt.get('channels')!r}")
        tracks = start.get("tracks")
        if not isinstance(tracks, list) or not tracks:
            raise TwilioError("start message has no tracks")
        if self.analyzed_track not in tracks:
            raise TwilioError(
                f"analyzed track {self.analyzed_track!r} not in stream tracks {tracks!r}"
            )
        self.stream_sid = str(stream_sid)
        self.call_sid = str(call_sid)
        self.tracks = [str(t) for t in tracks]
        self.started = True
        return []

    def _on_media(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if not self.started:
            raise TwilioError("media received before start")
        if self.stopped:
            raise TwilioError("media received after stop")
        if (msg.get("streamSid") or "") != self.stream_sid:
            raise TwilioError("media streamSid does not match the started stream")
        media = msg.get("media")
        if not isinstance(media, dict):
            raise TwilioError("media message has no media object")
        track = media.get("track")
        if track not in TRACKS:
            raise TwilioError(f"unknown media track {track!r}")
        state = self._state[track]
        state.messages += 1
        self._stats["media_messages"] += 1
        chunk = self._parse_int(media.get("chunk"), "media.chunk")
        timestamp_ms = self._parse_float(media.get("timestamp"), "media.timestamp")
        if track != self.analyzed_track:
            # Counted and validated, never decoded: the legs stay separate.
            self._stats["media_messages_other_track"] += 1
            return []
        payload = media.get("payload")
        if not isinstance(payload, str) or not payload:
            raise TwilioError("media message has no payload")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise TwilioError(f"media payload is not valid base64: {exc}") from exc
        if not raw:
            raise TwilioError("media payload decodes to zero bytes")
        # Duplicate: same chunk on this track at the same or an earlier
        # timestamp. Out of order: timestamp behind the previous payload's
        # end. Both are dropped, never replayed into the pipeline.
        if (
            state.last_chunk is not None
            and chunk <= state.last_chunk
            and timestamp_ms <= (state.last_timestamp_ms or 0.0)
        ):
            self._stats["duplicates_dropped"] += 1
            return []
        duration_ms = len(raw) / SAMPLES_PER_MS
        discontinuous = False
        if state.last_timestamp_ms is not None:
            # The next timestamp should land exactly at the end of the
            # previous payload; that duration is what defines "expected".
            expected_ms = state.last_timestamp_ms + (state.last_duration_ms or duration_ms)
            if timestamp_ms > expected_ms:
                gap_ms = timestamp_ms - expected_ms
                self._stats["gaps"] += 1
                self._stats["lost_samples"] += int(round(gap_ms * SAMPLES_PER_MS))
                discontinuous = True
                # A partial buffered frame straddles the gap: its samples
                # end at expected_ms while the new audio starts later.
                # Discard it rather than splicing two timelines into one
                # frame, and restart the buffer clock at the new message.
                if state.buffer.size:
                    self._stats["lost_samples"] += int(state.buffer.size)
                    state.buffer = np.empty(0, dtype=np.float32)
                state.buffer_t_ms = timestamp_ms
            elif timestamp_ms < state.last_timestamp_ms:
                self._stats["out_of_order_dropped"] += 1
                return []
        state.last_chunk = chunk
        state.last_timestamp_ms = timestamp_ms
        state.last_duration_ms = duration_ms
        if state.buffer_t_ms is None:
            state.buffer_t_ms = timestamp_ms
        samples = mulaw_decode(raw)
        state.buffer = np.concatenate([state.buffer, samples]) if state.buffer.size else samples
        self._stats["samples"] += int(samples.size)
        return self._drain_frames(state, track, discontinuous)

    def _on_stop(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if not self.started:
            raise TwilioError("stop received before start")
        if self.stopped:
            raise TwilioError("second stop message")
        if (msg.get("streamSid") or "") != self.stream_sid:
            raise TwilioError("stop streamSid does not match the started stream")
        stop = msg.get("stop")
        if isinstance(stop, dict) and stop.get("callSid") and stop["callSid"] != self.call_id:
            raise TwilioError(
                f"stop callSid {stop['callSid']!r} does not match call {self.call_id!r}"
            )
        # A partial trailing frame is discarded, not padded: padding
        # would invent audio the caller never sent. The count is kept
        # in stats so a short final chunk is visible, not silent.
        for state in self._state.values():
            self._stats["discarded_on_stop"] += int(state.buffer.size)
            state.buffer = np.empty(0, dtype=np.float32)
            state.buffer_t_ms = None
        self.stopped = True
        return []

    def _on_dtmf(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if not self.started:
            raise TwilioError("dtmf received before start")
        dtmf = msg.get("dtmf")
        if not isinstance(dtmf, dict) or not dtmf.get("digit"):
            raise TwilioError("dtmf message has no digit")
        self.dtmf_digits.append(str(dtmf["digit"]))
        self._stats["dtmf"] += 1
        return []

    def _on_mark(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if not self.started:
            raise TwilioError("mark received before start")
        mark = msg.get("mark")
        if not isinstance(mark, dict) or not mark.get("name"):
            raise TwilioError("mark message has no name")
        self.marks.append(str(mark["name"]))
        self._stats["marks"] += 1
        return []

    # -- helpers --------------------------------------------------------

    def _drain_frames(
        self, state: _TrackState, track: str, discontinuous: bool
    ) -> List[MediaFrame]:
        frames: List[MediaFrame] = []
        first = True
        while state.buffer.size >= FRAME_SAMPLES_8K:
            chunk_samples = state.buffer[:FRAME_SAMPLES_8K]
            state.buffer = state.buffer[FRAME_SAMPLES_8K:]
            t_sec = (state.buffer_t_ms or 0.0) / 1000.0
            frames.append(
                MediaFrame(
                    call_id=self.call_id,
                    stream_sid=self.stream_sid or "",
                    track=track,
                    seq=state.frames,
                    t_sec=t_sec,
                    samples=chunk_samples.copy(),
                    sample_rate=SAMPLE_RATE,
                    discontinuous=discontinuous and first,
                    source="twilio",
                )
            )
            state.frames += 1
            self._stats["frames"] += 1
            state.buffer_t_ms = (state.buffer_t_ms or 0.0) + FRAME_MS
            first = False
        return frames

    @staticmethod
    def _parse_int(value: Any, name: str) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise TwilioError(f"{name} is not an integer: {value!r}") from exc

    @staticmethod
    def _parse_float(value: Any, name: str) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise TwilioError(f"{name} is not a number: {value!r}") from exc
