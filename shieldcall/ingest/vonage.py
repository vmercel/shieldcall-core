"""Vonage Voice API WebSocket adapter (TDD v2.0 section 4, Media Ingest Fabric).

Translates Vonage's WebSocket protocol into the canonical 10 ms
frame envelope. Pure and transport-agnostic: it takes parsed JSON
text messages and raw binary audio and returns frames. It opens no
socket, holds no credential, and knows nothing about HTTP. The
serving layer owns auth and the websocket; see
shieldcall.serve.http_app for the /v1/ingest/vonage route that
drives this class.

Protocol (Vonage docs, "WebSocket Voice Chat API Guide",
developer.vonage.com, read 2026-11-21):
- The operator connects a call with an NCCO `connect` action whose
  endpoint is {"type": "websocket", "uri": "wss://...",
  "content-type": "audio/l16;rate=8000", "headers": {...}}.
  The rate is the operator's choice: 8000, 16000 (the doc's
  recommended rate for speech recognition) or 24000.
- Text, first: {"event": "websocket:connected",
    "content-type": "audio/l16;rate=16000", ...}
  The NCCO `headers` are flattened into top-level fields of this
  message alongside `event` and `content-type`. A reconnection
  sends the same message again with "reconnect_attempt": 1
  (counting up on later attempts).
- Binary: raw PCM audio captured from the caller, 16-bit signed
  little-endian, at the negotiated rate, each packet about 20 ms.
- Text, interleaved: {"event": "websocket:dtmf", "digit": "5",
    "duration": 260} (one event per keypress, digit 0-9, * or #,
    duration in milliseconds).
- Text, only as acks to outbound commands this adapter never
  sends: {"event": "websocket:cleared"} and
  {"event": "websocket:notify", "payload": {...}}.
- No stop event exists over the WebSocket. The end of the stream
  is the socket closing; call status arrives on the operator's
  HTTP eventUrl webhook, not on the socket.

What the protocol does NOT provide, and this adapter does not
pretend otherwise: no stream identifier, no call identifier, no
sequence number, no timestamp, and no per-message track label.
A lost packet is indistinguishable from unsent silence, so no
gap accounting is possible except after a reconnection, which
the docs say loses the audio sent during the gap. Presentation
time is therefore counted from the samples received, and the
single leg is reported as "inbound" (see ADR-012).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .frame import FRAME_SAMPLES_8K, MediaFrame

__all__ = ["VonageError", "VonageMediaStream"]

SAMPLE_RATE = 8000
"""The only sample rate this adapter accepts (see ADR-012)."""

TRACK_INBOUND = "inbound"
"""The single leg Vonage delivers; there is no track label on the wire."""

DTMF_DIGITS = frozenset("0123456789*#")

# JSON event names Vonage sends to the application (docs above).
EVENT_CONNECTED = "websocket:connected"
EVENT_DTMF = "websocket:dtmf"
EVENT_CLEARED = "websocket:cleared"
EVENT_NOTIFY = "websocket:notify"


class VonageError(ValueError):
    """A Vonage protocol violation: malformed message, wrong state, or
    an unsupported media format. The caller should close the stream;
    the adapter never guesses its way past one."""


class VonageMediaStream:
    """One Vonage WebSocket stream, from `websocket:connected` to the
    socket closing.

    - call_id: the pre-opened ShieldCall call id. The operator opens
      the call with the Vonage call uuid, and puts the same id in
      the websocket URI as a query parameter, because Vonage sends
      no call identifier over the socket itself (ADR-012 item 9).
      The adapter cannot check the binding in-band; the route owns
      it, and a distinct URI per call is an operator obligation.
    """

    def __init__(self, call_id: str):
        if not call_id:
            raise VonageError("call_id is required")
        self.call_id = call_id
        self.started = False
        self.finished = False
        self.content_type: Optional[str] = None
        # NCCO headers flattened into websocket:connected, recorded
        # for audit only. No header is treated as an identity: the
        # docs promise no particular header.
        self.headers: Dict[str, Any] = {}
        self.reconnect_attempt: int = 0
        self.dtmf_events: List[Dict[str, Any]] = []
        self.notify_payloads: List[Any] = []
        self._buffer = np.empty(0, dtype=np.float32)
        # Samples emitted as frames so far; the presentation clock,
        # counted from received samples because Vonage sends none.
        self._emitted_samples = 0
        self._frames = 0
        # True when a reconnection gap preceded the next frame: the
        # docs say audio sent during the gap is not delivered, so the
        # first frame after a reconnect is flagged discontinuous.
        # Cleared once that frame is emitted.
        self._gap_before_next_frame = False
        self._stats: Dict[str, int] = {
            "messages": 0,
            "audio_packets": 0,
            "audio_bytes": 0,
            "samples": 0,
            "frames": 0,
            "dtmf": 0,
            "notifications": 0,
            "cleared": 0,
            "reconnects": 0,
            "discarded_on_close": 0,
            "discarded_on_reconnect": 0,
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
        """Route one parsed Vonage JSON text message. Returns the
        canonical frames it produced (none: text messages never
        carry audio). Raises VonageError on any protocol violation."""
        if not isinstance(msg, dict):
            raise VonageError("message must be a JSON object")
        self._stats["messages"] += 1
        event = msg.get("event")
        if event == EVENT_CONNECTED:
            return self._on_connected(msg)
        if event == EVENT_DTMF:
            return self._on_dtmf(msg)
        if event == EVENT_CLEARED:
            return self._on_cleared(msg)
        if event == EVENT_NOTIFY:
            return self._on_notify(msg)
        raise VonageError(f"unknown Vonage event {event!r}")

    def handle_audio(self, data: bytes) -> List[MediaFrame]:
        """Translate one binary WebSocket message (raw L16 PCM) into
        canonical frames. Raises VonageError on any protocol
        violation."""
        if self.finished:
            raise VonageError("audio received after the stream finished")
        if not self.started:
            raise VonageError("audio received before websocket:connected")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise VonageError("audio must be bytes")
        raw = bytes(data)
        if not raw:
            raise VonageError("empty audio packet")
        if len(raw) % 2:
            raise VonageError(
                f"audio packet has odd byte count {len(raw)}: "
                "a truncated 16-bit sample"
            )
        self._stats["audio_packets"] += 1
        self._stats["audio_bytes"] += len(raw)
        # 16-bit signed little-endian PCM, scaled to [-1, 1]. The
        # divisor is 32768 so that -32768 maps to exactly -1.0.
        samples = (
            np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        )
        self._stats["samples"] += int(samples.size)
        self._buffer = (
            np.concatenate([self._buffer, samples])
            if self._buffer.size
            else samples
        )
        return self._drain_frames()

    def finish(self) -> None:
        """End the stream when the socket closes. Vonage sends no stop
        event, so this is route-side cleanup, and it is idempotent:
        a second call must not raise. A partial trailing frame is
        discarded, not padded: padding would invent audio the
        caller never sent. The count is kept in stats so a short
        final packet is visible, not silent."""
        if self.finished:
            return
        self._stats["discarded_on_close"] += int(self._buffer.size)
        self._buffer = np.empty(0, dtype=np.float32)
        self.finished = True

    # -- events ---------------------------------------------------------

    def _on_connected(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if self.finished:
            raise VonageError("websocket:connected received after finish")
        reconnect_attempt = self._parse_reconnect_attempt(
            msg.get("reconnect_attempt")
        )
        content_type = msg.get("content-type")
        self._check_content_type(content_type)
        if self.started:
            # A second connected on one instance is only legal as a
            # reconnect announcement. The route never sends one: it
            # creates a fresh instance per connection. Accepting it
            # here keeps the adapter honest about the gap it
            # implies: buffered audio from before the gap straddles
            # it and is discarded, not spliced across it.
            if reconnect_attempt < 1:
                raise VonageError("second websocket:connected message")
            self._stats["reconnects"] += 1
            self._stats["discarded_on_reconnect"] += int(self._buffer.size)
            self._buffer = np.empty(0, dtype=np.float32)
            self._gap_before_next_frame = True
        else:
            if reconnect_attempt >= 1:
                # A fresh instance whose first connected is already a
                # reconnect: audio was lost in the gap before this
                # connection, so the first frame is discontinuous.
                self._gap_before_next_frame = True
            self.started = True
        self.reconnect_attempt = reconnect_attempt
        self.content_type = str(content_type)
        self.headers = {
            key: value
            for key, value in msg.items()
            if key not in ("event", "content-type", "reconnect_attempt")
        }
        return []

    def _on_dtmf(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if self.finished:
            raise VonageError("websocket:dtmf received after finish")
        if not self.started:
            raise VonageError("websocket:dtmf received before websocket:connected")
        digit = msg.get("digit")
        if not isinstance(digit, str) or digit not in DTMF_DIGITS:
            raise VonageError(f"dtmf digit {digit!r} is not a keypad digit")
        duration_ms: Optional[int] = None
        if msg.get("duration") is not None:
            duration_ms = self._parse_int(msg.get("duration"), "dtmf duration")
            if duration_ms < 0:
                raise VonageError(f"dtmf duration {duration_ms} is negative")
        # DTMF is interleaved between audio frames; the digit and its
        # duration are recorded with no frame produced.
        self.dtmf_events.append({"digit": digit, "duration_ms": duration_ms})
        self._stats["dtmf"] += 1
        return []

    def _on_cleared(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if self.finished:
            raise VonageError("websocket:cleared received after finish")
        if not self.started:
            raise VonageError(
                "websocket:cleared received before websocket:connected"
            )
        # An ack for a clear command this adapter never sends.
        # Recorded and counted, not acted on.
        self._stats["cleared"] += 1
        return []

    def _on_notify(self, msg: Dict[str, Any]) -> List[MediaFrame]:
        if self.finished:
            raise VonageError("websocket:notify received after finish")
        if not self.started:
            raise VonageError(
                "websocket:notify received before websocket:connected"
            )
        # An ack for a notify command this adapter never sends.
        # The payload is recorded, not acted on.
        self.notify_payloads.append(msg.get("payload"))
        self._stats["notifications"] += 1
        return []

    # -- helpers --------------------------------------------------------

    def _drain_frames(self) -> List[MediaFrame]:
        frames: List[MediaFrame] = []
        first = True
        while self._buffer.size >= FRAME_SAMPLES_8K:
            chunk_samples = self._buffer[:FRAME_SAMPLES_8K]
            self._buffer = self._buffer[FRAME_SAMPLES_8K:]
            frames.append(
                MediaFrame(
                    call_id=self.call_id,
                    stream_sid="",
                    track=TRACK_INBOUND,
                    seq=self._frames,
                    t_sec=self._emitted_samples / SAMPLE_RATE,
                    samples=chunk_samples.copy(),
                    sample_rate=SAMPLE_RATE,
                    discontinuous=self._gap_before_next_frame and first,
                    source="vonage",
                )
            )
            self._frames += 1
            self._stats["frames"] += 1
            self._emitted_samples += FRAME_SAMPLES_8K
            first = False
        if frames:
            self._gap_before_next_frame = False
        return frames

    @staticmethod
    def _check_content_type(value: Any) -> None:
        """The negotiated format must be audio/l16 at 8000 Hz. Any
        other encoding or rate is refused, never silently
        resampled (ADR-012 item 4)."""
        if not isinstance(value, str) or not value.strip():
            raise VonageError("websocket:connected has no content-type")
        parts = [part.strip() for part in value.split(";")]
        encoding = parts[0].lower()
        params: Dict[str, str] = {}
        for part in parts[1:]:
            if "=" in part:
                key, _, val = part.partition("=")
                params[key.strip().lower()] = val.strip().strip('"')
        if encoding != "audio/l16":
            raise VonageError(
                f"unsupported Vonage content-type {value!r} "
                "(supported: audio/l16;rate=8000)"
            )
        rate = params.get("rate")
        if rate is None:
            raise VonageError(
                f"Vonage content-type {value!r} has no rate parameter"
            )
        try:
            rate_hz = int(rate)
        except ValueError as exc:
            raise VonageError(
                f"Vonage content-type rate {rate!r} is not an integer"
            ) from exc
        if rate_hz != SAMPLE_RATE:
            raise VonageError(
                f"unsupported Vonage sample rate {rate_hz} "
                f"(supported: {SAMPLE_RATE})"
            )

    @staticmethod
    def _parse_reconnect_attempt(value: Any) -> int:
        if value is None:
            return 0
        try:
            attempt = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise VonageError(
                f"reconnect_attempt {value!r} is not an integer"
            ) from exc
        if attempt < 0:
            raise VonageError(f"reconnect_attempt {attempt} is negative")
        return attempt

    @staticmethod
    def _parse_int(value: Any, name: str) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise VonageError(f"{name} is not an integer: {value!r}") from exc
