"""Shared RTP machinery for the Media Ingest Fabric (TDD v2.0 §4).

The Twilio adapter has its own JSON framing and does not use this
module. The UDP/RTP adapters (Asterisk External Media, ADR-009;
SIPREC, ADR-010) share one wire format, RFC 3550, and one
continuity discipline, so they share one implementation:
`_parse_rtp` for the datagram, `_seq_diff` / `_ts_diff` for
wraparound arithmetic, and `RtpStream` for the state machine that
turns a datagram flow into canonical 10 ms frames.

Continuity is judged from the RTP sequence number (the loss
detector) and the RTP timestamp (the presentation clock), both
with wraparound arithmetic. A forward jump in either is a gap:
the next frame is flagged discontinuous and the lost samples are
counted, never filled. A duplicate sequence number or a timestamp
behind the previous payload's end is dropped, never replayed. A
partial frame buffers across packets and is discarded, not
spliced, across a gap. `stop()` discards a partial trailing
frame, never padding it: padding would invent audio the sender
never sent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .frame import FRAME_MS, FRAME_SAMPLES_8K, MediaFrame

SAMPLE_RATE = 8000
"""Telephony adapters run at 8 kHz (G.711)."""

SAMPLES_PER_MS = SAMPLE_RATE / 1000.0

DecodeFn = Callable[[bytes], np.ndarray]
"""A codec decode function: encoded bytes -> float32 mono PCM in
[-1, 1]."""


class RtpError(ValueError):
    """An RTP protocol violation: malformed datagram, wrong state,
    unsupported format, or an SSRC change on a stream that refuses
    one. The caller should close the stream; the adapter never
    guesses its way past one."""


@dataclass(frozen=True)
class _RtpPacket:
    """One parsed RTP datagram (RFC 3550 header + payload)."""

    marker: bool
    payload_type: int
    seq: int
    timestamp: int
    ssrc: int
    payload: bytes


def _parse_rtp(data: bytes, error_cls: type = RtpError) -> _RtpPacket:
    """Parse one RTP datagram. Raises error_cls on anything that is
    not a well-formed RTP v2 packet with a payload."""
    if len(data) < 12:
        raise error_cls(f"RTP datagram too short: {len(data)} bytes")
    b0, b1 = data[0], data[1]
    version = (b0 >> 6) & 0x03
    if version != 2:
        raise error_cls(f"unsupported RTP version {version}")
    padding = bool(b0 & 0x20)
    extension = bool(b0 & 0x10)
    csrc_count = b0 & 0x0F
    marker = bool(b1 & 0x80)
    payload_type = b1 & 0x7F
    seq = int.from_bytes(data[2:4], "big")
    timestamp = int.from_bytes(data[4:8], "big")
    ssrc = int.from_bytes(data[8:12], "big")
    offset = 12 + 4 * csrc_count
    if len(data) < offset:
        raise error_cls("RTP datagram shorter than its CSRC list")
    if extension:
        if len(data) < offset + 4:
            raise error_cls("RTP datagram shorter than its extension header")
        ext_words = int.from_bytes(data[offset + 2:offset + 4], "big")
        offset += 4 + 4 * ext_words
        if len(data) < offset:
            raise error_cls("RTP datagram shorter than its extension")
    end = len(data)
    if padding:
        pad = data[-1]
        if pad == 0 or pad > end - offset:
            raise error_cls("RTP padding length is invalid")
        end -= pad
    payload = data[offset:end]
    if not payload:
        raise error_cls("RTP datagram has no payload")
    return _RtpPacket(
        marker=marker,
        payload_type=payload_type,
        seq=seq,
        timestamp=timestamp,
        ssrc=ssrc,
        payload=payload,
    )


def _seq_diff(new: int, old: int) -> int:
    """Signed sequence-number difference with 16-bit wraparound:
    in (-32768, 32768]. Positive means new is ahead of old."""
    diff = (new - old) & 0xFFFF
    if diff >= 0x8000:
        diff -= 0x10000
    return diff


def _ts_diff(new: int, old: int) -> int:
    """Signed timestamp difference with 32-bit wraparound."""
    diff = (new - old) & 0xFFFFFFFF
    if diff >= 0x80000000:
        diff -= 0x100000000
    return diff


class RtpStream:
    """One RTP stream, from first packet to `stop()`.

    - call_id: the pre-opened ShieldCall call id.
    - stream_sid: the provider's stream identifier; becomes the
      frame stream_sid.
    - track: the leg this stream carries ("inbound" is the remote
      caller, the leg ShieldCall scores; "outbound" is the local
      leg). Tracks are never mixed.
    - payload_type: the expected static RTP payload type; any
      other type is an RtpError, never a silent transcode.
    - decode: the codec decode function for that payload type.
    - source: adapter name recorded on every frame for audit
      ("asterisk", "siprec", ...); never used for branching.
    - ssrc_change: what an SSRC change mid-stream means.
      "error" (default) raises: a new SSRC is a different stream.
      "repin" resets the continuity state and flags the next
      frame discontinuous, for protocols whose RFC anticipates
      an SSRC change (SIPREC, RFC 7865 §8).
    - error_cls: the exception type raised for protocol
      violations, so each adapter keeps its own error class
      (AsteriskError, SiprecError) while sharing the machine.
    """

    def __init__(
        self,
        call_id: str,
        stream_sid: str,
        track: str,
        payload_type: int,
        decode: DecodeFn,
        source: str,
        ssrc_change: str = "error",
        error_cls: type = RtpError,
    ):
        if not call_id:
            raise error_cls("call_id is required")
        if ssrc_change not in ("error", "repin"):
            raise error_cls(f"unknown ssrc_change policy {ssrc_change!r}")
        self.call_id = call_id
        self.stream_sid = stream_sid
        self.track = track
        self.payload_type = payload_type
        self._decode = decode
        self.source = source
        self.ssrc_change = ssrc_change
        self._error_cls = error_cls
        self.started = False
        self.stopped = False
        self.ssrc: Optional[int] = None
        self._base_timestamp: Optional[int] = None
        self._last_seq: Optional[int] = None
        self._last_timestamp: Optional[int] = None
        self._last_payload_samples: int = 0
        self._buffer = np.empty(0, dtype=np.float32)
        # Presentation time (seconds from stream start) of the first
        # buffered sample not yet emitted as a frame.
        self._buffer_t_sec: Optional[float] = None
        self._frames = 0
        self._stats: Dict[str, int] = {
            "packets": 0,
            "frames": 0,
            "samples": 0,
            "gaps": 0,
            "lost_samples": 0,
            "duplicates_dropped": 0,
            "out_of_order_dropped": 0,
            "discarded_on_stop": 0,
            "marker_packets": 0,
            "ssrc_rejected": 0,
            "ssrc_repinned": 0,
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

    def handle_datagram(self, data: bytes) -> List[MediaFrame]:
        """Translate one raw UDP datagram into canonical frames.
        Returns the frames it produced (empty when the datagram was
        a duplicate or out of order). Raises RtpError on any
        protocol violation; the caller should close the stream."""
        if self.stopped:
            raise self._error_cls("datagram received after stop")
        if not isinstance(data, (bytes, bytearray)):
            raise self._error_cls("datagram must be bytes")
        packet = _parse_rtp(bytes(data), self._error_cls)
        if packet.payload_type != self.payload_type:
            raise self._error_cls(
                f"RTP payload type {packet.payload_type} does not match "
                f"expected {self.payload_type}"
            )
        if packet.marker:
            self._stats["marker_packets"] += 1
        self._stats["packets"] += 1
        if not self.started:
            # First packet pins the SSRC and the presentation clock.
            # The adapter does not assume the counters start at 0:
            # t_sec is measured from this packet's timestamp,
            # whatever it is.
            self.ssrc = packet.ssrc
            self._base_timestamp = packet.timestamp
            self._last_seq = packet.seq
            self._last_timestamp = packet.timestamp
            self._last_payload_samples = len(packet.payload)
            self._buffer_t_sec = 0.0
            self.started = True
            return self._ingest(packet, discontinuous=False)
        if packet.ssrc != self.ssrc:
            if self.ssrc_change == "repin":
                # The sender restarted or resolved an SSRC
                # conflict (RFC 7865 §8 anticipates this): reset
                # the continuity state and keep going, flagging
                # the next frame discontinuous. The buffered
                # partial frame belongs to the old epoch and is
                # discarded, not spliced across the reset.
                self._stats["ssrc_repinned"] += 1
                self._stats["gaps"] += 1
                if self._buffer.size:
                    self._stats["lost_samples"] += int(self._buffer.size)
                    self._buffer = np.empty(0, dtype=np.float32)
                self.ssrc = packet.ssrc
                self._base_timestamp = packet.timestamp
                self._last_seq = packet.seq
                self._last_timestamp = packet.timestamp
                self._last_payload_samples = len(packet.payload)
                self._buffer_t_sec = 0.0
                return self._ingest(packet, discontinuous=True)
            self._stats["ssrc_rejected"] += 1
            raise self._error_cls(
                f"RTP SSRC changed mid-stream ({packet.ssrc:#x} != "
                f"{self.ssrc:#x}); a new SSRC is a different stream"
            )
        seq_diff = _seq_diff(packet.seq, self._last_seq or 0)
        if seq_diff == 0:
            self._stats["duplicates_dropped"] += 1
            return []
        if seq_diff < 0:
            self._stats["out_of_order_dropped"] += 1
            return []
        ts_diff = _ts_diff(packet.timestamp, self._last_timestamp or 0)
        if ts_diff < 0:
            self._stats["out_of_order_dropped"] += 1
            return []
        discontinuous = False
        if seq_diff > 1:
            # Sequence jump: lost packets, each carrying as many
            # samples as the last payload did.
            lost_packets = seq_diff - 1
            self._stats["gaps"] += 1
            self._stats["lost_samples"] += lost_packets * self._last_payload_samples
            discontinuous = True
        elif ts_diff > self._last_payload_samples:
            # Sequence continuous but the presentation clock jumped:
            # a gap the sequence number did not see (e.g. a clock
            # reset upstream). Count it from the timestamp delta.
            self._stats["gaps"] += 1
            self._stats["lost_samples"] += ts_diff - self._last_payload_samples
            discontinuous = True
        elif ts_diff < self._last_payload_samples:
            # Sequence advanced but the clock did not: the packet
            # overlaps audio already received. Drop it rather than
            # double-counting those samples.
            self._stats["out_of_order_dropped"] += 1
            return []
        if discontinuous:
            # Restart the buffer clock at the new packet: the old
            # clock points at the pre-gap timeline, and stamping the
            # new audio with it would lie about when it was spoken.
            # A partial buffered frame straddles the gap: its samples
            # end before the new audio starts, so discard it rather
            # than splicing two timelines into one frame.
            self._buffer_t_sec = (
                _ts_diff(packet.timestamp, self._base_timestamp or 0) / SAMPLE_RATE
            )
            if self._buffer.size:
                self._stats["lost_samples"] += int(self._buffer.size)
                self._buffer = np.empty(0, dtype=np.float32)
        self._last_seq = packet.seq
        self._last_timestamp = packet.timestamp
        self._last_payload_samples = len(packet.payload)
        if self._buffer_t_sec is None:
            self._buffer_t_sec = _ts_diff(packet.timestamp, self._base_timestamp or 0) / SAMPLE_RATE
        return self._ingest(packet, discontinuous=discontinuous)

    def stop(self) -> None:
        """End the stream. A partial trailing frame is discarded,
        not padded: padding would invent audio the sender never
        sent. The count is kept in stats so a short final packet
        is visible, not silent."""
        if self.stopped:
            raise self._error_cls("second stop")
        self._stats["discarded_on_stop"] += int(self._buffer.size)
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_t_sec = None
        self.stopped = True

    # -- helpers --------------------------------------------------------

    def _ingest(self, packet: _RtpPacket, discontinuous: bool) -> List[MediaFrame]:
        samples = self._decode(packet.payload)
        self._buffer = (
            np.concatenate([self._buffer, samples]) if self._buffer.size else samples
        )
        self._stats["samples"] += int(samples.size)
        if self._buffer_t_sec is None:
            self._buffer_t_sec = (
                _ts_diff(packet.timestamp, self._base_timestamp or 0) / SAMPLE_RATE
            )
        return self._drain_frames(discontinuous)

    def _drain_frames(self, discontinuous: bool) -> List[MediaFrame]:
        frames: List[MediaFrame] = []
        first = True
        while self._buffer.size >= FRAME_SAMPLES_8K:
            chunk_samples = self._buffer[:FRAME_SAMPLES_8K]
            self._buffer = self._buffer[FRAME_SAMPLES_8K:]
            frames.append(
                MediaFrame(
                    call_id=self.call_id,
                    stream_sid=self.stream_sid,
                    track=self.track,
                    seq=self._frames,
                    t_sec=self._buffer_t_sec or 0.0,
                    samples=chunk_samples.copy(),
                    sample_rate=SAMPLE_RATE,
                    discontinuous=discontinuous and first,
                    source=self.source,
                )
            )
            self._frames += 1
            self._stats["frames"] += 1
            self._buffer_t_sec = (self._buffer_t_sec or 0.0) + FRAME_MS / 1000.0
            first = False
        return frames
