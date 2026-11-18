"""Asterisk External Media (RTP) adapter (TDD v2.0 §4, ADR-009).

Translates Asterisk ARI External Media RTP datagrams into the
canonical 10 ms frame envelope. Pure and transport-agnostic: it
takes raw UDP datagrams and returns frames. It opens no socket,
holds no credential, and knows nothing about ARI or HTTP. The
serving layer owns the UDP socket; see
shieldcall.serve.asterisk_udp for the listener that drives this
class.

Protocol (Asterisk docs, "Channels - Asterisk REST Interface",
POST /channels/externalMedia, since 16.6.0/17.1.0, read
2026-11-18; wire behaviour from production External Media reports,
consistent with RFC 3550):
- The operator creates an External Media channel with
  transport=udp, encapsulation=rtp, connection_type=client,
  direction=both, format=ulaw (or alaw), and external_host set to
  the listener's ip:port. Asterisk then sends RTP v2 datagrams to
  that port: one fixed SSRC per call, sequence numbers starting at
  0 and incrementing by 1 per packet, timestamps starting at 0 and
  incrementing by the payload size in samples (160 per 20 ms
  packet at 8 kHz), 160-byte payloads for ulaw/alaw.
- Only codecs with a static RTP payload type work over RTP
  external media (ulaw PT 0, alaw PT 8, gsm PT 3): there is no SDP
  exchange in which to negotiate a dynamic type, so slin16
  (dynamic PT 118) is not supported over RTP at all. This adapter
  pins format to ulaw or alaw and refuses anything else.
- RTP carries no call identifier and no credential. The binding
  between a UDP flow and a ShieldCall call is the port it arrives
  on: the listener is constructed per call, and the operator opens
  the call with the externalMedia channelId as the call_id. The
  first packet pins the SSRC; a different SSRC later is a
  different stream, not a continuation.

Continuity is judged from the RTP sequence number (the loss
detector) and the RTP timestamp (the presentation clock), both
with wraparound arithmetic. A forward jump in either is a gap: the
next frame is flagged discontinuous and the lost samples are
counted, never filled. A duplicate sequence number or a timestamp
behind the previous payload's end is dropped, never replayed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .frame import FRAME_MS, FRAME_SAMPLES_8K, MediaFrame
from .g711 import alaw_decode, mulaw_decode

SAMPLE_RATE = 8000
"""Asterisk External Media audio is 8 kHz (ulaw/alaw)."""

SAMPLES_PER_MS = SAMPLE_RATE / 1000.0

# Static RTP payload types (RFC 3551) for the formats Asterisk can
# send over RTP external media. GSM (PT 3) is listed by Asterisk but
# is not decoded here: the operator must use ulaw or alaw.
FORMATS = {"ulaw": 0, "alaw": 8}
TRACK_INBOUND = "inbound"
TRACK_OUTBOUND = "outbound"
TRACKS = (TRACK_INBOUND, TRACK_OUTBOUND)


class AsteriskError(ValueError):
    """An Asterisk RTP protocol violation: malformed datagram, wrong
    state, unsupported format, or an SSRC change. The caller should
    close the stream; the adapter never guesses its way past one."""


@dataclass(frozen=True)
class _RtpPacket:
    """One parsed RTP datagram (RFC 3550 header + payload)."""

    marker: bool
    payload_type: int
    seq: int
    timestamp: int
    ssrc: int
    payload: bytes


def _parse_rtp(data: bytes) -> _RtpPacket:
    """Parse one RTP datagram. Raises AsteriskError on anything that
    is not a well-formed RTP v2 packet with a payload."""
    if len(data) < 12:
        raise AsteriskError(f"RTP datagram too short: {len(data)} bytes")
    b0, b1 = data[0], data[1]
    version = (b0 >> 6) & 0x03
    if version != 2:
        raise AsteriskError(f"unsupported RTP version {version}")
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
        raise AsteriskError("RTP datagram shorter than its CSRC list")
    if extension:
        if len(data) < offset + 4:
            raise AsteriskError("RTP datagram shorter than its extension header")
        ext_words = int.from_bytes(data[offset + 2:offset + 4], "big")
        offset += 4 + 4 * ext_words
        if len(data) < offset:
            raise AsteriskError("RTP datagram shorter than its extension")
    end = len(data)
    if padding:
        pad = data[-1]
        if pad == 0 or pad > end - offset:
            raise AsteriskError("RTP padding length is invalid")
        end -= pad
    payload = data[offset:end]
    if not payload:
        raise AsteriskError("RTP datagram has no payload")
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


class AsteriskRtpStream:
    """One Asterisk External Media RTP stream, from first packet to
    `stop()`.

    - call_id: the pre-opened ShieldCall call id. The operator opens
      the call with the externalMedia channelId, and the listener
      binds that channel's UDP port to this stream.
    - format: "ulaw" (PT 0) or "alaw" (PT 8). Anything else is
      refused at construction: the adapter does not guess what a
      codec it was not told to expect sounds like.
    - track: the leg this channel carries ("inbound" by default:
      the remote caller, the leg ShieldCall scores). One
      externalMedia channel carries one leg; the operator creates
      one listener per leg.
    - channel_id: the externalMedia channel id; becomes the frame
      stream_sid. Defaults to call_id.
    """

    def __init__(
        self,
        call_id: str,
        format: str = "ulaw",
        track: str = TRACK_INBOUND,
        channel_id: Optional[str] = None,
    ):
        if not call_id:
            raise AsteriskError("call_id is required")
        if format not in FORMATS:
            raise AsteriskError(
                f"unsupported Asterisk format {format!r} (want 'ulaw' or 'alaw')"
            )
        if track not in TRACKS:
            raise AsteriskError(f"unknown track {track!r}")
        self.call_id = call_id
        self.format = format
        self.payload_type = FORMATS[format]
        self.track = track
        self.stream_sid = channel_id or call_id
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
        a duplicate or out of order). Raises AsteriskError on any
        protocol violation; the caller should close the stream."""
        if self.stopped:
            raise AsteriskError("datagram received after stop")
        if not isinstance(data, (bytes, bytearray)):
            raise AsteriskError("datagram must be bytes")
        packet = _parse_rtp(bytes(data))
        if packet.payload_type != self.payload_type:
            raise AsteriskError(
                f"RTP payload type {packet.payload_type} does not match "
                f"format {self.format!r} (want {self.payload_type})"
            )
        if packet.marker:
            self._stats["marker_packets"] += 1
        self._stats["packets"] += 1
        if not self.started:
            # First packet pins the SSRC and the presentation clock.
            # Asterisk starts both counters at 0 in practice, but the
            # adapter does not assume that: t_sec is measured from
            # this packet's timestamp, whatever it is.
            self.ssrc = packet.ssrc
            self._base_timestamp = packet.timestamp
            self._last_seq = packet.seq
            self._last_timestamp = packet.timestamp
            self._last_payload_samples = len(packet.payload)
            self._buffer_t_sec = 0.0
            self.started = True
            return self._ingest(packet, discontinuous=False)
        if packet.ssrc != self.ssrc:
            self._stats["ssrc_rejected"] += 1
            raise AsteriskError(
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
        not padded: padding would invent audio Asterisk never sent.
        The count is kept in stats so a short final packet is
        visible, not silent."""
        if self.stopped:
            raise AsteriskError("second stop")
        self._stats["discarded_on_stop"] += int(self._buffer.size)
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_t_sec = None
        self.stopped = True

    # -- helpers --------------------------------------------------------

    def _ingest(self, packet: _RtpPacket, discontinuous: bool) -> List[MediaFrame]:
        decode = mulaw_decode if self.format == "ulaw" else alaw_decode
        samples = decode(packet.payload)
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
                    source="asterisk",
                )
            )
            self._frames += 1
            self._stats["frames"] += 1
            self._buffer_t_sec = (self._buffer_t_sec or 0.0) + FRAME_MS / 1000.0
            first = False
        return frames
