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

The RTP parsing and continuity discipline live in the shared
shieldcall.ingest.rtp module (used by the SIPREC adapter too);
this module is only the Asterisk binding: format names, the
error type, and the stream class.
"""

from __future__ import annotations

from typing import Optional

from .g711 import alaw_decode, mulaw_decode
from .rtp import RtpError, RtpStream

SAMPLE_RATE = 8000
"""Asterisk External Media audio is 8 kHz (ulaw/alaw)."""

# Static RTP payload types (RFC 3551) for the formats Asterisk can
# send over RTP external media. GSM (PT 3) is listed by Asterisk but
# is not decoded here: the operator must use ulaw or alaw.
FORMATS = {"ulaw": 0, "alaw": 8}
TRACK_INBOUND = "inbound"
TRACK_OUTBOUND = "outbound"
TRACKS = (TRACK_INBOUND, TRACK_OUTBOUND)


class AsteriskError(RtpError):
    """An Asterisk RTP protocol violation: malformed datagram, wrong
    state, unsupported format, or an SSRC change. The caller should
    close the stream; the adapter never guesses its way past one."""


class AsteriskRtpStream(RtpStream):
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
        if format not in FORMATS:
            raise AsteriskError(
                f"unsupported Asterisk format {format!r} (want 'ulaw' or 'alaw')"
            )
        if track not in TRACKS:
            raise AsteriskError(f"unknown track {track!r}")
        decode = mulaw_decode if format == "ulaw" else alaw_decode
        super().__init__(
            call_id,
            stream_sid=channel_id or call_id,
            track=track,
            payload_type=FORMATS[format],
            decode=decode,
            source="asterisk",
            ssrc_change="error",
            error_cls=AsteriskError,
        )
        self.format = format
