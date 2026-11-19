"""Media Ingest Fabric: provider adapters translating wire audio into the
canonical 10 ms frame envelope (TDD v2.0 §4).
"""

from .asterisk import AsteriskError, AsteriskRtpStream
from .frame import FRAME_MS, FRAME_SAMPLES_8K, MediaFrame
from .g711 import alaw_decode, mulaw_decode
from .rtp import RtpError, RtpStream
from .siprec import (
    SiprecError,
    SiprecRecordingSession,
    SiprecSipHandler,
    parse_sdp_offer,
    parse_siprec_metadata,
)
from .twilio import TwilioError, TwilioMediaStream

__all__ = [
    "FRAME_MS",
    "FRAME_SAMPLES_8K",
    "MediaFrame",
    "AsteriskError",
    "AsteriskRtpStream",
    "RtpError",
    "RtpStream",
    "SiprecError",
    "SiprecRecordingSession",
    "SiprecSipHandler",
    "parse_sdp_offer",
    "parse_siprec_metadata",
    "TwilioError",
    "TwilioMediaStream",
    "alaw_decode",
    "mulaw_decode",
]
