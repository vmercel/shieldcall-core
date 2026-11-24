"""Opus decoding through the system libopus, via ctypes.

The Media Ingest Fabric's in-app adapter (ADR-015, TDD v2.0 section 4)
receives Opus packets as the app's Encoded Transform emits them. Python
has no Opus decoder in its standard library, and this repo pins no
Opus package, so decoding goes straight to the C library the system
already ships (libopus 1.4 on the dev host). Nothing here shells out
and nothing is invented: a packet that libopus rejects is an error,
and a host with no libopus refuses Opus streams at hello time rather
than pretending to decode them.

Only decoding is provided. Encoding belongs to the app (its WebRTC
stack), never to the sidecar; the test suite carries its own tiny
ctypes encoder so the decode path can be exercised end to end
without a second implementation in production code.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Optional

import numpy as np

__all__ = ["OpusDecoder", "OpusDecoderUnavailable", "OpusError", "load_libopus"]

# Opus output rates libopus accepts for decoding (RFC 7845 section 3).
OUTPUT_RATES = (8000, 12000, 16000, 24000, 48000)

# Longest frame libopus can return: 120 ms (RFC 7845 section 3.2).
_MAX_FRAME_MS = 120


class OpusError(ValueError):
    """libopus rejected a packet, or a decoder could not be created."""


class OpusDecoderUnavailable(RuntimeError):
    """No usable libopus was found on this host. Opus streams must be
    refused, never silently dropped or guessed past."""


_libopus: Optional[ctypes.CDLL] = None
_libopus_failed = False


def load_libopus() -> ctypes.CDLL:
    """Load libopus once, and configure the ctypes signatures.

    Raises OpusDecoderUnavailable when no library can be loaded, so
    the caller can refuse the stream with a plain-English reason
    instead of a ctypes traceback.
    """
    global _libopus, _libopus_failed
    if _libopus is not None:
        return _libopus
    if _libopus_failed:
        raise OpusDecoderUnavailable(
            "libopus is not available on this host: Opus decoding "
            "needs the system libopus library (see ADR-015 item 4)"
        )
    name = ctypes.util.find_library("opus") or "libopus.so.0"
    try:
        lib = ctypes.CDLL(name)
    except OSError as exc:
        _libopus_failed = True
        raise OpusDecoderUnavailable(
            f"could not load libopus ({name}): {exc}; Opus decoding "
            "needs the system libopus library (see ADR-015 item 4)"
        ) from exc
    lib.opus_decoder_create.argtypes = [
        ctypes.c_int32,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.opus_decoder_create.restype = ctypes.c_void_p
    lib.opus_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_short),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.opus_decode.restype = ctypes.c_int
    lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_decoder_destroy.restype = None
    lib.opus_strerror.argtypes = [ctypes.c_int]
    lib.opus_strerror.restype = ctypes.c_char_p
    _libopus = lib
    return lib


class OpusDecoder:
    """One Opus decoder, decoding to float32 mono PCM at `output_rate`.

    The decoder is created at the output rate, so libopus does the
    rate conversion internally: an Opus packet encoded at 48 kHz
    decoded at 8 kHz yields genuine 8 kHz audio, never upsampled
    audio pretending to carry wider bandwidth (TDD section 4.1).
    One decoder per stream; libopus decoder state is not shared.
    """

    def __init__(self, output_rate: int = 8000, channels: int = 1):
        if output_rate not in OUTPUT_RATES:
            raise OpusError(
                f"unsupported Opus output rate {output_rate} "
                f"(supported: {', '.join(map(str, OUTPUT_RATES))})"
            )
        if channels != 1:
            raise OpusError("only mono Opus is supported (channels must be 1)")
        self.output_rate = output_rate
        self.channels = channels
        self._lib = load_libopus()
        err = ctypes.c_int(0)
        handle = self._lib.opus_decoder_create(
            output_rate, channels, ctypes.byref(err)
        )
        if not handle or err.value != 0:
            raise OpusError(
                f"opus_decoder_create failed: "
                f"{self._lib.opus_strerror(err.value).decode('utf-8', 'replace')}"
            )
        self._handle = handle
        self._closed = False
        self._max_samples = output_rate * _MAX_FRAME_MS // 1000
        self._out = (ctypes.c_short * self._max_samples)()

    def decode(self, packet: bytes) -> np.ndarray:
        """Decode one Opus packet to float32 samples in [-1, 1].

        No packet-loss concealment is ever requested (decode_fec=0):
        a missing packet is a gap the adapter flags, never audio the
        decoder invents (ADR-015 item 7). An empty packet and a
        packet libopus rejects both raise OpusError.
        """
        if self._closed:
            raise OpusError("Opus decoder is closed")
        if not isinstance(packet, (bytes, bytearray)):
            raise OpusError("Opus packet must be bytes")
        if len(packet) == 0:
            raise OpusError("empty Opus packet: a lost packet is a gap, not a packet")
        count = self._lib.opus_decode(
            self._handle, bytes(packet), len(packet), self._out, self._max_samples, 0
        )
        if count < 0:
            raise OpusError(
                "libopus rejected the Opus packet: "
                f"{self._lib.opus_strerror(count).decode('utf-8', 'replace')}"
            )
        if count == 0:
            raise OpusError("Opus packet decoded to zero samples")
        samples = np.frombuffer(self._out, dtype=np.int16, count=count)
        return (samples.astype(np.float32) / 32768.0).copy()

    def close(self) -> None:
        """Destroy the decoder. Idempotent: closing is cleanup."""
        if not self._closed:
            self._closed = True
            if self._handle:
                self._lib.opus_decoder_destroy(self._handle)
                self._handle = None

    def __enter__(self) -> "OpusDecoder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
