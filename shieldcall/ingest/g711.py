"""G.711 codecs for the Media Ingest Fabric (TDD v2.0 §4).

Exact ITU-T G.711 mu-law and A-law decode tables, shared by the
provider adapters: Twilio sends mu-law, Asterisk External Media
sends mu-law or A-law, and the Telnyx adapter (PLANNED) will reuse
this module when its turn comes. Both tables decode to float32 mono
PCM in [-1, 1]. No encoder lives here: tests encode by nearest
match over the decode table, which is correct by construction
against the table itself.
"""

from __future__ import annotations

import numpy as np


def _build_mulaw_table() -> np.ndarray:
    """Exact G.711 mu-law decode table (ITU-T G.711), as float32 in
    [-1, 1]. Byte 0xFF decodes to 0.0, 0x00 to -32124/32768, 0x80 to
    +32124/32768; the table is verified against those vectors in
    tests/test_twilio_ingest.py."""
    table = np.empty(256, dtype=np.float32)
    for byte in range(256):
        u = (~byte) & 0xFF
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        sample = ((mantissa << 3) + 0x84) << exponent
        sample -= 0x84
        if sign:
            sample = -sample
        table[byte] = sample / 32768.0
    return table


def _build_alaw_table() -> np.ndarray:
    """Exact G.711 A-law decode table (ITU-T G.711), as float32 in
    [-1, 1]. The byte is XORed with 0x55 before decoding; bit 7 is
    the sign (set = positive). Known vectors, verified in
    tests/test_asterisk_ingest.py: 0xD5 -> +8/32768, 0x55 ->
    -8/32768, 0xA5 -> +16896/32768, 0x25 -> -16896/32768, and the
    maximum magnitude is 32256/32768."""
    table = np.empty(256, dtype=np.float32)
    for byte in range(256):
        a = byte ^ 0x55
        sign = a & 0x80
        exponent = (a >> 4) & 0x07
        mantissa = a & 0x0F
        if exponent == 0:
            sample = (mantissa << 4) + 8
        elif exponent == 1:
            sample = (mantissa << 4) + 0x108
        else:
            sample = ((mantissa << 4) + 0x108) << (exponent - 1)
        if not sign:
            sample = -sample
        table[byte] = sample / 32768.0
    return table


_MULAW_TABLE = _build_mulaw_table()
_ALAW_TABLE = _build_alaw_table()


def mulaw_decode(raw: bytes) -> np.ndarray:
    """Decode G.711 mu-law bytes to float32 mono PCM in [-1, 1]."""
    if not raw:
        return np.empty(0, dtype=np.float32)
    return _MULAW_TABLE[np.frombuffer(raw, dtype=np.uint8)].copy()


def alaw_decode(raw: bytes) -> np.ndarray:
    """Decode G.711 A-law bytes to float32 mono PCM in [-1, 1]."""
    if not raw:
        return np.empty(0, dtype=np.float32)
    return _ALAW_TABLE[np.frombuffer(raw, dtype=np.uint8)].copy()
