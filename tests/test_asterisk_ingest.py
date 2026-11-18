"""Asterisk External Media (RTP) adapter (ADR-009).

Protocol vectors come from Asterisk's own docs (POST
/channels/externalMedia, since 16.6.0/17.1.0) and production
External Media reports: RTP v2, one fixed SSRC per call, sequence
numbers starting at 0 incrementing by 1 per packet, timestamps
starting at 0 incrementing by the payload size in samples, 160-byte
payloads (20 ms) for ulaw/alaw at 8 kHz, and only static payload
types over RTP (ulaw PT 0, alaw PT 8).
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from shieldcall.ingest.asterisk import AsteriskError, AsteriskRtpStream
from shieldcall.ingest.frame import FRAME_SAMPLES_8K
from shieldcall.ingest.g711 import _ALAW_TABLE, _MULAW_TABLE, alaw_decode

CALL_ID = "ext-13094"
CHANNEL_ID = "ext-13094"
SSRC = 0x12345678


def _encode(table: np.ndarray, samples: np.ndarray) -> bytes:
    """G.711 encode (test helper only): pick the code whose decoded
    value is nearest to each sample. Correct by construction
    against the decode table; a wrong table would fail the known
    vector tests, not hide behind a second hand-derived table."""
    flat = np.asarray(samples, dtype=np.float64).ravel()
    idx = np.argmin(np.abs(table.astype(np.float64)[None, :] - flat[:, None]), axis=1)
    return idx.astype(np.uint8).tobytes()


def _mulaw_encode(samples: np.ndarray) -> bytes:
    return _encode(_MULAW_TABLE, samples)


def _alaw_encode(samples: np.ndarray) -> bytes:
    return _encode(_ALAW_TABLE, samples)


def _rtp(
    seq: int,
    ts: int,
    payload: bytes,
    ssrc: int = SSRC,
    pt: int = 0,
    marker: bool = False,
    version: int = 2,
    csrcs: tuple[int, ...] = (),
    extension: bytes = b"",
    pad: int = 0,
) -> bytes:
    """Build one RTP datagram (RFC 3550)."""
    b0 = (version << 6) | (1 if pad else 0) | (0x10 if extension else 0) | len(csrcs)
    b1 = (0x80 if marker else 0x00) | (pt & 0x7F)
    header = struct.pack(">BBHII", b0, b1, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc)
    for csrc in csrcs:
        header += struct.pack(">I", csrc)
    if extension:
        # Extension header: 16-bit profile, 16-bit length in 32-bit
        # words, then the extension body padded to a word boundary.
        body = extension + b"\x00" * ((-len(extension)) % 4)
        header += struct.pack(">HH", 0xBEDE, len(body) // 4) + body
    data = header + payload
    if pad:
        data += b"\x00" * (pad - 1) + bytes([pad])
    return data


def _sine(n, freq=440.0, amp=0.5):
    t = np.arange(n) / 8000.0
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _stream(format="ulaw", **kwargs):
    return AsteriskRtpStream(CALL_ID, format=format, channel_id=CHANNEL_ID, **kwargs)


def _packet(seq, ts, samples, **kwargs):
    fmt = kwargs.pop("format", "ulaw")
    pt = kwargs.pop("pt", 0 if fmt == "ulaw" else 8)
    encode = _mulaw_encode if fmt == "ulaw" else _alaw_encode
    return _rtp(seq, ts, encode(samples), pt=pt, **kwargs)


# -- G.711 A-law decode ------------------------------------------------


def test_alaw_known_vectors():
    # ITU-T G.711: 0xD5 is the smallest positive step, 0x55 the
    # smallest negative step; 0xA5 / 0x25 are a mid-range pair.
    assert alaw_decode(bytes([0xD5]))[0] == pytest.approx(8 / 32768.0)
    assert alaw_decode(bytes([0x55]))[0] == pytest.approx(-8 / 32768.0)
    assert alaw_decode(bytes([0xA5]))[0] == pytest.approx(16896 / 32768.0)
    assert alaw_decode(bytes([0x25]))[0] == pytest.approx(-16896 / 32768.0)
    assert alaw_decode(b"").size == 0


def test_alaw_max_magnitude():
    # Segment 7, mantissa 0xF: the largest A-law magnitude.
    assert alaw_decode(bytes([0x2A]))[0] == pytest.approx(-32256 / 32768.0)
    assert alaw_decode(bytes([0xAA]))[0] == pytest.approx(32256 / 32768.0)
    assert np.all(np.abs(alaw_decode(bytes(range(256)))) <= 1.0)


def test_alaw_round_trip_within_half_step():
    rng = np.random.RandomState(7)
    original = rng.uniform(-0.9, 0.9, 4096).astype(np.float32)
    decoded = alaw_decode(_alaw_encode(original))
    # Worst case is segment 7, where one A-law step is 1024/32768 =
    # 0.03125, so a correct round trip stays within a half step
    # (0.015625) of the original. A wrong table shifts values by a
    # whole segment or more and breaks this bound.
    assert np.max(np.abs(decoded - original)) < 0.02
    assert np.all(np.abs(decoded) <= 1.0)


# -- construction ------------------------------------------------------


def test_rejects_unknown_format():
    with pytest.raises(AsteriskError):
        AsteriskRtpStream(CALL_ID, format="slin16")
    with pytest.raises(AsteriskError):
        AsteriskRtpStream(CALL_ID, format="gsm")


def test_rejects_empty_call_id_and_unknown_track():
    with pytest.raises(AsteriskError):
        AsteriskRtpStream("")
    with pytest.raises(AsteriskError):
        AsteriskRtpStream(CALL_ID, track="sideways")


# -- happy path --------------------------------------------------------


def test_twenty_ms_packets_yield_two_frames_each():
    stream = _stream()
    frames = stream.handle_datagram(_packet(0, 0, _sine(160)))
    assert len(frames) == 2
    assert all(f.samples.size == FRAME_SAMPLES_8K for f in frames)
    assert [f.seq for f in frames] == [0, 1]
    assert frames[0].t_sec == pytest.approx(0.0)
    assert frames[1].t_sec == pytest.approx(0.01)
    assert frames[0].source == "asterisk"
    assert frames[0].track == "inbound"
    assert frames[0].stream_sid == CHANNEL_ID
    assert frames[0].call_id == CALL_ID
    assert frames[0].sample_rate == 8000
    assert not any(f.discontinuous for f in frames)
    frames = stream.handle_datagram(_packet(1, 160, _sine(160)))
    assert len(frames) == 2
    assert [f.seq for f in frames] == [2, 3]
    assert frames[0].t_sec == pytest.approx(0.02)
    assert stream.stats()["frames"] == 4


def test_ten_ms_packets_yield_one_frame_each():
    stream = _stream()
    frames = stream.handle_datagram(_packet(0, 0, _sine(80)))
    assert len(frames) == 1
    frames = stream.handle_datagram(_packet(1, 80, _sine(80)))
    assert len(frames) == 1
    assert frames[0].t_sec == pytest.approx(0.01)


def test_first_packet_arbitrary_timestamp():
    # Asterisk starts timestamps at 0 in practice, but the adapter
    # must not assume it: t_sec is measured from the first packet.
    stream = _stream()
    frames = stream.handle_datagram(_packet(0, 123456, _sine(160)))
    assert frames[0].t_sec == pytest.approx(0.0)
    frames = stream.handle_datagram(_packet(1, 123616, _sine(160)))
    assert frames[0].t_sec == pytest.approx(0.02)


def test_alaw_happy_path_decodes():
    stream = _stream(format="alaw")
    original = _sine(160)
    frames = stream.handle_datagram(_packet(0, 0, original, format="alaw"))
    assert len(frames) == 2
    decoded = np.concatenate([f.samples for f in frames])
    assert np.max(np.abs(decoded - original)) < 0.01


def test_ulaw_content_matches_samples():
    stream = _stream()
    original = _sine(160)
    frames = stream.handle_datagram(_packet(0, 0, original))
    decoded = np.concatenate([f.samples for f in frames])
    assert np.max(np.abs(decoded - original)) < 0.02


def test_partial_payload_buffers_across_packets():
    # A 120-byte payload is 15 ms: one frame now, 40 samples held.
    stream = _stream()
    frames = stream.handle_datagram(_packet(0, 0, _sine(120)))
    assert len(frames) == 1
    assert stream.stats()["samples"] == 120
    frames = stream.handle_datagram(_packet(1, 120, _sine(120)))
    # 40 buffered + 120 new = 160 samples = two frames.
    assert len(frames) == 2
    assert frames[0].t_sec == pytest.approx(0.01)


def test_marker_bit_counted_not_disruptive():
    stream = _stream()
    frames = stream.handle_datagram(_packet(0, 0, _sine(160), marker=True))
    assert len(frames) == 2
    assert stream.stats()["marker_packets"] == 1


# -- RTP header parsing ------------------------------------------------


def test_rejects_wrong_version():
    stream = _stream()
    with pytest.raises(AsteriskError):
        stream.handle_datagram(_packet(0, 0, _sine(160), version=0))


def test_rejects_short_datagram():
    stream = _stream()
    with pytest.raises(AsteriskError):
        stream.handle_datagram(b"\x80\x00\x00")


def test_rejects_empty_payload():
    stream = _stream()
    with pytest.raises(AsteriskError):
        stream.handle_datagram(_rtp(0, 0, b""))


def test_padding_is_stripped():
    stream = _stream()
    frames = stream.handle_datagram(_packet(0, 0, _sine(160), pad=4))
    assert len(frames) == 2
    assert stream.stats()["samples"] == 160


def test_rejects_invalid_padding():
    stream = _stream()
    # Padding byte claims 200 bytes of padding in a short packet.
    data = _rtp(0, 0, _mulaw_encode(_sine(160))) + b"\x00" * 3 + bytes([200])
    data = bytearray(data)
    data[0] |= 0x20  # set the padding bit
    with pytest.raises(AsteriskError):
        stream.handle_datagram(bytes(data))


def test_csrc_and_extension_are_skipped():
    stream = _stream()
    frames = stream.handle_datagram(
        _packet(0, 0, _sine(160), csrcs=(0x11111111, 0x22222222), extension=b"\x01\x02\x03")
    )
    assert len(frames) == 2
    assert stream.stats()["samples"] == 160


def test_rejects_wrong_payload_type():
    stream = _stream(format="ulaw")
    with pytest.raises(AsteriskError):
        stream.handle_datagram(_packet(0, 0, _sine(160), pt=8))
    # GSM (PT 3) is supported by Asterisk but not decoded here.
    with pytest.raises(AsteriskError):
        stream.handle_datagram(_packet(0, 0, _sine(160), pt=3))


def test_rejects_alaw_packet_on_ulaw_stream():
    stream = _stream(format="alaw")
    with pytest.raises(AsteriskError):
        stream.handle_datagram(_packet(0, 0, _sine(160), pt=0))


# -- continuity --------------------------------------------------------


def test_sequence_gap_flags_discontinuous_and_counts_loss():
    stream = _stream()
    stream.handle_datagram(_packet(0, 0, _sine(160)))
    frames = stream.handle_datagram(_packet(2, 320, _sine(160)))
    assert len(frames) == 2
    assert frames[0].discontinuous
    assert not frames[1].discontinuous
    assert frames[0].t_sec == pytest.approx(0.04)
    stats = stream.stats()
    assert stats["gaps"] == 1
    assert stats["lost_samples"] == 160  # one whole 20 ms packet


def test_sequence_gap_discards_partial_buffer():
    stream = _stream()
    stream.handle_datagram(_packet(0, 0, _sine(120)))  # 1 frame + 40 buffered
    frames = stream.handle_datagram(_packet(2, 320, _sine(160)))
    assert frames[0].discontinuous
    assert frames[0].t_sec == pytest.approx(0.04)
    stats = stream.stats()
    # 120 lost in the skipped packet (sized as the last payload,
    # 15 ms) + 40 discarded from the buffer.
    assert stats["lost_samples"] == 160


def test_duplicate_sequence_dropped():
    stream = _stream()
    stream.handle_datagram(_packet(0, 0, _sine(160)))
    frames = stream.handle_datagram(_packet(0, 0, _sine(160)))
    assert frames == []
    assert stream.stats()["duplicates_dropped"] == 1
    assert stream.stats()["frames"] == 2


def test_out_of_order_sequence_dropped():
    stream = _stream()
    stream.handle_datagram(_packet(0, 0, _sine(160)))
    stream.handle_datagram(_packet(1, 160, _sine(160)))
    frames = stream.handle_datagram(_packet(0, 0, _sine(160)))
    assert frames == []
    assert stream.stats()["out_of_order_dropped"] == 1


def test_timestamp_behind_dropped():
    stream = _stream()
    stream.handle_datagram(_packet(0, 0, _sine(160)))
    stream.handle_datagram(_packet(1, 160, _sine(160)))
    # Sequence advanced but the clock went backwards: overlap.
    frames = stream.handle_datagram(_packet(2, 80, _sine(160)))
    assert frames == []
    assert stream.stats()["out_of_order_dropped"] == 1


def test_timestamp_jump_with_continuous_sequence():
    stream = _stream()
    stream.handle_datagram(_packet(0, 0, _sine(160)))
    # Sequence continuous, clock jumped to 480 instead of 160: the
    # span [160, 480) never arrived, so 320 samples are lost.
    frames = stream.handle_datagram(_packet(1, 480, _sine(160)))
    assert frames[0].discontinuous
    assert frames[0].t_sec == pytest.approx(0.06)
    stats = stream.stats()
    assert stats["gaps"] == 1
    assert stats["lost_samples"] == 320


def test_ssrc_change_rejected():
    stream = _stream()
    stream.handle_datagram(_packet(0, 0, _sine(160)))
    with pytest.raises(AsteriskError):
        stream.handle_datagram(_packet(1, 160, _sine(160), ssrc=0x99999999))
    assert stream.stats()["ssrc_rejected"] == 1


def test_sequence_wraparound_is_continuous():
    stream = _stream()
    stream.handle_datagram(_packet(65534, 0, _sine(160)))
    frames = stream.handle_datagram(_packet(65535, 160, _sine(160)))
    assert not frames[0].discontinuous
    frames = stream.handle_datagram(_packet(0, 320, _sine(160)))
    assert not frames[0].discontinuous
    frames = stream.handle_datagram(_packet(1, 480, _sine(160)))
    assert not frames[0].discontinuous
    assert stream.stats()["gaps"] == 0


def test_timestamp_wraparound_is_continuous():
    stream = _stream()
    base = 0xFFFFFF00
    stream.handle_datagram(_packet(0, base, _sine(160)))
    frames = stream.handle_datagram(_packet(1, (base + 160) & 0xFFFFFFFF, _sine(160)))
    assert not frames[0].discontinuous
    assert frames[0].t_sec == pytest.approx(0.02)
    frames = stream.handle_datagram(_packet(2, (base + 320) & 0xFFFFFFFF, _sine(160)))
    assert frames[0].t_sec == pytest.approx(0.04)
    assert stream.stats()["gaps"] == 0


# -- stop --------------------------------------------------------------


def test_stop_discards_partial_frame_and_rejects_later_packets():
    stream = _stream()
    stream.handle_datagram(_packet(0, 0, _sine(120)))  # 40 samples buffered
    stream.stop()
    assert stream.is_stopped
    assert stream.stats()["discarded_on_stop"] == 40
    with pytest.raises(AsteriskError):
        stream.handle_datagram(_packet(1, 120, _sine(160)))
    with pytest.raises(AsteriskError):
        stream.stop()
