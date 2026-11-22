"""Amazon Connect KVS fragment adapter (ADR-013).

Covers shieldcall.ingest.amazon_connect: the EBML/MKV fragment
parser, track mapping by name, the MimeType format pin, the
ContactId binding check, per-track timecode clocks and gap
flagging, and buffering across fragments. Fragments are built
by a small EBML writer in this file to the structure Amazon
publishes (EBML header, Segment, Info, Tracks, Tags, Cluster
with SimpleBlocks); no AWS account or live fragment is needed.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from shieldcall.ingest.amazon_connect import (
    AmazonConnectError,
    AmazonConnectStream,
)

CALL_ID = "d111f871-5fe7-40b1-b5f7-6b67060de0c2"
FRAGMENT_NUMBER = "91343852333181531502161607852955530730504675150"
CLUSTER_TC = 1726677688724


# -- EBML writer ----------------------------------------------------------

def _size_vint(n: int) -> bytes:
    for length in range(1, 9):
        if n < (1 << (7 * length)) - 1:
            return ((1 << (7 * length)) + n).to_bytes(length, "big")
    raise ValueError("size too large")


def _el(element_id: int, payload: bytes, id_len: int = 1, unknown_size: bool = False) -> bytes:
    # An unknown size is an all-ones vint: length 8 is 0x01 followed
    # by seven 0xFF bytes, NOT eight 0xFF bytes (that would parse as
    # a 1-byte unknown size followed by garbage).
    size = b"\x01" + b"\xff" * 7 if unknown_size else _size_vint(len(payload))
    return element_id.to_bytes(id_len, "big") + size + payload


def _uint(element_id: int, value: int, id_len: int = 1) -> bytes:
    payload = value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")
    return _el(element_id, payload, id_len)


def _str(element_id: int, value: str, id_len: int = 1) -> bytes:
    return _el(element_id, value.encode("utf-8"), id_len)


def _float32(element_id: int, value: float, id_len: int = 1) -> bytes:
    return _el(element_id, struct.pack(">f", value), id_len)


def _simple_block(track: int, relative_ms: int, samples, flags: int = 0) -> bytes:
    payload = bytes([0x80 | track]) + struct.pack(">h", relative_ms) + bytes([flags])
    payload += struct.pack(f"<{len(samples)}h", *samples)
    return _el(0xA3, payload)


def _track_entry(number: int, name: str, audio: bool = True) -> bytes:
    body = _uint(0xD7, number)  # TrackNumber
    body += _uint(0x73C5, number, id_len=2)  # TrackUID
    body += _uint(0x83, 2)  # TrackType: audio
    body += _str(0x536E, name, id_len=2)  # Name
    body += _str(0x86, "A_AAC")  # CodecID (observed; never trusted, see ADR-013)
    body += _el(0x63A2, b"\x11\x90", id_len=2)  # CodecPrivate
    if audio:
        audio_body = _float32(0xB5, 8000.0) + _uint(0x9F, 1)  # SamplingFrequency, Channels
        body += _el(0xE1, audio_body)
    return _el(0xAE, body)


def _simple_tag(name: str, value: str) -> bytes:
    body = _str(0x45A3, name, id_len=2) + _str(0x4487, value, id_len=2)
    return _el(0x67C8, body, id_len=2)


def _tags(tags: dict) -> bytes:
    out = b""
    for name, value in tags.items():
        out += _el(0x1254C367, _el(0x7373, _simple_tag(name, value), id_len=2), id_len=4)
    return out


def _fragment(
    blocks,
    *,
    track_numbers=(1, 2),
    mime="audio/L16;rate=8000;channels=1;",
    contact_id=CALL_ID,
    cluster_timecode=CLUSTER_TC,
    fragment_number=FRAGMENT_NUMBER,
    include_tracks=True,
    include_info=True,
    timecode_scale=1000000,
    segment_unknown_size=False,
    cluster_unknown_size=False,
    extra_segment=b"",
    track_names=("AUDIO_FROM_CUSTOMER", "AUDIO_TO_CUSTOMER"),
    cluster_timecode_present=True,
    doc_type="matroska",
):
    ebml_body = (
        _uint(0x4286, 1, id_len=2)
        + _str(0x4282, doc_type, id_len=2)
    )
    ebml = _el(0x1A45DFA3, ebml_body, id_len=4)
    segment_body = b""
    if include_info:
        segment_body += _el(
            0x1549A966, _uint(0x2AD7B1, timecode_scale, id_len=3), id_len=4
        )
    if include_tracks:
        tracks_body = b"".join(
            _track_entry(number, name)
            for number, name in zip(track_numbers, track_names)
        )
        segment_body += _el(0x1654AE6B, tracks_body, id_len=4)
    tags = {}
    if mime is not None:
        tags["MimeType"] = mime
    if contact_id is not None:
        tags["ContactId"] = contact_id
    tags["InstanceId"] = "d5b716a2-10bb-45cb-bf7f-01a9864be591"
    if fragment_number is not None:
        tags["AWS_KINESISVIDEO_FRAGMENT_NUMBER"] = fragment_number
    segment_body += _tags(tags)
    cluster_body = b""
    if cluster_timecode_present:
        cluster_body += _uint(0xE7, cluster_timecode)
    for track, relative_ms, samples in blocks:
        cluster_body += _simple_block(track, relative_ms, samples)
    segment_body += _el(
        0x1F43B675, cluster_body, id_len=4, unknown_size=cluster_unknown_size
    )
    segment_body += extra_segment
    segment = _el(
        0x18538067, segment_body, id_len=4, unknown_size=segment_unknown_size
    )
    return ebml + segment


def _silence(n: int):
    return [0] * n


def _ramp(n: int):
    return [((i * 257) % 32768) - 16384 for i in range(n)]


def _stream():
    return AmazonConnectStream(call_id=CALL_ID)


# -- happy path ------------------------------------------------------------

def test_l16_known_vectors_round_trip():
    stream = _stream()
    samples = [-32768, -16384, 0, 16384, 32767] * 16
    frames = stream.feed_fragment(_fragment([(1, 0, samples)]))
    assert len(frames) == 1
    frame = frames[0]
    assert frame.track == "inbound"
    assert frame.seq == 0
    assert frame.t_sec == 0.0
    assert frame.sample_rate == 8000
    assert frame.source == "amazon_connect"
    assert frame.stream_sid == FRAGMENT_NUMBER
    assert frame.discontinuous is False
    expected = np.array(samples, dtype=np.float32) / 32768.0
    np.testing.assert_allclose(frame.samples, expected, atol=1e-6)


def test_track_mapping_comes_from_names_not_numbers():
    # FROM is track 2 and TO is track 1 here: numbers are not assumed.
    stream = _stream()
    frames = stream.feed_fragment(
        _fragment(
            [(2, 0, _silence(80)), (1, 0, _silence(80))],
            track_numbers=(2, 1),
        )
    )
    assert [f.track for f in frames] == ["inbound", "outbound"]
    assert frames[0].seq == 0 and frames[1].seq == 0
    assert stream.stats()["inbound_frames"] == 1
    assert stream.stats()["outbound_frames"] == 1


def test_customer_only_fragment_has_no_outbound_track():
    stream = _stream()
    frames = stream.feed_fragment(
        _fragment(
            [(1, 0, _silence(80))],
            track_numbers=(1,),
            track_names=("AUDIO_FROM_CUSTOMER",),
        )
    )
    assert len(frames) == 1 and frames[0].track == "inbound"


def test_t_sec_comes_from_cluster_and_block_timecodes():
    stream = _stream()
    frames = stream.feed_fragment(
        _fragment([(1, 0, _silence(160)), (1, 100, _silence(160))])
    )
    assert [f.t_sec for f in frames] == [0.0, 0.01, 0.1, 0.11]
    assert [f.seq for f in frames] == [0, 1, 2, 3]


def test_gap_flags_discontinuous_and_counts_lost_samples():
    stream = _stream()
    frames = stream.feed_fragment(
        _fragment([(1, 0, _silence(160)), (1, 60, _silence(80))])
    )
    assert frames[2].discontinuous is True
    assert frames[0].discontinuous is False
    assert stream.stats()["lost_samples"] == 320  # 40 ms at 8 kHz


def test_gap_flag_survives_a_block_that_emits_no_frame():
    stream = _stream()
    frames = stream.feed_fragment(
        _fragment(
            [
                (1, 0, _silence(160)),
                (1, 60, _silence(40)),
                (1, 65, _silence(40)),
            ]
        )
    )
    assert len(frames) == 3
    assert frames[2].discontinuous is True
    assert frames[2].t_sec == pytest.approx(0.06)


def test_overlap_or_out_of_order_block_is_refused():
    stream = _stream()
    with pytest.raises(AmazonConnectError, match="before the expected"):
        stream.feed_fragment(
            _fragment([(1, 0, _silence(160)), (1, 10, _silence(80))])
        )


def test_partial_frame_buffers_across_fragments():
    stream = _stream()
    first = stream.feed_fragment(_fragment([(1, 0, _silence(120))]))
    assert len(first) == 1
    second = stream.feed_fragment(_fragment([(1, 15, _silence(40))]))
    assert len(second) == 1
    assert second[0].seq == 1
    assert second[0].t_sec == pytest.approx(0.01)


def test_finish_discards_partial_frame_and_is_idempotent():
    stream = _stream()
    assert stream.feed_fragment(_fragment([(1, 0, _silence(40))])) == []
    stream.finish()
    stream.finish()
    assert stream.stats()["discarded_on_close"] == 40
    assert stream.is_finished


def test_messages_after_finish_raise():
    stream = _stream()
    stream.finish()
    with pytest.raises(AmazonConnectError, match="after the stream finished"):
        stream.feed_fragment(_fragment([(1, 0, _silence(80))]))


def test_empty_call_id_is_refused():
    with pytest.raises(AmazonConnectError, match="call_id is required"):
        AmazonConnectStream(call_id="")


# -- format pinning ---------------------------------------------------------

def test_wrong_mime_rate_is_refused():
    stream = _stream()
    with pytest.raises(AmazonConnectError, match="MimeType"):
        stream.feed_fragment(
            _fragment([(1, 0, _silence(80))], mime="audio/L16;rate=16000;channels=1;")
        )


def test_wrong_mime_channels_is_refused():
    stream = _stream()
    with pytest.raises(AmazonConnectError, match="MimeType"):
        stream.feed_fragment(
            _fragment([(1, 0, _silence(80))], mime="audio/L16;rate=8000;channels=2;")
        )


def test_missing_mime_on_first_fragment_is_refused():
    stream = _stream()
    with pytest.raises(AmazonConnectError, match="no MimeType"):
        stream.feed_fragment(_fragment([(1, 0, _silence(80))], mime=None))


def test_mime_change_mid_stream_is_refused():
    stream = _stream()
    stream.feed_fragment(_fragment([(1, 0, _silence(80))]))
    with pytest.raises(AmazonConnectError, match="MimeType changed"):
        stream.feed_fragment(
            _fragment(
                [(1, 80, _silence(80))],
                mime="audio/L16;rate=8000;channels=1",
            )
        )


def test_later_fragment_without_mime_reuses_the_first():
    stream = _stream()
    stream.feed_fragment(_fragment([(1, 0, _silence(80))]))
    frames = stream.feed_fragment(_fragment([(1, 10, _silence(80))], mime=None))
    assert len(frames) == 1


def test_audio_element_rate_mismatch_is_refused():
    stream = _stream()
    entry = _el(
        0xAE,
        _uint(0xD7, 1)
        + _uint(0x73C5, 1, id_len=2)
        + _uint(0x83, 2)
        + _str(0x536E, "AUDIO_FROM_CUSTOMER", id_len=2)
        + _str(0x86, "A_AAC")
        + _el(
            0xE1,
            _float32(0xB5, 16000.0) + _uint(0x9F, 1),
        ),
    )
    tracks = _el(0x1654AE6B, entry, id_len=4)
    ebml = _el(0x1A45DFA3, _str(0x4282, "matroska", id_len=2), id_len=4)
    segment_body = tracks + _tags({"MimeType": "audio/L16;rate=8000;channels=1;"})
    segment_body += _el(
        0x1F43B675,
        _uint(0xE7, CLUSTER_TC) + _simple_block(1, 0, _silence(80)),
        id_len=4,
    )
    with pytest.raises(AmazonConnectError, match="sample rate"):
        stream.feed_fragment(ebml + _el(0x18538067, segment_body, id_len=4))


# -- binding and tracks -----------------------------------------------------

def test_contact_id_mismatch_is_refused():
    stream = _stream()
    with pytest.raises(AmazonConnectError, match="ContactId"):
        stream.feed_fragment(
            _fragment([(1, 0, _silence(80))], contact_id="someone-else")
        )


def test_missing_contact_id_still_runs():
    stream = _stream()
    frames = stream.feed_fragment(
        _fragment([(1, 0, _silence(80))], contact_id=None)
    )
    assert len(frames) == 1
    assert stream.contact_id is None


def test_unknown_track_name_is_refused():
    stream = _stream()
    with pytest.raises(AmazonConnectError, match="unknown KVS track name"):
        stream.feed_fragment(
            _fragment(
                [(1, 0, _silence(80))],
                track_numbers=(1,),
                track_names=("AUDIO_SOMEWHERE_ELSE",),
            )
        )


def test_missing_customer_track_is_refused():
    stream = _stream()
    with pytest.raises(AmazonConnectError, match="AUDIO_FROM_CUSTOMER"):
        stream.feed_fragment(
            _fragment(
                [(1, 0, _silence(80))],
                track_numbers=(1,),
                track_names=("AUDIO_TO_CUSTOMER",),
            )
        )


def test_block_for_undeclared_track_is_refused():
    stream = _stream()
    with pytest.raises(AmazonConnectError, match="undeclared track number"):
        stream.feed_fragment(_fragment([(3, 0, _silence(80))]))


def test_track_map_change_mid_stream_is_refused():
    stream = _stream()
    stream.feed_fragment(_fragment([(1, 0, _silence(80))]))
    with pytest.raises(AmazonConnectError, match="track map changed"):
        stream.feed_fragment(
            _fragment([(2, 10, _silence(80))], track_numbers=(2, 1))
        )


# -- block-level violations --------------------------------------------------

def test_laced_block_is_refused():
    stream = _stream()
    block = _simple_block(1, 0, _silence(80), flags=0x02)
    cluster = _el(0x1F43B675, _uint(0xE7, CLUSTER_TC) + block, id_len=4)
    ebml = _el(0x1A45DFA3, _str(0x4282, "matroska", id_len=2), id_len=4)
    tracks = _el(
        0x1654AE6B, _track_entry(1, "AUDIO_FROM_CUSTOMER"), id_len=4
    )
    segment_body = tracks + _tags({"MimeType": "audio/L16;rate=8000;channels=1;"})
    segment_body += cluster
    with pytest.raises(AmazonConnectError, match="laced"):
        stream.feed_fragment(ebml + _el(0x18538067, segment_body, id_len=4))


def test_odd_byte_payload_is_refused():
    stream = _stream()
    payload = bytes([0x81]) + struct.pack(">h", 0) + bytes([0]) + b"\x00\x00\x00"
    cluster = _el(0x1F43B675, _uint(0xE7, CLUSTER_TC) + _el(0xA3, payload), id_len=4)
    ebml = _el(0x1A45DFA3, _str(0x4282, "matroska", id_len=2), id_len=4)
    tracks = _el(
        0x1654AE6B, _track_entry(1, "AUDIO_FROM_CUSTOMER"), id_len=4
    )
    segment_body = tracks + _tags({"MimeType": "audio/L16;rate=8000;channels=1;"})
    segment_body += cluster
    with pytest.raises(AmazonConnectError, match="odd byte count"):
        stream.feed_fragment(ebml + _el(0x18538067, segment_body, id_len=4))


def test_block_group_is_refused():
    stream = _stream()
    group = _el(0xA0, _el(0xA1, b"\x81\x00\x00\x00\x00\x00"), id_len=1)
    cluster = _el(0x1F43B675, _uint(0xE7, CLUSTER_TC) + group, id_len=4)
    ebml = _el(0x1A45DFA3, _str(0x4282, "matroska", id_len=2), id_len=4)
    tracks = _el(
        0x1654AE6B, _track_entry(1, "AUDIO_FROM_CUSTOMER"), id_len=4
    )
    segment_body = tracks + _tags({"MimeType": "audio/L16;rate=8000;channels=1;"})
    segment_body += cluster
    with pytest.raises(AmazonConnectError, match="BlockGroup"):
        stream.feed_fragment(ebml + _el(0x18538067, segment_body, id_len=4))


# -- EBML structure ----------------------------------------------------------

def test_truncated_fragment_is_refused():
    stream = _stream()
    good = _fragment([(1, 0, _silence(80))])
    with pytest.raises(AmazonConnectError, match="truncated"):
        stream.feed_fragment(good[:-5])


def test_trailing_bytes_after_segment_are_refused():
    stream = _stream()
    with pytest.raises(AmazonConnectError, match="exactly an EBML header"):
        stream.feed_fragment(
            _fragment([(1, 0, _silence(80))]) + _el(0xEC, b"\x00")
        )


def test_empty_fragment_is_refused():
    with pytest.raises(AmazonConnectError, match="empty fragment"):
        _stream().feed_fragment(b"")


def test_non_ebml_first_element_is_refused():
    with pytest.raises(AmazonConnectError):
        _stream().feed_fragment(b"\x00" * 10)


def test_wrong_doctype_is_refused():
    stream = _stream()
    with pytest.raises(AmazonConnectError, match="DocType"):
        stream.feed_fragment(
            _fragment([(1, 0, _silence(80))], doc_type="webm")
        )


def test_void_element_is_skipped():
    stream = _stream()
    void = _el(0xEC, b"\x00" * 6)
    frames = stream.feed_fragment(
        _fragment([(1, 0, _silence(80))], extra_segment=void)
    )
    assert len(frames) == 1


def test_unknown_size_segment_and_cluster_still_parse():
    stream = _stream()
    frames = stream.feed_fragment(
        _fragment(
            [(1, 0, _silence(80))],
            segment_unknown_size=True,
            cluster_unknown_size=True,
        )
    )
    assert len(frames) == 1
    # Tags after the unknown-size Cluster must still be read:
    # the ContactId binding and fragment number prove it.
    assert stream.contact_id == CALL_ID
    assert stream.fragment_number == FRAGMENT_NUMBER


def test_fragment_without_info_uses_default_timecode_scale():
    stream = _stream()
    frames = stream.feed_fragment(
        _fragment([(1, 0, _silence(80))], include_info=False)
    )
    assert len(frames) == 1


def test_cluster_without_timecode_is_refused():
    stream = _stream()
    with pytest.raises(AmazonConnectError, match="no Timecode"):
        stream.feed_fragment(
            _fragment([(1, 0, _silence(80))], cluster_timecode_present=False)
        )


def test_stats_accumulate_across_fragments():
    stream = _stream()
    stream.feed_fragment(_fragment([(1, 0, _silence(160)), (2, 0, _silence(80))]))
    stream.feed_fragment(_fragment([(1, 20, _silence(80))]))
    stats = stream.stats()
    assert stats["fragments"] == 2
    assert stats["blocks"] == 3
    assert stats["frames"] == 4
    assert stats["inbound_frames"] == 3
    assert stats["outbound_frames"] == 1
