"""FreeSWITCH mod_audio_fork adapter (ADR-014).

Protocol shapes follow the module's own README
(andrewvmail/freeswitch-modules, mod_audio_fork): the first
WebSocket frame is the operator's metadata text frame, then raw
L16 binary audio, with mid-call text frames of arbitrary shape
and no stop event. The pure translation is covered here; the
route is covered by tests/test_freeswitch_route.py.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from shieldcall.ingest.frame import FRAME_SAMPLES_8K
from shieldcall.ingest.freeswitch import FreeSwitchError, FreeSwitchMediaStream

CALL_ID = "aaaaaaaa-bbbb-cccc-dddd-0123456789ab"


def _l16(values) -> bytes:
    """Encode int16 sample values as little-endian L16 bytes."""
    return struct.pack(f"<{len(values)}h", *values)


def _metadata(**extra) -> str:
    msg = {"sample_rate": 8000, "mix_type": "mono"}
    msg.update(extra)
    return json.dumps(msg)


def _started_stream(**extra) -> FreeSwitchMediaStream:
    stream = FreeSwitchMediaStream(CALL_ID)
    stream.handle_text(_metadata(**extra))
    return stream


# -- lifecycle and metadata contract -------------------------------------


def test_call_id_required():
    with pytest.raises(FreeSwitchError, match="call_id is required"):
        FreeSwitchMediaStream("")


def test_metadata_pins_format_and_records_extras():
    stream = FreeSwitchMediaStream(CALL_ID)
    frames = stream.handle_text(
        _metadata(stream_id="stream-1", VID="Unknown", to="6173333456")
    )
    assert frames == []
    assert stream.is_started
    assert stream.sample_rate == 8000
    assert stream.mix_type == "mono"
    assert stream.stream_sid == "stream-1"
    assert stream.metadata["VID"] == "Unknown"
    assert stream.metadata["to"] == "6173333456"


def test_metadata_call_id_must_match_route_binding():
    stream = FreeSwitchMediaStream(CALL_ID)
    stream.handle_text(_metadata(call_id=CALL_ID))
    assert stream.is_started


def test_metadata_call_id_mismatch_refused():
    stream = FreeSwitchMediaStream(CALL_ID)
    with pytest.raises(FreeSwitchError, match="does not match"):
        stream.handle_text(_metadata(call_id="some-other-call"))


def test_metadata_call_id_must_be_string():
    stream = FreeSwitchMediaStream(CALL_ID)
    with pytest.raises(FreeSwitchError, match="call_id must be"):
        stream.handle_text(_metadata(call_id=42))


def test_metadata_stream_id_must_be_string():
    stream = FreeSwitchMediaStream(CALL_ID)
    with pytest.raises(FreeSwitchError, match="stream_id must be"):
        stream.handle_text(_metadata(stream_id=7))


def test_metadata_must_be_json():
    stream = FreeSwitchMediaStream(CALL_ID)
    with pytest.raises(FreeSwitchError, match="not JSON"):
        stream.handle_text("sample_rate=8000, mix_type=mono")


def test_metadata_must_be_json_object():
    for text in ('["mono"]', '"mono"', "8000", "null"):
        stream = FreeSwitchMediaStream(CALL_ID)
        with pytest.raises(FreeSwitchError, match="JSON object"):
            stream.handle_text(text)


def test_metadata_empty_frame_refused():
    for text in ("", "   "):
        stream = FreeSwitchMediaStream(CALL_ID)
        with pytest.raises(FreeSwitchError, match="empty"):
            stream.handle_text(text)


def test_metadata_missing_sample_rate_refused():
    stream = FreeSwitchMediaStream(CALL_ID)
    with pytest.raises(FreeSwitchError, match="no sample_rate"):
        stream.handle_text(json.dumps({"mix_type": "mono"}))


def test_metadata_missing_mix_type_refused():
    stream = FreeSwitchMediaStream(CALL_ID)
    with pytest.raises(FreeSwitchError, match="no mix_type"):
        stream.handle_text(json.dumps({"sample_rate": 8000}))


@pytest.mark.parametrize("rate", [8000, "8000", "8k", "8K", " 8k ", 8000.0])
def test_sample_rate_spellings_accepted(rate):
    stream = FreeSwitchMediaStream(CALL_ID)
    stream.handle_text(_metadata(sample_rate=rate))
    assert stream.sample_rate == 8000


@pytest.mark.parametrize("rate", [16000, "16k", 24000, "abc", "8.5k", True, 8000.5])
def test_sample_rate_rejected(rate):
    stream = FreeSwitchMediaStream(CALL_ID)
    with pytest.raises(FreeSwitchError):
        stream.handle_text(_metadata(sample_rate=rate))


def test_sixteen_k_refused_not_resampled():
    stream = FreeSwitchMediaStream(CALL_ID)
    with pytest.raises(FreeSwitchError, match="unsupported.*16000"):
        stream.handle_text(_metadata(sample_rate=16000))


@pytest.mark.parametrize("mix", ["mono", "Mono", " MONO ", "stereo"])
def test_mix_type_spellings_accepted(mix):
    stream = FreeSwitchMediaStream(CALL_ID)
    stream.handle_text(_metadata(mix_type=mix))
    assert stream.mix_type == mix.strip().lower()


def test_mixed_fork_refused():
    stream = FreeSwitchMediaStream(CALL_ID)
    with pytest.raises(FreeSwitchError, match="mixed forks are refused"):
        stream.handle_text(_metadata(mix_type="mixed"))


def test_unknown_mix_type_refused():
    stream = FreeSwitchMediaStream(CALL_ID)
    with pytest.raises(FreeSwitchError, match="unknown mix_type"):
        stream.handle_text(_metadata(mix_type="quad"))


def test_mix_type_must_be_string():
    stream = FreeSwitchMediaStream(CALL_ID)
    with pytest.raises(FreeSwitchError, match="mix_type.*not a string"):
        stream.handle_text(_metadata(mix_type=1))


def test_audio_before_metadata_refused():
    stream = FreeSwitchMediaStream(CALL_ID)
    with pytest.raises(FreeSwitchError, match="before the metadata"):
        stream.handle_audio(_l16([0] * 160))


# -- L16 decoding and framing --------------------------------------------


def test_l16_known_vectors():
    stream = _started_stream()
    frames = stream.handle_audio(_l16([0, 32767, -32768, 16384] * 20))
    assert len(frames) == 1
    assert frames[0].samples[:4].tolist() == pytest.approx(
        [0.0, 32767 / 32768, -1.0, 0.5]
    )
    assert frames[0].track == "inbound"
    assert frames[0].call_id == CALL_ID
    assert frames[0].sample_rate == 8000
    assert frames[0].source == "freeswitch"
    assert frames[0].discontinuous is False


def test_twenty_ms_packets_emit_two_frames_each():
    stream = _started_stream()
    frames = stream.handle_audio(_l16([0] * 160))
    assert len(frames) == 2
    assert [f.seq for f in frames] == [0, 1]
    assert frames[0].t_sec == pytest.approx(0.0)
    assert frames[1].t_sec == pytest.approx(0.01)


def test_frames_buffer_across_packet_boundaries():
    stream = _started_stream()
    # 100 samples: one full frame, 20 left over.
    assert len(stream.handle_audio(_l16([0] * 100))) == 1
    # 60 more: the leftover 20 plus 60 make the second frame.
    frames = stream.handle_audio(_l16([0] * 60))
    assert len(frames) == 1
    assert frames[0].seq == 1


def test_empty_audio_packet_refused():
    stream = _started_stream()
    with pytest.raises(FreeSwitchError, match="empty audio packet"):
        stream.handle_audio(b"")


def test_odd_byte_packet_refused():
    stream = _started_stream()
    with pytest.raises(FreeSwitchError, match="odd byte count"):
        stream.handle_audio(b"\x00\x00\x00")


def test_partial_trailing_frame_discarded_on_finish():
    stream = _started_stream()
    stream.handle_audio(_l16([0] * 100))
    stream.finish()
    assert stream.stats()["discarded_on_close"] == 20
    assert stream.is_finished


def test_finish_is_idempotent():
    stream = _started_stream()
    stream.finish()
    stream.finish()
    assert stream.is_finished


def test_messages_after_finish_rejected():
    stream = _started_stream()
    stream.finish()
    with pytest.raises(FreeSwitchError, match="after the stream finished"):
        stream.handle_audio(_l16([0] * 160))
    with pytest.raises(FreeSwitchError, match="after the stream finished"):
        stream.handle_text('{"event": "dtmf", "digit": "5"}')


# -- stereo ----------------------------------------------------------------


def test_stereo_deinterleaves_and_counts_other_leg():
    stream = _started_stream(mix_type="stereo")
    # 20 ms stereo: 160 interleaved pairs. Channel 0 ramps, channel 1
    # is a constant the frames must never contain.
    ch0 = list(range(160))
    ch1 = [20000] * 160
    interleaved = [v for pair in zip(ch0, ch1) for v in pair]
    frames = stream.handle_audio(_l16(interleaved))
    assert len(frames) == 2
    got = np.concatenate([f.samples for f in frames])
    expected = np.array(ch0, dtype=np.float32) / 32768.0
    assert got.tolist() == pytest.approx(expected.tolist())
    assert all(f.track == "inbound" for f in frames)
    assert stream.stats()["samples_other_track"] == 160
    assert stream.stats()["samples"] == 160


def test_stereo_truncated_pair_refused():
    stream = _started_stream(mix_type="stereo")
    # 6 bytes: one full pair plus a truncated second pair.
    with pytest.raises(FreeSwitchError, match="truncated interleaved pair"):
        stream.handle_audio(b"\x00\x00\x00\x00\x00\x00")


def test_stereo_negative_full_scale_round_trip():
    stream = _started_stream(mix_type="stereo")
    interleaved = [-32768, 32767] * 80
    frames = stream.handle_audio(_l16(interleaved))
    assert len(frames) == 1
    assert frames[0].samples.tolist() == pytest.approx([-1.0] * 80)


# -- mid-call text frames ---------------------------------------------------


def test_later_text_frames_recorded_verbatim_without_repinning():
    stream = _started_stream()
    # A DTMF notification, then a metadata-shaped frame declaring a
    # different format: neither may change the pinned format.
    assert stream.handle_text('{"event": "dtmf", "digit": "5"}') == []
    assert stream.handle_text(_metadata(sample_rate=16000, mix_type="stereo")) == []
    assert stream.sample_rate == 8000
    assert stream.mix_type == "mono"
    assert stream.text_events == [
        '{"event": "dtmf", "digit": "5"}',
        _metadata(sample_rate=16000, mix_type="stereo"),
    ]
    assert stream.stats()["text_events"] == 2
    frames = stream.handle_audio(_l16([0] * 160))
    assert len(frames) == 2


def test_stop_frame_is_just_a_text_event():
    # uuid_audio_fork stop sends a final text frame before the
    # socket closes; it carries no protocol meaning here.
    stream = _started_stream()
    stream.handle_audio(_l16([0] * 160))
    assert stream.handle_text('{"eof": 1}') == []
    stream.finish()
    assert stream.text_events == ['{"eof": 1}']


# -- stats ------------------------------------------------------------------


def test_stats_count_messages_packets_and_frames():
    stream = _started_stream(stream_id="s-1")
    stream.handle_audio(_l16([0] * 160))
    stream.handle_text("operator note")
    stream.finish()
    stats = stream.stats()
    assert stats["messages"] == 2
    assert stats["audio_packets"] == 1
    assert stats["audio_bytes"] == 320
    assert stats["samples"] == 160
    assert stats["frames"] == 2
    assert stats["discarded_on_close"] == 0


def test_stream_sid_reaches_frames():
    stream = _started_stream(stream_id="s-9")
    frames = stream.handle_audio(_l16([0] * 160))
    assert all(f.stream_sid == "s-9" for f in frames)
