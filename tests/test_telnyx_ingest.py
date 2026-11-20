"""Telnyx Media Streaming adapter (ADR-011).

Protocol vectors come from Telnyx's own docs ("Media Streaming over
Websockets", read 2026-11-20): connected/start/media/stop/dtmf/mark/
error shapes, snake_case fields, PCMU (G.711 mu-law) 8 kHz as the
declared default, PCMA as the other supported codec, chunk as ONE
counter shared by both tracks, and timestamp as an RTP timestamp in
codec clock units (one unit is one sample at 8 kHz), not
milliseconds. Payloads are 20 ms (160 byte) RTP payloads in the
tests below, matching the usual telephony packetisation; the adapter
itself buffers arbitrary lengths.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from shieldcall.ingest.frame import FRAME_SAMPLES_8K, MediaFrame
from shieldcall.ingest.g711 import _ALAW_TABLE, _MULAW_TABLE
from shieldcall.ingest.telnyx import TelnyxError, TelnyxMediaStream
from shieldcall.ingest.twilio import mulaw_decode

CALL_ID = "v2:T02llQxIyaRkhfRKxgAP8nY511EhFLizdvdUKJiSw8d6A9BborherQ"
STREAM_ID = "32DE0DEA-53CB-4B21-89A4-9E1819C043BC"


def _encode(samples: np.ndarray, table: np.ndarray) -> bytes:
    """G.711 encode (test helper only): pick the code whose decoded
    value is nearest to each sample. Correct by construction against
    the decode table; a wrong table would fail the known-vector test
    above, not hide behind a second hand-derived table."""
    flat = np.asarray(samples, dtype=np.float64).ravel()
    idx = np.argmin(np.abs(table.astype(np.float64)[None, :] - flat[:, None]), axis=1)
    return idx.astype(np.uint8).tobytes()


def _payload(samples: np.ndarray, encoding: str = "PCMU") -> str:
    table = _MULAW_TABLE if encoding == "PCMU" else _ALAW_TABLE
    return base64.b64encode(_encode(samples, table)).decode("ascii")


def _connected():
    return {"event": "connected", "version": "1.0.0"}


def _start(call_id=CALL_ID, encoding="PCMU", sample_rate=8000, channels=1):
    return {
        "event": "start",
        "sequence_number": "1",
        "start": {
            "user_id": "3E6F995F-85F7-4705-9741-53B116D28237",
            "call_control_id": call_id,
            "call_session_id": "ff55a038-6f5d-11ef-9692-02420aeffb1f",
            "from": "+13122010094",
            "to": "+13122123456",
            "tags": ["TAG1", "TAG2"],
            "client_state": "aGF2ZSBhIG5pY2UgZGF5ID1d",
            "media_format": {
                "encoding": encoding,
                "sample_rate": sample_rate,
                "channels": channels,
            },
        },
        "stream_id": STREAM_ID,
    }


def _media(chunk, timestamp_rtp, samples, track="inbound", encoding="PCMU"):
    return {
        "event": "media",
        "sequence_number": str(chunk + 3),
        "media": {
            "track": track,
            "chunk": str(chunk),
            "timestamp": str(timestamp_rtp),
            "payload": _payload(samples, encoding),
        },
        "stream_id": STREAM_ID,
    }


def _started_stream(**kwargs):
    stream = TelnyxMediaStream(CALL_ID, **kwargs)
    stream.handle_message(_connected())
    stream.handle_message(_start())
    return stream


def _sine(n, freq=440.0, amp=0.5):
    t = np.arange(n) / 8000.0
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# -- G.711 decode ------------------------------------------------------


def test_mulaw_known_vectors():
    # ITU-T G.711: 0xFF is silence, 0x00 / 0x80 are the extremes.
    assert mulaw_decode(bytes([0xFF]))[0] == pytest.approx(0.0)
    assert mulaw_decode(bytes([0x00]))[0] == pytest.approx(-32124 / 32768.0)
    assert mulaw_decode(bytes([0x80]))[0] == pytest.approx(32124 / 32768.0)
    assert mulaw_decode(b"").size == 0


def test_mulaw_round_trip_within_half_step():
    rng = np.random.RandomState(7)
    original = (rng.uniform(-0.9, 0.9, 4096)).astype(np.float32)
    decoded = mulaw_decode(_encode(original, _MULAW_TABLE))
    # Worst case is the top segment, where one mu-law step is
    # 1024/32768 = 0.03125, so a correct round trip stays within a
    # half step (0.0156) of the original.
    assert np.max(np.abs(decoded - original)) < 0.02
    assert np.all(np.abs(decoded) <= 1.0)


def test_alaw_known_vectors():
    from shieldcall.ingest.g711 import alaw_decode

    assert alaw_decode(bytes([0xD5]))[0] == pytest.approx(8 / 32768.0)
    assert alaw_decode(bytes([0x55]))[0] == pytest.approx(-8 / 32768.0)
    assert alaw_decode(bytes([0xA5]))[0] == pytest.approx(16896 / 32768.0)
    assert alaw_decode(bytes([0x25]))[0] == pytest.approx(-16896 / 32768.0)
    assert alaw_decode(b"").size == 0


def test_alaw_round_trip_within_half_step():
    from shieldcall.ingest.g711 import alaw_decode

    rng = np.random.RandomState(11)
    original = (rng.uniform(-0.9, 0.9, 4096)).astype(np.float32)
    decoded = alaw_decode(_encode(original, _ALAW_TABLE))
    assert np.max(np.abs(decoded - original)) < 0.02
    assert np.all(np.abs(decoded) <= 1.0)


# -- lifecycle ---------------------------------------------------------


def test_connected_then_start_happy_path():
    stream = TelnyxMediaStream(CALL_ID)
    assert stream.handle_message(_connected()) == []
    assert stream.handle_message(_start()) == []
    assert stream.is_started
    assert stream.stream_id == STREAM_ID
    assert stream.call_control_id == CALL_ID
    assert stream.call_session_id == "ff55a038-6f5d-11ef-9692-02420aeffb1f"
    assert stream.from_number == "+13122010094"
    assert stream.to_number == "+13122123456"
    assert stream.tags == ["TAG1", "TAG2"]
    assert stream.client_state == "aGF2ZSBhIG5pY2UgZGF5ID1d"
    assert stream.encoding == "PCMU"


def test_start_declares_no_tracks_list():
    # Telnyx declares no track list in start (unlike Twilio): the
    # operator selects the leg with stream_track at stream start.
    # The adapter must accept a start with no tracks field at all.
    stream = _started_stream()
    assert stream.is_started


def test_media_before_start_rejected():
    stream = TelnyxMediaStream(CALL_ID)
    stream.handle_message(_connected())
    with pytest.raises(TelnyxError):
        stream.handle_message(_media(1, 0, _sine(160)))


def test_stop_before_start_rejected():
    stream = TelnyxMediaStream(CALL_ID)
    with pytest.raises(TelnyxError):
        stream.handle_message(
            {"event": "stop", "stop": {"call_control_id": CALL_ID},
             "stream_id": STREAM_ID}
        )


def test_unknown_event_rejected():
    stream = _started_stream()
    with pytest.raises(TelnyxError):
        stream.handle_message({"event": "clear", "stream_id": STREAM_ID})


def test_non_object_message_rejected():
    stream = _started_stream()
    with pytest.raises(TelnyxError):
        stream.handle_message(["media"])  # type: ignore[arg-type]


def test_start_rejects_call_control_id_mismatch():
    stream = TelnyxMediaStream(CALL_ID)
    stream.handle_message(_connected())
    with pytest.raises(TelnyxError):
        stream.handle_message(_start(call_id="v2:other-call-control-id"))


def test_start_rejects_unsupported_format():
    for kwargs in (
        {"encoding": "G722"},
        {"encoding": "OPUS"},
        {"encoding": "L16"},
        {"encoding": "audio/x-mulaw"},
        {"sample_rate": 16000},
        {"channels": 2},
    ):
        stream = TelnyxMediaStream(CALL_ID)
        stream.handle_message(_connected())
        with pytest.raises(TelnyxError):
            stream.handle_message(_start(**kwargs))


def test_start_accepts_pcma():
    stream = TelnyxMediaStream(CALL_ID)
    stream.handle_message(_connected())
    stream.handle_message(_start(encoding="PCMA"))
    assert stream.encoding == "PCMA"


def test_start_rejects_missing_media_format_and_second_start():
    stream = TelnyxMediaStream(CALL_ID)
    stream.handle_message(_connected())
    bad = _start()
    del bad["start"]["media_format"]
    with pytest.raises(TelnyxError):
        stream.handle_message(bad)
    stream.handle_message(_start())
    with pytest.raises(TelnyxError):
        stream.handle_message(_start())


def test_media_wrong_stream_id_rejected():
    stream = _started_stream()
    msg = _media(1, 0, _sine(160))
    msg["stream_id"] = "other-stream-id"
    with pytest.raises(TelnyxError):
        stream.handle_message(msg)


def test_media_bad_base64_and_empty_payload_rejected():
    stream = _started_stream()
    msg = _media(1, 0, _sine(160))
    msg["media"]["payload"] = "not!!base64"
    with pytest.raises(TelnyxError):
        stream.handle_message(msg)
    msg2 = _media(1, 0, _sine(160))
    msg2["media"]["payload"] = ""
    with pytest.raises(TelnyxError):
        stream.handle_message(msg2)


def test_sequence_number_recorded_not_enforced():
    stream = _started_stream()
    stream.handle_message(_media(1, 0, _sine(160)))
    assert stream.last_sequence_number == 4
    # A skipped sequence number (Telnyx says numbers can be skipped)
    # must not break the stream.
    stream.handle_message(_media(2, 160, _sine(160)))
    assert stream.last_sequence_number == 5


# -- framing -----------------------------------------------------------


def test_one_20ms_chunk_yields_two_10ms_frames():
    stream = _started_stream()
    frames = stream.handle_message(_media(1, 0, _sine(160)))
    assert len(frames) == 2
    for i, frame in enumerate(frames):
        assert isinstance(frame, MediaFrame)
        assert frame.samples.shape == (FRAME_SAMPLES_8K,)
        assert frame.sample_rate == 8000
        assert frame.seq == i
        assert frame.t_sec == pytest.approx(i * 0.01)
        assert frame.track == "inbound"
        assert frame.source == "telnyx"
        assert frame.stream_sid == STREAM_ID
        assert frame.call_id == CALL_ID
        assert not frame.discontinuous
    assert stream.stats()["frames"] == 2
    assert stream.stats()["samples"] == 160


def test_partial_payload_buffers_across_messages():
    stream = _started_stream()
    frames = stream.handle_message(_media(1, 0, _sine(100)))
    assert len(frames) == 1  # 80 emitted, 20 buffered
    # 100 samples = 100 RTP timestamp units, so the next timestamp is 100.
    frames += stream.handle_message(_media(2, 100, _sine(160, freq=880.0)))
    # 20 buffered + 160 new = 180 -> two more frames, 20 left over.
    assert len(frames) == 3
    assert frames[1].t_sec == pytest.approx(0.01)
    assert frames[2].t_sec == pytest.approx(0.02)
    assert stream.stats()["frames"] == 3


def test_pcma_media_decodes_to_frames():
    stream = TelnyxMediaStream(CALL_ID)
    stream.handle_message(_connected())
    stream.handle_message(_start(encoding="PCMA"))
    frames = stream.handle_message(_media(1, 0, _sine(160), encoding="PCMA"))
    assert len(frames) == 2
    assert all(f.sample_rate == 8000 for f in frames)
    assert stream.stats()["samples"] == 160


def test_rtp_timestamp_gap_marks_discontinuity_and_counts_loss():
    stream = _started_stream()
    stream.handle_message(_media(1, 0, _sine(160)))
    # Next timestamp should be 160 (one unit is one sample at 8 kHz);
    # it arrives at 480: 320 samples (40 ms) lost.
    frames = stream.handle_message(_media(2, 480, _sine(160, freq=880.0)))
    assert len(frames) == 2
    assert frames[0].discontinuous
    assert not frames[1].discontinuous
    assert frames[0].t_sec == pytest.approx(480 / 8000.0)
    stats = stream.stats()
    assert stats["gaps"] == 1
    assert stats["lost_samples"] == 320


def test_gap_discards_partial_frame_instead_of_splicing():
    stream = _started_stream()
    stream.handle_message(_media(1, 0, _sine(100)))  # 1 frame + 20 buffered
    frames = stream.handle_message(_media(2, 480, _sine(160)))
    # The 20 buffered samples belonged to the pre-gap timeline; they
    # are discarded (counted as lost), not spliced into a post-gap frame.
    assert len(frames) == 2
    assert frames[0].t_sec == pytest.approx(480 / 8000.0)
    assert frames[0].discontinuous
    # 380 samples of gap (480 - 100) + 20 discarded buffered samples.
    assert stream.stats()["lost_samples"] == 380 + 20


def test_duplicate_chunk_dropped():
    stream = _started_stream()
    stream.handle_message(_media(1, 0, _sine(160)))
    frames = stream.handle_message(_media(1, 0, _sine(160, freq=880.0)))
    assert frames == []
    assert stream.stats()["duplicates_dropped"] == 1
    assert stream.stats()["frames"] == 2


def test_out_of_order_timestamp_dropped():
    stream = _started_stream()
    stream.handle_message(_media(1, 0, _sine(160)))
    stream.handle_message(_media(2, 160, _sine(160)))
    # Timestamp rewinds to 40 units with a fresh chunk number.
    frames = stream.handle_message(_media(3, 40, _sine(160, freq=880.0)))
    assert frames == []
    assert stream.stats()["out_of_order_dropped"] == 1


# -- shared chunk counter ----------------------------------------------


def test_chunk_gap_on_other_track_counted_not_flagged():
    # chunk is ONE counter for both tracks. A jump from 1 to 3 on the
    # outbound leg means Telnyx lost media somewhere, but the inbound
    # timeline is continuous, so no inbound frame is flagged.
    stream = _started_stream()
    stream.handle_message(_media(1, 0, _sine(160)))
    assert stream.handle_message(_media(3, 0, _sine(160), track="outbound")) == []
    assert stream.stats()["chunk_gaps"] == 1
    frames = stream.handle_message(_media(4, 160, _sine(160)))
    assert len(frames) == 2
    assert not any(f.discontinuous for f in frames)
    assert stream.stats()["gaps"] == 0


def test_chunk_gap_with_timestamp_gap_flags_discontinuity():
    stream = _started_stream()
    stream.handle_message(_media(1, 0, _sine(160)))
    # Chunk 2 never arrived and the timestamp jumps with it: 160
    # samples lost on the analysed leg itself.
    frames = stream.handle_message(_media(3, 320, _sine(160, freq=880.0)))
    assert len(frames) == 2
    assert frames[0].discontinuous
    assert stream.stats()["chunk_gaps"] == 1
    assert stream.stats()["gaps"] == 1
    assert stream.stats()["lost_samples"] == 160


# -- tracks ------------------------------------------------------------


def test_other_track_counted_never_decoded():
    stream = _started_stream()
    frames = stream.handle_message(_media(1, 0, _sine(160), track="outbound"))
    assert frames == []
    assert stream.stats()["media_messages_other_track"] == 1
    assert stream.stats()["frames"] == 0
    frames = stream.handle_message(_media(1, 0, _sine(160), track="inbound"))
    assert len(frames) == 2


def test_outbound_analyzed_track_emits_outbound_only():
    stream = TelnyxMediaStream(CALL_ID, analyzed_track="outbound")
    stream.handle_message(_connected())
    stream.handle_message(_start())
    assert stream.handle_message(_media(1, 0, _sine(160), track="inbound")) == []
    frames = stream.handle_message(_media(1, 0, _sine(160), track="outbound"))
    assert len(frames) == 2
    assert all(f.track == "outbound" for f in frames)


def test_unknown_track_rejected():
    stream = _started_stream()
    with pytest.raises(TelnyxError):
        stream.handle_message(_media(1, 0, _sine(160), track="sideways"))


# -- stop / dtmf / mark / error ----------------------------------------


def test_stop_discards_partial_frame_and_closes():
    stream = _started_stream()
    stream.handle_message(_media(1, 0, _sine(100)))  # 20 samples buffered
    assert stream.handle_message(
        {"event": "stop", "sequence_number": "5",
         "stop": {"call_control_id": CALL_ID}, "stream_id": STREAM_ID}
    ) == []
    assert stream.is_stopped
    assert stream.stats()["discarded_on_stop"] == 20
    with pytest.raises(TelnyxError):
        stream.handle_message(_media(2, 100, _sine(160)))


def test_stop_rejects_call_control_id_mismatch():
    stream = _started_stream()
    with pytest.raises(TelnyxError):
        stream.handle_message(
            {"event": "stop", "stop": {"call_control_id": "v2:other"},
             "stream_id": STREAM_ID}
        )


def test_dtmf_records_digit_with_occurred_at():
    stream = _started_stream()
    assert stream.handle_message(
        {"event": "dtmf", "stream_id": STREAM_ID,
         "occurred_at": "2025-06-05T08:54:19.698408Z",
         "sequence_number": "5", "dtmf": {"digit": "1"}}
    ) == []
    assert stream.dtmf_events == [
        {"digit": "1", "occurred_at": "2025-06-05T08:54:19.698408Z"}
    ]
    assert stream.stats()["dtmf"] == 1


def test_dtmf_before_start_rejected():
    stream = TelnyxMediaStream(CALL_ID)
    stream.handle_message(_connected())
    with pytest.raises(TelnyxError):
        stream.handle_message(
            {"event": "dtmf", "stream_id": STREAM_ID,
             "occurred_at": "2025-06-05T08:54:19.698408Z",
             "sequence_number": "5", "dtmf": {"digit": "1"}}
        )


def test_mark_recorded_without_frames():
    stream = _started_stream()
    assert stream.handle_message(
        {"event": "mark", "stream_id": STREAM_ID,
         "sequence_number": "5", "mark": {"name": "prompt-done"}}
    ) == []
    assert stream.marks == ["prompt-done"]
    assert stream.stats()["marks"] == 1


def test_mark_wrong_stream_id_rejected():
    stream = _started_stream()
    with pytest.raises(TelnyxError):
        stream.handle_message(
            {"event": "mark", "stream_id": "other",
             "sequence_number": "5", "mark": {"name": "x"}}
        )


def test_telnyx_error_frame_raises():
    stream = _started_stream()
    with pytest.raises(TelnyxError, match="100004"):
        stream.handle_message(
            {"event": "error",
             "payload": {"code": 100004, "title": "invalid_media",
                         "detail": "Media provided was not base64 encoded"},
             "stream_id": STREAM_ID}
        )
    assert stream.stats()["errors"] == 1


def test_telnyx_error_frame_without_payload_raises():
    stream = _started_stream()
    with pytest.raises(TelnyxError):
        stream.handle_message({"event": "error", "stream_id": STREAM_ID})


def test_reconnect_is_a_fresh_instance_not_a_second_start():
    # Telnyx reconnects are new WebSocket connections: a fresh
    # adapter accepts a new start whose sequence_number and chunk
    # continue from the dropped connection. A second start on the
    # SAME instance is still an error.
    first = _started_stream()
    first.handle_message(_media(1, 0, _sine(160)))
    with pytest.raises(TelnyxError):
        first.handle_message(_start())
    second = TelnyxMediaStream(CALL_ID)
    second.handle_message(_connected())
    resumed = _start()
    resumed["sequence_number"] = "9"
    second.handle_message(resumed)
    frames = second.handle_message(_media(50, 0, _sine(160)))
    assert len(frames) == 2
    assert second.last_sequence_number == 53
