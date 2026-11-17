"""Twilio Media Streams adapter (ADR-008).

Protocol vectors come from Twilio's own docs
("Media Streams - WebSocket Messages"): connected/start/media/stop/
dtmf/mark shapes, mu-law 8 kHz, chunk starting at "1", timestamps in
milliseconds from stream start, 20 ms (160 byte) payloads.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from shieldcall.ingest.frame import FRAME_SAMPLES_8K, MediaFrame
from shieldcall.ingest.twilio import TwilioError, TwilioMediaStream, mulaw_decode

CALL_ID = "CA11111111111111111111111111111111"
STREAM_SID = "MZ22222222222222222222222222222222"


def _mulaw_encode(samples: np.ndarray) -> bytes:
    """G.711 mu-law encode (test helper only): pick the code whose
    decoded value is nearest to each sample. Correct by construction
    against the decode table; a wrong table would fail the known-vector
    test above, not hide behind a second hand-derived table."""
    from shieldcall.ingest.twilio import _MULAW_TABLE

    table = _MULAW_TABLE.astype(np.float64)
    flat = np.asarray(samples, dtype=np.float64).ravel()
    idx = np.argmin(np.abs(table[None, :] - flat[:, None]), axis=1)
    return idx.astype(np.uint8).tobytes()


def _payload(samples: np.ndarray) -> str:
    return base64.b64encode(_mulaw_encode(samples)).decode("ascii")


def _connected():
    return {"event": "connected", "protocol": "Call", "version": "1.0.0"}


def _start(call_id=CALL_ID, tracks=("inbound",), encoding="audio/x-mulaw",
           sample_rate=8000, channels=1):
    return {
        "event": "start",
        "sequenceNumber": "1",
        "start": {
            "accountSid": "AC33333333333333333333333333333333",
            "streamSid": STREAM_SID,
            "callSid": call_id,
            "tracks": list(tracks),
            "mediaFormat": {
                "encoding": encoding,
                "sampleRate": sample_rate,
                "channels": channels,
            },
            "customParameters": {},
        },
        "streamSid": STREAM_SID,
    }


def _media(chunk, timestamp_ms, samples, track="inbound"):
    return {
        "event": "media",
        "sequenceNumber": str(chunk + 2),
        "media": {
            "track": track,
            "chunk": str(chunk),
            "timestamp": str(timestamp_ms),
            "payload": _payload(samples),
        },
        "streamSid": STREAM_SID,
    }


def _started_stream(**kwargs):
    stream = TwilioMediaStream(CALL_ID, **kwargs)
    stream.handle_message(_connected())
    stream.handle_message(_start())
    return stream


def _sine(n, freq=440.0, amp=0.5):
    t = np.arange(n) / 8000.0
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# -- mu-law decode -----------------------------------------------------


def test_mulaw_known_vectors():
    # ITU-T G.711: 0xFF is silence, 0x00 / 0x80 are the extremes.
    assert mulaw_decode(bytes([0xFF]))[0] == pytest.approx(0.0)
    assert mulaw_decode(bytes([0x00]))[0] == pytest.approx(-32124 / 32768.0)
    assert mulaw_decode(bytes([0x80]))[0] == pytest.approx(32124 / 32768.0)
    assert mulaw_decode(b"").size == 0


def test_mulaw_round_trip_within_half_step():
    rng = np.random.RandomState(7)
    original = (rng.uniform(-0.9, 0.9, 4096)).astype(np.float32)
    decoded = mulaw_decode(_mulaw_encode(original))
    # Worst case is the top segment, where one mu-law step is
    # 1024/32768 = 0.03125, so a correct round trip stays within a
    # half step (0.0156) of the original. A wrong table shifts values
    # by a whole segment or more and breaks this bound.
    assert np.max(np.abs(decoded - original)) < 0.02
    assert np.all(np.abs(decoded) <= 1.0)


# -- lifecycle ---------------------------------------------------------


def test_connected_then_start_happy_path():
    stream = TwilioMediaStream(CALL_ID)
    assert stream.handle_message(_connected()) == []
    assert stream.handle_message(_start()) == []
    assert stream.is_started
    assert stream.stream_sid == STREAM_SID
    assert stream.call_sid == CALL_ID


def test_connected_rejects_wrong_protocol():
    stream = TwilioMediaStream(CALL_ID)
    with pytest.raises(TwilioError):
        stream.handle_message({"event": "connected", "protocol": "Chat"})


def test_media_before_start_rejected():
    stream = TwilioMediaStream(CALL_ID)
    stream.handle_message(_connected())
    with pytest.raises(TwilioError):
        stream.handle_message(_media(1, 0, _sine(160)))


def test_stop_before_start_rejected():
    stream = TwilioMediaStream(CALL_ID)
    with pytest.raises(TwilioError):
        stream.handle_message({"event": "stop", "streamSid": STREAM_SID})


def test_unknown_event_rejected():
    stream = _started_stream()
    with pytest.raises(TwilioError):
        stream.handle_message({"event": "clear", "streamSid": STREAM_SID})


def test_non_object_message_rejected():
    stream = _started_stream()
    with pytest.raises(TwilioError):
        stream.handle_message(["media"])  # type: ignore[arg-type]


def test_start_rejects_call_sid_mismatch():
    stream = TwilioMediaStream(CALL_ID)
    stream.handle_message(_connected())
    with pytest.raises(TwilioError):
        stream.handle_message(_start(call_id="CA99999999999999999999999999999999"))


def test_start_rejects_unsupported_format():
    for kwargs in (
        {"encoding": "audio/x-alaw"},
        {"sample_rate": 16000},
        {"channels": 2},
    ):
        stream = TwilioMediaStream(CALL_ID)
        stream.handle_message(_connected())
        with pytest.raises(TwilioError):
            stream.handle_message(_start(**kwargs))


def test_start_rejects_missing_media_format_and_second_start():
    stream = TwilioMediaStream(CALL_ID)
    stream.handle_message(_connected())
    bad = _start()
    del bad["start"]["mediaFormat"]
    with pytest.raises(TwilioError):
        stream.handle_message(bad)
    stream.handle_message(_start())
    with pytest.raises(TwilioError):
        stream.handle_message(_start())


def test_start_rejects_analyzed_track_not_offered():
    stream = TwilioMediaStream(CALL_ID, analyzed_track="outbound")
    stream.handle_message(_connected())
    with pytest.raises(TwilioError):
        stream.handle_message(_start(tracks=("inbound",)))


def test_media_wrong_stream_sid_rejected():
    stream = _started_stream()
    msg = _media(1, 0, _sine(160))
    msg["streamSid"] = "MZother"
    with pytest.raises(TwilioError):
        stream.handle_message(msg)


def test_media_bad_base64_and_empty_payload_rejected():
    stream = _started_stream()
    msg = _media(1, 0, _sine(160))
    msg["media"]["payload"] = "not!!base64"
    with pytest.raises(TwilioError):
        stream.handle_message(msg)
    msg2 = _media(1, 0, _sine(160))
    msg2["media"]["payload"] = ""
    with pytest.raises(TwilioError):
        stream.handle_message(msg2)


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
        assert frame.source == "twilio"
        assert frame.stream_sid == STREAM_SID
        assert frame.call_id == CALL_ID
        assert not frame.discontinuous
    assert stream.stats()["frames"] == 2
    assert stream.stats()["samples"] == 160


def test_partial_payload_buffers_across_messages():
    stream = _started_stream()
    frames = stream.handle_message(_media(1, 0, _sine(100)))
    assert len(frames) == 1  # 80 emitted, 20 buffered
    frames += stream.handle_message(_media(2, 12, _sine(160, freq=880.0)))
    # 20 buffered + 160 new = 180 -> two more frames, 20 left over.
    assert len(frames) == 3
    assert frames[1].t_sec == pytest.approx(0.01)
    assert frames[2].t_sec == pytest.approx(0.02)
    assert stream.stats()["frames"] == 3


def test_timestamp_gap_marks_discontinuity_and_counts_loss():
    stream = _started_stream()
    stream.handle_message(_media(1, 0, _sine(160)))
    # Next chunk should start at 20 ms; it arrives at 60 ms: 40 ms lost.
    frames = stream.handle_message(_media(2, 60, _sine(160, freq=880.0)))
    assert len(frames) == 2
    assert frames[0].discontinuous
    assert not frames[1].discontinuous
    assert frames[0].t_sec == pytest.approx(0.06)
    stats = stream.stats()
    assert stats["gaps"] == 1
    assert stats["lost_samples"] == 320  # 40 ms at 8 kHz


def test_gap_discards_partial_frame_instead_of_splicing():
    stream = _started_stream()
    stream.handle_message(_media(1, 0, _sine(100)))  # 1 frame + 20 buffered
    frames = stream.handle_message(_media(2, 60, _sine(160)))
    # The 20 buffered samples belonged to the pre-gap timeline; they
    # are discarded (counted as lost), not spliced into a post-gap frame.
    assert len(frames) == 2
    assert frames[0].t_sec == pytest.approx(0.06)
    assert frames[0].discontinuous
    assert stream.stats()["lost_samples"] == 380 + 20  # 47.5 ms gap + 20 discarded


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
    stream.handle_message(_media(2, 20, _sine(160)))
    # Timestamp rewinds to 5 ms with a fresh chunk number.
    frames = stream.handle_message(_media(3, 5, _sine(160, freq=880.0)))
    assert frames == []
    assert stream.stats()["out_of_order_dropped"] == 1


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
    stream = TwilioMediaStream(CALL_ID, analyzed_track="outbound")
    stream.handle_message(_connected())
    stream.handle_message(_start(tracks=("inbound", "outbound")))
    assert stream.handle_message(_media(1, 0, _sine(160), track="inbound")) == []
    frames = stream.handle_message(_media(1, 0, _sine(160), track="outbound"))
    assert len(frames) == 2
    assert all(f.track == "outbound" for f in frames)


def test_unknown_track_rejected():
    stream = _started_stream()
    with pytest.raises(TwilioError):
        stream.handle_message(_media(1, 0, _sine(160), track="sideways"))


# -- stop / dtmf / mark ------------------------------------------------


def test_stop_discards_partial_frame_and_closes():
    stream = _started_stream()
    stream.handle_message(_media(1, 0, _sine(100)))  # 20 samples buffered
    assert stream.handle_message(
        {"event": "stop", "stop": {"callSid": CALL_ID}, "streamSid": STREAM_SID}
    ) == []
    assert stream.is_stopped
    assert stream.stats()["discarded_on_stop"] == 20
    with pytest.raises(TwilioError):
        stream.handle_message(_media(2, 12, _sine(160)))


def test_stop_rejects_call_sid_mismatch():
    stream = _started_stream()
    with pytest.raises(TwilioError):
        stream.handle_message(
            {"event": "stop", "stop": {"callSid": "CAother"}, "streamSid": STREAM_SID}
        )


def test_dtmf_and_mark_recorded_without_frames():
    stream = _started_stream()
    assert stream.handle_message(
        {"event": "dtmf", "dtmf": {"track": "inbound", "digit": "1"},
         "streamSid": STREAM_SID}
    ) == []
    assert stream.dtmf_digits == ["1"]
    assert stream.handle_message(
        {"event": "mark", "mark": {"name": "prompt-done"}, "streamSid": STREAM_SID}
    ) == []
    assert stream.marks == ["prompt-done"]
    assert stream.stats()["dtmf"] == 1
    assert stream.stats()["marks"] == 1
