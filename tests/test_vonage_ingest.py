"""Vonage Voice API WebSocket adapter (ADR-012).

Protocol shapes follow Vonage's own docs ("WebSocket Voice Chat
API Guide"): a websocket:connected text message carrying the
negotiated content-type, raw L16 binary audio, interleaved
websocket:dtmf events, and no stop event. The pure translation is
covered here; the route is covered by tests/test_vonage_route.py.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from shieldcall.ingest.frame import FRAME_SAMPLES_8K, MediaFrame
from shieldcall.ingest.vonage import VonageError, VonageMediaStream

CALL_ID = "aaaaaaaa-bbbb-cccc-dddd-0123456789ab"


def _l16(values) -> bytes:
    """Encode int16 sample values as little-endian L16 bytes."""
    return struct.pack(f"<{len(values)}h", *values)


def _connected(content_type="audio/l16;rate=8000", **extra):
    msg = {"event": "websocket:connected", "content-type": content_type}
    msg.update(extra)
    return msg


def _started_stream(**kwargs):
    stream = VonageMediaStream(CALL_ID, **kwargs)
    stream.handle_message(_connected())
    return stream


def _silence_packets(n_packets=1, samples_per_packet=160):
    return [_l16([0] * samples_per_packet) for _ in range(n_packets)]


# -- lifecycle and format ----------------------------------------------


def test_call_id_required():
    with pytest.raises(VonageError, match="call_id is required"):
        VonageMediaStream("")


def test_connected_records_content_type_and_headers():
    stream = VonageMediaStream(CALL_ID)
    frames = stream.handle_message(
        _connected(**{"caller-id": "447700900123", "prop1": "value1"})
    )
    assert frames == []
    assert stream.is_started
    assert stream.content_type == "audio/l16;rate=8000"
    assert stream.headers == {"caller-id": "447700900123", "prop1": "value1"}
    assert stream.reconnect_attempt == 0


def test_connected_content_type_parsing_is_case_insensitive():
    stream = VonageMediaStream(CALL_ID)
    stream.handle_message(_connected("Audio/L16; Rate=8000"))
    assert stream.is_started


def test_connected_missing_content_type_rejected():
    stream = VonageMediaStream(CALL_ID)
    with pytest.raises(VonageError, match="no content-type"):
        stream.handle_message({"event": "websocket:connected"})


@pytest.mark.parametrize("content_type", ["audio/l16;rate=16000", "audio/l16;rate=24000"])
def test_non_8k_rate_rejected(content_type):
    stream = VonageMediaStream(CALL_ID)
    with pytest.raises(VonageError, match="sample rate"):
        stream.handle_message(_connected(content_type))


def test_non_l16_encoding_rejected():
    stream = VonageMediaStream(CALL_ID)
    with pytest.raises(VonageError, match="content-type"):
        stream.handle_message(_connected("audio/x-mulaw;rate=8000"))


def test_content_type_without_rate_rejected():
    stream = VonageMediaStream(CALL_ID)
    with pytest.raises(VonageError, match="no rate"):
        stream.handle_message(_connected("audio/l16"))


def test_audio_before_connected_rejected():
    stream = VonageMediaStream(CALL_ID)
    with pytest.raises(VonageError, match="before websocket:connected"):
        stream.handle_audio(_l16([0] * 80))


def test_unknown_event_rejected():
    stream = _started_stream()
    with pytest.raises(VonageError, match="unknown Vonage event"):
        stream.handle_message({"event": "websocket:something-else"})


def test_non_object_message_rejected():
    stream = _started_stream()
    with pytest.raises(VonageError, match="JSON object"):
        stream.handle_message(["websocket:connected"])  # type: ignore[arg-type]


def test_second_connected_without_reconnect_rejected():
    stream = _started_stream()
    with pytest.raises(VonageError, match="second websocket:connected"):
        stream.handle_message(_connected())


# -- L16 decoding and framing ------------------------------------------


def test_l16_known_vector_decodes_exactly():
    stream = _started_stream()
    values = [0, 16384, -16384, 32767, -32768]
    padding = [0] * (FRAME_SAMPLES_8K - len(values))
    frames = stream.handle_audio(_l16(values + padding))
    assert len(frames) == 1
    frame = frames[0]
    assert isinstance(frame, MediaFrame)
    assert frame.call_id == CALL_ID
    assert frame.stream_sid == ""
    assert frame.track == "inbound"
    assert frame.seq == 0
    assert frame.t_sec == pytest.approx(0.0)
    assert frame.sample_rate == 8000
    assert frame.source == "vonage"
    assert frame.discontinuous is False
    expected = np.array(
        [0.0, 0.5, -0.5, 32767 / 32768.0, -1.0] + [0.0] * len(padding),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(frame.samples, expected)


def test_twenty_ms_packet_yields_two_frames_with_advancing_clock():
    stream = _started_stream()
    frames = stream.handle_audio(_l16([0] * 160))
    assert len(frames) == 2
    assert [f.seq for f in frames] == [0, 1]
    assert frames[0].t_sec == pytest.approx(0.0)
    assert frames[1].t_sec == pytest.approx(0.01)


def test_audio_buffers_across_packet_boundaries():
    stream = _started_stream()
    # 100 samples: one frame, 20 buffered. Then 60 more: the buffered
    # 20 plus 60 make the second frame, with nothing left over.
    assert len(stream.handle_audio(_l16([0] * 100))) == 1
    frames = stream.handle_audio(_l16([0] * 60))
    assert len(frames) == 1
    assert frames[0].seq == 1
    assert frames[0].t_sec == pytest.approx(0.01)
    assert stream.stats()["samples"] == 160


def test_small_packets_accumulate_before_any_frame():
    stream = _started_stream()
    assert stream.handle_audio(_l16([0] * 40)) == []
    assert stream.handle_audio(_l16([0] * 39)) == []
    frames = stream.handle_audio(_l16([0] * 1))
    assert len(frames) == 1
    assert frames[0].seq == 0


def test_empty_audio_packet_rejected():
    stream = _started_stream()
    with pytest.raises(VonageError, match="empty audio packet"):
        stream.handle_audio(b"")


def test_odd_byte_count_rejected_as_truncated_sample():
    stream = _started_stream()
    with pytest.raises(VonageError, match="odd byte count"):
        stream.handle_audio(b"\x00\x00\x00")


def test_non_bytes_audio_rejected():
    stream = _started_stream()
    with pytest.raises(VonageError, match="must be bytes"):
        stream.handle_audio("not bytes")  # type: ignore[arg-type]


# -- finish: no stop event, socket close ends the stream ----------------


def test_finish_discards_partial_frame_and_counts_it():
    stream = _started_stream()
    assert stream.handle_audio(_l16([0] * 40)) == []
    stream.finish()
    assert stream.is_finished
    assert stream.stats()["discarded_on_close"] == 40


def test_finish_is_idempotent():
    stream = _started_stream()
    stream.finish()
    stream.finish()
    assert stream.stats()["discarded_on_close"] == 0


def test_messages_after_finish_rejected():
    stream = _started_stream()
    stream.finish()
    with pytest.raises(VonageError, match="after the stream finished"):
        stream.handle_audio(_l16([0] * 80))
    with pytest.raises(VonageError, match="after finish"):
        stream.handle_message(_connected())
    with pytest.raises(VonageError, match="after finish"):
        stream.handle_message(
            {"event": "websocket:dtmf", "digit": "5", "duration": 260}
        )


# -- DTMF, cleared, notify ----------------------------------------------


def test_dtmf_recorded_with_duration_and_no_frames():
    stream = _started_stream()
    frames = stream.handle_message(
        {"event": "websocket:dtmf", "digit": "5", "duration": 260}
    )
    assert frames == []
    assert stream.dtmf_events == [{"digit": "5", "duration_ms": 260}]
    assert stream.stats()["dtmf"] == 1


def test_dtmf_without_duration_recorded_as_none():
    stream = _started_stream()
    stream.handle_message({"event": "websocket:dtmf", "digit": "#"})
    assert stream.dtmf_events == [{"digit": "#", "duration_ms": None}]


@pytest.mark.parametrize("digit", ["", "A", "12", None, 5])
def test_invalid_dtmf_digit_rejected(digit):
    stream = _started_stream()
    with pytest.raises(VonageError, match="keypad digit"):
        stream.handle_message(
            {"event": "websocket:dtmf", "digit": digit, "duration": 100}
        )


def test_negative_dtmf_duration_rejected():
    stream = _started_stream()
    with pytest.raises(VonageError, match="negative"):
        stream.handle_message(
            {"event": "websocket:dtmf", "digit": "5", "duration": -1}
        )


def test_dtmf_before_connected_rejected():
    stream = VonageMediaStream(CALL_ID)
    with pytest.raises(VonageError, match="before websocket:connected"):
        stream.handle_message(
            {"event": "websocket:dtmf", "digit": "5", "duration": 260}
        )


def test_cleared_and_notify_recorded_without_frames():
    stream = _started_stream()
    assert stream.handle_message({"event": "websocket:cleared"}) == []
    assert (
        stream.handle_message(
            {
                "event": "websocket:notify",
                "payload": {"customKey": "customValue"},
            }
        )
        == []
    )
    stats = stream.stats()
    assert stats["cleared"] == 1
    assert stats["notifications"] == 1
    assert stream.notify_payloads == [{"customKey": "customValue"}]


def test_cleared_before_connected_rejected():
    stream = VonageMediaStream(CALL_ID)
    with pytest.raises(VonageError, match="before websocket:connected"):
        stream.handle_message({"event": "websocket:cleared"})


# -- reconnection --------------------------------------------------------


def test_fresh_instance_with_reconnect_attempt_flags_first_frame():
    stream = VonageMediaStream(CALL_ID)
    stream.handle_message(_connected(reconnect_attempt=1))
    frames = stream.handle_audio(_l16([0] * 160))
    assert len(frames) == 2
    assert frames[0].discontinuous is True
    assert frames[1].discontinuous is False


def test_second_connected_with_reconnect_attempt_discards_partial():
    stream = _started_stream()
    assert stream.handle_audio(_l16([0] * 40)) == []
    stream.handle_message(_connected(reconnect_attempt=2))
    stats = stream.stats()
    assert stats["reconnects"] == 1
    assert stats["discarded_on_reconnect"] == 40
    frames = stream.handle_audio(_l16([0] * 80))
    assert len(frames) == 1
    assert frames[0].discontinuous is True
    assert frames[0].seq == 0


def test_negative_reconnect_attempt_rejected():
    stream = VonageMediaStream(CALL_ID)
    with pytest.raises(VonageError, match="negative"):
        stream.handle_message(_connected(reconnect_attempt=-1))


def test_non_integer_reconnect_attempt_rejected():
    stream = VonageMediaStream(CALL_ID)
    with pytest.raises(VonageError, match="not an integer"):
        stream.handle_message(_connected(reconnect_attempt="soon"))


# -- stats ---------------------------------------------------------------


def test_stats_count_packets_bytes_samples_and_frames():
    stream = _started_stream()
    stream.handle_audio(_l16([0] * 160))
    stream.handle_audio(_l16([0] * 160))
    stream.handle_message(
        {"event": "websocket:dtmf", "digit": "1", "duration": 100}
    )
    stats = stream.stats()
    assert stats["messages"] == 2
    assert stats["audio_packets"] == 2
    assert stats["audio_bytes"] == 640
    assert stats["samples"] == 320
    assert stats["frames"] == 4
    assert stats["dtmf"] == 1
