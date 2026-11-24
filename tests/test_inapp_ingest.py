"""In-app VoIP/WebRTC adapter: pure translation (ADR-015).

Protocol cases use a stub decoder returning a fixed 20 ms of
constant audio per packet, so they run on any host. The real
decode path is covered separately with a tiny ctypes Opus
encoder against the system libopus, and skipped where libopus
is absent.
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.util

import numpy as np
import pytest

from shieldcall.ingest.frame import FRAME_SAMPLES_8K, MediaFrame
from shieldcall.ingest.inapp import InAppError, InAppMediaStream
from shieldcall.ingest.opus import OpusDecoderUnavailable, OpusError

CALL_ID = "inapp-call-1"


class StubDecoder:
    """Returns `samples_per_decode` constant samples per packet."""

    def __init__(self, samples_per_decode: int = 160, value: float = 0.5):
        self.samples_per_decode = samples_per_decode
        self.value = value
        self.calls = 0
        self.closed = False

    def decode(self, packet: bytes):
        self.calls += 1
        return np.full(self.samples_per_decode, self.value, dtype=np.float32)

    def close(self):
        self.closed = True


class FailingDecoder:
    def decode(self, packet: bytes):
        raise OpusError("corrupted stream")

    def close(self):
        pass


def _hello(**extra):
    msg = {
        "type": "hello",
        "codec": "opus",
        "sample_rate": 48000,
        "channels": 1,
        "track": "far-end",
    }
    msg.update(extra)
    return msg


def _media(seq: int, timestamp: int, payload: bytes = b"\x01\x02\x03"):
    return {
        "type": "media",
        "seq": seq,
        "timestamp": timestamp,
        "payload": base64.b64encode(payload).decode("ascii"),
    }


def _stream(**kwargs):
    decoder = kwargs.pop("decoder", StubDecoder())
    return InAppMediaStream(call_id=CALL_ID, decoder=decoder)


def _feed(stream, messages):
    frames = []
    for msg in messages:
        frames.extend(stream.handle_message(msg))
    return frames


# -- hello ---------------------------------------------------------------


def test_media_before_hello_is_refused():
    stream = _stream()
    with pytest.raises(InAppError, match="hello"):
        stream.handle_message(_media(0, 0))


def test_non_dict_message_is_refused():
    stream = _stream()
    with pytest.raises(InAppError, match="JSON object"):
        stream.handle_message(["hello"])


def test_second_hello_is_refused():
    stream = _stream()
    stream.handle_message(_hello())
    with pytest.raises(InAppError, match="second hello"):
        stream.handle_message(_hello())


@pytest.mark.parametrize(
    "override,match",
    [
        ({"codec": "g711"}, "unsupported codec"),
        ({"codec": "OPUSX"}, "unsupported codec"),
        ({"sample_rate": 44100}, "unsupported sample_rate"),
        ({"sample_rate": "48000"}, "unsupported sample_rate"),
        ({"sample_rate": True}, "unsupported sample_rate"),
        ({"channels": 2}, "unsupported channels"),
        ({"channels": 0}, "unsupported channels"),
        ({"track": "mixed"}, "unknown track"),
        ({"track": "inbound"}, "unknown track"),
        ({"call_id": "another-call"}, "does not match"),
        ({"stream_id": ""}, "stream_id"),
        ({"consent_ref": 42}, "consent_ref"),
    ],
)
def test_hello_validation(override, match):
    stream = _stream()
    with pytest.raises(InAppError, match=match):
        stream.handle_message(_hello(**override))


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 48000])
def test_all_encoded_rates_accepted(rate):
    stream = _stream()
    stream.handle_message(_hello(sample_rate=rate))
    frames = _feed(stream, [_media(0, 0)])
    assert frames[0].source_rate == rate
    assert frames[0].sample_rate == 8000


def test_hello_records_stream_id_and_consent_ref_without_validating_it():
    # consent_ref is recorded for audit; no consent ledger exists
    # to validate it against (ADR-015 item 8). Recording it must
    # never be mistaken for a check.
    stream = _stream()
    stream.handle_message(
        _hello(stream_id="s-1", consent_ref="consent-123")
    )
    assert stream.stream_sid == "s-1"
    assert stream.consent_ref == "consent-123"


def test_hello_without_libopus_fails_closed(monkeypatch):
    import shieldcall.ingest.inapp as inapp_module

    def _unavailable(*args, **kwargs):
        raise OpusDecoderUnavailable("no libopus here")

    monkeypatch.setattr(inapp_module, "OpusDecoder", _unavailable)
    stream = InAppMediaStream(call_id=CALL_ID)
    with pytest.raises(InAppError, match="libopus"):
        stream.handle_message(_hello())


def test_unknown_message_type_is_refused():
    stream = _stream()
    stream.handle_message(_hello())
    with pytest.raises(InAppError, match="unknown message type"):
        stream.handle_message({"type": "ping"})


# -- happy path ----------------------------------------------------------


def test_in_order_packets_emit_frames_with_rtp_clock_timing():
    stream = _stream()
    frames = _feed(
        stream,
        [_hello()]
        + [_media(seq, seq * 960) for seq in range(5)],
    )
    assert len(frames) == 10
    assert [f.seq for f in frames] == list(range(10))
    assert [f.t_sec for f in frames] == pytest.approx(
        [i * 0.01 for i in range(10)]
    )
    for frame in frames:
        assert frame.samples.shape == (FRAME_SAMPLES_8K,)
        assert frame.sample_rate == 8000
        assert frame.source == "inapp"
        assert frame.source_codec == "opus"
        assert frame.source_rate == 48000
        assert frame.track == "inbound"
        assert frame.discontinuous is False
    assert stream.stats()["frames"] == 10
    assert stream.stats()["samples"] == 800


def test_frames_carry_no_faked_bandwidth():
    # Decoded at 8 kHz from 48 kHz Opus: the envelope says so.
    stream = _stream()
    frames = _feed(stream, [_hello(), _media(0, 0)])
    assert frames[0].sample_rate == 8000
    assert frames[0].source_rate == 48000


# -- near-end leg ----------------------------------------------------------


def test_near_end_is_counted_never_decoded_or_framed():
    decoder = StubDecoder()
    stream = _stream(decoder=decoder)
    stream.handle_message(_hello(track="near-end"))
    frames = _feed(stream, [_media(0, 0), _media(1, 960), _media(3, 2880)])
    assert frames == []
    assert decoder.calls == 0
    stats = stream.stats()
    assert stats["near_end_messages"] == 3
    assert stats["gaps"] == 1
    assert stats["lost_packets"] == 1


# -- ordering, gaps, wraparound --------------------------------------------


def test_out_of_order_within_window_emits_in_seq_order():
    stream = _stream()
    frames = _feed(
        stream,
        [_hello(), _media(1, 960), _media(0, 0), _media(2, 1920)],
    )
    assert len(frames) == 6
    assert [f.t_sec for f in frames] == pytest.approx(
        [i * 0.01 for i in range(6)]
    )
    assert all(f.discontinuous is False for f in frames)


def test_missing_packet_flushes_after_reorder_window_with_gap_flag():
    stream = _stream()
    frames = _feed(
        stream,
        [_hello()]
        + [_media(seq, seq * 960) for seq in (0, 2, 3, 4, 5)],
    )
    # seq 0 emits 2 frames at once; seqs 2..5 wait for seq 1 until
    # audio 60 ms past the gap has arrived, then flush with the
    # first post-gap frame flagged.
    assert len(frames) == 10
    assert frames[0].discontinuous is False
    assert frames[2].discontinuous is True
    assert all(f.discontinuous is False for f in frames[3:])
    stats = stream.stats()
    assert stats["gaps"] == 1
    assert stats["lost_packets"] == 1
    assert stats["lost_samples"] == 160


def test_late_packet_after_flush_is_dropped():
    stream = _stream()
    _feed(stream, [_hello()] + [_media(seq, seq * 960) for seq in (0, 2, 3, 4, 5)])
    frames = stream.handle_message(_media(1, 960))
    assert frames == []
    assert stream.stats()["late_dropped"] == 1


def test_duplicate_buffered_packet_is_dropped():
    stream = _stream()
    _feed(stream, [_hello(), _media(0, 0), _media(2, 1920)])
    frames = stream.handle_message(_media(2, 1920))
    assert frames == []
    assert stream.stats()["duplicates_dropped"] == 1


def test_sequence_wraparound_is_continuous():
    # Sequence numbers start at 0 on every fork, so wraparound is
    # only reachable by running the full 16-bit cycle. One-sample
    # stub packets keep that cheap: the point is the modular
    # sequence accounting, not the audio.
    stream = _stream(decoder=StubDecoder(samples_per_decode=1))
    stream.handle_message(_hello())
    # Timestamps must advance with the audio they carry: one
    # 8 kHz sample is 6 ticks on the 48 kHz clock.
    for seq in range(65536):
        stream.handle_message(_media(seq, seq * 6))
    assert stream.stats()["packets"] == 65536
    stream.handle_message(_media(0, 65536 * 6))
    stream.handle_message(_media(1, 65537 * 6))
    assert stream.stats()["packets"] == 65538
    assert stream.stats()["late_dropped"] == 0
    assert stream.stats()["gaps"] == 0


def test_wrapped_sequence_arriving_early_is_late_not_a_jump():
    # After seq 0, a packet numbered 65535 is the previous cycle:
    # it is dropped as late, never mistaken for a huge jump.
    stream = _stream()
    stream.handle_message(_hello())
    stream.handle_message(_media(0, 0))
    assert stream.handle_message(_media(65535, 0)) == []
    assert stream.stats()["late_dropped"] == 1


def test_timestamp_wraparound_keeps_the_clock_monotonic():
    stream = _stream()
    base = (1 << 32) - 960
    frames = _feed(
        stream,
        [_hello(), _media(0, base), _media(1, 0), _media(2, 960)],
    )
    assert [f.t_sec for f in frames] == pytest.approx(
        [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    )


def test_sender_clock_jump_with_contiguous_seq_is_a_gap():
    stream = _stream()
    frames = _feed(stream, [_hello(), _media(0, 0), _media(1, 9600)])
    assert len(frames) == 4
    assert frames[2].t_sec == pytest.approx(0.20)
    assert frames[2].discontinuous is True
    stats = stream.stats()
    assert stats["timestamp_jumps"] == 1
    assert stats["gaps"] == 1


def test_huge_sequence_jump_is_refused_not_resynced():
    stream = _stream()
    stream.handle_message(_hello())
    stream.handle_message(_media(0, 0))
    with pytest.raises(InAppError, match="sequence jump"):
        stream.handle_message(_media(1000, 960))


# -- payload validation ----------------------------------------------------


@pytest.mark.parametrize(
    "override,match",
    [
        ({"seq": -1}, "out of range"),
        ({"seq": 65536}, "out of range"),
        ({"seq": "0"}, "not an integer"),
        ({"seq": True}, "not an integer"),
        ({"timestamp": -1}, "out of range"),
        ({"timestamp": 1 << 32}, "out of range"),
        ({"payload": 123}, "base64"),
        ({"payload": "not base64!!"}, "not valid base64"),
        ({"payload": ""}, "empty"),
        ({"payload": base64.b64encode(b"x" * 4097).decode()}, "over the"),
    ],
)
def test_media_validation(override, match):
    stream = _stream()
    stream.handle_message(_hello())
    msg = _media(0, 0)
    msg.update(override)
    with pytest.raises(InAppError, match=match):
        stream.handle_message(msg)


def test_corrupt_opus_packet_is_an_error_not_silence():
    stream = _stream(decoder=FailingDecoder())
    stream.handle_message(_hello())
    with pytest.raises(InAppError, match="Opus decode failed"):
        stream.handle_message(_media(0, 0))


# -- stop / finish ---------------------------------------------------------


def test_stop_discards_partial_and_ends_stream():
    decoder = StubDecoder(samples_per_decode=40)  # 5 ms: never a frame
    stream = _stream(decoder=decoder)
    stream.handle_message(_hello())
    assert stream.handle_message(_media(0, 0)) == []
    assert stream.handle_message({"type": "stop"}) == []
    assert stream.is_finished
    assert stream.stats()["discarded_on_stop"] == 40
    with pytest.raises(InAppError, match="finished"):
        stream.handle_message(_media(1, 960))


def test_finish_is_idempotent_and_discards_partial():
    decoder = StubDecoder(samples_per_decode=40)
    stream = _stream(decoder=decoder)
    stream.handle_message(_hello())
    stream.handle_message(_media(0, 0))
    stream.finish()
    stream.finish()
    assert stream.stats()["discarded_on_close"] == 40
    assert decoder.closed


def test_finish_discards_buffered_reordered_packets():
    stream = _stream()
    stream.handle_message(_hello())
    stream.handle_message(_media(0, 0))
    stream.handle_message(_media(2, 1920))
    stream.finish()
    assert stream.stats()["pending_discarded"] == 1


# -- MediaFrame provenance defaults ----------------------------------------


def test_media_frame_provenance_defaults_mean_unknown():
    frame = MediaFrame(
        call_id=CALL_ID,
        stream_sid="",
        track="inbound",
        seq=0,
        t_sec=0.0,
        samples=np.zeros(FRAME_SAMPLES_8K, dtype=np.float32),
    )
    assert frame.source_codec == ""
    assert frame.source_rate == 0


# -- real Opus decode -------------------------------------------------------

_libopus_available = ctypes.util.find_library("opus") is not None


def _make_encoder(rate: int):
    lib = ctypes.CDLL(ctypes.util.find_library("opus"))
    lib.opus_encoder_create.argtypes = [
        ctypes.c_int32,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.opus_encoder_create.restype = ctypes.c_void_p
    lib.opus_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_short),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.opus_encode.restype = ctypes.c_int
    lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_encoder_destroy.restype = None
    err = ctypes.c_int(0)
    handle = lib.opus_encoder_create(rate, 1, 2048, ctypes.byref(err))
    assert handle and err.value == 0
    return lib, handle


def _encode_sine(rate: int, freq: float = 440.0, seconds: float = 0.02):
    """Encode `seconds` of sine at `rate` with real libopus."""
    lib, handle = _make_encoder(rate)
    try:
        n = int(rate * seconds)
        t = np.arange(n) / rate
        pcm = (0.4 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        out = ctypes.create_string_buffer(4000)
        count = lib.opus_encode(
            handle,
            pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
            n,
            out,
            4000,
        )
        assert count > 0
        return out.raw[:count]
    finally:
        lib.opus_encoder_destroy(handle)


@pytest.mark.skipif(not _libopus_available, reason="libopus not installed")
@pytest.mark.parametrize("rate", [8000, 16000, 24000, 48000])
def test_real_opus_round_trip_emits_frames(rate):
    stream = InAppMediaStream(call_id=CALL_ID)  # real decoder
    stream.handle_message(_hello(sample_rate=rate))
    ticks_per_packet = 48000 // 50  # 20 ms on the 48 kHz clock
    frames = _feed(
        stream,
        [_media(seq, seq * ticks_per_packet, _encode_sine(rate)) for seq in range(5)],
    )
    assert len(frames) == 10
    assert all(np.isfinite(f.samples).all() for f in frames)
    rms = float(np.sqrt(np.mean(np.concatenate([f.samples for f in frames]) ** 2)))
    assert rms > 0.01, "decoded audio must carry the encoded sine, not silence"
    assert frames[0].source_rate == rate


@pytest.mark.skipif(not _libopus_available, reason="libopus not installed")
def test_real_opus_corrupt_packet_is_refused():
    stream = InAppMediaStream(call_id=CALL_ID)
    stream.handle_message(_hello())
    with pytest.raises(InAppError, match="Opus decode failed"):
        stream.handle_message(_media(0, 0, b"\xff\xff\xff"))
