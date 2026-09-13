"""
P2-1: streaming ASR provider behind the circuit breaker.

Uses LoopbackTransport (no network, no API key) with a configurable
simulated provider RTT.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from shieldcall.linguistic.asr_bridge import PassthroughASR
from shieldcall.linguistic.deepgram_stream import (
    DeepgramStreamingASR,
    LoopbackTransport,
    TransportError,
    _to_pcm16_mono_16k,
)
from shieldcall.linguistic.provider import make_asr_from_env
from shieldcall.runtime.asr_gate import GatedASR
from shieldcall.runtime.breaker import BreakerState, CircuitBreaker

SR = 8000
FRAME = np.zeros(200, dtype=np.float32)  # 25 ms @ 8 kHz


def _make(loopback_ms: float = 20.0) -> DeepgramStreamingASR:
    return DeepgramStreamingASR(
        transport_factory=lambda: LoopbackTransport(latency_ms=loopback_ms)
    )


def test_loopback_returns_final_fragment_after_rtt():
    asr = _make(loopback_ms=50.0)
    try:
        got = asr.push_audio(FRAME, SR, 0.0)
        assert got == []  # nothing arrived yet; push never blocks
        deadline = time.monotonic() + 5.0
        frags = []
        while time.monotonic() < deadline and not frags:
            time.sleep(0.02)
            frags = asr.push_audio(FRAME, SR, 0.1)
        assert frags, "no fragment arrived within 5 s at 50 ms simulated RTT"
        assert frags[0].is_final
        assert frags[0].text == "hello this is a test"
        assert frags[0].confidence > 0.9
    finally:
        asr.close()


def test_push_audio_never_blocks_on_provider_rtt():
    asr = _make(loopback_ms=300.0)
    try:
        t0 = time.monotonic()
        for i in range(50):
            asr.push_audio(FRAME, SR, i * 0.025)
        dt = time.monotonic() - t0
        # 50 frames at 300 ms provider RTT: blocking code would take ~15 s.
        assert dt < 2.0, f"push_audio blocked: 50 pushes took {dt:.2f}s"
    finally:
        asr.close()


def test_8k_input_upsampled_to_16k_pcm16():
    pcm = _to_pcm16_mono_16k(FRAME, SR)
    assert len(pcm) == 800  # 200 samples @8k -> 400 @16k -> 800 bytes PCM16
    pcm2 = _to_pcm16_mono_16k(np.zeros(160, dtype=np.float32), 16000)
    assert len(pcm2) == 320
    assert _to_pcm16_mono_16k(np.zeros(0, dtype=np.float32), SR) == b""


def test_transport_failure_raises_so_breaker_can_trip():
    def bad_factory():
        raise TransportError("boom")

    asr = DeepgramStreamingASR(transport_factory=bad_factory)
    try:
        deadline = time.monotonic() + 5.0
        raised = False
        while time.monotonic() < deadline:
            try:
                asr.push_audio(FRAME, SR, 0.0)
            except TransportError:
                raised = True
                break
            time.sleep(0.02)
        assert raised, "transport error never surfaced within 5 s"
    finally:
        asr.close()


def test_gated_asr_opens_breaker_on_provider_failure():
    class Failing:
        def push_audio(self, *a):
            raise TransportError("down")

        def push_text(self, *a, **k):
            raise TransportError("down")

        def reset(self):
            pass

    breaker = CircuitBreaker(fail_threshold=3, reset_after_sec=60.0)
    gated = GatedASR(Failing(), breaker)
    for _ in range(3):
        assert gated.push_audio(FRAME, SR, 0.0) == []
    assert breaker.state is BreakerState.OPEN
    # Open breaker: provider not even consulted, empty result, fail-open.
    assert gated.push_audio(FRAME, SR, 0.0) == []


def test_push_text_bypasses_provider():
    asr = _make()
    try:
        frag = asr.push_text("suspicious call", 1.5)
        assert frag.text == "suspicious call"
        assert frag.is_final
    finally:
        asr.close()


def test_make_asr_from_env_defaults_to_passthrough():
    assert isinstance(make_asr_from_env({}), PassthroughASR)
    assert isinstance(make_asr_from_env({"SHIELDCALL_ASR_PROVIDER": "wat"}), PassthroughASR)


def test_make_asr_from_env_deepgram_needs_key():
    assert isinstance(
        make_asr_from_env({"SHIELDCALL_ASR_PROVIDER": "deepgram"}), PassthroughASR
    )
    asr = make_asr_from_env(
        {"SHIELDCALL_ASR_PROVIDER": "deepgram", "DEEPGRAM_API_KEY": "k"}
    )
    assert isinstance(asr, DeepgramStreamingASR)
    asr.close()
