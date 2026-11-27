"""Pipeline shadow wiring (ADR-018, TDD v2.0 section 10.3).

The cascade and the factorial model run beside CSCF in
shadow: decisions and Tier 1 scores are recorded beside
CSCF verdicts, and factorial states beside SDTG scores.
Nothing here may change a live verdict, delay the call
thread, or invent a Tier 1 score. Ground truth is
synthetic audio and transcript turns with known content.
No licensed ASVspoof data is touched, and no model is
downloaded: Tier 1 is tested through injected scorers.
"""

import threading

import numpy as np
import pytest

from shieldcall import ShieldCallPipeline, PipelineConfig
from shieldcall.acoustic.cascade import Tier1Result, Tier1UnavailableError
from shieldcall.fusion.engine import FusedRisk
from shieldcall.linguistic.asr_bridge import ScheduledTranscriptASR
from shieldcall.linguistic.factorial import FactorialDiscourseModel
from shieldcall.runtime.shadow import (
    REASON_NO_SCORER,
    REASON_QUEUE_FULL,
    REASON_TIER1_ERROR,
    REASON_TIER1_UNAVAILABLE,
    ShadowConfig,
    ShadowObserver,
    map_speaker,
)


def _sine(seconds, sr=8000):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 170 * t)).astype(np.float32)


def _speech_audio(seconds, sr=8000):
    """Syllable-burst audio the VAD accepts as speech.

    A steady sine adapts the VAD threshold up to its own
    energy, so almost no frame counts as speech and no
    2 s window ever completes. Gated harmonic bursts
    with quiet gaps keep the adaptive threshold below
    the burst energy, so most frames are speech.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    gate = (np.sin(2 * np.pi * 4 * t) > -0.2).astype(np.float32)
    vowel = (
        0.25 * np.sin(2 * np.pi * 150 * t)
        + 0.15 * np.sin(2 * np.pi * 300 * t)
        + 0.08 * np.sin(2 * np.pi * 450 * t)
    )
    audio = vowel * gate + 0.02 * rng.standard_normal(len(t))
    return np.clip(audio, -0.9, 0.9).astype(np.float32)


class _FakeScorer:
    """Tier 1 stand-in returning a fixed score, counting calls."""

    def __init__(self, score=0.75):
        self.score = score
        self.calls = 0

    def score_window(self, samples, sample_rate):
        self.calls += 1
        return Tier1Result(
            score=self.score,
            llr=None,
            calibrated=False,
            window_sec=float(len(samples) / sample_rate),
            model_sha256="fake-sha",
            backbone="fake",
            layers=(1, 2, 3, 4),
        )


class _GatedScorer(_FakeScorer):
    """Blocks in score_window until released, proving async scoring."""

    def __init__(self, score=0.75):
        super().__init__(score)
        self.entered = threading.Event()
        self.release = threading.Event()

    def score_window(self, samples, sample_rate):
        self.entered.set()
        self.release.wait(timeout=10.0)
        return super().score_window(samples, sample_rate)


class _UnavailableScorer:
    def score_window(self, samples, sample_rate):
        raise Tier1UnavailableError("backbone is not in the local cache")


class _ExplodingScorer:
    def score_window(self, samples, sample_rate):
        raise RuntimeError("boom")


def _feed_window(observer, n_hops=10, hop_samples=80, llr=1.0, t0=0.0):
    """Feed one 0.1 s window of speech hops; return the record."""
    record = None
    for i in range(n_hops):
        hop = np.full(hop_samples, 0.1, dtype=np.float32)
        record = observer.observe_frame(
            hop, True, t0 + i * 0.01, llr, False, None
        )
    return record


# --------------------------------------------------------------- speaker map


def test_map_speaker():
    assert map_speaker("remote") == "far"
    assert map_speaker("far_end") == "far"
    assert map_speaker("local") == "near"
    assert map_speaker("user") == "near"
    assert map_speaker("unknown") == "far"
    assert map_speaker("not-a-speaker") == "far"


def test_shadow_config_validation():
    with pytest.raises(ValueError):
        ShadowConfig(sample_rate=0)
    with pytest.raises(ValueError):
        ShadowConfig(requested_action_threshold=1.5)
    with pytest.raises(ValueError):
        ShadowConfig(queue_size=0)


# ------------------------------------------------------------ shadow disabled


def test_shadow_disabled_leaves_pipeline_unchanged():
    pipe = ShieldCallPipeline(config=PipelineConfig(shadow_enabled=False))
    try:
        assert pipe.shadow is None
        assert pipe.shadow_windows == []
        assert pipe.last_factorial is None
        risks = [ev.risk for ev in pipe.stream(_sine(3.0), 8000) if ev.risk is not None]
        assert len(risks) > 5
    finally:
        pipe.close()


def test_shadow_on_does_not_change_cscf_verdicts():
    audio = _sine(3.0)
    off = ShieldCallPipeline(config=PipelineConfig(shadow_enabled=False))
    on = ShieldCallPipeline(config=PipelineConfig(shadow_enabled=True))
    try:
        off_risks = [ev.risk.risk_score for ev in off.stream(audio, 8000) if ev.risk is not None]
        on_risks = [ev.risk.risk_score for ev in on.stream(audio, 8000) if ev.risk is not None]
    finally:
        off.close()
        on.close()
    assert off_risks == on_risks
    assert len(on_risks) > 5


# ------------------------------------------------------- factorial beside SDTG


def test_factorial_state_matches_standalone_model():
    pipe = ShieldCallPipeline()
    try:
        turns = [
            "Hello, this is your bank calling",
            "There is unusual activity on your account",
            "Buy gift cards immediately and read me the code",
        ]
        for i, text in enumerate(turns):
            pipe.push_transcript(text, float(i))
        standalone = FactorialDiscourseModel()
        expected = None
        for i, text in enumerate(turns):
            expected = standalone.update(text, float(i), "far")
        actual = pipe.last_factorial
        assert actual is not None
        assert actual.action == expected.action
        assert actual.fraud_score == pytest.approx(expected.fraud_score)
        assert actual.stage == expected.stage
        assert actual.hazard == pytest.approx(expected.hazard)
        assert len(pipe.shadow.factorial_states) == 3
    finally:
        pipe.close()


def test_factorial_rides_on_pipeline_events():
    schedule = [
        (0.5, "Hello from your bank"),
        (1.5, "Buy gift cards immediately, do not tell anyone"),
    ]
    pipe = ShieldCallPipeline(asr=ScheduledTranscriptASR(schedule))
    try:
        states = [ev.factorial for ev in pipe.stream(_sine(3.0), 8000) if ev.factorial is not None]
        assert states
        assert states[-1].action == "payment_or_gift_card"
        assert states[-1].fraud_score > 0.5
    finally:
        pipe.close()


# ------------------------------------------- exact form replaces the SDTG proxy


def test_requested_action_uses_factorial_not_sdtg_stage():
    observer = ShadowObserver(ShadowConfig(window_sec=0.1, sample_rate=8000, seed=7))
    try:
        # A harmful request, reported with an early SDTG stage:
        # the factorial exact form must still latch.
        observer.observe_turn(
            "Buy gift cards immediately", 0.0, "far", sdtg_stage="BENIGN"
        )
        record = _feed_window(observer)
        assert record.requested_action_latched is True
        assert "requested_action" in record.decision.reasons
        assert record.sdtg_stage == "BENIGN"
    finally:
        observer.close()


def test_sdtg_payment_stage_without_harmful_action_does_not_latch():
    observer = ShadowObserver(ShadowConfig(window_sec=0.1, sample_rate=8000, seed=7))
    try:
        # A benign turn reported at the PAYMENT stage: the old
        # SDTG proxy would have latched, the exact form must not.
        observer.observe_turn("Hello there", 0.0, "far", sdtg_stage="PAYMENT")
        record = _feed_window(observer, llr=-1.0)
        assert record.requested_action_latched is False
        assert "requested_action" not in record.decision.reasons
        assert record.sdtg_stage == "PAYMENT"
    finally:
        observer.close()


def test_local_turn_never_counts_as_requested_action():
    observer = ShadowObserver(ShadowConfig(window_sec=0.1, sample_rate=8000, seed=7))
    try:
        observer.observe_turn(
            "Buy gift cards immediately", 0.0, "local", sdtg_stage="BENIGN"
        )
        assert observer.last_factorial.action == "none"
        record = _feed_window(observer, llr=-1.0)
        assert record.requested_action_latched is False
    finally:
        observer.close()


def test_unknown_speaker_defaults_to_far_end():
    observer = ShadowObserver(ShadowConfig(window_sec=0.1, sample_rate=8000, seed=7))
    try:
        observer.observe_turn(
            "Buy gift cards immediately", 0.0, "unknown", sdtg_stage="BENIGN"
        )
        record = _feed_window(observer)
        assert record.requested_action_latched is True
    finally:
        observer.close()


# ------------------------------------------------------------- window records


def test_window_record_carries_llr_and_cscf_snapshot():
    observer = ShadowObserver(ShadowConfig(window_sec=0.1, sample_rate=8000, seed=7))
    try:
        risk = FusedRisk(
            timestamp_sec=0.05,
            risk_score=0.41,
            tier="SUSPICIOUS",
            acoustic_synth_prob=0.5,
            linguistic_fraud_prob=0.2,
            escalation_factor=1.0,
        )
        record = None
        for i in range(10):
            hop = np.full(80, 0.1, dtype=np.float32)
            record = observer.observe_frame(hop, True, i * 0.01, 1.0, False, risk)
        assert record is not None
        assert record.tier0_llr == pytest.approx(1.0)
        assert record.cscf_risk_score == pytest.approx(0.41)
        assert record.cscf_tier == "SUSPICIOUS"
        assert "tier0_llr" in record.decision.reasons
        assert "first_window" in record.decision.reasons
        assert record.tier1_unavailable_reason == REASON_NO_SCORER
    finally:
        observer.close()


def test_non_speech_does_not_fill_windows():
    observer = ShadowObserver(ShadowConfig(window_sec=0.1, sample_rate=8000, seed=7))
    try:
        for i in range(50):
            hop = np.zeros(80, dtype=np.float32)
            record = observer.observe_frame(hop, False, i * 0.01, 1.0, False, None)
            assert record is None
        assert observer.windows == []
    finally:
        observer.close()


def test_changepoint_latch_feeds_trigger():
    observer = ShadowObserver(ShadowConfig(window_sec=0.1, sample_rate=8000, seed=7))
    try:
        record = None
        for i in range(10):
            hop = np.full(80, 0.1, dtype=np.float32)
            record = observer.observe_frame(hop, True, i * 0.01, -1.0, i == 3, None)
        assert record.changepoint_latched is True
        assert "changepoint" in record.decision.reasons
    finally:
        observer.close()


# ------------------------------------------------------------ background Tier 1


def test_triggered_windows_scored_on_background_thread():
    pipe = ShieldCallPipeline()
    scorer = _GatedScorer()
    pipe.shadow.set_tier1_scorer(scorer)
    try:
        events = list(pipe.stream(_speech_audio(6.0), 8000))
        # The stream returned while Tier 1 was still gated:
        # scoring never delays the call thread.
        assert scorer.entered.wait(timeout=5.0)
        windows = pipe.shadow_windows
        assert len(windows) >= 1
        assert windows[0].tier1_score is None
        scorer.release.set()
        assert pipe.drain_shadow(timeout_sec=5.0)
        assert windows[0].tier1_score == pytest.approx(0.75)
        assert windows[0].tier1_model_sha256 == "fake-sha"
        assert windows[0].tier1_calibrated is False
        assert windows[0].tier1_unavailable_reason is None
    finally:
        scorer.release.set()
        pipe.close()


def test_no_scorer_records_reason_and_invents_nothing():
    pipe = ShieldCallPipeline()
    try:
        list(pipe.stream(_speech_audio(6.0), 8000))
        assert pipe.drain_shadow(timeout_sec=5.0)
        windows = pipe.shadow_windows
        assert len(windows) >= 1
        triggered = [w for w in windows if w.decision.triggered]
        assert triggered
        for window in triggered:
            assert window.tier1_score is None
            assert window.tier1_unavailable_reason == REASON_NO_SCORER
    finally:
        pipe.close()


def test_unavailable_scorer_latches_outage_and_factory_runs_once():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        raise Tier1UnavailableError("no backbone in cache")

    observer = ShadowObserver(
        ShadowConfig(window_sec=0.1, sample_rate=8000, seed=7), scorer_factory=factory
    )
    try:
        first = _feed_window(observer)
        second = _feed_window(observer, t0=0.2)
        assert observer.drain_shadow(timeout_sec=5.0)
        assert first.tier1_unavailable_reason == REASON_TIER1_UNAVAILABLE
        assert second.tier1_unavailable_reason == REASON_TIER1_UNAVAILABLE
        assert first.tier1_score is None and second.tier1_score is None
        assert calls["n"] == 1
    finally:
        observer.close()


def test_scorer_error_never_raises_into_call_path():
    pipe = ShieldCallPipeline()
    pipe.shadow.set_tier1_scorer(_ExplodingScorer())
    try:
        # Streaming must complete without the scorer's
        # exception reaching the caller.
        events = list(pipe.stream(_speech_audio(6.0), 8000))
        assert events
        assert pipe.drain_shadow(timeout_sec=5.0)
        triggered = [w for w in pipe.shadow_windows if w.decision.triggered]
        assert triggered
        assert all(w.tier1_unavailable_reason == REASON_TIER1_ERROR for w in triggered)
    finally:
        pipe.close()


def test_missing_model_file_latches_outage():
    pipe = ShieldCallPipeline(
        config=PipelineConfig(shadow_tier1_model_path="/nonexistent/model8.npz")
    )
    try:
        list(pipe.stream(_speech_audio(6.0), 8000))
        assert pipe.drain_shadow(timeout_sec=5.0)
        triggered = [w for w in pipe.shadow_windows if w.decision.triggered]
        assert triggered
        assert all(
            w.tier1_unavailable_reason == REASON_TIER1_UNAVAILABLE for w in triggered
        )
    finally:
        pipe.close()


# ------------------------------------------------------------- reset and queue


def test_reset_drops_stale_queued_windows():
    observer = ShadowObserver(ShadowConfig(window_sec=0.1, sample_rate=8000, seed=7))
    scorer = _GatedScorer()
    observer.set_tier1_scorer(scorer)
    try:
        stale = _feed_window(observer)
        assert scorer.entered.wait(timeout=5.0)
        observer.reset()
        assert observer.windows == []
        fresh = _feed_window(observer, t0=0.2)
        scorer.release.set()
        assert observer.drain_shadow(timeout_sec=5.0)
        # The stale window was never scored into the new call.
        assert stale.tier1_score is None
        assert fresh.tier1_score == pytest.approx(0.75)
        assert observer.windows == [fresh]
    finally:
        scorer.release.set()
        observer.close()


def test_full_queue_drops_oldest_unstarted_window():
    observer = ShadowObserver(
        ShadowConfig(window_sec=0.1, sample_rate=8000, seed=7, queue_size=1)
    )
    scorer = _GatedScorer()
    observer.set_tier1_scorer(scorer)
    try:
        first = _feed_window(observer)
        assert scorer.entered.wait(timeout=5.0)
        second = _feed_window(observer, t0=0.2)
        third = _feed_window(observer, t0=0.4)
        assert observer.dropped_windows == 1
        assert second.tier1_dropped is True
        assert second.tier1_unavailable_reason == REASON_QUEUE_FULL
        scorer.release.set()
        assert observer.drain_shadow(timeout_sec=5.0)
        assert first.tier1_score == pytest.approx(0.75)
        assert third.tier1_score == pytest.approx(0.75)
        assert second.tier1_score is None
    finally:
        scorer.release.set()
        observer.close()


def test_pipeline_reset_starts_shadow_fresh():
    pipe = ShieldCallPipeline(config=PipelineConfig(shadow_audit_seed=7))
    pipe.shadow.set_tier1_scorer(_FakeScorer())
    try:
        list(pipe.stream(_speech_audio(4.0), 8000))
        assert pipe.drain_shadow(timeout_sec=5.0)
        first_call = pipe.shadow_windows
        assert first_call
        pipe.reset()
        assert pipe.shadow_windows == []
        assert pipe.last_factorial is None
        list(pipe.stream(_speech_audio(4.0), 8000))
        assert pipe.drain_shadow(timeout_sec=5.0)
        second_call = pipe.shadow_windows
        assert second_call
        assert second_call[0].window_index == 0
        assert second_call[0].decision.reasons == first_call[0].decision.reasons
    finally:
        pipe.close()
