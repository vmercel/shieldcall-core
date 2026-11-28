"""Time-to-warning measurement (ADR-019, TDD v2.0 section 10.4).

The harness measures, on synthetic calls with known
evidence, how long the pipeline itself takes from the
end of discriminative evidence to a HIGH_RISK verdict,
reported per stage. ASR provider lag, network time, and
render time are absent in these runs and are reported
as absent, never as zero. No licensed ASVspoof data is
touched, no model is downloaded, and no live call is
scored. Ground truth is synthetic audio and transcript
turns with known content.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from shieldcall import ShieldCallPipeline, PipelineConfig
from shieldcall.acoustic.cascade import Tier1Result
from shieldcall.eval.time_to_warning import (
    Turn,
    WarningMeasurement,
    measure_time_to_warning,
    run_repeated,
    summarize_runs,
)
from shieldcall.linguistic.asr_bridge import ScheduledTranscriptASR
from shieldcall.runtime.shadow import ShadowConfig, ShadowObserver


def _speech_audio(seconds, sr=8000):
    """Gated-burst audio the VAD accepts as speech (ADR-018 pattern)."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    gate = (np.sin(2 * np.pi * 4 * t) > -0.2).astype(np.float32)
    vowel = (
        0.25 * np.sin(2 * np.pi * 150 * t)
        + 0.15 * np.sin(2 * np.pi * 300 * t)
        + 0.08 * np.sin(2 * np.pi * 450 * t)
    )
    return np.clip(vowel * gate + 0.02 * rng.standard_normal(len(t)), -0.9, 0.9).astype(np.float32)


def _scam_pipeline():
    schedule = [
        (0.5, "Hello from your bank"),
        (2.5, "Unusual activity detected, verify social security number and buy gift cards immediately, do not tell anyone"),
    ]
    return ShieldCallPipeline(
        config=PipelineConfig(shadow_audit_seed=0),
        asr=ScheduledTranscriptASR(schedule),
    )


class _FakeScorer:
    def score_window(self, samples, sample_rate):
        return Tier1Result(
            score=0.75,
            llr=None,
            calibrated=False,
            window_sec=float(len(samples) / sample_rate),
            model_sha256="fake-sha",
            backbone="fake",
            layers=(1, 2, 3, 4),
        )


class _StubPipeline:
    """Minimal pipeline stand-in with canned verdicts and a fake clock."""

    def __init__(self, events_by_chunk):
        self.events_by_chunk = events_by_chunk
        self.pushes = 0
        self.transcripts = []

    def reset(self):
        self.pushes = 0

    def push_transcript(self, text, timestamp_sec):
        self.transcripts.append((text, timestamp_sec))
        return SimpleNamespace()

    def push_audio(self, samples, sample_rate):
        events = self.events_by_chunk[self.pushes]
        self.pushes += 1
        return events

    def drain_shadow(self, timeout_sec=5.0):
        return True

    @property
    def shadow_windows(self):
        return []


def _event(tier, timestamp_sec):
    return SimpleNamespace(
        risk=SimpleNamespace(tier=tier),
        frame=SimpleNamespace(timestamp_sec=timestamp_sec),
    )


class _FakeClock:
    """Each call advances by a fixed step, so durations are exact."""

    def __init__(self, step=0.05):
        self.value = 0.0
        self.step = step

    def __call__(self):
        self.value += self.step
        return self.value


def _measurement(**overrides):
    base = dict(
        evidence_end_sec=1.5,
        audio_duration_sec=6.0,
        chunk_ms=100.0,
        n_chunks=60,
        warned=True,
        warned_before_evidence_end=False,
        first_high_risk_audio_sec=1.5,
        accumulation_audio_sec=0.0,
        provisional_cue=True,
        first_provisional_audio_sec=1.5,
        provisional_accumulation_audio_sec=0.0,
        deciding_chunk_index=15,
        deciding_chunk_processing_sec=0.05,
        chunk_processing_sec=[0.05] * 60,
        tier1_latency_sec=[],
        asr_lag_sec=0.1,
        render_sec=0.1,
        network_sec=0.1,
    )
    base.update(overrides)
    return WarningMeasurement(**base)


def test_scam_call_warns_within_accumulation_budget():
    audio = _speech_audio(6.0)
    measurement = measure_time_to_warning(
        _scam_pipeline(), audio, 8000, evidence_end_sec=2.5
    )
    assert measurement.warned
    assert not measurement.warned_before_evidence_end
    assert measurement.first_high_risk_audio_sec is not None
    assert measurement.first_high_risk_audio_sec >= 2.5 - 1e-9
    assert measurement.accumulation_audio_sec is not None
    assert 0.0 <= measurement.accumulation_audio_sec <= 0.7
    assert measurement.accumulation_within_budget is True
    assert measurement.deciding_chunk_processing_sec is not None
    assert measurement.deciding_chunk_processing_sec > 0.0
    assert len(measurement.chunk_processing_sec) == measurement.n_chunks
    # Out-of-process stages are absent, never zero.
    assert measurement.asr_lag_sec is None
    assert measurement.render_sec is None
    assert measurement.network_sec is None
    assert measurement.measured_only


def test_benign_call_never_warns():
    audio = _speech_audio(4.0)
    schedule = [(0.5, "Hello, how is the weather today"), (2.0, "Let us meet for lunch tomorrow")]
    pipe = ShieldCallPipeline(
        config=PipelineConfig(shadow_audit_seed=0),
        asr=ScheduledTranscriptASR(schedule),
    )
    measurement = measure_time_to_warning(pipe, audio, 8000, evidence_end_sec=2.0)
    assert not measurement.warned
    assert measurement.accumulation_audio_sec is None
    assert measurement.first_high_risk_audio_sec is None
    assert measurement.estimated_live_warning_sec is None
    assert measurement.accumulation_within_budget is None


def test_warning_before_evidence_end_is_flagged_not_negative():
    audio = _speech_audio(6.0)
    measurement = measure_time_to_warning(
        _scam_pipeline(), audio, 8000, evidence_end_sec=3.5
    )
    assert measurement.warned_before_evidence_end
    assert measurement.warned
    assert measurement.accumulation_audio_sec == 0.0
    assert measurement.estimated_live_warning_sec is not None
    assert measurement.estimated_live_warning_sec >= 0.0


def test_provisional_cue_reported_separately():
    audio = _speech_audio(6.0)
    measurement = measure_time_to_warning(
        _scam_pipeline(), audio, 8000, evidence_end_sec=2.5
    )
    assert measurement.provisional_cue
    assert measurement.provisional_accumulation_audio_sec is not None
    assert measurement.provisional_accumulation_audio_sec <= (
        measurement.accumulation_audio_sec or 0.0
    ) + 1e-9


def test_fake_clock_chunk_times_are_deterministic():
    # Chunk 0: SAFE. Chunk 1 contains the evidence end
    # (1.5 s) and warns at its first frame.
    stub = _StubPipeline(
        [
            [_event("SAFE", 0.0)],
            [_event("HIGH_RISK", 1.5)],
            [_event("HIGH_RISK", 1.6)],
        ]
    )
    audio = np.zeros(2400, dtype=np.float32)  # 0.3 s at 8 kHz, 3 chunks of 100 ms
    measurement = measure_time_to_warning(
        stub,
        audio,
        8000,
        evidence_end_sec=0.15,
        turns=[Turn(0.05, "hello")],
        chunk_ms=100.0,
        clock=_FakeClock(step=0.05),
    )
    assert measurement.warned
    assert measurement.chunk_processing_sec == pytest.approx([0.05, 0.05, 0.05])
    assert measurement.turn_processing_sec == pytest.approx([0.05])
    assert measurement.deciding_chunk_index == 1
    assert measurement.deciding_chunk_processing_sec == pytest.approx(0.05)
    assert measurement.steady_chunk_processing_p95_sec == pytest.approx(0.05)
    assert stub.transcripts == [("hello", 0.05)]


def test_tier1_latency_only_from_scored_windows():
    audio = _speech_audio(6.0)
    pipe = _scam_pipeline()
    pipe.shadow.set_tier1_scorer(_FakeScorer())
    measurement = measure_time_to_warning(pipe, audio, 8000, evidence_end_sec=2.5)
    assert measurement.shadow_windows_total > 0
    assert measurement.tier1_scored_windows > 0
    assert len(measurement.tier1_latency_sec) == measurement.tier1_scored_windows
    assert all(latency >= 0.0 for latency in measurement.tier1_latency_sec)
    assert measurement.tier1_latency_p95_sec is not None
    assert measurement.shadow_drained


def test_tier1_latency_absent_without_scorer():
    audio = _speech_audio(6.0)
    measurement = measure_time_to_warning(
        _scam_pipeline(), audio, 8000, evidence_end_sec=2.5
    )
    assert measurement.shadow_windows_total > 0
    assert measurement.tier1_scored_windows == 0
    assert measurement.tier1_latency_sec == []
    assert measurement.tier1_latency_p95_sec is None
    assert measurement.tier1_within_budget is None


def test_shadow_record_stamps():
    observer = ShadowObserver(ShadowConfig(seed=0), tier1_scorer=_FakeScorer())
    hop = np.full(80, 0.1, dtype=np.float32)
    record = None
    for i in range(250):
        record = observer.observe_frame(hop, True, i * 0.01, 1.0, False, None)
        if record is not None:
            break
    assert record is not None
    assert record.decision_monotonic_sec is not None
    assert observer.drain_shadow(5.0)
    assert record.tier1_score is not None
    assert record.tier1_monotonic_sec is not None
    assert record.tier1_monotonic_sec >= record.decision_monotonic_sec
    observer.close()


def test_shadow_record_tier1_stamp_absent_without_scorer():
    observer = ShadowObserver(ShadowConfig(seed=0))
    hop = np.full(80, 0.1, dtype=np.float32)
    record = None
    for i in range(250):
        record = observer.observe_frame(hop, True, i * 0.01, 1.0, False, None)
        if record is not None:
            break
    assert record is not None
    assert record.decision_monotonic_sec is not None
    assert record.tier1_monotonic_sec is None
    assert record.tier1_unavailable_reason is not None
    observer.close()


def test_percentile_is_nearest_rank():
    measurement = _measurement(chunk_processing_sec=[0.1, 0.2, 0.3, 0.4])
    assert measurement.chunk_processing_p50_sec == 0.2
    assert measurement.chunk_processing_p95_sec == 0.4
    assert measurement.steady_chunk_processing_p95_sec == 0.4


def test_budget_verdict_met_only_with_full_stages():
    summary = summarize_runs([_measurement(), _measurement()])
    assert summary.budget_verdict == "met"
    assert summary.warned_runs == 2
    assert summary.estimated_live_p95_sec == pytest.approx(0.35)


def test_budget_verdict_incomplete_when_stage_unmeasured():
    summary = summarize_runs([_measurement(asr_lag_sec=None)])
    assert summary.budget_verdict == "incomplete"
    assert any("unmeasured" in note for note in summary.budget_notes)


def test_budget_verdict_not_met_when_estimate_exceeds_two_seconds():
    summary = summarize_runs(
        [_measurement(accumulation_audio_sec=2.5, asr_lag_sec=0.1, render_sec=0.1, network_sec=0.1)]
    )
    assert summary.budget_verdict == "not met"


def test_summary_without_warnings_is_incomplete():
    summary = summarize_runs([_measurement(warned=False, accumulation_audio_sec=None, first_high_risk_audio_sec=None, deciding_chunk_processing_sec=None)])
    assert summary.budget_verdict == "incomplete"
    assert summary.warned_runs == 0


def test_run_repeated_summarises_stub_runs():
    def factory():
        return _StubPipeline(
            [[_event("SAFE", 0.0)], [_event("HIGH_RISK", 1.5)], [_event("HIGH_RISK", 1.6)]]
        )

    audio = np.zeros(2400, dtype=np.float32)
    measurements, summary = run_repeated(
        factory,
        audio,
        8000,
        2,
        evidence_end_sec=0.15,
        clock=_FakeClock(step=0.05),
        asr_lag_sec=0.1,
        render_sec=0.1,
        network_sec=0.1,
    )
    assert len(measurements) == 2
    assert summary.warned_runs == 2
    assert summary.budget_verdict == "met"


def test_invalid_inputs_rejected():
    pipe = _scam_pipeline()
    with pytest.raises(ValueError):
        measure_time_to_warning(pipe, np.zeros(0, dtype=np.float32), 8000, evidence_end_sec=0.0)
    with pytest.raises(ValueError):
        measure_time_to_warning(pipe, np.zeros(8000, dtype=np.float32), 8000, evidence_end_sec=2.0)
    with pytest.raises(ValueError):
        measure_time_to_warning(pipe, np.zeros(8000, dtype=np.float32), 8000, evidence_end_sec=0.5, chunk_ms=0.0)
    with pytest.raises(ValueError):
        measure_time_to_warning(pipe, np.zeros(8000, dtype=np.float32), 8000, evidence_end_sec=0.5, asr_lag_sec=-0.1)


def test_measurement_serialises_to_json():
    import json

    measurement = _measurement()
    summary = summarize_runs([measurement])
    assert json.loads(json.dumps(measurement.to_dict()))["warned"] is True
    assert json.loads(json.dumps(summary.to_dict()))["budget_verdict"] == "met"
