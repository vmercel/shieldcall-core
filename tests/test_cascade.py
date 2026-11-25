"""Risk-triggered acoustic cascade (ADR-016, TDD v2.0 section 6.5).

Ground truth is synthetic data with known distributions, so every
expected value below is derived, not asserted from the code under
test. Nothing here touches licensed ASVspoof data, and no model
is downloaded: the Tier 1 scorer is tested through an injected
embedding function plus a synthetic model file.
"""

import hashlib
import json
import math
import random

import numpy as np
import pytest

from shieldcall.acoustic.cascade import (
    CascadeConfig,
    CascadeRow,
    CascadeTrigger,
    Phase8Tier1Scorer,
    Tier1UnavailableError,
    WindowBuilder,
    evaluate_cascade,
    prob_to_llr,
)


# ---------------------------------------------------------------- prob_to_llr


def test_prob_to_llr_neutral_and_monotone():
    assert prob_to_llr(0.5) == pytest.approx(0.0)
    assert prob_to_llr(0.9) == pytest.approx(math.log(9.0))
    assert prob_to_llr(0.1) == pytest.approx(-math.log(9.0))
    assert prob_to_llr(0.9) > prob_to_llr(0.5) > prob_to_llr(0.1)


def test_prob_to_llr_clips_confident_inputs():
    assert math.isfinite(prob_to_llr(0.0))
    assert math.isfinite(prob_to_llr(1.0))
    assert prob_to_llr(0.0) < prob_to_llr(1.0)


def test_prob_to_llr_rejects_non_finite():
    with pytest.raises(ValueError):
        prob_to_llr(float("nan"))
    with pytest.raises(ValueError):
        prob_to_llr(float("inf"))


# ---------------------------------------------------------------- windows


def _builder(window_sec=0.1, sample_rate=8000):
    return WindowBuilder(CascadeConfig(window_sec=window_sec, sample_rate=sample_rate))


def test_window_builder_emits_at_two_seconds_of_speech():
    builder = _builder()
    frame = np.zeros(80, dtype=np.float32)
    result = None
    for i in range(10):
        result = builder.add(frame, True, t_sec=i * 0.01)
    assert result is not None
    assert result.samples.shape == (800,)
    assert result.window_index == 0
    assert result.start_sec == pytest.approx(0.0)
    assert result.end_sec == pytest.approx(0.09)
    assert builder.windows_completed == 1


def test_window_builder_ignores_non_speech():
    builder = _builder()
    speech = np.ones(80, dtype=np.float32)
    silence = np.zeros(80, dtype=np.float32)
    assert builder.add(silence, False, t_sec=0.0) is None
    assert builder.pending_samples == 0
    for i in range(9):
        assert builder.add(speech, True, t_sec=0.01 * i) is None
    assert builder.pending_samples == 720
    result = builder.add(speech, True, t_sec=0.09)
    assert result is not None
    # Silence contributed no samples: the window is all speech.
    assert float(np.mean(result.samples)) == pytest.approx(1.0)


def test_window_builder_discards_partial_on_reset():
    builder = _builder()
    builder.add(np.zeros(400, dtype=np.float32), True, t_sec=0.0)
    assert builder.pending_samples == 400
    assert builder.reset() == 400
    assert builder.dropped_partial_samples == 400
    assert builder.pending_samples == 0


def test_window_builder_rejects_bad_frames():
    builder = _builder()
    with pytest.raises(ValueError):
        builder.add(np.zeros((2, 80), dtype=np.float32), True, t_sec=0.0)
    with pytest.raises(ValueError):
        builder.add(np.array([np.nan], dtype=np.float32), True, t_sec=0.0)
    with pytest.raises(ValueError):
        builder.add(np.zeros(80, dtype=np.float32), True, t_sec=float("nan"))


# ---------------------------------------------------------------- triggers


def test_first_window_always_triggers():
    trigger = CascadeTrigger(CascadeConfig(audit_probability=0.0, seed=7))
    decision = trigger.decide(-5.0)
    assert decision.triggered
    assert decision.reasons == {"first_window"}
    assert decision.window_index == 0


def test_tier0_llr_trigger_is_strict():
    trigger = CascadeTrigger(CascadeConfig(tau0=0.0, audit_probability=0.0, seed=7))
    trigger.decide(-5.0)  # first window, consumed
    assert not trigger.decide(0.0).triggered  # equal to tau0: no trigger
    assert trigger.decide(0.001).triggered
    assert trigger.decide(0.001).reasons == {"tier0_llr"}


def test_changepoint_and_requested_action_trigger_alone():
    trigger = CascadeTrigger(CascadeConfig(audit_probability=0.0, seed=7))
    trigger.decide(-5.0)
    assert trigger.decide(-5.0, changepoint=True).reasons == {"changepoint"}
    assert trigger.decide(-5.0, requested_action=True).reasons == {"requested_action"}


def test_all_reasons_recorded_together():
    trigger = CascadeTrigger(CascadeConfig(audit_probability=1.0, seed=7))
    trigger.decide(-5.0)
    decision = trigger.decide(5.0, changepoint=True, requested_action=True)
    assert decision.reasons == {
        "tier0_llr",
        "changepoint",
        "requested_action",
        "random_audit",
    }
    assert decision.audit_triggered


def test_audit_never_fires_at_zero_probability():
    trigger = CascadeTrigger(CascadeConfig(audit_probability=0.0, seed=7))
    for _ in range(50):
        decision = trigger.decide(-5.0)
        assert "random_audit" not in decision.reasons


def test_audit_first_window_is_not_an_audit_reason():
    trigger = CascadeTrigger(CascadeConfig(audit_probability=1.0, seed=7))
    decision = trigger.decide(-5.0)
    assert "random_audit" not in decision.reasons
    assert decision.reasons == {"first_window"}


def test_audit_draws_are_seeded_and_reproducible():
    def draws(seed):
        trigger = CascadeTrigger(CascadeConfig(audit_probability=0.02, seed=seed))
        return [trigger.decide(-5.0).audit_draw for _ in range(200)]

    first, second = draws(42), draws(42)
    assert first == second
    rng = random.Random(42)
    assert first == [rng.random() for _ in range(200)]
    # Seeded rate is near the configured 2%: 200 draws, expect a few.
    trigger = CascadeTrigger(CascadeConfig(audit_probability=0.02, seed=42))
    audits = sum(
        1 for _ in range(200) if trigger.decide(-5.0).audit_triggered
    )
    assert 0 <= audits <= 12


def test_trigger_reset_reproduces_audit_stream():
    trigger = CascadeTrigger(CascadeConfig(audit_probability=0.5, seed=3))
    before = [trigger.decide(-5.0).audit_draw for _ in range(20)]
    trigger.reset()
    after = [trigger.decide(-5.0).audit_draw for _ in range(20)]
    assert before == after
    assert trigger.window_index == 20


def test_requested_action_stage_proxy():
    trigger = CascadeTrigger()
    assert trigger.is_requested_action("HARVEST")
    assert trigger.is_requested_action("PAYMENT")
    assert not trigger.is_requested_action("GREETING")
    assert not trigger.is_requested_action("BENIGN")
    custom = CascadeTrigger(CascadeConfig(requested_action_stages=("PAYMENT",)))
    assert not custom.is_requested_action("HARVEST")
    assert custom.is_requested_action("PAYMENT")


def test_trigger_config_validation():
    with pytest.raises(ValueError):
        CascadeConfig(audit_probability=1.5)
    with pytest.raises(ValueError):
        CascadeConfig(window_sec=0.0)
    with pytest.raises(ValueError):
        CascadeConfig(tau0=float("nan"))
    with pytest.raises(ValueError):
        CascadeTrigger().decide(float("nan"))


# ---------------------------------------------------------------- Tier 1 scorer


def _write_fake_model(path, kind="lr"):
    payload = {
        "m_kind": np.array(kind),
        "m_layers": np.array([1, 2, 3, 4]),
        "m_mu": np.zeros(6144, dtype=np.float32),
        "m_sd": np.ones(6144, dtype=np.float32),
        "m_w": np.full(6144, 0.001, dtype=np.float32),
        "m_b": np.array(0.5, dtype=np.float32),
        "meta": np.array(json.dumps({"backbone": "w2v2-base"})),
    }
    np.savez(path, **payload)
    return path


def _row_embed(value):
    def embed(samples16k):
        rows = np.empty((5, 1536), dtype=np.float64)
        for i in range(5):
            rows[i, :] = float(i) * value
        return rows

    return embed


def test_phase8_head_arithmetic_and_sign(tmp_path):
    model = _write_fake_model(tmp_path / "model8.npz")
    scorer = Phase8Tier1Scorer(str(model), embed_fn=_row_embed(1.0))
    result = scorer.score_window(np.zeros(16000, dtype=np.float32), 8000)
    # Flat features are rows 1..4: values 1, 2, 3, 4 over 1536 each.
    # Bona fide logit = 1536 * (1 + 2 + 3 + 4) * 0.001 + 0.5 = 15.86.
    # Spoof score is the negation.
    assert result.score == pytest.approx(-15.86)
    assert result.llr is None
    assert result.calibrated is False
    assert result.layers == (1, 2, 3, 4)
    assert result.model_sha256 == hashlib.sha256(model.read_bytes()).hexdigest()


def test_phase8_score_falls_as_bona_fide_evidence_rises(tmp_path):
    model = _write_fake_model(tmp_path / "model8.npz")
    zero = Phase8Tier1Scorer(str(model), embed_fn=_row_embed(0.0)).score_window(
        np.zeros(16000, dtype=np.float32), 8000
    )
    one = Phase8Tier1Scorer(str(model), embed_fn=_row_embed(1.0)).score_window(
        np.zeros(16000, dtype=np.float32), 8000
    )
    # Positive head weights mean larger features are bona fide evidence,
    # so the spoof score must fall.
    assert one.score < zero.score


def test_phase8_calibrator_is_applied_only_when_supplied(tmp_path):
    model = _write_fake_model(tmp_path / "model8.npz")

    class FakeCalibrator:
        def __init__(self):
            self.seen = []

        def llr(self, cell, score):
            self.seen.append((cell, score))
            return score * 2.0 + 1.0

    calibrator = FakeCalibrator()
    scorer = Phase8Tier1Scorer(
        str(model), embed_fn=_row_embed(0.0), calibrator=calibrator, cell="pstn"
    )
    result = scorer.score_window(np.zeros(16000, dtype=np.float32), 8000)
    assert result.calibrated is True
    assert result.llr == pytest.approx(result.score * 2.0 + 1.0)
    assert calibrator.seen == [("pstn", result.score)]


def test_phase8_resamples_8k_to_16k(tmp_path):
    model = _write_fake_model(tmp_path / "model8.npz")
    seen = {}

    def embed(samples16k):
        seen["n"] = int(samples16k.shape[0])
        return np.zeros((5, 1536), dtype=np.float64)

    scorer = Phase8Tier1Scorer(str(model), embed_fn=embed)
    scorer.score_window(np.zeros(16000, dtype=np.float32), 8000)
    assert seen["n"] == 32000
    scorer.score_window(np.zeros(16000, dtype=np.float32), 16000)
    assert seen["n"] == 16000


def test_phase8_missing_model_fails_closed(tmp_path):
    with pytest.raises(Tier1UnavailableError):
        Phase8Tier1Scorer(str(tmp_path / "no-such-model.npz"))


def test_phase8_unsupported_head_kind_fails_closed(tmp_path):
    model = _write_fake_model(tmp_path / "model8.npz", kind="mlp")
    with pytest.raises(Tier1UnavailableError):
        Phase8Tier1Scorer(str(model))


def test_phase8_missing_backbone_fails_closed(tmp_path):
    model = _write_fake_model(tmp_path / "model8.npz")
    scorer = Phase8Tier1Scorer(
        str(model),
        backbone_id="nonexistent/backbone-xyz",
        cache_dir=str(tmp_path / "empty-cache"),
    )
    with pytest.raises(Tier1UnavailableError):
        scorer.score_window(np.zeros(16000, dtype=np.float32), 8000)


def test_phase8_rejects_bad_windows(tmp_path):
    model = _write_fake_model(tmp_path / "model8.npz")
    scorer = Phase8Tier1Scorer(str(model), embed_fn=_row_embed(0.0))
    with pytest.raises(ValueError):
        scorer.score_window(np.zeros(16000, dtype=np.float32), 44100)
    with pytest.raises(ValueError):
        scorer.score_window(np.zeros(100, dtype=np.float32), 8000)
    with pytest.raises(ValueError):
        scorer.score_window(np.zeros(0, dtype=np.float32), 8000)


def test_phase8_embedding_layout_mismatch_fails_closed(tmp_path):
    model = _write_fake_model(tmp_path / "model8.npz")
    scorer = Phase8Tier1Scorer(
        str(model), embed_fn=lambda x: np.zeros((3, 1536), dtype=np.float64)
    )
    with pytest.raises(Tier1UnavailableError):
        scorer.score_window(np.zeros(16000, dtype=np.float32), 8000)


# ---------------------------------------------------------------- evaluation


def _rows(n_bona, bona_triggered, n_spoof, spoof_triggered, spoof_scores):
    rows = [CascadeRow(0, i < bona_triggered, 0.0) for i in range(n_bona)]
    rows += [
        CascadeRow(1, i < spoof_triggered, spoof_scores[i])
        for i in range(n_spoof)
    ]
    return rows


def test_evaluate_cascade_hand_computed_case():
    # 10 bona fide (2 triggered), 10 spoof (8 triggered).
    # Triggered spoof scores: 6 above threshold, 2 below.
    # Untriggered spoof scores: both above (always-on would catch them).
    scores = [1.0] * 6 + [-1.0] * 2 + [1.0] * 2
    report = evaluate_cascade(_rows(10, 2, 10, 8, scores), 0.0)
    assert report.trigger_rate == pytest.approx(0.5)
    assert report.benign_trigger_rate == pytest.approx(0.2)
    assert report.fnr_trigger == pytest.approx(0.2)
    assert report.always_on_miss_rate == pytest.approx(0.2)
    assert report.cascade_miss_rate == pytest.approx(0.4)
    assert report.miss_gap == pytest.approx(0.2)
    assert report.expected_cost == pytest.approx(3.0)
    assert report.cost_ratio == pytest.approx(3.0)
    assert report.passes_kill_criteria is False


def test_evaluate_cascade_kill_criteria_pass():
    scores = [1.0] * 10
    report = evaluate_cascade(_rows(10, 1, 10, 10, scores), 0.0)
    assert report.benign_trigger_rate == pytest.approx(0.1)
    assert report.always_on_miss_rate == 0.0
    assert report.cascade_miss_rate == 0.0
    assert report.passes_kill_criteria is True


def test_evaluate_cascade_miss_gap_boundary():
    # Benign trigger rate exactly 0.15; miss gap exactly 0.01: passes.
    scores = [1.0] * 89 + [-1.0] * 10 + [1.0]
    rows = _rows(20, 3, 100, 99, scores)
    report = evaluate_cascade(rows, 0.0)
    assert report.benign_trigger_rate == pytest.approx(0.15)
    assert report.always_on_miss_rate == pytest.approx(0.10)
    assert report.cascade_miss_rate == pytest.approx(0.11)
    assert report.miss_gap == pytest.approx(0.01)
    assert report.passes_kill_criteria is True
    # One more missed spoof window pushes the gap to 0.02: fails.
    scores2 = [1.0] * 88 + [-1.0] * 10 + [1.0] * 2
    report2 = evaluate_cascade(_rows(20, 3, 100, 98, scores2), 0.0)
    assert report2.miss_gap == pytest.approx(0.02)
    assert report2.passes_kill_criteria is False


def test_evaluate_cascade_validation():
    with pytest.raises(ValueError):
        evaluate_cascade([], 0.0)
    with pytest.raises(ValueError):
        evaluate_cascade([CascadeRow(2, True, 1.0)], 0.0)
    with pytest.raises(ValueError):
        evaluate_cascade([CascadeRow(1, True, float("nan"))], 0.0)
    with pytest.raises(ValueError):
        evaluate_cascade([CascadeRow(1, True, 1.0)], 0.0, tier1_cost=0.0)


def test_evaluate_cascade_no_spoof_rows():
    report = evaluate_cascade(_rows(10, 1, 0, 0, []), 0.0)
    assert report.fnr_trigger == 0.0
    assert report.always_on_miss_rate == 0.0
    assert report.cascade_miss_rate == 0.0
    assert report.passes_kill_criteria is True


# ---------------------------------------------------------------- caller flow


def test_builder_and_trigger_caller_flow():
    builder = WindowBuilder(CascadeConfig(window_sec=0.1, sample_rate=8000))
    trigger = CascadeTrigger(CascadeConfig(audit_probability=0.0, seed=1))
    frame = np.zeros(80, dtype=np.float32)
    window_llrs = [-1.0] * 10 + [2.0] * 10
    decisions = []
    for i, llr in enumerate(window_llrs):
        window = builder.add(frame, True, t_sec=i * 0.01)
        if window is not None:
            mean_llr = sum(window_llrs[window.window_index * 10 : (window.window_index + 1) * 10]) / 10.0
            decisions.append(trigger.decide(mean_llr))
    assert len(decisions) == 2
    assert decisions[0].reasons == {"first_window"}
    assert decisions[1].reasons == {"tier0_llr"}
