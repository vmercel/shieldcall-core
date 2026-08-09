from shieldcall.eval.harness import (
    generate_synthetic_benchmark,
    evaluate_acoustic_channel,
    evaluate_adaptation_recovery,
    run_basic_latency_test,
)
from shieldcall.eval.metrics import equal_error_rate, auc_roc, summarize_scores
from shieldcall.audio.channel import CodecProfile
from shieldcall.acoustic.features import extract_frame_features
import numpy as np


def test_metrics_eer_auc():
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.15, 0.3, 0.7, 0.8, 0.9, 0.75]
    eer = equal_error_rate(labels, scores)
    assert 0 <= eer <= 0.5
    assert auc_roc(labels, scores) > 0.8


def test_summarize_scores_compat():
    assert summarize_scores([0.1, 0.2], [0.8, 0.9]) < 0.3


def test_generate_and_eval_acoustic():
    samples = generate_synthetic_benchmark(n_human=6, n_synth=6, seed=0)
    assert len(samples) == 12
    result = evaluate_acoustic_channel(samples, CodecProfile.NARROWBAND, seed=0)
    assert result.n_samples == 12
    assert 0 <= result.eer_estimate <= 1
    assert result.mean_latency_ms >= 0


def test_adaptation_recovery_metric():
    m = evaluate_adaptation_recovery(n_shots=5, seed=0)
    assert "gap_reduction" in m
    assert m["mean_gap_after"] <= m["mean_gap_before"] + 1e-6


def test_latency_helper():
    lat = run_basic_latency_test(lambda x: extract_frame_features(x, 8000), n_frames=20)
    assert lat > 0
