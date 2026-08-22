"""P0 upgrade tests: locked lexicon, TCT-2, fusion ablations, privacy, agent sim."""

from __future__ import annotations

import json

import numpy as np
import pytest

from shieldcall.agent.agent import DefenseAgent
from shieldcall.agent.belief import Perception
from shieldcall.agent.hypotheses import Action, Hypothesis
from shieldcall.agent.simulator import compare_policies
from shieldcall.audio.channel import ChannelConfig, CodecProfile, TelephonyChannelTwin
from shieldcall.eval.corpora.independent_scripts import independent_scripts
from shieldcall.eval.metrics import auc_roc, bootstrap_ci
from shieldcall.eval.protocols import fusion_ablation_from_pairs, linguistic_ablation_protocol
from shieldcall.eval.vocoders import vocode
from shieldcall.fusion.engine import FusionEngine
from shieldcall.linguistic.discourse import LEXICON_LOCK, STAGE_EMISSIONS, emission_only_score, path_only_score
from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime


def test_lexicon_lock_is_stable():
    assert len(LEXICON_LOCK) == 16
    frozen = json.dumps(STAGE_EMISSIONS, sort_keys=True, separators=(",", ":"))
    import hashlib

    assert hashlib.sha256(frozen.encode()).hexdigest()[:16] == LEXICON_LOCK


def test_independent_set_has_scam_and_benign():
    scripts = independent_scripts()
    assert len(scripts) >= 30
    assert any(s.is_scam for s in scripts)
    assert any(not s.is_scam for s in scripts)
    assert all(s.split == "independent" for s in scripts)


def test_wide_lexicon_is_not_identical_to_path_on_independent():
    scripts = independent_scripts()
    wide = [emission_only_score([t[1] for t in s.turns]) for s in scripts]
    path = [path_only_score([t[1] for t in s.turns]) for s in scripts]
    assert wide != path


def test_linguistic_ablation_runs():
    res = linguistic_ablation_protocol(confirmatory=True, asr_wer=0.0, seed=0)
    assert "ling_narrow_keywords" in res
    assert "ling_wide_lexicon" in res
    assert "ling_sdtg" in res
    assert "ling_ntm" in res
    for v in res.values():
        assert 0.0 <= v.auc <= 1.0
        assert v.n_samples >= 30


def test_tct2_profiles_change_waveform():
    rng = np.random.RandomState(0)
    x = rng.randn(4000).astype(np.float32) * 0.2
    sr = 8000
    clean = TelephonyChannelTwin(ChannelConfig(profile=CodecProfile.CLEAN)).apply(x, sr)
    for p in (CodecProfile.OPUS_NB, CodecProfile.NEURAL_CODEC, CodecProfile.G729_LIKE):
        y = TelephonyChannelTwin(ChannelConfig(profile=p, seed=1)).apply(x, sr)
        assert y.shape == clean.shape
        assert not np.allclose(y, clean, atol=1e-3)


def test_neural_quant_vocoder_runs():
    rng = np.random.RandomState(1)
    x = rng.randn(3000).astype(np.float32) * 0.15
    y = vocode(x, 8000, "neural_quant")
    assert y.shape == x.shape
    assert np.max(np.abs(y)) <= 1.0 + 1e-5


def test_production_pipeline_has_tct_off():
    cfg = PipelineConfig()
    assert cfg.channel is None


def test_fusion_or_and_floors_are_not_the_same():
    or_s = FusionEngine.calibrated_or(0.8, 0.1)
    fl = FusionEngine.floors_only(0.8, 0.1)
    naive = FusionEngine.naive_sum(0.8, 0.1)
    assert or_s > naive  # OR recovers complementary cells better than average
    assert fl >= 0.5  # spoof-probe floor
    att_a = FusionEngine().combine_streams(0.2, 0.2, attestation="A")
    att_u = FusionEngine().combine_streams(0.2, 0.2, attestation="unsigned")
    assert att_u > att_a


def test_bootstrap_ci_contains_point():
    labels = [0, 0, 0, 1, 1, 1, 1, 0]
    scores = [0.1, 0.2, 0.15, 0.9, 0.8, 0.7, 0.85, 0.3]
    pt, lo, hi = bootstrap_ci(labels, scores, auc_roc, n_boot=200, seed=0)
    assert lo <= pt <= hi


def test_agent_sim_warns_on_se_and_does_not_challenge_benign_much():
    table = compare_policies(n_per_class=12, n_steps=4, seed=0)
    se = table["agent"][Hypothesis.SOCIAL_ENGINEERING.value]
    ben = table["agent"][Hypothesis.BENIGN.value]
    assert se.false_challenge <= 0.25
    assert ben.false_challenge <= 0.35
    assert se.missed_harvest <= 0.35


def test_audit_trace_does_not_include_nonce_field():
    ag = DefenseAgent()
    ag.step(Perception(0.5, synth=0.9, fraud=0.1, risk=0.75, regime="deepfake_probe"))
    ag.step(Perception(1.5, synth=0.92, fraud=0.1, risk=0.8, regime="deepfake_probe"))
    dump = ag.trace_dicts()
    for row in dump:
        assert "nonce" not in row["tool"]


def test_two_sessions_do_not_share_cusum_or_belief():
    rt = SidecarRuntime(max_calls=3)
    a = rt.open_call("iso-a")
    b = rt.open_call("iso-b")
    assert a.pipeline.cusum is not b.pipeline.cusum
    assert a.pipeline.fusion is not b.pipeline.fusion
    assert a.agent is not b.agent


def test_fusion_ablation_keys():
    from shieldcall.eval.protocols import PairScore

    pairs = [
        PairScore(0.2, 0.8, 0.85, 0.44, 0.7, 1, "social_engineering"),
        PairScore(0.8, 0.2, 0.82, 0.44, 0.7, 1, "spoof_probe"),
        PairScore(0.1, 0.1, 0.1, 0.1, 0.1, 0, "safe"),
        PairScore(0.8, 0.8, 0.9, 0.8, 0.85, 1, "dual"),
    ]
    res = fusion_ablation_from_pairs(pairs)
    assert res["fuse_floors"].extras["disc_recall@0.5"] >= 0.5
    assert "fuse_calibrated_or" in res
