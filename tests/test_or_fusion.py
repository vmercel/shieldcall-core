"""Calibrated OR fusion in shadow beside CSCF (ADR-006, TDD v2.0 section 8.1).

Ground truth is synthetic data with known distributions, so every
expected value below is derived, not asserted from the code under test.
Nothing here touches licensed ASVspoof data.
"""

import math

import numpy as np
import pytest

from shieldcall.acoustic.scorer import AcousticScore
from shieldcall.fusion.engine import FusionEngine
from shieldcall.fusion.or_fusion import (
    ATTESTATION_DELTA,
    OrFusion,
    ShadowFusion,
    ShadowRow,
    attestation_delta,
    fit_rho,
    fused_cllr,
    fused_threat,
    joint_prob,
    noisy_or,
    posterior,
    severity,
    shadow_compare,
)


def _acoustic(prob, t=0.0):
    return AcousticScore(
        timestamp_sec=t,
        frame_index=0,
        synthetic_prob=float(prob),
        confidence=0.9,
        is_speech=True,
        features=np.zeros(4),
    )


def _linguistic(prob, t=0.0):
    from shieldcall.linguistic.scorer import LinguisticScore

    return LinguisticScore(
        timestamp_sec=t, fraud_prob=float(prob), confidence=0.9
    )


def test_envelope_bounds_and_endpoints():
    """Proposition 1(a)(c): max <= p_threat <= noisy-OR; rho=1 is max, rho=0 is noisy-OR."""
    rng = np.random.default_rng(0)
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        for _ in range(200):
            p_v, p_s = rng.uniform(0.0, 1.0, 2)
            p = fused_threat(p_v, p_s, rho)
            assert max(p_v, p_s) - 1e-12 <= p <= noisy_or(p_v, p_s) + 1e-12
    for _ in range(200):
        p_v, p_s = rng.uniform(0.0, 1.0, 2)
        assert fused_threat(p_v, p_s, 0.0) == pytest.approx(noisy_or(p_v, p_s))
        assert fused_threat(p_v, p_s, 1.0) == pytest.approx(max(p_v, p_s))


def test_envelope_monotone():
    """Proposition 1(b): p_threat is non-decreasing in each argument."""
    rng = np.random.default_rng(1)
    for rho in (0.0, 0.4, 1.0):
        for _ in range(100):
            p_v, p_s = rng.uniform(0.0, 1.0, 2)
            step = rng.uniform(0.0, 1.0 - max(p_v, p_s))
            assert fused_threat(p_v + step, p_s, rho) >= fused_threat(p_v, p_s, rho) - 1e-12
            assert fused_threat(p_v, p_s + step, rho) >= fused_threat(p_v, p_s, rho) - 1e-12


def test_severity_range_and_endpoints():
    """sev in [0, 1]; rho=0 gives pV*pS/noisy-OR, rho=1 gives min/max."""
    rng = np.random.default_rng(2)
    for _ in range(200):
        p_v, p_s = rng.uniform(0.01, 1.0, 2)
        for rho in (0.0, 0.5, 1.0):
            s = severity(p_v, p_s, rho)
            assert 0.0 <= s <= 1.0 + 1e-12
        assert severity(p_v, p_s, 0.0) == pytest.approx(
            p_v * p_s / noisy_or(p_v, p_s)
        )
        assert severity(p_v, p_s, 1.0) == pytest.approx(
            min(p_v, p_s) / max(p_v, p_s)
        )
    assert severity(0.0, 0.0, 0.5) == 0.0
    # Joint probability matches eq. 18 at both endpoints.
    assert joint_prob(0.3, 0.7, 0.0) == pytest.approx(0.21)
    assert joint_prob(0.3, 0.7, 1.0) == pytest.approx(0.3)


def test_posterior_prior_and_attestation():
    """Eq. 17: zero LLR returns the prior; attestation shifts only pS."""
    assert posterior(0.0, 0.05) == pytest.approx(0.05)
    assert posterior(0.0, 0.5) == pytest.approx(0.5)
    # log-odds of 0.05 plus 0.18 nats.
    expected = 1.0 / (1.0 + math.exp(-(math.log(0.05 / 0.95) + 0.18)))
    assert posterior(0.0, 0.05, 0.18) == pytest.approx(expected)
    assert attestation_delta("A") == 0.0
    assert attestation_delta("B") == pytest.approx(0.07)
    assert attestation_delta("C") == pytest.approx(0.18)
    assert attestation_delta("unsigned") == pytest.approx(0.18)
    assert max(ATTESTATION_DELTA.values()) <= 0.2


def test_fit_rho_recovers_known_dependence():
    """Labels drawn from the true fused model at rho=0.6: the fit recovers it."""
    rng = np.random.default_rng(3)
    rho_true = 0.6
    n = 6000
    p_v = rng.uniform(0.0, 1.0, n)
    p_s = rng.uniform(0.0, 1.0, n)
    p_threat = np.array([fused_threat(a, b, rho_true) for a, b in zip(p_v, p_s)])
    labels = (rng.uniform(0.0, 1.0, n) < p_threat).astype(int)
    result = fit_rho(p_v, p_s, labels)
    assert abs(result.rho - rho_true) <= 0.15
    assert result.n == n
    # The fit is never worse than either endpoint on its own fit data.
    assert result.cllr <= result.cllr_rho0 + 1e-9
    assert result.cllr <= result.cllr_rho1 + 1e-9


def test_fit_rho_prefers_max_when_streams_redundant():
    """When the label is exactly max(pV, pS) (full redundancy), rho fits near 1."""
    rng = np.random.default_rng(4)
    n = 4000
    p_v = rng.uniform(0.0, 1.0, n)
    p_s = rng.uniform(0.0, 1.0, n)
    p_threat = np.maximum(p_v, p_s)
    labels = (rng.uniform(0.0, 1.0, n) < p_threat).astype(int)
    result = fit_rho(p_v, p_s, labels)
    assert result.rho >= 0.8
    assert result.cllr < result.cllr_rho0


def test_fit_rho_degenerate_inputs():
    """Empty or single-class fit data stays uninformative instead of crashing."""
    empty = fit_rho([], [], [])
    assert empty.rho == 0.0 and empty.n == 0
    one_class = fit_rho([0.2, 0.8], [0.3, 0.9], [0, 0])
    assert one_class.rho == 0.0 and one_class.n == 2


def test_fused_llr_subtracts_threat_prior():
    """Regression: the Cllr LLR is logit(p_threat) minus logit(prior).

    Using logit(p_threat) alone smuggles the threat prior into the
    score and biases the rho fit toward whichever rho implies a prior
    closest to 0.5; that bug was caught by the rho-recovery test.
    """
    from shieldcall.fusion.or_fusion import _fused_llrs

    p_v = [0.8, 0.2]
    p_s = [0.7, 0.3]
    llr_half = _fused_llrs(p_v, p_s, 0.0, prior=0.5)
    p_threat = np.array([noisy_or(a, b) for a, b in zip(p_v, p_s)])
    assert llr_half == pytest.approx(np.log(p_threat / (1.0 - p_threat)))
    llr_shifted = _fused_llrs(p_v, p_s, 0.0, prior=0.2)
    shift = math.log(0.2 / 0.8)
    assert llr_shifted == pytest.approx(llr_half - shift)


def test_leaky_integration_converges_and_caps():
    """Constant LLR 1 with beta=0.9 tends to 1/(1-0.9)=10, capped at 8."""
    fusion = OrFusion(prior_v=0.5, prior_s=0.5, rho=0.0)
    result = None
    for _ in range(200):
        result = fusion.update(1.0, 0.0)
    # Smoothed acoustic LLR capped at 8: pV = sigma(8).
    assert result.p_v == pytest.approx(1.0 / (1.0 + math.exp(-8.0)), rel=1e-6)
    # Linguistic LLR stayed 0, so pS is the prior.
    assert result.p_s == pytest.approx(0.5)
    fusion.reset()
    result = fusion.update(0.0, 0.0)
    assert result.p_v == pytest.approx(0.5)
    assert result.p_s == pytest.approx(0.5)


def test_or_fusion_update_matches_closed_form():
    """One tick with beta=0 is exactly eq. 17 + eq. 19 on the raw LLRs."""
    fusion = OrFusion(prior_v=0.05, prior_s=0.05, rho=0.5, beta=0.0)
    result = fusion.update(2.0, -1.0, attestation="B")
    p_v = posterior(2.0, 0.05)
    p_s = posterior(-1.0, 0.05, 0.07)
    assert result.p_v == pytest.approx(p_v)
    assert result.p_s == pytest.approx(p_s)
    assert result.p_threat == pytest.approx(fused_threat(p_v, p_s, 0.5))
    assert result.severity == pytest.approx(severity(p_v, p_s, 0.5))
    assert result.rho == 0.5


def test_shadow_runner_leaves_cscf_untouched():
    """CSCF risk is identical with and without the shadow wrapper."""
    rng = np.random.default_rng(5)
    shadow = ShadowFusion(cscf=FusionEngine(), or_fusion=OrFusion(rho=0.3))
    plain = FusionEngine()
    risks_shadow, risks_plain = [], []
    for t in range(20):
        p_ac = float(rng.uniform(0.0, 1.0))
        p_li = float(rng.uniform(0.0, 1.0))
        ac, li = _acoustic(p_ac, t * 0.1), _linguistic(p_li, t * 0.1)
        risk, _ = shadow.update(ac, li, llr_ac=0.0, llr_lang=0.0)
        plain.update_acoustic(_acoustic(p_ac, t * 0.1))
        plain.update_linguistic(_linguistic(p_li, t * 0.1))
        risks_plain.append(plain.fuse(timestamp_sec=t * 0.1).risk_score)
        risks_shadow.append(risk)
    assert risks_shadow == pytest.approx(risks_plain)
    assert len(shadow.rows) == 20
    report = shadow.report()
    assert report["n"] == 20
    assert report["n_labelled"] == 0
    assert report["cscf_cllr"] is None
    assert 0.0 <= report["disagreement_rate"] <= 1.0


def test_shadow_compare_scores_both_systems():
    """On labelled rows the report scores CSCF and OR fusion side by side."""
    rng = np.random.default_rng(6)
    rows = []
    for _ in range(500):
        y = int(rng.uniform() < 0.5)
        # OR fusion well calibrated: threat probit tracks the label.
        p_or = rng.beta(6, 2) if y else rng.beta(2, 6)
        # CSCF deliberately noisier.
        p_cscf = float(np.clip(p_or + rng.normal(0, 0.15), 0.01, 0.99))
        rows.append(ShadowRow(cscf_risk=p_cscf, or_threat=p_or, or_severity=0.2, label=y))
    report = shadow_compare(rows)
    assert report["n"] == 500 and report["n_labelled"] == 500
    assert report["or_fusion_cllr"] < report["cscf_cllr"]
    assert 0.0 <= report["or_fusion_eer"] <= 0.5
    assert 0.0 <= report["disagreement_rate"] <= 1.0
    for key in ("cscf_min_dcf", "cscf_act_dcf", "or_fusion_min_dcf", "or_fusion_act_dcf"):
        assert report[key] is not None


def test_shadow_compare_empty_rows():
    """An empty shadow run reports n=0 instead of crashing."""
    report = shadow_compare([])
    assert report["n"] == 0
    assert report["disagreement_rate"] is None
    assert report["cscf_cllr"] is None
