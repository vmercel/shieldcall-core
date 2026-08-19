import numpy as np

from shieldcall.acoustic.scorer import AcousticScore
from shieldcall.fusion.conformal import StreamingConformalCalibrator
from shieldcall.fusion.engine import FusionEngine
from shieldcall.fusion.explain import classify_regime, explain_risk
from shieldcall.linguistic.scorer import LinguisticScore


def _ac(synth, conf=0.8, speech=True):
    return AcousticScore(
        timestamp_sec=1.0,
        frame_index=10,
        synthetic_prob=synth,
        confidence=conf,
        is_speech=speech,
        features=np.zeros(64, dtype=np.float32),
    )


def _li(fraud, esc=1.0, groups=None):
    return LinguisticScore(
        timestamp_sec=1.0,
        fraud_prob=fraud,
        confidence=0.8,
        active_groups=groups or [],
        escalation_factor=esc,
        discourse_stage="PAYMENT",
        progression_depth=5,
    )


def test_safe_when_low():
    fe = FusionEngine(use_conformal=False)
    fe.update_acoustic(_ac(0.1))
    fe.update_linguistic(_li(0.05))
    r = fe.fuse(1.0)
    assert r.tier == "SAFE"
    assert r.risk_score < 0.35


def test_high_risk_scam_language():
    fe = FusionEngine(use_conformal=False)
    for _ in range(5):
        fe.update_acoustic(_ac(0.25))
        fe.update_linguistic(_li(0.9, esc=1.5, groups=["payment_urgency", "identity_harvest"]))
    r = fe.fuse(2.0)
    assert r.risk_score > 0.5
    assert r.tier in ("HIGH_RISK", "SUSPICIOUS")
    assert r.threat_explanation is not None
    assert r.regime in ("social_engineering", "dual_threat", "agreement_low", "deepfake_probe")


def test_coactivation_increases_risk():
    fe = FusionEngine(use_conformal=False)
    fe.update_acoustic(_ac(0.7))
    fe.update_linguistic(_li(0.7, groups=["payment_urgency"]))
    r = fe.fuse(1.0)
    assert r.coactivation > 0.3


def test_regime_classification():
    assert classify_regime(0.1, 0.8) == "social_engineering"
    assert classify_regime(0.8, 0.1) == "deepfake_probe"
    assert classify_regime(0.8, 0.8) == "dual_threat"


def test_counterfactuals_present():
    ex = explain_risk(0.9, "HIGH_RISK", synth=0.2, fraud=0.9, escalation=1.5, groups=["payment_urgency"])
    assert ex.counterfactuals
    assert any(c.action == "neutralize_language" for c in ex.counterfactuals)


def test_conformal_bounds():
    cal = StreamingConformalCalibrator(alpha=0.1)
    for s in np.linspace(0.1, 0.9, 50):
        cal.observe(float(s))
    v = cal.calibrate(0.5)
    assert v.lower <= v.risk_score <= v.upper
