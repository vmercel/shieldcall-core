"""
Cross-Stream Causal Fusion (CSCF)
=================================

Joint temporal fusion of acoustic authenticity and linguistic
fraud-intent that goes far beyond weighted averages.

Key mechanisms:
  1. Asymmetric stream weights (language often leads in vishing).
  2. Joint-score trajectory (rising risk is itself a feature).
  3. Cross-modal co-activation: both streams hot within a causal
     window  ->  super-additive risk (deepfake-enabled social engineering).
  4. Disagreement regimes: high fraud + low synth = human social
     engineer; high synth + low fraud = deepfake probe / spoof.
  5. Conformal calibration bands + optional abstention.
  6. Counterfactual explanations for every decision.

This cross-modal causal structure is the core scientific claim of
ShieldCall relative to single-stream liveness or keyword systems.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np

from ..acoustic.scorer import AcousticScore
from ..linguistic.scorer import LinguisticScore
from .conformal import StreamingConformalCalibrator, ConformalVerdict
from .explain import explain_risk, ThreatExplanation


@dataclass
class FusedRisk:
    timestamp_sec: float
    risk_score: float
    tier: str  # SAFE | SUSPICIOUS | HIGH_RISK | ABSTAIN
    acoustic_synth_prob: float
    linguistic_fraud_prob: float
    escalation_factor: float
    active_linguistic_groups: List[str] = field(default_factory=list)
    explanation: str = ""
    coactivation: float = 0.0
    regime: str = "agreement_low"
    conformal_lower: float = 0.0
    conformal_upper: float = 1.0
    abstain: bool = False
    discourse_stage: str = ""
    progression_depth: int = 0
    threat_explanation: Optional[ThreatExplanation] = None


class FusionEngine:
    """Streaming CSCF fusion engine."""

    def __init__(
        self,
        acoustic_weight: float = 0.40,
        linguistic_weight: float = 0.60,
        suspicious_threshold: float = 0.35,
        high_risk_threshold: float = 0.62,
        history_len: int = 40,
        coactivation_window: int = 15,
        use_conformal: bool = True,
        conformal_alpha: float = 0.1,
    ):
        self.acoustic_weight = acoustic_weight
        self.linguistic_weight = linguistic_weight
        self.suspicious_threshold = suspicious_threshold
        self.high_risk_threshold = high_risk_threshold
        self.coactivation_window = coactivation_window
        self.use_conformal = use_conformal

        self._history: Deque[float] = deque(maxlen=history_len)
        self._ac_hist: Deque[float] = deque(maxlen=coactivation_window)
        self._li_hist: Deque[float] = deque(maxlen=coactivation_window)

        self._last_acoustic: Optional[AcousticScore] = None
        self._last_linguistic: Optional[LinguisticScore] = None
        self._calibrator = StreamingConformalCalibrator(
            alpha=conformal_alpha,
            suspicious_threshold=suspicious_threshold,
            high_risk_threshold=high_risk_threshold,
        )

    def reset(self) -> None:
        self._history.clear()
        self._ac_hist.clear()
        self._li_hist.clear()
        self._last_acoustic = None
        self._last_linguistic = None
        self._calibrator.reset()

    def update_acoustic(self, score: AcousticScore) -> None:
        # Sticky speech scores: non-speech frames must not wipe acoustic
        # evidence (otherwise deepfake-probe risk collapses between words).
        if score.is_speech:
            self._last_acoustic = score
            self._ac_hist.append(score.synthetic_prob)
        elif self._last_acoustic is None:
            self._last_acoustic = score

    def update_linguistic(self, score: LinguisticScore) -> None:
        self._last_linguistic = score
        self._li_hist.append(score.fraud_prob)

    def _coactivation(self) -> float:
        """
        Measure concurrent elevation of both streams inside the causal window.
        Uses soft-AND of recent *means* (not lifetime max) so stale peaks decay.
        """
        if not self._ac_hist or not self._li_hist:
            return 0.0
        ac_recent = list(self._ac_hist)[-min(5, len(self._ac_hist)) :]
        li_recent = list(self._li_hist)[-min(5, len(self._li_hist)) :]
        ac = float(np.mean(ac_recent))
        li = float(np.mean(li_recent))
        # Soft AND
        return float(np.sqrt(max(ac, 0) * max(li, 0)))

    def fuse(self, timestamp_sec: Optional[float] = None) -> FusedRisk:
        ac = self._last_acoustic
        li = self._last_linguistic

        # Prefer peak recent acoustic evidence (sticky last speech + window max)
        synth_last = ac.synthetic_prob if ac and ac.is_speech else (
            ac.synthetic_prob if ac else 0.0
        )
        if ac and not ac.is_speech and self._ac_hist:
            synth_last = float(self._ac_hist[-1])
        synth_peak = float(max(self._ac_hist)) if self._ac_hist else synth_last
        # Blend peak (probe sensitivity) with last (stability)
        synth = 0.55 * synth_peak + 0.45 * synth_last
        # Residual cue when available strengthens deepfake evidence under mild language
        residual_cue = float(getattr(ac, "residual_cue", 0.0) or 0.0) if ac else 0.0
        synth_eff = float(np.clip(max(synth, 0.65 * residual_cue + 0.35 * synth), 0.0, 1.0))

        ac_conf = ac.confidence if ac and (ac.is_speech or self._ac_hist) else 0.0
        fraud = li.fraud_prob if li else 0.0
        li_conf = li.confidence if li else 0.0
        escalation = li.escalation_factor if li else 1.0
        groups = li.active_groups if li else []
        stage = li.discourse_stage if li else ""
        depth = li.progression_depth if li else 0

        ts = timestamp_sec
        if ts is None:
            ts = (ac.timestamp_sec if ac else None) or (li.timestamp_sec if li else 0.0)

        # Confidence-weighted stream blend (use synth_eff for decision path)
        aw = self.acoustic_weight * (0.5 + 0.5 * ac_conf)
        lw = self.linguistic_weight * (0.5 + 0.5 * li_conf)
        wsum = aw + lw + 1e-8
        base = (aw * synth_eff + lw * fraud) / wsum * (self.acoustic_weight + self.linguistic_weight)

        # Joint trajectory
        self._history.append(base)
        if len(self._history) >= 5:
            recent = list(self._history)[-5:]
            earlier = list(self._history)[:-5] or recent[:1]
            rise = np.mean(recent) - np.mean(earlier)
            if rise > 0.06:
                base *= 1.0 + 1.8 * min(rise, 0.35)

        # Escalation from linguistic stream
        base *= min(escalation, 2.2)

        # Cross-modal co-activation super-additivity
        coact = self._coactivation()
        if coact > 0.35:
            base *= 1.0 + 0.55 * coact

        # Regime-specific adjustments (core CSCF novelty vs naive weighted sum)
        # Social engineering: human (or uncertain) voice + hostile script  -> 
        # language dominates. Without this, acoustic "human" evidence would
        # incorrectly suppress a clear vishing path.
        if fraud >= 0.45 and synth_eff < 0.50:
            se_floor = 0.90 * fraud * min(max(escalation, 1.0), 2.0)
            if depth >= 3:
                se_floor = max(se_floor, 0.55 + 0.08 * min(depth, 6))
            base = max(base, se_floor)
        # Deepfake probe: synthetic residual/voice with mild language still elevates
        if synth_eff >= 0.35 and fraud < 0.40:
            base = max(base, 0.85 * synth_eff + 0.05 * fraud)
        # Dual threat: both elevated  ->  super-linear (beyond coactivation)
        if synth_eff >= 0.45 and fraud >= 0.45:
            base = max(base, 0.5 * (synth_eff + fraud) + 0.25 * min(synth_eff, fraud))

        # Expose effective acoustic probability in output fields
        synth = synth_eff

        risk = float(np.clip(base, 0.0, 1.0))

        # Tier
        if risk >= self.high_risk_threshold:
            tier = "HIGH_RISK"
        elif risk >= self.suspicious_threshold:
            tier = "SUSPICIOUS"
        else:
            tier = "SAFE"

        conf_verdict: Optional[ConformalVerdict] = None
        lo, hi, abstain = risk, risk, False
        if self.use_conformal:
            conf_verdict = self._calibrator.calibrate(risk)
            lo, hi = conf_verdict.lower, conf_verdict.upper
            if conf_verdict.abstain:
                tier = "ABSTAIN"
                abstain = True
            elif conf_verdict.tier != "ABSTAIN":
                # Prefer conformal tier when more conservative on high risk
                if conf_verdict.tier == "HIGH_RISK":
                    tier = "HIGH_RISK"

        threat = explain_risk(
            risk_score=risk,
            tier=tier,
            synth=synth,
            fraud=fraud,
            escalation=escalation,
            groups=groups,
            discourse_stage=stage,
            progression_depth=depth,
            coactivation=coact,
            suspicious_threshold=self.suspicious_threshold,
            high_risk_threshold=self.high_risk_threshold,
            acoustic_weight=self.acoustic_weight,
            linguistic_weight=self.linguistic_weight,
        )

        return FusedRisk(
            timestamp_sec=ts,
            risk_score=risk,
            tier=tier,
            acoustic_synth_prob=synth,
            linguistic_fraud_prob=fraud,
            escalation_factor=escalation,
            active_linguistic_groups=groups,
            explanation=threat.summary,
            coactivation=coact,
            regime=threat.regime,
            conformal_lower=lo,
            conformal_upper=hi,
            abstain=abstain,
            discourse_stage=stage,
            progression_depth=depth,
            threat_explanation=threat,
        )
