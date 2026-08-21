"""
Cross-stream score fusion (CSCF)
================================

Rule-based fusion of acoustic authenticity and linguistic fraud-intent.

This is not causal inference. "Causal window" only means the two streams
are aligned in time (no future frames). Mechanisms:

  1. Confidence-weighted blend of the two streams.
  2. Trajectory: a rising score slightly increases risk.
  3. Co-activation: both streams elevated in the recent window.
  4. Disagreement regimes as explicit floors:
       high fraud + not-high synth  ->  social-engineering floor
       high synth + low fraud       ->  spoof-probe floor (confidence-gated)
       both high                    ->  dual-threat floor
  5. Optional heuristic uncertainty band (not split-conformal coverage).
  6. Counterfactuals by re-running this same scoring function.

The claim to test is operational: on calls labeled by *threat* (scam
language OR synthetic voice), this should beat a naive weighted sum,
especially on disagreement cells. If it does not, drop the claim.
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
        self._last_coact: float = 0.0
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
        self._last_coact = 0.0
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

    def _effective_synth(self) -> tuple[float, float]:
        ac = self._last_acoustic
        synth_last = ac.synthetic_prob if ac and ac.is_speech else (
            ac.synthetic_prob if ac else 0.0
        )
        if ac and not ac.is_speech and self._ac_hist:
            synth_last = float(self._ac_hist[-1])
        synth_peak = float(max(self._ac_hist)) if self._ac_hist else synth_last
        synth = 0.55 * synth_peak + 0.45 * synth_last
        residual_cue = float(getattr(ac, "residual_cue", 0.0) or 0.0) if ac else 0.0
        synth_eff = float(np.clip(max(synth, 0.65 * residual_cue + 0.35 * synth), 0.0, 1.0))
        ac_conf = ac.confidence if ac and (ac.is_speech or self._ac_hist) else 0.0
        return synth_eff, ac_conf

    def combine_streams(
        self,
        synth: float,
        fraud: float,
        ac_conf: float = 0.8,
        li_conf: float = 0.8,
        escalation: float = 1.0,
        depth: int = 0,
        coact: float = 0.0,
        apply_trajectory: bool = True,
    ) -> float:
        """Score from stream components. Used by fuse() and counterfactuals."""
        aw = self.acoustic_weight * (0.5 + 0.5 * ac_conf)
        lw = self.linguistic_weight * (0.5 + 0.5 * li_conf)
        wsum = aw + lw + 1e-8
        base = (aw * synth + lw * fraud) / wsum * (self.acoustic_weight + self.linguistic_weight)

        if apply_trajectory and len(self._history) >= 5:
            recent = list(self._history)[-5:]
            earlier = list(self._history)[:-5] or recent[:1]
            rise = float(np.mean(recent) - np.mean(earlier))
            if rise > 0.06:
                base *= 1.0 + 1.2 * min(rise, 0.25)

        base *= min(max(escalation, 1.0), 1.8)

        if coact > 0.45 and synth >= 0.40 and fraud >= 0.40:
            base *= 1.0 + 0.35 * coact

        # Social engineering: hostile script, voice not clearly synthetic.
        if fraud >= 0.45 and li_conf >= 0.35 and synth < 0.55:
            se_floor = 0.88 * fraud * min(max(escalation, 1.0), 1.6)
            if depth >= 3:
                se_floor = max(se_floor, 0.52 + 0.06 * min(depth, 6))
            base = max(base, se_floor)

        # Spoof probe: only if acoustic evidence is actually confident.
        if synth >= 0.55 and ac_conf >= 0.45 and fraud < 0.40:
            base = max(base, 0.80 * synth + 0.08 * fraud)

        if synth >= 0.50 and fraud >= 0.50:
            base = max(base, 0.5 * (synth + fraud) + 0.20 * min(synth, fraud))

        return float(np.clip(base, 0.0, 1.0))

    @staticmethod
    def naive_sum(synth: float, fraud: float, acoustic_weight: float = 0.40, linguistic_weight: float = 0.60) -> float:
        return float(np.clip(acoustic_weight * synth + linguistic_weight * fraud, 0.0, 1.0))

    def fuse(self, timestamp_sec: Optional[float] = None) -> FusedRisk:
        ac = self._last_acoustic
        li = self._last_linguistic

        synth_eff, ac_conf = self._effective_synth()
        fraud = li.fraud_prob if li else 0.0
        li_conf = li.confidence if li else 0.0
        escalation = li.escalation_factor if li else 1.0
        groups = li.active_groups if li else []
        stage = li.discourse_stage if li else ""
        depth = li.progression_depth if li else 0

        ts = timestamp_sec
        if ts is None:
            ts = (ac.timestamp_sec if ac else None) or (li.timestamp_sec if li else 0.0)

        coact = self._coactivation()
        self._last_coact = coact
        risk = self.combine_streams(
            synth=synth_eff,
            fraud=fraud,
            ac_conf=ac_conf,
            li_conf=li_conf,
            escalation=escalation,
            depth=depth,
            coact=coact,
            apply_trajectory=True,
        )
        self._history.append(risk)
        synth = synth_eff

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
            score_fn=self.combine_streams,
            ac_conf=ac_conf,
            li_conf=li_conf,
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
