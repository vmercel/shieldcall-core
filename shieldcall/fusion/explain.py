"""
Counterfactual Threat Explanations (CTE)
========================================

Given a fused risk decision, compute minimal interventions that would
drop the call into a safer tier. These are not post-hoc LLM stories —
they are gradient-free, exact counterfactuals over the fusion inputs:

  - Reduce linguistic fraud probability
  - Reduce acoustic synthetic probability
  - Break co-activation
  - Flatten escalation trajectory

Used for operator UIs, audit logs, and user-facing warnings that name
*what would need to change* for the call to look legitimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class CounterfactualAction:
    action: str
    detail: str
    delta_risk: float


@dataclass
class ThreatExplanation:
    summary: str
    drivers: List[str]
    counterfactuals: List[CounterfactualAction] = field(default_factory=list)
    regime: str = "unknown"  # agreement | social_engineering | deepfake_probe | dual_threat


def classify_regime(
    synth: float,
    fraud: float,
    synth_thr: float = 0.4,
    fraud_thr: float = 0.4,
) -> str:
    high_s = synth >= synth_thr
    high_f = fraud >= fraud_thr
    if high_s and high_f:
        return "dual_threat"
    if high_f and not high_s:
        return "social_engineering"  # human voice, scam language
    if high_s and not high_f:
        return "deepfake_probe"  # synthetic voice, weak language
    return "agreement_low"


def explain_risk(
    risk_score: float,
    tier: str,
    synth: float,
    fraud: float,
    escalation: float,
    groups: List[str],
    discourse_stage: str = "",
    progression_depth: int = 0,
    coactivation: float = 0.0,
    suspicious_threshold: float = 0.35,
    high_risk_threshold: float = 0.62,
    acoustic_weight: float = 0.40,
    linguistic_weight: float = 0.60,
) -> ThreatExplanation:
    regime = classify_regime(synth, fraud)
    drivers: List[str] = []

    if fraud > 0.4:
        g = ", ".join(groups) if groups else "patterns"
        drivers.append(f"linguistic fraud signals ({g})")
    if synth > 0.4:
        drivers.append("elevated synthetic-voice evidence (STRF/prototypes)")
    if escalation > 1.3:
        drivers.append("rising threat trajectory")
    if coactivation > 0.3:
        drivers.append("cross-modal co-activation (language + voice anomalies together)")
    if progression_depth >= 4:
        drivers.append(
            f"scam discourse progression depth={progression_depth}"
            + (f" stage={discourse_stage}" if discourse_stage else "")
        )
    if not drivers:
        drivers.append("no strong indicators")

    regime_blurb = {
        "dual_threat": "Both voice authenticity and language intent look hostile.",
        "social_engineering": "Voice may be human; the *script* is the weapon.",
        "deepfake_probe": "Synthetic voice cues dominate; language is secondary.",
        "agreement_low": "Streams agree on low threat.",
    }[regime]

    summary = f"{tier}: risk={risk_score:.2f}. {regime_blurb}"
    if drivers and drivers[0] != "no strong indicators":
        summary += " Drivers: " + "; ".join(drivers) + "."

    # Counterfactuals: what minimal change drops below high_risk / suspicious
    cfs: List[CounterfactualAction] = []
    target = suspicious_threshold - 0.02 if tier == "SUSPICIOUS" else high_risk_threshold - 0.02
    if risk_score > target:
        # Approximate risk ≈ clip((aw*synth + lw*fraud) * esc * (1+coact_bonus))
        base = acoustic_weight * synth + linguistic_weight * fraud
        esc = max(escalation, 1.0)

        # CF1: drop fraud to 0
        r1 = float(np.clip(acoustic_weight * synth * esc, 0, 1))
        if r1 < risk_score - 0.05:
            cfs.append(
                CounterfactualAction(
                    action="neutralize_language",
                    detail="If scam-language cues were absent, risk would fall to ~{:.2f}".format(r1),
                    delta_risk=risk_score - r1,
                )
            )
        # CF2: drop synth to 0
        r2 = float(np.clip(linguistic_weight * fraud * esc, 0, 1))
        if r2 < risk_score - 0.05:
            cfs.append(
                CounterfactualAction(
                    action="authenticate_voice",
                    detail="If voice were confirmed human, risk would fall to ~{:.2f}".format(r2),
                    delta_risk=risk_score - r2,
                )
            )
        # CF3: flatten escalation
        r3 = float(np.clip(base, 0, 1))
        if r3 < risk_score - 0.05:
            cfs.append(
                CounterfactualAction(
                    action="flatten_trajectory",
                    detail="Without rising escalation, risk would fall to ~{:.2f}".format(r3),
                    delta_risk=risk_score - r3,
                )
            )
        # CF4: break coactivation
        if coactivation > 0.2:
            cfs.append(
                CounterfactualAction(
                    action="break_coactivation",
                    detail="Desynchronizing voice and language anomalies would reduce joint confidence.",
                    delta_risk=0.1 * coactivation,
                )
            )

    return ThreatExplanation(
        summary=summary,
        drivers=drivers,
        counterfactuals=cfs,
        regime=regime,
    )
