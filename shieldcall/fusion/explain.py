"""
Threat explanations and input-space counterfactuals.

Given a fused risk decision, re-run the *same* scoring function with
intervened inputs (zero fraud, zero synth, no escalation, no coactivation).
That is a finite difference on the fusion function, not an LLM narrative
and not a search for a minimal perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

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
    score_fn: Optional[Callable[..., float]] = None,
    ac_conf: float = 0.8,
    li_conf: float = 0.8,
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

    cfs: List[CounterfactualAction] = []
    target = suspicious_threshold - 0.02 if tier == "SUSPICIOUS" else high_risk_threshold - 0.02

    def _score(s: float, f: float, esc: float, c: float) -> float:
        if score_fn is not None:
            return float(
                score_fn(
                    synth=s,
                    fraud=f,
                    ac_conf=ac_conf,
                    li_conf=li_conf,
                    escalation=esc,
                    depth=progression_depth,
                    coact=c,
                    apply_trajectory=False,
                )
            )
        return float(np.clip(acoustic_weight * s + linguistic_weight * f, 0, 1))

    if risk_score > target:
        r1 = _score(synth, 0.0, 1.0, 0.0)
        if r1 < risk_score - 0.05:
            cfs.append(
                CounterfactualAction(
                    action="neutralize_language",
                    detail="If scam-language cues were absent, risk would fall to ~{:.2f}".format(r1),
                    delta_risk=risk_score - r1,
                )
            )
        r2 = _score(0.0, fraud, escalation, 0.0)
        if r2 < risk_score - 0.05:
            cfs.append(
                CounterfactualAction(
                    action="authenticate_voice",
                    detail="If voice were confirmed human, risk would fall to ~{:.2f}".format(r2),
                    delta_risk=risk_score - r2,
                )
            )
        r3 = _score(synth, fraud, 1.0, coactivation)
        if r3 < risk_score - 0.05:
            cfs.append(
                CounterfactualAction(
                    action="flatten_trajectory",
                    detail="Without rising escalation, risk would fall to ~{:.2f}".format(r3),
                    delta_risk=risk_score - r3,
                )
            )
        if coactivation > 0.2:
            r4 = _score(synth, fraud, escalation, 0.0)
            cfs.append(
                CounterfactualAction(
                    action="break_coactivation",
                    detail="Removing joint elevation would move risk to ~{:.2f}".format(r4),
                    delta_risk=max(risk_score - r4, 0.0),
                )
            )

    return ThreatExplanation(
        summary=summary,
        drivers=drivers,
        counterfactuals=cfs,
        regime=regime,
    )
