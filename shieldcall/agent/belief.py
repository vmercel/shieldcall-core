"""Heuristic belief over call hypotheses.

This is not a calibrated Bayesian network. Each percept multiplies a
hand-written likelihood and we renormalize. The numbers are declared
here so they can be attacked; they are not fitted to ASVspoof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np

from .hypotheses import ALL, Hypothesis


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


@dataclass
class Perception:
    """Sufficient statistics. No waveform, no transcript text."""

    timestamp_sec: float
    synth: float = 0.0
    fraud: float = 0.0
    handoff_score: float = 0.0
    handoff_pvalue: float = 1.0
    coverage_gap: float = 0.0
    regime: str = ""
    risk: float = 0.0
    abstain_band: bool = False


@dataclass
class Belief:
    p: Dict[Hypothesis, float] = field(
        default_factory=lambda: {
            Hypothesis.BENIGN: 0.80,
            Hypothesis.SOCIAL_ENGINEERING: 0.08,
            Hypothesis.SYNTHETIC_FULL: 0.05,
            Hypothesis.HANDOFF: 0.04,
            Hypothesis.UNKNOWN_FAMILY: 0.03,
        }
    )

    def as_dict(self) -> Dict[str, float]:
        return {h.value: float(self.p[h]) for h in ALL}

    def entropy(self) -> float:
        ent = 0.0
        for v in self.p.values():
            if v > 1e-12:
                ent -= v * np.log2(v)
        return float(ent)

    def mode(self) -> Hypothesis:
        return max(ALL, key=lambda h: self.p[h])

    def mass(self, *hs: Hypothesis) -> float:
        return float(sum(self.p[h] for h in hs))


def likelihoods(perc: Perception) -> Dict[Hypothesis, float]:
    s = _clip01(perc.synth)
    f = _clip01(perc.fraud)
    g = _clip01(perc.coverage_gap)
    ho = _clip01(perc.handoff_score)
    if perc.handoff_pvalue < 0.15:
        ho = max(ho, 0.55)
    eps = 0.02
    return {
        Hypothesis.BENIGN: eps + (1 - s) * (1 - f) * (1 - ho) * (1 - 0.5 * g),
        Hypothesis.SOCIAL_ENGINEERING: eps + f * (1 - 0.7 * s) * (1 - 0.4 * ho),
        Hypothesis.SYNTHETIC_FULL: eps + s * (1 - 0.5 * f) * (1 - 0.5 * ho),
        Hypothesis.HANDOFF: eps + 0.35 * ho + 0.4 * ho * f + 0.2 * s * f,
        Hypothesis.UNKNOWN_FAMILY: eps + g * (0.6 + 0.4 * s),
    }


def update(belief: Belief, perc: Perception, inertia: float = 0.65) -> Belief:
    """Multiply current belief by likelihood, mix with inertia, renormalize."""
    like = likelihoods(perc)
    raw: Dict[Hypothesis, float] = {}
    for h in ALL:
        raw[h] = (inertia * belief.p[h] + (1 - inertia) * like[h]) * like[h]
    z = sum(raw.values()) + 1e-12
    return Belief(p={h: raw[h] / z for h in ALL})


def entropy(p: Mapping[Hypothesis, float]) -> float:
    ent = 0.0
    for v in p.values():
        if v > 1e-12:
            ent -= float(v) * np.log2(float(v))
    return float(ent)
