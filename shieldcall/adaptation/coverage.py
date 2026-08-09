"""
Coverage-Debt Tracker
=====================

When a new voice-cloning synthesizer appears, global classifiers fail
silently until the next retraining cycle. Coverage debt is the lag
between *emergence of a new attack family* and *detector competence*.

This module tracks:
  - OOD / coverage-gap rates from prototype memory
  - Time-to-recovery after few-shot adaptation
  - Per-family competence scores

It turns adaptation from a vague aspiration into a measurable control
loop — a first-class research metric for the national-interest agenda.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
import time

import numpy as np


@dataclass
class CoverageSnapshot:
    timestamp: float
    mean_gap: float
    high_gap_rate: float  # fraction of frames with gap > threshold
    n_observations: int
    families_known: int
    debt_index: float  # 0 = fully covered, 1 = severe debt


@dataclass
class FamilyCompetence:
    family: str
    n_examples: int
    mean_gap_after_adapt: float
    recovery_frames: int
    competent: bool


class CoverageDebtTracker:
    def __init__(
        self,
        gap_threshold: float = 0.55,
        window: int = 500,
        competence_examples: int = 5,
        competence_gap: float = 0.35,
    ):
        self.gap_threshold = gap_threshold
        self.window = window
        self.competence_examples = competence_examples
        self.competence_gap = competence_gap
        self._gaps: Deque[float] = deque(maxlen=window)
        self._family_examples: Dict[str, int] = defaultdict(int)
        self._family_gaps: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100))
        self._adapt_events: List[dict] = []

    def observe_gap(self, gap: float, family: str = "unknown") -> None:
        self._gaps.append(float(gap))
        if family and family != "unknown":
            self._family_gaps[family].append(float(gap))

    def register_adaptation(
        self,
        family: str,
        is_synthetic: bool,
        gap_before: float,
        gap_after: Optional[float] = None,
    ) -> None:
        self._family_examples[family] += 1
        self._adapt_events.append(
            {
                "t": time.time(),
                "family": family,
                "is_synthetic": is_synthetic,
                "gap_before": gap_before,
                "gap_after": gap_after,
            }
        )
        if gap_after is not None:
            self._family_gaps[family].append(gap_after)

    def snapshot(self) -> CoverageSnapshot:
        if not self._gaps:
            return CoverageSnapshot(
                timestamp=time.time(),
                mean_gap=0.0,
                high_gap_rate=0.0,
                n_observations=0,
                families_known=len(self._family_examples),
                debt_index=0.0,
            )
        gaps = np.array(self._gaps, dtype=np.float64)
        high = float(np.mean(gaps > self.gap_threshold))
        mean_gap = float(np.mean(gaps))
        # Debt index blends OOD rate with lack of family coverage
        family_factor = 1.0 / (1.0 + 0.3 * len(self._family_examples))
        debt = float(np.clip(0.6 * high + 0.4 * mean_gap * family_factor, 0.0, 1.0))
        return CoverageSnapshot(
            timestamp=time.time(),
            mean_gap=mean_gap,
            high_gap_rate=high,
            n_observations=len(gaps),
            families_known=len(self._family_examples),
            debt_index=debt,
        )

    def family_competence(self, family: str) -> FamilyCompetence:
        n = self._family_examples.get(family, 0)
        gaps = list(self._family_gaps.get(family, []))
        mean_gap = float(np.mean(gaps)) if gaps else 1.0
        competent = n >= self.competence_examples and mean_gap <= self.competence_gap
        return FamilyCompetence(
            family=family,
            n_examples=n,
            mean_gap_after_adapt=mean_gap,
            recovery_frames=len(gaps),
            competent=competent,
        )

    def reset(self) -> None:
        self._gaps.clear()
        self._family_examples.clear()
        self._family_gaps.clear()
        self._adapt_events.clear()
