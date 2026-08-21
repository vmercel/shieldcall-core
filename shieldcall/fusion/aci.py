"""Adaptive conformal inference (ACI) of Gibbs & Candès, NeurIPS 2021.

We implement the one-parameter online update

    α_{t+1} = Π_{[α_min, α_max]}( α_t + γ (α − err_t) )

where err_t is 1 if the interval misses the label, and the interval
half-width is the empirical (1 − α_t) quantile of stored residuals
|score − label|.

This is the published ACI recursion. It is not a finite-sample
coverage theorem under arbitrary dependence; Gibbs & Candès give
asymptotic / tracking guarantees under distribution shift. We report
*empirical* coverage on our stream. If coverage is not near 1 − α,
we say so.

Reference: Gibbs, Candès, "Adaptive Conformal Inference Under
Distribution Shift", NeurIPS 2021, arXiv:2106.00170.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ACIVerdict:
    lower: float
    upper: float
    alpha_t: float
    err: Optional[int]
    covered: Optional[bool]


class AdaptiveConformal:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.05,
        window: int = 400,
        alpha_min: float = 0.01,
        alpha_max: float = 0.5,
        initial_width: float = 0.25,
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.alpha_t = float(alpha)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.initial_width = initial_width
        self._residuals: Deque[float] = deque(maxlen=window)
        self.errors: List[int] = []
        self.alphas: List[float] = []

    def reset(self) -> None:
        self.alpha_t = self.alpha
        self._residuals.clear()
        self.errors = []
        self.alphas = []

    def _quantile(self) -> float:
        if len(self._residuals) < 8:
            return self.initial_width
        arr = np.asarray(self._residuals, dtype=np.float64)
        q_level = float(np.clip(1.0 - self.alpha_t, 0.0, 1.0))
        return float(np.quantile(arr, q_level))

    def interval(self, score: float) -> Tuple[float, float]:
        q = self._quantile()
        lo = float(np.clip(score - q, 0.0, 1.0))
        hi = float(np.clip(score + q, 0.0, 1.0))
        return lo, hi

    def observe(self, score: float, label: float) -> ACIVerdict:
        """Form the interval from *past* residuals, then update with this pair."""
        lo, hi = self.interval(score)
        y = float(label)
        covered = lo - 1e-12 <= y <= hi + 1e-12
        err = 0 if covered else 1
        self.errors.append(err)
        residual = abs(float(score) - y)
        self._residuals.append(residual)
        self.alpha_t = float(
            np.clip(
                self.alpha_t + self.gamma * (self.alpha - err),
                self.alpha_min,
                self.alpha_max,
            )
        )
        self.alphas.append(self.alpha_t)
        return ACIVerdict(
            lower=lo, upper=hi, alpha_t=self.alpha_t, err=err, covered=covered
        )

    def empirical_coverage(self) -> float:
        if not self.errors:
            return float("nan")
        return 1.0 - float(np.mean(self.errors))
