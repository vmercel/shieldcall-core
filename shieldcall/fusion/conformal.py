"""
Conformal Streaming Risk (CSR)
==============================

Distribution-free uncertainty quantification for high-stakes call
screening. Maintains a calibration buffer of nonconformity scores and
emits prediction sets / abstention when the model is not confident
enough under the user-chosen error rate.

Unlike fixed thresholds, conformal bands adapt to the operating
distribution  -  critical when new synthesizers shift acoustic scores.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np


@dataclass
class ConformalVerdict:
    risk_score: float
    lower: float
    upper: float
    tier: str
    abstain: bool
    set_size: int
    alpha: float


class StreamingConformalCalibrator:
    """
    Sliding-window conformal calibrator for continuous risk scores.

    Nonconformity = |score - label| when labels available; when only
    scores flow, we use a self-calibrating residual against a slow EMA
    to produce interval width (conservative under covariate shift).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        window: int = 200,
        abstain_width: float = 0.45,
        suspicious_threshold: float = 0.35,
        high_risk_threshold: float = 0.62,
    ):
        self.alpha = alpha
        self.window = window
        self.abstain_width = abstain_width
        self.suspicious_threshold = suspicious_threshold
        self.high_risk_threshold = high_risk_threshold
        self._residuals: Deque[float] = deque(maxlen=window)
        self._ema: Optional[float] = None
        self._ema_beta = 0.05

    def reset(self) -> None:
        self._residuals.clear()
        self._ema = None

    def observe(self, score: float, label: Optional[float] = None) -> None:
        if label is not None:
            self._residuals.append(abs(float(score) - float(label)))
        else:
            if self._ema is None:
                self._ema = float(score)
            else:
                self._ema = (1 - self._ema_beta) * self._ema + self._ema_beta * float(score)
            self._residuals.append(abs(float(score) - self._ema))

    def _quantile(self) -> float:
        if len(self._residuals) < 10:
            return 0.25  # conservative default width half
        arr = np.array(self._residuals, dtype=np.float64)
        # Split conformal style: (1-alpha)(1+1/n) quantile
        n = len(arr)
        q_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        return float(np.quantile(arr, q_level))

    def calibrate(self, risk_score: float) -> ConformalVerdict:
        q = self._quantile()
        # Cap interval half-width so a single score jump cannot force permanent abstention
        q = float(min(q, 0.35))
        lo = float(np.clip(risk_score - q, 0.0, 1.0))
        hi = float(np.clip(risk_score + q, 0.0, 1.0))
        width = hi - lo

        # Set size: how many discrete tiers still possible inside [lo,hi]
        candidates = []
        for t_name, thr_lo, thr_hi in [
            ("SAFE", 0.0, self.suspicious_threshold),
            ("SUSPICIOUS", self.suspicious_threshold, self.high_risk_threshold),
            ("HIGH_RISK", self.high_risk_threshold, 1.01),
        ]:
            if hi >= thr_lo and lo < thr_hi:
                candidates.append(t_name)

        # Abstain only on true ambiguity: interval spans SAFE and a threat tier
        ambiguous = "SAFE" in candidates and len(candidates) > 1 and width >= self.abstain_width
        abstain = ambiguous

        if risk_score >= self.high_risk_threshold and lo >= self.suspicious_threshold:
            tier = "HIGH_RISK"
            abstain = False
        elif risk_score >= self.high_risk_threshold:
            tier = "HIGH_RISK"
        elif risk_score >= self.suspicious_threshold:
            tier = "SUSPICIOUS"
        else:
            tier = "SAFE"

        if abstain:
            tier = "ABSTAIN"

        self.observe(risk_score)

        return ConformalVerdict(
            risk_score=float(risk_score),
            lower=lo,
            upper=hi,
            tier=tier,
            abstain=abstain,
            set_size=len(candidates),
            alpha=self.alpha,
        )
