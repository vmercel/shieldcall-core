"""Late-fusion baselines used in experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover
    LogisticRegression = None


class LogisticLateFusion:
    """Logistic regression on (synth_prob, fraud_prob). Requires sklearn."""

    def __init__(self):
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn is required for LogisticLateFusion")
        self.model = LogisticRegression(max_iter=500, class_weight="balanced")
        self._fitted = False

    def fit(self, synth: Sequence[float], fraud: Sequence[float], labels: Sequence[int]) -> None:
        x = np.column_stack([np.asarray(synth, dtype=float), np.asarray(fraud, dtype=float)])
        y = np.asarray(labels, dtype=int)
        self.model.fit(x, y)
        self._fitted = True

    def predict_proba(self, synth: float, fraud: float) -> float:
        if not self._fitted:
            return 0.4 * synth + 0.6 * fraud
        x = np.array([[synth, fraud]], dtype=float)
        return float(self.model.predict_proba(x)[0, 1])
