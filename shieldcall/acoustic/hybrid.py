"""Hybrid-H: logistic regression on residual embeddings (no deep net).

This is the candidate "physics features + learned linear head" baseline.
It is not AASIST. Report it next to residual+PMA, not as a neural vocoder
detector, until a published encoder is in the same table.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover
    LogisticRegression = None


class HybridHScorer:
    def __init__(self, seed: int = 0):
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn is required for HybridHScorer")
        self.clf = LogisticRegression(
            max_iter=400,
            solver="liblinear",
            class_weight="balanced",
            C=0.5,
            random_state=seed,
        )
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.fitted = False

    def _whiten(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            return X
        return (X - self.mean_) / self.std_

    def fit(self, embeddings: Sequence[np.ndarray], is_synthetic: Iterable[bool]) -> "HybridHScorer":
        X = np.stack([np.asarray(e, dtype=np.float64).ravel() for e in embeddings], axis=0)
        y = np.asarray(list(is_synthetic), dtype=int)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-6
        self.clf.fit(self._whiten(X), y)
        self.fitted = True
        return self

    def score(self, embedding: np.ndarray) -> float:
        if not self.fitted:
            return 0.5
        x = np.asarray(embedding, dtype=np.float64).ravel().reshape(1, -1)
        proba = self.clf.predict_proba(self._whiten(x))[0]
        classes = list(self.clf.classes_)
        if 1 in classes:
            return float(proba[classes.index(1)])
        return float(proba[-1])
