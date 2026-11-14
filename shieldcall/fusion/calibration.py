"""Per-cell LLR calibration (TDD v2.0 sections 5, 6.6; ADR-005).

Every acoustic score ``z`` in channel cell ``c`` is mapped to a
log-likelihood ratio by a cell-specific affine map (TDD eq. 9)::

    l = a_c * z + b_c,   (a_c, b_c) = argmin Cllr on held-out data of cell c

Cllr is TDD eq. 10 (base-2 log). Fitting minimises Cllr, which is the
logistic loss up to the constant factor 1/log(2), so it is convex in
``(a, b)`` and solved here with damped Newton iterations in numpy; no
new dependency, no sklearn at scoring time.

Labels follow the repo convention: 1 = spoof (target), 0 = bona fide.
LLRs are log p(x | spoof) / p(x | bona fide), so a high LLR means spoof.

The fit data must be held out from the data the scorer was trained on;
this module takes explicit fit data and never splits it, so the caller
owns the train / calibration / test separation.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def cllr(labels: Sequence[int], llrs: Sequence[float]) -> float:
    """Log-likelihood-ratio cost, TDD eq. 10, in bits.

    ``Cllr = 1/(2 Ns) sum_{i in spoof} log2(1 + exp(-l_i))
              + 1/(2 Nb) sum_{j in bona} log2(1 + exp(l_j))``

    0 is perfect, 1.0 is the uninformative (all LLRs zero) value.
    """
    labels = np.asarray(labels, dtype=int)
    llrs = np.asarray(llrs, dtype=float)
    if len(labels) == 0:
        return 1.0
    spoof = labels == 1
    bona = labels == 0
    n_spoof = max(int(spoof.sum()), 1)
    n_bona = max(int(bona.sum()), 1)
    term_spoof = np.sum(np.log2(1.0 + np.exp(-np.clip(llrs[spoof], -500.0, 500.0))))
    term_bona = np.sum(np.log2(1.0 + np.exp(np.clip(llrs[bona], -500.0, 500.0))))
    return float(0.5 * term_spoof / n_spoof + 0.5 * term_bona / n_bona)


def _pav(values: np.ndarray) -> np.ndarray:
    """Pool adjacent violators: isotonic (non-decreasing) fit of ``values``.

    Returns the fitted values in the original order. ``values`` must
    already be sorted by score ascending.
    """
    levels: list = []  # each [value, weight, count]
    for v in values:
        level = [float(v), 1.0, 1]
        while levels and levels[-1][0] > level[0] + 1e-12:
            prev = levels.pop()
            total_w = prev[1] + level[1]
            level = [
                (prev[0] * prev[1] + level[0] * level[1]) / total_w,
                total_w,
                prev[2] + level[2],
            ]
        levels.append(level)
    out = np.empty(len(values))
    i = 0
    for value, _weight, count in levels:
        out[i : i + count] = value
        i += count
    return out


def cllr_min(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Minimum Cllr over monotone recalibrations of ``scores``.

    The optimal monotone map is the isotonic (PAV) fit of the labels
    against the sorted scores; its Cllr isolates discrimination from
    calibration. PAV posteriors are clipped to [1e-6, 1 - 1e-6] so
    perfectly separable data gives a small finite value, not infinity.
    Always ``cllr_min <= cllr`` of any monotone map of the same scores.
    """
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(labels) == 0:
        return 1.0
    order = np.argsort(scores, kind="stable")
    fitted = _pav(labels[order])
    p = np.clip(fitted, 1e-6, 1.0 - 1e-6)
    llrs = np.log(p / (1.0 - p))
    return cllr(labels[order], llrs)


def fit_affine_llr(
    scores: Sequence[float],
    labels: Sequence[int],
    ridge: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> Tuple[float, float]:
    """Fit ``(a, b)`` minimising Cllr for LLR ``l = a * z + b``.

    The objective is the logistic loss (Cllr up to a constant factor),
    convex in ``(a, b)``. Solved by damped Newton iterations on the
    design ``[z, 1]`` with target labels 1 = spoof. A tiny ridge keeps
    the Hessian invertible when a cell's scores are (near-)constant.
    Returns ``(a, b)`` as plain floats.
    """
    z = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(z) == 0:
        return 0.0, 0.0
    if len(np.unique(y)) < 2:
        # One class only: the map is unidentifiable; stay uninformative.
        return 0.0, 0.0
    # Standardise the design for conditioning; unscale at the end.
    mu = float(np.mean(z))
    sd = float(np.std(z)) or 1.0
    x = (z - mu) / sd
    design = np.column_stack([x, np.ones_like(x)])
    w = np.zeros(2)
    # Start from the uninformative map: a = 0, b = logit(prior).
    prior = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
    w[1] = np.log(prior / (1.0 - prior))
    for _ in range(max_iter):
        p = _sigmoid(design @ w)
        grad = design.T @ (p - y) / len(y) + ridge * w
        s = np.clip(p * (1.0 - p), 1e-12, None)
        hess = (design * s[:, None]).T @ design / len(y) + ridge * np.eye(2)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        # Damp the step so the loss never increases.
        loss = _logistic_loss(design, y, w)
        t = 1.0
        while t > 1e-8:
            cand = w - t * step
            if _logistic_loss(design, y, cand) <= loss + 1e-12:
                break
            t *= 0.5
        w = w - t * step
        if float(np.max(np.abs(t * step))) < tol:
            break
    a = float(w[0] / sd)
    b = float(w[1] - w[0] * mu / sd)
    return a, b


def _logistic_loss(design: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    eta = np.clip(design @ w, -500.0, 500.0)
    return float(np.mean(np.logaddexp(0.0, eta) - y * eta))


class PerCellCalibrator:
    """Cell-specific affine LLR maps with a pooled fallback (ADR-005).

    A cell is fitted only when it holds at least ``min_class_count``
    trials of each class; otherwise the pooled map is used and the cell
    is recorded in ``fallback_cells``. Unknown cells at transform time
    also fall back to the pooled map. The fallback is a deliberate
    fail-open for scoring and is visible through ``fallback_cells``,
    not silent.
    """

    def __init__(self, min_class_count: int = 20):
        self.min_class_count = int(min_class_count)
        self.maps: Dict[str, Tuple[float, float]] = {}
        self.pooled: Tuple[float, float] = (0.0, 0.0)
        self.fallback_cells: set = set()

    def fit(self, cells: Mapping[str, Tuple[Sequence[float], Sequence[int]]]) -> "PerCellCalibrator":
        """Fit one map per cell plus the pooled map.

        ``cells`` maps a cell key to ``(scores, labels)`` on held-out
        data. Returns self.
        """
        all_scores: list = []
        all_labels: list = []
        for key, (scores, labels) in cells.items():
            scores = list(scores)
            labels = list(labels)
            all_scores.extend(scores)
            all_labels.extend(labels)
            n_spoof = sum(1 for y in labels if y == 1)
            n_bona = sum(1 for y in labels if y == 0)
            if n_spoof >= self.min_class_count and n_bona >= self.min_class_count:
                self.maps[str(key)] = fit_affine_llr(scores, labels)
            else:
                self.fallback_cells.add(str(key))
        self.pooled = fit_affine_llr(all_scores, all_labels) if all_scores else (0.0, 0.0)
        for key in self.fallback_cells:
            self.maps[str(key)] = self.pooled
        return self

    def map_for(self, cell: str) -> Tuple[float, float]:
        """Affine map for ``cell``; the pooled map when unknown."""
        return self.maps.get(str(cell), self.pooled)

    def llr(self, cell: str, score: float) -> float:
        """Calibrated LLR of one raw score in ``cell``."""
        a, b = self.map_for(cell)
        return float(a * score + b)

    def transform(self, cell: str, scores: Sequence[float]) -> np.ndarray:
        """Calibrated LLRs for raw ``scores`` in ``cell``."""
        a, b = self.map_for(cell)
        return a * np.asarray(scores, dtype=float) + b
