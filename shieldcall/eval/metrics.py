"""Detection and calibration metrics for telephony-condition evaluation."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def roc_curve(
    labels: Sequence[int], scores: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fpr, tpr, thresholds."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.unique(scores)[::-1]
    tpr_list, fpr_list = [], []
    n_pos = max((labels == 1).sum(), 1)
    n_neg = max((labels == 0).sum(), 1)
    for th in thresholds:
        pred = scores >= th
        tpr_list.append(((pred) & (labels == 1)).sum() / n_pos)
        fpr_list.append(((pred) & (labels == 0)).sum() / n_neg)
    return np.array(fpr_list), np.array(tpr_list), thresholds


def equal_error_rate(labels: Sequence[int], scores: Sequence[float]) -> float:
    """EER via threshold sweep."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) == 0:
        return 1.0
    thresholds = np.linspace(0, 1, 201)
    best = 1.0
    eer = 1.0
    n_pos = max((labels == 1).sum(), 1)
    n_neg = max((labels == 0).sum(), 1)
    for th in thresholds:
        pred = scores >= th
        fpr = ((pred) & (labels == 0)).sum() / n_neg
        fnr = ((~pred) & (labels == 1)).sum() / n_pos
        diff = abs(fpr - fnr)
        if diff < best:
            best = diff
            eer = 0.5 * (fpr + fnr)
    return float(eer)


def auc_roc(labels: Sequence[int], scores: Sequence[float]) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    if len(fpr) < 2:
        return 0.5
    # Ensure sorted by fpr
    order = np.argsort(fpr)
    y = tpr[order]
    x = fpr[order]
    # NumPy 2.0 removed np.trapz in favor of np.trapezoid
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if trapz is None:
        return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(x)))
    return float(trapz(y, x))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    labels = labels[order]
    tp = 0
    precisions = []
    for i, y in enumerate(labels, start=1):
        if y == 1:
            tp += 1
            precisions.append(tp / i)
    if not precisions:
        return 0.0
    return float(np.mean(precisions))


def brier_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    return float(np.mean((scores - labels) ** 2))


def expected_calibration_error(
    labels: Sequence[int], scores: Sequence[float], n_bins: int = 10
) -> float:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (scores >= bins[i]) & (scores < bins[i + 1] if i < n_bins - 1 else scores <= bins[i + 1])
        if not np.any(mask):
            continue
        conf = scores[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.sum() / len(scores)) * abs(acc - conf)
    return float(ece)


def summarize_scores(
    human_scores: List[float], synthetic_scores: List[float]
) -> float:
    """Backward-compatible rough EER from two score lists."""
    labels = [0] * len(human_scores) + [1] * len(synthetic_scores)
    scores = list(human_scores) + list(synthetic_scores)
    return equal_error_rate(labels, scores)
