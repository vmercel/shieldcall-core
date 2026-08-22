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
    n_pos = max(int(labels.sum()), 1)
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


def bootstrap_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    fn,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (point, lo, hi) for ``fn(labels, scores)`` via case bootstrap."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    point = float(fn(labels, scores))
    if len(labels) < 4:
        return point, point, point
    rng = np.random.RandomState(seed)
    stats = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        stats.append(float(fn(labels[idx], scores[idx])))
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1.0 - alpha / 2))
    return point, lo, hi


def precision_at_fpr(
    labels: Sequence[int], scores: Sequence[float], target_fpr: float = 0.05
) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, thr = roc_curve(labels, scores)
    if len(fpr) == 0:
        return 0.0
    # operating point: largest TPR with FPR <= target
    ok = np.where(fpr <= target_fpr + 1e-12)[0]
    if len(ok) == 0:
        i = int(np.argmin(fpr))
    else:
        i = int(ok[np.argmax(tpr[ok])])
    th = thr[i] if i < len(thr) else 0.5
    pred = scores >= th
    tp = int(((pred) & (labels == 1)).sum())
    fp = int(((pred) & (labels == 0)).sum())
    return float(tp / max(tp + fp, 1))


def min_dcf(
    labels: Sequence[int],
    scores: Sequence[float],
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Normalized min DCF (ASVspoof-style, default 0.05 prior)."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = max((labels == 1).sum(), 1)
    n_neg = max((labels == 0).sum(), 1)
    best = 1.0
    for th in np.unique(scores):
        pred = scores >= th
        fnr = ((~pred) & (labels == 1)).sum() / n_pos
        fpr = ((pred) & (labels == 0)).sum() / n_neg
        dcf = c_miss * p_target * fnr + c_fa * (1.0 - p_target) * fpr
        best = min(best, dcf)
    denom = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return float(best / max(denom, 1e-12))


def recall_at_threshold(labels: Sequence[int], scores: Sequence[float], th: float = 0.5) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = scores >= th
    n_pos = max((labels == 1).sum(), 1)
    return float(((pred) & (labels == 1)).sum() / n_pos)


def fpr_at_threshold(labels: Sequence[int], scores: Sequence[float], th: float = 0.5) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = scores >= th
    n_neg = max((labels == 0).sum(), 1)
    return float(((pred) & (labels == 0)).sum() / n_neg)
