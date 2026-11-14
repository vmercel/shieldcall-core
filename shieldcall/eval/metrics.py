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


def tpr_at_fpr(
    labels: Sequence[int], scores: Sequence[float], target_fpr: float = 0.05
) -> float:
    """Maximum TPR with empirical FPR at most ``target_fpr``."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, _ = roc_curve(labels, scores)
    if len(fpr) == 0:
        return 0.0
    ok = np.where(fpr <= target_fpr + 1e-12)[0]
    if len(ok) == 0:
        return float(tpr[int(np.argmin(fpr))])
    return float(np.max(tpr[ok]))


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


# ---------------------------------------------------------------------------
# Actual-DCF and per-cell calibration reporting (TDD v2.0 sections 6.6, 15.4;
# ADR-005). EER and min DCF are oracle thresholds; a deployed system is
# judged at the threshold it actually uses, so actDCF is the DCF at the
# Bayes threshold on calibrated LLRs, with the same normaliser as min_dcf
# so the two are directly comparable (gate G2: actDCF within 10% of minDCF).
# ---------------------------------------------------------------------------

def bayes_threshold(
    p_target: float = 0.05, c_miss: float = 1.0, c_fa: float = 1.0
) -> float:
    """LLR threshold from Bayes' rule (TDD v2.0 section 6.6).

    Decide spoof when ``l > log(C_fa / C_miss) - logit(p_target)``:
    deciding spoof costs ``C_fa * P(bona | x)`` and deciding bona fide
    costs ``C_miss * P(spoof | x)``, so spoof wins when the posterior
    odds exceed ``C_fa / C_miss``. With the defaults (``p_target=0.05``,
    equal costs) the threshold is ``log(19)`` ≈ 2.944.
    """
    import math

    p = min(max(float(p_target), 1e-12), 1.0 - 1e-12)
    return float(math.log(float(c_fa) / float(c_miss)) - math.log(p / (1.0 - p)))


def dcf_at_threshold(
    labels: Sequence[int],
    llrs: Sequence[float],
    threshold: float,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Normalised DCF at one fixed LLR threshold (``llr > threshold``)."""
    labels = np.asarray(labels, dtype=int)
    llrs = np.asarray(llrs, dtype=float)
    n_pos = max((labels == 1).sum(), 1)
    n_neg = max((labels == 0).sum(), 1)
    pred = llrs > threshold
    fnr = ((~pred) & (labels == 1)).sum() / n_pos
    fpr = ((pred) & (labels == 0)).sum() / n_neg
    dcf = c_miss * p_target * fnr + c_fa * (1.0 - p_target) * fpr
    denom = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return float(dcf / max(denom, 1e-12))


def act_dcf(
    labels: Sequence[int],
    llrs: Sequence[float],
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Actual DCF at the Bayes threshold the system actually uses."""
    return dcf_at_threshold(
        labels, llrs, bayes_threshold(p_target, c_miss, c_fa), p_target, c_miss, c_fa
    )


def per_cell_report(
    cells,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    n_boot: int = 0,
):
    """Per-cell and pooled EER / minDCF / actDCF / Cllr / Cllr-min.

    ``cells`` maps a cell key (e.g. an ASVspoof 2021 LA codec: C1 none,
    C2 alaw, C3 pstn, C4 g722, C5 ulaw, C6 gsm, C7 opus) to
    ``(labels, llrs)`` with labels 1 = spoof and ``llrs`` calibrated
    log-likelihood ratios. Returns ``{"cells": {...}, "pooled": {...},
    "actdcf_spread": float | None, "pooled_min_dcf": float | None}``.

    Each row carries ``n``, ``n_spoof``, ``n_bona``, ``eer``,
    ``min_dcf``, ``act_dcf``, ``cllr``, ``cllr_min`` and, when
    ``n_boot > 0``, ``eer_ci`` as a ``(lo, hi)`` bootstrap interval.
    Rows for cells holding only one class carry ``None`` for the
    metrics that need both classes and are excluded from the spread.
    ``actdcf_spread`` is the best-to-worst codec actDCF spread, the
    quantity the TDD section 5 acceptance criterion requires
    per-cell calibration to narrow without raising pooled minDCF.
    """
    from ..fusion.calibration import cllr, cllr_min

    def _row(labels, llrs):
        labels = np.asarray(labels, dtype=int)
        llrs = np.asarray(llrs, dtype=float)
        n_spoof = int((labels == 1).sum())
        n_bona = int((labels == 0).sum())
        row = {"n": int(len(labels)), "n_spoof": n_spoof, "n_bona": n_bona}
        if n_spoof == 0 or n_bona == 0 or len(labels) == 0:
            row.update(
                {"eer": None, "min_dcf": None, "act_dcf": None, "cllr": None, "cllr_min": None}
            )
            return row
        # EER is invariant to monotone transforms; evaluate it on the
        # posterior so the [0, 1] threshold grid in equal_error_rate
        # applies (raw LLRs range over the whole real line).
        posterior = 1.0 / (1.0 + np.exp(-np.clip(llrs, -500.0, 500.0)))
        row["eer"] = equal_error_rate(labels, posterior)
        row["min_dcf"] = min_dcf(labels, posterior, p_target, c_miss, c_fa)
        row["act_dcf"] = act_dcf(labels, llrs, p_target, c_miss, c_fa)
        row["cllr"] = cllr(labels, llrs)
        row["cllr_min"] = cllr_min(labels, llrs)
        if n_boot > 0:
            _, lo, hi = bootstrap_ci(labels, posterior, equal_error_rate, n_boot=n_boot)
            row["eer_ci"] = (lo, hi)
        return row

    out_cells = {}
    pooled_labels: list = []
    pooled_llrs: list = []
    for key, (labels, llrs) in cells.items():
        out_cells[str(key)] = _row(labels, llrs)
        pooled_labels.extend(list(labels))
        pooled_llrs.extend(list(llrs))
    pooled = _row(pooled_labels, pooled_llrs)
    spreads = [
        row["act_dcf"]
        for row in out_cells.values()
        if row["act_dcf"] is not None
    ]
    return {
        "cells": out_cells,
        "pooled": pooled,
        "actdcf_spread": (max(spreads) - min(spreads)) if spreads else None,
        "pooled_min_dcf": pooled["min_dcf"],
    }
