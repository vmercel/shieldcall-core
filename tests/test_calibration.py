"""Per-cell LLR calibration and actDCF / Cllr reporting (ADR-005).

Ground truth is synthetic data with known distributions, so every
expected value below is derived, not asserted from the code under test.
Nothing here touches licensed ASVspoof data.
"""

import math

import numpy as np
import pytest

from shieldcall.eval.metrics import (
    act_dcf,
    bayes_threshold,
    dcf_at_threshold,
    min_dcf,
    per_cell_report,
)
from shieldcall.fusion.calibration import (
    PerCellCalibrator,
    cllr,
    cllr_min,
    fit_affine_llr,
)


def _synthetic_llrs(rng, n_spoof=400, n_bona=400, sep=2.0, shift=0.0):
    """Genuine LLRs from two Gaussians with equal priors and unit variance.

    Draw ``x`` from N(+sep/2, 1) for spoof and N(-sep/2, 1) for bona
    fide; the log-likelihood ratio of ``x`` is then exactly ``sep * x``,
    so the returned values are true LLRs, not just scores. (Drawing
    N(+sep/2, 1) / N(-sep/2, 1) and calling the draw itself an LLR
    would be wrong by a factor of ``sep``: the LLR of that draw is
    ``sep`` times the draw.) ``shift`` adds a constant miscalibration
    offset to every LLR.
    """
    x_spoof = rng.normal(sep / 2.0, 1.0, n_spoof)
    x_bona = rng.normal(-sep / 2.0, 1.0, n_bona)
    labels = np.r_[np.ones(n_spoof), np.zeros(n_bona)]
    llrs = sep * np.r_[x_spoof, x_bona] + shift
    return labels, llrs


def test_cllr_uninformative_is_one_bit():
    labels = np.array([1, 1, 0, 0])
    assert cllr(labels, np.zeros(4)) == pytest.approx(1.0)


def test_cllr_perfect_is_near_zero():
    labels = np.array([1, 1, 0, 0])
    assert cllr(labels, np.array([20.0, 20.0, -20.0, -20.0])) < 1e-6


def test_cllr_known_value():
    # One spoof trial at l = 0 and one bona fide trial at l = 0:
    # Cllr = 0.5 * log2(2) + 0.5 * log2(2) = 1.
    assert cllr([1, 0], [0.0, 0.0]) == pytest.approx(1.0)
    # Spoof at l = 1, bona at l = -1: each term log2(1 + e^-1).
    expected = math.log2(1.0 + math.exp(-1.0))
    assert cllr([1, 0], [1.0, -1.0]) == pytest.approx(expected)


def test_cllr_min_never_above_cllr_and_zero_when_separable():
    rng = np.random.RandomState(0)
    labels, llrs = _synthetic_llrs(rng)
    assert cllr_min(labels, llrs) <= cllr(labels, llrs) + 1e-9
    labels_sep = np.array([1, 1, 1, 0, 0, 0])
    scores_sep = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
    assert cllr_min(labels_sep, scores_sep) < 0.01
    # Cllr-min of a constant score is the uninformative value: PAV fits
    # the prior everywhere, so no discrimination is possible.
    assert cllr_min(labels, np.zeros(len(labels))) == pytest.approx(1.0, abs=0.02)


def test_fit_affine_llr_recovers_known_map():
    rng = np.random.RandomState(1)
    # True LLR l ~ spoof N(1, 1), bona N(-1, 1); raw score z = (l - 3) / 2,
    # so the true map back is l = 2 * z + 3.
    labels, true_llrs = _synthetic_llrs(rng, sep=2.0)
    z = (true_llrs - 3.0) / 2.0
    a, b = fit_affine_llr(z, labels)
    assert a == pytest.approx(2.0, rel=0.15)
    assert b == pytest.approx(3.0, rel=0.15)


def test_fit_affine_llr_improves_cllr_on_held_out_data():
    rng = np.random.RandomState(2)
    labels_tr, llrs_tr = _synthetic_llrs(rng, sep=2.0, shift=1.5)
    labels_te, llrs_te = _synthetic_llrs(rng, sep=2.0, shift=1.5)
    # Raw scores are the true LLRs shifted by a miscalibration offset.
    z_tr = llrs_tr + 2.0
    z_te = llrs_te + 2.0
    assert cllr(labels_te, z_te) > 1.0  # shifted LLRs are worse than useless
    a, b = fit_affine_llr(z_tr, labels_tr)
    fitted = a * z_te + b
    assert cllr(labels_te, fitted) < cllr(labels_te, z_te)
    assert cllr(labels_te, fitted) < 0.6  # near the true-LLR value


def test_fit_affine_llr_degenerate_inputs():
    assert fit_affine_llr([], []) == (0.0, 0.0)
    assert fit_affine_llr([1.0, 2.0], [1, 1]) == (0.0, 0.0)


def test_bayes_threshold_defaults_and_unequal_costs():
    # Defaults: p = 0.05, equal costs -> log(19).
    assert bayes_threshold() == pytest.approx(math.log(19.0))
    # Misses cost 10x false alarms: the threshold must fall so fewer
    # spoofs are missed.
    assert bayes_threshold(0.05, c_miss=10.0, c_fa=1.0) < bayes_threshold()
    # False alarms cost 10x misses: the threshold must rise.
    assert bayes_threshold(0.05, c_miss=1.0, c_fa=10.0) > bayes_threshold()


def test_act_dcf_matches_min_dcf_when_calibrated():
    rng = np.random.RandomState(3)
    labels, llrs = _synthetic_llrs(rng, n_spoof=2000, n_bona=2000, sep=3.0)
    # True LLRs: the Bayes threshold is (near-)optimal, so actDCF must
    # sit close to minDCF.
    assert act_dcf(labels, llrs) == pytest.approx(min_dcf(labels, llrs), rel=0.15)


def test_act_dcf_penalises_miscalibration_that_min_dcf_cannot_see():
    rng = np.random.RandomState(4)
    labels, llrs = _synthetic_llrs(rng, n_spoof=2000, n_bona=2000, sep=3.0)
    shifted = llrs + 3.0  # same discrimination, wrong calibration
    # minDCF sweeps thresholds, so it cannot see the shift at all.
    assert min_dcf(labels, shifted) == pytest.approx(min_dcf(labels, llrs), rel=1e-6)
    # actDCF uses the fixed Bayes threshold, so it must get much worse.
    assert act_dcf(labels, shifted) > 2.0 * act_dcf(labels, llrs)


def test_act_dcf_unequal_costs_threshold_is_bayes_not_flipped():
    rng = np.random.RandomState(5)
    labels, llrs = _synthetic_llrs(rng, n_spoof=2000, n_bona=2000, sep=3.0)
    # With misses 10x as costly, the Bayes threshold must sit near the
    # optimum: actDCF close to minDCF. A flipped cost ratio would put
    # the threshold on the wrong side and blow actDCF up.
    assert act_dcf(labels, llrs, c_miss=10.0, c_fa=1.0) == pytest.approx(
        min_dcf(labels, llrs, c_miss=10.0, c_fa=1.0), rel=0.2
    )
    assert act_dcf(labels, llrs, c_miss=1.0, c_fa=10.0) == pytest.approx(
        min_dcf(labels, llrs, c_miss=1.0, c_fa=10.0), rel=0.2
    )


def test_dcf_at_threshold_monotone_in_threshold():
    labels = np.array([1, 1, 0, 0])
    llrs = np.array([2.0, 0.5, -0.5, -2.0])
    low = dcf_at_threshold(labels, llrs, -10.0)
    mid = dcf_at_threshold(labels, llrs, 0.0)
    high = dcf_at_threshold(labels, llrs, 10.0)
    # Very low threshold: everything called spoof -> all false alarms.
    assert low == pytest.approx(0.95 / 0.05, rel=1e-6)  # (1-p)/min(p, 1-p)
    # Very high threshold: everything called bona fide -> all misses.
    assert high == pytest.approx(0.05 / 0.05, rel=1e-6)
    assert mid < low and mid < high


def test_per_cell_calibrator_fits_per_cell_and_flags_low_support():
    rng = np.random.RandomState(6)
    # Cell A scores are shifted +2 relative to cell B; the fitted maps
    # must differ, and each cell's fitted LLRs must beat the raw scores.
    cells = {}
    for key, shift in (("C1", 2.0), ("C3", -2.0)):
        labels, llrs = _synthetic_llrs(rng, n_spoof=100, n_bona=100, sep=2.0)
        cells[key] = (llrs + shift, labels)
    # One starved cell: 3 trials of each class, below the 20 minimum.
    cells["C7"] = (np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]),
                   np.array([1, 1, 1, 0, 0, 0]))
    cal = PerCellCalibrator().fit(cells)
    assert cal.maps["C1"] != cal.maps["C3"]
    assert "C7" in cal.fallback_cells
    assert cal.maps["C7"] == cal.pooled
    for key, (scores, labels) in cells.items():
        if key == "C7":
            continue
        assert cllr(labels, cal.transform(key, scores)) < cllr(labels, scores)
    # Unknown cells fall back to the pooled map instead of raising.
    assert cal.llr("unknown-cell", 1.0) == pytest.approx(
        cal.pooled[0] * 1.0 + cal.pooled[1]
    )


def test_per_cell_report_narrows_actdcf_spread_after_calibration():
    # Two cells with the same discrimination but opposite score offsets:
    # one global threshold treats them very differently, so the raw
    # actDCF spread is wide. Per-cell calibration must narrow it,
    # mirroring the TDD section 5 acceptance criterion on synthetic data.
    rng = np.random.RandomState(7)
    raw_cells = {}
    cal_cells = {}
    for key, shift in (("C1", 2.5), ("C3", -2.5)):
        labels, llrs = _synthetic_llrs(rng, n_spoof=500, n_bona=500, sep=2.5)
        raw_cells[key] = (labels, llrs + shift)
        cal_cells[key] = (llrs + shift, labels)
    raw = per_cell_report(raw_cells)
    cal = PerCellCalibrator().fit(cal_cells)
    fitted_cells = {
        key: (labels, cal.transform(key, scores))
        for key, (scores, labels) in cal_cells.items()
    }
    fitted = per_cell_report(fitted_cells)
    assert raw["actdcf_spread"] > 0.3
    assert fitted["actdcf_spread"] < raw["actdcf_spread"]
    # ... without raising pooled minDCF (the TDD acceptance criterion).
    # Per-cell maps are monotone within each cell but not globally, so
    # pooled discrimination may improve as the cells align; it must
    # not get worse.
    assert fitted["pooled_min_dcf"] <= raw["pooled_min_dcf"] + 0.02


def test_per_cell_report_rows_and_degenerate_cells():
    rng = np.random.RandomState(8)
    labels, llrs = _synthetic_llrs(rng, n_spoof=50, n_bona=50)
    report = per_cell_report({"C1": (labels, llrs)}, n_boot=20)
    row = report["cells"]["C1"]
    assert row["n"] == 100 and row["n_spoof"] == 50 and row["n_bona"] == 50
    assert 0.0 < row["eer"] < 0.5
    assert 0.0 < row["min_dcf"] < 1.0
    assert row["cllr_min"] <= row["cllr"] + 1e-9
    lo, hi = row["eer_ci"]
    assert lo <= row["eer"] <= hi
    assert report["pooled"]["eer"] == pytest.approx(row["eer"])
    assert report["actdcf_spread"] == pytest.approx(0.0)
    # A single-class cell reports None metrics and is excluded from
    # the spread instead of crashing the report.
    report2 = per_cell_report({
        "C1": (labels, llrs),
        "C2": (np.ones(10, dtype=int), np.zeros(10)),
    })
    assert report2["cells"]["C2"]["eer"] is None
    assert report2["actdcf_spread"] == pytest.approx(0.0)
    assert report2["pooled"]["eer"] is not None
