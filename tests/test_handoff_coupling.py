"""SAPC and CUSUM: timing tests that do not need recorded speech."""

import numpy as np

from shieldcall.acoustic.changepoint import StreamingCUSUM, estimate_mean_shift_time
from shieldcall.fusion.aci import AdaptiveConformal
from shieldcall.fusion.coupling import gaussian_coupling, permutation_pvalue, StageAlignedCoupling


def test_cusum_fires_after_downward_shift():
    c = StreamingCUSUM(k=0.1, h=0.5, burn_in=10, ignore_first_sec=0.0, min_alarm_gap_sec=0.2)
    t = 0.0
    for _ in range(15):
        c.update(0.85, t)
        t += 0.05
    fired = None
    for _ in range(40):
        al = c.update(0.15, t)
        if al is not None and fired is None:
            fired = al.timestamp_sec
        t += 0.05
    assert fired is not None


def test_cusum_fires_after_upward_shift():
    c = StreamingCUSUM(k=0.1, h=0.5, burn_in=10, ignore_first_sec=0.0, min_alarm_gap_sec=0.2)
    t = 0.0
    for _ in range(15):
        c.update(0.15, t)
        t += 0.05
    fired = None
    for _ in range(40):
        al = c.update(0.85, t)
        if al is not None and fired is None:
            fired = al.timestamp_sec
        t += 0.05
    assert fired is not None
    assert fired >= 15 * 0.05 - 1e-6


def test_coupling_higher_when_aligned():
    stages = [1.0, 2.0, 3.0]
    aligned = [1.05, 2.02, 2.95]
    unaligned = [0.1, 1.55, 2.4]
    c_a = gaussian_coupling(stages, aligned, sigma_sec=0.3, first_alarm_only=False)
    c_u = gaussian_coupling(stages, unaligned, sigma_sec=0.3, first_alarm_only=False)
    assert c_a > c_u


def test_permutation_aligned_has_smaller_p():
    stages = [1.0, 2.0, 3.0, 4.0]
    alarms = [1.02, 2.05, 3.0, 3.97]
    aligned = permutation_pvalue(stages, alarms, period=5.0, sigma_sec=0.3, n_perm=99, seed=0)
    shifted = permutation_pvalue(stages, [0.4, 1.3, 2.7, 4.4], period=5.0, sigma_sec=0.3, n_perm=99, seed=1)
    assert aligned.statistic > shifted.statistic
    assert aligned.p_value <= shifted.p_value + 1e-9
    assert aligned.score > shifted.score


def test_empty_processes_score_zero():
    sapc = StageAlignedCoupling()
    sapc.observe_stage("GREETING", 0.1)
    r = sapc.evaluate()
    assert r.score == 0.0
    assert r.p_value == 1.0


def test_mean_shift_localizes_step():
    times = [i * 0.01 for i in range(80)]
    vals = [0.2] * 40 + [0.9] * 40
    loc = estimate_mean_shift_time(vals, times, win=8)
    assert loc is not None
    assert 0.30 < loc[0] < 0.50


def test_aci_updates_and_reports_coverage():
    aci = AdaptiveConformal(alpha=0.1, gamma=0.05, window=200)
    rng = np.random.RandomState(0)
    for i in range(300):
        y = float(rng.rand() > 0.5)
        score = float(np.clip(y * 0.7 + 0.15 + 0.05 * rng.randn(), 0, 1))
        aci.observe(score, y)
    cov = aci.empirical_coverage()
    assert 0.5 <= cov <= 1.0
    assert len(aci.errors) == 300
    assert aci.alpha_min <= aci.alpha_t <= aci.alpha_max
