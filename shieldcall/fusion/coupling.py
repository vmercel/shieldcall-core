"""Stage-aligned production change (SAPC).

Question this module answers
----------------------------
PartialSpoof and related work ask whether *any* frame is synthetic.
This module asks a different, narrower question:

    Are acoustic production *change-points* closer to discourse-stage
    *transitions* than they would be if the two point processes were
    independent, given the same margins?

The null is a circular time-shift of the acoustic alarms (preserves
inter-alarm spacing, destroys alignment with stages). The p-value is
the Monte Carlo tail
    p = (1 + #{C_null >= C_obs}) / (B + 1)
which is valid under the shift-null, not under every possible
dependence. If aligned and unaligned handoffs (same mix ratio, only
timing differs) are not separated, the claim is false.

This is not causal identification in Pearl's sense. It is a
permutation test on two observed point processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CouplingResult:
    statistic: float
    p_value: float
    n_stages: int
    n_alarms: int
    score: float  # statistic * (1 - p), in [0, 1]


def gaussian_coupling(
    stage_times: Sequence[float],
    alarm_times: Sequence[float],
    sigma_sec: float = 0.45,
    first_alarm_only: bool = True,
) -> float:
    """Kernel proximity of production change(s) to stage times.

    Default uses the *first* CUSUM alarm only. Frame-level jitter otherwise
    saturates the statistic. C = max_k exp( -(t_k - tau0)^2 / (2 sigma^2) ).
    Empty side -> 0.
    """
    if not stage_times or not alarm_times:
        return 0.0
    stages = np.asarray(stage_times, dtype=np.float64)
    alarms = np.asarray(alarm_times, dtype=np.float64)
    if first_alarm_only:
        alarms = np.array([np.min(alarms)], dtype=np.float64)
    var = 2.0 * sigma_sec * sigma_sec
    d2 = (stages[:, None] - alarms[None, :]) ** 2
    hits = np.exp(-np.min(d2, axis=1) / var)
    return float(np.max(hits) if first_alarm_only else np.mean(hits))


def circular_shift(times: Sequence[float], offset: float, period: float) -> List[float]:
    if period <= 1e-9:
        return [float(t) for t in times]
    return [float((t + offset) % period) for t in times]


def permutation_pvalue(
    stage_times: Sequence[float],
    alarm_times: Sequence[float],
    period: float,
    sigma_sec: float = 0.45,
    n_perm: int = 199,
    seed: int = 0,
    first_alarm_only: bool = True,
) -> CouplingResult:
    """Monte Carlo p-value under circular shifts of the alarm process."""
    obs = gaussian_coupling(
        stage_times, alarm_times, sigma_sec=sigma_sec, first_alarm_only=first_alarm_only
    )
    k = len(stage_times)
    m = len(alarm_times)
    if k == 0 or m == 0 or period <= 1e-9:
        return CouplingResult(statistic=obs, p_value=1.0, n_stages=k, n_alarms=m, score=0.0)
    rng = np.random.RandomState(seed)
    ge = 0
    for _ in range(n_perm):
        off = float(rng.uniform(0.0, period))
        src = [min(alarm_times)] if first_alarm_only else list(alarm_times)
        null_alarms = circular_shift(src, off, period)
        c_null = gaussian_coupling(
            stage_times, null_alarms, sigma_sec=sigma_sec, first_alarm_only=first_alarm_only
        )
        if c_null >= obs - 1e-12:
            ge += 1
    p = (1.0 + ge) / (n_perm + 1.0)
    score = float(np.clip(obs * (1.0 - p), 0.0, 1.0))
    return CouplingResult(statistic=float(obs), p_value=float(p), n_stages=k, n_alarms=m, score=score)


class StageAlignedCoupling:
    """Streaming buffers + on-demand permutation test."""

    def __init__(
        self,
        sigma_sec: float = 0.45,
        n_perm: int = 199,
        seed: int = 0,
        stage_allowlist: Tuple[str, ...] = ("HARVEST", "PAYMENT", "THREAT", "SECRECY"),
    ):
        self.sigma_sec = sigma_sec
        self.n_perm = n_perm
        self.seed = seed
        self.stage_allowlist = stage_allowlist
        self.stage_times: List[float] = []
        self._last_stage: str = ""
        self.alarm_times: List[float] = []
        self._tmax = 0.0

    def reset(self) -> None:
        self.stage_times = []
        self._last_stage = ""
        self.alarm_times = []
        self._tmax = 0.0

    def observe_stage(self, stage: str, timestamp_sec: float) -> None:
        self._tmax = max(self._tmax, float(timestamp_sec))
        if not stage:
            return
        if stage == self._last_stage:
            return
        # Ignore the first emission so the opening GREETING does not
        # couple to burn-in CUSUM noise.
        prev = self._last_stage
        self._last_stage = stage
        if prev == "":
            return
        if self.stage_allowlist and stage not in self.stage_allowlist:
            return
        self.stage_times.append(float(timestamp_sec))

    def observe_alarm(self, timestamp_sec: float) -> None:
        self._tmax = max(self._tmax, float(timestamp_sec))
        self.alarm_times.append(float(timestamp_sec))

    def evaluate(self) -> CouplingResult:
        period = max(self._tmax, 1e-3)
        return permutation_pvalue(
            self.stage_times,
            self.alarm_times,
            period=period,
            sigma_sec=self.sigma_sec,
            n_perm=self.n_perm,
            seed=self.seed,
            first_alarm_only=True,
        )
