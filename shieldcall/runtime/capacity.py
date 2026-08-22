"""Capacity and cost arithmetic.

Scale unit is the *call*, not the HTTP request. A worker that spends
``ms_per_frame`` on each 10 ms hop holds::

    calls_per_core = (hop_ms / ms_per_frame) * utilization

GPU is not required. Cost is vCPU-hours plus the (optional) ASR bill,
which this package does not own.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityPlan:
    ms_per_frame: float
    hop_ms: float
    utilization: float
    calls_per_core: float
    cores_for_target: int
    n_plus_1_cores: int
    target_concurrent_calls: int
    usd_per_vcpu_hour: float
    monthly_usd_illustrative: float
    bytes_per_call_sec: int
    notes: str


def plan_capacity(
    ms_per_frame: float,
    *,
    hop_ms: float = 10.0,
    utilization: float = 0.70,
    target_concurrent_calls: int = 1000,
    usd_per_vcpu_hour: float = 0.04,
    hours_per_month: float = 730.0,
    pcm_sr: int = 8000,
    pcm_bytes: int = 2,
) -> CapacityPlan:
    if ms_per_frame <= 0:
        raise ValueError("ms_per_frame must be positive")
    raw = (hop_ms / ms_per_frame) * utilization
    calls_per_core = max(0.1, raw)
    cores = int(-(-target_concurrent_calls // calls_per_core))  # ceil
    n1 = cores + max(1, cores // 10)  # +10% or at least one spare
    monthly = n1 * usd_per_vcpu_hour * hours_per_month
    return CapacityPlan(
        ms_per_frame=float(ms_per_frame),
        hop_ms=float(hop_ms),
        utilization=float(utilization),
        calls_per_core=float(calls_per_core),
        cores_for_target=cores,
        n_plus_1_cores=n1,
        target_concurrent_calls=int(target_concurrent_calls),
        usd_per_vcpu_hour=float(usd_per_vcpu_hour),
        monthly_usd_illustrative=float(monthly),
        bytes_per_call_sec=int(pcm_sr * pcm_bytes),
        notes=(
            "Illustrative CPU cost only (no ASR, no GPU). "
            "usd_per_vcpu_hour is a parameter, not a vendor quote. "
            "Channel-twin simulation is eval-only and must be off in live workers."
        ),
    )
