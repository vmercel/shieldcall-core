"""Guard the invariants that docs/HOSTED_INFERENCE_DESIGN.md relies on.

These tests assert internal consistency of the capacity arithmetic and
that the SLO constants the design doc cites have not drifted. They do
not hardcode the doc's tables: if the code changes, the doc author
re-runs plan_capacity and updates the tables.
"""

from shieldcall.runtime.capacity import plan_capacity
from shieldcall.runtime.slo import SLO


def test_slo_constants_cited_by_design_doc():
    slo = SLO()
    assert slo.hop_ms == 10.0
    assert slo.frame_budget_ms == 8.0
    assert slo.realtime_min_x == 1.25
    assert slo.detector_ready_min_free == 0.10
    assert slo.asr_fail_threshold == 5
    assert slo.asr_reset_sec == 30.0


def test_capacity_arithmetic_internal_consistency():
    # Measured p95 on the engineering VM (see design doc section 3.1).
    for target in (100, 1000, 10000):
        p = plan_capacity(4.691, target_concurrent_calls=target)
        # The pool must actually hold the target load.
        assert p.calls_per_core * p.cores_for_target >= target
        # N+1: at least one spare, at least +10% for larger pools.
        assert p.n_plus_1_cores >= p.cores_for_target + 1
        # Cost arithmetic is exactly what the doc claims it is.
        assert p.monthly_usd_illustrative == p.n_plus_1_cores * p.usd_per_vcpu_hour * 730.0
        # Bandwidth assumption: 8 kHz 16-bit mono.
        assert p.bytes_per_call_sec == 16000


def test_capacity_scales_linearly_with_load():
    p100 = plan_capacity(4.691, target_concurrent_calls=100)
    p1000 = plan_capacity(4.691, target_concurrent_calls=1000)
    assert p1000.cores_for_target >= 9 * p100.cores_for_target
    assert p1000.cores_for_target <= 11 * p100.cores_for_target


def test_faster_frames_need_fewer_cores():
    slow = plan_capacity(8.0, target_concurrent_calls=1000)
    fast = plan_capacity(4.0, target_concurrent_calls=1000)
    assert fast.cores_for_target < slow.cores_for_target
