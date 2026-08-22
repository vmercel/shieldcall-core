"""Runtime contracts: isolation, shed/fail-open, breaker, capacity math."""

from __future__ import annotations

import numpy as np
import pytest

from shieldcall.agent.hypotheses import Action
from shieldcall.runtime.breaker import BreakerState, CircuitBreaker
from shieldcall.runtime.capacity import plan_capacity
from shieldcall.runtime.runtime import ShedError, SidecarRuntime
from shieldcall.runtime.slo import SLO


def _tone(seconds=0.3, sr=8000):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 170 * t)).astype(np.float32), sr


def test_sessions_are_isolated():
    rt = SidecarRuntime(max_calls=4)
    a = rt.open_call("call-a")
    b = rt.open_call("call-b")
    assert a.pipeline is not b.pipeline
    assert a.agent is not b.agent
    a.push_transcript("verify your social security number now", 0.5)
    b.push_transcript("your dentist appointment is tomorrow", 0.5)
    audio, sr = _tone()
    a.push_audio(audio, sr)
    b.push_audio(audio, sr)
    assert a.agent.belief.p is not b.agent.belief.p
    # A heard a harvest-ish script; B did not share that state.
    assert a.agent.belief.p != b.agent.belief.p


def test_shed_is_fail_open_and_skips_compute():
    rt = SidecarRuntime(max_calls=1)
    a = rt.open_call("one")
    assert a.shed is False
    b = rt.open_call("two")
    assert b.shed is True
    audio, sr = _tone()
    out = b.push_audio(audio, sr)
    assert out[0].shed is True
    assert b.frames == 0
    assert b.last_action is Action.MONITOR
    h = rt.health()
    assert h.live is True
    assert h.ready is False
    assert h.shed_total >= 1


def test_hard_reject_raises():
    rt = SidecarRuntime(max_calls=1)
    rt.open_call("one")
    with pytest.raises(ShedError):
        rt.open_call("two", hard_reject=True)


def test_close_frees_capacity():
    rt = SidecarRuntime(max_calls=1)
    rt.open_call("one")
    rt.close_call("one")
    c = rt.open_call("three")
    assert c.shed is False
    assert rt.health().ready is False  # 1/1 used, unready for *new* calls


def test_breaker_opens_and_half_opens():
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    br = CircuitBreaker(fail_threshold=3, reset_after_sec=10.0, clock=now)
    for _ in range(3):
        br.record_failure()
    assert br.state is BreakerState.OPEN
    assert br.allow() is False
    clock["t"] = 11.0
    assert br.allow() is True
    assert br.state is BreakerState.HALF_OPEN
    br.record_success()
    assert br.state is BreakerState.CLOSED


def test_capacity_formula():
    plan = plan_capacity(5.0, hop_ms=10.0, utilization=0.7, target_concurrent_calls=100)
    # (10/5)*0.7 = 1.4 calls/core; 100/1.4 ≈ 72 cores
    assert plan.calls_per_core == pytest.approx(1.4)
    assert plan.cores_for_target == 72
    assert plan.n_plus_1_cores == 72 + 7
    assert plan.bytes_per_call_sec == 16000
    assert plan.monthly_usd_illustrative > 0


def test_slo_frame_budget():
    slo = SLO(frame_budget_ms=8.0)
    assert slo.within_frame_budget(6.3)
    assert not slo.within_frame_budget(9.0)


def test_over_budget_marks_unready():
    rt = SidecarRuntime(max_calls=8)
    rt.open_call("a")
    assert rt.health().ready is True
    rt.observe_frame_ms(20.0)
    assert rt.health().ready is False
    assert "frame time" in rt.health().detail
