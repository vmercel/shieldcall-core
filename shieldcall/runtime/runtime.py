"""Sidecar worker: admission, isolation, health, fail-open shed.

A load balancer pins a call_id to one worker (affinity). This object
is that worker. It never hairpins RTP. It never shares a pipeline
across two call_ids.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from ..agent.agent import DefenseAgent
from ..linguistic.asr_bridge import ASRBridge, PassthroughASR
from ..pipeline import PipelineConfig, ShieldCallPipeline
from .asr_gate import GatedASR
from .breaker import CircuitBreaker
from .health import Health
from .session import CallSession
from .slo import SLO


class ShedError(Exception):
    """Raised only if the caller asked for a hard reject instead of a shed session."""


class SidecarRuntime:
    def __init__(
        self,
        *,
        max_calls: int = 32,
        slo: Optional[SLO] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        asr: Optional[ASRBridge] = None,
        ewma_alpha: float = 0.2,
    ):
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        self.max_calls = max_calls
        self.slo = slo or SLO()
        self.pipeline_config = pipeline_config or PipelineConfig()
        self._asr = asr
        self._sessions: Dict[str, CallSession] = {}
        self._lock = threading.RLock()
        self._shed_total = 0
        self._ms_ewma = 0.0
        self._alpha = ewma_alpha
        self.breaker = CircuitBreaker(
            fail_threshold=self.slo.asr_fail_threshold,
            reset_after_sec=self.slo.asr_reset_sec,
        )

    def open_call(self, call_id: str, *, hard_reject: bool = False) -> CallSession:
        with self._lock:
            existing = self._sessions.get(call_id)
            if existing is not None and not existing.closed:
                return existing
            if len(self._active()) >= self.max_calls:
                self._shed_total += 1
                if hard_reject:
                    raise ShedError(f"at capacity ({self.max_calls}); fail-open")
                # Shed session: same API, zero compute, MONITOR-only.
                shed = CallSession(
                    call_id=call_id,
                    pipeline=ShieldCallPipeline(self.pipeline_config, asr=self._asr),
                    agent=DefenseAgent(),
                    breaker=self.breaker,
                    shed=True,
                )
                self._sessions[call_id] = shed
                return shed
            inner = self._asr or PassthroughASR()
            pipe = ShieldCallPipeline(self.pipeline_config, asr=GatedASR(inner, self.breaker))
            sess = CallSession(
                call_id=call_id,
                pipeline=pipe,
                agent=DefenseAgent(),
                breaker=self.breaker,
            )
            self._sessions[call_id] = sess
            return sess

    def close_call(self, call_id: str):
        with self._lock:
            sess = self._sessions.pop(call_id, None)
            if sess is None:
                return []
            return sess.close()

    def observe_frame_ms(self, ms: float) -> None:
        if self._ms_ewma <= 0:
            self._ms_ewma = ms
        else:
            self._ms_ewma = self._alpha * ms + (1.0 - self._alpha) * self._ms_ewma

    def health(self) -> Health:
        with self._lock:
            active = len(self._active())
            free = 1.0 - (active / self.max_calls)
            ready = free >= self.slo.detector_ready_min_free
            over_budget = (
                self._ms_ewma > 0 and not self.slo.within_frame_budget(self._ms_ewma)
            )
            if over_budget:
                ready = False
            detail = "ok"
            if active >= self.max_calls:
                detail = "at capacity; shedding new calls (fail-open)"
            elif over_budget:
                detail = "frame time above SLO; unready"
            return Health(
                live=True,
                ready=ready,
                active_calls=active,
                max_calls=self.max_calls,
                shed_total=self._shed_total,
                asr_breaker=self.breaker.state.value,
                ms_per_frame_ewma=self._ms_ewma,
                detail=detail,
            )

    def _active(self) -> Dict[str, CallSession]:
        return {k: s for k, s in self._sessions.items() if not s.closed and not s.shed}
