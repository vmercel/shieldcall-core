"""One call, one session. State never crosses call_id."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..agent.agent import Decision, DefenseAgent
from ..agent.hypotheses import Action
from ..pipeline import PipelineConfig, PipelineEvent, ShieldCallPipeline
from .breaker import CircuitBreaker


@dataclass
class SessionEvent:
    pipeline: Optional[PipelineEvent]
    decision: Optional[Decision]
    shed: bool
    asr_bypassed: bool


@dataclass
class CallSession:
    call_id: str
    pipeline: ShieldCallPipeline
    agent: DefenseAgent
    breaker: CircuitBreaker
    shed: bool = False
    frames: int = 0
    decisions: int = 0
    last_action: Action = Action.MONITOR
    closed: bool = False
    _trace: List[Decision] = field(default_factory=list)

    def push_audio(self, samples: np.ndarray, sample_rate: int) -> List[SessionEvent]:
        if self.closed:
            raise RuntimeError(f"session {self.call_id} is closed")
        if self.shed:
            # Fail-open: do not spend CPU, do not interrupt the call.
            return [SessionEvent(None, None, shed=True, asr_bypassed=True)]
        asr_ok = self.breaker.allow()
        if not asr_ok:
            # Freeze linguistic input; acoustic still runs via pipeline,
            # but we skip injecting new text. Audio ASR is gated below.
            pass
        events: List[SessionEvent] = []
        for ev in self.pipeline.push_audio(samples, sample_rate):
            self.frames += 1
            decision = None
            asr_bypassed = not asr_ok
            if ev.risk is not None:
                perc = self.agent.perceive_risk(ev.risk)
                decision = self.agent.step(perc)
                self.last_action = decision.action
                self.decisions += 1
                self._trace.append(decision)
            events.append(SessionEvent(ev, decision, shed=False, asr_bypassed=asr_bypassed))
        return events

    def push_transcript(self, text: str, timestamp_sec: float) -> None:
        if self.closed or self.shed:
            return
        if not self.breaker.allow():
            self.breaker.record_failure()
            return
        try:
            self.pipeline.push_transcript(text, timestamp_sec)
            self.breaker.record_success()
        except Exception:
            self.breaker.record_failure()
            raise

    def close(self) -> List[Decision]:
        self.closed = True
        return list(self._trace)
