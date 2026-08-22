"""Circuit breaker for the ASR dependency.

Linguistic scoring needs transcripts. Acoustic scoring does not.
When ASR fails or times out, the breaker opens and the worker keeps
the residual path; the agent sees fraud_prob freeze and can still act
on synth / SAPC / coverage gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    fail_threshold: int = 5
    reset_after_sec: float = 30.0
    failures: int = 0
    state: BreakerState = BreakerState.CLOSED
    opened_at: float = 0.0
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)

    def allow(self) -> bool:
        if self.state is BreakerState.CLOSED:
            return True
        if self.state is BreakerState.OPEN:
            if self.clock() - self.opened_at >= self.reset_after_sec:
                self.state = BreakerState.HALF_OPEN
                return True
            return False
        return True  # half-open: one trial

    def record_success(self) -> None:
        self.failures = 0
        self.state = BreakerState.CLOSED

    def record_failure(self) -> None:
        self.failures += 1
        if self.state is BreakerState.HALF_OPEN or self.failures >= self.fail_threshold:
            self.state = BreakerState.OPEN
            self.opened_at = self.clock()
