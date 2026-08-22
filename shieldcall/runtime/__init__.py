"""Production runtime contracts for the research prototype.

ShieldCall is a sidecar, not a media hairpin. This package implements
the in-process seams a CPaaS worker would actually use:

- one isolated session per call (pipeline + agent)
- admission control / shed under concurrency limits
- fail-open on the telephone call (no interrupt when unhealthy)
- fail-closed on actuation (shed sessions only MONITOR)
- circuit breaker around ASR
- health / readiness
- capacity math from measured milliseconds per frame

Nothing here is a carrier deployment. The tests check the contracts.
"""

from .breaker import CircuitBreaker, BreakerState
from .capacity import CapacityPlan, plan_capacity
from .health import Health
from .runtime import SidecarRuntime, ShedError
from .session import CallSession, SessionEvent
from .slo import SLO

__all__ = [
    "BreakerState",
    "CallSession",
    "CapacityPlan",
    "CircuitBreaker",
    "Health",
    "SLO",
    "SessionEvent",
    "ShedError",
    "SidecarRuntime",
    "plan_capacity",
]
