"""Liveness vs readiness.

Live: process is up (always true if this object exists).
Ready: the worker will accept a *new* call. At capacity it stays live
and becomes unready so a load balancer drains it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Health:
    live: bool
    ready: bool
    active_calls: int
    max_calls: int
    shed_total: int
    asr_breaker: str
    ms_per_frame_ewma: float
    detail: str
