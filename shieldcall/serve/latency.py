"""Per-route latency histograms and 4xx/5xx rate export (P2-6c, Phase 1).

The hosted-inference design doc (section 6) asks for per-route latency
histograms and 4xx/5xx rates exported from /health. This module
implements them with no new dependencies:

- ``RouteLatencyTracker``: thread-safe fixed-bucket cumulative
  histograms keyed by ``"METHOD /route/template"``. Route TEMPLATES are
  used (``/v1/calls/{call_id}``), never concrete paths, so a flood of
  distinct call ids cannot blow up metric cardinality.
- Percentiles are ESTIMATED from bucket midpoints (``p50_ms_est``,
  ``p95_ms_est``) and labelled as estimates. They are operational
  signals, not measurement instruments.
- Buckets are overridable via ``SHIELDCALL_LATENCY_BUCKETS="1,5,25,..."``
  (milliseconds). Empty or invalid values fall back to the default
  ladder.
- Status-class counts (2xx/4xx/5xx/other) per route give the error-rate
  half of the contract; 2xx is kept so rates are computable.

Websocket connections are intentionally not tracked: they are
long-lived, so a duration histogram would be dominated by idle hold
time rather than request latency.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Tuple

# Default bucket edges in milliseconds: fine near the 8 ms frame budget,
# coarser further out. "+Inf" is always implicit as the final bucket.
DEFAULT_BUCKETS_MS: Tuple[float, ...] = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)


def buckets_from_env(env_var: str = "SHIELDCALL_LATENCY_BUCKETS") -> Tuple[float, ...]:
    """Parse bucket edges from the environment; default ladder on bad input."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return DEFAULT_BUCKETS_MS
    try:
        edges = sorted({float(p) for p in raw.split(",") if p.strip()})
    except ValueError:
        return DEFAULT_BUCKETS_MS
    if len(edges) < 2 or any(e <= 0 for e in edges):
        return DEFAULT_BUCKETS_MS
    return tuple(edges)


def _status_class(status_code: int) -> str:
    if 200 <= status_code < 300:
        return "2xx"
    if 400 <= status_code < 500:
        return "4xx"
    if 500 <= status_code < 600:
        return "5xx"
    return "other"


class RouteLatencyTracker:
    """Thread-safe cumulative per-route latency histograms."""

    def __init__(self, buckets: Tuple[float, ...] = DEFAULT_BUCKETS_MS):
        self._buckets = tuple(buckets)
        self._lock = threading.Lock()
        # route -> {"count", "sum_ms", "bucket_counts", "status": {"2xx","4xx","5xx","other"}}
        self._routes: Dict[str, Dict[str, Any]] = {}

    @property
    def buckets(self) -> Tuple[float, ...]:
        return self._buckets

    def _entry(self, route: str) -> Dict[str, Any]:
        entry = self._routes.get(route)
        if entry is None:
            entry = {
                "count": 0,
                "sum_ms": 0.0,
                "bucket_counts": [0] * len(self._buckets),
                "status": {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0},
            }
            self._routes[route] = entry
        return entry

    def record(self, route: str, elapsed_ms: float, status_code: int) -> None:
        """Record one completed HTTP exchange for a route template key.

        Buckets are CUMULATIVE (Prometheus convention): an observation
        increments every bucket whose edge is at or above the value.
        """
        if elapsed_ms < 0:
            elapsed_ms = 0.0
        with self._lock:
            entry = self._entry(route)
            entry["count"] += 1
            entry["sum_ms"] += elapsed_ms
            counts = entry["bucket_counts"]
            for i, edge in enumerate(self._buckets):
                if elapsed_ms <= edge:
                    counts[i] += 1
            entry["status"][_status_class(status_code)] += 1

    def _percentile_est(self, cumulative_counts: List[int], total: int, quantile: float) -> float:
        """Estimate a percentile from bucket midpoints (conservative input
        guard: returns 0.0 on empty data). Expects CUMULATIVE counts."""
        if total <= 0:
            return 0.0
        target = quantile * total
        lower = 0.0
        for edge, cumulative in zip(self._buckets, cumulative_counts):
            if cumulative >= target:
                return (lower + edge) / 2.0
            lower = edge
        # Everything fell past the last edge: estimate at twice the top edge.
        return self._buckets[-1] * 2.0

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """JSON-safe snapshot: per-route counts, cumulative buckets,
        mean, p50/p95 estimates, and status-class counts."""
        with self._lock:
            out: Dict[str, Dict[str, Any]] = {}
            for route, entry in self._routes.items():
                count = entry["count"]
                buckets = {str(edge): c for edge, c in zip(self._buckets, entry["bucket_counts"])}
                buckets["+Inf"] = count
                out[route] = {
                    "count": count,
                    "sum_ms": round(entry["sum_ms"], 3),
                    "mean_ms": round(entry["sum_ms"] / count, 3) if count else 0.0,
                    "buckets": buckets,
                    "p50_ms_est": round(self._percentile_est(entry["bucket_counts"], count, 0.50), 3),
                    "p95_ms_est": round(self._percentile_est(entry["bucket_counts"], count, 0.95), 3),
                    "status": dict(entry["status"]),
                }
            return out

    def reset(self) -> None:
        """Drop all recorded data (used by tests)."""
        with self._lock:
            self._routes.clear()
