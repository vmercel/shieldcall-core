"""Per-route latency histograms + 4xx/5xx rates (P2-6c, Phase 1).

Covers the tracker unit contract (cumulative buckets, status classes,
percentile estimates, env bucket override) and the end-to-end wiring:
the middleware keys by route TEMPLATE (not concrete call ids) and
/health exports the snapshot.
"""

from __future__ import annotations

import threading

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve import latency
from shieldcall.serve.http_app import create_app
from shieldcall.serve.latency import RouteLatencyTracker, buckets_from_env


def _client():
    cfg = PipelineConfig(channel=None)
    rt = SidecarRuntime(max_calls=4, pipeline_config=cfg)
    return TestClient(create_app(rt))


# --- Unit contract ----------------------------------------------------


def test_buckets_are_cumulative():
    t = RouteLatencyTracker(buckets=(1.0, 5.0, 10.0))
    t.record("GET /x", 3.0, 200)
    snap = t.snapshot()["GET /x"]
    assert snap["buckets"] == {"1.0": 0, "5.0": 1, "10.0": 1, "+Inf": 1}
    assert snap["count"] == 1
    assert snap["status"] == {"2xx": 1, "4xx": 0, "5xx": 0, "other": 0}


def test_observation_above_top_edge_still_counts():
    t = RouteLatencyTracker(buckets=(1.0, 5.0))
    t.record("GET /x", 120.0, 200)
    snap = t.snapshot()["GET /x"]
    assert snap["buckets"] == {"1.0": 0, "5.0": 0, "+Inf": 1}
    # p95 estimate degrades honestly past the ladder top.
    assert snap["p95_ms_est"] == 10.0


def test_status_classes_split():
    t = RouteLatencyTracker(buckets=(1.0, 5.0))
    t.record("GET /x", 0.5, 200)
    t.record("GET /x", 0.5, 201)
    t.record("GET /x", 0.5, 404)
    t.record("GET /x", 0.5, 429)
    t.record("GET /x", 0.5, 500)
    t.record("GET /x", 0.5, 101)
    status = t.snapshot()["GET /x"]["status"]
    assert status == {"2xx": 2, "4xx": 2, "5xx": 1, "other": 1}


def test_percentile_estimates_use_bucket_midpoints():
    t = RouteLatencyTracker(buckets=(10.0, 20.0))
    for _ in range(9):
        t.record("GET /x", 5.0, 200)  # lands in the 10 ms bucket
    t.record("GET /x", 15.0, 200)  # lands in the 20 ms bucket
    snap = t.snapshot()["GET /x"]
    # p50: 9 of 10 obs under the 10 ms edge -> midpoint of [0, 10].
    assert snap["p50_ms_est"] == 5.0
    # p95: first cumulative count reaching 9.5 is the 20 ms bucket.
    assert snap["p95_ms_est"] == 15.0
    # mean is exact, not estimated.
    assert snap["mean_ms"] == pytest.approx(6.0)


def test_empty_tracker_snapshots_empty():
    assert RouteLatencyTracker().snapshot() == {}


def test_env_bucket_override(monkeypatch):
    monkeypatch.setenv("SHIELDCALL_LATENCY_BUCKETS", "2,4,8")
    assert buckets_from_env() == (2.0, 4.0, 8.0)


def test_env_bucket_override_rejects_garbage(monkeypatch):
    monkeypatch.setenv("SHIELDCALL_LATENCY_BUCKETS", "abc")
    assert buckets_from_env() == latency.DEFAULT_BUCKETS_MS
    monkeypatch.setenv("SHIELDCALL_LATENCY_BUCKETS", "-3,0")
    assert buckets_from_env() == latency.DEFAULT_BUCKETS_MS
    monkeypatch.setenv("SHIELDCALL_LATENCY_BUCKETS", "")
    assert buckets_from_env() == latency.DEFAULT_BUCKETS_MS


def test_record_is_thread_safe():
    t = RouteLatencyTracker()
    threads = [
        threading.Thread(target=lambda: [t.record("GET /x", 3.0, 200) for _ in range(100)])
        for _ in range(8)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    snap = t.snapshot()["GET /x"]
    assert snap["count"] == 800
    assert snap["buckets"]["+Inf"] == 800


# --- Wiring -----------------------------------------------------------


def test_health_exports_per_route_template():
    c = _client()
    r = c.post("/v1/calls", json={})
    assert r.status_code == 200
    cid = r.json()["call_id"]
    r = c.get(f"/v1/calls/{cid}/decision")
    assert r.status_code == 200
    body = c.get("/health").json()
    route_latency = body["route_latency"]
    # Keyed by TEMPLATE: two concrete decision URLs share one entry.
    open_entry = route_latency["POST /v1/calls"]
    assert open_entry["count"] >= 1
    assert open_entry["status"]["2xx"] >= 1
    assert open_entry["buckets"]["+Inf"] == open_entry["count"]
    assert open_entry["mean_ms"] >= 0
    assert open_entry["p50_ms_est"] >= 0
    assert open_entry["p95_ms_est"] >= open_entry["p50_ms_est"]
    assert "GET /v1/calls/{call_id}/decision" in route_latency


def test_unmatched_paths_recorded_without_cardinality():
    c = _client()
    c.get("/v1/does-not-exist")
    c.get("/v1/also-missing")
    entry = c.get("/health").json()["route_latency"]["GET unmatched"]
    assert entry["count"] == 2
    assert entry["status"]["4xx"] == 2


def test_http_errors_are_recorded_as_4xx():
    c = _client()
    # Unknown script -> 404 on the /inject route; the call must exist first.
    cid = c.post("/v1/calls", json={}).json()["call_id"]
    r = c.post(f"/v1/calls/{cid}/inject", json={"script_id": "nope"})
    assert r.status_code == 404
    entry = c.get("/health").json()["route_latency"]["POST /v1/calls/{call_id}/inject"]
    assert entry["status"]["4xx"] >= 1


def test_snapshot_is_json_safe():
    c = _client()
    c.get("/health")
    body = c.get("/health").json()  # must not raise on serialisation
    for route, entry in body["route_latency"].items():
        assert set(entry) == {
            "count",
            "sum_ms",
            "mean_ms",
            "buckets",
            "p50_ms_est",
            "p95_ms_est",
            "status",
        }
        # Buckets cumulative: monotone non-decreasing up to +Inf == count.
        values = [entry["buckets"][str(e)] for e in latency.DEFAULT_BUCKETS_MS]
        assert values == sorted(values)
        assert entry["buckets"]["+Inf"] == entry["count"]
