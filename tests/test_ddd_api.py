"""DDD application + newly published detector endpoints."""

from __future__ import annotations

import base64

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from shieldcall.pipeline import PipelineConfig
from shieldcall.serve.http_app import create_app
from shieldcall.runtime.runtime import SidecarRuntime


def _client():
    cfg = PipelineConfig(channel=None)
    rt = SidecarRuntime(max_calls=4, pipeline_config=cfg)
    return TestClient(create_app(rt)), rt


def test_capabilities_lists_every_endpoint():
    c, _ = _client()
    r = c.get("/v1/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert body["fail_open"] is True
    assert body["hang_up"] is False
    assert body["actuation"] == "recommend_only"
    joined = " ".join(body["endpoints"])
    for path in (
        "/health",
        "/ready",
        "/v1/calls",
        "/v1/calls/{id}/chunk",
        "/v1/score/linguistic",
        "/v1/score/acoustic",
        "/v1/score/fuse",
        "/openapi.json",
    ):
        assert path in joined


def test_ready_and_list_and_get_call():
    c, _ = _client()
    assert c.get("/ready").status_code == 200
    opened = c.post("/v1/calls", json={"call_id": "ddd-1"}).json()
    listed = c.get("/v1/calls").json()["calls"]
    assert any(row["call_id"] == opened["call_id"] for row in listed)
    got = c.get(f"/v1/calls/{opened['call_id']}")
    assert got.status_code == 200
    assert got.json()["last_action"] in {"monitor", "warn", "challenge", "escalate", "abstain", "adapt"}


def test_chunk_and_decision():
    c, _ = _client()
    cid = c.post("/v1/calls", json={"call_id": "ddd-chunk"}).json()["call_id"]
    ev = c.post(
        f"/v1/calls/{cid}/chunk",
        json={"t": 1.2, "text": "This is the IRS. Buy gift cards and wire the money now."},
    )
    assert ev.status_code == 200
    body = ev.json()
    assert body["type"] == "event"
    assert body["fraud"] >= 0.2
    dec = c.get(f"/v1/calls/{cid}/decision")
    assert dec.status_code == 200
    assert dec.json()["call_id"] == cid


def test_stateless_score_linguistic_and_fuse():
    c, _ = _client()
    li = c.post("/v1/score/linguistic", json={"text": "Your grandson is in jail. Send bitcoin immediately."})
    assert li.status_code == 200
    assert li.json()["fraud"] > 0.2
    fused = c.post("/v1/score/fuse", json={"fraud": 0.8, "synth": 0.1})
    assert fused.status_code == 200
    assert fused.json()["risk"] > 0.4
    assert "regime" in fused.json()


def test_stateless_score_acoustic():
    c, _ = _client()
    pcm = (np.zeros(1600, dtype=np.int16)).tobytes()
    b64 = base64.b64encode(pcm).decode("ascii")
    r = c.post("/v1/score/acoustic", json={"sr": 8000, "pcm_s16le_b64": b64})
    assert r.status_code == 200
    assert "synth" in r.json()


def test_unknown_call_is_404_not_hangup():
    c, _ = _client()
    assert c.post("/v1/calls/missing/chunk", json={"text": "hello"}).status_code == 404
    assert c.get("/v1/calls/missing").status_code == 404
