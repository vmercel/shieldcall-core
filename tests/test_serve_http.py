"""HTTP sidecar contracts: TCT off, traces have no nonce, inject works."""

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


def test_health_and_channel_twin_off():
    c, rt = _client()
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["live"] is True
    assert body["channel_twin"] is False
    assert body["actuation"] == "recommend_only"
    assert rt.pipeline_config.channel is None


def test_open_inject_dentist_and_scam():
    c, _ = _client()
    cid = c.post("/v1/calls", json={"call_id": "lab-test"}).json()["call_id"]
    dent = c.post(f"/v1/calls/{cid}/inject", json={"script_id": "ind_b01"})
    assert dent.status_code == 200
    scam = c.post(f"/v1/calls/{cid}/inject", json={"script_id": "ind_s01"})
    assert scam.status_code == 200
    assert scam.json()["fraud"] >= dent.json()["fraud"]
    tr = c.get(f"/v1/calls/{cid}/trace")
    assert tr.status_code == 200
    for row in tr.json()["trace"]:
        assert "nonce" not in row.get("tool", {})
        assert "nonce" not in str(row.get("tool", {})).lower()


def test_audio_roundtrip_and_404():
    c, _ = _client()
    assert c.post("/v1/calls/nope/audio", json={"sr": 8000, "pcm_s16le_b64": "AAA="}).status_code == 404
    cid = c.post("/v1/calls", json={}).json()["call_id"]
    pcm = (np.zeros(1600, dtype=np.int16)).tobytes()
    b64 = base64.b64encode(pcm).decode("ascii")
    r = c.post(f"/v1/calls/{cid}/audio", json={"sr": 8000, "pcm_s16le_b64": b64})
    assert r.status_code == 200
    assert r.json()["action"] in {"monitor", "warn", "challenge", "escalate", "abstain", "adapt"}
    c.delete(f"/v1/calls/{cid}")


def test_lab_home_and_scripts():
    c, _ = _client()
    home = c.get("/")
    assert home.status_code == 200
    assert b"ShieldCall Detector Lab" in home.content
    scripts = c.get("/v1/scripts")
    assert scripts.status_code == 200
    ids = {s["id"] for s in scripts.json()["scripts"]}
    assert "ind_b01" in ids
    assert "ind_s01" in ids


def test_websocket_transcript():
    c, _ = _client()
    cid = c.post("/v1/calls", json={"call_id": "ws1"}).json()["call_id"]
    with c.websocket_connect(f"/v1/calls/{cid}/stream") as ws:
        ws.send_json({"type": "transcript", "t": 0.5, "text": "Hello, this is a reminder that you have a cleaning tomorrow."})
        ev = ws.receive_json()
        assert ev["type"] == "event"
        assert "action" in ev
