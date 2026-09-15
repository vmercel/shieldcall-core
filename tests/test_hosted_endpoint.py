"""P2-5: prototype hosted endpoint. Feature flag, fail-closed auth, quota, analyze."""

from __future__ import annotations

import base64

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from shieldcall.eval.corpora.independent_scripts import independent_scripts
from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve import hosted
from shieldcall.serve.http_app import create_app


def _rt():
    cfg = PipelineConfig(channel=None)
    return SidecarRuntime(max_calls=4, pipeline_config=cfg)


def _clear_hosted_env(monkeypatch):
    for k in (
        "SHIELDCALL_HOSTED_ENDPOINT",
        "SHIELDCALL_SIDECAR_TOKEN",
        "SHIELDCALL_SIDECAR_TOKENS",
        "SHIELDCALL_HOSTED_QUOTA_PER_MIN",
    ):
        monkeypatch.delenv(k, raising=False)


def _script_turns(script_id):
    for s in independent_scripts():
        if s.script_id == script_id:
            return [{"t": float(i) * 0.8, "text": text} for i, (_, text) in enumerate(s.turns)]
    raise AssertionError(f"unknown script {script_id}")


def test_hosted_routes_absent_when_flag_off(monkeypatch):
    _clear_hosted_env(monkeypatch)
    c = TestClient(create_app(_rt()))
    assert c.get("/hosted/v1/status").status_code == 404
    assert c.post("/hosted/v1/analyze", json={}).status_code == 404


def test_fail_closed_when_flag_on_but_no_token(monkeypatch):
    _clear_hosted_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_HOSTED_ENDPOINT", "1")
    with pytest.raises(RuntimeError, match="Refusing to serve unauthenticated"):
        create_app(_rt())


def _hosted_client(monkeypatch, **env):
    _clear_hosted_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_HOSTED_ENDPOINT", "1")
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKENS", "k1:tokA,k2:tokB")
    monkeypatch.setenv("SHIELDCALL_HOSTED_QUOTA_PER_MIN", "1000")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return TestClient(create_app(_rt()))


def test_status_auth_shapes(monkeypatch):
    c = _hosted_client(monkeypatch)
    assert c.get("/hosted/v1/status").status_code == 401
    assert c.get("/hosted/v1/status", headers={"Authorization": "Bearer nope"}).status_code == 403
    r = c.get("/hosted/v1/status", headers={"Authorization": "Bearer tokA"})
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["tokens_configured"] == 2
    assert body["token_id"] == "k1"
    # Second rotatable token also accepted.
    r2 = c.get("/hosted/v1/status", headers={"Authorization": "Bearer tokB"})
    assert r2.json()["token_id"] == "k2"


def test_legacy_single_token_supported(monkeypatch):
    _clear_hosted_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_HOSTED_ENDPOINT", "1")
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "legacy-secret")
    c = TestClient(create_app(_rt()))
    r = c.get("/hosted/v1/status", headers={"Authorization": "Bearer legacy-secret"})
    assert r.json()["token_id"] == "default"


def test_analyze_benign_vs_scam(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="shieldcall.serve.hosted")
    c = _hosted_client(monkeypatch)
    h = {"Authorization": "Bearer tokA"}
    benign = c.post(
        "/hosted/v1/analyze",
        json={"transcript": _script_turns("ind_b01"), "client_ref": "benign"},
        headers=h,
    )
    scam = c.post(
        "/hosted/v1/analyze",
        json={"transcript": _script_turns("ind_s01"), "client_ref": "scam"},
        headers=h,
    )
    assert benign.status_code == 200
    assert scam.status_code == 200
    b, s = benign.json(), scam.json()
    assert b["token_id"] == "k1" and s["token_id"] == "k1"
    assert b["client_ref"] == "benign" and s["client_ref"] == "scam"
    assert b["tier"] in {"SAFE", "WATCH", "HIGH_RISK"}
    assert s["fraud"] >= b["fraud"], (b["fraud"], s["fraud"])
    assert b["n_turns"] > 0
    # Stateless per request: no hosted sessions linger in the runtime.
    assert c.app.state.runtime.list_calls() == []
    # Token id is logged with each request.
    assert "token_id=k1" in caplog.text


def test_analyze_rejects_bad_audio(monkeypatch):
    c = _hosted_client(monkeypatch)
    h = {"Authorization": "Bearer tokA"}
    r = c.post("/hosted/v1/analyze", json={"audio": {"sr": 8000, "pcm_s16le_b64": "!!!"}},
               headers=h)
    assert r.status_code == 400


def test_analyze_accepts_audio_chunk(monkeypatch):
    c = _hosted_client(monkeypatch)
    h = {"Authorization": "Bearer tokA"}
    pcm = (np.zeros(1600, dtype=np.int16)).tobytes()
    b64 = base64.b64encode(pcm).decode("ascii")
    r = c.post(
        "/hosted/v1/analyze",
        json={"audio": {"sr": 8000, "pcm_s16le_b64": b64}},
        headers=h,
    )
    assert r.status_code == 200
    assert r.json()["tier"] in {"SAFE", "WATCH", "HIGH_RISK"}


def test_quota_429_with_retry_after(monkeypatch):
    c = _hosted_client(monkeypatch, SHIELDCALL_HOSTED_QUOTA_PER_MIN="1")
    h = {"Authorization": "Bearer tokA"}
    first = c.post("/hosted/v1/analyze", json={}, headers=h)
    assert first.status_code == 200
    second = c.post("/hosted/v1/analyze", json={}, headers=h)
    assert second.status_code == 429
    assert "Retry-After" in second.headers
    assert int(second.headers["Retry-After"]) > 0
    # Quota is per token id: the other token still has budget.
    other = c.post("/hosted/v1/analyze", json={},
                   headers={"Authorization": "Bearer tokB"})
    assert other.status_code == 200


def test_parse_tokens_skips_malformed(monkeypatch):
    _clear_hosted_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKENS", "k1:tokA,,bad-no-colon,:novalue,k2:tokB")
    tokens = hosted.parse_tokens()
    assert tokens == {"k1": "tokA", "k2": "tokB"}
