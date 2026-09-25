"""Fail-closed auth for the main sidecar API (P2-4 gap fix).

create_app() must refuse to start when no bearer token is configured, unless
the explicit local-dev escape hatch SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED
is set. When tokens are configured, the detector routes require a valid
Bearer token (401 missing, 403 wrong), accepting both the legacy single
token and the rotatable id:value pairs with constant-time comparison.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve import http_app
from shieldcall.serve.http_app import create_app


def _client():
    cfg = PipelineConfig(channel=None)
    rt = SidecarRuntime(max_calls=4, pipeline_config=cfg)
    return TestClient(create_app(rt))


def _no_auth_env(monkeypatch):
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKEN", raising=False)
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.delenv("SHIELDCALL_HOSTED_ENDPOINT", raising=False)
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)


def test_create_app_raises_without_token_or_escape_hatch(monkeypatch):
    _no_auth_env(monkeypatch)
    with pytest.raises(RuntimeError, match="Refusing to serve"):
        create_app()


def test_create_app_allows_explicit_unauthenticated_lab(monkeypatch):
    _no_auth_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", "1")
    c = _client()
    assert c.get("/health").status_code == 200
    # Detector routes still serve in lab mode without a token.
    r = c.post("/v1/calls", json={})
    assert r.status_code == 200
    assert r.json()["call_id"]


def test_detector_routes_require_token_when_configured(monkeypatch):
    _no_auth_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "lab-secret")
    c = _client()
    # /health stays public for liveness probes.
    assert c.get("/health").status_code == 200
    # Detector routes: 401 without a token, 403 with a wrong one.
    assert c.post("/v1/calls", json={}).status_code == 401
    assert (
        c.post("/v1/calls", json={}, headers={"Authorization": "Bearer wrong"}).status_code
        == 403
    )
    r = c.post("/v1/calls", json={}, headers={"Authorization": "Bearer lab-secret"})
    assert r.status_code == 200
    cid = r.json()["call_id"]
    # Token is enforced on the other detector routes too.
    assert c.get(f"/v1/calls/{cid}/trace").status_code == 401
    assert (
        c.get(
            f"/v1/calls/{cid}/trace", headers={"Authorization": "Bearer lab-secret"}
        ).status_code
        == 200
    )
    assert c.post("/v1/score/fuse", json={"fraud": 0.1, "synth": 0.2}).status_code == 401
    assert (
        c.post(
            "/v1/score/fuse",
            json={"fraud": 0.1, "synth": 0.2},
            headers={"Authorization": "Bearer lab-secret"},
        ).status_code
        == 200
    )


def test_rotatable_tokens_accepted_on_main_routes(monkeypatch):
    _no_auth_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKENS", "k1:tokA,k2:tokB")
    c = _client()
    for presented in ("tokA", "tokB"):
        r = c.post("/v1/calls", json={}, headers={"Authorization": f"Bearer {presented}"})
        assert r.status_code == 200, presented
    assert (
        c.post("/v1/calls", json={}, headers={"Authorization": "Bearer tokC"}).status_code
        == 403
    )


def test_module_level_app_is_none_when_unconfigured(monkeypatch):
    # Import-time creation must not crash tooling; it logs and leaves app None.
    # Reload with a bare environment to exercise the except path.
    import importlib

    _no_auth_env(monkeypatch)
    reloaded = importlib.reload(http_app)
    try:
        assert reloaded.app is None
        # The supported entry point still fails closed.
        with pytest.raises(RuntimeError, match="Refusing to serve"):
            reloaded.create_app()
    finally:
        monkeypatch.setenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", "1")
        importlib.reload(http_app)
