"""Pinned CORS for the sidecar API (Phase 1, item 5 of 5).

SHIELDCALL_CORS used to default to "*" (any browser page could call the
detector API). Now: unset/blank means no cross-origin access at all,
explicit origins are validated and pinned at startup, and "*" requires
the explicit dev-only SHIELDCALL_CORS_ALLOW_WILDCARD=1 escape hatch,
refusing to boot without it.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve import http_app
from shieldcall.serve.http_app import CorsConfig, cors_config_from_env, create_app

HATCH = "SHIELDCALL_CORS_ALLOW_WILDCARD"


def _client():
    cfg = PipelineConfig(channel=None)
    rt = SidecarRuntime(max_calls=4, pipeline_config=cfg)
    return TestClient(create_app(rt))


def _clean_cors_env(monkeypatch):
    monkeypatch.delenv("SHIELDCALL_CORS", raising=False)
    monkeypatch.delenv(HATCH, raising=False)


def _preflight(client, origin):
    return client.options(
        "/v1/calls",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization",
        },
    )


# --- unit: cors_config_from_env -------------------------------------------


def test_unset_means_no_cross_origin(monkeypatch):
    _clean_cors_env(monkeypatch)
    assert cors_config_from_env() == CorsConfig(origins=(), wildcard=False)


def test_blank_means_no_cross_origin(monkeypatch):
    _clean_cors_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_CORS", "   ")
    assert cors_config_from_env() == CorsConfig(origins=(), wildcard=False)


def test_wildcard_refuses_without_hatch(monkeypatch):
    _clean_cors_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_CORS", "*")
    with pytest.raises(RuntimeError, match="dev-only"):
        cors_config_from_env()


def test_wildcard_mixed_with_origins_still_needs_hatch(monkeypatch):
    _clean_cors_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_CORS", "*, https://app.example.com")
    with pytest.raises(RuntimeError, match="dev-only"):
        cors_config_from_env()


def test_wildcard_allowed_under_explicit_hatch(monkeypatch, caplog):
    _clean_cors_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_CORS", "*")
    monkeypatch.setenv(HATCH, "1")
    with caplog.at_level("WARNING", logger="shieldcall.serve.http_app"):
        cfg = cors_config_from_env()
    assert cfg == CorsConfig(origins=(), wildcard=True)
    assert "SHIELDCALL_CORS_ALLOW_WILDCARD is set" in caplog.text


@pytest.mark.parametrize(
    "raw",
    [
        "notaurl",
        "ftp://host.example.com",
        "ws://host.example.com",
        "https://host.example.com/call/path",
        "https://host.example.com?q=1",
        "https://host.example.com#frag",
        "https://user:pass@host.example.com",
        "http://",
        "://missing-scheme.example.com",
        "https://host.example.com:badport",
        "",
    ],
)
def test_invalid_origins_refuse_to_boot(monkeypatch, raw):
    _clean_cors_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_CORS", raw or "https://ok.example.com, notaurl")
    with pytest.raises(RuntimeError, match="Invalid SHIELDCALL_CORS"):
        cors_config_from_env()


def test_origins_normalized_and_deduped(monkeypatch):
    _clean_cors_env(monkeypatch)
    monkeypatch.setenv(
        "SHIELDCALL_CORS",
        "HTTPS://App.Example.COM/, https://app.example.com , http://lab.local:8080",
    )
    assert cors_config_from_env() == CorsConfig(
        origins=("https://app.example.com", "http://lab.local:8080"),
        wildcard=False,
    )


# --- integration: middleware behavior -------------------------------------


def test_default_denies_cross_origin_but_serves_same_origin(monkeypatch):
    _clean_cors_env(monkeypatch)
    c = _client()
    # Preflight from a browser origin: no allow-origin header -> denied.
    r = _preflight(c, "https://app.example.com")
    assert r.headers.get("access-control-allow-origin") is None
    # Same-origin / non-browser traffic unaffected.
    assert c.get("/health").status_code == 200
    body = c.get("/health").json()
    assert body["cors"] == {"origins": [], "wildcard": False}


def test_pinned_origins_allowlisted_only(monkeypatch):
    _clean_cors_env(monkeypatch)
    monkeypatch.setenv(
        "SHIELDCALL_CORS", "https://app.example.com, https://admin.example.com:8443"
    )
    c = _client()
    r = _preflight(c, "https://app.example.com")
    assert r.headers.get("access-control-allow-origin") == "https://app.example.com"
    r = _preflight(c, "https://admin.example.com:8443")
    assert (
        r.headers.get("access-control-allow-origin")
        == "https://admin.example.com:8443"
    )
    # A port mismatch is a different origin.
    r = _preflight(c, "https://admin.example.com")
    assert r.headers.get("access-control-allow-origin") is None
    r = _preflight(c, "https://evil.example.com")
    assert r.headers.get("access-control-allow-origin") is None
    body = c.get("/health").json()
    assert body["cors"] == {
        "origins": ["https://app.example.com", "https://admin.example.com:8443"],
        "wildcard": False,
    }


def test_wildcard_hatch_allows_any_origin(monkeypatch):
    _clean_cors_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_CORS", "*")
    monkeypatch.setenv(HATCH, "1")
    c = _client()
    r = _preflight(c, "https://anything.example.com")
    assert r.headers.get("access-control-allow-origin") == "*"
    body = c.get("/health").json()
    assert body["cors"] == {"origins": [], "wildcard": True}


def test_wildcard_without_hatch_refuses_to_boot(monkeypatch):
    _clean_cors_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_CORS", "*")
    with pytest.raises(RuntimeError, match="unacknowledged wildcard"):
        create_app()


def test_invalid_origin_refuses_to_boot(monkeypatch):
    _clean_cors_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_CORS", "https://ok.example.com, /absolute/path")
    with pytest.raises(RuntimeError, match="Invalid SHIELDCALL_CORS"):
        create_app()
