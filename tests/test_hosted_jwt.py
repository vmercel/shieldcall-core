"""P2-6f: app-origin JWT auth on the hosted prototype.

The hosted endpoint (/hosted/v1/*) previously accepted only static bearer
tokens. It now also accepts Supabase Auth access tokens when
SHIELDCALL_APP_JWT_JWKS_URL is configured, with static tokens tried first
(exactly the ordering the main API uses). JWT principals authenticate as
token_id "jwt:<sub>" and get per-user quota through the ordinary quota
path; the analyze route is stateless (ephemeral session per request), so
no call-ownership bookkeeping applies.

The JWKS is faked per test (ES256 keypair minted in-process, fetch
monkeypatched); no network, no real Supabase project.
"""

from __future__ import annotations

import base64
import time

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("jwt")
from fastapi.testclient import TestClient

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

import jwt as pyjwt

from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve import app_jwt, http_app

ISSUER = "https://fake-project.supabase.co/auth/v1"
JWKS_URL = "https://fake-project.supabase.co/auth/v1/jwks"


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class FakeKeys:
    """One ES256 keypair plus its JWKS document and a failing fetch hook."""

    def __init__(self, kid: str = "test-k1"):
        self.kid = kid
        self.key = ec.generate_private_key(ec.SECP256R1())
        pub = self.key.public_key().public_numbers()
        self.jwk = {
            "kty": "EC",
            "crv": "P-256",
            "x": _b64u(pub.x.to_bytes(32, "big")),
            "y": _b64u(pub.y.to_bytes(32, "big")),
            "kid": kid,
        }
        self.priv_pem = self.key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        self.jwks = {"keys": [self.jwk]}
        self.fetch_calls = 0
        self.fail_fetch = False

    def install(self, monkeypatch):
        def fake_fetch(url, timeout):
            self.fetch_calls += 1
            assert url == JWKS_URL
            if self.fail_fetch:
                raise app_jwt.AppJwtUnavailable("boom")
            return self.jwks

        monkeypatch.setattr(app_jwt, "_fetch_jwks", fake_fetch)

    def mint(self, sub="user-1", role="authenticated", aud="authenticated",
             iss=ISSUER, exp_offset=600, kid=None, alg="ES256", key=None):
        claims = {"sub": sub, "role": role, "aud": aud,
                  "exp": time.time() + exp_offset}
        if iss is not None:
            claims["iss"] = iss
        headers = {"kid": self.kid if kid is None else kid}
        return pyjwt.encode(
            claims, key or self.priv_pem, algorithm=alg, headers=headers
        )


def _rt():
    cfg = PipelineConfig(channel=None)
    return SidecarRuntime(max_calls=4, pipeline_config=cfg)


def _clear(monkeypatch):
    for k in (
        "SHIELDCALL_HOSTED_ENDPOINT",
        "SHIELDCALL_SIDECAR_TOKEN",
        "SHIELDCALL_SIDECAR_TOKENS",
        "SHIELDCALL_HOSTED_QUOTA_PER_MIN",
        "SHIELDCALL_APP_JWT_JWKS_URL",
        "SHIELDCALL_APP_JWT_ISSUER",
        "SHIELDCALL_APP_JWT_CACHE_TTL",
        "SHIELDCALL_APP_JWT_TIMEOUT",
        "SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED",
    ):
        monkeypatch.delenv(k, raising=False)
    # The http_app JWT verifier is a process-wide singleton with its own
    # JWKS cache; reset it so each test builds against its own fake keys.
    monkeypatch.setattr(http_app, "_jwt_auth_instance", None)
    monkeypatch.setattr(http_app, "_jwt_auth_env_key", None)


def _client(keys, monkeypatch, static_token="static-secret", quota="1000",
            install_keys=True):
    _clear(monkeypatch)
    if install_keys:
        keys.install(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_HOSTED_ENDPOINT", "1")
    monkeypatch.setenv("SHIELDCALL_APP_JWT_JWKS_URL", JWKS_URL)
    monkeypatch.setenv("SHIELDCALL_APP_JWT_ISSUER", ISSUER)
    if static_token is not None:
        monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", static_token)
    monkeypatch.setenv("SHIELDCALL_HOSTED_QUOTA_PER_MIN", quota)
    return TestClient(http_app.create_app(_rt()))


def _bearer(token):
    return {"headers": {"Authorization": f"Bearer {token}"}}


_BENIGN = {"transcript": [{"t": 0.0, "text": "Hello, this is the bank calling about your account."}]}


def test_jwt_accepted_on_status_and_analyze(monkeypatch):
    keys = FakeKeys()
    c = _client(keys, monkeypatch)
    token = keys.mint()
    r = c.get("/hosted/v1/status", **_bearer(token))
    assert r.status_code == 200
    body = r.json()
    assert body["token_id"] == "jwt:user-1"
    assert body["jwt_configured"] is True
    r = c.post("/hosted/v1/analyze", json=_BENIGN, **_bearer(token))
    assert r.status_code == 200
    assert r.json()["token_id"] == "jwt:user-1"


def test_static_token_still_accepted_when_both_configured(monkeypatch):
    keys = FakeKeys()
    c = _client(keys, monkeypatch)
    r = c.get("/hosted/v1/status",
              headers={"Authorization": "Bearer static-secret"})
    assert r.status_code == 200
    assert r.json()["token_id"] == "default"


def test_jwt_per_user_quota_isolation(monkeypatch):
    keys = FakeKeys()
    c = _client(keys, monkeypatch, quota="1")
    a = keys.mint(sub="user-a")
    b = keys.mint(sub="user-b")
    assert c.post("/hosted/v1/analyze", json=_BENIGN, **_bearer(a)).status_code == 200
    r = c.post("/hosted/v1/analyze", json=_BENIGN, **_bearer(a))
    assert r.status_code == 429
    assert r.headers["Retry-After"]
    # user-b has her own independent budget under the same hosted scope.
    r = c.post("/hosted/v1/analyze", json=_BENIGN, **_bearer(b))
    assert r.status_code == 200
    assert r.json()["token_id"] == "jwt:user-b"


def test_jwt_only_deployment_boots(monkeypatch):
    # No static tokens at all: the JWKS URL alone satisfies the fail-closed
    # startup check for both the main API and the hosted routes (parity).
    keys = FakeKeys()
    c = _client(keys, monkeypatch, static_token=None)
    r = c.post("/hosted/v1/analyze", json=_BENIGN, **_bearer(keys.mint()))
    assert r.status_code == 200


def test_flag_on_but_no_static_and_no_jwks_still_refuses_to_boot(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setenv("SHIELDCALL_HOSTED_ENDPOINT", "1")
    with pytest.raises(RuntimeError, match="Refusing to serve unauthenticated"):
        http_app.create_app(_rt())


def test_bad_jwt_is_403(monkeypatch):
    keys = FakeKeys()
    c = _client(keys, monkeypatch)
    # Token with the right kid but signed by a different key.
    other = FakeKeys()
    r = c.get("/hosted/v1/status", **_bearer(other.mint(kid=keys.kid)))
    assert r.status_code == 403
    # Expired token.
    r = c.get("/hosted/v1/status", **_bearer(keys.mint(exp_offset=-10)))
    assert r.status_code == 403


def test_missing_bearer_is_401_with_jwt_configured(monkeypatch):
    keys = FakeKeys()
    c = _client(keys, monkeypatch)
    assert c.get("/hosted/v1/status").status_code == 401
    assert c.post("/hosted/v1/analyze", json=_BENIGN).status_code == 401


def test_jwks_outage_is_503_fail_closed(monkeypatch):
    keys = FakeKeys()
    c = _client(keys, monkeypatch)
    keys.fail_fetch = True
    token = keys.mint()
    r = c.post("/hosted/v1/analyze", json=_BENIGN, **_bearer(token))
    assert r.status_code == 503


def test_status_reports_jwt_not_configured_without_jwks(monkeypatch):
    # Static-only deployment: the new key is present and False.
    _clear(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setenv("SHIELDCALL_HOSTED_ENDPOINT", "1")
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "static-secret")
    c = TestClient(http_app.create_app(_rt()))
    r = c.get("/hosted/v1/status",
              headers={"Authorization": "Bearer static-secret"})
    assert r.status_code == 200
    assert r.json()["jwt_configured"] is False
