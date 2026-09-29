"""App-origin JWT auth for the sidecar API (Phase 1 item 4).

App-origin traffic (the ShieldCallAI mobile app, signed in with Supabase
Auth) may present the user's Supabase access token as the Bearer token
instead of the static sidecar token. The static token keeps working
unchanged for SBC/sidecar integrations.

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
from starlette.websockets import WebSocketDisconnect

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

import jwt as pyjwt

from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve import app_jwt, http_app
from shieldcall.serve.app_jwt import AppJwtAuth
from shieldcall.serve.http_app import create_app

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


def _jwt_env(monkeypatch, with_static_token=True):
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKEN", raising=False)
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.delenv("SHIELDCALL_HOSTED_ENDPOINT", raising=False)
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    # The http_app JWT verifier is a process-wide singleton with its own
    # JWKS cache; reset it so each test validates against its own fake keys.
    monkeypatch.setattr(http_app, "_jwt_auth_instance", None)
    monkeypatch.setattr(http_app, "_jwt_auth_env_key", None)
    if with_static_token:
        monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "static-secret")
    monkeypatch.setenv("SHIELDCALL_APP_JWT_JWKS_URL", JWKS_URL)
    monkeypatch.setenv("SHIELDCALL_APP_JWT_ISSUER", ISSUER)


def _client():
    cfg = PipelineConfig(channel=None)
    rt = SidecarRuntime(max_calls=8, pipeline_config=cfg)
    return TestClient(create_app(rt))


def _bearer(token):
    return {"headers": {"Authorization": f"Bearer {token}"}}


# --- unit: the verifier itself -------------------------------------------

def test_from_env_none_when_unset(monkeypatch):
    monkeypatch.delenv("SHIELDCALL_APP_JWT_JWKS_URL", raising=False)
    assert AppJwtAuth.from_env() is None


def test_valid_token_yields_principal(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    auth = AppJwtAuth(JWKS_URL, ISSUER)
    principal = auth.validate(keys.mint(sub="user-1"))
    assert principal.sub == "user-1"
    assert principal.token_id == "jwt:user-1"
    assert keys.fetch_calls == 1


def test_jwks_cached_across_validations(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    auth = AppJwtAuth(JWKS_URL, ISSUER)
    auth.validate(keys.mint(sub="a"))
    auth.validate(keys.mint(sub="b"))
    assert keys.fetch_calls == 1


def test_expired_token_rejected(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    auth = AppJwtAuth(JWKS_URL, ISSUER)
    with pytest.raises(app_jwt.AppJwtInvalid, match="expired"):
        auth.validate(keys.mint(exp_offset=-10))


def test_service_role_rejected(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    auth = AppJwtAuth(JWKS_URL, ISSUER)
    with pytest.raises(app_jwt.AppJwtInvalid, match="role"):
        auth.validate(keys.mint(role="service_role"))


def test_anon_role_rejected(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    auth = AppJwtAuth(JWKS_URL, ISSUER)
    with pytest.raises(app_jwt.AppJwtInvalid, match="role"):
        auth.validate(keys.mint(role="anon"))


def test_wrong_audience_rejected(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    auth = AppJwtAuth(JWKS_URL, ISSUER)
    with pytest.raises(app_jwt.AppJwtInvalid):
        auth.validate(keys.mint(aud="wrong"))


def test_wrong_issuer_rejected(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    auth = AppJwtAuth(JWKS_URL, ISSUER)
    with pytest.raises(app_jwt.AppJwtInvalid, match="issuer"):
        auth.validate(keys.mint(iss="https://evil.example/auth/v1"))


def test_hs256_token_rejected_even_if_signed(monkeypatch):
    # A leaked HS256 service/anon key must never pass as a user token.
    keys = FakeKeys()
    keys.install(monkeypatch)
    auth = AppJwtAuth(JWKS_URL, ISSUER)
    hs = pyjwt.encode(
        {"sub": "user-1", "role": "authenticated", "aud": "authenticated",
         "exp": time.time() + 600},
        "some-hs256-secret",
        algorithm="HS256",
    )
    with pytest.raises(app_jwt.AppJwtInvalid, match="alg"):
        auth.validate(hs)


def test_alg_none_rejected(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    auth = AppJwtAuth(JWKS_URL, ISSUER)
    none_tok = pyjwt.encode(
        {"sub": "u", "role": "authenticated", "aud": "authenticated",
         "exp": time.time() + 600},
        key="",
        algorithm="none",
    )
    with pytest.raises(app_jwt.AppJwtInvalid):
        auth.validate(none_tok)


def test_unknown_kid_triggers_one_refresh_then_fails(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    auth = AppJwtAuth(JWKS_URL, ISSUER)
    # Warm the cache with a valid token first (one fetch).
    auth.validate(keys.mint(sub="warm"))
    assert keys.fetch_calls == 1
    other = FakeKeys(kid="rotated-k2")
    with pytest.raises(app_jwt.AppJwtInvalid, match="key id"):
        auth.validate(other.mint(kid="rotated-k2", key=other.priv_pem))
    # Exactly one refresh for the kid miss, then fail.
    assert keys.fetch_calls == 2


def test_key_rotation_succeeds_after_refresh(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    auth = AppJwtAuth(JWKS_URL, ISSUER)
    other = FakeKeys(kid="rotated-k2")
    # Simulate rotation: the fetch now returns the new key set.
    keys.jwks = {"keys": [other.jwk]}
    principal = auth.validate(other.mint(kid="rotated-k2", key=other.priv_pem))
    assert principal.sub == "user-1"


def test_jwks_outage_is_fail_closed(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    keys.fail_fetch = True
    auth = AppJwtAuth(JWKS_URL, ISSUER)
    with pytest.raises(app_jwt.AppJwtUnavailable):
        auth.validate(keys.mint())


# --- route level ----------------------------------------------------------

def test_jwt_opens_owns_and_serves_call(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    _jwt_env(monkeypatch)
    c = _client()
    tok1 = keys.mint(sub="user-1")
    tok2 = keys.mint(sub="user-2")

    r = c.post("/v1/calls", json={"call_id": "jwt-call"}, **_bearer(tok1))
    assert r.status_code == 200, r.text
    # Owner reads fine.
    assert c.get("/v1/calls/jwt-call/trace", **_bearer(tok1)).status_code == 200
    assert c.get("/v1/calls/jwt-call", **_bearer(tok1)).status_code == 200
    # Another user is shut out of every per-call surface.
    assert c.get("/v1/calls/jwt-call/trace", **_bearer(tok2)).status_code == 403
    assert c.get("/v1/calls/jwt-call", **_bearer(tok2)).status_code == 403
    assert (
        c.post("/v1/calls/jwt-call/transcript", json={"text": "hi"},
               **_bearer(tok2)).status_code == 403
    )
    # A static-token integrator is unrestricted (unchanged behavior).
    static = _bearer("static-secret")
    assert c.get("/v1/calls/jwt-call/trace", **static).status_code == 200
    c.delete("/v1/calls/jwt-call", **_bearer(tok1))


def test_jwt_cannot_touch_integrator_call(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    _jwt_env(monkeypatch)
    c = _client()
    static = _bearer("static-secret")
    cid = c.post("/v1/calls", json={"call_id": "sbc-call"}, **static).json()["call_id"]
    tok = keys.mint(sub="user-1")
    assert c.get(f"/v1/calls/{cid}/trace", **_bearer(tok)).status_code == 403
    assert c.get(f"/v1/calls/{cid}", **_bearer(tok)).status_code == 403
    c.delete(f"/v1/calls/{cid}", **static)


def test_jwt_cannot_reopen_others_call(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    _jwt_env(monkeypatch)
    c = _client()
    tok1 = keys.mint(sub="user-1")
    tok2 = keys.mint(sub="user-2")
    c.post("/v1/calls", json={"call_id": "shared-id"}, **_bearer(tok1))
    r = c.post("/v1/calls", json={"call_id": "shared-id"}, **_bearer(tok2))
    assert r.status_code == 403
    c.delete("/v1/calls/shared-id", **_bearer(tok1))


def test_list_calls_filtered_for_jwt_user(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    _jwt_env(monkeypatch)
    c = _client()
    tok1 = keys.mint(sub="user-1")
    tok2 = keys.mint(sub="user-2")
    static = _bearer("static-secret")
    c.post("/v1/calls", json={"call_id": "mine-1"}, **_bearer(tok1))
    c.post("/v1/calls", json={"call_id": "theirs-1"}, **_bearer(tok2))
    c.post("/v1/calls", json={"call_id": "sbc-1"}, **static)

    seen1 = {x["call_id"] for x in c.get("/v1/calls", **_bearer(tok1)).json()["calls"]}
    assert seen1 == {"mine-1"}, seen1
    seen_static = {x["call_id"] for x in c.get("/v1/calls", **static).json()["calls"]}
    assert {"mine-1", "theirs-1", "sbc-1"} <= seen_static
    for cid in ("mine-1", "theirs-1", "sbc-1"):
        c.delete(f"/v1/calls/{cid}", **static)


def test_bad_jwt_gets_403_and_missing_gets_401(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    _jwt_env(monkeypatch)
    c = _client()
    assert c.post("/v1/calls", json={}).status_code == 401
    assert c.post("/v1/calls", json={}, **_bearer("not-a-jwt")).status_code == 403
    assert (
        c.post("/v1/calls", json={}, **_bearer(keys.mint(exp_offset=-10))).status_code
        == 403
    )


def test_jwt_disabled_without_jwks_url(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    _jwt_env(monkeypatch)
    monkeypatch.delenv("SHIELDCALL_APP_JWT_JWKS_URL")
    c = _client()
    # A perfectly good user token is rejected like any bad static token.
    assert (
        c.post("/v1/calls", json={}, **_bearer(keys.mint())).status_code == 403
    )
    # Static token still works.
    r = c.post("/v1/calls", json={}, **_bearer("static-secret"))
    assert r.status_code == 200
    c.delete(f"/v1/calls/{r.json()['call_id']}", **_bearer("static-secret"))


def test_jwks_outage_is_503_not_fail_open(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    keys.fail_fetch = True
    _jwt_env(monkeypatch)
    c = _client()
    tok = keys.mint()
    r = c.post("/v1/calls", json={}, **_bearer(tok))
    assert r.status_code == 503, r.text
    # Static tokens are unaffected by the JWKS outage.
    r = c.post("/v1/calls", json={}, **_bearer("static-secret"))
    assert r.status_code == 200
    c.delete(f"/v1/calls/{r.json()['call_id']}", **_bearer("static-secret"))


def test_jwt_only_deployment_boots_and_serves(monkeypatch):
    # No static token at all: the JWKS URL satisfies the fail-closed
    # startup check, and the app authenticates with its user JWT.
    keys = FakeKeys()
    keys.install(monkeypatch)
    _jwt_env(monkeypatch, with_static_token=False)
    c = _client()
    tok = keys.mint(sub="app-user")
    r = c.post("/v1/calls", json={}, **_bearer(tok))
    assert r.status_code == 200, r.text
    cid = r.json()["call_id"]
    assert c.get(f"/v1/calls/{cid}/trace", **_bearer(tok)).status_code == 200
    # No static token exists, so a static-looking bearer is rejected.
    assert c.post("/v1/calls", json={}, **_bearer("static-secret")).status_code == 403
    assert c.delete(f"/v1/calls/{cid}", **_bearer(tok)).status_code == 200


def test_per_user_new_call_quota(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    _jwt_env(monkeypatch)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_QUOTA_PER_MIN", "2")
    monkeypatch.setenv("SHIELDCALL_QUOTA_DB_PATH", ":memory:")
    c = _client()
    tok1 = keys.mint(sub="quota-u1")
    tok2 = keys.mint(sub="quota-u2")
    opened = []
    for i in range(2):
        r = c.post("/v1/calls", json={"call_id": f"q1-{i}"}, **_bearer(tok1))
        assert r.status_code == 200, r.text
        opened.append(("q1-%d" % i, tok1))
    # Third open for user-1 is throttled...
    r = c.post("/v1/calls", json={"call_id": "q1-2"}, **_bearer(tok1))
    assert r.status_code == 429, r.text
    assert "Retry-After" in r.headers
    # ...while user-2 still has a full independent budget.
    r = c.post("/v1/calls", json={"call_id": "q2-0"}, **_bearer(tok2))
    assert r.status_code == 200, r.text
    opened.append(("q2-0", tok2))
    static = _bearer("static-secret")
    for cid, tok in opened:
        c.delete(f"/v1/calls/{cid}", **static)


def test_websocket_enforces_ownership(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    _jwt_env(monkeypatch)
    c, rt = _client(), None
    tok1 = keys.mint(sub="ws-u1")
    tok2 = keys.mint(sub="ws-u2")
    cid = c.post("/v1/calls", json={"call_id": "ws-owned"}, **_bearer(tok1)).json()["call_id"]

    def connect(cid, token):
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with c.websocket_connect(
                f"/v1/calls/{cid}/stream", **_bearer(token)
            ) as ws:
                ws.send_json({"type": "transcript", "t": 0.0, "text": "hello"})
                ws.receive_json()
        return excinfo.value.code

    assert connect(cid, tok2) == 4403
    # The owner still streams.
    with c.websocket_connect(f"/v1/calls/{cid}/stream", **_bearer(tok1)) as ws:
        ws.send_json({"type": "transcript", "t": 0.0, "text": "hello"})
        ev = ws.receive_json()
    assert ev["type"] == "event"
    c.delete(f"/v1/calls/{cid}", **_bearer(tok1))


def test_websocket_auth_still_4401_for_bad_token(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    _jwt_env(monkeypatch)
    c = _client()
    tok = keys.mint(sub="ws-u1")
    cid = c.post("/v1/calls", json={"call_id": "ws-auth"}, **_bearer(tok)).json()["call_id"]
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with c.websocket_connect(f"/v1/calls/{cid}/stream") as ws:
            ws.send_json({"type": "transcript", "t": 0.0, "text": "x"})
            ws.receive_json()
    assert excinfo.value.code == 4401
    c.delete(f"/v1/calls/{cid}", **_bearer(tok))


def test_close_releases_ownership(monkeypatch):
    keys = FakeKeys()
    keys.install(monkeypatch)
    _jwt_env(monkeypatch)
    c = _client()
    tok1 = keys.mint(sub="rel-u1")
    tok2 = keys.mint(sub="rel-u2")
    cid = c.post("/v1/calls", json={"call_id": "rel-call"}, **_bearer(tok1)).json()["call_id"]
    assert c.delete(f"/v1/calls/{cid}", **_bearer(tok1)).status_code == 200
    # After close, another user may open the same id fresh.
    r = c.post("/v1/calls", json={"call_id": "rel-call"}, **_bearer(tok2))
    assert r.status_code == 200, r.text
    c.delete(f"/v1/calls/{cid}", **_bearer(tok2))
