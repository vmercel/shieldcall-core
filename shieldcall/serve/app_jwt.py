"""Supabase Auth JWT validation for app-origin traffic.

Phase 1 item 4 of docs/HOSTED_INFERENCE_DESIGN.md (section 2): app-origin
traffic (the ShieldCallAI mobile app, signed in with Supabase Auth) may
present the user's Supabase access token as the Bearer token instead of
the static sidecar token. The static bearer token remains for SBC/sidecar
integrations and keeps working unchanged.

Env knobs (read lazily so tests can set them per case):
  SHIELDCALL_APP_JWT_JWKS_URL   Project JWKS endpoint, e.g.
                                https://<project-ref>.supabase.co/auth/v1/jwks.
                                Unset -> app-JWT auth is disabled entirely and
                                user tokens are rejected like any bad token.
  SHIELDCALL_APP_JWT_ISSUER     Expected `iss` claim, e.g.
                                https://<project-ref>.supabase.co/auth/v1.
                                Optional but recommended: defense in depth
                                past the JWKS signature check.
  SHIELDCALL_APP_JWT_CACHE_TTL  JWKS cache seconds. Default 3600.
  SHIELDCALL_APP_JWT_TIMEOUT    JWKS fetch timeout seconds. Default 5.

Contract (fail closed, never fail open):
- Only asymmetric algorithms actually carried by the JWKS (RS256/ES256)
  are accepted. ``alg=none`` and symmetric HS* tokens are rejected, so a
  leaked anon/service key can never pass as a user identity.
- The token must carry ``role == "authenticated"`` and
  ``aud == "authenticated"``. service_role and anon tokens are rejected
  even if they verify against the JWKS.
- A JWKS fetch failure at request time raises AppJwtUnavailable (mapped
  to HTTP 503 by callers): the server refuses to authenticate rather
  than guessing.
- ``kid`` misses trigger exactly one JWKS refresh (key rotation), then
  fail.
"""

from __future__ import annotations

import logging
import os
import json
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import httpx
import jwt

log = logging.getLogger(__name__)

# Asymmetric algorithms only. HS* is deliberately excluded: Supabase
# service_role/anon keys are HS256 and must never authenticate as a user.
_ALLOWED_ALGS = ("RS256", "ES256")


class AppJwtError(Exception):
    """Base class for app-JWT auth failures."""


class AppJwtInvalid(AppJwtError):
    """The presented token is not a valid app-user token (HTTP 403)."""


class AppJwtUnavailable(AppJwtError):
    """The JWKS could not be fetched, so the token cannot be validated
    (HTTP 503). Fail closed: never pass the request through."""


@dataclass(frozen=True)
class JwtPrincipal:
    """An authenticated app user. ``token_id`` is the quota/logging
    identity: per-user budgets fall out of the existing per-token-id
    quota store with no special casing."""

    sub: str

    @property
    def token_id(self) -> str:
        return f"jwt:{self.sub}"


def _fetch_jwks(url: str, timeout: float) -> dict:
    """Fetch and parse the JWKS document. Raises AppJwtUnavailable on any
    transport or parse failure (fail closed)."""
    try:
        resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
        doc = resp.json()
    except Exception as exc:
        raise AppJwtUnavailable(f"cannot fetch JWKS from {url}: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("keys"), list):
        raise AppJwtUnavailable(f"malformed JWKS document from {url}")
    return doc


def _key_for_alg(jwk: dict, alg: str):
    """Build a PyJWT verification key from one JWK dict. Raises
    AppJwtInvalid for keys we cannot use."""
    try:
        if alg == "RS256":
            return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
        if alg == "ES256":
            return jwt.algorithms.ECAlgorithm.from_jwk(json.dumps(jwk))
    except Exception as exc:
        raise AppJwtInvalid(f"unusable JWK (kid={jwk.get('kid')}): {exc}") from exc
    raise AppJwtInvalid(f"unsupported alg {alg}")


class AppJwtAuth:
    """Validates Supabase Auth access tokens against the project JWKS."""

    def __init__(
        self,
        jwks_url: str,
        expected_issuer: Optional[str] = None,
        cache_ttl: float = 3600.0,
        timeout: float = 5.0,
    ) -> None:
        self.jwks_url = jwks_url
        self.expected_issuer = expected_issuer
        self.cache_ttl = max(1.0, cache_ttl)
        self.timeout = max(0.5, timeout)
        self._lock = threading.Lock()
        self._cached_at: float = 0.0
        self._keys: Dict[str, dict] = {}

    @classmethod
    def from_env(cls) -> Optional["AppJwtAuth"]:
        """None when SHIELDCALL_APP_JWT_JWKS_URL is unset (JWT auth off)."""
        url = os.environ.get("SHIELDCALL_APP_JWT_JWKS_URL", "").strip()
        if not url:
            return None
        issuer = os.environ.get("SHIELDCALL_APP_JWT_ISSUER", "").strip() or None
        ttl_raw = os.environ.get("SHIELDCALL_APP_JWT_CACHE_TTL", "3600").strip()
        timeout_raw = os.environ.get("SHIELDCALL_APP_JWT_TIMEOUT", "5").strip()
        try:
            ttl = float(ttl_raw)
        except ValueError:
            ttl = 3600.0
        try:
            timeout = float(timeout_raw)
        except ValueError:
            timeout = 5.0
        return cls(url, issuer, ttl, timeout)

    def _refresh_keys(self) -> None:
        doc = _fetch_jwks(self.jwks_url, self.timeout)
        keys: Dict[str, dict] = {}
        for jwk in doc["keys"]:
            if isinstance(jwk, dict) and jwk.get("kid"):
                keys[str(jwk["kid"])] = jwk
        with self._lock:
            self._keys = keys
            self._cached_at = time.monotonic()

    def _keys_snapshot(self) -> Dict[str, dict]:
        with self._lock:
            fresh = (time.monotonic() - self._cached_at) < self.cache_ttl
            return dict(self._keys) if fresh else {}

    def validate(self, token: str) -> JwtPrincipal:
        """Validate a Bearer token as a Supabase app-user JWT.

        Returns the JwtPrincipal on success; raises AppJwtInvalid (bad
        token) or AppJwtUnavailable (JWKS unreachable, fail closed).
        """
        try:
            header = jwt.get_unverified_header(token)
        except Exception as exc:
            raise AppJwtInvalid(f"malformed token: {exc}") from exc
        alg = header.get("alg", "")
        if alg not in _ALLOWED_ALGS:
            # Catches alg=none, HS256 service/anon keys, and anything else
            # the JWKS cannot have signed.
            raise AppJwtInvalid(f"unexpected token alg {alg!r}")
        kid = str(header.get("kid") or "")

        keys = self._keys_snapshot()
        if not keys or kid not in keys:
            # Cache empty/expired or rotation: exactly one refresh, then fail.
            self._refresh_keys()
            keys = self._keys_snapshot()
        jwk = keys.get(kid)
        if jwk is None:
            if len(keys) == 1 and not kid:
                jwk = next(iter(keys.values()))
            else:
                raise AppJwtInvalid("token key id not in JWKS")
        key = _key_for_alg(jwk, alg)
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=[alg],
                audience="authenticated",
                options={"require": ["exp", "sub"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AppJwtInvalid("token expired") from exc
        except jwt.InvalidTokenError as exc:
            raise AppJwtInvalid(f"token rejected: {exc}") from exc
        if claims.get("role") != "authenticated":
            # service_role / anon tokens must never authenticate as a user,
            # even if one somehow verified against this JWKS.
            raise AppJwtInvalid("token role is not an authenticated user")
        if self.expected_issuer and claims.get("iss") != self.expected_issuer:
            raise AppJwtInvalid("unexpected token issuer")
        sub = str(claims.get("sub") or "")
        if not sub:
            raise AppJwtInvalid("token has no subject")
        return JwtPrincipal(sub=sub)
