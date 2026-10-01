"""FastAPI worker around SidecarRuntime.

The telephone (or Expo lab loopback) does not traverse this process.
Channel twin is off. Recommend-only: this API never hangs up a call.
"""

from __future__ import annotations

import base64
import hmac
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.requests import Request

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

from ..application.detector import DetectorApplication
from ..eval.corpora.independent_scripts import independent_scripts
from ..linguistic.provider import make_asr_from_env
from ..pipeline import PipelineConfig
from ..runtime.runtime import SidecarRuntime
from ..runtime.session import CallSession, SessionEvent
from . import hosted
from . import quota_store as quota_store_mod
from . import app_jwt as app_jwt_mod
from .app_jwt import AppJwtAuth, AppJwtInvalid, AppJwtUnavailable
from .latency import RouteLatencyTracker, buckets_from_env
from .quota_store import QuotaStoreError

# Quota scope for the main detector API. Budgets are independent per
# route family: the main API ("sidecar") and the hosted prototype
# ("hosted") never eat each other's budget.
SIDECAR_QUOTA_SCOPE = "sidecar"


@dataclass(frozen=True)
class AuthPrincipal:
    """Who authenticated this request.

    - Static bearer token -> token_id is the configured token id,
      jwt_sub is None (a trusted integrator: SBC/sidecar/lab).
    - Supabase app-user JWT -> token_id is "jwt:<sub>" (per-user quota
      and logging identity), jwt_sub is the user's sub claim. A JWT
      principal may only touch call sessions it opened itself.
    - Lab escape hatch -> token_id "lab", jwt_sub None.
    """

    token_id: str
    jwt_sub: Optional[str] = None


_jwt_auth_instance: Optional[AppJwtAuth] = None
_jwt_auth_env_key: Optional[tuple] = None


def _app_jwt_auth() -> Optional[AppJwtAuth]:
    """App-origin JWT auth, read lazily so tests can set env per case.

    The instance is cached process-wide and rebuilt only when the
    JWT env knobs change, so the JWKS document cache inside
    AppJwtAuth actually pays off across requests.
    """
    global _jwt_auth_instance, _jwt_auth_env_key
    key = (
        os.environ.get("SHIELDCALL_APP_JWT_JWKS_URL", "").strip(),
        os.environ.get("SHIELDCALL_APP_JWT_ISSUER", "").strip(),
        os.environ.get("SHIELDCALL_APP_JWT_CACHE_TTL", "").strip(),
        os.environ.get("SHIELDCALL_APP_JWT_TIMEOUT", "").strip(),
    )
    if _jwt_auth_env_key != key:
        _jwt_auth_env_key = key
        _jwt_auth_instance = AppJwtAuth.from_env()
    return _jwt_auth_instance

def _sidecar_tokens() -> Dict[str, str]:
    """Configured bearer tokens, read lazily so tests can set env per case."""
    return hosted.parse_tokens()


def _unauthenticated_allowed() -> bool:
    return os.environ.get("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _require_sidecar_auth_configured() -> None:
    """Fail closed at startup: serving the detector API without a bearer
    token is only allowed under the explicit local-dev escape hatch
    SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED=1. A configured app-JWT JWKS
    URL also counts as auth configured (JWT-only deployments serve the
    app without a static token; SBC integrations then have no credential,
    which is the operator's explicit choice)."""
    if _sidecar_tokens():
        return
    if _app_jwt_auth() is not None:
        log.info(
            "no static sidecar token configured; serving with app-JWT auth only "
            "(SHIELDCALL_APP_JWT_JWKS_URL set)"
        )
        return
    if _unauthenticated_allowed():
        log.warning(
            "SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED is set: the sidecar API "
            "serves without a bearer token. Never use in production."
        )
        return
    raise RuntimeError(
        "No sidecar bearer token is configured (set SHIELDCALL_SIDECAR_TOKENS "
        "or SHIELDCALL_SIDECAR_TOKEN, or SHIELDCALL_APP_JWT_JWKS_URL for "
        "app-JWT-only operation). Refusing to serve the detector API "
        "unauthenticated. For local lab use only, set "
        "SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED=1."
    )


# ---------------------------------------------------------------------------
# Pinned CORS (Phase 1, item 5)
# ---------------------------------------------------------------------------

_CORS_WILDCARD_HATCH = "SHIELDCALL_CORS_ALLOW_WILDCARD"


@dataclass(frozen=True)
class CorsConfig:
    """Effective CORS policy for the sidecar API.

    - origins: pinned allowlist (scheme://host[:port] tuples). Empty means
      no cross-origin browser access at all.
    - wildcard: True only under the explicit dev-only wildcard escape
      hatch; the middleware then answers "*" like the old default.
    """

    origins: Tuple[str, ...] = ()
    wildcard: bool = False


def _cors_wildcard_allowed() -> bool:
    return os.environ.get(_CORS_WILDCARD_HATCH, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _validate_cors_origin(entry: str) -> str:
    """Normalize one SHIELDCALL_CORS entry to scheme://host[:port].

    Rejects anything that is not a bare http(s) origin: paths, queries,
    fragments, userinfo, non-http schemes, or missing hosts. Raises
    RuntimeError with the offending value so a misconfigured deployment
    fails loudly at startup instead of serving a silently wrong policy.
    """
    candidate = entry.strip().rstrip("/")
    parsed = urlparse(candidate)
    try:
        port: Optional[int] = parsed.port
    except ValueError:
        # Non-numeric port (e.g. "https://host:bad"): fail closed below.
        port = None
        invalid_port = True
    else:
        invalid_port = False
    ok = (
        not invalid_port
        and parsed.scheme.lower() in {"http", "https"}
        and parsed.hostname
        and not parsed.username
        and not parsed.password
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
    )
    if not ok:
        raise RuntimeError(
            f"Invalid SHIELDCALL_CORS origin {entry!r}: expected a bare "
            "'https://host' or 'https://host:port' (http(s) only, no path, "
            "query, fragment, or credentials)."
        )
    origin = f"{parsed.scheme.lower()}://{parsed.hostname.lower()}"
    if port:
        origin += f":{port}"
    return origin


def cors_config_from_env() -> CorsConfig:
    """Build the CORS policy from the environment, fail-closed.

    SHIELDCALL_CORS is a comma-separated origin allowlist. Unset or blank
    means NO cross-origin access: the same-origin lab UI and native app
    clients (which browsers do not subject to CORS) keep working, while no
    browser page from another origin can call the API. This replaces the
    old default of "*", which the design doc reserves for dev-only use:
    a "*" entry now requires the explicit
    SHIELDCALL_CORS_ALLOW_WILDCARD=1 escape hatch and refuses to boot
    without it. Malformed origins also refuse to boot.
    """
    raw = os.environ.get("SHIELDCALL_CORS", "")
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    if any(e == "*" for e in entries):
        if not _cors_wildcard_allowed():
            raise RuntimeError(
                "SHIELDCALL_CORS contains '*' (wildcard CORS). Wildcard CORS "
                "is dev-only: set SHIELDCALL_CORS_ALLOW_WILDCARD=1 explicitly "
                "for local development, or pin SHIELDCALL_CORS to the app "
                "origin(s) for production. Refusing to serve with an "
                "unacknowledged wildcard."
            )
        log.warning(
            "SHIELDCALL_CORS_ALLOW_WILDCARD is set: the sidecar API allows "
            "cross-origin requests from ANY origin. Dev-only; never use in "
            "production."
        )
        return CorsConfig(origins=(), wildcard=True)
    origins: List[str] = []
    for entry in entries:
        origins.append(_validate_cors_origin(entry))
    # Dedupe, order preserved.
    seen = set()
    unique = [o for o in origins if not (o in seen or seen.add(o))]
    return CorsConfig(origins=tuple(unique), wildcard=False)


class OpenCallBody(BaseModel):
    call_id: Optional[str] = None


class TranscriptBody(BaseModel):
    t: float = 0.0
    text: str = ""


class AudioBody(BaseModel):
    sr: int = 8000
    pcm_s16le_b64: str = Field(..., min_length=1)


class InjectBody(BaseModel):
    script_id: Optional[str] = None
    turns: Optional[List[str]] = None


class ChunkBody(BaseModel):
    t: float = 0.0
    text: str = ""
    sr: int = 8000
    pcm_s16le_b64: Optional[str] = None


class LinguisticScoreBody(BaseModel):
    text: str = ""
    t: float = 0.0


class AcousticScoreBody(BaseModel):
    sr: int = 8000
    pcm_s16le_b64: str = Field(..., min_length=1)


class FuseScoreBody(BaseModel):
    fraud: float = 0.0
    synth: float = 0.0
    t: float = 0.0


def _sidecar_quota_per_min() -> int:
    """Per-token-id budget of new-call opens per minute on the main API.
    0 disables throttling."""
    raw = os.environ.get("SHIELDCALL_SIDECAR_QUOTA_PER_MIN", "60").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 60


def _enforce_new_call_quota(qstore, token_id: str, limit: int) -> None:
    """429 when the per-token new-call budget is spent, or when the quota
    store is unreachable. A dead store freezes new-call opens; the
    audio/score paths keep serving existing calls (never fail open)."""
    if limit <= 0:
        return
    try:
        decision = qstore.consume(SIDECAR_QUOTA_SCOPE, token_id, limit)
    except QuotaStoreError:
        log.error(
            "sidecar quota store unreachable; freezing new-call opens", exc_info=True
        )
        raise HTTPException(
            status_code=429,
            detail="quota store unavailable",
            headers={"Retry-After": "60"},
        )
    if not decision.allowed:
        raise HTTPException(
            status_code=429,
            detail="sidecar quota exceeded",
            headers={"Retry-After": str(decision.retry_after)},
        )


def _check_token(authorization: Optional[str]) -> AuthPrincipal:
    """Authenticate the request. Static bearer tokens are tried first
    (constant-time compare); when app-JWT auth is configured
    (SHIELDCALL_APP_JWT_JWKS_URL), a Supabase user access token is
    accepted as the Bearer token instead.

    Returns the AuthPrincipal. Raises 401 (missing), 403 (bad token),
    503 (auth not configured, or the JWKS is unreachable so the token
    cannot be validated -- fail closed, never pass through).
    """
    tokens = _sidecar_tokens()
    jwt_auth = _app_jwt_auth()
    if not tokens and jwt_auth is None:
        # create_app refuses to start in this state; this is defense in depth
        # in case the environment changed after startup.
        if _unauthenticated_allowed():
            return AuthPrincipal(token_id="lab")
        raise HTTPException(status_code=503, detail="sidecar auth not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    presented = authorization.split(" ", 1)[1]
    for tid, value in tokens.items():
        if hmac.compare_digest(presented, value):
            return AuthPrincipal(token_id=tid)
    if jwt_auth is not None:
        try:
            principal = jwt_auth.validate(presented)
        except AppJwtUnavailable as exc:
            raise HTTPException(
                status_code=503, detail=f"app JWT unavailable: {exc}"
            ) from exc
        except AppJwtInvalid as exc:
            raise HTTPException(
                status_code=403, detail=f"bad app JWT: {exc}"
            ) from exc
        return AuthPrincipal(token_id=principal.token_id, jwt_sub=principal.sub)
    raise HTTPException(status_code=403, detail="bad token")


def _require_call_access(app: FastAPI, call_id: str, principal: AuthPrincipal) -> CallSession:
    """Return the session for call_id if the principal may touch it.

    404 when the call does not exist. A JWT (app-user) principal may only
    access sessions it opened itself (403 otherwise); static-token
    principals are trusted integrators and are unrestricted.
    """
    sess = app.state.runtime.get_call(call_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="unknown call_id")
    if principal.jwt_sub is not None:
        owner = app.state.call_owners.get(call_id)
        if owner != principal.jwt_sub:
            raise HTTPException(status_code=403, detail="not your call")
    return sess


def _pcm_to_float(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    if len(raw) < 2 or len(raw) % 2:
        raise ValueError("pcm must be even-length s16le")
    pcm = np.frombuffer(raw, dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def _last_event_dict(sess: CallSession, events: Optional[List[SessionEvent]] = None) -> Dict[str, Any]:
    risk = None
    if events:
        for ev in reversed(events):
            if ev.pipeline is not None and ev.pipeline.risk is not None:
                risk = ev.pipeline.risk
                break
    out: Dict[str, Any] = {
        "type": "event",
        "call_id": sess.call_id,
        "action": sess.last_action.value,
        "shed": sess.shed,
        "t": float(risk.timestamp_sec) if risk else 0.0,
        "synth": float(risk.acoustic_synth_prob) if risk else 0.0,
        "fraud": float(risk.linguistic_fraud_prob) if risk else 0.0,
        "risk": float(risk.risk_score) if risk else 0.0,
        "regime": risk.regime if risk else "",
        "tier": risk.tier if risk else "SAFE",
        "stage": risk.discourse_stage if risk else "",
        "n_turns": int(sess.n_turns),
        "last_text": sess.last_text,
    }
    return out


def _script_turns(script_id: Optional[str], turns: Optional[List[str]]) -> List[str]:
    if turns:
        return [t for t in turns if t and t.strip()]
    if not script_id:
        raise HTTPException(status_code=400, detail="script_id or turns required")
    for s in independent_scripts():
        if s.script_id == script_id:
            return [text for _, text in s.turns]
    raise HTTPException(status_code=404, detail=f"unknown script_id {script_id}")


def create_app(runtime: Optional[SidecarRuntime] = None) -> FastAPI:
    # Fail closed at startup: the detector API must not serve unauthenticated
    # (P2-4 gap; the hosted endpoint already had this from P2-5).
    _require_sidecar_auth_configured()
    # Persistent per-client quota store (P2-6 Phase 1). Fail closed at
    # startup too: a deployment that cannot open its quota database must
    # not boot and serve unthrottled.
    try:
        qstore = quota_store_mod.QuotaStore(quota_store_mod.default_quota_db_path())
    except QuotaStoreError as exc:
        raise RuntimeError(
            f"Cannot open sidecar quota store ({exc}). Refusing to serve "
            "without quota accounting."
        ) from exc
    sidecar_quota_per_min = _sidecar_quota_per_min()
    # Prototype hosted endpoint (P2-5): fail closed at startup when the flag
    # is on but no bearer token is configured.
    hosted_auth = None
    if hosted.hosted_enabled():
        hosted_auth = hosted.HostedAuth.from_env()
        if not hosted_auth.tokens and hosted_auth.jwt_auth is None:
            raise RuntimeError(
                "SHIELDCALL_HOSTED_ENDPOINT is enabled but no sidecar token is "
                "configured (set SHIELDCALL_SIDECAR_TOKENS or "
                "SHIELDCALL_SIDECAR_TOKEN, or SHIELDCALL_APP_JWT_JWKS_URL "
                "for app-user JWTs). Refusing to serve unauthenticated."
            )
    cfg = PipelineConfig(channel=None, use_conformal=True, fuse_every_n_frames=5)
    rt = runtime or SidecarRuntime(max_calls=8, pipeline_config=cfg, asr=make_asr_from_env())
    if rt.pipeline_config.channel is not None:
        raise RuntimeError("live sidecar must not enable the channel twin")

    app = FastAPI(
        title="ShieldCall Core Detector API",
        version="0.8.0",
        description=(
            "Domain-driven detector sidecar. Recommend-only. Fail-open. "
            "Does not hang up. Does not join the carrier path. "
            "Live OpenAPI at /docs and /openapi.json."
        ),
    )
    app.state.runtime = rt
    detector = DetectorApplication(rt)
    app.state.detector = detector
    app.state.quota_store = qstore
    # Call-session ownership for app-JWT principals (Phase 1, item 4):
    # call_id -> owner sub, or None for sessions opened by a static
    # bearer token (trusted integrator: SBC/sidecar/lab, unrestricted).
    # A JWT principal may only touch sessions it opened itself.
    app.state.call_owners: Dict[str, Optional[str]] = {}
    # Per-route latency histograms + 4xx/5xx rates (P2-6c, Phase 1). Buckets
    # are env-overridable via SHIELDCALL_LATENCY_BUCKETS.
    app.state.latency = RouteLatencyTracker(buckets_from_env())

    @app.middleware("http")
    async def record_route_latency(request: Request, call_next):
        """Time every HTTP exchange and file it under the route TEMPLATE
        (e.g. "POST /v1/calls/{call_id}"), never the concrete path, so
        distinct call ids cannot blow up metric cardinality. Unmatched
        paths (404s) land on "METHOD unmatched". Exceptions are recorded
        as 5xx and re-raised to the outer ServerError middleware, which
        still converts them to a 500 response. Websockets are untouched:
        Starlette 'http' middleware only sees HTTP traffic, and WS hold
        time would poison a latency histogram anyway."""
        tracker: RouteLatencyTracker = request.app.state.latency
        t0 = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", None)
            key = f"{request.method} {path}" if path else f"{request.method} unmatched"
            tracker.record(key, (time.perf_counter() - t0) * 1000.0, status_code)

    # Pinned CORS (Phase 1, item 5). Unset -> no cross-origin access
    # at all; "*" is dev-only behind SHIELDCALL_CORS_ALLOW_WILDCARD=1;
    # malformed origins refuse to boot (raises inside cors_config_from_env).
    # Native app clients and the same-origin lab UI are unaffected by any
    # of these settings: browsers are the only CORS consumers.
    cors = cors_config_from_env()
    app.state.cors = cors
    if cors.wildcard:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    elif cors.origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors.origins),
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        log.info(
            "CORS pinned to %d origin(s): %s",
            len(cors.origins),
            ", ".join(cors.origins),
        )
    else:
        log.info(
            "SHIELDCALL_CORS is unset: cross-origin browser access is denied; "
            "same-origin lab UI and native app clients are unaffected. Pin "
            "SHIELDCALL_CORS to the app origin(s) if a browser client needs "
            "to call this API."
        )

    @app.get("/")
    def lab_home():
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="lab UI missing")
        return FileResponse(page, media_type="text/html")

    @app.get("/v1/scripts")
    def scripts():
        items = []
        for s in independent_scripts():
            items.append(
                {
                    "id": s.script_id,
                    "family": s.family,
                    "is_scam": s.is_scam,
                    "n_turns": len(s.turns),
                }
            )
        return {"scripts": items}

    @app.get("/health")
    def health():
        body = detector.health()
        body["quota"] = app.state.quota_store.stats()
        body["route_latency"] = app.state.latency.snapshot()
        cors = app.state.cors
        body["cors"] = {"origins": list(cors.origins), "wildcard": cors.wildcard}
        return body

    @app.get("/ready")
    def ready():
        body = detector.ready()
        if not body["ready"]:
            raise HTTPException(status_code=503, detail=body)
        return body

    @app.get("/v1/capabilities")
    def capabilities():
        return detector.capabilities()

    @app.post("/v1/calls")
    def open_call(body: OpenCallBody, authorization: Optional[str] = Header(default=None)):
        principal = _check_token(authorization)
        _enforce_new_call_quota(app.state.quota_store, principal.token_id, sidecar_quota_per_min)
        cid = (body.call_id or "").strip() or f"lab-{uuid.uuid4().hex[:12]}"
        sess = rt.open_call(cid)
        # open_call is idempotent: record the owner on first open, and
        # refuse when a JWT principal tries to (re)open someone else's
        # call. Static-token integrators own nothing (unrestricted).
        owners: Dict[str, Optional[str]] = app.state.call_owners
        if principal.jwt_sub is not None:
            existing = owners.get(sess.call_id, "absent")
            if existing == "absent":
                owners[sess.call_id] = principal.jwt_sub
            elif existing != principal.jwt_sub:
                raise HTTPException(status_code=403, detail="not your call")
        else:
            owners.setdefault(sess.call_id, None)
        return {"call_id": sess.call_id, "shed": sess.shed}

    @app.delete("/v1/calls/{call_id}")
    def close_call(call_id: str, authorization: Optional[str] = Header(default=None)):
        principal = _check_token(authorization)
        _require_call_access(app, call_id, principal)
        trace = rt.close_call(call_id)
        app.state.call_owners.pop(call_id, None)
        return {"call_id": call_id, "n_decisions": len(trace)}

    @app.get("/v1/calls/{call_id}/trace")
    def trace(call_id: str, authorization: Optional[str] = Header(default=None)):
        principal = _check_token(authorization)
        sess = _require_call_access(app, call_id, principal)
        return {"call_id": call_id, "trace": sess.agent.trace_dicts()}

    @app.post("/v1/calls/{call_id}/transcript")
    def transcript(
        call_id: str,
        body: TranscriptBody,
        authorization: Optional[str] = Header(default=None),
    ):
        principal = _check_token(authorization)
        sess = _require_call_access(app, call_id, principal)
        sess.push_transcript(body.text, float(body.t))
        # Drive one hop of silence so fusion emits after linguistic update.
        sr = 8000
        hop = np.zeros(int(sr * 0.05), dtype=np.float32)
        events = sess.push_audio(hop, sr)
        return _last_event_dict(sess, events)

    @app.post("/v1/calls/{call_id}/audio")
    def audio(
        call_id: str,
        body: AudioBody,
        authorization: Optional[str] = Header(default=None),
    ):
        principal = _check_token(authorization)
        sess = _require_call_access(app, call_id, principal)
        try:
            samples = _pcm_to_float(body.pcm_s16le_b64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        t0 = time.perf_counter()
        events = sess.push_audio(samples, int(body.sr) or 8000)
        rt.observe_frame_ms((time.perf_counter() - t0) * 1000.0)
        return _last_event_dict(sess, events)

    @app.post("/v1/calls/{call_id}/inject")
    def inject(
        call_id: str,
        body: InjectBody,
        authorization: Optional[str] = Header(default=None),
    ):
        principal = _check_token(authorization)
        sess = _require_call_access(app, call_id, principal)
        turns = _script_turns(body.script_id, body.turns)
        sr = 8000
        last = None
        for i, text in enumerate(turns):
            sess.push_transcript(text, float(i) * 0.8)
            noise = (0.01 * np.random.RandomState(i).randn(sr // 5)).astype(np.float32)
            last = sess.push_audio(noise, sr)
        return {"call_id": call_id, "n_turns": len(turns), **_last_event_dict(sess, last)}

    @app.get("/v1/calls")
    def list_calls(authorization: Optional[str] = Header(default=None)):
        principal = _check_token(authorization)
        body = detector.list_calls()
        if principal.jwt_sub is not None:
            # An app user sees only the calls they opened.
            owners: Dict[str, Optional[str]] = app.state.call_owners
            body = {
                "calls": [
                    c
                    for c in body.get("calls", [])
                    if owners.get(c.get("call_id")) == principal.jwt_sub
                ]
            }
        return body

    @app.get("/v1/calls/{call_id}")
    def get_call(call_id: str, authorization: Optional[str] = Header(default=None)):
        principal = _check_token(authorization)
        _require_call_access(app, call_id, principal)
        body = detector.get_call(call_id)
        if body is None:
            raise HTTPException(status_code=404, detail="unknown call_id")
        return body

    @app.get("/v1/calls/{call_id}/decision")
    def last_decision(call_id: str, authorization: Optional[str] = Header(default=None)):
        principal = _check_token(authorization)
        _require_call_access(app, call_id, principal)
        body = detector.decision(call_id)
        if body is None:
            raise HTTPException(status_code=404, detail="unknown call_id")
        return body

    @app.post("/v1/calls/{call_id}/chunk")
    def chunk(
        call_id: str,
        body: ChunkBody,
        authorization: Optional[str] = Header(default=None),
    ):
        principal = _check_token(authorization)
        _require_call_access(app, call_id, principal)
        samples = None
        if body.pcm_s16le_b64:
            try:
                samples = _pcm_to_float(body.pcm_s16le_b64)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        ev = detector.ingest_chunk(call_id, body.text, body.t, samples, body.sr)
        if ev is None:
            raise HTTPException(status_code=404, detail="unknown call_id")
        return ev

    @app.post("/v1/score/linguistic")
    def score_linguistic(body: LinguisticScoreBody, authorization: Optional[str] = Header(default=None)):
        _check_token(authorization)
        return detector.score_linguistic(body.text, body.t)

    @app.post("/v1/score/acoustic")
    def score_acoustic(body: AcousticScoreBody, authorization: Optional[str] = Header(default=None)):
        _check_token(authorization)
        try:
            samples = _pcm_to_float(body.pcm_s16le_b64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return detector.score_acoustic(samples, body.sr)

    @app.post("/v1/score/fuse")
    def score_fuse(body: FuseScoreBody, authorization: Optional[str] = Header(default=None)):
        _check_token(authorization)
        return detector.score_fuse(body.fraud, body.synth, body.t)

    @app.websocket("/v1/calls/{call_id}/stream")
    async def stream(ws: WebSocket, call_id: str):
        await ws.accept()
        principal: Optional[AuthPrincipal] = None
        if _sidecar_tokens() or _app_jwt_auth() is not None or not _unauthenticated_allowed():
            proto = ws.headers.get("authorization") or ""
            try:
                principal = _check_token(proto)
            except HTTPException:
                await ws.close(code=4401)
                return
        sess = rt.get_call(call_id)
        if sess is None:
            # No implicit call creation (P2-6b): the only way to open a call
            # is POST /v1/calls, which enforces the per-token new-call quota
            # from the persistent quota store. Opening the session here would
            # let a client bypass that gate and create unbounded sessions
            # (resource-exhaustion vector).
            await ws.close(code=4404, reason="unknown call_id")
            return
        if principal is not None and principal.jwt_sub is not None:
            owner = app.state.call_owners.get(call_id)
            if owner != principal.jwt_sub:
                await ws.close(code=4403, reason="not your call")
                return
        try:
            while True:
                msg = await ws.receive_json()
                kind = msg.get("type")
                if kind == "close":
                    break
                if kind == "transcript":
                    sess.push_transcript(str(msg.get("text") or ""), float(msg.get("t") or 0.0))
                    hop = np.zeros(400, dtype=np.float32)
                    events = sess.push_audio(hop, 8000)
                    await ws.send_json(_last_event_dict(sess, events))
                elif kind == "audio":
                    samples = _pcm_to_float(str(msg.get("pcm_s16le_b64") or ""))
                    events = sess.push_audio(samples, int(msg.get("sr") or 8000))
                    await ws.send_json(_last_event_dict(sess, events))
                else:
                    await ws.send_json({"type": "error", "detail": f"unknown type {kind}"})
        except WebSocketDisconnect:
            return
        except Exception as exc:
            try:
                await ws.send_json({"type": "error", "detail": str(exc)})
            except Exception:
                pass

    if hosted_auth is not None:
        hosted.register_hosted_routes(app, rt, hosted_auth)

    return app


try:
    app = create_app()
except RuntimeError as exc:
    # Module import must not crash when auth is unconfigured (tests, docs
    # builds). create_app() is the supported entry point and still fails
    # closed; uvicorn entry points must call it after configuring env.
    log.warning("shieldcall.serve.http_app: %s", exc)
    app = None
