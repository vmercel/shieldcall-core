"""Prototype hosted inference endpoint (P2-5).

Implements the auth + quota contract from docs/HOSTED_INFERENCE_DESIGN.md
(section 2) behind a feature flag. This is a prototype: single detector
route, in-process per-client quota, stateless per-request analysis. Not
production until it has a persistent quota store, structured metrics, and
the Phase 1 sticky pool from the design doc.

Env knobs:
  SHIELDCALL_HOSTED_ENDPOINT   Set to 1/true/yes/on to register /hosted/*
                               routes. Default: off.
  SHIELDCALL_SIDECAR_TOKEN     Legacy single bearer token. When the hosted
                               endpoint is enabled and no token is configured,
                               create_app raises instead of serving (fail
                               closed).
  SHIELDCALL_SIDECAR_TOKENS    Rotatable token list: comma-separated
                               "id:value" pairs (e.g. "k1:abc,k2:def"). The
                               id of the token that authenticated a request
                               is logged with the request. Takes precedence
                               over SHIELDCALL_SIDECAR_TOKEN when set.
  SHIELDCALL_HOSTED_QUOTA_PER_MIN
                               Per-token-id budget of analyze calls per
                               minute. Default 60. 0 disables throttling.
"""

from __future__ import annotations

import base64
import hmac
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0


def hosted_enabled() -> bool:
    return os.environ.get("SHIELDCALL_HOSTED_ENDPOINT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def parse_tokens() -> Dict[str, str]:
    """Return {token_id: token_value}. Empty when nothing is configured."""
    multi = os.environ.get("SHIELDCALL_SIDECAR_TOKENS", "").strip()
    tokens: Dict[str, str] = {}
    if multi:
        for item in multi.split(","):
            item = item.strip()
            if not item or ":" not in item:
                continue
            tid, _, value = item.partition(":")
            tid, value = tid.strip(), value.strip()
            if tid and value:
                tokens[tid] = value
    legacy = os.environ.get("SHIELDCALL_SIDECAR_TOKEN", "").strip()
    if legacy and not tokens:
        tokens["default"] = legacy
    return tokens


class TranscriptTurn(BaseModel):
    t: float = 0.0
    text: str = ""


class AudioChunk(BaseModel):
    sr: int = 8000
    pcm_s16le_b64: str = Field(..., min_length=1)


class AnalyzeBody(BaseModel):
    transcript: List[TranscriptTurn] = Field(default_factory=list)
    audio: Optional[AudioChunk] = None
    client_ref: Optional[str] = Field(default=None, max_length=128)


class HostedAuth:
    """Bearer auth with rotatable token ids and an in-process per-client quota."""

    def __init__(self, tokens: Dict[str, str], quota_per_min: int = 60):
        self.tokens = tokens
        self.quota_per_min = max(0, quota_per_min)
        # token_id -> (window_start_epoch, count)
        self._usage: Dict[str, Tuple[float, int]] = {}

    @classmethod
    def from_env(cls) -> "HostedAuth":
        quota_raw = os.environ.get("SHIELDCALL_HOSTED_QUOTA_PER_MIN", "60").strip()
        try:
            quota = int(quota_raw)
        except ValueError:
            quota = 60
        return cls(parse_tokens(), quota_per_min=quota)

    def authenticate(self, authorization: Optional[str]) -> str:
        """Return the token id that authenticated the request.

        Raises 401 when the bearer credential is missing, 403 when it does
        not match any configured token. Constant-time compare per token.
        """
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        presented = authorization.split(" ", 1)[1]
        for tid, value in self.tokens.items():
            if hmac.compare_digest(presented, value):
                return tid
        raise HTTPException(status_code=403, detail="bad token")

    def quota_wait_seconds(self, token_id: str) -> Optional[int]:
        """Consume one budget unit. Return seconds to wait when over quota."""
        if self.quota_per_min <= 0:
            return None
        now = time.time()
        start, count = self._usage.get(token_id, (now, 0))
        if now - start >= WINDOW_SECONDS:
            start, count = now, 0
        if count >= self.quota_per_min:
            return int(WINDOW_SECONDS - (now - start)) + 1
        self._usage[token_id] = (start, count + 1)
        return None

    def reset_usage(self) -> None:
        self._usage.clear()


def _pcm_to_float(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    if len(raw) < 2 or len(raw) % 2:
        raise ValueError("pcm must be even-length s16le")
    pcm = np.frombuffer(raw, dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def _require(auth: HostedAuth, authorization: Optional[str]) -> str:
    tid = auth.authenticate(authorization)
    wait = auth.quota_wait_seconds(tid)
    if wait is not None:
        raise HTTPException(
            status_code=429,
            detail="hosted quota exceeded",
            headers={"Retry-After": str(wait)},
        )
    return tid


def _risk_dict(sess, events) -> Dict[str, Any]:
    risk = None
    if events:
        for ev in reversed(events):
            if ev.pipeline is not None and ev.pipeline.risk is not None:
                risk = ev.pipeline.risk
                break
    return {
        "call_id": sess.call_id,
        "action": sess.last_action.value,
        "tier": risk.tier if risk else "SAFE",
        "risk_score": float(risk.risk_score) if risk else 0.0,
        "fraud": float(risk.linguistic_fraud_prob) if risk else 0.0,
        "synth": float(risk.acoustic_synth_prob) if risk else 0.0,
        "regime": risk.regime if risk else "",
        "stage": risk.discourse_stage if risk else "",
        "n_turns": int(sess.n_turns),
        "t": float(risk.timestamp_sec) if risk else 0.0,
    }


def register_hosted_routes(app: FastAPI, runtime, auth: HostedAuth) -> None:
    """Mount the prototype hosted routes. Caller must have fail-closed
    already: raises RuntimeError if no token is configured."""
    if not auth.tokens:
        raise RuntimeError(
            "SHIELDCALL_HOSTED_ENDPOINT is enabled but no sidecar token is "
            "configured (set SHIELDCALL_SIDECAR_TOKENS or "
            "SHIELDCALL_SIDECAR_TOKEN). Refusing to serve unauthenticated."
        )
    router = APIRouter(prefix="/hosted/v1", tags=["hosted"])

    @router.get("/status")
    def status(authorization: Optional[str] = Header(default=None)):
        tid = auth.authenticate(authorization)
        return {
            "enabled": True,
            "prototype": True,
            "tokens_configured": len(auth.tokens),
            "quota_per_min": auth.quota_per_min,
            "token_id": tid,
        }

    @router.post("/analyze")
    def analyze(body: AnalyzeBody, authorization: Optional[str] = Header(default=None)):
        tid = _require(auth, authorization)
        call_id = f"hosted-{uuid.uuid4().hex[:12]}"
        sess = runtime.open_call(call_id)
        try:
            events = []
            for turn in body.transcript:
                if turn.text and turn.text.strip():
                    sess.push_transcript(turn.text, float(turn.t))
                    # Drive one 50 ms silence hop so fusion emits after the
                    # linguistic update (same pump as the lab /transcript route).
                    events = sess.push_audio(np.zeros(400, dtype=np.float32), 8000)
            if body.audio is not None:
                try:
                    samples = _pcm_to_float(body.audio.pcm_s16le_b64)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                t0 = time.perf_counter()
                events = sess.push_audio(samples, int(body.audio.sr) or 8000)
                runtime.observe_frame_ms((time.perf_counter() - t0) * 1000.0)
            out = _risk_dict(sess, events)
        finally:
            runtime.close_call(call_id)
        out["token_id"] = tid
        out["client_ref"] = body.client_ref
        log.info(
            "hosted analyze token_id=%s client_ref=%s tier=%s risk=%.3f turns=%d",
            tid,
            body.client_ref,
            out["tier"],
            out["risk_score"],
            out["n_turns"],
        )
        return out

    app.include_router(router)
    app.state.hosted_auth = auth
