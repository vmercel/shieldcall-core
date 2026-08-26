"""FastAPI worker around SidecarRuntime.

The telephone (or Expo lab loopback) does not traverse this process.
Channel twin is off. Recommend-only: this API never hangs up a call.
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).resolve().parent / "static"

from ..eval.corpora.independent_scripts import independent_scripts
from ..pipeline import PipelineConfig
from ..runtime.runtime import SidecarRuntime
from ..runtime.session import CallSession, SessionEvent

TOKEN = os.environ.get("SHIELDCALL_SIDECAR_TOKEN", "").strip()


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


def _check_token(authorization: Optional[str]) -> None:
    if not TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    if authorization.split(" ", 1)[1] != TOKEN:
        raise HTTPException(status_code=403, detail="bad token")


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
    cfg = PipelineConfig(channel=None, use_conformal=True, fuse_every_n_frames=5)
    rt = runtime or SidecarRuntime(max_calls=8, pipeline_config=cfg)
    if rt.pipeline_config.channel is not None:
        raise RuntimeError("live sidecar must not enable the channel twin")

    app = FastAPI(title="ShieldCall sidecar", version="0.7.0-mvp")
    app.state.runtime = rt
    origins = os.environ.get("SHIELDCALL_CORS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins if o.strip()],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
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
        h = rt.health()
        return {
            "live": h.live,
            "ready": h.ready,
            "active_calls": h.active_calls,
            "max_calls": h.max_calls,
            "shed_total": h.shed_total,
            "detail": h.detail,
            "channel_twin": False,
            "actuation": "recommend_only",
        }

    @app.post("/v1/calls")
    def open_call(body: OpenCallBody, authorization: Optional[str] = Header(default=None)):
        _check_token(authorization)
        cid = (body.call_id or "").strip() or f"lab-{uuid.uuid4().hex[:12]}"
        sess = rt.open_call(cid)
        return {"call_id": sess.call_id, "shed": sess.shed}

    @app.delete("/v1/calls/{call_id}")
    def close_call(call_id: str, authorization: Optional[str] = Header(default=None)):
        _check_token(authorization)
        trace = rt.close_call(call_id)
        return {"call_id": call_id, "n_decisions": len(trace)}

    @app.get("/v1/calls/{call_id}/trace")
    def trace(call_id: str, authorization: Optional[str] = Header(default=None)):
        _check_token(authorization)
        sess = rt.get_call(call_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="unknown call_id")
        return {"call_id": call_id, "trace": sess.agent.trace_dicts()}

    @app.post("/v1/calls/{call_id}/transcript")
    def transcript(
        call_id: str,
        body: TranscriptBody,
        authorization: Optional[str] = Header(default=None),
    ):
        _check_token(authorization)
        sess = rt.get_call(call_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="unknown call_id")
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
        _check_token(authorization)
        sess = rt.get_call(call_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="unknown call_id")
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
        _check_token(authorization)
        sess = rt.get_call(call_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="unknown call_id")
        turns = _script_turns(body.script_id, body.turns)
        sr = 8000
        last = None
        for i, text in enumerate(turns):
            sess.push_transcript(text, float(i) * 0.8)
            noise = (0.01 * np.random.RandomState(i).randn(sr // 5)).astype(np.float32)
            last = sess.push_audio(noise, sr)
        return {"call_id": call_id, "n_turns": len(turns), **_last_event_dict(sess, last)}

    @app.websocket("/v1/calls/{call_id}/stream")
    async def stream(ws: WebSocket, call_id: str):
        await ws.accept()
        if TOKEN:
            proto = ws.headers.get("authorization") or ""
            try:
                _check_token(proto)
            except HTTPException:
                await ws.close(code=4401)
                return
        sess = rt.get_call(call_id)
        if sess is None:
            sess = rt.open_call(call_id)
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

    return app


app = create_app()
