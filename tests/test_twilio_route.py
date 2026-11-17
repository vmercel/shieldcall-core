"""Twilio ingest websocket route (ADR-008).

Covers the /v1/ingest/twilio route in shieldcall.serve.http_app: the
per-integration query token, the pre-opened-call gate (P2-6b), the
callSid match, and the full connected -> start -> media -> stop flow
into a live session. Protocol shapes follow Twilio's docs; the pure
translation is covered by tests/test_twilio_ingest.py.
"""

from __future__ import annotations

import base64

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve.http_app import create_app

CALL_ID = "CA11111111111111111111111111111111"
STREAM_SID = "MZ22222222222222222222222222222222"
SILENCE_20MS = base64.b64encode(bytes([0xFF]) * 160).decode("ascii")


def _client():
    cfg = PipelineConfig(channel=None)
    rt = SidecarRuntime(max_calls=4, pipeline_config=cfg)
    return TestClient(create_app(rt)), rt


def _connected():
    return {"event": "connected", "protocol": "Call", "version": "1.0.0"}


def _start(call_id=CALL_ID):
    return {
        "event": "start",
        "sequenceNumber": "1",
        "start": {
            "accountSid": "AC33333333333333333333333333333333",
            "streamSid": STREAM_SID,
            "callSid": call_id,
            "tracks": ["inbound"],
            "mediaFormat": {
                "encoding": "audio/x-mulaw",
                "sampleRate": 8000,
                "channels": 1,
            },
            "customParameters": {},
        },
        "streamSid": STREAM_SID,
    }


def _media(chunk=1, timestamp_ms=0, payload=SILENCE_20MS):
    return {
        "event": "media",
        "sequenceNumber": str(chunk + 2),
        "media": {
            "track": "inbound",
            "chunk": str(chunk),
            "timestamp": str(timestamp_ms),
            "payload": payload,
        },
        "streamSid": STREAM_SID,
    }


def _stop():
    return {
        "event": "stop",
        "stop": {"callSid": CALL_ID},
        "streamSid": STREAM_SID,
    }


def _close_code(c, url, send=None, **kwargs):
    """Connect, optionally send one message, return the close code."""
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with c.websocket_connect(url, **kwargs) as ws:
            if send is not None:
                ws.send_json(send)
            ws.receive_json()
    return excinfo.value.code


def _twilio_token_env(monkeypatch, token="twilio-secret"):
    # The app itself must boot: keep a sidecar token so create_app's
    # fail-closed startup check passes, and delete the lab hatch so the
    # Twilio route's own auth decision is what the test exercises.
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "sidecar-secret")
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("SHIELDCALL_TWILIO_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_TWILIO_TOKENS", f"twilio:{token}")


# -- auth -----------------------------------------------------------------


def test_token_required_when_configured(monkeypatch):
    _twilio_token_env(monkeypatch)
    c, _ = _client()
    assert _close_code(c, f"/v1/ingest/twilio?call_id={CALL_ID}") == 4401
    assert (
        _close_code(c, f"/v1/ingest/twilio?call_id={CALL_ID}&token=wrong") == 4401
    )


def test_no_twilio_tokens_configured_closes_4401(monkeypatch):
    # App boots on its sidecar token, but the Twilio route has no
    # per-integration token configured and the lab hatch is off: fail closed.
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "sidecar-secret")
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("SHIELDCALL_TWILIO_TOKENS", raising=False)
    c, _ = _client()
    assert _close_code(c, f"/v1/ingest/twilio?call_id={CALL_ID}") == 4401


def test_wrong_token_closes_4401_before_call_gate(monkeypatch):
    _twilio_token_env(monkeypatch)
    c, _ = _client()
    # Unknown call id too, but auth is checked first: still 4401.
    assert (
        _close_code(c, "/v1/ingest/twilio?call_id=ghost&token=wrong") == 4401
    )


def test_valid_token_reaches_call_gate(monkeypatch):
    _twilio_token_env(monkeypatch)
    c, rt = _client()
    assert (
        _close_code(c, "/v1/ingest/twilio?call_id=ghost&token=twilio-secret")
        == 4404
    )
    assert rt.get_call("ghost") is None


# -- call gate ------------------------------------------------------------


def test_unknown_call_closed_4404_and_no_session_created():
    c, rt = _client()  # lab hatch on via conftest: no Twilio tokens needed
    assert _close_code(c, f"/v1/ingest/twilio?call_id=ghost") == 4404
    assert rt.get_call("ghost") is None


def test_missing_call_id_closed_4404():
    c, _ = _client()
    assert _close_code(c, "/v1/ingest/twilio") == 4404


# -- happy path -----------------------------------------------------------


def test_full_stream_flows_into_session_and_stop_closes_1000():
    c, rt = _client()
    cid = c.post("/v1/calls", json={"call_id": CALL_ID}).json()["call_id"]
    with c.websocket_connect(f"/v1/ingest/twilio?call_id={cid}") as ws:
        ws.send_json(_connected())
        ws.send_json(_start())
        ws.send_json(_media())
        ev = ws.receive_json()
        assert ev["type"] == "event"
        assert ev["call_id"] == cid
        ws.send_json(_media(chunk=2, timestamp_ms=20))
        ev = ws.receive_json()
        assert ev["type"] == "event"
        ws.send_json(_stop())
        final = ws.receive_json()
        assert final["type"] == "event"
        assert final["call_id"] == cid
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
        assert excinfo.value.code == 1000
    sess = rt.get_call(cid)
    assert sess is not None
    assert sess.frames > 0
    c.delete(f"/v1/calls/{cid}")


def test_call_sid_mismatch_sends_error_and_closes_4400():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    with c.websocket_connect(f"/v1/ingest/twilio?call_id={CALL_ID}") as ws:
        ws.send_json(_connected())
        ws.send_json(_start(call_id="CA99999999999999999999999999999999"))
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "callSid" in err["detail"]
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
        assert excinfo.value.code == 4400


def test_malformed_media_sends_error_and_closes_4400():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    with c.websocket_connect(f"/v1/ingest/twilio?call_id={CALL_ID}") as ws:
        ws.send_json(_connected())
        ws.send_json(_start())
        ws.send_json(_media(payload="not!!base64"))
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "base64" in err["detail"]
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
        assert excinfo.value.code == 4400


def test_media_before_start_sends_error_and_closes_4400():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    with c.websocket_connect(f"/v1/ingest/twilio?call_id={CALL_ID}") as ws:
        ws.send_json(_media())
        err = ws.receive_json()
        assert err["type"] == "error"
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
        assert excinfo.value.code == 4400


def test_capabilities_lists_twilio_ingest():
    c, _ = _client()
    body = c.get("/v1/capabilities").json()
    assert "WS /v1/ingest/twilio" in body["endpoints"]
