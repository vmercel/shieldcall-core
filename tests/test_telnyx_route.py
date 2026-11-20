"""Telnyx ingest websocket route (ADR-011).

Covers the /v1/ingest/telnyx route in shieldcall.serve.http_app: the
per-integration query token, the pre-opened-call gate (P2-6b), the
call_control_id match, and the full connected -> start -> media ->
stop flow into a live session, in both PCMU and PCMA. Protocol
shapes follow Telnyx's docs; the pure translation is covered by
tests/test_telnyx_ingest.py.
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

CALL_ID = "v2:T02llQxIyaRkhfRKxgAP8nY511EhFLizdvdUKJiSw8d6A9BborherQ"
STREAM_ID = "32DE0DEA-53CB-4B21-89A4-9E1819C043BC"
SILENCE_20MS_PCMU = base64.b64encode(bytes([0xFF]) * 160).decode("ascii")
# A-law silence is 0xD5.
SILENCE_20MS_PCMA = base64.b64encode(bytes([0xD5]) * 160).decode("ascii")


def _client():
    cfg = PipelineConfig(channel=None)
    rt = SidecarRuntime(max_calls=4, pipeline_config=cfg)
    return TestClient(create_app(rt)), rt


def _connected():
    return {"event": "connected", "version": "1.0.0"}


def _start(call_id=CALL_ID, encoding="PCMU"):
    return {
        "event": "start",
        "sequence_number": "1",
        "start": {
            "user_id": "3E6F995F-85F7-4705-9741-53B116D28237",
            "call_control_id": call_id,
            "call_session_id": "ff55a038-6f5d-11ef-9692-02420aeffb1f",
            "from": "+13122010094",
            "to": "+13122123456",
            "tags": ["TAG1"],
            "client_state": "aGF2ZSBhIG5pY2UgZGF5ID1d",
            "media_format": {
                "encoding": encoding,
                "sample_rate": 8000,
                "channels": 1,
            },
        },
        "stream_id": STREAM_ID,
    }


def _media(chunk=1, timestamp_rtp=0, payload=SILENCE_20MS_PCMU):
    return {
        "event": "media",
        "sequence_number": str(chunk + 3),
        "media": {
            "track": "inbound",
            "chunk": str(chunk),
            "timestamp": str(timestamp_rtp),
            "payload": payload,
        },
        "stream_id": STREAM_ID,
    }


def _stop():
    return {
        "event": "stop",
        "sequence_number": "5",
        "stop": {"call_control_id": CALL_ID},
        "stream_id": STREAM_ID,
    }


def _close_code(c, url, send=None, **kwargs):
    """Connect, optionally send one message, return the close code."""
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with c.websocket_connect(url, **kwargs) as ws:
            if send is not None:
                ws.send_json(send)
            ws.receive_json()
    return excinfo.value.code


def _telnyx_token_env(monkeypatch, token="telnyx-secret"):
    # The app itself must boot: keep a sidecar token so create_app's
    # fail-closed startup check passes, and delete the lab hatch so the
    # Telnyx route's own auth decision is what the test exercises.
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "sidecar-secret")
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("SHIELDCALL_TELNYX_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_TELNYX_TOKENS", f"telnyx:{token}")


# -- auth -----------------------------------------------------------------


def test_token_required_when_configured(monkeypatch):
    _telnyx_token_env(monkeypatch)
    c, _ = _client()
    assert _close_code(c, f"/v1/ingest/telnyx?call_id={CALL_ID}") == 4401
    assert (
        _close_code(c, f"/v1/ingest/telnyx?call_id={CALL_ID}&token=wrong") == 4401
    )


def test_no_telnyx_tokens_configured_closes_4401(monkeypatch):
    # App boots on its sidecar token, but the Telnyx route has no
    # per-integration token configured and the lab hatch is off: fail closed.
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "sidecar-secret")
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("SHIELDCALL_TELNYX_TOKENS", raising=False)
    c, _ = _client()
    assert _close_code(c, f"/v1/ingest/telnyx?call_id={CALL_ID}") == 4401


def test_wrong_token_closes_4401_before_call_gate(monkeypatch):
    _telnyx_token_env(monkeypatch)
    c, _ = _client()
    # Unknown call id too, but auth is checked first: still 4401.
    assert (
        _close_code(c, "/v1/ingest/telnyx?call_id=ghost&token=wrong") == 4401
    )


def test_valid_token_reaches_call_gate(monkeypatch):
    _telnyx_token_env(monkeypatch)
    c, rt = _client()
    assert (
        _close_code(c, "/v1/ingest/telnyx?call_id=ghost&token=telnyx-secret")
        == 4404
    )
    assert rt.get_call("ghost") is None


# -- call gate ------------------------------------------------------------


def test_unknown_call_closed_4404_and_no_session_created():
    c, rt = _client()  # lab hatch on via conftest: no Telnyx tokens needed
    assert _close_code(c, f"/v1/ingest/telnyx?call_id=ghost") == 4404
    assert rt.get_call("ghost") is None


def test_missing_call_id_closed_4404():
    c, _ = _client()
    assert _close_code(c, "/v1/ingest/telnyx") == 4404


# -- happy path -----------------------------------------------------------


def test_full_stream_flows_into_session_and_stop_closes_1000():
    c, rt = _client()
    cid = c.post("/v1/calls", json={"call_id": CALL_ID}).json()["call_id"]
    with c.websocket_connect(f"/v1/ingest/telnyx?call_id={cid}") as ws:
        ws.send_json(_connected())
        ws.send_json(_start())
        ws.send_json(_media())
        ev = ws.receive_json()
        assert ev["type"] == "event"
        assert ev["call_id"] == cid
        # Second payload: 160 samples, so the RTP timestamp advances by 160.
        ws.send_json(_media(chunk=2, timestamp_rtp=160))
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


def test_pcma_stream_flows_into_session():
    c, rt = _client()
    cid = c.post("/v1/calls", json={"call_id": CALL_ID}).json()["call_id"]
    with c.websocket_connect(f"/v1/ingest/telnyx?call_id={cid}") as ws:
        ws.send_json(_connected())
        ws.send_json(_start(encoding="PCMA"))
        ws.send_json(_media(payload=SILENCE_20MS_PCMA))
        ev = ws.receive_json()
        assert ev["type"] == "event"
        # A second payload: the pipeline's 25 ms analysis window needs
        # more than one 20 ms message before it emits a frame, and the
        # RTP timestamp advances by 160 samples per payload.
        ws.send_json(_media(chunk=2, timestamp_rtp=160, payload=SILENCE_20MS_PCMA))
        ev = ws.receive_json()
        assert ev["type"] == "event"
        ws.send_json(_stop())
        final = ws.receive_json()
        assert final["type"] == "event"
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
        assert excinfo.value.code == 1000
    sess = rt.get_call(cid)
    assert sess is not None
    assert sess.frames > 0
    c.delete(f"/v1/calls/{cid}")


def test_call_control_id_mismatch_sends_error_and_closes_4400():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    with c.websocket_connect(f"/v1/ingest/telnyx?call_id={CALL_ID}") as ws:
        ws.send_json(_connected())
        ws.send_json(_start(call_id="v2:other-call-control-id"))
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "call_control_id" in err["detail"]
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
        assert excinfo.value.code == 4400


def test_malformed_media_sends_error_and_closes_4400():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    with c.websocket_connect(f"/v1/ingest/telnyx?call_id={CALL_ID}") as ws:
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
    with c.websocket_connect(f"/v1/ingest/telnyx?call_id={CALL_ID}") as ws:
        ws.send_json(_media())
        err = ws.receive_json()
        assert err["type"] == "error"
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
        assert excinfo.value.code == 4400


def test_telnyx_error_frame_sends_error_and_closes_4400():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    with c.websocket_connect(f"/v1/ingest/telnyx?call_id={CALL_ID}") as ws:
        ws.send_json(_connected())
        ws.send_json(_start())
        ws.send_json(
            {"event": "error",
             "payload": {"code": 100003, "title": "malformed_frame",
                         "detail": "bad frame"},
             "stream_id": STREAM_ID}
        )
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "100003" in err["detail"]
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
        assert excinfo.value.code == 4400


def test_capabilities_lists_telnyx_ingest():
    c, _ = _client()
    body = c.get("/v1/capabilities").json()
    assert "WS /v1/ingest/telnyx" in body["endpoints"]
