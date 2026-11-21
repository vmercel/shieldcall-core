"""Vonage ingest websocket route (ADR-012).

Covers the /v1/ingest/vonage route in shieldcall.serve.http_app: the
per-integration query token, the pre-opened-call gate (P2-6b), and
the full websocket:connected -> binary L16 audio -> socket close
flow into a live session. Protocol shapes follow Vonage's docs;
the pure translation is covered by tests/test_vonage_ingest.py.
"""

from __future__ import annotations

import struct

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve.http_app import create_app

CALL_ID = "aaaaaaaa-bbbb-cccc-dddd-0123456789ab"
# 20 ms of L16 silence at 8 kHz: 160 samples, 320 bytes.
SILENCE_20MS = struct.pack("<160h", *([0] * 160))


def _client():
    cfg = PipelineConfig(channel=None)
    rt = SidecarRuntime(max_calls=4, pipeline_config=cfg)
    return TestClient(create_app(rt)), rt


def _connected(content_type="audio/l16;rate=8000"):
    return {"event": "websocket:connected", "content-type": content_type}


def _close_code(c, url, send=None, **kwargs):
    """Connect, optionally send one JSON message, return the close code."""
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with c.websocket_connect(url, **kwargs) as ws:
            if send is not None:
                ws.send_json(send)
            ws.receive_json()
    return excinfo.value.code


def _close_code_raw(c, url, send_bytes=None, send_text=None):
    """Connect, optionally send one raw message, return the close code.

    The route sends an error JSON before closing with 4400, so the
    first receive returns that JSON and only the second receive
    raises WebSocketDisconnect. A close with no message (4401)
    raises on the first receive instead.
    """
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with c.websocket_connect(url) as ws:
            if send_bytes is not None:
                ws.send_bytes(send_bytes)
            if send_text is not None:
                ws.send_text(send_text)
            ws.receive_json()
            ws.receive_json()
    return excinfo.value.code


def _vonage_token_env(monkeypatch, token="vonage-secret"):
    # The app itself must boot: keep a sidecar token so create_app's
    # fail-closed startup check passes, and delete the lab hatch so the
    # Vonage route's own auth decision is what the test exercises.
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "sidecar-secret")
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("SHIELDCALL_VONAGE_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_VONAGE_TOKENS", f"vonage:{token}")


# -- auth -----------------------------------------------------------------


def test_token_required_when_configured(monkeypatch):
    _vonage_token_env(monkeypatch)
    c, _ = _client()
    assert _close_code(c, f"/v1/ingest/vonage?call_id={CALL_ID}") == 4401
    assert (
        _close_code(c, f"/v1/ingest/vonage?call_id={CALL_ID}&token=wrong") == 4401
    )


def test_no_vonage_tokens_configured_closes_4401(monkeypatch):
    # App boots on its sidecar token, but the Vonage route has no
    # per-integration token configured and the lab hatch is off: fail closed.
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "sidecar-secret")
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("SHIELDCALL_VONAGE_TOKENS", raising=False)
    c, _ = _client()
    assert _close_code(c, f"/v1/ingest/vonage?call_id={CALL_ID}") == 4401


def test_wrong_token_closes_4401_before_call_gate(monkeypatch):
    _vonage_token_env(monkeypatch)
    c, _ = _client()
    # Unknown call id too, but auth is checked first: still 4401.
    assert (
        _close_code(c, "/v1/ingest/vonage?call_id=ghost&token=wrong") == 4401
    )


def test_valid_token_reaches_call_gate(monkeypatch):
    _vonage_token_env(monkeypatch)
    c, rt = _client()
    assert (
        _close_code(c, "/v1/ingest/vonage?call_id=ghost&token=vonage-secret")
        == 4404
    )
    assert rt.get_call("ghost") is None


# -- call gate ------------------------------------------------------------


def test_unknown_call_closed_4404_and_no_session_created():
    c, rt = _client()  # lab hatch on via conftest: no Vonage tokens needed
    assert _close_code(c, f"/v1/ingest/vonage?call_id=ghost") == 4404
    assert rt.get_call("ghost") is None


def test_missing_call_id_closed_4404():
    c, _ = _client()
    assert _close_code(c, "/v1/ingest/vonage") == 4404


# -- happy path -----------------------------------------------------------


def test_full_stream_flows_into_session_and_close_ends_it():
    c, rt = _client()
    cid = c.post("/v1/calls", json={"call_id": CALL_ID}).json()["call_id"]
    with c.websocket_connect(f"/v1/ingest/vonage?call_id={cid}") as ws:
        ws.send_json(_connected())
        ws.send_bytes(SILENCE_20MS)
        ev = ws.receive_json()
        assert ev["type"] == "event"
        assert ev["call_id"] == cid
        # A DTMF event interleaved between audio packets must not
        # break the stream, and the next packet still flows.
        ws.send_json({"event": "websocket:dtmf", "digit": "5", "duration": 260})
        ws.send_bytes(SILENCE_20MS)
        ev = ws.receive_json()
        assert ev["type"] == "event"
        assert ev["call_id"] == cid
    # Exiting the context closes the socket: Vonage has no stop
    # event, so the close is the end of the stream.
    sess = rt.get_call(cid)
    assert sess is not None
    assert sess.frames > 0
    c.delete(f"/v1/calls/{cid}")


def test_reconnect_connected_flags_no_error_on_fresh_connection():
    c, rt = _client()
    cid = c.post("/v1/calls", json={"call_id": CALL_ID}).json()["call_id"]
    with c.websocket_connect(f"/v1/ingest/vonage?call_id={cid}") as ws:
        ws.send_json(
            {
                "event": "websocket:connected",
                "content-type": "audio/l16;rate=8000",
                "reconnect_attempt": 1,
            }
        )
        ws.send_bytes(SILENCE_20MS)
        ev = ws.receive_json()
        assert ev["type"] == "event"
        # The pipeline's 25 ms analysis window needs more than one
        # 20 ms packet before it emits a frame, as in the Telnyx
        # route test.
        ws.send_bytes(SILENCE_20MS)
        ev = ws.receive_json()
        assert ev["type"] == "event"
    sess = rt.get_call(cid)
    assert sess is not None
    assert sess.frames > 0
    c.delete(f"/v1/calls/{cid}")


# -- protocol errors ------------------------------------------------------


def test_unsupported_rate_sends_error_and_closes_4400():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    with c.websocket_connect(f"/v1/ingest/vonage?call_id={CALL_ID}") as ws:
        ws.send_json(_connected("audio/l16;rate=16000"))
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "sample rate" in err["detail"]
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
        assert excinfo.value.code == 4400


def test_audio_before_connected_sends_error_and_closes_4400():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    assert (
        _close_code_raw(
            c, f"/v1/ingest/vonage?call_id={CALL_ID}", send_bytes=SILENCE_20MS
        )
        == 4400
    )


def test_malformed_json_text_sends_error_and_closes_4400():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    assert (
        _close_code_raw(
            c,
            f"/v1/ingest/vonage?call_id={CALL_ID}",
            send_text="this is not json",
        )
        == 4400
    )


def test_odd_byte_packet_sends_error_and_closes_4400():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    with c.websocket_connect(f"/v1/ingest/vonage?call_id={CALL_ID}") as ws:
        ws.send_json(_connected())
        ws.send_bytes(b"\x00\x00\x00")
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "odd byte count" in err["detail"]
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
        assert excinfo.value.code == 4400


def test_capabilities_lists_vonage_ingest():
    c, _ = _client()
    body = c.get("/v1/capabilities").json()
    assert "WS /v1/ingest/vonage" in body["endpoints"]
