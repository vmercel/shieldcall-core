"""In-app VoIP/WebRTC ingest websocket route (ADR-015).

Covers the /v1/ingest/inapp route in shieldcall.serve.http_app:
the per-integration query token, the pre-opened-call gate (P2-6b),
and the hello -> Opus media -> stop flow into a live session.
The pure translation is covered by tests/test_inapp_ingest.py;
the Opus packets here are real libopus output, skipped where
libopus is absent.
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.util

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve.http_app import create_app

CALL_ID = "aaaaaaaa-bbbb-cccc-dddd-0123456789ab"

_libopus_available = ctypes.util.find_library("opus") is not None


def _client():
    cfg = PipelineConfig(channel=None)
    rt = SidecarRuntime(max_calls=4, pipeline_config=cfg)
    return TestClient(create_app(rt)), rt


def _encode_sine(rate: int = 48000, freq: float = 440.0):
    lib = ctypes.CDLL(ctypes.util.find_library("opus"))
    lib.opus_encoder_create.argtypes = [
        ctypes.c_int32,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.opus_encoder_create.restype = ctypes.c_void_p
    lib.opus_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_short),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.opus_encode.restype = ctypes.c_int
    lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_encoder_destroy.restype = None
    err = ctypes.c_int(0)
    handle = lib.opus_encoder_create(rate, 1, 2048, ctypes.byref(err))
    assert handle and err.value == 0
    try:
        n = rate // 50  # 20 ms
        t = np.arange(n) / rate
        pcm = (0.4 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        out = ctypes.create_string_buffer(4000)
        count = lib.opus_encode(
            handle,
            pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
            n,
            out,
            4000,
        )
        assert count > 0
        return out.raw[:count]
    finally:
        lib.opus_encoder_destroy(handle)


def _hello(**extra):
    msg = {
        "type": "hello",
        "codec": "opus",
        "sample_rate": 48000,
        "channels": 1,
        "track": "far-end",
    }
    msg.update(extra)
    return msg


def _media(seq: int, timestamp: int, packet: bytes):
    return {
        "type": "media",
        "seq": seq,
        "timestamp": timestamp,
        "payload": base64.b64encode(packet).decode("ascii"),
    }


def _close_code(c, url, send=None):
    """Connect, optionally send one JSON message, return the close code.

    The route sends an error JSON before closing on a protocol
    error, so the first receive returns that JSON and the second
    raises WebSocketDisconnect with the close code.
    """
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with c.websocket_connect(url) as ws:
            if send is not None:
                ws.send_json(send)
            ws.receive_json()
            ws.receive_json()
    return excinfo.value.code


def _inapp_token_env(monkeypatch, token="inapp-secret"):
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "sidecar-secret")
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("SHIELDCALL_INAPP_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_INAPP_TOKENS", f"inapp:{token}")


# -- auth -------------------------------------------------------------------


def test_token_required_when_configured(monkeypatch):
    _inapp_token_env(monkeypatch)
    c, _ = _client()
    assert _close_code(c, f"/v1/ingest/inapp?call_id={CALL_ID}") == 4401
    assert (
        _close_code(
            c, f"/v1/ingest/inapp?call_id={CALL_ID}&token=wrong"
        )
        == 4401
    )


def test_no_inapp_tokens_configured_closes_4401(monkeypatch):
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "sidecar-secret")
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("SHIELDCALL_INAPP_TOKENS", raising=False)
    c, _ = _client()
    assert _close_code(c, f"/v1/ingest/inapp?call_id={CALL_ID}") == 4401


def test_wrong_token_closes_4401_before_call_gate(monkeypatch):
    _inapp_token_env(monkeypatch)
    c, _ = _client()
    assert (
        _close_code(c, "/v1/ingest/inapp?call_id=ghost&token=wrong") == 4401
    )


def test_valid_token_reaches_call_gate(monkeypatch):
    _inapp_token_env(monkeypatch)
    c, rt = _client()
    assert (
        _close_code(
            c, "/v1/ingest/inapp?call_id=ghost&token=inapp-secret"
        )
        == 4404
    )
    assert rt.get_call("ghost") is None


# -- call gate --------------------------------------------------------------


def test_unknown_call_closed_4404_and_no_session_created():
    c, rt = _client()  # lab hatch on via conftest: no tokens needed
    assert _close_code(c, "/v1/ingest/inapp?call_id=ghost") == 4404
    assert rt.get_call("ghost") is None


def test_missing_call_id_closed_4404():
    c, _ = _client()
    assert _close_code(c, "/v1/ingest/inapp") == 4404


# -- happy path -------------------------------------------------------------


@pytest.mark.skipif(not _libopus_available, reason="libopus not installed")
def test_full_stream_flows_into_session_and_stop_closes_1000():
    c, rt = _client()
    cid = c.post("/v1/calls", json={"call_id": CALL_ID}).json()["call_id"]
    packet = _encode_sine()
    with c.websocket_connect(f"/v1/ingest/inapp?call_id={cid}") as ws:
        ws.send_json(_hello())
        ws.send_json(_media(0, 0, packet))
        event = ws.receive_json()
        assert isinstance(event, dict)
        ws.send_json(_media(1, 960, packet))
        ws.receive_json()
        ws.send_json({"type": "stop"})
        ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
        assert excinfo.value.code == 1000
    sess = rt.get_call(cid)
    assert sess is not None
    assert sess.frames > 0
    c.delete(f"/v1/calls/{cid}")


@pytest.mark.skipif(not _libopus_available, reason="libopus not installed")
def test_hello_call_id_mismatch_closes_4400():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    code = _close_code(
        c,
        f"/v1/ingest/inapp?call_id={CALL_ID}",
        send=_hello(call_id="another-call"),
    )
    assert code == 4400
    c.delete(f"/v1/calls/{CALL_ID}")


@pytest.mark.skipif(not _libopus_available, reason="libopus not installed")
def test_media_before_hello_closes_4400():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    code = _close_code(
        c,
        f"/v1/ingest/inapp?call_id={CALL_ID}",
        send=_media(0, 0, _encode_sine()),
    )
    assert code == 4400
    c.delete(f"/v1/calls/{CALL_ID}")


def test_capabilities_lists_inapp_ingest():
    c, _ = _client()
    body = c.get("/v1/capabilities").json()
    assert "WS /v1/ingest/inapp" in body["endpoints"]
