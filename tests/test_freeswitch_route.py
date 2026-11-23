"""FreeSWITCH mod_audio_fork ingest websocket route (ADR-014).

Covers the /v1/ingest/freeswitch route in shieldcall.serve.http_app:
the per-integration query token, the pre-opened-call gate (P2-6b),
and the full metadata -> binary L16 audio -> socket close flow
into a live session. Protocol shapes follow the module's README;
the pure translation is covered by tests/test_freeswitch_ingest.py.

The route sends nothing back over the socket (ADR-014 item 10):
mod_audio_fork reads server JSON as control events, so the error
cases here assert that the first receive raises WebSocketDisconnect
directly, with no error JSON preceding the close.
"""

from __future__ import annotations

import json
import struct

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve.http_app import create_app

CALL_ID = "aaaaaaaa-bbbb-cccc-dddd-0123456789ab"
# 20 ms of L16 silence at 8 kHz mono: 160 samples, 320 bytes.
SILENCE_20MS = struct.pack("<160h", *([0] * 160))
# 20 ms of L16 silence at 8 kHz stereo: 160 interleaved pairs, 640 bytes.
SILENCE_20MS_STEREO = struct.pack("<320h", *([0] * 320))


def _client():
    cfg = PipelineConfig(channel=None)
    rt = SidecarRuntime(max_calls=4, pipeline_config=cfg)
    return TestClient(create_app(rt)), rt


def _metadata(**extra):
    msg = {"sample_rate": 8000, "mix_type": "mono"}
    msg.update(extra)
    return json.dumps(msg)


def _close_code(c, url, send_text=None, send_bytes=None):
    """Connect, optionally send one message, return the close code.

    The route never sends JSON, so the first receive after the
    server closes raises WebSocketDisconnect directly. If the
    route ever started sending frames back, this helper would
    return the sent JSON instead of raising, and the test would
    fail on the attribute access: that is the guard.
    """
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with c.websocket_connect(url) as ws:
            if send_text is not None:
                ws.send_text(send_text)
            if send_bytes is not None:
                ws.send_bytes(send_bytes)
            ws.receive_json()
    return excinfo.value.code


def _freeswitch_token_env(monkeypatch, token="freeswitch-secret"):
    # The app itself must boot: keep a sidecar token so create_app's
    # fail-closed startup check passes, and delete the lab hatch so the
    # FreeSWITCH route's own auth decision is what the test exercises.
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "sidecar-secret")
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("SHIELDCALL_FREESWITCH_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_FREESWITCH_TOKENS", f"freeswitch:{token}")


# -- auth -----------------------------------------------------------------


def test_token_required_when_configured(monkeypatch):
    _freeswitch_token_env(monkeypatch)
    c, _ = _client()
    assert _close_code(c, f"/v1/ingest/freeswitch?call_id={CALL_ID}") == 4401
    assert (
        _close_code(
            c, f"/v1/ingest/freeswitch?call_id={CALL_ID}&token=<redacted>"
        )
        == 4401
    )


def test_no_freeswitch_tokens_configured_closes_4401(monkeypatch):
    # App boots on its sidecar token, but the FreeSWITCH route has no
    # per-integration token configured and the lab hatch is off: fail closed.
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "sidecar-secret")
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("SHIELDCALL_FREESWITCH_TOKENS", raising=False)
    c, _ = _client()
    assert _close_code(c, f"/v1/ingest/freeswitch?call_id={CALL_ID}") == 4401


def test_wrong_token_closes_4401_before_call_gate(monkeypatch):
    _freeswitch_token_env(monkeypatch)
    c, _ = _client()
    # Unknown call id too, but auth is checked first: still 4401.
    assert (
        _close_code(c, "/v1/ingest/freeswitch?call_id=ghost&token=<redacted>")
        == 4401
    )


def test_valid_token_reaches_call_gate(monkeypatch):
    _freeswitch_token_env(monkeypatch)
    c, rt = _client()
    assert (
        _close_code(
            c, "/v1/ingest/freeswitch?call_id=ghost&token=freeswitch-secret"
        )
        == 4404
    )
    assert rt.get_call("ghost") is None


# -- call gate ------------------------------------------------------------


def test_unknown_call_closed_4404_and_no_session_created():
    c, rt = _client()  # lab hatch on via conftest: no FreeSWITCH tokens needed
    assert _close_code(c, "/v1/ingest/freeswitch?call_id=ghost") == 4404
    assert rt.get_call("ghost") is None


def test_missing_call_id_closed_4404():
    c, _ = _client()
    assert _close_code(c, "/v1/ingest/freeswitch") == 4404


# -- happy path -----------------------------------------------------------


def test_full_stream_flows_into_session_and_close_ends_it():
    c, rt = _client()
    cid = c.post("/v1/calls", json={"call_id": CALL_ID}).json()["call_id"]
    with c.websocket_connect(f"/v1/ingest/freeswitch?call_id={cid}") as ws:
        ws.send_text(_metadata())
        ws.send_bytes(SILENCE_20MS)
        # A mid-call text frame (a DTMF notification, say) must not
        # break the stream, and the next packet still flows.
        ws.send_text('{"event": "dtmf", "digit": "5"}')
        ws.send_bytes(SILENCE_20MS)
        # The route sends nothing back: no receive here. The socket
        # closing ends the stream, as for Vonage.
    sess = rt.get_call(cid)
    assert sess is not None
    assert sess.frames > 0
    c.delete(f"/v1/calls/{cid}")


def test_stereo_stream_flows_into_session():
    c, rt = _client()
    cid = c.post("/v1/calls", json={"call_id": CALL_ID}).json()["call_id"]
    with c.websocket_connect(f"/v1/ingest/freeswitch?call_id={cid}") as ws:
        ws.send_text(_metadata(mix_type="stereo"))
        ws.send_bytes(SILENCE_20MS_STEREO)
        ws.send_bytes(SILENCE_20MS_STEREO)
    sess = rt.get_call(cid)
    assert sess is not None
    assert sess.frames > 0
    c.delete(f"/v1/calls/{cid}")


def test_metadata_call_id_matching_uri_binding_flows():
    c, rt = _client()
    cid = c.post("/v1/calls", json={"call_id": CALL_ID}).json()["call_id"]
    with c.websocket_connect(f"/v1/ingest/freeswitch?call_id={cid}") as ws:
        ws.send_text(_metadata(call_id=CALL_ID))
        ws.send_bytes(SILENCE_20MS)
        ws.send_bytes(SILENCE_20MS)
    sess = rt.get_call(cid)
    assert sess is not None
    assert sess.frames > 0
    c.delete(f"/v1/calls/{cid}")


def test_metadata_call_id_mismatch_closes_4400_with_no_json():
    c, rt = _client()
    cid = c.post("/v1/calls", json={"call_id": CALL_ID}).json()["call_id"]
    # A fork whose metadata names a different call is refused with
    # 4400 and no JSON first: the first receive raises directly.
    assert (
        _close_code(
            c,
            f"/v1/ingest/freeswitch?call_id={cid}",
            send_text=_metadata(call_id="another-call"),
        )
        == 4400
    )
    c.delete(f"/v1/calls/{cid}")


# -- protocol errors ------------------------------------------------------


def test_audio_before_metadata_closes_4400_with_no_json():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    assert (
        _close_code(
            c,
            f"/v1/ingest/freeswitch?call_id={CALL_ID}",
            send_bytes=SILENCE_20MS,
        )
        == 4400
    )


def test_malformed_metadata_closes_4400_with_no_json():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    assert (
        _close_code(
            c,
            f"/v1/ingest/freeswitch?call_id={CALL_ID}",
            send_text="this is not json",
        )
        == 4400
    )


def test_mixed_metadata_closes_4400_with_no_json():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    assert (
        _close_code(
            c,
            f"/v1/ingest/freeswitch?call_id={CALL_ID}",
            send_text=_metadata(mix_type="mixed"),
        )
        == 4400
    )


def test_sixteen_k_metadata_closes_4400_with_no_json():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    assert (
        _close_code(
            c,
            f"/v1/ingest/freeswitch?call_id={CALL_ID}",
            send_text=_metadata(sample_rate=16000),
        )
        == 4400
    )


def test_odd_byte_packet_closes_4400_with_no_json():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    with c.websocket_connect(f"/v1/ingest/freeswitch?call_id={CALL_ID}") as ws:
        ws.send_text(_metadata())
        ws.send_bytes(b"\x00\x00\x00")
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
        assert excinfo.value.code == 4400


def test_capabilities_lists_freeswitch_ingest():
    c, _ = _client()
    body = c.get("/v1/capabilities").json()
    assert "WS /v1/ingest/freeswitch" in body["endpoints"]
