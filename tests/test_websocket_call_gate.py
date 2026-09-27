"""Websocket call gate (P2-6b).

The websocket used to open a call session implicitly for an unknown
call_id. That bypassed the per-token new-call quota enforced on
POST /v1/calls (persistent store, P2-6a) and allowed unbounded session
creation from a single connection. The websocket must now only attach
to a call that was opened through POST /v1/calls; unknown ids are
rejected with close code 4404 and no session is created.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve.http_app import create_app


def _client():
    cfg = PipelineConfig(channel=None)
    rt = SidecarRuntime(max_calls=4, pipeline_config=cfg)
    return TestClient(create_app(rt)), rt


def _connect_unknown(c, cid, **kwargs):
    """Connect a websocket to an unknown call_id; return the close code.

    The server closes immediately after accept, so the disconnect
    surfaces on the first receive, not on send.
    """
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with c.websocket_connect(f"/v1/calls/{cid}/stream", **kwargs) as ws:
            ws.send_json({"type": "transcript", "t": 0.0, "text": "hello"})
            ws.receive_json()
    return excinfo.value.code


def test_unknown_call_id_closed_4404_and_no_session_created():
    c, rt = _client()
    cid = "ghost-call-that-was-never-opened"
    assert rt.get_call(cid) is None
    code = _connect_unknown(c, cid)
    assert code == 4404
    # The rejected connection must not have materialised a session.
    assert rt.get_call(cid) is None


def test_known_call_still_streams_after_post_open():
    c, rt = _client()
    cid = c.post("/v1/calls", json={"call_id": "ws-known"}).json()["call_id"]
    with c.websocket_connect(f"/v1/calls/{cid}/stream") as ws:
        ws.send_json({"type": "transcript", "t": 0.5, "text": "hello from a real call"})
        ev = ws.receive_json()
    assert ev["type"] == "event"
    assert ev["call_id"] == cid
    c.delete(f"/v1/calls/{cid}")


def _token_env(monkeypatch, token="ws-secret"):
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKEN", raising=False)
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", token)


def _auth(token):
    return {"headers": {"Authorization": f"Bearer {token}"}}


def test_auth_checked_before_call_gate(monkeypatch):
    _token_env(monkeypatch)
    c, _ = _client()
    # Missing / wrong Bearer <redacted> still closes with 4401, even for an unknown id.
    assert _connect_unknown(c, "nope") == 4401
    assert _connect_unknown(c, "nope", **_auth("wrong")) == 4401
    # A valid token reaches the gate: unknown id -> 4404, not a session.
    assert _connect_unknown(c, "nope", **_auth("ws-secret")) == 4404


def test_authenticated_known_call_streams(monkeypatch):
    _token_env(monkeypatch)
    c, _ = _client()
    cid = c.post("/v1/calls", json={"call_id": "ws-authed"}, **_auth("ws-secret")).json()[
        "call_id"
    ]
    with c.websocket_connect(
        f"/v1/calls/{cid}/stream", **_auth("ws-secret")
    ) as ws:
        ws.send_json({"type": "transcript", "t": 0.1, "text": "authenticated stream"})
        ev = ws.receive_json()
    assert ev["type"] == "event"
    assert ev["call_id"] == cid
