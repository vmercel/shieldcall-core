"""Amazon Connect fragment POST route (ADR-013).

Covers POST /v1/ingest/amazon-connect in
shieldcall.serve.http_app: the per-integration query token, the
pre-opened-call gate (P2-6b), and the full fragment POST flow
into a live session. The pure translation is covered by
tests/test_amazon_connect_ingest.py; the fragment builder here
is a compact copy of that file's writer.
"""

from __future__ import annotations

import struct

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve.http_app import create_app

CALL_ID = "d111f871-5fe7-40b1-b5f7-6b67060de0c2"
CLUSTER_TC = 1726677688724
VALID_TOKEN = "valid-token-1"
WRONG_TOKEN = "wrong-token-2"


def _client():
    cfg = PipelineConfig(channel=None)
    rt = SidecarRuntime(max_calls=4, pipeline_config=cfg)
    return TestClient(create_app(rt)), rt


# -- minimal EBML fragment builder -----------------------------------------

def _size_vint(n: int) -> bytes:
    for length in range(1, 9):
        if n < (1 << (7 * length)) - 1:
            return ((1 << (7 * length)) + n).to_bytes(length, "big")
    raise ValueError("size too large")


def _el(element_id: int, payload: bytes, id_len: int = 1) -> bytes:
    return element_id.to_bytes(id_len, "big") + _size_vint(len(payload)) + payload


def _uint(element_id: int, value: int, id_len: int = 1) -> bytes:
    return _el(element_id, value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big"), id_len)


def _str(element_id: int, value: str, id_len: int = 1) -> bytes:
    return _el(element_id, value.encode("utf-8"), id_len)


def _simple_block(track: int, relative_ms: int, n_samples: int) -> bytes:
    payload = bytes([0x80 | track]) + struct.pack(">h", relative_ms) + b"\x00"
    payload += struct.pack(f"<{n_samples}h", *([0] * n_samples))
    return _el(0xA3, payload)


def _track_entry(number: int, name: str) -> bytes:
    body = (
        _uint(0xD7, number)
        + _uint(0x73C5, number, id_len=2)
        + _uint(0x83, 2)
        + _str(0x536E, name, id_len=2)
        + _str(0x86, "A_AAC")
    )
    return _el(0xAE, body)


def _simple_tag(name: str, value: str) -> bytes:
    body = _str(0x45A3, name, id_len=2) + _str(0x4487, value, id_len=2)
    return _el(0x67C8, body, id_len=2)


def _fragment(blocks, contact_id=CALL_ID):
    ebml = _el(0x1A45DFA3, _str(0x4282, "matroska", id_len=2), id_len=4)
    tracks = _el(
        0x1654AE6B,
        _track_entry(1, "AUDIO_FROM_CUSTOMER")
        + _track_entry(2, "AUDIO_TO_CUSTOMER"),
        id_len=4,
    )
    tags_body = b""
    for name, value in (
        ("MimeType", "audio/L16;rate=8000;channels=1;"),
        ("ContactId", contact_id),
        ("AWS_KINESISVIDEO_FRAGMENT_NUMBER", "9134"),
    ):
        tags_body += _el(
            0x1254C367, _el(0x7373, _simple_tag(name, value), id_len=2), id_len=4
        )
    cluster_body = _uint(0xE7, CLUSTER_TC)
    for track, relative_ms, n_samples in blocks:
        cluster_body += _simple_block(track, relative_ms, n_samples)
    cluster = _el(0x1F43B675, cluster_body, id_len=4)
    segment = _el(0x18538067, tracks + tags_body + cluster, id_len=4)
    return ebml + segment


def _url(call_id=CALL_ID, token=None):
    url = f"/v1/ingest/amazon-connect?call_id={call_id}"
    if token is not None:
        url += f"&token={token}"
    return url


def _token_env(monkeypatch, token=VALID_TOKEN):
    # The app itself must boot: keep a sidecar token so create_app's
    # fail-closed startup check passes, and delete the lab hatch so
    # the route's own auth decision is what the test exercises.
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "sidecar-secret")
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("SHIELDCALL_AMAZON_CONNECT_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_AMAZON_CONNECT_TOKENS", f"connect:{token}")


# -- auth -------------------------------------------------------------------

def test_token_required_when_configured(monkeypatch):
    _token_env(monkeypatch)
    c, _ = _client()
    body = _fragment([(1, 0, 160)])
    assert c.post(_url(), content=body).status_code == 401
    assert c.post(_url(token=WRONG_TOKEN), content=body).status_code == 401


def test_no_tokens_configured_refuses_401(monkeypatch):
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKEN", "sidecar-secret")
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("SHIELDCALL_AMAZON_CONNECT_TOKENS", raising=False)
    c, _ = _client()
    assert c.post(_url(), content=b"junk").status_code == 401


def test_wrong_token_checked_before_call_gate(monkeypatch):
    _token_env(monkeypatch)
    c, rt = _client()
    # Unknown call id too, but auth is checked first: still 401.
    resp = c.post(_url(call_id="ghost", token=WRONG_TOKEN), content=b"junk")
    assert resp.status_code == 401
    assert rt.get_call("ghost") is None


def test_valid_token_reaches_call_gate(monkeypatch):
    _token_env(monkeypatch)
    c, rt = _client()
    resp = c.post(_url(call_id="ghost", token=VALID_TOKEN), content=b"junk")
    assert resp.status_code == 404
    assert rt.get_call("ghost") is None


# -- call gate ---------------------------------------------------------------

def test_unknown_call_404_and_no_session_created():
    c, rt = _client()  # lab hatch on via conftest: no tokens needed
    resp = c.post(_url(call_id="ghost"), content=_fragment([(1, 0, 160)]))
    assert resp.status_code == 404
    assert rt.get_call("ghost") is None


def test_missing_call_id_400():
    c, _ = _client()
    resp = c.post("/v1/ingest/amazon-connect", content=_fragment([(1, 0, 160)]))
    assert resp.status_code == 400


def test_empty_body_400():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    assert c.post(_url(), content=b"").status_code == 400


# -- happy path ----------------------------------------------------------------

def test_full_fragment_flow_into_session():
    c, rt = _client()
    cid = c.post("/v1/calls", json={"call_id": CALL_ID}).json()["call_id"]
    # Two contiguous 20 ms fragments: the pipeline's 25 ms analysis
    # window needs more than one before it emits a frame.
    first = c.post(
        _url(), content=_fragment([(1, 0, 160), (2, 0, 80)])
    )
    assert first.status_code == 200
    assert first.json()["frames"] == 2
    assert first.json()["outbound_frames"] == 1
    assert first.json()["call_id"] == cid
    second = c.post(_url(), content=_fragment([(1, 20, 160), (2, 20, 80)]))
    assert second.status_code == 200
    assert second.json()["frames"] == 2
    sess = rt.get_call(cid)
    assert sess is not None
    assert sess.frames > 0
    c.delete(f"/v1/calls/{cid}")


def test_malformed_fragment_400_and_session_survives():
    c, rt = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    resp = c.post(_url(), content=b"this is not an MKV fragment")
    assert resp.status_code == 400
    assert rt.get_call(CALL_ID) is not None
    c.delete(f"/v1/calls/{CALL_ID}")


def test_contact_id_mismatch_400():
    c, _ = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    resp = c.post(
        _url(), content=_fragment([(1, 0, 160)], contact_id="someone-else")
    )
    assert resp.status_code == 400
    assert "ContactId" in resp.json()["detail"]
    c.delete(f"/v1/calls/{CALL_ID}")


def test_close_call_drops_fragment_state():
    c, rt = _client()
    c.post("/v1/calls", json={"call_id": CALL_ID})
    assert c.post(_url(), content=_fragment([(1, 0, 160)])).status_code == 200
    c.delete(f"/v1/calls/{CALL_ID}")
    # The call is gone, so a fragment for it is a 404, not a stale
    # stream resuming: no session is created by the POST.
    assert c.post(_url(), content=_fragment([(1, 20, 160)])).status_code == 404
    assert rt.get_call(CALL_ID) is None
    # Reopening the same call starts a fresh stream: the identical
    # first fragment is accepted again instead of tripping the
    # overlap check on the dropped stream's clock.
    c.post("/v1/calls", json={"call_id": CALL_ID})
    assert c.post(_url(), content=_fragment([(1, 0, 160)])).status_code == 200
    c.delete(f"/v1/calls/{CALL_ID}")


def test_capabilities_lists_amazon_connect_ingest():
    c, _ = _client()
    body = c.get("/v1/capabilities").json()
    assert "POST /v1/ingest/amazon-connect" in body["endpoints"]
