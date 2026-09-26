"""P2-6a: persistent quota store (SQLite fixed-window accounting).

Covers the store unit behavior (limits, windows, isolation, persistence,
atomicity, pruning, open failures) and the route-level contract:
per-token 429 with Retry-After on ingest routes, and a dead quota store
freezing new-call opens (429 "quota store unavailable") while the
audio/score paths keep serving existing calls.
"""

from __future__ import annotations

import base64
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve import hosted
from shieldcall.serve.http_app import create_app
from shieldcall.serve.quota_store import (
    QuotaStore,
    QuotaStoreError,
    default_quota_db_path,
)


def _rt():
    cfg = PipelineConfig(channel=None)
    return SidecarRuntime(max_calls=8, pipeline_config=cfg)


def _authed_client(monkeypatch, **env):
    """App with real bearer tokens (no escape hatch), :memory: quota store."""
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKENS", "k1:tokA,k2:tokB")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return TestClient(create_app(_rt()))


def _h():
    return {"Authorization": "Bearer tokA"}


# ---------------------------------------------------------------- store unit


def test_consume_within_limit_counts_down(tmp_path):
    s = QuotaStore(str(tmp_path / "q.db"))
    try:
        d1 = s.consume("sidecar", "k1", 3, now=1000.0)
        assert d1.allowed and d1.remaining == 2 and d1.retry_after == 0
        d2 = s.consume("sidecar", "k1", 3, now=1000.0)
        assert d2.allowed and d2.remaining == 1
        d3 = s.consume("sidecar", "k1", 3, now=1000.0)
        assert d3.allowed and d3.remaining == 0 and d3.count == 3
        d4 = s.consume("sidecar", "k1", 3, now=1000.0)
        assert not d4.allowed
        assert d4.retry_after > 0
        assert d4.remaining == 0 and d4.count == 4
    finally:
        s.close()


def test_scopes_and_tokens_isolated(tmp_path):
    s = QuotaStore(str(tmp_path / "q.db"))
    try:
        assert s.consume("sidecar", "k1", 1, now=1000.0).allowed
        assert not s.consume("sidecar", "k1", 1, now=1000.0).allowed
        # Other token unaffected.
        assert s.consume("sidecar", "k2", 1, now=1000.0).allowed
        # Other scope unaffected: hosted budget is independent.
        assert s.consume("hosted", "k1", 1, now=1000.0).allowed
    finally:
        s.close()


def test_window_rollover(tmp_path):
    s = QuotaStore(str(tmp_path / "q.db"))
    try:
        # Windows are epoch-aligned: now=1000 falls in window [960, 1020).
        assert s.consume("sidecar", "k1", 1, now=1000.0).allowed
        assert not s.consume("sidecar", "k1", 1, now=1019.0).allowed
        d = s.consume("sidecar", "k1", 1, now=1020.0)
        assert d.allowed and d.remaining == 0
    finally:
        s.close()


def test_persists_across_reopen(tmp_path):
    path = str(tmp_path / "q.db")
    s = QuotaStore(path)
    s.consume("sidecar", "k1", 3, now=1000.0)
    s.consume("sidecar", "k1", 3, now=1000.0)
    s.close()
    s2 = QuotaStore(path)
    try:
        d = s2.consume("sidecar", "k1", 3, now=1000.0)
        # Budget continued from the previous process lifetime, not reset.
        assert d.allowed and d.count == 3
        assert not s2.consume("sidecar", "k1", 3, now=1000.0).allowed
    finally:
        s2.close()


def test_prunes_old_windows(tmp_path):
    path = str(tmp_path / "q.db")
    s = QuotaStore(path)
    try:
        s.consume("sidecar", "k1", 100, now=1000.0)
        s.consume("sidecar", "k1", 100, now=2000.0)
        conn = sqlite3.connect(path)
        rows = conn.execute("SELECT window_start FROM quota_windows").fetchall()
        conn.close()
        # Only the current window row survives.
        assert rows == [(1980,)]
    finally:
        s.close()


def test_concurrent_consume_is_atomic(tmp_path):
    s = QuotaStore(str(tmp_path / "q.db"))
    allowed = []
    lock = threading.Lock()
    try:
        def worker():
            for _ in range(10):
                d = s.consume("sidecar", "k1", 50, now=1000.0)
                with lock:
                    allowed.append(d.allowed)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 80 consumes against a limit of 50: exactly 50 allowed, no lost
        # increments.
        assert sum(allowed) == 50
        assert len(allowed) == 80
    finally:
        s.close()


def test_unwritable_path_raises_at_open(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    with pytest.raises(QuotaStoreError, match="cannot open quota database"):
        QuotaStore(str(blocker / "q.db"))


def test_consume_rejects_nonpositive_limit(tmp_path):
    s = QuotaStore(str(tmp_path / "q.db"))
    try:
        with pytest.raises(ValueError):
            s.consume("sidecar", "k1", 0, now=1000.0)
    finally:
        s.close()


def test_stats_and_reset(tmp_path):
    s = QuotaStore(str(tmp_path / "q.db"))
    try:
        s.consume("sidecar", "k1", 1, now=1000.0)
        s.consume("sidecar", "k1", 1, now=1000.0)
        stats = s.stats()
        assert stats["backend"] == "sqlite"
        assert stats["consumed"] == 1 and stats["rejected"] == 1
        assert stats["store_errors"] == 0
        s.reset()
        assert s.consume("sidecar", "k1", 1, now=1000.0).allowed
    finally:
        s.close()


def test_default_path_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SHIELDCALL_QUOTA_DB_PATH", str(tmp_path / "custom.db"))
    assert default_quota_db_path() == str(tmp_path / "custom.db")
    monkeypatch.delenv("SHIELDCALL_QUOTA_DB_PATH")
    assert default_quota_db_path().endswith(str(Path(".shieldcall") / "quota.db"))


# ------------------------------------------------------- route-level contract


def test_create_app_fails_closed_on_bad_store(monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKENS", "k1:tokA")
    monkeypatch.setenv("SHIELDCALL_QUOTA_DB_PATH", str(blocker / "q.db"))
    with pytest.raises(RuntimeError, match="Cannot open sidecar quota store"):
        create_app(_rt())


def test_v1_calls_quota_429_with_retry_after(monkeypatch):
    c = _authed_client(monkeypatch, SHIELDCALL_SIDECAR_QUOTA_PER_MIN="1")
    first = c.post("/v1/calls", json={}, headers=_h())
    assert first.status_code == 200
    call_id = first.json()["call_id"]
    second = c.post("/v1/calls", json={}, headers=_h())
    assert second.status_code == 429
    assert second.json()["detail"] == "sidecar quota exceeded"
    assert "Retry-After" in second.headers
    assert int(second.headers["Retry-After"]) > 0
    # Quota is per token id: the other token still has budget.
    other = c.post(
        "/v1/calls", json={}, headers={"Authorization": "Bearer tokB"}
    )
    assert other.status_code == 200
    # The first call is unaffected by the quota freeze.
    pcm = base64.b64encode(np.zeros(160, dtype=np.int16).tobytes()).decode("ascii")
    audio = c.post(
        f"/v1/calls/{call_id}/audio",
        json={"sr": 8000, "pcm_s16le_b64": pcm},
        headers=_h(),
    )
    assert audio.status_code == 200


def test_store_error_freezes_new_calls_but_audio_serves(monkeypatch):
    c = _authed_client(monkeypatch)
    first = c.post("/v1/calls", json={}, headers=_h())
    assert first.status_code == 200
    call_id = first.json()["call_id"]

    class BrokenStore:
        def consume(self, *args, **kwargs):
            raise QuotaStoreError("disk gone")

    monkeypatch.setattr(c.app.state, "quota_store", BrokenStore())
    frozen = c.post("/v1/calls", json={}, headers=_h())
    assert frozen.status_code == 429
    assert frozen.json()["detail"] == "quota store unavailable"
    assert frozen.headers["Retry-After"] == "60"
    # Existing call's audio path keeps serving: never fail open, never
    # brick an in-progress call.
    pcm = base64.b64encode(np.zeros(160, dtype=np.int16).tobytes()).decode("ascii")
    audio = c.post(
        f"/v1/calls/{call_id}/audio",
        json={"sr": 8000, "pcm_s16le_b64": pcm},
        headers=_h(),
    )
    assert audio.status_code == 200


def test_quota_disabled_when_zero(monkeypatch):
    c = _authed_client(monkeypatch, SHIELDCALL_SIDECAR_QUOTA_PER_MIN="0")
    for _ in range(5):
        r = c.post("/v1/calls", json={}, headers=_h())
        assert r.status_code == 200


def test_quota_db_path_env_wiring(monkeypatch, tmp_path):
    db = tmp_path / "wired.db"
    c = _authed_client(monkeypatch, SHIELDCALL_QUOTA_DB_PATH=str(db))
    r = c.post("/v1/calls", json={}, headers=_h())
    assert r.status_code == 200
    assert db.exists()
    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT scope, token_id FROM quota_windows").fetchall()
    conn.close()
    assert ("sidecar", "k1") in rows


def test_health_exports_quota_stats(monkeypatch):
    c = _authed_client(monkeypatch)
    body = c.get("/health").json()
    quota = body["quota"]
    assert quota["backend"] == "sqlite"
    assert quota["path"] == ":memory:"
    assert quota["consumed"] >= 0


def test_hosted_status_exports_quota_stats(monkeypatch):
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setenv("SHIELDCALL_HOSTED_ENDPOINT", "1")
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKENS", "k1:tokA")
    c = TestClient(create_app(_rt()))
    r = c.get("/hosted/v1/status", headers=_h())
    assert r.status_code == 200
    quota = r.json()["quota"]
    assert quota["backend"] == "sqlite"
    assert quota["path"] == ":memory:"


def test_hosted_quota_still_429s(monkeypatch):
    monkeypatch.delenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setenv("SHIELDCALL_HOSTED_ENDPOINT", "1")
    monkeypatch.setenv("SHIELDCALL_SIDECAR_TOKENS", "k1:tokA,k2:tokB")
    monkeypatch.setenv("SHIELDCALL_HOSTED_QUOTA_PER_MIN", "1")
    c = TestClient(create_app(_rt()))
    assert c.post("/hosted/v1/analyze", json={}, headers=_h()).status_code == 200
    r = c.post("/hosted/v1/analyze", json={}, headers=_h())
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_hosted_auth_direct_construction_is_hermetic():
    # Direct construction (no from_env) must not touch the real DB path.
    auth = hosted.HostedAuth({"k1": "tokA"}, quota_per_min=2)
    assert auth.check_quota("k1") is None
    assert auth.check_quota("k1") is None
    assert auth.check_quota("k1") is not None
    auth.reset_usage()
    assert auth.check_quota("k1") is None
