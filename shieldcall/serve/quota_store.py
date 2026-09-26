"""Persistent per-client quota store (P2-6 Phase 1, 2026-09-26).

SQLite-backed fixed-window quota accounting shared by the sidecar's
ingest routes (main ``POST /v1/calls`` and hosted
``POST /hosted/v1/analyze``). This replaces the P2-5 in-process dict,
which reset on every worker restart and could not be shared across
workers, so a restarted or multi-worker deployment silently granted
extra budget.

Fail behavior (production contract, docs/HOSTED_INFERENCE_DESIGN.md
section 2 item 3): the store never fails open. A ``consume`` that cannot
reach the store raises :class:`QuotaStoreError`; callers translate that
into a 429 on NEW-CALL ingest routes ("quota store unavailable",
Retry-After 60) while the audio/score paths keep serving existing
calls. In other words: a dead quota store freezes new-call opens, it
never disables throttling.

Design notes:
- One row per (scope, token_id, window_start); the upsert and the
  count read run inside a single ``BEGIN IMMEDIATE`` transaction, so
  concurrent workers cannot lose increments.
- Scopes keep budgets independent per route family: the main API uses
  scope ``"sidecar"``, the hosted prototype uses ``"hosted"``. A call
  to one route never eats the other's budget.
- WAL journal mode + busy_timeout so two worker processes sharing one
  database file do not wedge each other.
- Stdlib only (sqlite3): no new dependencies for the sidecar.

Env:
  SHIELDCALL_QUOTA_DB_PATH   Path to the SQLite file. Default
                             ``~/.shieldcall/quota.db``. The value
                             ``:memory:`` is honored (used by the test
                             suite). The parent directory is created when
                             missing; failure to open the database raises
                             QuotaStoreError at startup so a misconfigured
                             deployment fails closed instead of serving
                             with a dead quota store.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 60

_UPSERT = """
INSERT INTO quota_windows (scope, token_id, window_start, count)
VALUES (?, ?, ?, 1)
ON CONFLICT (scope, token_id, window_start)
DO UPDATE SET count = quota_windows.count + 1
"""

_SELECT_COUNT = """
SELECT count FROM quota_windows
WHERE scope = ? AND token_id = ? AND window_start = ?
"""

_PRUNE = "DELETE FROM quota_windows WHERE window_start < ?"


class QuotaStoreError(Exception):
    """The quota store could not serve a request.

    Callers must treat this as "throttling state unknown": freeze
    new-call ingest with a 429, never serve unthrottled.
    """


class QuotaDecision(NamedTuple):
    allowed: bool
    retry_after: int  # seconds to wait when not allowed, else 0
    remaining: int  # budget left in this window (0 when not allowed)
    count: int  # requests seen in this window including this one


def default_quota_db_path() -> str:
    raw = os.environ.get("SHIELDCALL_QUOTA_DB_PATH", "").strip()
    if raw:
        return raw
    return str(Path.home() / ".shieldcall" / "quota.db")


class QuotaStore:
    """Fixed-window per-(scope, token) quota accounting on SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        # Process-local counters for /health export. They are
        # deliberately not persisted: they describe this worker only.
        self.consumed = 0
        self.rejected = 0
        self.store_errors = 0
        try:
            if db_path == ":memory:":
                conn = sqlite3.connect(":memory:", check_same_thread=False)
            else:
                path = Path(db_path).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS quota_windows (
                    scope TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    window_start INTEGER NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (scope, token_id, window_start)
                )"""
            )
            conn.commit()
        except (sqlite3.Error, OSError) as exc:
            raise QuotaStoreError(
                f"cannot open quota database at {db_path!r}: {exc}"
            ) from exc
        self._conn = conn

    def consume(
        self,
        scope: str,
        token_id: str,
        limit: int,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        now: float | None = None,
    ) -> QuotaDecision:
        """Consume one budget unit. ``limit`` must be positive; callers
        disable throttling by not calling consume (limit 0), not by
        passing 0 here."""
        if limit <= 0:
            raise ValueError("limit must be positive; skip consume() to disable throttling")
        window = max(1, int(window_seconds))
        t = time.time() if now is None else float(now)
        window_start = (int(t) // window) * window
        key = (scope, token_id, window_start)
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute("BEGIN IMMEDIATE")
                cur.execute(_UPSERT, key)
                row = cur.execute(_SELECT_COUNT, key).fetchone()
                count = int(row[0]) if row else 0
                # Keep at most ~2 windows per key so the table stays tiny.
                cur.execute(_PRUNE, (window_start - window,))
                self._conn.commit()
            except Exception as exc:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                self.store_errors += 1
                raise QuotaStoreError(f"quota store consume failed: {exc}") from exc
        allowed = count <= limit
        if allowed:
            self.consumed += 1
            return QuotaDecision(True, 0, limit - count, count)
        self.rejected += 1
        retry_after = int(window_start + window - t) + 1
        return QuotaDecision(False, max(1, retry_after), 0, count)

    def reset(self) -> None:
        """Clear all accounting rows (tests, admin). Counters are kept."""
        with self._lock:
            try:
                self._conn.execute("DELETE FROM quota_windows")
                self._conn.commit()
            except Exception as exc:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                self.store_errors += 1
                raise QuotaStoreError(f"quota store reset failed: {exc}") from exc

    def stats(self) -> dict:
        """Process-local counters for /health export."""
        return {
            "backend": "sqlite",
            "path": self.db_path,
            "consumed": self.consumed,
            "rejected": self.rejected,
            "store_errors": self.store_errors,
        }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
