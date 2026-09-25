"""Shared pytest fixtures for the shieldcall-core test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _lab_unauthenticated(monkeypatch):
    """Existing HTTP tests exercise the lab API without bearer tokens.

    The sidecar is fail-closed by default (see
    shieldcall.serve.http_app._require_sidecar_auth_configured); this fixture
    opts the suite into the explicit local-dev escape hatch so those tests
    keep testing detector behavior rather than auth plumbing. Auth itself is
    covered by tests/test_sidecar_auth.py, which overrides this env per case.
    """
    monkeypatch.setenv("SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKEN", raising=False)
    monkeypatch.delenv("SHIELDCALL_SIDECAR_TOKENS", raising=False)
    monkeypatch.delenv("SHIELDCALL_HOSTED_ENDPOINT", raising=False)
