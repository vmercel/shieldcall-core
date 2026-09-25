#!/usr/bin/env python3
"""Lab/CPaaS sidecar HTTP+WS on 0.0.0.0:8765. Channel twin off. Not a hairpin."""

from __future__ import annotations

import os

import _repo  # noqa: F401

import uvicorn


def main() -> None:
    host = os.environ.get("SHIELDCALL_SIDECAR_HOST", "0.0.0.0")
    port = int(os.environ.get("SHIELDCALL_SIDECAR_PORT", "8765"))
    from shieldcall.serve import http_app

    if http_app.app is None:
        raise SystemExit(
            "Refusing to serve the sidecar API without a bearer token. Set "
            "SHIELDCALL_SIDECAR_TOKENS (id:value pairs) or "
            "SHIELDCALL_SIDECAR_TOKEN, or for local lab use only set "
            "SHIELDCALL_SIDECAR_ALLOW_UNAUTHENTICATED=1."
        )
    print(f"ShieldCall sidecar  http://{host}:{port}/health")
    print("Channel twin: OFF. Actuation: recommend-only. Fail-open if this process dies.")
    uvicorn.run("shieldcall.serve.http_app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
