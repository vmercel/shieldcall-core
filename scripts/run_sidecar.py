#!/usr/bin/env python3
"""Lab/CPaaS sidecar HTTP+WS on 0.0.0.0:8765. Channel twin off. Not a hairpin."""

from __future__ import annotations

import os

import _repo  # noqa: F401

import uvicorn


def main() -> None:
    host = os.environ.get("SHIELDCALL_SIDECAR_HOST", "0.0.0.0")
    port = int(os.environ.get("SHIELDCALL_SIDECAR_PORT", "8765"))
    print(f"ShieldCall sidecar  http://{host}:{port}/health")
    print("Channel twin: OFF. Actuation: recommend-only. Fail-open if this process dies.")
    uvicorn.run("shieldcall.serve.http_app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
