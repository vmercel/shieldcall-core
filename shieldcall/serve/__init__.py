"""HTTP/WebSocket sidecar for lab and CPaaS-style forks. Not a media hairpin."""

from .http_app import app, create_app

__all__ = ["app", "create_app"]
