"""
ASR provider selection for the sidecar telephone path.

``make_asr_from_env`` is what the serving sidecar (``shieldcall.serve``)
calls at startup. Selection is explicit and fail-safe:

- ``SHIELDCALL_ASR_PROVIDER=deepgram`` + ``DEEPGRAM_API_KEY`` set
  -> live Deepgram streaming provider.
- ``SHIELDCALL_ASR_PROVIDER=deepgram`` without a key
  -> warn, fall back to ``PassthroughASR`` (never crash the sidecar).
- unset / anything else -> ``PassthroughASR`` (previous behavior).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .asr_bridge import ASRBridge, PassthroughASR
from .deepgram_stream import DeepgramStreamingASR

log = logging.getLogger(__name__)


def make_asr_from_env(env: Optional[dict] = None) -> ASRBridge:
    src = env if env is not None else os.environ
    provider = (src.get("SHIELDCALL_ASR_PROVIDER") or "").strip().lower()
    if provider == "deepgram":
        key = (src.get("DEEPGRAM_API_KEY") or "").strip()
        if not key:
            log.warning(
                "SHIELDCALL_ASR_PROVIDER=deepgram but DEEPGRAM_API_KEY is not set; "
                "using PassthroughASR (no live transcription)"
            )
            return PassthroughASR()
        return DeepgramStreamingASR(api_key=key)
    if provider:
        log.warning("unknown SHIELDCALL_ASR_PROVIDER=%r; using PassthroughASR", provider)
    return PassthroughASR()
