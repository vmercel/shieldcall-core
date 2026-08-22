"""ASR adapter that respects the circuit breaker."""

from __future__ import annotations

from typing import List

import numpy as np

from ..linguistic.asr_bridge import ASRBridge, TranscriptFragment
from .breaker import CircuitBreaker


class GatedASR(ASRBridge):
    def __init__(self, inner: ASRBridge, breaker: CircuitBreaker):
        self.inner = inner
        self.breaker = breaker

    def push_audio(
        self, samples: np.ndarray, sample_rate: int, timestamp_sec: float
    ) -> List[TranscriptFragment]:
        if not self.breaker.allow():
            return []
        try:
            out = self.inner.push_audio(samples, sample_rate, timestamp_sec)
            self.breaker.record_success()
            return out
        except Exception:
            self.breaker.record_failure()
            return []

    def push_text(self, text: str, timestamp_sec: float, is_final: bool = True) -> TranscriptFragment:
        if not self.breaker.allow():
            self.breaker.record_failure()
            return TranscriptFragment(text="", timestamp_sec=timestamp_sec, is_final=is_final, confidence=0.0)
        try:
            frag = self.inner.push_text(text, timestamp_sec, is_final=is_final)
            self.breaker.record_success()
            return frag
        except Exception:
            self.breaker.record_failure()
            raise

    def reset(self) -> None:
        self.inner.reset()
