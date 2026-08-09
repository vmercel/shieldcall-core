"""
ASR bridge  -  pluggable streaming transcript interface.

ShieldCall never requires a specific ASR vendor. Any system that can
emit partial/final text fragments with timestamps can feed the
linguistic stream. Includes a null ASR and a mock ASR for demos/tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class TranscriptFragment:
    text: str
    timestamp_sec: float
    is_final: bool = True
    confidence: float = 1.0
    speaker: str = "remote"  # remote | local | unknown


class ASRBridge(ABC):
    """Abstract streaming ASR adapter."""

    @abstractmethod
    def push_audio(self, samples: np.ndarray, sample_rate: int, timestamp_sec: float) -> List[TranscriptFragment]:
        """Push audio; return zero or more new transcript fragments."""
        ...

    @abstractmethod
    def push_text(self, text: str, timestamp_sec: float, is_final: bool = True) -> TranscriptFragment:
        """Direct text injection (when ASR runs out-of-process)."""
        ...

    def reset(self) -> None:
        pass


class PassthroughASR(ASRBridge):
    """Text-only bridge: ignores audio, accepts external transcripts."""

    def push_audio(self, samples: np.ndarray, sample_rate: int, timestamp_sec: float) -> List[TranscriptFragment]:
        return []

    def push_text(self, text: str, timestamp_sec: float, is_final: bool = True) -> TranscriptFragment:
        return TranscriptFragment(text=text, timestamp_sec=timestamp_sec, is_final=is_final)


class ScheduledTranscriptASR(ASRBridge):
    """
    Demo/eval ASR that emits scheduled (time, text) pairs when stream
    time reaches each cue  -  no real speech recognition required.
    """

    def __init__(self, schedule: Optional[List[tuple[float, str]]] = None):
        self.schedule = list(schedule or [])
        self._idx = 0
        self._clock = 0.0

    def reset(self) -> None:
        self._idx = 0
        self._clock = 0.0

    def push_audio(self, samples: np.ndarray, sample_rate: int, timestamp_sec: float) -> List[TranscriptFragment]:
        self._clock = timestamp_sec
        out: List[TranscriptFragment] = []
        while self._idx < len(self.schedule) and self.schedule[self._idx][0] <= self._clock + 1e-6:
            t, text = self.schedule[self._idx]
            out.append(TranscriptFragment(text=text, timestamp_sec=t, is_final=True))
            self._idx += 1
        return out

    def push_text(self, text: str, timestamp_sec: float, is_final: bool = True) -> TranscriptFragment:
        return TranscriptFragment(text=text, timestamp_sec=timestamp_sec, is_final=is_final)
