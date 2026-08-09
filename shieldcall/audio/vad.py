"""
Streaming multi-feature voice activity detection for telephony audio.

Combines energy, zero-crossing rate, and spectral flatness with hangover
logic so short unvoiced consonants are not chopped — critical for both
acoustic residual analysis and transcript alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np


@dataclass
class VADDecision:
    is_speech: bool
    energy: float
    zcr: float
    flatness: float
    confidence: float


class StreamingVAD:
    """Adaptive-threshold VAD with hangover for telephone speech."""

    def __init__(
        self,
        energy_percentile: float = 30.0,
        hangover_frames: int = 8,
        history: int = 80,
    ):
        self.energy_percentile = energy_percentile
        self.hangover_frames = hangover_frames
        self._energy_hist: deque[float] = deque(maxlen=history)
        self._hangover = 0

    def reset(self) -> None:
        self._energy_hist.clear()
        self._hangover = 0

    def decide(self, frame: np.ndarray) -> VADDecision:
        x = frame.astype(np.float32)
        energy = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        zcr = float(np.mean(np.abs(np.diff(np.sign(x + 1e-12)))))

        # Spectral flatness (noise ~1, tone/speech lower)
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) + 1e-8
        log_mean = float(np.mean(np.log(spec)))
        mean = float(np.mean(spec))
        flatness = float(np.exp(log_mean) / (mean + 1e-8))

        self._energy_hist.append(energy)
        if len(self._energy_hist) >= 10:
            thr = float(np.percentile(list(self._energy_hist), self.energy_percentile))
            thr = max(thr * 1.5, 0.005)
        else:
            thr = 0.01

        raw_speech = energy > thr and flatness < 0.85
        if raw_speech:
            self._hangover = self.hangover_frames
            is_speech = True
        elif self._hangover > 0:
            self._hangover -= 1
            is_speech = True
        else:
            is_speech = False

        # Confidence: distance from threshold + structure
        conf = float(np.clip((energy / (thr + 1e-8) - 0.5) * 0.4 + (1.0 - flatness) * 0.4, 0, 1))
        if not is_speech:
            conf = 1.0 - conf

        return VADDecision(
            is_speech=is_speech,
            energy=energy,
            zcr=zcr,
            flatness=flatness,
            confidence=conf,
        )
