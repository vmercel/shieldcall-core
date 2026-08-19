"""
Telephony-aware streaming preprocessor.

Integrates Channel Twin (optional online simulation), multi-feature VAD,
resampling, and framing so both acoustic and linguistic streams see a
consistent, realistic view of the call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
from scipy import signal

from .channel import ChannelConfig, TelephonyChannelTwin
from .vad import StreamingVAD, VADDecision


@dataclass
class Frame:
    """One analysis frame ready for dual-stream scoring."""

    samples: np.ndarray  # float32 mono
    sample_rate: int
    timestamp_sec: float
    is_speech: bool
    frame_index: int
    vad_confidence: float = 1.0
    energy: float = 0.0


class TelephonyPreprocessor:
    """
    Streaming preprocessor for the dual-stream engine.

    Design goals:
    - Operate on successive short frames (default 25 ms, 10 ms hop).
    - Support 8 kHz (narrowband telephony) and 16 kHz wideband.
    - Optional live Channel Twin injection for stress-testing.
    - Multi-feature VAD decisions both streams can trust.
    """

    def __init__(
        self,
        target_sr: int = 8000,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        energy_vad_threshold: float = 0.01,  # kept for API compat; VAD is adaptive
        apply_telephone_bandlimit: bool = True,
        channel_config: Optional[ChannelConfig] = None,
        use_advanced_vad: bool = True,
    ):
        self.target_sr = target_sr
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.energy_vad_threshold = energy_vad_threshold
        self.apply_telephone_bandlimit = apply_telephone_bandlimit
        self.use_advanced_vad = use_advanced_vad

        self.frame_len = int(target_sr * frame_ms / 1000.0)
        self.hop_len = int(target_sr * hop_ms / 1000.0)

        self._buffer = np.zeros(0, dtype=np.float32)
        self._frame_index = 0
        self._time_sec = 0.0
        self._vad = StreamingVAD()
        self._channel: Optional[TelephonyChannelTwin] = None
        if channel_config is not None:
            self._channel = TelephonyChannelTwin(channel_config)

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._frame_index = 0
        self._time_sec = 0.0
        self._vad.reset()

    def set_channel(self, config: Optional[ChannelConfig]) -> None:
        self._channel = TelephonyChannelTwin(config) if config else None

    def _bandlimit_telephone(self, x: np.ndarray, sr: int) -> np.ndarray:
        if sr < 8000:
            return x
        nyq = sr / 2.0
        low = 300.0 / nyq
        high = min(3400.0 / nyq, 0.99)
        b, a = signal.butter(4, [low, high], btype="band")
        return signal.lfilter(b, a, x).astype(np.float32)

    def _resample_if_needed(self, samples: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.target_sr:
            return samples.astype(np.float32)
        gcd = np.gcd(orig_sr, self.target_sr)
        up = self.target_sr // gcd
        down = orig_sr // gcd
        resampled = signal.resample_poly(samples, up, down)
        return resampled.astype(np.float32)

    def push(self, samples: np.ndarray, sample_rate: int) -> list[Frame]:
        if samples.ndim > 1:
            samples = samples.mean(axis=-1)

        samples = samples.astype(np.float32)

        # Optional channel twin *before* analysis resampling so codec
        # artifacts form at the source rate when possible
        if self._channel is not None:
            samples = self._channel.apply(samples, sample_rate)

        samples = self._resample_if_needed(samples, sample_rate)

        if self.apply_telephone_bandlimit and self._channel is None:
            samples = self._bandlimit_telephone(samples, self.target_sr)

        peak = np.max(np.abs(samples)) + 1e-8
        if peak > 1.0:
            samples = samples / peak

        self._buffer = np.concatenate([self._buffer, samples])

        frames: list[Frame] = []
        while len(self._buffer) >= self.frame_len:
            frame_samples = self._buffer[: self.frame_len].copy()
            self._buffer = self._buffer[self.hop_len :]

            if self.use_advanced_vad:
                vad: VADDecision = self._vad.decide(frame_samples)
                is_speech = vad.is_speech
                vad_conf = vad.confidence
                energy = vad.energy
            else:
                energy = float(np.sqrt(np.mean(frame_samples ** 2)))
                is_speech = energy > self.energy_vad_threshold
                vad_conf = 1.0 if is_speech else 0.0

            frames.append(
                Frame(
                    samples=frame_samples,
                    sample_rate=self.target_sr,
                    timestamp_sec=self._time_sec,
                    is_speech=is_speech,
                    frame_index=self._frame_index,
                    vad_confidence=vad_conf,
                    energy=energy,
                )
            )
            self._frame_index += 1
            self._time_sec += self.hop_ms / 1000.0

        return frames

    def stream_from_array(
        self, samples: np.ndarray, sample_rate: int, chunk_ms: float = 100.0
    ) -> Iterator[Frame]:
        self.reset()
        chunk_len = max(1, int(sample_rate * chunk_ms / 1000.0))
        for start in range(0, len(samples), chunk_len):
            chunk = samples[start : start + chunk_len]
            for frame in self.push(chunk, sample_rate):
                yield frame
