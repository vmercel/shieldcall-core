"""
Telephony Channel Twin (TCT)
============================

A generative model of the telephone path that real calls traverse before
reaching any detector. Unlike clean-lab evaluation, TCT synthesizes the
compound distortions that erase or invent deepfake cues:

  - G.711 mu-law quantization (8-bit, companded)
  - Telephone bandlimiting (300-3400 Hz)
  - Packet loss with waveform-domain concealment artifacts
  - Additive line noise and SNR control
  - Optional jitter / frame drop clustering

Novelty: most deepfake systems train and evaluate on clean or lightly
noisy audio. ShieldCall treats the *channel itself* as a first-class
stochastic process and forces every acoustic decision through TCT so
scores generalize under real PSTN / VoIP conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from scipy import signal


class CodecProfile(str, Enum):
    CLEAN = "clean"
    NARROWBAND = "narrowband"
    G711_ULAW = "g711_ulaw"
    HARSH_VOIP = "harsh_voip"
    DEGRADED_PSTN = "degraded_pstn"


@dataclass
class ChannelConfig:
    profile: CodecProfile = CodecProfile.NARROWBAND
    snr_db: Optional[float] = None          # None = no additive noise
    packet_loss_rate: float = 0.0           # 0..1
    packet_ms: float = 20.0
    seed: Optional[int] = None


def _ulaw_encode_decode(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """mu-law compress  ->  8-bit quantize  ->  expand (G.711 approximation)."""
    x = np.clip(x.astype(np.float64), -1.0, 1.0)
    sign = np.sign(x)
    abs_x = np.abs(x)
    compressed = sign * np.log1p(mu * abs_x) / np.log1p(mu)
    # 8-bit mid-riser quantization
    levels = 256
    q = np.round((compressed + 1.0) * 0.5 * (levels - 1))
    q = np.clip(q, 0, levels - 1)
    dequant = (q / (levels - 1)) * 2.0 - 1.0
    expanded = sign * (1.0 / mu) * (np.expm1(np.abs(dequant) * np.log1p(mu)))
    return expanded.astype(np.float32)


def _bandlimit(x: np.ndarray, sr: int, low: float = 300.0, high: float = 3400.0) -> np.ndarray:
    if sr < 8000:
        return x.astype(np.float32)
    nyq = sr / 2.0
    lo = max(low / nyq, 1e-4)
    hi = min(high / nyq, 0.99)
    if lo >= hi:
        return x.astype(np.float32)
    b, a = signal.butter(4, [lo, hi], btype="band")
    return signal.lfilter(b, a, x).astype(np.float32)


def _add_noise(x: np.ndarray, snr_db: float, rng: np.random.RandomState) -> np.ndarray:
    power = float(np.mean(x ** 2) + 1e-12)
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = rng.randn(len(x)).astype(np.float32) * np.sqrt(noise_power)
    return (x + noise).astype(np.float32)


def _packet_loss(
    x: np.ndarray,
    sr: int,
    loss_rate: float,
    packet_ms: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Drop packets and apply simple PLC (packet loss concealment):
    repeat last good packet with mild attenuation  -  a common source of
    artificial periodicity that naive deepfake detectors misread.
    """
    if loss_rate <= 0.0:
        return x
    pkt = max(1, int(sr * packet_ms / 1000.0))
    out = x.copy()
    n_packets = int(np.ceil(len(x) / pkt))
    last_good = np.zeros(pkt, dtype=np.float32)
    for i in range(n_packets):
        start = i * pkt
        end = min(len(x), start + pkt)
        if rng.rand() < loss_rate:
            # Conceal with attenuated last good packet (PLC artifact)
            fill = last_good[: end - start] * 0.7
            out[start:end] = fill
        else:
            last_good = np.zeros(pkt, dtype=np.float32)
            last_good[: end - start] = out[start:end]
    return out


class TelephonyChannelTwin:
    """
    Stochastic telephone channel. Apply to any waveform to stress-test
    acoustic scorers under the conditions that matter for real calls.
    """

    PROFILE_DEFAULTS = {
        CodecProfile.CLEAN: dict(bandlimit=False, ulaw=False, snr_db=None, packet_loss_rate=0.0),
        CodecProfile.NARROWBAND: dict(bandlimit=True, ulaw=False, snr_db=None, packet_loss_rate=0.0),
        CodecProfile.G711_ULAW: dict(bandlimit=True, ulaw=True, snr_db=35.0, packet_loss_rate=0.0),
        CodecProfile.HARSH_VOIP: dict(bandlimit=True, ulaw=True, snr_db=20.0, packet_loss_rate=0.08),
        CodecProfile.DEGRADED_PSTN: dict(bandlimit=True, ulaw=True, snr_db=12.0, packet_loss_rate=0.15),
    }

    def __init__(self, config: Optional[ChannelConfig] = None):
        self.config = config or ChannelConfig()
        self._rng = np.random.RandomState(self.config.seed)

    def apply(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        x = samples.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=-1).astype(np.float32)

        # Peak normalize lightly before channel so quantizer sees full range
        peak = float(np.max(np.abs(x)) + 1e-8)
        if peak > 1.0:
            x = x / peak

        defaults = self.PROFILE_DEFAULTS[self.config.profile]
        if defaults["bandlimit"]:
            x = _bandlimit(x, sample_rate)
        if defaults["ulaw"]:
            x = _ulaw_encode_decode(x)

        snr = self.config.snr_db if self.config.snr_db is not None else defaults["snr_db"]
        if snr is not None:
            x = _add_noise(x, snr, self._rng)

        plr = (
            self.config.packet_loss_rate
            if self.config.packet_loss_rate > 0
            else defaults["packet_loss_rate"]
        )
        if plr > 0:
            x = _packet_loss(x, sample_rate, plr, self.config.packet_ms, self._rng)

        # Final soft clip
        return np.clip(x, -1.0, 1.0).astype(np.float32)

    def apply_batch(
        self, samples: np.ndarray, sample_rate: int, n_variants: int = 1
    ) -> list[np.ndarray]:
        """Generate multiple stochastic channel realizations (for robust eval)."""
        out = []
        for i in range(n_variants):
            # reseed per variant for diversity while remaining reproducible from base seed
            if self.config.seed is not None:
                self._rng = np.random.RandomState(self.config.seed + i * 9973)
            out.append(self.apply(samples, sample_rate))
        return out
