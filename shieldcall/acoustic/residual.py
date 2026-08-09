"""
Spectral-Temporal Residual Fingerprinting (STRF) — residual core
================================================================

Neural vocoders (HiFi-GAN, WaveNet-family, diffusion vocoders) leave
characteristic structure in the *residual after a classical harmonic-plus-
noise (HPN) decomposition*:

  - Overly smooth residual envelopes (too little micro-variation)
  - Periodic grid artifacts from upsampling kernels
  - Phase-derivative irregularities that survive bandlimiting differently
    than natural glottal noise

Under telephone channels these cues weaken but do not vanish; STRF
extracts residual statistics that remain informative after G.711 /
narrowband distortion — the regime commercial liveness systems treat
as out-of-scope lab noise.

This module performs a lightweight HPN-style residual extraction without
heavy pretrained models so the core stays deployable on-device / edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class ResidualFingerprint:
    residual_energy_ratio: float
    residual_flatness: float
    residual_kurtosis: float
    residual_modulation: float
    harmonic_to_noise: float
    phase_irregularity: float
    grid_artifact_score: float
    vector: np.ndarray  # compact embedding used by the scorer


def _estimate_f0(x: np.ndarray, sr: int) -> float:
    """Autocorrelation F0 estimate in Hz; 0 if unvoiced / unclear."""
    n = len(x)
    if n < 64:
        return 0.0
    # Center, remove DC, window
    x = x - np.mean(x)
    w = x * np.hanning(n)
    energy = float(np.mean(w ** 2))
    if energy < 1e-8:
        return 0.0
    acf = np.correlate(w, w, mode="full")[n - 1 :]
    acf = acf / (acf[0] + 1e-8)
    min_lag = max(2, int(sr / 400))
    max_lag = min(len(acf) - 1, int(sr / 60))
    if max_lag <= min_lag:
        return 0.0
    seg = acf[min_lag:max_lag]
    peak_i = int(np.argmax(seg))
    # Slightly lower threshold so breathy voiced speech still locks
    if seg[peak_i] < 0.18:
        return 0.0
    lag = peak_i + min_lag
    return float(sr / lag)


def _harmonic_model(x: np.ndarray, sr: int, f0: float, n_harm: int = 8) -> np.ndarray:
    """Least-squares sum of cos/sin harmonics at f0..n*f0."""
    n = len(x)
    t = np.arange(n) / sr
    if f0 < 60 or f0 > 400:
        return np.zeros(n, dtype=np.float64)
    cols = []
    for k in range(1, n_harm + 1):
        freq = k * f0
        if freq >= sr / 2:
            break
        cols.append(np.cos(2 * np.pi * freq * t))
        cols.append(np.sin(2 * np.pi * freq * t))
    if not cols:
        return np.zeros(n, dtype=np.float64)
    A = np.column_stack(cols)
    try:
        coef, *_ = np.linalg.lstsq(A, x.astype(np.float64), rcond=None)
        return (A @ coef).astype(np.float64)
    except np.linalg.LinAlgError:
        return np.zeros(n, dtype=np.float64)


def _phase_irregularity(x: np.ndarray) -> float:
    """Variance of instantaneous phase derivative (Hilbert)."""
    if len(x) < 16:
        return 0.0
    analytic = signal.hilbert(x)
    phase = np.unwrap(np.angle(analytic))
    dphase = np.diff(phase)
    return float(np.std(dphase))


def _grid_artifact_score(residual: np.ndarray, sr: int) -> float:
    """
    Detect weak periodic energy at common neural-upsampling rates
    (e.g. related to 256x / 128x hop patterns folded into band).
    Score is relative spectral peakiness in residual.
    """
    if len(residual) < 32:
        return 0.0
    mag = np.abs(np.fft.rfft(residual * np.hanning(len(residual)))) + 1e-8
    # Peak-to-median ratio in upper half of spectrum
    mid = len(mag) // 2
    upper = mag[mid:]
    if len(upper) < 4:
        return 0.0
    return float(np.max(upper) / (np.median(upper) + 1e-8) - 1.0)


def extract_residual_fingerprint(frame: np.ndarray, sr: int = 8000) -> ResidualFingerprint:
    x = frame.astype(np.float64)
    n = len(x)
    if n < 16:
        z = np.zeros(16, dtype=np.float32)
        return ResidualFingerprint(0, 0, 0, 0, 0, 0, 0, z)

    # Pre-emphasis
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])

    f0 = _estimate_f0(x, sr)
    harmonic = _harmonic_model(x, sr, f0)
    residual = x - harmonic

    e_x = float(np.mean(x ** 2) + 1e-12)
    e_r = float(np.mean(residual ** 2) + 1e-12)
    e_h = float(np.mean(harmonic ** 2) + 1e-12)

    residual_energy_ratio = e_r / e_x
    harmonic_to_noise = e_h / e_r

    mag_r = np.abs(np.fft.rfft(residual * np.hanning(n))) + 1e-8
    log_mag = np.log(mag_r)
    residual_flatness = float(np.exp(np.mean(log_mag)) / (np.mean(mag_r) + 1e-8))

    # Excess kurtosis of residual (natural glottal noise is heavier-tailed)
    m = residual - np.mean(residual)
    m2 = float(np.mean(m ** 2) + 1e-12)
    m4 = float(np.mean(m ** 4))
    residual_kurtosis = m4 / (m2 ** 2) - 3.0

    env = np.abs(signal.hilbert(residual))
    residual_modulation = float(np.var(env) / (np.mean(env ** 2) + 1e-12))

    phase_irr = _phase_irregularity(x)
    grid = _grid_artifact_score(residual, sr)

    vector = np.array(
        [
            residual_energy_ratio,
            residual_flatness,
            residual_kurtosis,
            residual_modulation,
            np.log1p(harmonic_to_noise),
            phase_irr,
            grid,
            f0 / 400.0,
            e_r,
            e_h,
            np.mean(log_mag),
            np.std(log_mag),
            np.percentile(mag_r, 90) / (np.percentile(mag_r, 10) + 1e-8),
            float(np.max(np.abs(residual))),
            float(np.mean(np.abs(np.diff(residual)))),
            float(np.std(residual)),
        ],
        dtype=np.float32,
    )

    return ResidualFingerprint(
        residual_energy_ratio=residual_energy_ratio,
        residual_flatness=residual_flatness,
        residual_kurtosis=float(residual_kurtosis),
        residual_modulation=residual_modulation,
        harmonic_to_noise=float(harmonic_to_noise),
        phase_irregularity=phase_irr,
        grid_artifact_score=float(np.clip(grid, 0, 20)),
        vector=vector,
    )
