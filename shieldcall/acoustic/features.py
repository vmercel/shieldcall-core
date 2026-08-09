"""
STRF + classical spectral features for telephony deepfake detection.

Produces a fixed 64-D embedding per frame that combines:
  - classical spectral / prosodic cues robust under bandlimiting
  - Spectral-Temporal Residual Fingerprint (STRF) components
  - short-term dynamics placeholders filled by the scorer via history
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew

from .residual import extract_residual_fingerprint

FEATURE_DIM = 64


def extract_frame_features(frame: np.ndarray, sr: int = 8000) -> np.ndarray:
    """
    Extract a 64-D float32 feature vector from one short frame.

    Dimensions 0-15 : classical spectral / temporal
    Dimensions 16-31: STRF residual fingerprint (padded)
    Dimensions 32-47: cepstral / MFCC-lite under telephone band
    Dimensions 48-63: higher-order / modulation / reserved for aggregation
    """
    x = frame.astype(np.float32)
    n = len(x)
    if n < 16:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    # Pre-emphasis
    xp = np.append(x[0], x[1:] - 0.97 * x[:-1])

    window = np.hanning(n)
    spec = np.fft.rfft(xp * window)
    mag = np.abs(spec) + 1e-8
    log_mag = np.log(mag)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    centroid = float(np.sum(freqs * mag) / np.sum(mag))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / np.sum(mag)))
    cumsum = np.cumsum(mag)
    rolloff = float(freqs[min(np.searchsorted(cumsum, 0.85 * cumsum[-1]), len(freqs) - 1)])
    flatness = float(np.exp(np.mean(log_mag)) / (np.mean(mag) + 1e-8))
    flux = float(np.mean(np.diff(mag) ** 2))
    mid = len(mag) // 2
    hf_ratio = float(np.sum(mag[mid:]) / (np.sum(mag) + 1e-8))
    zcr = float(np.mean(np.abs(np.diff(np.sign(xp)))))
    energy = float(np.sqrt(np.mean(xp ** 2)))
    sk = float(skew(xp))
    ku = float(kurtosis(xp))

    # Autocorrelation peak (periodicity)
    if n > 64:
        acf = np.correlate(xp, xp, mode="full")[n - 1 :]
        acf = acf / (acf[0] + 1e-8)
        min_lag = max(1, int(sr / 400))
        max_lag = min(len(acf) - 1, int(sr / 50))
        peak = float(np.max(acf[min_lag:max_lag])) if max_lag > min_lag else 0.0
    else:
        peak = 0.0

    envelope = np.abs(signal.hilbert(xp))
    mod_energy = float(np.var(envelope))

    classical = np.array(
        [
            centroid / (sr / 2),
            bandwidth / (sr / 2),
            rolloff / (sr / 2),
            flatness,
            flux,
            hf_ratio,
            zcr,
            energy,
            sk,
            ku,
            peak,
            mod_energy,
            np.mean(log_mag),
            np.std(log_mag),
            np.percentile(mag, 25),
            np.percentile(mag, 75),
        ],
        dtype=np.float32,
    )

    # STRF residual fingerprint
    fp = extract_residual_fingerprint(x, sr)
    strf = fp.vector
    if len(strf) < 16:
        strf = np.pad(strf, (0, 16 - len(strf)))
    else:
        strf = strf[:16]

    # MFCC-lite: log-mel filterbank DCT (telephone-aware 20 bands  ->  16 coeffs)
    n_mels = 20
    # Simple triangular mel filters on existing magnitude
    mel_lo = 2595 * np.log10(1 + 300 / 700)  # telephone low
    mel_hi = 2595 * np.log10(1 + min(3400, sr / 2 - 1) / 700)
    mel_points = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, len(mag) - 1)

    fbank = np.zeros(n_mels, dtype=np.float64)
    for m in range(n_mels):
        left, center, right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, center):
            fbank[m] += mag[k] * (k - left) / (center - left)
        for k in range(center, min(right, len(mag))):
            fbank[m] += mag[k] * (right - k) / (right - center)
    log_fbank = np.log(fbank + 1e-8)
    # DCT-II
    mfcc = np.zeros(16, dtype=np.float32)
    for i in range(16):
        mfcc[i] = float(
            np.sum(log_fbank * np.cos(np.pi * i * (np.arange(n_mels) + 0.5) / n_mels))
        )

    # Higher-order / modulation stats (16)
    # AM/FM proxies + spectral entropy
    p = mag / (np.sum(mag) + 1e-8)
    spectral_entropy = float(-np.sum(p * np.log(p + 1e-12)))
    # Sub-band energy ratios (4 bands in telephone range)
    bands = np.array_split(mag, 4)
    band_e = np.array([float(np.sum(b)) for b in bands], dtype=np.float32)
    band_e = band_e / (np.sum(band_e) + 1e-8)
    delta_env = float(np.mean(np.abs(np.diff(envelope))))
    crest = float(np.max(np.abs(xp)) / (energy + 1e-8))
    higher = np.zeros(16, dtype=np.float32)
    higher[0] = spectral_entropy
    higher[1:5] = band_e
    higher[5] = delta_env
    higher[6] = crest
    higher[7] = fp.residual_energy_ratio
    higher[8] = fp.residual_flatness
    higher[9] = fp.grid_artifact_score
    higher[10] = fp.phase_irregularity
    higher[11] = fp.harmonic_to_noise if np.isfinite(fp.harmonic_to_noise) else 0.0
    higher[12] = float(np.mean(np.abs(np.diff(log_mag))))
    if len(log_mag) > 3 and float(np.std(log_mag)) > 1e-6:
        with np.errstate(all="ignore"):
            higher[13] = float(skew(log_mag))
            higher[14] = float(kurtosis(log_mag))
            if not np.isfinite(higher[13]):
                higher[13] = 0.0
            if not np.isfinite(higher[14]):
                higher[14] = 0.0
    else:
        higher[13] = 0.0
        higher[14] = 0.0
    higher[15] = energy

    feats = np.concatenate([classical, strf.astype(np.float32), mfcc, higher])
    # Sanitize
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if len(feats) < FEATURE_DIM:
        feats = np.pad(feats, (0, FEATURE_DIM - len(feats)))
    else:
        feats = feats[:FEATURE_DIM]
    return feats


def aggregate_temporal(history: list[np.ndarray], max_frames: int = 20) -> np.ndarray:
    """
    Temporal aggregation for streaming context: mean, std, delta-mean.
    Returns FEATURE_DIM vector.
    """
    if not history:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    stacked = np.stack(history[-max_frames:], axis=0)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    if len(stacked) >= 2:
        delta = np.mean(np.diff(stacked, axis=0), axis=0)
    else:
        delta = np.zeros_like(mean)
    # Blend into fixed dim
    agg = 0.5 * mean + 0.3 * std + 0.2 * delta
    return agg.astype(np.float32)
