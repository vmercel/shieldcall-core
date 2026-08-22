"""Classical vocoders used as *controlled* spoof conditions.

These are not HiFi-GAN / WaveNet / commercial TTS. They resynthesize
real speech so that linguistic content is matched and only the
production model changes. That is the honest analogue of logical-access
spoofing that we can run without the ASVspoof license.

Conditions
----------
- ``lpc``: low-order LPC vocoder with pulse/noise excitation.
- ``griffin_lim``: magnitude STFT + iterative phase reconstruction.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal

from ..acoustic.residual import _estimate_f0

VocoderName = Literal["lpc", "griffin_lim", "pulse_formant", "neural_quant"]


def _frames(x: np.ndarray, frame_len: int, hop: int) -> list[np.ndarray]:
    out = []
    if len(x) < frame_len:
        pad = np.zeros(frame_len, dtype=np.float64)
        pad[: len(x)] = x
        return [pad]
    for start in range(0, len(x) - frame_len + 1, hop):
        out.append(x[start : start + frame_len].astype(np.float64))
    return out


def _levinson_durbin(r: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Return LPC polynomial ``a`` (a[0]=1) and residual energy."""
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    e = float(r[0] + 1e-12)
    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -acc / (e + 1e-12)
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new
        e *= max(1.0 - k * k, 1e-8)
    return a, e


def lpc_vocode(
    x: np.ndarray,
    sr: int,
    order: int = 12,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """LPC vocoder: replace the residual with pulse train or white noise."""
    x = x.astype(np.float64)
    frame_len = max(32, int(sr * frame_ms / 1000.0))
    hop = max(8, int(sr * hop_ms / 1000.0))
    window = np.hanning(frame_len)
    y = np.zeros(len(x) + frame_len, dtype=np.float64)
    wsum = np.zeros_like(y)
    rng = np.random.RandomState(0)
    pulse_phase = 0.0

    for i, start in enumerate(range(0, max(1, len(x) - frame_len + 1), hop)):
        frame = x[start : start + frame_len]
        if len(frame) < frame_len:
            break
        fw = frame * window
        # Autocorrelation 0..order
        r = np.array(
            [np.dot(fw[: frame_len - k], fw[k:]) for k in range(order + 1)],
            dtype=np.float64,
        )
        a, e = _levinson_durbin(r, order)
        f0 = _estimate_f0(frame.astype(np.float32), sr)
        gain = np.sqrt(max(e, 1e-12) / frame_len)
        if f0 >= 70:
            period = sr / float(f0)
            idx = []
            t = pulse_phase
            n = 0
            while n < frame_len:
                if t >= period:
                    t -= period
                    idx.append(n)
                t += 1.0
                n += 1
            pulse_phase = t
            exc = np.zeros(frame_len, dtype=np.float64)
            if idx:
                exc[np.array(idx, dtype=int)] = gain * np.sqrt(period)
            else:
                exc[0] = gain
        else:
            exc = rng.randn(frame_len) * gain
            pulse_phase = 0.0
        synth = signal.lfilter([1.0], a, exc)
        y[start : start + frame_len] += synth * window
        wsum[start : start + frame_len] += window

    wsum = np.maximum(wsum, 1e-8)
    y = y[: len(x)] / wsum[: len(x)]
    peak = np.max(np.abs(y)) + 1e-8
    return np.clip((y / peak) * 0.9, -1.0, 1.0).astype(np.float32)


def griffin_lim_vocode(
    x: np.ndarray,
    sr: int,
    n_iter: int = 8,
    nperseg: int = 256,
    noverlap: int = 192,
) -> np.ndarray:
    """Griffin–Lim reconstruction from the magnitude spectrogram."""
    x = x.astype(np.float64)
    _, _, zxx = signal.stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(zxx)
    rng = np.random.RandomState(1)
    phase = rng.uniform(-np.pi, np.pi, size=mag.shape)
    reconstructed = x
    for _ in range(n_iter):
        zhat = mag * np.exp(1j * phase)
        _, reconstructed = signal.istft(zhat, fs=sr, nperseg=nperseg, noverlap=noverlap)
        reconstructed = reconstructed[: len(x)]
        _, _, z2 = signal.stft(reconstructed, fs=sr, nperseg=nperseg, noverlap=noverlap)
        phase = np.angle(z2)
    peak = np.max(np.abs(reconstructed)) + 1e-8
    y = (reconstructed / peak) * 0.9
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return np.clip(y[: len(x)], -1.0, 1.0).astype(np.float32)


def pulse_formant_vocode(x: np.ndarray, sr: int) -> np.ndarray:
    """Robotic pulse-train + 3-formant vocoder (clearly non-glottal)."""
    x = x.astype(np.float64)
    f0 = _estimate_f0(x.astype(np.float32), sr)
    if f0 < 70:
        f0 = 110.0
    period = max(int(sr / f0), 2)
    exc = np.zeros(len(x), dtype=np.float64)
    exc[::period] = 1.0
    y = exc
    for freq, bw in ((500.0, 90.0), (1500.0, 110.0), (2500.0, 140.0)):
        r = np.exp(-np.pi * bw / sr)
        theta = 2.0 * np.pi * freq / sr
        y = signal.lfilter([1.0], [1.0, -2.0 * r * np.cos(theta), r * r], y)
    analytic = signal.hilbert(x)
    env = np.abs(analytic)
    wn = min(8.0 / (sr / 2.0), 0.99)
    b, a = signal.butter(1, wn, btype="low")
    env = np.abs(signal.lfilter(b, a, env))
    env = env / (np.max(env) + 1e-8)
    y = y / (np.max(np.abs(y)) + 1e-8) * env * 0.9
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def neural_quant_vocode(x: np.ndarray, sr: int, bits: int = 5, n_iter: int = 8) -> np.ndarray:
    """STFT magnitude quantization + phase noise (Encodec/DAC-like artifacts).

    This is not a licensed neural codec. It is a *telephony-like neural-codec
    surrogate*: coarse codebook on log-magnitudes and scrambled residual
    phase. Headline tables may use it; do not call it HiFi-GAN or Encodec.
    """
    x = x.astype(np.float64)
    nper = 256
    nover = 192
    _, _, z = signal.stft(x, fs=sr, nperseg=nper, noverlap=nover)
    mag = np.abs(z)
    logm = np.log1p(mag)
    levels = 2 ** bits
    lo, hi = float(logm.min()), float(logm.max() + 1e-8)
    q = np.round((logm - lo) / (hi - lo) * (levels - 1))
    rec = np.expm1(q / (levels - 1) * (hi - lo) + lo)
    rng = np.random.RandomState(3)
    phase = np.angle(z) + rng.randn(*z.shape) * 0.65
    # extra high-band phase scramble (neural codecs smear 3–4 kHz)
    freqs = np.linspace(0, sr / 2, z.shape[0])
    high = freqs > 2400
    phase[high] = rng.uniform(-np.pi, np.pi, size=phase[high].shape)
    rec_c = rec * np.exp(1j * phase)
    _, y = signal.istft(rec_c, fs=sr, nperseg=nper, noverlap=nover)
    y = y[: len(x)]
    peak = np.max(np.abs(y)) + 1e-8
    return np.clip(y / peak * 0.95, -1.0, 1.0).astype(np.float32)


def vocode(x: np.ndarray, sr: int, name: VocoderName) -> np.ndarray:
    if name == "lpc":
        return lpc_vocode(x, sr)
    if name == "griffin_lim":
        return griffin_lim_vocode(x, sr)
    if name == "pulse_formant":
        return pulse_formant_vocode(x, sr)
    if name == "neural_quant":
        return neural_quant_vocode(x, sr)
    raise ValueError(f"unknown vocoder {name}")
