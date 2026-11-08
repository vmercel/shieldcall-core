"""ShieldCall Phase-3 feature engineering (numpy/scipy only).

Design elements (measured against ablations in docs/results/PHASE3_ITERATIONS.md; none is claimed novel until prior-art review):
  * Band-locked front end: every view, whatever its source rate, is mapped to the 8 kHz telephone domain before analysis,
    so train and test share one bandwidth.
  * Phase channel: modified group-delay cepstrum (MGD) of each frame.
  * Fine-structure channel ("harmonic residual"): cepstral peak, pitch-quefrency, HNR proxy, band flatness, flux.
Each function returns (T, D) float32 arrays, utterance-level CMVN applied by ``build``.
"""
from __future__ import annotations
import numpy as np
from scipy.fft import dct
from .lfcc import resample, lfcc

SR = 8000
WIN, HOP, NFFT = 200, 80, 512


def _frames(x: np.ndarray) -> np.ndarray:
    x = np.append(x[0], x[1:] - 0.97 * x[:-1]).astype(np.float32)
    if len(x) < WIN + HOP:
        x = np.pad(x, (0, WIN + HOP - len(x)))
    n = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def to_band(x: np.ndarray, sr: int) -> np.ndarray:
    return resample(x, sr, SR) if sr != SR else x.astype(np.float32)


def mgd(fr: np.ndarray, n_ceps: int = 20, gamma: float = 0.9, alpha: float = 0.4) -> np.ndarray:
    w = np.hamming(WIN).astype(np.float32)
    xw = fr * w
    yw = xw * np.arange(1, WIN + 1, dtype=np.float32)[None, :]
    X = np.fft.rfft(xw, NFFT); Y = np.fft.rfft(yw, NFFT)
    logm = np.log(np.abs(X) + 1e-8)
    cep = np.fft.irfft(logm, NFFT)
    cep[:, 30:NFFT - 30] = 0                      # cepstrally smoothed envelope S
    S = np.exp(np.fft.rfft(cep, NFFT).real)
    tau = (X.real * Y.real + X.imag * Y.imag) / (S ** (2 * gamma) + 1e-8)
    tau = np.sign(tau) * np.abs(tau) ** alpha
    return dct(tau[:, :256], type=2, axis=1, norm="ortho")[:, 1:n_ceps + 1].astype(np.float32)


def fine(fr: np.ndarray) -> np.ndarray:
    w = np.hamming(WIN).astype(np.float32)
    X = np.fft.rfft(fr * w, NFFT)
    p = np.abs(X) ** 2 + 1e-10
    lp = np.log(p)
    cep = np.fft.irfft(lp, NFFT)[:, :NFFT // 2]
    lo, hi = 20, 133                               # pitch 60-400 Hz at 8 kHz
    seg = cep[:, lo:hi]
    pk = seg.max(1); pos = seg.argmax(1).astype(np.float32) / (hi - lo)
    rest = np.abs(cep[:, hi:200]).mean(1)
    prom = pk - seg.mean(1)
    # band flatness over 4 bands of 0-4 kHz
    edges = [0, 64, 128, 192, 257]
    flat = np.stack([np.exp(np.log(p[:, a:b]).mean(1)) / p[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:])], 1)
    flux = np.r_[0, np.sqrt(((np.diff(lp, axis=0)) ** 2).mean(1))]
    en = lp.mean(1)
    zc = (np.abs(np.diff(np.sign(fr + 1e-9), axis=1)) > 0).mean(1)
    return np.concatenate([np.stack([pk, pos, rest, prom, flux, en, zc], 1), np.log(flat + 1e-6)], 1).astype(np.float32)


def cmvn(f: np.ndarray) -> np.ndarray:
    return ((f - f.mean(0)) / (f.std(0) + 1e-6)).astype(np.float32)


def build(x: np.ndarray, sr: int, sets=("lfcc",)) -> np.ndarray:
    """Concatenate feature channels (T, D). sets subset of: lfcc, mgd, fine."""
    x8 = to_band(x, sr)
    parts = []
    if "lfcc" in sets:
        parts.append(lfcc(x8, SR, n_ceps=20))     # already CMVN'd
    if {"mgd", "fine"} & set(sets):
        fr = _frames(x8)
        if "mgd" in sets: parts.append(cmvn(mgd(fr)))
        if "fine" in sets: parts.append(cmvn(fine(fr)))
    n = min(p.shape[0] for p in parts)
    return np.concatenate([p[:n] for p in parts], 1)
