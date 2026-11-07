"""Numpy LFCC front end + silence trim (reference baseline; no torch/librosa)."""
from __future__ import annotations
import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly
from math import gcd


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def trim_silence(x: np.ndarray, sr: int, rel_db: float = -40.0, frame_ms: float = 20.0) -> np.ndarray:
    """Cut leading/trailing frames below ``rel_db`` of the loudest frame.
    ASVspoof silence duration is a known shortcut; the benchmark removes it."""
    n = int(sr * frame_ms / 1000)
    m = len(x) // n
    if m < 3:
        return x
    rms = np.sqrt((x[: m * n].reshape(m, n) ** 2).mean(1) + 1e-12)
    keep = np.where(rms > rms.max() * 10 ** (rel_db / 20))[0]
    if len(keep) == 0:
        return x
    return x[keep[0] * n:(keep[-1] + 1) * n]


_FB = {}


def _filterbank(sr: int, nfft: int, n_filt: int = 20) -> np.ndarray:
    key = (sr, nfft, n_filt)
    if key in _FB:
        return _FB[key]
    edges = np.linspace(0, sr / 2, n_filt + 2)
    bins = np.fft.rfftfreq(nfft, 1 / sr)
    fb = np.zeros((n_filt, len(bins)), np.float32)
    for i in range(n_filt):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (bins - lo) / (c - lo)
        dn = (hi - bins) / (hi - c)
        fb[i] = np.clip(np.minimum(up, dn), 0, None)
    _FB[key] = fb
    return fb


def lfcc(x: np.ndarray, sr: int, n_ceps: int = 20, win_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    """(T, 3*n_ceps) LFCC + delta + delta-delta, utterance-level CMVN."""
    x = np.asarray(x, np.float32)
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    w, h = int(sr * win_ms / 1000), int(sr * hop_ms / 1000)
    if len(x) < w + h:
        x = np.pad(x, (0, w + h - len(x)))
    nfr = 1 + (len(x) - w) // h
    idx = np.arange(w)[None, :] + h * np.arange(nfr)[:, None]
    fr = x[idx] * np.hamming(w).astype(np.float32)
    nfft = 1 << (w - 1).bit_length()
    p = np.abs(np.fft.rfft(fr, nfft)) ** 2
    e = np.log(p @ _filterbank(sr, nfft).T + 1e-10)
    c = dct(e, type=2, axis=1, norm="ortho")[:, :n_ceps]

    def delta(a):
        pad = np.pad(a, ((2, 2), (0, 0)), mode="edge")
        return (2 * (pad[4:] - pad[:-4]) + (pad[3:-1] - pad[1:-3])) / 10.0

    d1 = delta(c)
    d2 = delta(d1)
    f = np.concatenate([c, d1, d2], 1)
    return ((f - f.mean(0)) / (f.std(0) + 1e-6)).astype(np.float32)
