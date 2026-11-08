"""Truncated wav2vec2 front end (CNN + first N transformer layers) with optional spectral-gate denoising.

Finding (docs/results/PHASE4_ITERATIONS.md): on 8 kHz telephone audio the early layers carry most spoof evidence,
so the encoder can be cut after layer 4 for ~3x less compute with no loss in accuracy.
"""
from __future__ import annotations
import numpy as np, torch

_M = {}


def model(n_layers: int = 4):
    if n_layers not in _M:
        from transformers import Wav2Vec2Model
        m = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()
        m.encoder.layers = m.encoder.layers[:n_layers]
        _M[n_layers] = m
    return _M[n_layers]


def spectral_gate(x: np.ndarray, sr: int = 16000, n_fft: int = 512, hop: int = 128, floor: float = 0.1) -> np.ndarray:
    """Minimum-statistics noise estimate (10th percentile of per-bin power over time) + Wiener-style gain."""
    from scipy.signal import stft, istft
    f, t, Z = stft(x, sr, nperseg=n_fft, noverlap=n_fft - hop)
    P = np.abs(Z) ** 2
    noise = np.percentile(P, 10, axis=1, keepdims=True) * 2.0
    gain = np.maximum(1 - noise / (P + 1e-12), floor) ** 0.5
    _, y = istft(Z * gain, sr, nperseg=n_fft, noverlap=n_fft - hop)
    return y[: len(x)].astype(np.float32)


@torch.no_grad()
def embed(x16: np.ndarray, n_layers: int = 4, denoise: bool = False, max_samples: int = 40000) -> np.ndarray:
    x = x16[:max_samples]
    if denoise:
        x = spectral_gate(x)
    x = (x - x.mean()) / (x.std() + 1e-6)
    hs = model(n_layers)(torch.from_numpy(x.astype(np.float32))[None], output_hidden_states=True).hidden_states
    return np.concatenate([np.r_[h[0].mean(0).numpy(), h[0].std(0).numpy()] for h in hs[1:]]).astype(np.float16)
