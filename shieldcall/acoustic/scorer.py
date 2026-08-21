"""
Streaming acoustic authenticity scorer with Prototype Memory Adaptation.

Combines:
  1. STRF residual heuristics (channel-robust synthetic cues)
  2. Prototype Memory (Mahalanobis distance to human vs synthetic manifolds)
  3. Optional online adaptation when confirmed labels arrive

The interface stays stable so FusionEngine never needs to change when
the internal model is upgraded to a neural backend.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence

import numpy as np

from .features import extract_frame_features, aggregate_temporal, FEATURE_DIM
from .residual import extract_residual_fingerprint
from ..audio.preprocessor import Frame


@dataclass
class AcousticScore:
    timestamp_sec: float
    frame_index: int
    synthetic_prob: float  # 0 = likely human, 1 = likely synthetic
    confidence: float
    is_speech: bool
    features: np.ndarray
    residual_cue: float = 0.0
    prototype_cue: float = 0.0
    embedding: Optional[np.ndarray] = None


class PrototypeMemory:
    """
    Online prototype memory for human vs synthetic embeddings.

    Novelty: rather than freezing a global classifier that fails on the
    next unseen vocoder (coverage debt), we maintain class-conditional
    prototypes with diagonal Mahalanobis metrics that accept few-shot
    updates in milliseconds.
    """

    def __init__(self, dim: int = FEATURE_DIM, max_prototypes: int = 64, seed: int = 42):
        self.dim = dim
        self.max_prototypes = max_prototypes
        self._rng = np.random.RandomState(seed)
        self.human: List[np.ndarray] = []
        self.synthetic: List[np.ndarray] = []
        self._human_mean: Optional[np.ndarray] = None
        self._human_var: Optional[np.ndarray] = None
        self._synth_mean: Optional[np.ndarray] = None
        self._synth_var: Optional[np.ndarray] = None
        self._bootstrap()

    def fit(self, embeddings: Sequence[np.ndarray], is_synthetic: Sequence[bool]) -> None:
        """Replace bootstrap priors with labeled embeddings from real data."""
        self.human = []
        self.synthetic = []
        self._human_mean = None
        self._synth_mean = None
        for emb, y in zip(embeddings, is_synthetic):
            self.add(np.asarray(emb, dtype=np.float32), bool(y))
        if not self.human or not self.synthetic:
            self._bootstrap()

    def _bootstrap(self) -> None:
        """
        Weak random priors so the scorer is callable before any labeled data.
        These are not speech. Call ``fit`` before reporting acoustic metrics.
        """
        for _ in range(12):
            h = self._rng.randn(self.dim).astype(np.float32) * 0.15
            h[7] = abs(h[7]) + 0.08   # energy
            h[10] = 0.4 + 0.2 * self._rng.rand()  # periodicity
            # STRF region: more residual structure for natural speech
            h[16] = 0.25 + 0.1 * self._rng.rand()
            h[18] = 0.5 + 0.5 * self._rng.rand()  # residual kurtosis prior
            self.human.append(h)

            s = self._rng.randn(self.dim).astype(np.float32) * 0.15
            s[7] = abs(s[7]) + 0.05
            s[10] = 0.6 + 0.2 * self._rng.rand()
            s[16] = 0.08 + 0.05 * self._rng.rand()  # low residual energy ratio
            s[17] = 0.7 + 0.2 * self._rng.rand()    # high residual flatness
            s[22] = 2.0 + self._rng.rand()          # grid artifact
            self.synthetic.append(s)
        self._recompute_stats()

    def _recompute_stats(self) -> None:
        if self.human:
            H = np.stack(self.human, axis=0)
            self._human_mean = np.mean(H, axis=0)
            self._human_var = np.var(H, axis=0) + 0.05
        if self.synthetic:
            S = np.stack(self.synthetic, axis=0)
            self._synth_mean = np.mean(S, axis=0)
            self._synth_var = np.var(S, axis=0) + 0.05

    def _maha(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        # Floor variance so high-dim distances stay numerically stable
        v = np.maximum(var, 0.05)
        return float(np.sqrt(np.mean(((x - mean) ** 2) / v)))  # RMS Mahalanobis (scale-free in dim)

    def score(self, embedding: np.ndarray) -> float:
        """Return synthetic probability in [0,1] from relative distances."""
        if self._human_mean is None or self._synth_mean is None:
            return 0.5
        d_h = self._maha(embedding, self._human_mean, self._human_var)
        d_s = self._maha(embedding, self._synth_mean, self._synth_var)
        # Softmax over negative distances (temperature keeps early priors soft)
        logits = np.array([-d_h, -d_s], dtype=np.float64) / 0.35
        logits -= logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()
        return float(probs[1])

    def add(self, embedding: np.ndarray, is_synthetic: bool) -> None:
        emb = embedding.astype(np.float32).copy()
        bucket = self.synthetic if is_synthetic else self.human
        bucket.append(emb)
        if len(bucket) > self.max_prototypes:
            # Drop oldest (FIFO)  -  keeps memory responsive to new synth families
            del bucket[0 : len(bucket) - self.max_prototypes]
        self._recompute_stats()

    def coverage_gap(self, embedding: np.ndarray) -> float:
        """
        How far is this sample from *both* manifolds?
        High gap  ->  unknown synthesizer / domain shift (coverage debt signal).
        """
        if self._human_mean is None or self._synth_mean is None:
            return 1.0
        d_h = self._maha(embedding, self._human_mean, self._human_var)
        d_s = self._maha(embedding, self._synth_mean, self._synth_var)
        # RMS-Mahalanobis ~1 under-manifold; >>2 is strongly OOD
        return float(np.clip(min(d_h, d_s) / 2.5, 0.0, 1.0))


def _strf_heuristic_synth_prob(frame: np.ndarray, sr: int) -> float:
    """
    Physics-informed synthetic probability from residual fingerprint.

    Cues that survive telephony (and separate vocoder-like from breathy speech):
      - periodic grid / upsampling peaks in the residual spectrum
      - when F0 locks: unusually low residual energy + high HNR
      - over-stable phase structure

    Important: high residual energy after a *failed* F0 lock is often natural
    noise (breath, frication)  -  not a synthetic cue. Early versions inverted
    this and scored humans as synthetic.
    """
    fp = extract_residual_fingerprint(frame, sr)
    f0_norm = float(fp.vector[7]) if len(fp.vector) > 7 else 0.0  # f0/400
    voiced = f0_norm > 0.05

    score = 0.12

    # 1) Grid / upsampling artifact  -  strongest telephony-stable cue
    score += float(np.clip(fp.grid_artifact_score / 12.0, 0.0, 0.55))

    if voiced:
        # 2) Overly complete harmonic model fit (little residual left)
        score += float(np.clip(0.55 - fp.residual_energy_ratio, 0.0, 0.55)) * 0.9
        # 3) High harmonic-to-noise after lock
        score += float(np.clip(np.log1p(fp.harmonic_to_noise) / 4.0, 0.0, 0.25))
        # 4) Residual too flat once harmonics removed
        score += float(np.clip(fp.residual_flatness - 0.45, 0.0, 0.4)) * 0.5
    else:
        # Unvoiced / no lock: residual≈signal. Prefer human prior.
        # Only mild elevates if residual is unnaturally peaky (grid without F0).
        if fp.grid_artifact_score > 5:
            score += 0.15
        else:
            score -= 0.05  # breathy residual  ->  slightly more human

    # 5) Phase irregularity: natural speech tends higher; very low  ->  synthetic
    if fp.phase_irregularity < 0.8 and voiced:
        score += 0.08

    return float(np.clip(score, 0.0, 1.0))


class AcousticDeepfakeScorer:
    """
    Streaming acoustic authenticity scorer.

    Fusion of STRF heuristics + prototype memory, with confidence that
    grows as history accumulates and shrinks under coverage-gap (OOD).
    """

    def __init__(
        self,
        history_frames: int = 30,
        seed: int = 42,
        strf_weight: float = 0.45,
        prototype_weight: float = 0.55,
    ):
        self.history_frames = history_frames
        self.strf_weight = strf_weight
        self.prototype_weight = prototype_weight
        self._history: Deque[np.ndarray] = deque(maxlen=history_frames)
        self._rng = np.random.RandomState(seed)
        self.memory = PrototypeMemory(dim=FEATURE_DIM, seed=seed)
        self._is_fitted = True
        # Backward-compat: expose a trivial sklearn-like flag
        self._model = None

    def reset(self) -> None:
        self._history.clear()

    def score_frame(self, frame: Frame) -> AcousticScore:
        feats = extract_frame_features(frame.samples, frame.sample_rate)
        self._history.append(feats)

        if not frame.is_speech:
            return AcousticScore(
                timestamp_sec=frame.timestamp_sec,
                frame_index=frame.frame_index,
                synthetic_prob=0.0,
                confidence=0.0,
                is_speech=False,
                features=feats,
                residual_cue=0.0,
                prototype_cue=0.0,
                embedding=feats,
            )

        agg = aggregate_temporal(list(self._history))
        residual_cue = _strf_heuristic_synth_prob(frame.samples, frame.sample_rate)
        prototype_cue = self.memory.score(agg)
        gap = self.memory.coverage_gap(agg)

        synthetic_prob = (
            self.strf_weight * residual_cue + self.prototype_weight * prototype_cue
        )
        # Mild temporal smoothing via history length
        synthetic_prob = float(np.clip(synthetic_prob, 0.0, 1.0))

        conf = 0.35 + 0.45 * min(len(self._history) / self.history_frames, 1.0)
        conf *= 1.0 - 0.5 * gap  # OOD  ->  lower confidence
        conf = float(np.clip(conf, 0.05, 0.95))

        return AcousticScore(
            timestamp_sec=frame.timestamp_sec,
            frame_index=frame.frame_index,
            synthetic_prob=synthetic_prob,
            confidence=conf,
            is_speech=True,
            features=feats,
            residual_cue=residual_cue,
            prototype_cue=prototype_cue,
            embedding=agg,
        )

    def fit(self, embeddings: Sequence[np.ndarray], is_synthetic: Sequence[bool]) -> None:
        """Fit prototype memory on labeled frame embeddings."""
        self.memory.fit(embeddings, is_synthetic)
        self._is_fitted = True

    def adapt(self, features: np.ndarray, is_synthetic: bool) -> None:
        """Online few-shot update into prototype memory."""
        self.memory.add(features, is_synthetic=is_synthetic)

    def adapt_from_score(self, score: AcousticScore, is_synthetic: bool) -> None:
        emb = score.embedding if score.embedding is not None else score.features
        self.adapt(emb, is_synthetic)
