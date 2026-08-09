"""
Evaluation harness for telephony-condition benchmarks.

Measures:
  - EER / AUC under clean vs Channel Twin profiles
  - Per-frame latency
  - Linguistic-only / acoustic-only / fused ablations
  - Coverage-debt recovery after few-shot adaptation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence
import time

import numpy as np

from .metrics import equal_error_rate, auc_roc, average_precision, brier_score, expected_calibration_error, summarize_scores
from ..audio.channel import TelephonyChannelTwin, ChannelConfig, CodecProfile
from ..audio.preprocessor import TelephonyPreprocessor
from ..acoustic.scorer import AcousticDeepfakeScorer
from ..linguistic.scorer import LinguisticFraudScorer
from ..fusion.engine import FusionEngine
from ..pipeline import ShieldCallPipeline, PipelineConfig
from ..linguistic.asr_bridge import ScheduledTranscriptASR


@dataclass
class EvalSample:
    audio: np.ndarray
    sample_rate: int
    is_synthetic: bool
    transcript: str = ""
    condition: str = "clean"
    family: str = "unknown"


@dataclass
class EvalResult:
    condition: str
    n_samples: int
    eer_estimate: float
    mean_latency_ms: float
    auc: float = 0.5
    average_precision: float = 0.0
    brier: float = 1.0
    ece: float = 1.0
    notes: str = ""
    extras: Dict[str, float] = field(default_factory=dict)


def run_basic_latency_test(engine_push_fn: Callable, n_frames: int = 200) -> float:
    times = []
    dummy = np.random.randn(200).astype(np.float32) * 0.1
    for _ in range(n_frames):
        t0 = time.perf_counter()
        engine_push_fn(dummy)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def _synth_tone(sr: int, duration: float, freq: float, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """Human-like proxy: few harmonics + breathy/glottal residual noise."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = 0.35 * np.sin(2 * np.pi * freq * t)
    x += 0.12 * np.sin(2 * np.pi * 2 * freq * t + 0.1)
    x += 0.06 * np.sin(2 * np.pi * 3 * freq * t)
    # Natural-ish residual: higher energy, heavier tails, amplitude modulation
    breath = noise * rng.randn(len(t))
    breath *= 1.0 + 0.4 * np.sin(2 * np.pi * 4.0 * t)  # syllabic AM
    x += breath
    # Mild shimmer / jitter
    x += 0.02 * np.sin(2 * np.pi * (freq + 3.0 * np.sin(2 * np.pi * 5 * t)) * t)
    return np.clip(x, -1, 1).astype(np.float32)


def _synth_vocoder_like(sr: int, duration: float, freq: float, seed: int = 0) -> np.ndarray:
    """
    Synthetic proxy for neural vocoder output: many phase-locked harmonics,
    near-zero residual noise, periodic grid modulation — for offline eval
    without real TTS model weights.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = np.zeros_like(t)
    for k in range(1, 16):
        # Near-perfect harmonic series (overly clean phase lock)
        x += (0.5 / k) * np.sin(2 * np.pi * k * freq * t)
    # Tiny residual (too clean vs human breathiness)
    x += 0.002 * rng.randn(len(t))
    # Upsampling / hop-grid artifact
    x *= 1.0 + 0.06 * np.sin(2 * np.pi * 200 * t)
    x *= 1.0 + 0.04 * np.sin(2 * np.pi * 400 * t)
    peak = np.max(np.abs(x)) + 1e-8
    x = x / peak * 0.9
    return np.clip(x, -1, 1).astype(np.float32)


def generate_synthetic_benchmark(
    n_human: int = 20,
    n_synth: int = 20,
    sr: int = 8000,
    duration: float = 1.5,
    seed: int = 0,
) -> List[EvalSample]:
    rng = np.random.RandomState(seed)
    samples: List[EvalSample] = []
    for i in range(n_human):
        f0 = rng.uniform(100, 220)
        audio = _synth_tone(sr, duration, f0, noise=0.04, seed=seed + i)
        samples.append(
            EvalSample(
                audio=audio,
                sample_rate=sr,
                is_synthetic=False,
                transcript="Hello, this is a reminder about your appointment tomorrow.",
                family="human",
            )
        )
    for i in range(n_synth):
        f0 = rng.uniform(100, 220)
        audio = _synth_vocoder_like(sr, duration, f0, seed=seed + 1000 + i)
        samples.append(
            EvalSample(
                audio=audio,
                sample_rate=sr,
                is_synthetic=True,
                transcript="We detected unusual activity. Please verify your social security number immediately and purchase gift cards.",
                family="vocoder_proxy",
            )
        )
    return samples


def evaluate_acoustic_channel(
    samples: Sequence[EvalSample],
    channel_profile: CodecProfile = CodecProfile.NARROWBAND,
    seed: int = 0,
) -> EvalResult:
    channel = TelephonyChannelTwin(ChannelConfig(profile=channel_profile, seed=seed))
    scorer = AcousticDeepfakeScorer(seed=seed)
    pre = TelephonyPreprocessor(target_sr=8000)
    labels, scores, latencies = [], [], []

    for s in samples:
        audio = channel.apply(s.audio, s.sample_rate)
        pre.reset()
        scorer.reset()
        frame_scores = []
        for frame in pre.stream_from_array(audio, s.sample_rate, chunk_ms=100.0):
            t0 = time.perf_counter()
            sc = scorer.score_frame(frame)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if sc.is_speech:
                frame_scores.append(sc.synthetic_prob)
        score = float(np.mean(frame_scores)) if frame_scores else 0.0
        scores.append(score)
        labels.append(1 if s.is_synthetic else 0)

    eer = equal_error_rate(labels, scores)
    return EvalResult(
        condition=channel_profile.value,
        n_samples=len(samples),
        eer_estimate=eer,
        mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        auc=auc_roc(labels, scores),
        average_precision=average_precision(labels, scores),
        brier=brier_score(labels, scores),
        ece=expected_calibration_error(labels, scores),
        notes="acoustic-only under channel twin",
    )


def evaluate_fused_pipeline(
    samples: Sequence[EvalSample],
    channel_profile: CodecProfile = CodecProfile.NARROWBAND,
) -> EvalResult:
    labels, scores, latencies = [], [], []
    for s in samples:
        schedule = [(0.3, s.transcript)] if s.transcript else []
        asr = ScheduledTranscriptASR(schedule)
        pipe = ShieldCallPipeline(
            config=PipelineConfig(
                channel=ChannelConfig(profile=channel_profile, seed=0),
                use_conformal=False,  # stable tiers for metric sweep
            ),
            asr=asr,
        )
        t0 = time.perf_counter()
        last_risk = 0.0
        for ev in pipe.stream(s.audio, s.sample_rate):
            if ev.risk is not None:
                last_risk = ev.risk.risk_score
        latencies.append((time.perf_counter() - t0) * 1000.0)
        scores.append(last_risk)
        # Positive label: either synthetic audio OR high-fraud transcript path
        # For dual-stream eval we label by synthetic flag for acoustic stress;
        # fraud transcripts still drive fused risk on synth samples in generator.
        labels.append(1 if s.is_synthetic else 0)

    return EvalResult(
        condition=f"fused_{channel_profile.value}",
        n_samples=len(samples),
        eer_estimate=equal_error_rate(labels, scores),
        mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        auc=auc_roc(labels, scores),
        average_precision=average_precision(labels, scores),
        brier=brier_score(labels, scores),
        ece=expected_calibration_error(labels, scores),
        notes="full dual-stream pipeline",
    )


def evaluate_adaptation_recovery(
    n_shots: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Measure how quickly prototype memory reduces coverage gap after
    few-shot exposure to a new 'family'.
    """
    rng = np.random.RandomState(seed)
    scorer = AcousticDeepfakeScorer(seed=seed)
    # New synthesizer family: offset cluster in embedding space
    family_center = scorer.memory._synth_mean.copy() if scorer.memory._synth_mean is not None else np.zeros(scorer.memory.dim, dtype=np.float32)
    family_center = family_center + rng.randn(scorer.memory.dim).astype(np.float32) * 1.2 + 0.8
    # Measure gap on a fixed probe *before* any shots
    probe = family_center + rng.randn(scorer.memory.dim).astype(np.float32) * 0.05
    gap_before = scorer.memory.coverage_gap(probe)
    for i in range(n_shots):
        emb = family_center + rng.randn(scorer.memory.dim).astype(np.float32) * 0.08
        scorer.adapt(emb, is_synthetic=True)
    gap_after = scorer.memory.coverage_gap(probe)
    return {
        "mean_gap_before": float(gap_before),
        "mean_gap_after": float(gap_after),
        "gap_reduction": float(gap_before - gap_after),
        "n_shots": float(n_shots),
    }


def run_full_benchmark(seed: int = 0) -> Dict[str, EvalResult]:
    samples = generate_synthetic_benchmark(seed=seed)
    results = {}
    for profile in [
        CodecProfile.CLEAN,
        CodecProfile.NARROWBAND,
        CodecProfile.G711_ULAW,
        CodecProfile.HARSH_VOIP,
    ]:
        results[f"acoustic_{profile.value}"] = evaluate_acoustic_channel(samples, profile, seed=seed)
    results["fused_narrowband"] = evaluate_fused_pipeline(samples, CodecProfile.NARROWBAND)
    adapt = evaluate_adaptation_recovery(seed=seed)
    results["adaptation"] = EvalResult(
        condition="adaptation_recovery",
        n_samples=int(adapt["n_shots"]),
        eer_estimate=0.0,
        mean_latency_ms=0.0,
        notes="few-shot coverage gap recovery",
        extras=adapt,
    )
    return results
