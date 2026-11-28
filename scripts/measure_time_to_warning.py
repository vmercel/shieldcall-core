#!/usr/bin/env python3
"""ADR-019: measure time-to-warning end to end, in process, on synthetic calls.

Runs one synthetic scam call and one benign call several
times through fresh pipelines and prints a JSON report
with the per-stage figures from TDD section 10.4:
accumulation, deciding-chunk processing, Tier 1 latency,
and the estimated live warning time. ASR provider lag,
render time, and network time are unmeasured in this
lab run and are reported as absent, never as zero, so
the full 2 s p95 verdict is incomplete by construction.
No licensed data, no network, no ASR provider, and no
model file are used; Tier 1 latency is therefore absent
unless the operator supplies a scorer, which this
script does not.

Usage: python scripts/measure_time_to_warning.py [--runs N]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _repo  # noqa: F401  (adds the repo root to sys.path)

from shieldcall import PipelineConfig, ShieldCallPipeline
from shieldcall.eval.time_to_warning import (
    measure_time_to_warning,
    run_repeated,
    summarize_runs,
)
from shieldcall.linguistic.asr_bridge import ScheduledTranscriptASR

SR = 8000


def _speech_audio(seconds: float) -> np.ndarray:
    """Gated-burst audio the VAD accepts as speech (ADR-018 pattern)."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    gate = (np.sin(2 * np.pi * 4 * t) > -0.2).astype(np.float32)
    vowel = (
        0.25 * np.sin(2 * np.pi * 150 * t)
        + 0.15 * np.sin(2 * np.pi * 300 * t)
        + 0.08 * np.sin(2 * np.pi * 450 * t)
    )
    return np.clip(vowel * gate + 0.02 * rng.standard_normal(len(t)), -0.9, 0.9).astype(np.float32)


def _scam_factory():
    schedule = [
        (0.5, "Hello from your bank"),
        (
            2.5,
            "Unusual activity detected, verify social security number "
            "and buy gift cards immediately, do not tell anyone",
        ),
    ]
    return ShieldCallPipeline(
        config=PipelineConfig(shadow_audit_seed=0),
        asr=ScheduledTranscriptASR(schedule),
    )


def _benign_factory():
    schedule = [
        (0.5, "Hello, how is the weather today"),
        (2.0, "Let us meet for lunch tomorrow"),
    ]
    return ShieldCallPipeline(
        config=PipelineConfig(shadow_audit_seed=0),
        asr=ScheduledTranscriptASR(schedule),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="repetitions per call")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    audio = _speech_audio(6.0)
    scam_measurements, scam_summary = run_repeated(
        _scam_factory, audio, SR, args.runs, evidence_end_sec=2.5
    )
    benign = measure_time_to_warning(
        _benign_factory(), audio, SR, evidence_end_sec=2.0
    )
    report = {
        "measurement": "time-to-warning, in process only (ADR-019, TDD 10.4)",
        "environment": "lab run on the local host (%s); not reference hardware" % platform.node(),
        "unmeasured_stages": [
            "asr_lag_sec (no ASR provider)",
            "render_sec (no app render)",
            "network_sec (no network)",
            "tier1_latency_sec (no Tier 1 scorer configured)",
        ],
        "budget": {
            "warning_p95_sec": 2.0,
            "accumulation_sec": 0.7,
            "tier1_sec": 0.2,
            "deciding_chunk_sec": 0.2,
        },
        "scam_call": {
            "evidence_end_sec": 2.5,
            "runs": [m.to_dict() for m in scam_measurements],
            "summary": scam_summary.to_dict(),
        },
        "benign_call": benign.to_dict(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
