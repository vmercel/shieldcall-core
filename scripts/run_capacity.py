#!/usr/bin/env python3
"""Measure ms/frame and print a capacity / cost plan.

This is the operational number. It is host-specific. Do not paste it
into the paper as a universal constant; paste the *formula* and this
host's reading as an example.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from shieldcall.pipeline import PipelineConfig, ShieldCallPipeline
from shieldcall.runtime.capacity import plan_capacity
from shieldcall.runtime.slo import SLO

ROOT = Path(__file__).resolve().parents[1]


def measure_ms_per_frame(seconds: float = 2.0, sr: int = 8000) -> dict:
    rng = np.random.default_rng(0)
    audio = (0.08 * rng.standard_normal(int(sr * seconds))).astype(np.float32)
    pipe = ShieldCallPipeline(PipelineConfig(use_conformal=False, fuse_every_n_frames=5))
    list(pipe.stream(audio[: sr // 5], sr))  # warmup
    pipe.reset()
    t0 = time.perf_counter()
    n = 0
    for _ in pipe.stream(audio, sr, chunk_ms=20):
        n += 1
    wall = time.perf_counter() - t0
    ms = 1000.0 * wall / max(n, 1)
    return {
        "audio_seconds": seconds,
        "frames": n,
        "wall_seconds": round(wall, 4),
        "ms_per_frame": round(ms, 3),
        "realtime_x": round(seconds / wall, 3),
    }


def main() -> None:
    slo = SLO()
    m = measure_ms_per_frame()
    plan = plan_capacity(m["ms_per_frame"], hop_ms=slo.hop_ms, target_concurrent_calls=1000)
    out = {
        "slo_hop_ms": slo.hop_ms,
        "slo_frame_budget_ms": slo.frame_budget_ms,
        "within_budget": slo.within_frame_budget(m["ms_per_frame"]),
        "measurement": m,
        "plan_1000_concurrent": {
            "calls_per_core": round(plan.calls_per_core, 3),
            "cores": plan.cores_for_target,
            "n_plus_1_cores": plan.n_plus_1_cores,
            "bytes_per_call_sec": plan.bytes_per_call_sec,
            "monthly_usd_illustrative": round(plan.monthly_usd_illustrative, 0),
            "usd_per_vcpu_hour_parameter": plan.usd_per_vcpu_hour,
            "notes": plan.notes,
        },
    }
    dest = ROOT / "docs" / "results" / "capacity.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
