#!/usr/bin/env python3
"""P2-3: multi-seed confirmatory reruns of the paper experiments.

Re-runs the seed-dependent paper-experiment components (acoustic,
operational fusion, adaptation) across SEEDS and reports mean, sample SD,
and 95% t-confidence intervals per metric. The linguistic protocol takes
no seed and is therefore not re-run.

Writes docs/results/multiseed.json with per-seed values and the summary.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

import _repo  # noqa: F401

from shieldcall.audio.channel import CodecProfile
from shieldcall.eval.protocols import (
    acoustic_protocol,
    adaptation_protocol,
    operational_fusion_protocol,
)
from shieldcall.eval.speech_data import speech_available

SEEDS = (0, 1, 2, 3, 4)

# t_{0.975, df} for df = n-1 (n=5 -> 2.776)
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def summarize(values: list[float]) -> dict:
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    half = T_975[n - 1] * sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "values": [round(v, 4) for v in values],
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "ci95": [round(mean - half, 4), round(mean + half, 4)],
    }


def main() -> None:
    t0 = time.perf_counter()
    if not speech_available():
        print("Mini LibriSpeech missing. Run: python scripts/download_speech.py")
        raise SystemExit(2)
    out_dir = Path("docs/results")

    per_seed: dict = {}
    # metric_key -> list of values across seeds
    agg: dict[str, list[float]] = {}

    def record(metric: str, value: float) -> None:
        agg.setdefault(metric, []).append(float(value))

    for seed in SEEDS:
        print(f"--- seed {seed} ---", flush=True)
        ac, scorer, train, test, test_spoof = acoustic_protocol(
            n_train_speakers=5,
            n_test_speakers=5,
            utt_per_speaker=2,
            vocoders=("pulse_formant",),
            profiles=(CodecProfile.CLEAN, CodecProfile.NARROWBAND),
            seed=seed,
        )
        seed_rec: dict = {"acoustic": {}, "operational": {}, "adaptation": {}}
        for k, v in ac.items():
            seed_rec["acoustic"][k] = {"eer": v.eer_estimate, "auc": v.auc, "n": v.n_samples}
            record(f"acoustic/{k}/eer", v.eer_estimate)
            record(f"acoustic/{k}/auc", v.auc)
            print(f"  {k:42s} EER={v.eer_estimate:.3f} AUC={v.auc:.3f}")

        spoof_key = "pulse_formant" if "pulse_formant" in test_spoof else next(iter(test_spoof))
        _, _, op = operational_fusion_protocol(scorer, test, test_spoof[spoof_key], seed=seed)
        for k, v in op.items():
            seed_rec["operational"][k] = {"eer": v.eer_estimate, "auc": v.auc, "n": v.n_samples}
            record(f"operational/{k}/eer", v.eer_estimate)
            record(f"operational/{k}/auc", v.auc)
            rec = v.extras.get("disagreement_recall@0.5")
            if rec is not None:
                record(f"operational/{k}/disagreement_recall@0.5", rec)
            print(f"  {k:42s} EER={v.eer_estimate:.3f} AUC={v.auc:.3f}")

        adapt = adaptation_protocol(train, test, n_shots=5, seed=seed)
        seed_rec["adaptation"] = adapt
        for mk in ("eer_before", "eer_after", "eer_reduction"):
            record(f"adaptation/{mk}", adapt[mk])
        print(
            f"  adaptation: EER before={adapt['eer_before']:.3f} "
            f"after={adapt['eer_after']:.3f} reduction={adapt['eer_reduction']:.3f}"
        )
        per_seed[str(seed)] = seed_rec

    summary = {metric: summarize(vals) for metric, vals in sorted(agg.items())}
    payload = {
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "summary": summary,
        "seconds": time.perf_counter() - t0,
        "notes": (
            "Multi-seed reruns of the seed-dependent paper-experiment components. "
            "The linguistic protocol takes no seed and was not re-run. "
            "CI95 uses the t-distribution (df=n-1)."
        ),
    }
    (out_dir / "multiseed.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_dir / 'multiseed.json'} in {payload['seconds']:.1f}s")

    print("\nSummary (mean +/- 95% CI):")
    for metric, s in summary.items():
        lo, hi = s["ci95"]
        print(f"  {metric:52s} {s['mean']:.3f}  [{lo:.3f}, {hi:.3f}]  (sd={s['sd']:.3f})")


if __name__ == "__main__":
    main()
