#!/usr/bin/env python3
"""Falsify (or support) stage-aligned production change.

Task: distinguish aligned vs unaligned handoffs with the same vocoded
fraction. Mean synthetic probability should not separate them; SAPC
should, if the timing hypothesis is real on this construction.
"""

from __future__ import annotations

import json
from pathlib import Path

from shieldcall.eval.handoff import run_handoff_protocol
from shieldcall.eval.metrics import auc_roc
from shieldcall.eval.speech_data import speech_available
from shieldcall.fusion.aci import AdaptiveConformal
from shieldcall.fusion.coupling import permutation_pvalue
import numpy as np


def sapc_on_clean_point_processes(n: int = 40, seed: int = 0) -> dict:
    """Software check: when alarms *are* aligned to stages, SAPC ranks them.

    This is not speech. It only shows the statistic does what the formula says.
    """
    rng = np.random.RandomState(seed)
    y: list[int] = []
    scores: list[float] = []
    wins = 0
    stages = [5.0, 6.2]
    for i in range(n):
        aligned = permutation_pvalue(
            stages, [5.05 + 0.04 * rng.randn()], period=10.0, n_perm=99, seed=i
        )
        unaligned = permutation_pvalue(
            stages, [1.1 + 0.04 * rng.randn()], period=10.0, n_perm=99, seed=i + 1000
        )
        y.extend([1, 0])
        scores.extend([aligned.score, unaligned.score])
        if aligned.score > unaligned.score:
            wins += 1
    return {
        "n_pairs": float(n),
        "pair_win_rate": float(wins / n),
        "auc": auc_roc(y, scores),
        "note": "synthetic point processes, not audio",
    }


def _aci_demo(n: int = 400, seed: int = 0) -> dict:
    """Empirical coverage of ACI vs a frozen quantile, with a mid-stream shift."""
    rng = np.random.RandomState(seed)
    aci = AdaptiveConformal(alpha=0.1, gamma=0.08, window=250)
    frozen_res = []
    frozen_miss = []
    for i in range(n):
        # Label process shifts at the midpoint: more positives later.
        p = 0.25 if i < n // 2 else 0.75
        y = float(rng.rand() < p)
        score = float(np.clip(0.2 + 0.6 * y + 0.08 * rng.randn(), 0, 1))
        aci.observe(score, y)
        r = abs(score - y)
        if i < 40:
            frozen_res.append(r)
            continue
        q = float(np.quantile(frozen_res, 0.9)) if frozen_res else 0.25
        frozen_miss.append(0 if abs(score - y) <= q else 1)
    return {
        "aci_coverage": aci.empirical_coverage(),
        "aci_target": 0.9,
        "frozen_quantile_coverage": 1.0 - float(np.mean(frozen_miss)) if frozen_miss else float("nan"),
        "n": n,
    }


def main() -> None:
    out = Path("docs/results/handoff_experiment.json")
    payload: dict = {}
    print("SAPC on clean synthetic point processes (not audio)", flush=True)
    syn = sapc_on_clean_point_processes()
    payload["sapc_synthetic"] = syn
    print(
        f"  pair_win={syn['pair_win_rate']:.3f}  AUC={syn['auc']:.3f}  ({syn['note']})",
        flush=True,
    )

    print("ACI coverage demo (synthetic labels, mid-stream prevalence shift)", flush=True)
    aci = _aci_demo()
    payload["aci"] = aci
    print(
        f"  ACI empirical coverage={aci['aci_coverage']:.3f} "
        f"(target {aci['aci_target']:.2f}); "
        f"frozen quantile coverage={aci['frozen_quantile_coverage']:.3f}",
        flush=True,
    )

    if not speech_available():
        print("No Mini LibriSpeech; skip handoff audio protocol.")
        payload["error"] = "no_speech"
        out.write_text(json.dumps(payload, indent=2))
        raise SystemExit(2)

    print("Handoff protocol (pulse-formant tail, held-out script h01)", flush=True)
    h = run_handoff_protocol(n_test_speakers=6, utt_per_speaker=3, seed=0)
    payload["handoff"] = h
    for k, v in h.items():
        print(f"  {k:24s}  {v:.4f}", flush=True)

    # Honest gates: mean synth must be similar; coupling must rank aligned higher.
    mix_ok = abs(h["mean_synth_aligned"] - h["mean_synth_unaligned"]) < 0.12
    timing_ok = h["auc_coupling_score"] > h["auc_mean_synth"] + 0.05
    pair_ok = h["pair_win_rate"] >= 0.6 and h["n_pairs"] >= 8.0
    print("Gates", flush=True)
    print(f"  [{'PASS' if mix_ok else 'FAIL'}] |mean_synth aligned-unaligned| < 0.12 (matched mix)", flush=True)
    print(f"  [{'PASS' if timing_ok else 'FAIL'}] coupling AUC > mean-synth AUC + 0.05 on audio", flush=True)
    print(f"  [{'PASS' if pair_ok else 'FAIL'}] pair win rate >= 0.6 on audio", flush=True)
    syn_ok = syn["auc"] >= 0.9 and syn["pair_win_rate"] >= 0.9
    print(f"  [{'PASS' if syn_ok else 'FAIL'}] SAPC statistic ranks clean point processes (AUC>=0.9)", flush=True)
    aci_ok = abs(aci["aci_coverage"] - aci["aci_target"]) <= 0.08
    print(f"  [{'PASS' if aci_ok else 'FAIL'}] ACI empirical coverage within 0.08 of target", flush=True)
    payload["gates"] = {
        "matched_mix": mix_ok,
        "audio_coupling_beats_mean": timing_ok,
        "audio_pair_win": pair_ok,
        "synthetic_sapc": syn_ok,
        "aci_coverage": aci_ok,
        "audio_handoff_claim": bool(mix_ok and timing_ok and pair_ok),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}", flush=True)
    if not syn_ok or not aci_ok:
        raise SystemExit(1)
    if not payload["gates"]["audio_handoff_claim"]:
        print("Audio timing claim NOT supported. Synthetic SAPC and ACI stand; do not cite handoff AUC.")
    else:
        print("Audio handoff timing gates passed.")


if __name__ == "__main__":
    main()
