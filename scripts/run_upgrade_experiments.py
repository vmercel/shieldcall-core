#!/usr/bin/env python3
"""Confirmatory upgrade suite (lexicon-locked). Writes docs/results/upgrade_experiments.json."""

from __future__ import annotations

import json
import platform
import socket
import time
from pathlib import Path

import numpy as np

from shieldcall.agent.simulator import compare_policies
from shieldcall.audio.channel import CodecProfile
from shieldcall.eval.asvspoof import available as asvspoof_available
from shieldcall.eval.protocols import (
    acoustic_protocol,
    fusion_ablation_from_pairs,
    linguistic_ablation_protocol,
    operational_fusion_protocol,
)
from shieldcall.eval.speech_data import speech_available
from shieldcall.linguistic.discourse import LEXICON_LOCK


def _ser(m):
    return {
        "hypothesis": m.hypothesis,
        "n": m.n,
        "missed_harvest": m.missed_harvest,
        "false_challenge": m.false_challenge,
        "false_warn": m.false_warn,
        "mean_cost": m.mean_cost,
        "challenge_rate": m.challenge_rate,
    }


def main() -> None:
    t0 = time.perf_counter()
    out_dir = Path("docs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "lexicon_lock": LEXICON_LOCK,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "asvspoof": asvspoof_available(),
        "speech": speech_available(),
    }

    print("1. Linguistic ablation (independent, locked lexicon)", flush=True)
    ling = linguistic_ablation_protocol(confirmatory=True, asr_wer=0.0, seed=0)
    ling_asr = linguistic_ablation_protocol(confirmatory=True, asr_wer=0.25, seed=0)
    payload["linguistic_independent"] = {
        k: {"auc": v.auc, "eer": v.eer_estimate, "n": v.n_samples, "extras": v.extras, "notes": v.notes}
        for k, v in {**ling, **ling_asr}.items()
    }
    for k, v in {**ling, **ling_asr}.items():
        print(f"  {k:28s} AUC={v.auc:.3f} trap={v.extras.get('trap_mean', 0):.3f}")

    print("2. Agent simulator vs threshold", flush=True)
    sim = compare_policies(n_per_class=20, n_steps=5, seed=0)
    payload["agent_sim"] = {
        pol: {h: _ser(m) for h, m in rows.items()} for pol, rows in sim.items()
    }
    print(
        "  SE missed_harvest agent="
        f"{sim['agent']['social_engineering'].missed_harvest:.2f} "
        f"thr={sim['threshold']['social_engineering'].missed_harvest:.2f} "
        f"SE false_challenge agent={sim['agent']['social_engineering'].false_challenge:.2f}"
    )

    if speech_available():
        print("3. Acoustic LPC + neural_quant (pulse-formant not headline)", flush=True)
        ac, scorer, train, test, test_spoof = acoustic_protocol(
            n_train_speakers=3,
            n_test_speakers=3,
            utt_per_speaker=1,
            vocoders=("lpc", "neural_quant"),
            profiles=(CodecProfile.CLEAN, CodecProfile.NARROWBAND),
            seed=0,
        )
        payload["acoustic"] = {
            k: {"auc": v.auc, "eer": v.eer_estimate, "n": v.n_samples, "notes": v.notes}
            for k, v in ac.items()
        }
        for k, v in ac.items():
            print(f"  {k:42s} AUC={v.auc:.3f} EER={v.eer_estimate:.3f} n={v.n_samples}")

        print("4. Fusion ablation on operational cells", flush=True)
        key = "neural_quant" if "neural_quant" in test_spoof else next(iter(test_spoof))
        _, pairs, _ = operational_fusion_protocol(scorer, test, test_spoof[key], seed=0)
        fuse = fusion_ablation_from_pairs(pairs)
        payload["fusion_ablation"] = {
            k: {"auc": v.auc, "eer": v.eer_estimate, "n": v.n_samples, "extras": v.extras}
            for k, v in fuse.items()
        }
        for k, v in fuse.items():
            print(
                f"  {k:24s} AUC={v.auc:.3f} disc={v.extras.get('disc_recall@0.5', 0):.2f} "
                f"safeFPR={v.extras.get('safe_fpr@0.5', 0):.2f}"
            )
    else:
        print("Mini LibriSpeech missing; skip acoustic/fusion", flush=True)

    payload["seconds"] = time.perf_counter() - t0
    path = out_dir / "upgrade_experiments.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path} in {payload['seconds']:.1f}s")


if __name__ == "__main__":
    main()
