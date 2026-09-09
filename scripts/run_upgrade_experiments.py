#!/usr/bin/env python3
"""Confirmatory upgrade suite (lexicon-locked). Writes docs/results/upgrade_experiments.json."""

from __future__ import annotations

import json
import platform
import socket
import time
from pathlib import Path

import _repo  # noqa: F401  — repo root on sys.path when run as python scripts/...

from shieldcall.agent.simulator import compare_policies
from shieldcall.audio.channel import CodecProfile
from shieldcall.eval.asvspoof import available as asvspoof_available
from shieldcall.eval.corpora.independent_scripts import independent_corpus_hash
from shieldcall.eval.protocols import (
    acoustic_protocol,
    fusion_ablation_from_pairs,
    linguistic_ablation_protocol,
    operational_fusion_protocol,
)
from shieldcall.eval.speech_data import speech_available
from shieldcall.linguistic.discourse import LEXICON_LOCK
from shieldcall.linguistic.lock import current_lexicon_lock


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


def _gate(name: str, ok: bool, detail: str) -> dict:
    print(f"  GATE {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    return {"name": name, "ok": bool(ok), "detail": detail}


def main() -> None:
    t0 = time.perf_counter()
    out_dir = Path("docs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    gates = []
    payload: dict = {
        "lexicon_lock_stage": LEXICON_LOCK,
        "lexicon_lock": current_lexicon_lock(),
        "corpus_hash": independent_corpus_hash(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "asvspoof": asvspoof_available(),
        "speech": speech_available(),
    }

    print("1. Linguistic ablation (independent, locked lexicon)", flush=True)
    ling = linguistic_ablation_protocol(confirmatory=True, asr_wer=0.0, seed=0)
    ling_noise = linguistic_ablation_protocol(confirmatory=True, asr_wer=0.25, seed=0)
    # keep *_asr keys for plot compatibility but notes say text_noise
    payload["linguistic_independent"] = {
        k: {"auc": v.auc, "eer": v.eer_estimate, "n": v.n_samples, "extras": v.extras, "notes": v.notes}
        for k, v in {**ling, **ling_noise}.items()
    }
    for k, v in {**ling, **ling_noise}.items():
        print(
            f"  {k:28s} AUC={v.auc:.3f} trap={v.extras.get('trap_mean', 0):.3f} n={v.n_samples}",
            flush=True,
        )
    n_ling = next(iter(ling.values())).n_samples
    wide = ling["ling_wide_lexicon"].auc
    sdtg = ling["ling_sdtg"].auc
    gates.append(_gate("linguistic_n", n_ling >= 40, f"n={n_ling}"))
    gates.append(_gate("sdtg_not_above_wide", sdtg <= wide + 0.02, f"sdtg={sdtg:.3f} wide={wide:.3f}"))

    print("2. Agent class-conditional simulator vs threshold", flush=True)
    sim = compare_policies(n_per_class=20, n_steps=5, seed=0)
    payload["agent_sim"] = {
        pol: {h: _ser(m) for h, m in rows.items()} for pol, rows in sim.items()
    }
    print(
        "  SE false_challenge agent="
        f"{sim['agent']['social_engineering'].false_challenge:.2f} "
        f"handoff challenge_rate agent={sim['agent']['handoff'].challenge_rate:.2f} "
        f"thr={sim['threshold']['handoff'].challenge_rate:.2f}",
        flush=True,
    )

    if speech_available():
        try:
            import soundfile as _sf  # noqa: F401
        except ImportError:
            raise SystemExit(
                "Mini LibriSpeech is present but `soundfile` is not installed.\n"
                "FLAC clips cannot be decoded. From the repo root run:\n"
                "  pip install -r requirements.txt\n"
                "then re-run: python scripts/run_upgrade_experiments.py"
            )
        print("3. Acoustic LPC headline (n_test speakers=8, 2 utt) + Hybrid-H", flush=True)
        ac, scorer, train, test, test_spoof = acoustic_protocol(
            n_train_speakers=8,
            n_test_speakers=8,
            utt_per_speaker=2,
            vocoders=("lpc",),
            profiles=(CodecProfile.CLEAN, CodecProfile.NARROWBAND),
            seed=0,
            include_hybrid=True,
        )
        payload["acoustic"] = {
            k: {"auc": v.auc, "eer": v.eer_estimate, "n": v.n_samples, "notes": v.notes}
            for k, v in ac.items()
        }
        for k, v in ac.items():
            print(f"  {k:42s} AUC={v.auc:.3f} EER={v.eer_estimate:.3f} n={v.n_samples}", flush=True)
        lpc_n = ac.get("ac_lpc_narrowband")
        if lpc_n:
            gates.append(_gate("acoustic_n", lpc_n.n_samples >= 20, f"n={lpc_n.n_samples}"))
        neural_keys = [k for k in ac if "neural_quant" in k or "pulse_formant" in k]
        gates.append(_gate("no_easy_vocoder_in_headline", not neural_keys, f"keys={neural_keys}"))

        print("4. Fusion ablation on operational cells (independent scripts)", flush=True)
        key = next(iter(test_spoof))
        _, pairs, _ = operational_fusion_protocol(scorer, test, test_spoof[key], seed=0)
        fuse = fusion_ablation_from_pairs(pairs)
        payload["fusion_ablation"] = {
            k: {"auc": v.auc, "eer": v.eer_estimate, "n": v.n_samples, "extras": v.extras}
            for k, v in fuse.items()
        }
        for k, v in fuse.items():
            print(
                f"  {k:24s} n={v.n_samples} disc@0.5={v.extras.get('disc_recall@0.5', 0):.2f} "
                f"safeFPR@0.5={v.extras.get('safe_fpr@0.5', 0):.2f} "
                f"tpr@fpr0.05={v.extras.get('tpr@fpr0.05', 0):.2f}",
                flush=True,
            )
        n_fuse = next(iter(fuse.values())).n_samples
        gates.append(_gate("fusion_n", n_fuse >= 40, f"n={n_fuse}"))

        print("5. Closed-loop agent on pipeline scores", flush=True)
        from shieldcall.eval.agent_closed_loop import compare_closed_loop

        clo = compare_closed_loop(
            test,
            test_spoof[key],
            n_per_class=4,
            acoustic_scorer=scorer,
        )
        payload["agent_closed_loop"] = clo
        se = clo["agent"]["social_engineering"]
        hd = clo["agent"]["handoff"]
        print(
            f"  SE missed={se['missed_harvest']:.2f} SE challenge={se['false_challenge']:.2f} "
            f"handoff challenge_rate={hd['challenge_rate']:.2f}",
            flush=True,
        )
    else:
        print("Mini LibriSpeech missing; skip acoustic/fusion/closed-loop", flush=True)
        gates.append(_gate("speech_present", False, "run scripts/download_speech.py"))

    payload["gates"] = gates
    payload["seconds"] = time.perf_counter() - t0
    path = out_dir / "upgrade_experiments.json"
    path.write_text(json.dumps(payload, indent=2))
    failed = [g for g in gates if not g["ok"]]
    print(f"Wrote {path} in {payload['seconds']:.1f}s; gates failed={len(failed)}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
