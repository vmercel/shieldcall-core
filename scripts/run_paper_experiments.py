#!/usr/bin/env python3
"""Official experiment suite for the paper. Refuses to run on sine-wave proxies."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from shieldcall.audio.channel import CodecProfile
from shieldcall.eval.protocols import (
    acoustic_protocol,
    adaptation_protocol,
    linguistic_protocol,
    operational_fusion_protocol,
)
from shieldcall.eval.speech_data import speech_available
from shieldcall.eval.asvspoof import available as asvspoof_available


def _row(name: str, eer: float, auc: float, extra: str = "") -> str:
    return f"  {name:42s}  EER={eer:.3f}  AUC={auc:.3f}  {extra}"


def main() -> None:
    t0 = time.perf_counter()
    out_dir = Path("docs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {"asvspoof": asvspoof_available()}

    print("1. Linguistic held-out scripts", flush=True)
    ling = linguistic_protocol()
    payload["linguistic"] = {
        k: {"eer": v.eer_estimate, "auc": v.auc, "n": v.n_samples, "notes": v.notes, "extras": v.extras}
        for k, v in ling.items()
    }
    for k, v in ling.items():
        print(_row(k, v.eer_estimate, v.auc, v.notes))

    if not speech_available():
        print("\nMini LibriSpeech missing. Run: python scripts/download_speech.py")
        payload["error"] = "no_speech"
        (out_dir / "paper_experiments.json").write_text(json.dumps(payload, indent=2))
        raise SystemExit(2)

    print("\n2. Acoustic: speaker-disjoint bona fide vs vocoded")
    ac, scorer, train, test, test_spoof = acoustic_protocol(
        n_train_speakers=5,
        n_test_speakers=5,
        utt_per_speaker=2,
        vocoders=("pulse_formant",),
        profiles=(CodecProfile.CLEAN, CodecProfile.NARROWBAND),
        seed=0,
    )
    payload["acoustic"] = {
        k: {"eer": v.eer_estimate, "auc": v.auc, "n": v.n_samples, "notes": v.notes}
        for k, v in ac.items()
    }
    for k, v in ac.items():
        print(_row(k, v.eer_estimate, v.auc, v.notes))

    print("\n3. Operational fusion (threat = scam OR vocoded)")
    spoof_key = "pulse_formant" if "pulse_formant" in test_spoof else next(iter(test_spoof))
    _, pairs, op = operational_fusion_protocol(
        scorer, test, test_spoof[spoof_key], seed=0
    )
    payload["operational"] = {
        k: {
            "eer": v.eer_estimate,
            "auc": v.auc,
            "n": v.n_samples,
            "extras": v.extras,
        }
        for k, v in op.items()
    }
    payload["operational_cells"] = [
        {"cell": p.cell, "y": p.y, "synth": p.synth, "fraud": p.fraud, "cscf": p.cscf, "naive": p.naive}
        for p in pairs
    ]
    for k, v in op.items():
        extra = " ".join(f"{kk}={vv:.3f}" for kk, vv in v.extras.items())
        print(_row(k, v.eer_estimate, v.auc, extra))

    print("\n4. Adaptation: pulse-formant memory, LPC k-shot", flush=True)
    adapt = adaptation_protocol(train, test, n_shots=5, seed=0)
    payload["adaptation"] = adapt
    print(
        f"  unseen LPC EER before={adapt['eer_before']:.3f} "
        f"after={adapt['eer_after']:.3f} reduction={adapt['eer_reduction']:.3f}",
        flush=True,
    )

    # Honest gates (no -0.05 fudge)
    held_sdtg = ling["ling_heldout_patterns_sdtg"].auc
    held_pat = ling["ling_heldout_patterns"].auc
    cscf_auc = op["op_cscf"].auc
    naive_auc = op["op_naive"].auc
    cscf_rec = op["op_cscf"].extras.get("disagreement_recall@0.5", 0.0)
    naive_rec = op["op_naive"].extras.get("disagreement_recall@0.5", 0.0)
    ac_pf = ac.get("ac_pulse_formant_clean") or ac.get("ac_pulse_formant_narrowband")
    ac_ok = ac_pf is not None and (
        (ac.get("ac_pulse_formant_clean") and ac["ac_pulse_formant_clean"].auc > 0.65)
        or (ac.get("ac_pulse_formant_narrowband") and ac["ac_pulse_formant_narrowband"].auc > 0.65)
    )

    checks = [
        ("held-out SDTG AUC >= patterns-only", held_sdtg + 1e-9 >= held_pat),
        (
            "operational CSCF disagreement recall >= naive sum",
            cscf_rec + 1e-9 >= naive_rec,
        ),
        (
            "pulse-formant acoustic AUC (clean or narrowband) > 0.65",
            ac_ok,
        ),
    ]
    print("\nGates")
    failed = 0
    payload["gates"] = []
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        payload["gates"].append({"name": name, "ok": bool(ok)})
        if not ok:
            failed += 1

    payload["seconds"] = time.perf_counter() - t0
    (out_dir / "paper_experiments.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_dir / 'paper_experiments.json'} in {payload['seconds']:.1f}s")
    if failed:
        raise SystemExit(1)
    print("All paper gates passed.")


if __name__ == "__main__":
    main()
