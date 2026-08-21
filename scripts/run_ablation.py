#!/usr/bin/env python3
"""Ablations for novelty claims.

Linguistic tests always run (author-written scripts).
Acoustic and fusion tests require Mini LibriSpeech and refuse sine-wave
proxies as evidence for claim language.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shieldcall.eval.protocols import linguistic_protocol
from shieldcall.eval.speech_data import speech_available


def main() -> None:
    print("ShieldCall Core  -  Ablations")
    print("=" * 72)
    print("\n1. Linguistic / stage tracker (author-written scripts)")
    ling = linguistic_protocol()
    for k, v in ling.items():
        print(f"  {k:40s}  EER={v.eer_estimate:.4f}  AUC={v.auc:.4f}  n={v.n_samples}  {v.notes}")

    held_sdtg = ling["ling_heldout_patterns_sdtg"].auc
    held_pat = ling["ling_heldout_patterns"].auc
    traps = ling.get("ling_heldout_trap_mean")
    trap_ok = traps is None or traps.eer_estimate < 0.45

    checks = [
        ("held-out patterns+SDTG AUC >= patterns-only", held_sdtg + 1e-9 >= held_pat),
        ("held-out isolated-keyword benign mean fraud < 0.45", trap_ok),
    ]

    if speech_available():
        print("\n2. Speech-based gates  ->  scripts/run_paper_experiments.py")
        print("  (not duplicated here; run that script for acoustic/fusion numbers)")
    else:
        print("\n2. Acoustic/fusion SKIPPED (no Mini LibriSpeech).")
        print("  Sine-wave proxies are not evidence. python scripts/download_speech.py")

    print("\n" + "=" * 72)
    print("Gates:")
    failed = 0
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failed += 1
    if failed:
        raise SystemExit(1)
    print("Linguistic gates passed.")
    if not speech_available():
        print("Acoustic/fusion claims remain untested on real speech.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
