# Lab notebook (timestamped)

Protocol hashes and experiment locks. Append-only. Do not rewrite history.

## 2026-08-22 — claim freeze (TNCD upgrade start)

- Checklist: `CRITIQUES/2026-08-22-shieldcall-eb2-niw-q1-upgrade-checklist.md`
- Locked lexicon file: `shieldcall/linguistic/discourse.py` (`STAGE_EMISSIONS`)
- Lexicon lock id: `LEXICON_LOCK` constant in that module (sha256 of canonical JSON)
- Frozen hyperparameters: `configs/paper.yaml`
- Contamination log: `docs/lab/CONTAMINATION.md`
- Current confirmatory tables: `docs/results/paper_experiments.json`, `docs/results/handoff_experiment.json`
- Speech corpus: Mini LibriSpeech `dev-clean-2` (OpenSLR 31), local `data/LibriSpeech/`
- ASVspoof: **not present**. Loader is `shieldcall/eval/asvspoof.py`. Set `SHIELDCALL_ASVSPOOF_ROOT` when licensed.
- Decision: author-written `heldout_scripts()` remain a **sanity** split. Confirmatory linguistic numbers use `independent_scripts` (second-pass corpus, lexicon frozen first).
- Pulse-formant remains a **unit/easy** condition. `neural_quant` STFT quantization was also easy (AUC 1.0, *n*=6) — treat it like pulse-formant, not as Encodec. LPC remains the hard classical control.
- 2026-08-22 confirmatory JSON: `docs/results/upgrade_experiments.json`
  - independent wide-lexicon AUC 0.839 vs narrow 0.547; SDTG 0.820 (does not beat wide).
  - agent sim: SE missed_harvest 0, false_challenge 0.

## 2026-08-22 — standard-uplift pass (checklist Part V)

- Combined lexicon lock: `shieldcall/linguistic/lock.py`.
- Independent set: detector-aware benign tells removed; extra tropes added; still same-lab (not a hired second writer).
- LTM rename: `LinearTrajectoryModel` (no wide-bag feature).
- Acoustic headline vocoder: LPC only; Hybrid-H logistic on residual embeddings.
- TCT Opus/G.729/neural flagged as caricatures.
- Closed-loop agent: `eval/agent_closed_loop.py`.
- Fusion extras: TPR at FPR 0.05 / 0.50.
- Gates in `run_upgrade_experiments.py` (linguistic n, acoustic n, no easy vocoder in headline, fusion n).
- README/RESEARCH no longer lead with contaminated 0.88.
- Confirmatory run 2026-08-22 (`upgrade_experiments.json`, 533 s, all gates PASS):
  - ling n=48 corpus `7879918a4fe08c5d`: narrow AUC 0.596 trap 0.24; wide 0.836 trap 0.10; SDTG 0.817 trap 0.33; LTM 0.733 trap 0.10.
  - LPC residual n=32: clean AUC 0.773; narrowband 0.586.
  - Hybrid-H n=32: clean 0.922 / NB 0.910 (64-D, small n — exploratory).
  - Fusion n=40: CSCF disc@0.5=0.70 safeFPR=0.50 tpr@fpr0.05=0.23; calibrated-OR disc=1.00 but safeFPR=1.00.
  - Closed-loop n=4/class: SE challenge_rate=1.0 (policy does not survive real scores yet).

## How to hash a result JSON

```bash
shasum -a 256 docs/results/*.json
```
