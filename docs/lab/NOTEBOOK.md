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

## How to hash a result JSON

```bash
shasum -a 256 docs/results/*.json
```
