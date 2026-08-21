# Research status

## Claim → code → experiment

| Claim | Code | Experiment | Status |
|-------|------|------------|--------|
| Telephone channel simulation | `audio/channel.py` | paper acoustic narrowband | Implemented |
| Residual frame features | `acoustic/residual.py` | pulse-formant vs LPC | Implemented; LPC fails under bandlimit |
| Prototype memory fit on speech | `acoustic/scorer.py` `fit()` | speaker-disjoint protocol | Implemented |
| Scam stage tracker | `linguistic/discourse.py` | held-out scripts | Implemented |
| Keyword + stage blend | `linguistic/scorer.py` | same | Implemented |
| Disagreement fusion | `fusion/engine.py` | operational protocol | Implemented; recall vs FPR tradeoff |
| Uncertainty band | `fusion/conformal.py` | unit tests | Heuristic, not CP |
| Counterfactuals by re-score | `fusion/explain.py` | unit tests | Implemented |
| Challenge+voice check | `adaptation/hooks.py` | unit tests | Transcript plus synth cap |
| Production ASR | `asr_bridge.py` | — | Interface only |
| ASVspoof numbers | `eval/asvspoof.py` | set `SHIELDCALL_ASVSPOOF_ROOT` | Loader only |
| Production change-point | `acoustic/changepoint.py` | unit tests | Implemented (CUSUM + mean-shift) |
| Stage-aligned coupling | `fusion/coupling.py` | synthetic AUC 1.0; LibriSpeech splices **fail** | Method yes; audio claim **no** |
| Gibbs–Candès ACI | `fusion/aci.py` | coverage 0.885 vs 0.90 | Implemented |

## What “works” means here

1. Stage tracker beats keywords on held-out paraphrases.
2. Pulse-formant vocoding of real speech is detected at 8 kHz after bandlimiting.
3. LPC vocoding is **not** detected after bandlimiting with this front end.
4. Fusion labeled by *threat* (scam or vocoded) raises disagreement recall vs naive sum and raises safe-cell FPR.

## Reproduce

```bash
source .venv/bin/activate
python scripts/download_speech.py
pytest -q
python scripts/run_paper_experiments.py
```

Snapshot: `docs/results/ablation_latest.txt`, `docs/results/paper_experiments.json`.

## Still required for a stronger scientific claim

- ASVspoof LA through the channel simulator, same table as AASIST or RawNet2.
- Labeled real call transcripts (not author-written English).
- Larger speaker set; report confidence intervals.
- Neural vocoders (HiFi-GAN / official TTS), not only pulse-formant and LPC.
- Human study of explanations.

## Publication checklist

- [x] Ablation tables under `docs/results/`
- [x] arXiv draft under `paper/`
- [ ] ASVspoof run
- [ ] Timestamped lab notebook beyond git
- [ ] Provisional patent (optional; do not file on unmeasured neural-TTS claims)
- [ ] Independent users, pilots, or letters
