# Novelty: what is measured

Read with `RESEARCH.md` (claim → code → test) and `paper/main.pdf`.

A claim is **in scope** only if it is implemented, has a number in `docs/results/`, and is differentiated from the cited literature. Everything else is roadmap.

## Supported by current data

1. **Stage tracker vs keywords on paraphrases.** Held-out author-written scripts: keyword AUC 0.42, keywords+stages AUC 0.88. Code: `linguistic/discourse.py`, `eval/corpora/vishing_scripts.py`.

2. **Disagreement-aware fusion vs naive sum, on operational labels** (threat = scam language OR vocoded voice). Disagreement recall at 0.5: CSCF 1.00 vs naive 0.30. Cost: safe-cell FPR 0.40 vs 0.00. Code: `fusion/engine.py`.

3. **Pulse-formant vocoder vs bona fide LibriSpeech** at 8 kHz and after bandlimiting, speaker-disjoint, residual+prototype scorer. Easy condition, EER 0. Easy is allowed if labeled easy.

## Negative results (also in scope)

- **LPC vocoder after narrowband:** residual features at chance (AUC 0.49).
- **Five-shot PMA on unseen LPC:** EER 0.50 → 0.45. Not a coverage-debt success story.

## Not claimed

| Phrase we do not use | Why |
|----------------------|-----|
| Best ASVspoof EER | Corpus not run |
| Causal fusion / causal inference | Time-aligned score rules only |
| Conformal coverage guarantee | EMA residual band |
| Neural vocoder / HiFi-GAN detection | Pulse-formant and LPC only |
| Production ASR | Interface only |
| Keyword fraud detection as novel | Baseline |
| Weighted fusion as novel | Naive sum is the baseline |

## Reproduce

```bash
pytest -q
python scripts/download_speech.py
python scripts/run_paper_experiments.py
```
