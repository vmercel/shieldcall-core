# Research status

Confirmatory tables: `docs/results/upgrade_experiments.json` from
`python scripts/run_upgrade_experiments.py`.
Author held-out linguistic numbers in `paper_experiments.json` are **sanity
only** (lexicon and paraphrases share an author).

## Claim → code → experiment

| Claim | Code | Experiment | Status |
|-------|------|------------|--------|
| Telephone channel simulation | `audio/channel.py` | TCT-2 caricature profiles + classical NB | Implemented; Opus/G.729/neural names are DSP caricatures |
| Residual frame features | `acoustic/residual.py` | LPC vs bona fide | LPC under bandlimit remains hard |
| Prototype memory | `acoustic/scorer.py` `fit()` | speaker-disjoint | Implemented |
| Hybrid-H logistic head | `acoustic/hybrid.py` | same protocol, `hy_*` rows | Linear head on residual embeddings; not AASIST |
| Wide locked lexicon vs narrow keywords | `discourse.wide_lexicon_score` | independent set | Wide bag beats narrow |
| Stage HMM vs wide bag | `linguistic/discourse.py` | independent set | **Kill:** SDTG does not beat wide bag |
| Linear trajectory model | `linguistic/ntm.py` `LinearTrajectoryModel` | train-fit, independent eval | Baseline; not neural |
| Disagreement fusion | `fusion/engine.py` | operational OR-label | Report disc-recall and TPR at FPR, not AUC 1.0 |
| Synthetic text noise | `linguistic/asr_noise.py` | independent + noise knob | **Not ASR** |
| Production ASR | `asr_bridge.py` | — | Interface + unused Whisper hook |
| ASVspoof numbers | `eval/asvspoof.py` | `SHIELDCALL_ASVSPOOF_ROOT` | Loader only |
| SAPC | `fusion/coupling.py` | LibriSpeech splices | Audio claim **not** supported |
| Agent (scripted / sim) | `agent/` | `compare_policies` | Class-conditional means |
| Agent closed-loop | `eval/agent_closed_loop.py` | pipeline scores | Implemented; small n |

## What “works” means here (confirmatory)

1. A **wide frozen lexicon** beats **narrow keywords** on the independent set.
2. The **HMM path prior does not beat** that wide bag (discourse novelty dropped).
3. Residual features remain weak on **LPC** after telephone-band filtering.
4. Fusion floors / calibrated-OR change complementary-cell recall vs a weighted sum, at a false-alarm cost. Lead with that tradeoff, not ranking AUC.

## Reproduce

```bash
source .venv/bin/activate
python scripts/download_speech.py
pytest -q
python scripts/run_upgrade_experiments.py
```

## Still required for a stronger scientific claim

- ASVspoof 5 (or 2019 LA) through TCT, **same table as AASIST or RawNet2**.
- A true second-writer or public transcript dump (current independent set is same-lab tropes without detector-aware benign tells).
- Real ASR-in-the-loop (Whisper or telephony ASR), not `synthetic_text_noise`.
- Neural vocoders / streaming VC, not STFT-quant surrogates.
- Gold SAPC *n*≥100.
- Named-instance capacity; CPaaS playback.

## Publication checklist

- [x] Ablation tables under `docs/results/`
- [x] Journal draft under `paper/` (not submitted)
- [x] Timestamped lab notebook (`docs/lab/NOTEBOOK.md`)
- [ ] ASVspoof run
- [ ] Venue template (elsarticle / IEEEtran)
- [ ] Provisional patent
- [ ] Independent users, pilots, or letters
