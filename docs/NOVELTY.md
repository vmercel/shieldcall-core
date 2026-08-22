# Novelty: target claims vs current claims

A claim is **in scope** only if it is implemented, numbered in `docs/results/`,
and differentiated from the cited literature. The upgrade checklist’s
*target* claims are the endeavor. Current claims are baselines until
independent data says otherwise.

Locked lexicon: `LEXICON_LOCK` in `shieldcall/linguistic/discourse.py`.
Frozen hyperparameters: `configs/paper.yaml`.
Contamination log: `docs/lab/CONTAMINATION.md`.

## Target (TNCD) vs current (v0.6)

| Claim | Current (measured) | Target | Status |
|-------|--------------------|--------|--------|
| Stage tracker recovers paraphrases keywords miss | Author held-out AUC 0.88 vs 0.42 — **contaminated** | Independent set, locked lexicon, wide-keyword control | **Confirmatory:** independent wide-lexicon AUC ~0.84 vs narrow ~0.55; SDTG 0.82 **does not beat** wide bag. Discourse-HMM novelty **dropped** (kill criterion). |
| OR-label fusion vs naive sum | Complementary recall 1.00 vs 0.30, safe FPR 0.40 vs 0.00, *n*=40 constructed cells | DET + floors vs blend vs calibrated-OR on joint trajectories | Floors/OR ablations implemented; still constructed cells until more audio |
| Pulse-formant after bandlimit | EER 0, *n*=20, easy | Demoted to unit test | Headline vocoders: LPC, neural_quant |
| Residual vs LPC+NB | AUC 0.49 | Hybrid-H vs published neural baseline on TCT-2 | Residual still at chance until Hybrid-H/AASIST table exists |
| SAPC harvest-timed handoff | Synthetic AUC 1.00; audio 0.47, *n*=6 | Gold *n*≥100 streaming-VC | Protocol exists; audio claim **not supported** |
| ACI coverage | 0.885 vs frozen 0.931 | Real fusion streams; demote if frozen wins | Frozen still wins on the reported stream |
| Agent warn-on-SE | 3 scripted traces | Simulator + closed-loop sensor scores vs threshold | Simulator table in `upgrade_experiments.json` |
| U.S.-deployable detector | Denied in README | CPaaS trial | **Not claimed** |

## Supported by current confirmatory data

1. **Wide locked lexicon vs narrow keywords on independent scripts** (not author held-out). See `docs/results/upgrade_experiments.json`.
2. **SDTG path prior does not beat the wide bag** on that set. The HMM is a baseline, not a contribution.
3. **Disagreement floors / calibrated-OR vs naive sum** on operational cells (FPR bill is the number to lead with).
4. **LPC after narrowband remains hard** for residual features.
5. **Agent simulator:** social-engineering traces prefer warn; challenge rate on SE is the wasted-nonce metric.

## Negative results (keep)

- LPC narrowband residual AUC ~0.49.
- SAPC audio AUC 0.47 (*n* small; not a powered gold construction yet).
- ACI does not beat a frozen quantile on the reported synthetic stream.
- Five-shot PMA on unseen LPC: EER 0.50 → 0.45.
- SDTG ≉ wide lexicon on independent text.

## Not claimed

| Phrase | Why |
|--------|-----|
| Best ASVspoof EER | Corpus not run |
| Causal fusion | Non-anticipative score rules |
| Conformal coverage guarantee | EMA / ACI measured, not guaranteed |
| HiFi-GAN / Encodec detection | neural_quant is a surrogate |
| Production ASR | Noise model + optional Whisper |
| Discourse HMM as novel | Kill criterion fired |
| U.S.-deployable | Not measured |

## Reproduce

```bash
pytest -q
python scripts/run_paper_experiments.py   # sanity / historical
python scripts/run_upgrade_experiments.py # confirmatory
```
