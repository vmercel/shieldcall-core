# Novelty & Differentiation

## The hard problems we claim

1. **Telephony generalization of deepfake cues**  
   Lab-clean detectors collapse under G.711 + packet-loss concealment. STRF + TCT make the channel part of the model, not an afterthought.

2. **Joint low-latency linguistic × acoustic fusion**  
   Vishing is rarely “fake voice only” or “bad words only.” CSCF models co-activation and disagreement regimes that single-stream products cannot express.

3. **Coverage debt for new synthesizers**  
   The attack surface moves weekly. PMA + CoverageDebtTracker turn “we’ll retrain next quarter” into a measurable control loop with few-shot recovery time.

4. **Calibrated decisions under shift**  
   CSR abstains when conformal width is too large — safer than a confident wrong tier on an unseen vocoder.

## What we are *not* claiming (yet)

- State-of-the-art EER on ASVspoof / WaveFake leaderboards (no large trained net in-box).
- Production ASR quality (pluggable bridge only).
- Legal admissibility of scores without human review.

## Evaluation doctrine

Every release should report:

- Acoustic EER/AUC under `{clean, narrowband, g711_ulaw, harsh_voip}`
- Fused pipeline metrics on dual-stream scenarios
- Mean per-frame latency
- Adaptation gap reduction after *k*-shot family exposure
- Coverage-debt index on held-out OOD families

```bash
python scripts/run_benchmark.py
pytest -q
```

## Path to stronger models

Interfaces are stable:

- Replace `PrototypeMemory.score` with a neural embedding + distance head (ONNX).
- Keep `AcousticDeepfakeScorer.score_frame` signature.
- Continue feeding PMA with human-review and challenge-response labels.

The fusion, discourse, conformal, and channel layers remain valuable regardless of the acoustic backbone.
