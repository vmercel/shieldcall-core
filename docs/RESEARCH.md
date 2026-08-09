# Research Implementation Status

This document ties **claims → code → measurable experiments**. If a claim has no test or ablation, it is marked as **aspirational**.

## 1. Claim → code map

| Research claim | Primary code | Experiment / test | Status |
|----------------|--------------|-------------------|--------|
| Telephony channel twin (TCT) | `audio/channel.py` | `tests/test_audio.py`, channel profiles in benchmark | **Implemented** |
| Multi-feature streaming VAD | `audio/vad.py` | `tests/test_audio.py` | **Implemented** |
| STRF residual fingerprint | `acoustic/residual.py`, `features.py` | `tests/test_acoustic.py`, acoustic EER in harness | **Implemented** (proxy audio; not ASVspoof-trained) |
| Prototype memory + adapt | `acoustic/scorer.py` PMA | `evaluate_adaptation_recovery`, tests | **Implemented** |
| Scam discourse trajectory (SDTG) | `linguistic/discourse.py` | `tests/test_linguistic.py` | **Implemented** |
| Pattern + discourse blend | `linguistic/scorer.py` | scam vs benign tests | **Implemented** |
| CSCF co-activation + regimes | `fusion/engine.py` | `tests/test_fusion.py` | **Implemented** |
| Conformal streaming risk | `fusion/conformal.py` | unit tests + demo bands | **Implemented** |
| Counterfactual explanations | `fusion/explain.py` | `tests/test_fusion.py` | **Implemented** |
| Coverage-debt tracker | `adaptation/coverage.py` | pipeline adapt test, recovery metric | **Implemented** |
| Challenge-response protocol | `adaptation/hooks.py` | `tests/test_pipeline.py` | **Implemented** |
| Full dual-stream pipeline | `pipeline.py` | e2e test, demo | **Implemented** |
| SSL / large neural CM | — | — | **Not implemented** (API reserved) |
| Production ASR | `asr_bridge.py` (passthrough/scheduled) | — | **Interface only** |
| Public ASVspoof numbers | — | — | **Not yet** |

## 2. Evaluation doctrine (what “works” means here)

We do **not** claim “world’s best deepfake detector.” We claim a **system** that:

1. Runs **streaming** dual-stream scoring under **TCT** conditions.  
2. Separates **proxy** human-like vs vocoder-like acoustics via STRF+PMA.  
3. Elevates risk on **scam script progression**, not only keywords.  
4. Fuses with **regimes + co-activation**, not only weighted average.  
5. **Measures** coverage debt and reduces gap with few-shot PMA.  
6. Emits **calibrated bands** and **counterfactuals**.

### How to reproduce

```bash
source .venv/bin/activate
pytest -q
python scripts/run_benchmark.py
python scripts/run_ablation.py
```

### Latest local synthetic results (re-run to refresh)

Run `python scripts/run_benchmark.py`. Example structure:

| Condition | Metric | Interpretation |
|-----------|--------|----------------|
| acoustic_* under TCT | EER, AUC, latency | Channel robustness of STRF+PMA on **proxy** waveforms |
| fused_narrowband | EER, AUC | Dual-stream on scam transcripts + audio |
| adaptation | gap_before, gap_after, reduction | PMA coverage recovery |

**Caveat:** Proxy vocoder-like tones are **not** WaveNet/HiFi-GAN speech. They stress residual/grid cues. Real speech training is required for publication-grade acoustic claims.

## 3. Required ablations (scientific honesty)

`scripts/run_ablation.py` reports:

1. **Linguistic-only** risk on scam vs benign transcripts.  
2. **Acoustic-only** under clean vs harsh_voip.  
3. **Naive sum fusion** vs **CSCF** (co-activation + regimes + trajectory).  
4. **Pattern-only** vs **pattern+SDTG** fraud scores.  
5. **PMA gap** before/after *k* shots.

If CSCF does not beat naive sum on the dual-stream suite, the fusion novelty claim is **not supported** by current data—fix algorithms before patent language freezes.

## 4. Publication / patent evidence package checklist

- [ ] Timestamped invention notebook / git tags  
- [ ] Prior-art search log (this folder + counsel search)  
- [ ] Ablation tables committed under `docs/results/`  
- [ ] System diagrams (ARCHITECTURE.md)  
- [ ] Pseudocode matching claims  
- [ ] Provisional specification draft  
- [ ] Inventor declarations  

## 5. Next research sprints (priority order)

1. **Import ASVspoof (or similar) + TCT degrade** → report real EER curves.  
2. **Label 50–200 call scripts** with SDTG stages → supervised transition learning.  
3. **Neural embedding head** optional behind PMA (same interface).  
4. **Human study**: operators prefer CTE explanations vs raw scores.  
5. **Latency profile** on mobile (ONNX export).
