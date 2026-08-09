# Research Implementation Status

This document ties **claims to code to measurable experiments**. If a claim has no test or ablation, it is marked as **aspirational**.

![Claim to code to experiment map](figures/claim_code_map.png)

## 1. Claim to code map

| Research claim | Primary code | Experiment / test | Status |
|----------------|--------------|-------------------|--------|
| Telephony channel twin (TCT) | `audio/channel.py` | `tests/test_audio.py`, channel profiles in benchmark | **Implemented** |
| Multi-feature streaming VAD | `audio/vad.py` | `tests/test_audio.py` | **Implemented** |
| STRF residual fingerprint | `acoustic/residual.py`, `features.py` | `tests/test_acoustic.py`, acoustic EER in harness | **Implemented** (proxy audio; not ASVspoof-trained) |
| Prototype memory and adapt | `acoustic/scorer.py` PMA | `evaluate_adaptation_recovery`, tests | **Implemented** |
| Scam discourse trajectory (SDTG) | `linguistic/discourse.py` | `tests/test_linguistic.py` | **Implemented** |
| Pattern and discourse blend | `linguistic/scorer.py` | scam vs benign tests | **Implemented** |
| CSCF co-activation and regimes | `fusion/engine.py` | `tests/test_fusion.py`, disagreement ablations | **Implemented** |
| Conformal streaming risk | `fusion/conformal.py` | unit tests and demo bands | **Implemented** |
| Counterfactual explanations | `fusion/explain.py` | `tests/test_fusion.py` | **Implemented** |
| Coverage-debt tracker | `adaptation/coverage.py` | pipeline adapt test, recovery metric | **Implemented** |
| Challenge-response protocol | `adaptation/hooks.py` | `tests/test_pipeline.py` | **Implemented** |
| Full dual-stream pipeline | `pipeline.py` | end-to-end test, demo | **Implemented** |
| SSL / large neural countermeasure | (none) | (none) | **Not implemented** (API reserved) |
| Production ASR | `asr_bridge.py` (passthrough/scheduled) | (none) | **Interface only** |
| Public ASVspoof numbers | (none) | (none) | **Not yet** |

## 2. Evaluation doctrine (what "works" means here)

We do **not** claim "world's best deepfake detector." We claim a **system** that:

1. Runs **streaming** dual-stream scoring under **TCT** conditions.
2. Separates **proxy** human-like versus vocoder-like acoustics via STRF and PMA.
3. Elevates risk on **scam script progression**, not only keywords.
4. Fuses with **regimes and co-activation**, not only weighted average.
5. **Measures** coverage debt and reduces gap with few-shot PMA.
6. Emits **calibrated bands** and **counterfactuals**.

### How to reproduce

```bash
source .venv/bin/activate
pytest -q
python scripts/run_benchmark.py
python scripts/run_ablation.py
python scripts/render_diagrams.py
```

### Latest local synthetic results (re-run to refresh)

Run `python scripts/run_benchmark.py` and `python scripts/run_ablation.py`. Saved snapshot: `docs/results/ablation_latest.txt`.

| Condition | Metric | Interpretation |
|-----------|--------|----------------|
| acoustic_* under TCT | EER, AUC, latency | Channel robustness of STRF and PMA on proxy waveforms |
| fused_narrowband | EER, AUC | Dual-stream on scam transcripts plus audio |
| disagreement regimes | CSCF vs naive sum | Social engineering and deepfake probe |
| adaptation | gap before, after, reduction | PMA coverage recovery |

**Caveat:** Proxy vocoder-like tones are not WaveNet or HiFi-GAN speech. They stress residual and grid cues. Real speech training is required for publication-grade acoustic claims.

## 3. Required ablations (scientific honesty)

`scripts/run_ablation.py` reports:

1. **Linguistic-only** risk on scam versus benign transcripts.
2. **Acoustic-only** under clean versus harsh VoIP.
3. **Naive sum fusion** versus **CSCF** (co-activation, regimes, trajectory).
4. **Pattern-only** versus **pattern plus SDTG** fraud scores.
5. **PMA gap** before and after k shots.
6. **Disagreement regimes** (human voice with scam script; synthetic voice with mild language).

If CSCF does not beat naive sum on the dual-stream disagreement suite where it should, the fusion novelty claim is **not supported** by current data. Fix algorithms before patent language freezes.

## 4. Publication and patent evidence package checklist

- [ ] Timestamped invention notebook and git tags
- [ ] Prior-art search log (this folder plus counsel search)
- [ ] Ablation tables committed under `docs/results/`
- [ ] System diagrams as rendered PNG under `docs/figures/`
- [ ] Pseudocode matching claims
- [ ] Provisional specification draft
- [ ] Inventor declarations

## 5. Next research sprints (priority order)

1. Import ASVspoof (or similar) plus TCT degrade; report real EER curves.
2. Label 50 to 200 call scripts with SDTG stages; supervised transition learning.
3. Neural embedding head optional behind PMA (same interface).
4. Human study: operators prefer CTE explanations versus raw scores.
5. Latency profile on mobile (ONNX export).
