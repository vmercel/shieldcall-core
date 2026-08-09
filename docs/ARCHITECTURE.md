# ShieldCall Core Architecture

## Mission

Real-time joint detection of **voice synthesis** and **linguistic fraud intent** on telephone audio — under the channel conditions that actually exist (narrowband, G.711, packet loss), with **measurable adaptation** when new synthesizers appear.

## Why this does not exist elsewhere

| Capability | Typical commercial stack | ShieldCall Core |
|---|---|---|
| Acoustic liveness | Strong (Pindrop-class) on contact-center audio | STRF residual fingerprints designed to *survive* telephony |
| Fraud language | Keyword / topic models offline | Streaming **discourse trajectory graph** (script structure) |
| Fusion | Alert rules or late score average | **Cross-Stream Causal Fusion** with co-activation & regimes |
| New synthesizers | Wait for retraining | **Prototype Memory + coverage-debt loop** (few-shot) |
| Uncertainty | Fixed thresholds | **Conformal streaming risk** with abstention |
| Explainability | Generic text | **Counterfactual** interventions on fusion inputs |
| Evaluation | Clean or proprietary sets | **Channel Twin** first-class in every benchmark |

## Data flow

```
mic / RTP / file
      │
      ▼
 TelephonyPreprocessor ── optional Telephony Channel Twin (TCT)
      │ frames + VAD
      ├──────────────────────────────┐
      ▼                              ▼
 AcousticDeepfakeScorer        ASRBridge → LinguisticFraudScorer
  • STRF residual FP             • pattern groups (tactical)
  • Prototype Memory (PMA)       • Scam Discourse Graph (strategic)
      │                              │
      └──────────┬───────────────────┘
                 ▼
         FusionEngine (CSCF)
           • trajectory
           • co-activation
           • regime logic
           • conformal bands (CSR)
           • counterfactuals (CTE)
                 │
                 ▼
     FusedRisk + CoverageDebtTracker
```

## Novel subsystems (research claims)

1. **TCT — Telephony Channel Twin**  
   Stochastic generative model of the phone path (µ-law, bandlimit, PLC, SNR). Every acoustic claim is stress-tested through TCT.

2. **STRF — Spectral-Temporal Residual Fingerprinting**  
   Harmonic-plus-noise residual statistics targeting neural vocoder artifacts that remain after narrowband distortion.

3. **SDTG — Scam Discourse Trajectory Graph**  
   HMM-style stage machine over scam scripts; path score captures *structure*, not only keywords.

4. **CSCF — Cross-Stream Causal Fusion**  
   Super-additive co-activation, disagreement regimes (social engineering vs deepfake probe), joint trajectory.

5. **PMA + Coverage Debt**  
   Online Mahalanobis prototypes; OOD gap is a first-class metric; few-shot recovery is measured.

6. **CSR — Conformal Streaming Risk**  
   Distribution-free intervals and abstention for high-stakes decisions.

7. **CTE — Counterfactual Threat Explanations**  
   Exact minimal interventions over fusion inputs for audit and UI.

## Package map

```
shieldcall/
  audio/        preprocessor, VAD, channel twin
  acoustic/     STRF features, residual, prototype scorer
  linguistic/   patterns, discourse graph, ASR bridge
  fusion/       CSCF engine, conformal, explain
  adaptation/   buffers, challenge-response, coverage debt
  eval/         metrics, harness, synthetic benchmark
  demo/         streaming demo
  pipeline.py   single entrypoint
  config.py     YAML profiles
```

## Latency budget (target)

| Stage | Budget |
|---|---|
| Frame + VAD | < 1 ms |
| STRF + features | < 5 ms |
| Acoustic score | < 2 ms |
| Linguistic update | < 1 ms |
| Fusion + conformal | < 1 ms |
| **Total / frame** | **≪ 20 ms** (real-time at 10 ms hop) |

## Version

0.2.0 — full research skeleton with runnable science, tests, and channel-aware eval. Neural weight files and large telephony corpora are the next training milestone.
