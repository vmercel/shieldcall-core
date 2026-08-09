# Prior Art Landscape and Differentiation

**Purpose:** Ground ShieldCall Core novelty in published literature and commercial practice so claims are researchable and examinable, not marketing slogans.

**Status of this document:** Technical prior-art memo for inventors and counsel. Not a formal freedom-to-operate (FTO) opinion. A registered patent attorney should run paid searches (USPTO, Espacenet, Google Patents, Derwent) before filing.

![Dual-stream architecture under review](figures/architecture_pipeline.png)

---

## 1. Problem statement (what the field already knows)

Telephone fraud increasingly combines:

1. **Synthetic or cloned voice** (TTS, voice conversion, neural vocoders), and/or
2. **Social-engineering language** (vishing scripts: authority, urgency, harvest, payment, secrecy).

These are usually treated as **separate products**:

| Domain | Typical systems | Gap |
|--------|-----------------|-----|
| Acoustic anti-spoofing | ASVspoof challenge systems, contact-center liveness, SSL front-ends (wav2vec/AASIST) | Strong on lab conditions; degrades under telephony codecs, packet loss, and PLC [1][2][3] |
| Linguistic fraud / vishing NLP | Keyword rules, call-center speech analytics, text classifiers | Ignore voice authenticity and script trajectory as a structured process |
| Multimodal audio and text phishing | Weighted score average of text model and audio model [4] | Late fusion only; no causal co-activation window, disagreement regimes, or coverage-debt control loop |
| Few-shot spoof adaptation | Prototypical nets, GP adaptation for new generators [5][6] | Acoustic-only; not fused with a live fraud-intent stream or challenge-response enrollment |

ShieldCall targets the **joint, streaming, telephony-conditioned** problem with an explicit **adaptation and coverage-debt** control loop.

---

## 2. Acoustic deepfake under telephone channels

### What exists

- **ASVspoof** series (2015 to 2024/5): primary benchmark for logical-access spoofing; recent editions stress codec and adversarial filtering degradation [3].
- **Channel-effect studies:** codecs, device impulse responses, and telephone transmission damage spoof cues; data augmentation (packet loss, codec) helps but is not always a full channel twin for evaluation doctrine [1][2].
- **Commercial telephony deepfake detection:** systems trained with telephone simulation for live-call reliability [7].
- **Features and models:** LFCC, CQCC, residual and phase cues, SSL embeddings with AASIST-style backends [3].

### What is still hard (research consensus)

1. Cues that survive narrowband plus mu-law plus PLC are not the same as clean-lab cues.
2. New vocoders create coverage holes until the next retrain.
3. Streaming and low-latency constraints limit heavy offline models on-device.

### ShieldCall acoustic position (implemented)

![TCT and STRF](figures/tct_strf.png)

| Component | Implementation | Relation to art |
|-----------|----------------|-----------------|
| **TCT** | Stochastic G.711-like mu-law, bandlimit, SNR, packet-loss with PLC concealment | Builds on channel-augmentation literature; makes channel a first-class generative twin for stress-test and optional online injection |
| **STRF** | Harmonic-plus-noise residual fingerprint: residual energy ratio, flatness, kurtosis, modulation, phase irregularity, grid-artifact score | Related to residual/HNR and spectral artifact work; grid peakiness under bandlimit is optimized for vocoder upsampling plus telephony |
| **PMA** | Online class-conditional prototypes, RMS-Mahalanobis score, FIFO family memory | Related to few-shot prototypical anti-spoofing [5]; streaming coverage-gap signal feeds fusion and debt tracker |

**Honest limit:** In-repo acoustic models are physics-informed plus prototype, not wav2vec-scale SSL. Leaderboard state-of-the-art EER on ASVspoof is not claimed. The architecture and telephony evaluation doctrine are the research claim until large corpora are trained.

---

## 3. Linguistic fraud and vishing

### What exists

- Keyword and TF-IDF style scam detectors.
- Neural text classifiers on phishing corpora.
- Multimodal voice-phishing systems fusing text and audio via weighted average [4].
- Discourse and dialogue-act work in general NLP (not typically shipping as a streaming scam-script stage machine on live calls).

### ShieldCall linguistic position (implemented)

![SDTG](figures/sdtg_stages.png)

| Component | Implementation | Differentiation |
|-----------|----------------|-----------------|
| Pattern groups | Weighted multi-family regex (authority, payment, secrecy, harvest, and related) | Tactical layer (not novel alone) |
| **SDTG** | Streaming stage machine over GREETING through THREAT with log-transition path score | Strategic layer: fraud as script progression, not bag-of-words |
| Blend | fraud probability blends patterns and path score, then applies escalation | Trajectory plus structure |

**Honest limit:** SDTG emissions are pattern-driven; a learned neural dialogue model on labeled call graphs would strengthen the claim further.

---

## 4. Multimodal fusion

### What exists

- Weighted sum or max aggregation of modality scores [4].
- Audio-visual deepfake fusion (different problem: face and voice video) [8].
- Generic late fusion and stacking in machine learning.

### What is rare or absent in telephony fraud products

1. Cross-modal co-activation inside a causal time window (both streams elevated together).
2. Disagreement regimes as first-class states:
 - high fraud and low synth: social engineering (human voice)
 - high synth and low fraud: deepfake probe or spoof
 - both high: dual threat
3. Joint score trajectory (rising fused risk is itself a feature).
4. Conformal intervals and abstention on the fused risk (distribution-free under shift) [9].
5. Counterfactual interventions on fusion inputs (not LLM post-hoc text).

### ShieldCall fusion position (implemented as CSCF, CSR, and CTE)

![CSCF regimes](figures/cscf_regimes.png)

See `shieldcall/fusion/engine.py`, `conformal.py`, and `explain.py`.

---

## 5. Adaptation and coverage debt

### What exists

- Few-shot synthetic speech detection via prototypical or attention prototypes [5].
- Gaussian-process adaptation for deepfake detectors [6].
- Generic out-of-distribution and domain-shift literature.

### ShieldCall position (implemented)

![Adaptation loop](figures/adaptation_loop.png)

| Concept | Implementation |
|---------|----------------|
| Coverage gap | Distance to both human and synthetic manifolds (out-of-distribution when far from both) |
| Debt index | Rolling high-gap rate plus family coverage factor |
| Recovery metric | Gap before and after k-shot family exposure (`evaluate_adaptation_recovery`) |
| Challenge-response | Interactive liveness challenge that can label adaptation examples |

**Claim scope:** Not "we invented few-shot learning." The claim is the closed-loop telephony dual-stream system that measures and reduces coverage debt while fused fraud-intent continues to run.

---

## 6. Patent-relevant claim themes (for counsel)

Draft themes that map to **specific technical combinations** (Alice-friendly: improve a technical process, not "detect fraud on a computer"):

1. **Method** of streaming dual-stream risk estimation on telephone audio comprising residual fingerprinting under a telephony channel model, discourse-stage path scoring on ASR fragments, and co-activation fusion with regime classification.
2. **System** maintaining online prototype memory and emitting a coverage-debt signal that gates conformal abstention and triggers challenge-response enrollment.
3. **Method** of generating counterfactual risk explanations by minimal interventions on acoustic versus linguistic fusion inputs.
4. **Apparatus** executing the above at frame hop of 20 ms or less on edge or mobile.

**Avoid bare claims:** "using AI to detect scam calls," "fusing audio and text," "keyword matching."

---

## 7. References (entry points for formal search)

1. Cohen et al., data augmentation in voice anti-spoofing (packet loss and channel effects).
2. Zhang et al., empirical study on channel effects for synthetic voice spoofing countermeasures. arXiv:2104.01320.
3. ASVspoof challenge series and ASVspoof 5 reports (codec and adversarial filtering gaps).
4. Kim et al., multimodal voice phishing (text and audio weighted fusion). Appl. Sci. 2025.
5. Garg et al., few-shot synthetic speech detection via self-attentive prototypical networks. arXiv:2508.13320.
6. Glazer et al., few-shot deepfake detection adaptation with Gaussian processes. Interspeech 2025.
7. Commercial telephony deepfake pipelines (telephone channel simulation in training).
8. Multimodal audio-visual deepfake score fusion literature (video domain).
9. Conformal prediction (Vovk et al.; Angelopoulos and Bates tutorials) applied here to streaming risk.

**Search queries for counsel, Espacenet, or Google Patents:**

```
voice spoofing OR deepfake detection telephone OR codec OR "packet loss"
vishing OR "voice phishing" multimodal fusion acoustic linguistic
prototypical network spoofing few-shot
"synthetic speech" adaptation online learning call center
```

---

## 8. What must still be done for scientific (not only product) novelty

| Milestone | Why |
|-----------|-----|
| Train and evaluate on public ASVspoof plus telephony-degraded copies via TCT | External comparability |
| Collect or license labeled vishing transcripts with stage annotations | Validate SDTG versus keyword-only ablation |
| Ablation table: acoustic-only / linguistic-only / sum-fusion / CSCF | Prove co-activation and regimes add value |
| Held-out unseen vocoder families for PMA recovery curves | Prove coverage-debt loop |
| Optional SSL backbone behind the same `score_frame` API | Competitiveness without breaking claims |

These are tracked as engineering milestones in `docs/RESEARCH.md` and `scripts/run_ablation.py`.
