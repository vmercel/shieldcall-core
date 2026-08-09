# Prior Art Landscape & Differentiation

**Purpose:** Ground ShieldCall Core novelty in published literature and commercial practice so claims are *researchable and examinable*, not marketing slogans.

**Status of this document:** Technical prior-art memo for inventors and counsel. Not a formal freedom-to-operate (FTO) opinion. A registered patent attorney should run paid searches (USPTO, Espacenet, Google Patents, Derwent) before filing.

---

## 1. Problem statement (what the field already knows)

Telephone fraud increasingly combines:

1. **Synthetic / cloned voice** (TTS, VC, neural vocoders), and/or  
2. **Social-engineering language** (vishing scripts: authority → urgency → harvest → payment → secrecy).

These are usually treated as **separate products**:

| Domain | Typical systems | Gap |
|--------|-----------------|-----|
| Acoustic anti-spoofing | ASVspoof challenge systems, contact-center liveness (e.g. Pindrop-class), SSL front-ends (wav2vec/AASIST) | Strong on lab/LA conditions; **degrades under telephony codecs, packet loss, PLC** [1][2][3] |
| Linguistic fraud / vishing NLP | Keyword rules, call-center speech analytics, text classifiers | **Ignore voice authenticity** and **script trajectory** as a structured process |
| Multimodal “audio+text” phishing | Weighted score average of text model + audio model [4] | Late fusion only; no **causal co-activation window**, no **disagreement regimes**, no **coverage-debt control loop** |
| Few-shot spoof adaptation | Prototypical nets, GP adaptation for new generators [5][6] | Acoustic-only; not fused with **live fraud-intent stream** or **challenge-response enrollment** |

ShieldCall targets the **joint, streaming, telephony-conditioned** problem with an explicit **adaptation / coverage-debt** control loop.

---

## 2. Acoustic deepfake under telephone channels

### What exists

- **ASVspoof** series (2015–2024/5): primary benchmark for logical-access spoofing; recent editions stress **codec and adversarial filtering** degradation [3].  
- **Channel-effect studies:** codecs, device IR, telephone transmission damage spoof cues; data augmentation (packet loss, codec) helps but is not a full channel twin for *evaluation doctrine* [1][2].  
- **Commercial telephony deepfake detection:** e.g. systems trained with telephone simulation (packet loss, jitter, bandwidth, codecs) for live-call reliability [7].  
- **Features / models:** LFCC, CQCC, residual/phase cues, SSL embeddings + AASIST-style backends [3].

### What is still hard (research consensus)

1. Cues that survive **narrowband + µ-law + PLC** are not the same as clean-lab cues.  
2. **New vocoders** create coverage holes until the next retrain.  
3. **Streaming / low-latency** constraints (frame budgets ≪ 20 ms) limit heavy offline models on-device.

### ShieldCall acoustic position (implemented)

| Component | Implementation | Relation to art |
|-----------|----------------|-----------------|
| **TCT** | Stochastic G.711-like µ-law, bandlimit, SNR, packet-loss + PLC concealment | Builds on channel-augmentation literature; makes channel a **first-class generative twin** for both stress-test and optional online injection |
| **STRF** | Harmonic+noise residual fingerprint: residual energy ratio, flatness, kurtosis, modulation, phase irregularity, **grid-artifact score** | Related to residual/HNR and spectral artifact work; **grid peakiness under bandlimit** is optimized for vocoder upsampling + telephony |
| **PMA** | Online class-conditional prototypes, RMS-Mahalanobis score, FIFO family memory | Related to few-shot prototypical anti-spoofing [5]; **streaming + coverage-gap signal** feeds fusion and debt tracker |

**Honest limit:** In-repo acoustic models are **physics-informed + prototype**, not wav2vec-scale SSL. Leaderboard SOTA EER on ASVspoof is *not* claimed. The **architecture and telephony evaluation doctrine** are the research claim until large corpora are trained.

---

## 3. Linguistic fraud / vishing

### What exists

- Keyword and TF-IDF style scam detectors.  
- Neural text classifiers on phishing corpora.  
- Multimodal voice-phishing systems fusing text and audio via **weighted average** [4].  
- Discourse / dialogue act work in general NLP (not typically shipping as **streaming scam-script HMM** on live calls).

### ShieldCall linguistic position (implemented)

| Component | Implementation | Differentiation |
|-----------|----------------|-----------------|
| Pattern groups | Weighted multi-family regex (authority, payment, secrecy, harvest, …) | Tactical layer (not novel alone) |
| **SDTG** | Streaming stage machine over GREETING→…→THREAT with log-transition path score | **Strategic layer:** fraud as *script progression*, not bag-of-words |
| Blend | `fraud_prob = blend(patterns, path_score) × escalation` | Trajectory + structure |

**Honest limit:** SDTG emissions are pattern-driven; a learned neural dialogue model on labeled call graphs would strengthen the claim further.

---

## 4. Multimodal fusion

### What exists

- Weighted sum / max aggregation of modality scores [4].  
- Audio-visual deepfake fusion (different problem: face+voice video) [8].  
- Generic late fusion / stacking in ML.

### What is rare or absent in telephony fraud products

1. **Cross-modal co-activation** inside a causal time window (both streams elevated *together*).  
2. **Disagreement regimes** as first-class states:  
   - high fraud + low synth → **social engineering (human voice)**  
   - high synth + low fraud → **deepfake probe / spoof**  
   - both high → **dual threat**  
3. **Joint score trajectory** (rising fused risk is itself a feature).  
4. **Conformal intervals + abstention** on the fused risk (distribution-free under shift) [9].  
5. **Counterfactual interventions** on fusion inputs (not LLM post-hoc text).

### ShieldCall fusion position (implemented as CSCF + CSR + CTE)

See `shieldcall/fusion/engine.py`, `conformal.py`, `explain.py`.

---

## 5. Adaptation & coverage debt

### What exists

- Few-shot synthetic speech detection via prototypical / attention prototypes [5].  
- GP-based adaptation for deepfake detectors [6].  
- Generic OOD / domain-shift literature.

### ShieldCall position (implemented)

| Concept | Implementation |
|---------|----------------|
| **Coverage gap** | Distance to *both* human and synthetic manifolds (OOD when far from both) |
| **Debt index** | Rolling high-gap rate + family coverage factor |
| **Recovery metric** | Gap before/after *k*-shot family exposure (`evaluate_adaptation_recovery`) |
| **Challenge-response** | Interactive liveness challenge that can *label* adaptation examples |

**Claim scope:** Not “we invented few-shot learning.” Claim is the **closed-loop telephony dual-stream system** that *measures and reduces* coverage debt while fused fraud-intent continues to run.

---

## 6. Patent-relevant claim themes (for counsel)

Draft themes that map to **specific technical combinations** (Alice-friendly: improve a technical process, not “detect fraud on a computer”):

1. **Method** of streaming dual-stream risk estimation on telephone audio comprising residual fingerprinting under a telephony channel model, discourse-stage path scoring on ASR fragments, and co-activation fusion with regime classification.  
2. **System** maintaining online prototype memory and emitting a coverage-debt signal that gates conformal abstention and triggers challenge-response enrollment.  
3. **Method** of generating counterfactual risk explanations by minimal interventions on acoustic vs linguistic fusion inputs.  
4. **Apparatus** executing the above at frame hop ≤ 20 ms on edge/mobile.

**Avoid bare claims:** “using AI to detect scam calls,” “fusing audio and text,” “keyword matching.”

---

## 7. References (entry points for formal search)

1. Cohen et al., data augmentation in voice anti-spoofing (packet loss / channel effects).  
2. Zhang et al., empirical study on channel effects for synthetic voice spoofing CMs — arXiv:2104.01320.  
3. ASVspoof challenge series / ASVspoof 5 reports (codec & adversarial filtering gaps).  
4. Kim et al., multimodal voice phishing (text+audio weighted fusion) — *Appl. Sci.* 2025.  
5. Garg et al., few-shot synthetic speech detection via self-attentive prototypical networks — arXiv:2508.13320.  
6. Glazer et al., few-shot deepfake detection adaptation with Gaussian processes — Interspeech 2025.  
7. Commercial telephony deepfake pipelines (telephone channel simulation in training).  
8. Multimodal A/V deepfake score fusion literature (video domain).  
9. Conformal prediction (Vovk et al.; Angelopoulos & Bates tutorials) applied here to streaming risk.

**Search queries for counsel / Espacenet / Google Patents:**

```
voice spoofing OR deepfake detection telephone OR codec OR "packet loss"
vishing OR "voice phishing" multimodal fusion acoustic linguistic
prototypical network spoofing few-shot
"synthetic speech" adaptation online learning call center
```

---

## 8. What must still be done for *scientific* (not only product) novelty

| Milestone | Why |
|-----------|-----|
| Train/eval on **public** ASVspoof + **telephony-degraded** copies via TCT | External comparability |
| Collect / license **labeled vishing transcripts** with stage annotations | Validate SDTG vs keyword-only ablation |
| Ablation table: acoustic-only / linguistic-only / sum-fusion / CSCF | Prove co-activation & regimes add value |
| Held-out **unseen vocoder families** for PMA recovery curves | Prove coverage-debt loop |
| Optional SSL backbone behind same `score_frame` API | Competitiveness without breaking claims |

These are tracked as engineering milestones in `docs/RESEARCH.md` and `scripts/run_ablation.py`.
