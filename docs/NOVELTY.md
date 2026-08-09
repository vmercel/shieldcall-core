# Novelty — Evidence-Based (Not Marketing)

**Read this with:** `PRIOR_ART.md` (landscape), `RESEARCH.md` (claim→code→test), `PATENT_PATHWAY.md` (filing links).

## Rule

A feature is **novel for our purposes** only if:

1. It is **implemented** in this repo, and  
2. It has a **measurable ablation or experiment**, and  
3. It is **differentiated** from cited prior art in `PRIOR_ART.md`.

Anything else is labeled **roadmap**, not novelty.

## Differentiated technical contributions

### 1. Joint streaming telephony dual-stream stack (system claim)

**Not claimed as novel alone:** keyword fraud detection; generic deepfake classification; weighted audio+text average (exists in multimodal vishing papers).

**Claimed combination:** continuous **STRF acoustic** + **SDTG linguistic path** + **CSCF co-activation/regimes** under **TCT** channel conditions, with **PMA coverage-debt** feedback.

- Code: `pipeline.py`, full package  
- Evidence: `scripts/run_benchmark.py`, `scripts/run_ablation.py`  
- Prior art gap: late-fusion multimodal vishing lacks causal co-activation, regimes, debt loop, telephony residual stack as one system

### 2. STRF under explicit Channel Twin evaluation doctrine

- Code: `acoustic/residual.py`, `audio/channel.py`  
- Evidence: acoustic EER/AUC across clean / narrowband / harsh_voip  
- Prior art: channel augmentation exists; we **standardize TCT as generative twin** for stress + optional live injection and **grid residual** cues tuned for bandlimited vocoder artifacts  
- Limit: not yet ASVspoof-trained SSL — do not claim SOTA EER

### 3. SDTG — scam script as streaming stage machine

- Code: `linguistic/discourse.py`  
- Evidence: path_score scam ≫ benign; progression depth on scripts  
- Prior art: dialogue acts exist in NLP; rare as **real-time vishing path likelihood** fused with acoustics

### 4. CSCF — co-activation, disagreement regimes, joint trajectory

- Code: `fusion/engine.py`  
- Evidence: ablation CSCF vs naive sum / unimodal  
- Prior art: weighted sum multimodal [Kim et al. 2025 class]; we add **regime taxonomy** and **temporal co-activation**

### 5. PMA + coverage-debt control loop + challenge-response

- Code: `acoustic/scorer.py`, `adaptation/coverage.py`, `adaptation/hooks.py`  
- Evidence: gap reduction after k-shot; debt snapshot  
- Prior art: few-shot prototypical anti-spoofing papers exist; we couple debt **into fused call risk operations**

### 6. CSR + CTE on fused call risk

- Code: `fusion/conformal.py`, `fusion/explain.py`  
- Evidence: unit tests; demo intervals/counterfactuals  
- Prior art: conformal prediction is general; application to **streaming dual-stream telephony risk with abstention** is the system-level claim

## What we explicitly do **not** claim

| Overclaim | Reality |
|-----------|---------|
| Best ASVspoof EER | No public ASVspoof training run in-repo yet |
| Invented deep learning | PMA/STRF are lightweight; SSL optional future |
| Invented keyword fraud detection | Patterns are tactical baseline |
| Invented weighted fusion | Naive sum is our baseline to beat |
| Legal/regulatory certification | Research system only |

## Reproduce evidence

```bash
pytest -q
python scripts/run_benchmark.py
python scripts/run_ablation.py   # falsification gates
```

If a gate fails, **fix the algorithm** before updating patent language.
