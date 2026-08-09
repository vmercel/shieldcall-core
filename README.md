# ShieldCall Core

Research-grade **streaming dual-stream detection engine** for real-time joint **linguistic fraud-intent** and **acoustic synthesis-artifact** analysis under **telephone conditions**.

This repository is the technical core for a differentiated national-interest research and engineering agenda. It is deliberately separate from the consumer Expo application so detection science can be developed, evaluated, and claimed independently.

## Research stack (v0.2) — implemented, measured, prior-art grounded

| Acronym | Subsystem | Role |
|---------|-----------|------|
| **TCT** | Telephony Channel Twin | Stochastic phone-path simulation (µ-law, bandlimit, PLC, SNR) |
| **STRF** | Spectral-Temporal Residual Fingerprinting | Vocoder residual cues that survive narrowband |
| **SDTG** | Scam Discourse Trajectory Graph | Streaming scam-script stage machine |
| **CSCF** | Cross-Stream Causal Fusion | Co-activation, regimes, joint trajectory |
| **PMA** | Prototype Memory Adaptation | Few-shot online acoustic updates |
| **CSR** | Conformal Streaming Risk | Uncertainty bands + abstention |
| **CTE** | Counterfactual Threat Explanations | Minimal interventions that drop risk |

**Evidence, not slogans:**

- [docs/PRIOR_ART.md](docs/PRIOR_ART.md) — landscape vs this system  
- [docs/RESEARCH.md](docs/RESEARCH.md) — claim → code → experiment map  
- [docs/NOVELTY.md](docs/NOVELTY.md) — what is / is not claimed  
- [docs/PATENT_PATHWAY.md](docs/PATENT_PATHWAY.md) — **official USPTO/WIPO filing links**  
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system design  

```bash
pytest -q
python scripts/run_benchmark.py
python scripts/run_ablation.py   # falsification gates for novelty claims
```

## Architecture

```
Incoming audio stream (8 kHz / 16 kHz mono)
        |
        v
+-------------------+
| Telephony         |  TCT + VAD + framing
| Preprocessor      |
+-------------------+
        |
        +--------------------+
        |                    |
        v                    v
+---------------+    +------------------+
| Acoustic      |    | Linguistic       |
| STRF + PMA    |    | patterns + SDTG  |
+---------------+    +------------------+
        |                    |
        +---------+----------+
                  |
                  v
        +-------------------+
        | CSCF Fusion       |  trajectory, co-activation,
        | CSR + CTE         |  conformal bands, counterfactuals
        +-------------------+
                  |
                  v
        Risk · tier · regime · coverage-debt signal
```

## Getting started

```bash
cd shieldcall-core
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# or: pip install -r requirements.txt && pip install -e .

python -m shieldcall.demo.stream_demo
python scripts/run_benchmark.py
pytest -q
```

### Config profiles

- `configs/default.yaml` — balanced telephony
- `configs/telephony_harsh.yaml` — VoIP / degraded PSTN stress
- `configs/research_sensitive.yaml` — high-sensitivity research

```python
from shieldcall.config import load_config, pipeline_config_from_dict
from shieldcall import ShieldCallPipeline

cfg = pipeline_config_from_dict(load_config("configs/default.yaml"))
pipe = ShieldCallPipeline(config=cfg)
```

### Programmatic API

```python
import numpy as np
from shieldcall import ShieldCallPipeline, PipelineConfig
from shieldcall.linguistic.asr_bridge import ScheduledTranscriptASR

asr = ScheduledTranscriptASR([(0.5, "Verify your SSN and buy gift cards immediately")])
pipe = ShieldCallPipeline(asr=asr)
audio = np.random.randn(8000).astype(np.float32) * 0.01  # replace with real audio
for ev in pipe.stream(audio, 8000):
    if ev.risk:
        print(ev.risk.tier, ev.risk.risk_score, ev.risk.explanation)
```

## Package layout

```
shieldcall/
  audio/         TCT, VAD, preprocessor
  acoustic/      STRF features, residual, prototype scorer
  linguistic/    patterns, SDTG, ASR bridge
  fusion/        CSCF, conformal CSR, CTE explanations
  adaptation/    PMA buffers, challenge-response, coverage debt
  eval/          metrics + channel-aware harness
  demo/          streaming demo
  pipeline.py    unified streaming entrypoint
configs/         YAML operating profiles
tests/           pytest suite
docs/            architecture & novelty
```

## Status

**v0.2.0** — Full research-oriented implementation: interfaces, dual-stream science, channel twin eval, adaptation loop, conformal risk, counterfactuals, tests, and configs.

Still ahead for production claims: large telephony-trained STRF/neural weights, production ASR wiring, and held-out multi-synthesizer corpora.

## Relation to the consumer app

The Expo/React Native application can later call this core (local service, ONNX Runtime, or FFI) for threat scores. Ghost Mode, AI Dialer, and other product features remain outside this research core.

## License

Proprietary / All rights reserved for the time being.
