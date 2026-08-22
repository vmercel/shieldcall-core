# ShieldCall Core

Streaming detector for **vishing language** and **vocoded speech** on telephone-bandwidth audio.

This is a research prototype (v0.6), not a certified product and not a state-of-the-art ASVspoof system. The measured claims, and the things we explicitly do not claim, are in `docs/NOVELTY.md` and the paper in `paper/`.

**v0.6 runtime:** the library is a sidecar, not a media hairpin. `shieldcall.runtime.SidecarRuntime` isolates one session per call, sheds under concurrency limits (fail-open on the telephone path), and trips an ASR circuit breaker. Capacity is `calls/core = (hop_ms / ms_frame) * util`, measured by `python scripts/run_capacity.py`. Design: `docs/SYSTEM_DESIGN.md`. ADR: `docs/ADR-003-sidecar-runtime.md`. This is not a carrier deployment.

**v0.5 agent:** the pipeline is a sensor. `shieldcall.agent.DefenseAgent` holds a belief over five call hypotheses and chooses monitor / challenge / warn / escalate / adapt / abstain by information gain minus interruption cost. It never sees raw audio. It is not an LLM. Demo: `python scripts/run_agent_demo.py`. ADR: `docs/ADR-002-belief-state-defense-agent.md`.

**Journal manuscript** (Information Fusion / TASLP target; do not submit to Computers \& Security): `paper/main.pdf`  
**Reproduce confirmatory tables:** `python scripts/run_upgrade_experiments.py`

## What it does

On a shared 8 kHz timeline it:

1. Simulates telephone-channel distortion (bandlimit, µ-law, packet loss) when asked.
2. Scores residual / harmonic artifacts on speech frames, with optional prototype memory fit on labeled clips.
3. Scores transcript fragments with a keyword layer plus a scam-script stage tracker.
4. Fuses the two streams with disagreement rules (human voice + scam script vs vocoded voice + mild language).

There is no production ASR in this repository. Linguistic experiments inject text. Acoustic experiments use Mini LibriSpeech plus vocoders, not ASVspoof, unless you point `SHIELDCALL_ASVSPOOF_ROOT` at a licensed copy.

## Results (what is actually measured)

From `docs/results/ablation_latest.txt` / `scripts/run_paper_experiments.py`:

| Test | Result |
|------|--------|
| Held-out paraphrased scam scripts | Keyword AUC **0.42**; keywords+stages AUC **0.88** |
| Pulse-formant vocoder vs LibriSpeech (speaker-disjoint, 8 kHz / narrowband) | EER **0.00**, AUC **1.00** (easy condition) |
| LPC vocoder vs LibriSpeech after bandlimiting | AUC **0.49** (at chance; negative result) |
| Operational fusion, threat = scam **or** vocoded | CSCF disagreement recall **1.00** vs naive sum **0.30**; CSCF safe-cell FPR **0.40** vs naive **0.00** |
| ASVspoof | **Not run** |
| SAPC on synthetic point processes | AUC **1.00** (formula check, not speech) |
| SAPC aligned vs unaligned vocoded splices | AUC **0.47** — **not supported** |
| ACI coverage vs target 0.90 | **0.885** (synthetic labeled stream) |

Sine-wave unit tests still exist. They are not evidence.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_speech.py          # Mini LibriSpeech into ./data
python scripts/run_paper_experiments.py    # official numbers
pytest -q
python -m shieldcall.demo.stream_demo
```

Configs: `configs/default.yaml`, `configs/telephony_harsh.yaml`, `configs/research_sensitive.yaml`.

## Layout

```
shieldcall/     engine
scripts/        download, paper experiments, demo
tests/          unit tests (including synthetic sanity checks)
docs/           architecture, system design (scale / reliability / cost), novelty, results
paper/          arXiv draft (LaTeX + PDF)
data/           Mini LibriSpeech (gitignored; download script)
```

## Proposed endeavor (plain language)

Build and evaluate a **U.S.-deployable telephony detector** that flags (a) scam-script progression on call transcripts and (b) vocoded/synthetic speech after telephone distortion, with disagreement-aware fusion so a human vishing call is not suppressed by a “human-sounding” voice score. Current evidence is a reproducible prototype and a preprint, not a production deployment.

## What this is not

- Not ASVspoof SOTA.
- Not a filed patent.
- Not a carrier integration, user study, or legal/compliance certification.
- Not twelve months of public iteration (the git log is short; that is a fact).

## License

Proprietary / all rights reserved for the time being.
