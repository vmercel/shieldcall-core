# ShieldCall Core

Streaming detector for **vishing language** and **vocoded speech** on telephone-bandwidth audio.

## Live MVP (record this)

The current shippable slice is a **Detector Lab**, not a carrier product. It runs next to a consented live call, scores two streams, and never hangs up.

```bash
pip install -r requirements-serve.txt
python scripts/run_sidecar.py
# Chrome: http://127.0.0.1:8765
```

1. Check consent. Start a live session. Click **Listen on microphone**.
2. Put a phone on speaker next to the laptop, or speak both sides of a call in the room.
3. Watch **linguistic** (vishing language), **acoustic** (vocoded / synthetic), and **fused risk**. The action is recommend-only.
4. Optional A/B in the same take: inject dentist reminder (`ind_b01`) then grandparent-bond (`ind_s01`).

Recording recipe: [`docs/DEMO.md`](docs/DEMO.md).

**Recorded walkthrough:** add the clip URL here after capture (YouTube unlisted or a GitHub Release). Until then the live lab is the demo.

This is a research prototype (v0.7 MVP), not a certified product and not a state-of-the-art ASVspoof system. The measured claims, and the things we explicitly do not claim, are in `docs/NOVELTY.md` and the paper in `paper/`.

**v0.6 runtime:** the library is a sidecar, not a media hairpin. `shieldcall.runtime.SidecarRuntime` isolates one session per call, sheds under concurrency limits (fail-open on the telephone path), and trips an ASR circuit breaker. Capacity is `calls/core = (hop_ms / ms_frame) * util`, measured by `python scripts/run_capacity.py`. Design: `docs/SYSTEM_DESIGN.md`. ADR: `docs/ADR-003-sidecar-runtime.md`. This is not a carrier deployment.

**v0.5 agent:** the pipeline is a sensor. `shieldcall.agent.DefenseAgent` holds a belief over five call hypotheses and chooses monitor / challenge / warn / escalate / adapt / abstain by information gain minus interruption cost. It never sees raw audio. It is not an LLM. Demo: `python scripts/run_agent_demo.py`. ADR: `docs/ADR-002-belief-state-defense-agent.md`.

**Journal manuscript** (Information Fusion / TASLP target; do not submit to Computers \& Security): `paper/main.pdf`  
**Reproduce confirmatory tables:** `python scripts/run_upgrade_experiments.py`  
Author held-out linguistic numbers from `run_paper_experiments.py` are **sanity only**.

## What it does

On a shared 8 kHz timeline it:

1. Simulates telephone-channel distortion (bandlimit, µ-law, packet loss) when asked.
2. Scores residual / harmonic artifacts on speech frames, with optional prototype memory fit on labeled clips.
3. Scores transcript fragments with a keyword layer plus a scam-script stage tracker.
4. Fuses the two streams with disagreement rules (human voice + scam script vs vocoded voice + mild language).

There is no production ASR in this repository. Linguistic experiments inject text. Acoustic experiments use Mini LibriSpeech plus vocoders, not ASVspoof, unless you point `SHIELDCALL_ASVSPOOF_ROOT` at a licensed copy.

## Results (what is actually measured)

Confirmatory source: `docs/results/upgrade_experiments.json` (`run_upgrade_experiments.py`).
Re-run that script after pulling; the table below is the **claim set**, not a frozen scoreboard.

| Test | Status |
|------|--------|
| Independent scripts, locked lexicon | Wide frozen bag beats narrow keywords. **SDTG does not beat the wide bag** (discourse novelty dropped). |
| Author held-out keywords 0.42 vs stages 0.88 | **Sanity / contaminated** — do not cite as confirmatory. |
| LPC vs LibriSpeech after bandlimiting | Headline acoustic condition; residual features remain weak. |
| Pulse-formant / `neural_quant` | Easy unit conditions — **not headlines**. |
| Operational fusion (OR-label) | Report complementary-cell recall and TPR at FPR, not AUC 1.0. |
| ASVspoof | **Not run** |
| SAPC audio splices | **Not supported** |
| ACI vs frozen quantile | Frozen still wins on the reported synthetic stream |
| Agent | Simulator + closed-loop pipeline scores; likelihoods heuristic |

Sine-wave unit tests still exist. They are not evidence.

## Setup

```bash
# from the repo root; scripts bootstrap sys.path so `pip install -e .` is optional
pip install -r requirements.txt            # numpy/scipy/sklearn/pyyaml/soundfile
python scripts/download_speech.py          # Mini LibriSpeech into ./data
python scripts/run_upgrade_experiments.py  # confirmatory numbers
pip install -r requirements-serve.txt
python scripts/run_sidecar.py              # Detector Lab at http://127.0.0.1:8765
python scripts/run_paper_experiments.py    # historical / sanity
pytest -q
```

To install the package into the current env (conda `base` or a venv):

```bash
pip install -e ".[dev]"
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
