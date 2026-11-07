# ShieldCall Core

![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square) ![Status](https://img.shields.io/badge/Status-research%20prototype-yellow?style=flat-square) ![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square)

Streaming detector for **vishing language** and **vocoded speech** on telephone-bandwidth audio.

## Who uses what

**People use the phone app** (Live Protect in the ShieldCall client). That listens on the handset microphone while the other phone is on speaker. It does not hang up.

**This repo is the detector.** The HTTP sidecar is for the phone on the LAN and for engine work. The Chrome page at `/` is a developer lab, not the product.

```bash
pip install -r requirements-serve.txt
python scripts/run_sidecar.py
# phone: EXPO_PUBLIC_SHIELDCALL_URL=http://<lan-ip>:8765
# engine lab only: http://127.0.0.1:8765
```

1. Consent, then listen on the phone (or, for engine work, on the Chrome lab).
2. Linguistic meter: vishing language. Acoustic meter: vocoded / synthetic. Fusion is recommend-only.
3. Optional A/B in the lab: inject dentist reminder (`ind_b01`) then grandparent-bond (`ind_s01`).

Recording recipe for the engine lab: [`docs/DEMO.md`](docs/DEMO.md).

This is a research prototype (v0.7 MVP), not a certified product and not a state-of-the-art ASVspoof system. The measured claims, and the things we explicitly do not claim, are in `docs/NOVELTY.md` and the paper in `paper/`.

**v0.6 runtime:** the library is a sidecar, not a media hairpin. `shieldcall.runtime.SidecarRuntime` isolates one session per call, sheds under concurrency limits (fail-open on the telephone path), and trips an ASR circuit breaker. Capacity is `calls/core = (hop_ms / ms_frame) * util`, measured by `python scripts/run_capacity.py`. Design: `docs/SYSTEM_DESIGN.md`. ADR: `docs/ADR-003-sidecar-runtime.md`. This is not a carrier deployment.

**v0.5 agent:** the pipeline is a sensor. `shieldcall.agent.DefenseAgent` holds a belief over five call hypotheses and chooses monitor / challenge / warn / escalate / adapt / abstain by information gain minus interruption cost. It never sees raw audio. It is not an LLM. Demo: `python scripts/run_agent_demo.py`. ADR: `docs/ADR-002-belief-state-defense-agent.md`.

**Manuscript:** `paper/main.pdf`  
**Reproduce confirmatory tables:** `python scripts/run_upgrade_experiments.py`  
Author held-out linguistic numbers from `run_paper_experiments.py` are **sanity only**.

## What it does

On a shared 8 kHz timeline it:

1. Simulates telephone-channel distortion (bandlimit, µ-law, packet loss) when asked.
2. Scores residual / harmonic artifacts on speech frames, with optional prototype memory fit on labeled clips.
3. Scores transcript fragments with a keyword layer plus a scam-script stage tracker.
4. Fuses the two streams with disagreement rules (human voice + scam script vs vocoded voice + mild language).

There is no production ASR in this repository. Linguistic experiments inject text. Early acoustic experiments used Mini LibriSpeech plus vocoders; the Phase 1-2 benchmark below uses ASVspoof 2019 LA. (Optional loader: `SHIELDCALL_ASVSPOOF_ROOT` at a licensed copy.)

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
| ASVspoof 2019 LA (subset, locked splits) | **Run** (Phase 1-2, 2026-11-07). Reference LFCC-GMM baseline: EER 28-34% on unseen attacks, worse under noise; the repo's own acoustic scorer is near chance (EER 41-47%). See [`docs/results/BENCHMARK_PHASE1_2.md`](docs/results/BENCHMARK_PHASE1_2.md). |
| SAPC audio splices | **Not supported** |
| ACI vs frozen quantile | Frozen still wins on the reported synthetic stream |
| Agent | Simulator + closed-loop pipeline scores; likelihoods heuristic |

Sine-wave unit tests still exist. They are not evidence.

## Benchmark (Phase 1 data foundation, Phase 2 baselines)

Locked, reproducible, and deliberately modest. Data card: [`data/DATA_CARD.md`](data/DATA_CARD.md). Protocol: [`docs/lab/ANNOTATION_PROTOCOL.md`](docs/lab/ANNOTATION_PROTOCOL.md). Lock file with hashes: [`docs/results/phase1_lock.json`](docs/results/phase1_lock.json). Full tables: [`docs/results/BENCHMARK_PHASE1_2.md`](docs/results/BENCHMARK_PHASE1_2.md).

* **Data**: ASVspoof 2019 LA subset (ODC-By; 67 held-out speakers; 13 attack systems not used for training), 14 channel conditions including real G.711, AMR-NB, Opus codecs via FFmpeg; 1,600 LLM-generated scam/benign dialogues (Apache-2.0).
* **Acoustic** (EER, held-out speakers and attacks): LFCC-GMM trained on clean audio 28.0% clean, 33.9% G.711, 36.8% at 20 dB noise, 43.4% at 0 dB; trained with telephone augmentation 30.4% G.711, 33.2% at 10 dB. The repository's STRF/prototype scorer: 41-47% (near chance) on a subset.
* **Language** (synthetic dialogues, English): repo scorer EER 9.5%, rising to 10.8% at 30% simulated ASR word error; a TF-IDF classifier reaches 0% because the generator leaves stylistic artifacts, so these numbers say little about real calls.
* **Not covered**: real recordings, real scam transcripts, languages other than English, post-2019 voice cloning, AMR-WB/EVS/G.729.

Reproduce: see the bottom of `docs/lab/ANNOTATION_PROTOCOL.md`. Needs `pip install av pandas` in addition to `requirements.txt`.

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

Licensed under the [Apache License 2.0](LICENSE). This is a research prototype and is provided without warranty.

## Feedback and support

If you work on telephone fraud, speech anti-spoofing, or consumer protection and find this useful, please [star the repository](https://github.com/vmercel/shieldcall-core) and open an issue with feedback, test results, or questions. Independent review is welcome. The companion phone client is [ShieldCallAI](https://github.com/vmercel/ShieldCallAI).
