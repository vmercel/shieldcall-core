# React call app × ShieldCall sidecar

Canonical checklist:  
`../../CRITIQUES/2026-08-22-react-call-sidecar-implementation-checklist.md`  
(or the copy in that folder if this repo is cloned alone — the sections below are the working copy).

This document is the **repo-local** implementation checklist for Experiment 0.

---

## Rules

- Sidecar, not hairpin. Channel twin off. Recommend-only UI.
- Agent never sees waveform/transcript. No nonce in traces.
- CHALLENGE is logged, not a blocking quiz, until benign false-challenge is measured.

## Run

```bash
# core
pip install -r requirements-serve.txt
python scripts/run_sidecar.py          # 0.0.0.0:8765
# Chrome: http://127.0.0.1:8765  (live mic + dual-stream meters)

# app (sibling repo GitHub/ShieldCall)
# Settings → Detector Lab → consent → Connect
# Type a live turn, or inject dentist then a scam paraphrase
```

Video recipe: `docs/DEMO.md`.

## API

| Method | Path |
|--------|------|
| GET | `/health` |
| POST | `/v1/calls` body `{call_id?}` |
| DELETE | `/v1/calls/{id}` |
| GET | `/v1/calls/{id}/trace` |
| POST | `/v1/calls/{id}/transcript` `{t, text}` |
| POST | `/v1/calls/{id}/audio` `{sr, pcm_s16le_b64}` |
| POST | `/v1/calls/{id}/inject` `{script_id}` or `{turns:[]}` |
| WS | `/v1/calls/{id}/stream` |

PCM: mono s16le 8 kHz, 200 ms chunks, base64 in JSON.

## Experiment 0 cells

| Cell | Fixture |
|------|---------|
| Safe | `ind_b01` |
| SE | `ind_s01` |

Save traces to `docs/results/lab_exp0.json` when five repeats exist.
