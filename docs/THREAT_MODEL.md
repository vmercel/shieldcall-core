# Threat model (STRIDE-light)

Intended use: research sidecar scoring a **fork** of call audio/text. RTP is not hairpinned. Actuation is fail-closed; media is fail-open.

## Assets

- Live call audio (forked copy)
- Transcript fragments
- Sufficient statistics (synth, fraud, SAPC, ACI, regime)
- Audit traces
- Prototype memory
- Liveness nonce

## Adversaries

| # | Actor | Goal | Path |
|---|-------|------|------|
| A1 | Human social engineer | Harvest credentials with a bona fide voice | Linguistic path only |
| A2 | Offline TTS / vocoder | Impersonate a trusted speaker for the whole call | Acoustic path |
| A3 | Streaming VC handoff | Rapport is human; harvest is converted | SAPC + dual stream |
| A4 | Keyword-aware paraphraser | Avoid frozen lexicon | Independent-set degradation |
| A5 | Score-query adaptive attacker | Stay in the safe cell given black-box scores | Fusion floors |
| A6 | Detector-DoS | Force shed so the call is undefended | Runtime capacity |

## Evaluated now

- A1: independent scripts + agent warn-not-challenge policy.
- A2: LPC and neural-quant vocoders through TCT-2 (pulse-formant is sanity only).
- A3: SAPC protocol (currently a powered-or-pilot negative on pulse-formant splices).
- A4: independent paraphrases vs locked lexicon (wide bag vs narrow).

A5 and A6 are specified; A5 adaptive search is not a confirmatory table yet.

## Privacy / legal

- Decision maker sees scores, not waveforms or transcripts (ADR-002).
- Nonce is not written to audit exports.
- Sidecar stores decisions. Recording-consent / TCPA / GDPR must be satisfied **before** a live-call pilot (not claimed here).

## Security of the sidecar

- Session isolation: one pipeline, CUSUM, belief, and agent per `call_id`.
- Shed under load: MONITOR only, RTP untouched.
- Poisoned prototypes and audit injection are residual risks (TIFS-scope).
