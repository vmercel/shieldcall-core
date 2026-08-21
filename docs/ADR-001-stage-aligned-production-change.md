# ADR-001: Stage-aligned production change (SAPC) and ACI

## Status

Accepted as *implemented methods*. Audio handoff *claim* is **not** accepted.

## Context

Utterance-level deepfake scores miss mid-call switches (human rapport, then a vocoded harvest). PartialSpoof localises fake *segments* but does not ask whether the fake *onset* lines up with a vishing stage. Score averaging cannot see timing.

## Decision

1. Detect production change with Page CUSUM online and a two-window mean-shift offline.
2. Score **coupling** between that time and harvest/payment/threat/secrecy stage times with a Gaussian kernel.
3. Calibrate with a circular-shift permutation p-value (not Pearl causality).
4. Replace the old EMA “conformal” story, for labeled streams, with Gibbs–Candès adaptive conformal inference (ACI).

## Alternatives considered

- Frame-level PartialSpoof classifiers — different question (any fake frame vs timing with language).
- Unique-information PID fusion — deferred; needs a reliable joint histogram.
- Tuning SAPC until LibriSpeech splices passed — rejected as p-hacking.

## Evidence

- Synthetic point processes: SAPC ranks aligned vs unaligned (unit tests + `sapc_on_clean_point_processes`).
- ACI: empirical coverage 0.885 vs target 0.90 on a synthetic prevalence shift; a frozen quantile was *not* worse on that stream.
- Pulse-formant splices into Mini LibriSpeech: matched mix, but coupling did **not** beat mean synthetic probability (claim not supported).

## Consequences

- Positive: the question is well-posed and falsifiable; the code exists; ACI is the published recursion.
- Negative: do not cite SAPC as detecting real vocoded handoffs until a cleaner construction or PartialSpoof+transcripts shows an effect.
