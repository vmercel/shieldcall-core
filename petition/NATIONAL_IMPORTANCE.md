# National Importance Addendum

This addendum is the prong-1 exhibit. It maps public U.S. facts onto the
specific ShieldCall endeavor. It is not a TAM slide.

## 1. Harm (problem exhibits, not endorsements)

- FTC Consumer Sentinel: impersonation remains a top-loss category. Public
  compilations through 2025–2026 put impersonation losses in the
  multi-billion-dollar range (CFA citing FTC: on the order of **$3.5B in
  2025**; older Americans on the order of **$445M in 2024**). Attach the
  latest Sentinel tables at filing, not a blog recap.
- FCC 8 February 2024 Declaratory Ruling: AI-generated voices are “artificial”
  under the TCPA. The ruling is a **regulatory hook**; it does not authenticate
  media. Citation to be pinned to the DA/FCC number in the exhibit binder.
- FTC Voice Cloning Challenge (winners April 2024): the U.S. government has
  already named detection and prevention of cloned-voice fraud as a
  consumer-protection R&D priority.
- CISA / IC3 / AARP Fraud Watch: downstream beneficiaries are the public,
  especially older adults targeted by government- and family-impersonation
  calls.

## 2. STIR/SHAKEN limitation

STIR/SHAKEN attests **caller ID**, not voice authenticity and not script.
A signed “A” attestation can still carry a human vishing script or a
vocoded harvest. That gap is the national-importance wedge: the missing
public-interest layer between number authentication and a human operator.

## 3. OSTP CET 2024 map (one sentence each)

| CET subfield | ShieldCall component |
|--------------|----------------------|
| AI assurance and assessment | Disjunctive OR-label evaluation doctrine; public negatives |
| Planning, reasoning, and decision making | Sufficient-statistic agent with interruption cost |
| AI safety, trust, security, responsible use | Fail-open media, fail-closed actuation, abstention |
| Communications and network security | Telephony Channel Twin as evaluation law; sidecar beside RTP |
| Privacy-enhancing technologies | Decision maker never sees waveform or transcript |
| Data fusion | CSCF / calibrated-OR of acoustic and linguistic streams |
| Adaptive network controls | Coverage-debt enroll of new synthesizer families |

## 4. Why dual-stream, mid-call, budgeted

- Human vishing is invisible to ASVspoof-style utterance EER.
- Vocoded probes are invisible to keyword products.
- Mid-call production change timed to harvest is invisible to both
  whole-utterance anti-spoofing and bag-of-words NLP.
- A liveness nonce wasted on a live social engineer is a product failure.
  The agent’s warn-on-SE policy is the control-layer response.

## 5. U.S. insertion points

**Primary:** CPaaS media fork (Twilio / Bandwidth / Vonage) — SIPREC or
media-stream copy into the sidecar. RTP stays on the CPaaS path.

**Backup:** contact-centre SIPREC; carrier analytics sidecar; handset OS
call-screening as a later, on-device port.

See `docs/ADR-003-sidecar-runtime.md` and `docs/figures/deployment_sidecar.png`.

## 6. Prospective impact without TAM

If trialed on a regional credit-union CPaaS, the operating metric is
precision at a declared alert budget (FPR 1% and 5%), not market share.
Officers should discount any slide that leads with TAM/SAM/SOM of
“the fraud-detection market.”

## 7. What this addendum does not claim

- No live U.S. caller is protected by the current prototype.
- No carrier letter is attached yet (P1).
- No government endorsement of the petitioner is implied by citing FTC/FCC.
