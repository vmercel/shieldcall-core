# Model Release Standard

A ShieldCall model release is eligible for a pilot only when every requirement below is documented in a versioned release record.

## Evidence requirements

| Requirement | Required evidence |
|---|---|
| Dataset provenance | Dataset manifest with source, version, consent or license reference, split, language, codec condition, attack family, and sample counts. |
| Leakage control | Verification that speakers and materially linked examples do not overlap between training and locked test splits. |
| Baselines | Acoustic-only, linguistic-only, and simple late-fusion baselines measured on the same locked test set. |
| Calibration | Held-out calibration report with Brier score, expected calibration error, interval coverage, and abstention rate. |
| Robustness | Results sliced by codec, channel, attack family, ASR confidence band, language, and other lawful, documented categories. |
| Safety | Warning policy review, false-positive analysis, emergency-call exclusion, and user-comprehension test. |
| Security | SBOM, dependency scan, secrets scan, signed artifact, and rollback target. |
| Privacy | Consent and retention review, data minimization review, and approved deletion workflow. |

## Release decision

The release owner must document the model version, code commit, dataset fingerprints, calibration artifact, threshold policy, supported channels, known limitations, owners, and rollback plan. A release may not claim external validity beyond the populations and channel conditions tested.

## Monitoring requirements

Each deployed release must emit redacted operational measurements: request volume, latency, warning rate, abstention rate, error rate, model version, and user-reported false alerts. Model drift or a material change in call channel, ASR, language mix, or attack family requires re-evaluation before thresholds are changed.

## Rollback

A tested rollback to the immediately preceding approved model must be available before pilot deployment. Model updates based on adaptation candidates require a governed release, reviewer rationale, and traceability to approved candidate identifiers.
