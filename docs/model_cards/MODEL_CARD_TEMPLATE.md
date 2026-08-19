# ShieldCall Model Card Template

## Model identity

| Field | Value |
|---|---|
| Model name | [name] |
| Version | [semantic or registry version] |
| Code commit | [full commit SHA] |
| Owner | [owner] |
| Release status | research, internal evaluation, pilot, or production |
| Supported channels | [authorized VoIP, enterprise integration, or other supported channel] |

## Intended use

State the supported user, communication channel, language, model task, and warning behavior. State explicitly that the model assists fraud-risk verification and does not establish criminal conduct, voice identity, or a guarantee of safety.

## Data and provenance

Identify the dataset manifest fingerprint, all data sources, consent or license references, date range, language distribution, channel conditions, attack families, and exclusions. Confirm that the locked test split was not used for selection or threshold tuning.

## Evaluation

| Metric | Calibration split | Locked test split | Notes |
|---|---:|---:|---|
| EER or task-appropriate acoustic metric | [value] | [value] | [by attack family and codec] |
| ROC-AUC | [value] | [value] | [class balance] |
| PR-AUC | [value] | [value] | [positive definition] |
| Brier score | [value] | [value] | [probability quality] |
| Expected calibration error | [value] | [value] | [binning method] |
| Coverage and abstention | [value] | [value] | [interval method] |
| End-to-end latency | [value] | [value] | [device or service profile] |

Report slices by supported language, codec, attack family, ASR confidence, and any other lawful and ethically collected dimension relevant to deployment.

## Known limitations

Describe unsupported channels, languages, threat types, noisy conditions, potential false positives, potential false negatives, and user-safety limitations. Never replace this section with generic statements.

## Privacy and security

State whether raw audio is persisted, consent purpose, retention duration, encryption controls, model-improvement eligibility, audit fields, access controls, and incident-response owner.

## Deployment and rollback

Record the threshold policy, warning text version, monitoring dashboard, alert-rate limits, preceding approved model version, rollback procedure, and release approvers.
