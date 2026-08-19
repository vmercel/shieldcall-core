# Privacy and Data Governance

## Scope

This document defines the minimum data-governance requirements for a ShieldCall MVP that analyzes authorized VoIP or enterprise-contact-center communications. It does not authorize collection or analysis of every native carrier call. Android call screening may be used for caller identification and call-response workflows, but it is not treated as a general call-audio capture path.

## Data minimization

The supported MVP should process only the data necessary to create a time-bounded fraud-risk warning. Raw audio should remain on device or within the authorized integration whenever possible. The core service should prefer ephemeral audio chunks, time-stamped transcripts, model scores, warning decisions, consent identifiers, and redacted audit metadata. It must not store raw audio by default.

| Data category | Default handling | Retention rule |
|---|---|---|
| Raw audio | Do not persist by default | Retain only with separate explicit consent, a documented purpose, encrypted storage, and a defined expiry. |
| Transcript fragments | Process transiently where feasible | Retain only the minimum redacted text required for an opted-in case review. |
| Risk score and warning | Store as redacted audit metadata | Retain according to the tenant's documented fraud-prevention purpose and deletion schedule. |
| Consent record | Store consent identifier, scope, time, and expiration | Retain for the period needed to demonstrate lawful processing. |
| Model-improvement candidate | Store feature vector and provenance, not raw audio | Retain only after governed review and according to model-release policy. |

## Consent requirements

Before analyzing audio, the supported channel must collect a clear and affirmative consent record for the purpose `fraud_risk_analysis`. The record must identify the subject, purpose, time of capture, expiration if any, and method of withdrawal. Consent for real-time warning does not automatically authorize model improvement, research use, or long-term audio retention. Those purposes require separate consent or another documented lawful basis reviewed by counsel.

## Model-improvement governance

No live call sample may alter a model automatically. Adaptation candidates must use the governed intake workflow and include a consent record, source, label method, collection channel, quality score, model version, and reviewer rationale. An approved release must be versioned, auditable, and reversible. Raw audio should not be entered into the adaptation registry.

## Access, deletion, and incident response

Production deployments must enforce least-privilege access and tenant separation. They must provide a documented route for consent withdrawal, data deletion requests, and incident reports. Any suspected disclosure of raw communications, transcripts, identifiers, credentials, or unapproved model-training data must be handled under the incident-response runbook and investigated without copying sensitive content into public tickets.

## Prohibited practices

The MVP must not rely on undocumented operating-system workarounds to capture call audio. It must not make automated criminal accusations, block emergency calls based solely on a model score, retain audio silently, or use user communications for training without the required authorization and governance record.
