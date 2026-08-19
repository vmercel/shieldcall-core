# ShieldCall MVP Integration Scope

## Product boundary

ShieldCall Core is not a general-purpose call-recording system. The MVP analyzes only audio delivered by a supported, authorized integration and a current `fraud_risk_analysis` consent record. The reference implementation enforces this policy through `CallSession.authorize_audio_analysis()`.

## Supported first-release channels

| Channel | MVP behavior | Audio analysis | Status |
|---|---|---|---|
| User-controlled VoIP application | Real-time warnings during calls that the application originates or receives through its own authorized media path | Supported after consent | Primary MVP path |
| Enterprise contact center | API or media-stream integration with documented customer and agent consent | Supported after consent and tenant setup | Primary B2B pilot path |
| Android call screening | Caller-number labeling, reputation checks, silence or reject actions as permitted by platform rules | Not supported through the core contract | Companion feature only |

## Explicit exclusions

The MVP does not claim to analyze the content of every native cellular call. It does not use accessibility abuse, undocumented recording workarounds, or hidden capture mechanisms. It does not automatically block calls based on model output. Emergency-calling paths are outside the model-control boundary.

## Integration handshake

1. The integration authenticates the tenant and creates an `AnalysisConsent` record with the `fraud_risk_analysis` purpose.
2. The integration creates a `CallSession` with a supported VoIP or enterprise channel.
3. The integration sends short-lived `AudioChunk` objects only for that session.
4. ShieldCall emits a warning and redacted audit metadata. The integration decides how to present the warning to the user.
5. The integration must provide a means to withdraw consent and request deletion according to the tenant's data-governance policy.

## User-warning principles

Warnings must use cautious language. They should advise users to pause, avoid sharing money or sensitive information, and verify through an official channel. They must not declare that a caller is a criminal or that a voice is definitively synthetic.
