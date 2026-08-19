# Security Policy

## Security posture

ShieldCall Core is a research and engineering library. It is not a consumer-facing fraud service, does not provide a hosted endpoint, and must not be used as the sole basis for blocking a call, accusing a caller of fraud, or making a financial decision.

A production implementation must treat call-derived information, risk scores, transcripts, and consent records as sensitive data. It must use authenticated and authorized service boundaries, encryption in transit and at rest, tenant isolation, audit logs, secret management, dependency scanning, and a documented incident-response process.

## Supported versions

Only the latest release on the default branch is supported for security fixes. Development branches and local research experiments are not production releases.

## Reporting a vulnerability

Do not open a public issue for a potential security vulnerability. Report it privately to **[security contact to be configured before public release]** with a concise description, reproduction details, impacted component, and any proof of concept that does not expose real call data.

The maintainer should acknowledge a report within five business days, assess severity, coordinate a remediation timeline, and publish a release note after a fix is available. Credit will be offered to reporters who request it.

## Data restrictions

Raw call audio, transcripts, caller identifiers, access tokens, and secret keys must never be committed to this repository. Test fixtures must be synthetic, public, or collected under documented consent and licensing terms. Model adaptation candidates must enter only through the governed review workflow and must not contain raw audio.
