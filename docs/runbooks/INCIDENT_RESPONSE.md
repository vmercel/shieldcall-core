# Incident Response Runbook

## Purpose

This runbook covers suspected security disclosures, consent failures, model-risk incidents, and service outages in a ShieldCall pilot. It must be adapted to the deployed environment before any live pilot begins.

## Immediate actions

| Incident type | First action | Escalation |
|---|---|---|
| Suspected raw-audio or transcript exposure | Stop affected data flow, preserve minimal forensic metadata, rotate relevant credentials, and restrict access. | Security lead, privacy lead, tenant owner, and counsel as appropriate. |
| Consent failure or unauthorized analysis | Stop analysis for the affected integration or tenant, preserve consent and audit records, and begin deletion or containment review. | Privacy lead and tenant owner. |
| Material false-alert spike | Disable the affected model or threshold policy, roll back to the prior approved release, and notify pilot operations. | Model owner and product safety lead. |
| Missed high-risk pattern discovered during review | Preserve the governed case record, assess whether it is a data, ASR, model, or policy issue, and do not silently retrain. | Model owner and review panel. |
| Service outage or elevated latency | Switch to a clear unavailable state, not a silent fail-open fraud claim. Preserve health metrics and restore through tested rollback procedures. | On-call engineering owner. |

## Evidence handling

Public tickets and chat messages must not contain raw communications, unredacted transcripts, caller identifiers, secret values, or full consent records. Use the approved secure incident store. Record only the incident identifier, affected tenant, model version, integration channel, time window, containment action, and owner in ordinary operational systems.

## Model rollback

A model rollback requires the release identifier, previous approved model identifier, reason, owner, and verification that the deployment is serving the intended preceding version. A rollback must be documented in the model registry and post-incident review.

## Post-incident review

Within five business days, record the root cause, user impact, data impact, timeline, remediation, tests added, owner, and whether policy, model, platform, or documentation changes are required. If the incident concerns a vulnerability, follow `SECURITY.md` for responsible disclosure handling.
