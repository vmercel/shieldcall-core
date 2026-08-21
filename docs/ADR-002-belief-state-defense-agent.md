# ADR-002: Belief-state call-defense agent (not an LLM wrapper)

## Status

Accepted

## Context

The detector is a sensor. A live call still requires *actions*: wait, challenge, warn, escalate, abstain, or enroll a new voice family. Those actions cost user trust. An LLM that “reasons” over transcripts would add latency, privacy risk, and an untested decision maker. We need agency in the Russell–Norvig sense: percepts, a world model, a utility, and tools.

## Decision

Put a **discrete-hypothesis agent** on top of ShieldCall:

1. The agent never sees raw audio or full transcripts — only sufficient statistics (scores, regime, coupling, coverage gap).
2. World model = five mutually exclusive hypotheses: benign, social engineering, fully synthetic, mid-call handoff, unknown family.
3. Belief is a normalized distribution updated by heuristic likelihoods (not a calibrated Bayes net; we say so).
4. Actions are experiments. The planner maximises expected information gain minus interruption cost minus delayed-threat risk.
5. At most one liveness challenge per call (budget). A **human** social engineer will pass a nonce; the policy therefore prefers **warn/escalate** on that hypothesis and saves the challenge for synthetic / unknown-family / handoff.
6. No LLM in the decision path. Natural-language rationale is a template over the belief.

## Alternatives considered

**LLM ReAct agent over the transcript**
Rejected: unmeasured, non-private, slow, and would re-introduce keyword theater in prose form.

**Open-loop thresholds only**
Rejected: cannot spend an interruption budget, cannot distinguish “I should wait” from “I should probe.”

**Full POMDP solver**
Rejected: no transition model we trust; overclaim. Info-gain on a 5-atom belief is the honest middle.

## Consequences

### Positive
- Decisions are auditable traces.
- Matches the actual science (disjunctive threats, handoff as a hypothesis).
- Privacy: scores in, actions out.

### Negative
- Likelihoods are hand-set; we do not claim optimal Bayesian experimental design on real calls.
- Agent quality is measured by *action choice on scripted percepts*, not by ASVspoof EER.

## References
- ADR-001 (SAPC / ACI)
- Russell & Norvig, *AIMA* (rational agents)
- Lindley, on-information for experiments
