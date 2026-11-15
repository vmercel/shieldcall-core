# ADR-006: Calibrated OR fusion in shadow beside CSCF

## Status

Accepted (2026-11-15). Implements TDD v2.0 §8.1 (calibrated OR fusion
with a monotone envelope, Proposition 1) and the G7 shadow gate
(§11.2). Second slice of roadmap stage 2 ("Baselines, calibration and
fusion"), building on ADR-005 per-cell LLR calibration. CSCF stays the
production default; the OR rule runs in shadow until it passes G2 +
G7.

## Context

CSCF (the built fusion layer, `fusion/engine.py`) blends the acoustic
and linguistic streams with a confidence-weighted base and hand-set
regime floors. The floors exist because a plain weighted average
dilutes a strong single-stream signal: a cloned voice with benign
words, or a human social engineer with a clean voice, can both score
below threshold. TDD v2.0 §8.1 replaces the hand-built floors with a
rule that has the property by construction: the fused threat is the
probability of the OR of the two events (synthetic voice V, fraud
script S), with one fitted dependence parameter rho. Proposition 1
bounds it between the strongest stream and noisy-OR, so no signal is
ever averaged away and evidence can never hurt.

Stage 2a (ADR-005) supplies the input the rule needs: calibrated LLRs
per channel cell. What this ADR adds is the rule itself, the rho fit,
the severity channel, and the shadow harness that lets the two rules
run side by side without touching the production decision.

## Decision

1. **Per-stream posteriors from calibrated LLRs (TDD eq. 17).**
   `pV = sigma(l_ac + logit(pi_V))`,
   `pS = sigma(l_lang + logit(pi_S) + delta_att)`, with deployment
   priors pi_V = pi_S = 0.05 by default (the repo's p_target
   convention; base rates to be set from measured traffic, not tuned)
   and the STIR/SHAKEN term delta_att in {0, 0.07, 0.18} nats for A,
   B, C or unsigned, bounded by |delta_att| <= 0.2. Attestation moves
   the prior odds of a scam, never the verdict.
2. **Fused threat (TDD eq. 18-19).**
   `P(V and S) = (1 - rho) pV pS + rho min(pV, pS)`,
   `p_threat = (1 - rho) noisy-OR + rho max(pV, pS)`,
   with rho in [0, 1]. The implementation (`fusion/or_fusion.py`)
   exposes `joint_prob`, `noisy_or`, `fused_threat` and `severity` as
   pure functions, and the envelope properties are tested directly:
   max <= p_threat <= noisy-OR, monotonicity in each argument, rho =
   0 is noisy-OR, rho = 1 is max.
3. **Rho is fitted, not tuned.** `fit_rho` minimises the Cllr of the
   fused output on paired development calls: a coarse grid over [0, 1]
   plus golden-section refinement in the best bracket, so the fit is
   never worse than either endpoint on the fit data. A synthetic
   recovery test (labels drawn from the true fused model at rho = 0.6)
   recovers rho within 0.15; a fully redundant case fits rho >= 0.8.
4. **The fused LLR subtracts the threat prior.** The Cllr is defined
   on true log-likelihood ratios, and the fused output is a posterior,
   so the LLR is `logit(p_threat) - logit(pi_threat)`. The threat
   prior is the model-implied one, `pi_V + pi_S - P(V and S)` at the
   stream priors, or the empirical label rate of the scored set when
   the stream priors are not given. Using `logit(p_threat)` alone
   smuggles the prior into the score and biases the fit toward
   whichever rho implies a prior closest to 0.5; that bug was caught
   by the rho-recovery test during development and is locked in by a
   regression test.
5. **Severity is carried separately.** `sev = P(V and S) / p_threat`
   (0 when there is no threat) is passed alongside the probability so
   the agent can choose stronger guidance on dual threats (an
   AI-voiced scam) without the probability itself meaning something
   else. The v1 co-activation boost, which multiplied the score, is
   not reproduced.
6. **Temporal smoothing is leaky integration.** Per-stream LLRs
   accumulate as `l_bar_t = beta l_bar_{t-1} + l_t` with beta = 0.9
   per fusion tick and a cap |l_bar| <= 8. This replaces v1's sticky
   synthetic score and trajectory boost with one rule that has a
   likelihood interpretation.
7. **Shadow mode, no decision-path change.** `ShadowFusion` feeds the
   same ticks to a CSCF engine and an `OrFusion`, returns the CSCF
   risk untouched, and records paired rows. `shadow_compare` reports
   per-system Cllr / EER / minDCF / actDCF (prior-corrected LLRs, same
   conversion for both) plus the tier-disagreement rate at the CSCF
   thresholds (0.35 / 0.62). A test verifies CSCF risk is bit-identical
   with and without the shadow wrapper.
8. **Release gate.** The OR rule becomes the production default only
   when it passes G2 (pooled and per-cell Cllr no worse than the
   incumbent; actDCF within 10% of minDCF) and G7 (at least 7 days
   scored in shadow; disagreement with the incumbent reviewed). Both
   gates need paired development calls with labels, which do not
   exist yet in the repo: fitting rho on real data and the shadow run
   are blocked on that data, not on code.

## Consequences

- No live decision path changes in this commit: CSCF stays the
  production fusion, no sidecar or app threshold moves, no new
  dependency (numpy only).
- The rho fit and the shadow report are honest about their inputs:
  they take explicit paired data and never split it, so the caller
  owns the train / calibration / test separation, as in ADR-005.
- Full validation on real paired calls still needs the data: the
  synthetic tests prove the rule, the fit, and the envelope, not the
  value of rho on real traffic.
