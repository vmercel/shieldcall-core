# ADR-005: Per-cell LLR calibration and actDCF / Cllr reporting

## Status

Accepted (2026-11-14). Implements TDD v2.0 §5 (Channel Passport, per-cell
calibration), §6.6 (calibrated outputs), §15.4 (metrics) and the G2 release
gate (§11.2). First slice of roadmap stage 2 ("Baselines, calibration and
fusion"). The calibrated OR fusion rule (§8.1) and its shadow run beside CSCF
are a separate, later item that builds on this one.

## Context

Phase 8 is measured and locked (11.78% pooled EER, 0.442 min t-DCF on the
full ASVspoof 2021 LA set), but the same model scores 9.09% EER on
uncompressed audio and 16.35% on PSTN-transmitted audio (TDD v2.0 §5). A
score means different things on different channels, so one global threshold
is wrong: the per-codec actDCF spread is the quantity that has to shrink,
and EER / min t-DCF alone cannot show it, because both are oracle
thresholds. TDD v2.0 §15.4 therefore requires actDCF at the threshold the
system actually uses, plus Cllr and Cllr-min, reported per channel cell with
bootstrap confidence intervals.

No part of this existed in the repo before this ADR: `shieldcall/eval/
metrics.py` had EER, min DCF, AUC, Brier and ECE, but no actual-DCF and no
Cllr; `shieldcall/fusion/` had a conformal calibrator and a `calibrated_or`
static helper (noisy-OR of two probabilities), but no LLR calibration and no
per-cell maps.

## Decision

1. **Scores become LLRs per cell.** Every acoustic score `z` in channel cell
   `c` is mapped to a log-likelihood ratio by a cell-specific affine map
   (TDD eq. 9):
   `l = a_c * z + b_c`, with `(a_c, b_c) = argmin Cllr` on held-out data of
   cell `c`. Fitting minimises Cllr (logistic loss, convex in `(a, b)`),
   implemented as damped Newton iterations in numpy, no new dependency.
   The fit is done on held-out data only; the module takes explicit fit
   data and never splits it, so the caller owns the train / calibration /
   test separation.
2. **Cells are the ASVspoof 2021 LA codec conditions** for evaluation
   (C1 none, C2 alaw, C3 pstn, C4 g722, C5 ulaw, C6 gsm, C7 opus), and any
   caller-supplied cell key at runtime. The Channel Passport itself
   (blind channel estimation, §5.1) is a later item; this ADR only defines
   the calibration that consumes its cell label.
3. **Low-support cells fall back to the pooled map.** A cell is fitted only
   when it holds at least 20 spoof and 20 bona fide trials
   (`min_class_count`, overridable); otherwise the pooled map is used and
   the cell is recorded in `fallback_cells` for audit. An unknown cell at
   transform time also falls back to the pooled map. This is a deliberate
   fail-open for scoring: a hard failure in the scoring path is worse than
   a pooled calibration, and the fallback is visible, not silent. Runtime
   abstention on out-of-support channels (TDD §5.2 item 3) is a separate
   Passport item, not smuggled in here.
4. **Report actDCF, Cllr and Cllr-min per cell and pooled.** actDCF is the
   normalised DCF at the Bayes threshold `l* = log(C_fa / C_miss) -
   logit(pi)` (TDD §6.6; with the repo's default `pi = 0.05`,
   `C_miss = C_fa = 1`, `l* = log(19) ≈ 2.944`). The normaliser matches the
   existing `min_dcf` so actDCF and minDCF are directly comparable, which
   is what gate G2 needs ("actDCF within 10% of minDCF"). Cllr is TDD eq.
   10 (base-2 log). Cllr-min is the minimum over monotone recalibrations,
   computed with the pool-adjacent-violators algorithm; PAV posteriors are
   clipped to `[1e-6, 1 - 1e-6]` so perfectly separable data gives a small
   finite value instead of infinity.
5. **Module placement follows the TDD module map.** The calibration service
   lives in `shieldcall/fusion/calibration.py` (TDD §17 table). The
   reporting metrics (`bayes_threshold`, `act_dcf`, `per_cell_report`)
   live in `shieldcall/eval/metrics.py` next to `min_dcf`, and import Cllr
   from the calibration module. No import cycle: `fusion.calibration`
   imports only numpy.
6. **Acceptance criterion (TDD §5, "Evidence status").** On ASVspoof 2021
   LA, per-cell calibration must narrow the spread between the best and
   worst codec actDCF without raising pooled minDCF. The report function
   returns that spread (`actdcf_spread`) and the pooled minDCF so the
   criterion is checkable in one call. It cannot be run here yet: no
   licensed ASVspoof data is present locally (see Consequences).

## Alternatives considered

**One global calibration map.** Rejected: it is exactly the failure §5
documents. A single affine map cannot fix a per-codec shift, and the
actDCF spread is the metric that would stay wide.

**Isotonic (non-parametric) calibration per cell.** Deferred: strictly more
flexible, but needs far more per-cell data to avoid overfitting, and the
TDD specifies the affine map (eq. 9). The PAV-based Cllr-min already gives
the isotonic reference for comparison.

**Platt scaling via sklearn LogisticRegression.** Rejected: adds a runtime
dependency to the scoring path and hides the optimisation. The Newton
solver is ~30 lines, convex, and tested against known ground truth.

**Cllr in nats instead of bits.** Rejected: TDD eq. 10 is base-2; reporting
in bits keeps the numbers comparable with the ASVspoof literature the TDD
cites.

## Consequences

- `shieldcall/fusion/calibration.py` is new: `cllr`, `cllr_min`,
  `fit_affine_llr`, `PerCellCalibrator`. `shieldcall/eval/metrics.py`
  gains `bayes_threshold`, `act_dcf`, `per_cell_report`.
- The G2 gate is now computable: pooled and per-cell Cllr, actDCF vs
  minDCF, all per cell. The gate itself (pass/fail on a candidate) is not
  implemented here; it needs a candidate scored on real data.
- Full validation on ASVspoof 2021 LA is still blocked on licensed data
  (the P2-2 data gap, unchanged). The mechanism is validated on synthetic
  data in `tests/test_calibration.py`, including a constructed two-cell
  case where per-cell calibration narrows the actDCF spread, mirroring
  the TDD acceptance criterion.
- The next stage-2 item (calibrated OR fusion in shadow beside CSCF) now
  has calibrated LLRs to consume; it must not re-derive calibration.
- Nothing in this change alters any live decision path: CSCF remains the
  production fusion, and no threshold in the sidecar or app changes.
