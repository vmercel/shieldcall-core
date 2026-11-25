# ADR-016: Risk-triggered acoustic cascade (Tier 1 in shadow)

## Status

Accepted (2026-11-25). Implements the first slice of TDD v2.0
section 6.5 (risk-triggered acoustic cascade) and roadmap stage 4
("Integration"): Tier 1 live behind the cascade. This slice
delivers the cascade decision logic, the 2 s window assembly,
the Phase 8 Tier 1 scorer, and the offline kill-criteria
evaluator, all in shadow. It changes no live decision path:
CSCF stays the production fusion, no sidecar or app threshold
moves, and Tier 1 never scores a live call in this slice.
Pipeline wiring, background-thread execution, and the 7-day
shadow gate (G7) are follow-ups listed in item 11.

## Context

The gap TDD v2.0 section 6.5 names is exact. The best
measured acoustic recipe runs only in the evaluation harness,
while the live path uses STRF alone. Phase 8 measured the
truncated wav2vec2 encoder (transformer layers 1 to 4,
pooled statistics, logistic head) at 11.78% pooled EER and
0.442 min t-DCF on the full ASVspoof 2021 LA evaluation
(148,176 trials, model hash in `phases/phase8/PROTOCOL.md`).
That number describes whole clips scored offline. It says
nothing about 2 s windows scored live, and this ADR does not
reuse it as a live claim.

The cascade closes the gap with what is already built.
STRF runs on every frame (Tier 0, built). The Phase 8
encoder runs on a 2 s window only when a trigger fires
(Tier 1). TDD section 6.5 lists five triggers, an expected
cost model, and a design target. Section 10.3 adds the
runtime rule that Tier 1 runs on a separate thread and
cannot delay the 10 ms acoustic cadence. Table 31 fixes the
kill criteria before the work starts: the cascade continues
only if the trigger rate is at most 0.15 on benign traffic
and the cascade miss rate is within 1 point of always-on
Tier 1; otherwise Tier 1 runs always-on over fewer calls
per core, or the distilled student is prioritised.

Three of the five triggers cannot be built exactly as TDD
specifies them today, and the differences are recorded
here rather than smoothed over.

First, trigger (ii) in TDD section 6.5 is a change-point
posterior above 0.5, which belongs to the BOCPD detector of
section 6.7. BOCPD is PLANNED, not built. What is built is
the one-sided StreamingCUSUM alarm (`shieldcall/acoustic/
changepoint.py`), already used by the pipeline to feed the
retired SAPC coupling. This slice therefore takes a latched
CUSUM alarm during the window as the change-point trigger.
A CUSUM alarm is a cumulative-excess statistic crossing a
fixed threshold, not a posterior, and it is named as an
alarm everywhere in the code. Replacing it with the BOCPD
posterior is a follow-up, not a rename.

Second, trigger (iii) is the language stream reaching a
requested-action state, which belongs to the factorial
discourse model of section 7.3. That model is PLANNED, not
built. What is built is the SDTG stage tracker
(`shieldcall/linguistic/discourse.py`), whose stages
HARVEST and PAYMENT are the stages at which the far end
asks for credentials, codes, or payment. This slice
therefore takes a latched SDTG stage in HARVEST or PAYMENT
during the window as the requested-action trigger. The
stage set is a configuration value, not a hardcoded
constant, so the factorial model's action set can replace
it without changing the trigger code. The SDTG is the
auditable floor, not the factorial model, and it is named
as such.

Third, trigger (i) needs a Tier 0 LLR. The live scorer
emits a synthetic probability, not an LLR. Stage 2a
(ADR-005) fitted per-cell affine LLR maps, but no fitted
map exists for live traffic, and none is invented here.
This slice accepts a Tier 0 LLR from the caller and
provides `prob_to_llr`, a clipped logit of the Tier 0
probability, for callers with no fitted calibrator. A
logit of an uncalibrated probability is an uncalibrated
LLR, and it is flagged as such. The trigger threshold
tau0 is a configuration value with a neutral default of
0.0 (an LLR of zero means spoof and bona fide are equally
likely under the calibration in use). That default is not
a fitted operating point. TDD section 6.5 says tau0 is
set low, for high recall and poor precision, so that the
expensive tier does the deciding. The fitted value must
come from benign shadow traffic, measured with the
evaluator in item 8, before any live use. Until then the
trigger rate target is a measurement to report, not a
property the code claims.

## Decision

1. **Window assembly is speech only, 2 s, sample counted.**
   `WindowBuilder` accumulates far-end speech samples into
   fixed 2 s windows at the Tier 0 sample rate (8 kHz,
   16,000 samples). Non-speech frames contribute nothing,
   so a window is 2 s of far-end speech, not 2 s of wall
   time. A window carries its start and end timestamps
   from the frames that filled it, and its index. A
   partial window at reset is discarded and counted, never
   scored as a short window. The builder takes sample
   arrays and a speech flag per frame, so it works the
   same for preprocessor frames and ingest frames, and it
   knows nothing about detectors.

2. **Five triggers, one decision, reasons recorded.**
   `CascadeTrigger.decide` takes the window Tier 0 LLR, a
   latched change-point alarm flag, and a latched
   requested-action flag, and returns a `CascadeDecision`
   with every reason that held. The reasons are:
   `first_window` (window index 0, every call, giving
   each call one strong baseline reading), `tier0_llr`
   (window Tier 0 LLR above tau0), `changepoint`
   (CUSUM alarm latched during the window),
   `requested_action` (SDTG HARVEST or PAYMENT stage
   latched during the window), and `random_audit`
   (a uniform draw below q = 0.02). A window triggers
   when any reason holds. The window Tier 0 LLR is the
   mean of the per-frame Tier 0 LLRs over the speech
   frames in the window. A mean is used rather than a
   maximum because a single noisy frame must not spend
   a Tier 1 pass on its own; the choice is stated here
   so it can be revisited from shadow data.

3. **Random audits use a caller-controlled seed, and
   production must not fix one.** The audit draw comes
   from a `random.Random` instance. A seed may be
   supplied for tests and reproducible offline runs.
   With no seed the generator is seeded from operating
   system entropy. A fixed seed in production would let
   a prober (adversary A5 in the TDD threat model) learn
   exactly which windows are audited, which defeats the
   purpose of the audit. The code refuses no seed, but
   this ADR states the rule: fixed seeds are for tests
   and offline measurement only. One draw is consumed
   for every window, including the first, so the draw
   sequence does not depend on which reasons held. The
   audit reason is recorded only after the first window,
   since the first window triggers anyway.

4. **Tier 1 is the Phase 8 head, loaded lazily, and
   fails closed.** `Phase8Tier1Scorer` loads
   `phases/phase8/model8.npz` (kind `lr`, layers 1 to 4,
   mean and standard deviation vectors, weights, bias)
   and scores a 2 s window through the truncated
   wav2vec2-base encoder. The head score in the model
   file is a bona fide logit: higher means bona fide.
   The scorer negates it, so its `score` is a spoof
   score where higher means more spoof-like, matching
   the repo convention that label 1 is spoof. The
   encoder is loaded only on the first scored window,
   from the local Hugging Face cache only
   (`local_files_only`), so construction never downloads
   a model and scoring never waits on a network. If
   torch, transformers, the cached backbone, or the
   model file is missing, scoring raises
   `Tier1UnavailableError`. There is no fallback score
   and no silent 0.5. The SHA-256 of the model file is
   recorded on the result for provenance; the signature
   is not verified in this slice (gate G8 remains open).

5. **No silence trim, no arbitrary resampling, per-window
   normalisation.** The Phase 8 evaluation trimmed leading
   and trailing silence per clip and capped clips at 4 s.
   Live windows are fixed 2 s slices of speech, so
   trimming would change what a window is, and it is not
   applied. Each window is normalised by its own mean and
   standard deviation, matching the per-clip normalisation
   of the evaluation harness applied at window scale.
   Input at 8 kHz is upsampled to the 16 kHz backbone
   rate by an exact factor of 2; input at 16 kHz passes
   through; any other rate is refused. The mismatch
   between whole-clip evaluation and windowed scoring is
   a measurement gap, not a detail: the shadow gate must
   re-measure Tier 1 on 2 s windows before any live use,
   and the 11.78% EER figure must not be quoted for
   windowed scores.

6. **Calibration is explicit, never assumed.**
   `Tier1Result` carries `score` (the uncalibrated spoof
   logit) and `llr`, which is present only when the
   caller supplies a fitted `PerCellCalibrator` (ADR-005)
   for the window's channel cell. Without a calibrator,
   `llr` is None and `calibrated` is False. No caller
   in this slice may treat the raw score as an LLR.
   The calibrator must be fitted on held-out Tier 1
   window scores, not on the Phase 8 clip scores, and
   fitting it is a follow-up owned by the shadow gate.

7. **Tier 1 never touches the live decision path in this
   slice.** The cascade module has no reference to the
   pipeline, the fusion engine, or the serve layer. The
   pipeline is untouched. CSCF stays the production
   fusion, and the calibrated OR fusion stays in shadow
   beside it (ADR-006). A later slice wires the cascade
   into the pipeline in shadow mode on a background
   thread, following TDD section 10.3: Tier 1 results
   merge when they arrive and never delay the 10 ms
   cadence. That wiring is deliberately not in this
   slice, because the trigger operating point, the
   windowed Tier 1 measurement, and the thread model
   each need the shadow evidence this slice makes
   measurable first.

8. **The kill criteria are measured offline, on Tier 1
   scores for every window.** `evaluate_cascade` takes
   rows of (label, triggered, tier1_score) and a Tier 1
   threshold, and reports the trigger rate, the benign
   trigger rate, FNRtrig (the fraction of synthetic
   windows Tier 0 failed to escalate), the always-on
   miss rate, the cascade miss rate, and the expected
   cost E[C] = C0 + r*C1 from TDD eq. 8. The evaluator
   requires a Tier 1 score for every row, including
   untriggered ones, because the cascade miss rate can
   only be measured against always-on Tier 1 by scoring
   Tier 1 on windows the cascade would have skipped.
   That is an offline harness property: a live cascade
   never computes those scores. The kill check passes
   only when the benign trigger rate is at most 0.15
   and the cascade miss rate is within 1 point (0.01)
   of the always-on miss rate. The Tier 1 cost default
   is 4.0 Tier 0 window units, which is the TDD bound
   ("at most about four Tier 0 windows"), not a
   measurement. Measured Tier 1 CPU time per window
   must replace it before the cost claim is used for
   capacity planning.

9. **Disagreement and input randomisation are deferred,
   stated plainly.** TDD section 6.9 specifies a random
   channel transform before Tier 1 and treating large
   STRF/Tier 1 disagreement as coverage-gap evidence.
   Neither is in this slice. Windows reach Tier 1
   untransformed, and disagreement is not computed.
   Both are follow-ups; the window and score types
   already carry what they need (window samples, Tier 0
   LLR, Tier 1 score) so neither follow-up changes the
   trigger interface.

10. **No new dependency.** The cascade module uses
    NumPy only. SciPy is used for the exact 8 kHz to
    16 kHz resampling and is already a declared
    dependency. Torch and transformers stay optional:
    they are required only to run the Phase 8 scorer,
    never to import the module or run the test suite.
    The test suite uses an injected embedding function
    for the scorer, so no model download and no GPU
    are needed to test the head arithmetic, the sign
    convention, or the failure paths.

11. **Follow-ups, in order.** (a) Pipeline shadow
    wiring on a background thread, recording cascade
    decisions and Tier 1 scores beside CSCF verdicts
    without changing them (TDD section 10.3). (b) Tau0
    set from benign shadow traffic to meet the r <=
    0.15 target, and the windowed Tier 1 re-measurement
    on labelled 2 s windows. (c) Per-cell calibration
    of Tier 1 window scores on held-out data (ADR-005
    maps, fitted for windows). (d) BOCPD posterior
    replacing the CUSUM alarm proxy, and the factorial
    requested-action state replacing the SDTG stage
    proxy. (e) Input randomisation and disagreement
    handling (TDD section 6.9). (f) Seven-day shadow
    run (G7) and the release gates G1, G2, and G6
    before any live use. (g) Time-to-warning measured
    end to end (TDD section 10.4), which needs the
    wired cascade plus ASR and rendering timing, and
    is its own slice. Nothing in this slice pre-empts
    any of these, and none of them is claimed done.

## Consequences

The cascade can now be exercised offline: windows can be
assembled from recorded calls, triggers can be evaluated
with a seeded audit stream, Tier 1 can be scored where a
cached backbone and the Phase 8 model exist, and the kill
criteria can be computed from the resulting rows. What
cannot yet be claimed: any trigger rate on real traffic,
any windowed Tier 1 accuracy, any latency or cost figure,
or any effect on live verdicts. The 11.78% Phase 8 EER
remains a whole-clip offline result.

## Verification

Tests in `tests/test_cascade.py` use synthetic data with
known ground truth, following the stage 2a and 2b test
pattern. They cover: each trigger reason in isolation
and in combination, seeded audit determinism and rate,
speech-only window assembly and partial-window discard,
the logit conversion and its clipping, the Phase 8 head
arithmetic and sign convention through an injected
embedding function, refusal paths for missing models and
unsupported sample rates, and hand-computed kill-criteria
cases including the pass and fail boundaries. The full
core suite must stay green. No licensed ASVspoof data is
touched, and no live call is scored.
