# ADR-018: Pipeline shadow wiring (cascade and factorial, background thread)

## Status

Accepted (2026-11-27). Implements ADR-016 follow-up (a)
(pipeline shadow wiring on a background thread) and ADR-017
follow-ups (a) and (b) (pipeline shadow wiring for the
factorial model, and the cascade proxy replacement). It
changes no live decision path: CSCF stays the production
fusion, the calibrated OR fusion stays in shadow beside it
(ADR-006), and no sidecar or app threshold moves. Tier 1
never scores a live call in this slice unless the operator
supplies a Tier 1 scorer, and even then its scores are
recorded beside CSCF verdicts and never fed back into them.

## Context

Three modules exist but none of them runs on a live call.
The cascade (ADR-016) decides, per 2 s speech window,
whether Tier 1 should look, and scores windows offline.
The factorial discourse model (ADR-017) tracks stage,
tactics, and requested action per transcript turn, and
exposes `requested_action_active` as the exact form that
was to replace the SDTG-stage proxy in the cascade trigger.
The pipeline runs STRF on every frame and the SDTG on
every transcript turn, and fuses them with CSCF. Nothing
joins the three. TDD section 10.3 requires the join to
run Tier 1 on a separate thread that cannot delay the
10 ms acoustic cadence, and Table 31 requires the shadow
evidence (trigger rate, windowed Tier 1 accuracy, lead
time) before any live use. This slice is that join, in
shadow only.

Two interface facts shape the design, and both are
recorded rather than smoothed over.

First, pipeline frames overlap. The preprocessor emits
25 ms frames on a 10 ms hop, so consecutive frames share
15 ms of audio. Feeding whole frames to the cascade
`WindowBuilder` would double-count audio, fill a 2 s
window in under 2 s of speech, and hand Tier 1 a window
built from overlapping slices. The shadow observer
therefore advances the window with the newest hop of
each frame only: the last `hop_ms` of samples, stamped
with the time of that hop. A window is then 2 s of
far-end speech at hop cadence, exactly as ADR-016
defines it. The hop length comes from the pipeline
configuration, so a pipeline configured with a
different hop advances its windows by that hop, and a
window is still 2 s of speech. Non-speech frames
contribute nothing, as in ADR-016.

Second, the pipeline has no speaker labels on its
transcript path. `push_transcript` takes text and a
timestamp only, and ASR fragments carry a speaker of
`remote`, `local`, or `unknown`. The factorial model
needs a speaker because only far-end turns can
constitute a requested action (ADR-017 item 4). The
observer maps `remote` to far, `local` to near, and
`unknown` to far, which is the safe default ADR-017
states for a feed that does not diarise. The
`push_transcript` path has no speaker at all, so it is
treated as far-end, for the same reason. The mapping
is stated in the code so a diarising ASR can replace
it without changing the observer interface.

## Decision

1. **One observer, owned by the pipeline, disabled
   only by configuration.** `ShadowObserver` lives in
   `shieldcall/runtime/shadow.py`. The pipeline
   constructs one when `PipelineConfig.shadow_enabled`
   is true (the default) and none when it is false.
   Disabling shadow changes nothing else: the
   pipeline, fusion, and serve layers behave exactly
   as before, and no thread is started. The observer
   holds the `WindowBuilder`, the `CascadeTrigger`,
   the `FactorialDiscourseModel`, the per-window
   Tier 0 LLR accumulator, the changepoint and
   requested-action latches, and the record lists.
   It has no reference to the fusion engine or the
   serve layer, so shadow state can never leak into
   a verdict except through the records the pipeline
   deliberately attaches to its events.

2. **Factorial runs beside the SDTG, on the calling
   thread.** The factorial lexicon filters are cheap
   (regular expressions over one turn), and running
   them on the calling thread keeps turn order exact:
   the factorial state attached to an event is the
   state after that event's turns, with no reordering
   a queue could introduce. Every transcript turn the
   pipeline scores with the SDTG is also fed to the
   factorial model, from both `push_transcript` and
   the ASR fragments inside `_process_frame`. The
   factorial state is attached to the pipeline event
   that carries the linguistic score for that turn,
   and the pipeline exposes the latest state as
   `last_factorial`. The SDTG remains the live language
   stream; the factorial fraud score is recorded, never
   fused. The factorial model keeps its own SDTG
   instance, as ADR-017 built it; the two SDTG states
   are independent and may disagree, and the record
   keeps both visible.

3. **The cascade proxy is replaced by the exact form.**
   The requested-action latch for a window is set
   when `requested_action_active` holds for any
   factorial turn during that window, at the
   configured threshold (default 0.5, the helper's
   default). The latched SDTG stage in HARVEST or
   PAYMENT no longer feeds the trigger. A window
   with no transcript turn has no requested action,
   which is correct: nothing was asked. The SDTG
   stage at window end is still recorded on the
   window record, beside the factorial action, so
   the proxy and the exact form can be compared
   from shadow data. `CascadeTrigger` itself is
   unchanged: it still takes a boolean, and the
   offline harness keeps using the stage proxy
   where no factorial state exists.

4. **Tier 1 runs on one background thread, and an
   outage is recorded, never hidden.** When a window
   completes, the observer decides the trigger on the
   calling thread (cheap, deterministic) and, if the
   window triggered, hands the window to a single
   daemon worker thread through a bounded queue. The
   worker scores it and fills in the window record.
   The calling thread never waits for Tier 1, so the
   10 ms cadence cannot be delayed by a slow encoder.
   The record is created at decision time with the
   Tier 1 fields empty and the most recent CSCF risk
   attached; the worker fills the score in place, so
   a record read later shows the score, and a record
   read early honestly shows none yet. The queue is
   bounded (64 windows); a full queue drops the
   oldest unstarted window and counts it, rather
   than growing without bound or blocking the call.
   If no Tier 1 scorer is configured, triggered
   windows are recorded with the reason `no scorer
   configured` and no score is invented. If scoring
   raises `Tier1UnavailableError`, the observer
   latches the outage: later windows are recorded
   with the same reason and no further load is
   attempted, so a missing backbone does not retry
   a doomed load on every window. Any other
   exception from the scorer is caught, recorded as
   `tier1_error`, and never propagates into the call
   path: shadow code must not fail a call.

5. **Tier 1 is injected, never constructed by
   default.** The pipeline constructs no scorer on
   its own. `PipelineConfig.shadow_tier1_model_path`
   names a Phase 8 model file; when set, the
   observer builds a `Phase8Tier1Scorer` lazily, on
   the first triggered window, from the local cache
   only, exactly as ADR-016 specifies. When unset,
   which is the default, no model file is read, no
   backbone is loaded, and no network is touched.
   Tests inject a scorer directly (a fake embedding
   function, as in the cascade tests) through
   `ShadowObserver.set_tier1_scorer`, so no model
   download is needed to test the wiring. The
   scorer factory, when supplied, is called at most
   once; a factory that raises latches the outage
   like any other scorer failure.

6. **Records are the only output, and they carry
   their own provenance.** A `ShadowWindowRecord`
   carries the window index and timestamps, the
   `CascadeDecision` with every reason, the window
   Tier 0 LLR, the latched changepoint and
   requested-action flags, the factorial action and
   hazard at window end, the SDTG stage at window
   end, the most recent CSCF risk (score and tier,
   or None when no fusion has run yet), the Tier 1
   score and calibration flag when scored, and the
   unavailability reason when not scored. Records
   are appended in window order and are never
   rewritten except for the Tier 1 fields the worker
   fills in. `drain_shadow` waits, up to a timeout,
   for the worker to finish everything queued, so
   tests and offline runs can read complete records
   without sleeping. `reset` starts a new call:
   it clears the records, the window builder, the
   trigger, the factorial model, and the latches,
   bumps a generation counter so queued windows
   from the previous call are dropped rather than
   scored into the new call's records, and keeps
   the worker thread alive. `close` stops the
   worker. The pipeline calls `reset` from its own
   reset and exposes `drain_shadow` and
   `shadow_windows` for callers that need complete
   shadow data.

7. **Audit seeds stay caller-controlled.** The
   cascade audit draws come from the
   `CascadeConfig` the observer is built with.
   `PipelineConfig.shadow_audit_seed` defaults to
   None, which seeds from operating system entropy,
   the only correct choice for production (ADR-016
   item 3). Fixed seeds are for tests and
   reproducible offline runs only, and the pipeline
   configuration documents that.

8. **No new dependency, no serve change.** The
   observer uses the standard library plus NumPy,
   like the cascade and factorial modules. The
   serve layer is untouched in this slice: sessions
   already consume pipeline events, and the shadow
   record rides on the event that completes each
   window. Exposing shadow records on a serve
   endpoint, and the 7-day shadow gate (G7), are
   follow-ups. Time-to-warning measured end to end
   (ADR-016 follow-up (g)) is its own slice: it
   needs the wired cascade plus ASR and rendering
   timing, and nothing here claims it.

## Consequences

The cascade and the factorial model now run on live
pipeline traffic in shadow: every 2 s of far-end
speech produces a trigger decision beside the CSCF
verdict that was in force, every transcript turn
produces a factorial state beside the SDTG score,
and triggered windows are scored by Tier 1 off the
call thread where a scorer is configured. What
cannot yet be claimed: any trigger rate on real
traffic (tau0 is still the neutral default, to be
set from benign shadow traffic), any windowed Tier 1
accuracy, any lead time on real calls, or any effect
on live verdicts. The 11.78% Phase 8 EER remains a
whole-clip offline result and is not claimed for
2 s windows. The SDTG proxy comparison the records
enable is a measurement, not a decision, until the
shadow gate says otherwise.

## Verification

Tests in `tests/test_shadow_wiring.py` use
synthetic audio and transcript turns with known
ground truth, following the cascade and factorial
test pattern. They cover: shadow disabled leaves
the pipeline byte-identical in behaviour (no
records, no factorial state, no thread); the
factorial state rides on events and matches a
standalone factorial model fed the same turns;
the speaker mapping (local turns never count as
requested actions, unknown turns default to
far-end); the requested-action latch uses the
factorial exact form, not the SDTG stage (a
payment turn that the SDTG leaves at an early
stage still latches, and a HARVEST stage with no
harmful action does not); windows complete on
speech only and carry the window Tier 0 LLR and
the latest CSCF risk; triggered windows are
scored on the background thread without delaying
the calling thread, and `drain_shadow` makes the
scores visible; a missing scorer records the
reason and invents no score; a failing scorer
latches the outage and never raises into the
call path; reset drops stale queued windows and
starts the audit stream fresh; a full queue
drops the oldest unstarted window and counts it.
The full core suite must stay green. No licensed
ASVspoof data is touched, and no live call is
scored.
