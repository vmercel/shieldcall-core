# ADR-019: Time-to-warning measured end to end, in process (TDD section 10.4)

## Status

Accepted (2026-11-28). Implements ADR-016 follow-up (g)
and the TDD section 10.4 budget as a measurement. It
changes no live decision path: CSCF stays the production
fusion, no threshold moves, and no serve or app change.
What it adds is a harness that measures, on synthetic
calls with known evidence, how long the pipeline itself
takes from the end of discriminative evidence to a
HIGH_RISK verdict, reported per stage so a regression
is attributable. It does not claim the 2 s p95 budget
is met on real devices or networks. That claim needs
real ASR lag, real network time, and real rendering
time, none of which exist in this slice.

## Context

TDD section 10.4 defines the objective exactly: at most
2 s at p95, measured from the moment discriminative
evidence ends (the last frame of a cloned utterance, or
the end of a harmful request) to the moment the warning
is rendered. Figure 21 budgets the critical path at
1.2 s: ingest, decode and jitter at most 100 ms,
acoustic evidence accumulation at most 700 ms, Tier 1
inference at most 200 ms on CPU, stable-prefix ASR and
discourse update at most 700 ms in parallel, and fusion,
agent, signed event and render at most 200 ms. The
remaining 0.8 s is headroom for network variance and
ASR lag on hosted recognizers. The same section
requires time-to-warning to be reported separately for
ASR lag, model inference, evidence onset, and
rendering. It also requires a provisional SUSPICIOUS
cue early when semantic evidence needs longer than the
budget, upgraded later rather than held until certainty.

Until now nothing measured any of this. Per-frame cost
(section 10.1, 4.69 ms p95) says nothing about how
quickly a user is warned, and the cascade, factorial,
and shadow wiring slices (ADR-016, ADR-017, ADR-018)
each stated that time-to-warning was still unmeasured.
The wired pipeline now has everything the in-process
part of the measurement needs: chunked audio in,
scheduled or pushed transcript turns, CSCF verdicts on
pipeline events, and shadow windows with background
Tier 1 scores. The missing pieces are outside the
process: an ingest adapter's network and decode time,
a hosted ASR provider's lag, and the app's render time.
Those stay unmeasured here, reported as absent, never
as zero.

Two facts shape the design, and both are recorded
rather than smoothed over.

First, a laboratory harness feeds audio faster than
real time. Wall-clock time from feeding the evidence
chunk to receiving the verdict is then only the
processing time of the chunks fed after the evidence,
not the time a live caller waits while the audio
arrives. The measurement therefore reports two
quantities that must not be conflated. Accumulation
is audio time: the frame timestamp of the first
HIGH_RISK event minus the evidence end time, which
is how much post-evidence audio the pipeline consumed
before warning. In a live call that audio arrives at
1x speed, so accumulation audio seconds are also wall
seconds of waiting. Processing is wall time: how long
the deciding push took, from ingest of its chunk to
the verdict being returned. The estimated live warning
time is the sum of accumulation, processing, ASR lag,
and render time, and it is only an estimate of the
full budget when ASR lag and render time are supplied
from real measurements. When they are not supplied,
the estimate is labelled measured-only and no budget
verdict is claimed for the full 2 s objective.

Second, a warning can precede the declared evidence
end. If earlier evidence already pushes the pipeline
to HIGH_RISK, the first warning frame timestamp falls
before the evidence end time, and accumulation would
go negative. That is a mis-specified evidence end,
not a faster warning: the evidence that sufficed ended
earlier. The harness reports it plainly as
warned-before-evidence-end, clamps accumulation at
zero, and keeps the first warning. It never reports a
negative time-to-warning.

## Decision

1. **One harness module, no pipeline change.** The
   measurement lives in
   `shieldcall/eval/time_to_warning.py`. It drives an
   existing `ShieldCallPipeline` through its public
   `push_audio` and `push_transcript` calls, in
   fixed-size chunks, and reads verdicts off the
   returned `PipelineEvent` objects. It constructs no
   pipeline, loads no model, and touches no serve
   route, so it cannot change a live verdict. A
   caller supplies the pipeline (or a factory that
   builds one per run), the audio, the evidence end
   time in audio seconds, and optionally transcript
   turns with their audio times. Turns may also arrive
   through the pipeline's own ASR, as in the
   `ScheduledTranscriptASR` eval pattern; in that case
   the turns list is empty and the discourse update
   time is inside the chunk processing time, which the
   report says.

2. **The clock is injectable, and defaults to the
   monotonic clock.** Wall times use a caller-supplied
   clock callable, defaulting to `time.monotonic`.
   A fake clock makes the arithmetic deterministic in
   tests. Real runs report lab-machine wall times:
   the shared engineering VM, not reference hardware.
   The report carries no hardware claim, and the
   script names the machine it ran on only as the
   local host. Per-chunk processing times are kept in
   full, not averaged away: the first chunk carries
   warm-up cost, and the report shows the first chunk
   separately from the steady-state p95 over the
   remaining chunks, so warm-up can neither hide nor
   inflate the steady figure.

3. **Warning and provisional cue are defined on
   CSCF tiers, from the evidence chunk onward.** The
   warning is the first event with tier HIGH_RISK
   returned by the chunk that contains the evidence
   end time, or any later chunk. The provisional cue
   is the first event with tier SUSPICIOUS or
   HIGH_RISK from the same point onward, matching
   the TDD requirement to show a cue early and
   upgrade it. Events from earlier chunks are
   examined only to set the warned-before-evidence-end
   flag. A run that never reaches HIGH_RISK reports
   warned as false with accumulation absent, never
   with a zero. ABSTAIN events are not warnings and
   not cues.

4. **Stages are reported separately, in TDD 10.4
   terms.** Each run reports: accumulation in audio
   seconds; deciding-chunk processing in wall
   seconds; turn (discourse update) processing where
   turns were pushed; Tier 1 latency per scored shadow
   window (item 5); and ASR lag, render time, and
   network time as caller-supplied values that default
   to absent. The estimated live warning time is
   accumulation plus deciding-chunk processing plus
   turn processing plus any supplied ASR, render, and
   network times. Budget checks are per component
   against Figure 21: accumulation at most 700 ms,
   Tier 1 latency at most 200 ms, deciding-chunk
   processing at most 200 ms. The full 2 s p95 check
   runs only over repeated runs (item 6) and only
   when no stage is absent; otherwise the verdict is
   incomplete, stated as such.

5. **Shadow windows gain monotonic timestamps, and
   nothing else changes in the shadow module.**
   `ShadowWindowRecord` gains two additive fields,
   `decision_monotonic_sec` set when the record is
   created at window completion, and
   `tier1_monotonic_sec` set when the background
   worker fills the Tier 1 score. Both default to
   None: a record for a window that was not
   triggered, or was never scored, keeps None for
   the Tier 1 stamp, and no latency is computed
   from it. The harness drains the shadow queue
   after feeding, reads the stamps, and reports one
   Tier 1 latency per scored window. A latency is
   never computed from a record whose score arrived
   from a previous call: the generation guard in
   ADR-018 already prevents that fill, and the
   stamp is set in the same locked section as the
   score.

6. **Repeated runs give the p95; one run does not.**
   `summarize_runs` takes the measurements from
   repeated runs of the same call and reports p50
   and p95 (nearest-rank, stated in the code) for
   accumulation, deciding-chunk processing, Tier 1
   latency, and the estimated live warning time. The
   2 s budget verdict compares the p95 of the
   estimated live warning time against 2.0 s, and
   only when every run in the summary has all stages
   supplied. A single run, however fast, never
   yields a p95 claim.

7. **A runnable script, synthetic calls only.** 
   `scripts/measure_time_to_warning.py` builds one
   synthetic scam call (gated-burst speech audio the
   VAD accepts, plus scheduled scam turns) and one
   benign call, runs each several times through fresh
   pipelines, and prints a JSON report with the
   per-component figures and the budget verdicts. It
   uses no licensed data, no network, no ASR
   provider, and no model file, so Tier 1 latency is
   absent in its output unless the operator supplies
   a scorer, which the script does not. Its output
   is a lab measurement, labelled as such in the JSON.

## Consequences

The in-process part of time-to-warning is now
measurable and attributable: accumulation, processing,
discourse update, and Tier 1 latency are separate
numbers with separate budgets, and ASR lag, render,
and network time are visible absences rather than
silent zeros. What cannot yet be claimed: the 2 s
p95 budget on real devices or networks, any Tier 1
latency on a real backbone (no scorer is configured
by default), any ingest-adapter latency (adapters
are outside this harness), or any effect on live
verdicts. The per-chunk times also expose the
pipeline's real-time factor on the machine that ran
them; if the steady-state chunk cost exceeds the
audio it carries, that is a finding for the G6
latency gate, not a number to average away.

## Verification

Tests in `tests/test_time_to_warning.py` use
synthetic audio and transcript turns with known
ground truth, following the cascade, factorial, and
shadow test pattern. They cover: a scam call warns,
with accumulation at or after the evidence end and
inside the 700 ms accumulation budget on the lab
machine; a benign call never warns and reports
accumulation absent; a warning before the declared
evidence end is flagged and never yields a negative
time; the provisional cue is reported separately
from the warning; chunk processing times are
recorded per chunk with a fake clock, and the
first chunk is excluded from the steady-state p95;
Tier 1 latency is computed only from stamped,
scored windows, and stays absent when no scorer is
configured; percentile and budget logic on
hand-computed cases, including the incomplete
verdict when ASR or render time is absent; repeated
runs summarise to a p95 that a single run cannot
claim. The full core suite must stay green. No
licensed ASVspoof data is touched, no model is
downloaded, and no live call is scored.

Follow-ups, not in this slice: serve-layer stamps
(ingest arrival to last-decision response), real ASR
lag measured against a hosted recognizer, app render
timing on a device, a reference-hardware run for the
G6 gate, tau0 from benign shadow traffic, windowed
Tier 1 re-measurement, and the 7-day shadow gate G7.
