# ADR-017: Factorial discourse model (stage, tactic, requested action)

## Status

Accepted (2026-11-26). Implements TDD v2.0 section 7.3
(factorial discourse model) and section 7.4 (benign-mention
guard), with the section 7.5 mandatory TF-IDF prefix baseline
for lead-time comparison. This slice delivers the model, the
guard, the hazard and fraud score, and the offline lead-time
evaluator, all in shadow. It changes no live decision path:
the pipeline, the fusion engine, and the cascade keep using
the SDTG stage tracker, CSCF stays the production fusion, and
no sidecar or app threshold moves. Pipeline wiring, learned
transitions, and real-call evaluation are follow-ups listed
in item 9.

## Context

One stage index conflates three things a warning must
separate: how far along a script the call has got, which
persuasion tactics are active, and what the caller is asking
the user to do. A single alarming word is not a script, and
a script that has reached the payment ask is not the same
as one that is still establishing authority. TDD section 7.3
therefore tracks a factorial latent state per turn,
z = (s, u, a): the SDTG stage s, a vector u of active
persuasion tactics, and the requested action a.

Two gaps in the current code motivate this slice. First,
the language stream scores a call with the SDTG alone, so
it cannot tell a credential request from a payment request,
and it cannot say whether a harmful request is coming
before it arrives. Second, the cascade (ADR-016) triggers
on a latched SDTG stage in HARVEST or PAYMENT as a proxy
for the requested-action state, because the exact form did
not exist. This slice builds the exact form as a standalone
module; replacing the proxy in the cascade is a follow-up,
not part of this slice, so the cascade interface does not
change here.

One defect in the TDD text is recorded rather than silently
repaired. The action set A in section 7.3 is cut off at the
page edge after "remote acc". The set is completed as remote
access, which is the only reading consistent with the rest
of the document: Table 16 in section 9.5 gives guidance for
a "remote-access request", and the tech-support pattern group
in the existing scorer already treats remote access as a
distinct request. The completed set has six actions, and
the completion is stated here so a reader of the TDD alone
can see exactly what was assumed.

## Decision

1. **Factorial state, factorised filtering.** The model
   tracks (stage, tactics, action) per turn. TDD eq. 15
   filters the joint state exactly; this slice filters the
   three factors independently and combines their outputs,
   which is the approximation TDD itself calls
   approximate factorisation. The stage factor is the
   existing SDTG forward recursion, reused rather than
   reimplemented, so the stage beliefs, the frozen
   lexicon, and the LEXICON_LOCK stay the auditable
   floor. The tactic and action factors are new Bayes
   filters over their own lexicons, defined in items 3
   and 4. No factor reads another factor's posterior
   during filtering; the hazard in item 6 is the only
   place the factors meet. A joint filter with learned
   cross-factor transitions is a follow-up that needs
   labelled transcripts, which do not exist in the repo.

2. **Stage with an explicit duration floor.** TDD
   specifies a hidden semi-Markov chain, so a stage
   cannot flicker on one turn. The SDTG belief itself
   is a plain HMM and can change its argmax on a single
   turn, so this slice adds a dwell floor on top of it:
   the committed stage changes only after the same new
   stage has been the SDTG argmax for
   `min_stage_turns` consecutive turns (default 2). The
   committed stage, the SDTG argmax, and the full stage
   posterior are all reported, so the floor is visible
   rather than hidden. This is a dwell floor, not a
   fitted duration distribution: no duration parameters
   are learned, and none are claimed. The floor applies
   to the reported stage only; the SDTG belief, path
   score, depth, and visited stages are reported
   exactly as the SDTG computes them.

3. **Six tactics, independent Bernoulli chains with
   persistence.** The tactic set is fixed by TDD:
   authority, urgency, fear or threat, reward, liking
   or rapport, and secrecy or isolation, aligned with
   the PsyScam and EARS taxonomies. Each tactic has its
   own binary filter: it persists with probability 0.85
   when active and enters with probability 0.10 when
   inactive, and a turn that hits the tactic lexicon is
   seven to nine times as likely under the active state
   as under the inactive state (hit likelihoods 0.70
   and 0.08). These are prior constants, stated in the
   code and in the config, not fitted values: no
   labelled tactic data exists to fit them from. Each
   tactic lexicon is a small explicit pattern list, and
   the tactic lexicon set carries its own SHA-256 lock
   hash, computed the same way as LEXICON_LOCK, so any
   later edit to a tactic pattern is detectable. A
   tactic stays active across turns with no new
   evidence only through the persistence rule, decaying
   toward the prior; it is never latched forever.

4. **Requested action, far-end speaker only, six
   actions.** The action set is: none, personal data,
   one-time code or credential, payment or gift card,
   wire or crypto transfer, and remote access. Each
   action except none has an explicit pattern list
   drawn from the HARVEST and PAYMENT stage patterns
   and the existing pattern groups, plus remote-access
   patterns from the tech-support group; the action
   lexicon set carries its own SHA-256 lock hash. The
   action belief is an emission score per turn,
   smoothed toward the previous belief with a stay
   weight of 0.6, so one turn of evidence moves the
   belief without replacing it outright. The harmful
   set is exactly TDD eq. 16: all actions except none
   and personal data. Personal data is therefore not
   harmful in this model, even though asking for it is
   serious: the TDD definition is followed exactly, and
   the exclusion is stated here rather than corrected
   silently. A turn from the near-end speaker (the
   protected user) never updates the action belief and
   reports action none for that turn; the user's words
   still update stage and tactic evidence, per TDD
   section 7.4. The caller must supply the speaker;
   when no speaker is given, the turn is treated as
   far-end, which is the only safe default for a
   transcript feed that does not diarise, and the
   assumption is stated in the code.

5. **Benign-mention guard, all three features.**
   Benign calls talk about fraud constantly, and three
   features keep them from looking like scams. (i)
   Speaker role: item 4, the user's words never
   constitute a request. (ii) Reported speech and
   negation: a far-end turn matching a reported-speech
   pattern ("they told me to buy gift cards", "the
   caller said") or a negation-of-request pattern ("we
   will never ask for your code", "no legitimate
   organisation asks for payment this way") has its
   action emission blended 75 percent of the way toward
   uniform, and the guard is recorded on the state.
   Down-weighting is not deletion: a guarded turn can
   still move the action belief if its evidence is
   strong, and the stage and tactic updates are not
   down-weighted. (iii) Inbound-direction prior: when
   the immediately preceding user turn requested an
   action, a far-end turn requesting the same action is
   down-weighted the same way, because the user
   initiated it. The guard patterns are explicit lists
   in the module, not a language model judgement, and
   they are tested against the exact example sentences
   TDD gives.

6. **Hazard and fraud score, TDD eq. 16.** The primary
   output is the hazard eta: the probability that a
   harmful action is requested within the next k turns
   (default k = 3), computed as
   1 - (1 - p_harm)^k where p_harm is the current
   posterior probability that the requested action is
   harmful. The independence across future turns is a
   stated approximation: the model does not predict how
   the action distribution will evolve, it projects the
   current harmful probability forward. The fraud score
   passed to fusion is f = max(eta, p_harm), which is
   eta for every k >= 1, so f is the hazard. A harmful
   request already made therefore always counts, and a
   merely probable harmful request counts through the
   horizon. The hazard is action based only: stage
   depth and tactic evidence reach fusion through the
   separate fields on the state, not folded into f, so
   a threatening stage with no current request scores
   low on f and high on depth, and the distinction is
   preserved for the caller.

7. **Lead time is the target, measured against the
   mandatory baseline.** For a call whose first harmful
   request occurs at time th, and whose first warning
   at a fixed false-alert budget occurs at tw, the lead
   time is L = th - tw in seconds, positive when the
   warning comes first. `evaluate_lead_time` reports
   the distribution of L over scam calls, the number
   missed (no warning at or before the harmful
   request), and false alerts per 100 benign calls, at
   a stated threshold. `compare_lead_time` sets the
   threshold for each model at the same false-alert
   budget: it scans the observed scores from the lowest
   upward and picks the most sensitive threshold whose
   benign false-alert rate is at or below the budget, so
   the comparison is at equal false-alert rate as TDD
   Table 31 requires.
   The baseline is a TF-IDF prefix scorer with a
   logistic regression fitted by gradient descent in
   NumPy, trained on whole-call texts from the same
   split and scored on cumulative prefixes, per TDD
   section 7.5. The baseline is deliberately plain: no
   pretrained encoder, no language model, fixed
   training settings, deterministic. The kill
   criterion is fixed before any real-call measurement:
   the factorial model continues only if its median
   lead time beats the baseline at equal false-alert
   rate; otherwise the shipped language scoring stays
   the TF-IDF prefix score plus the SDTG stage trace.
   On the synthetic dialogues in the test suite both
   models separate easily, which proves the harness
   runs, not that either model wins on real calls.

8. **Shadow only, no learned encoder, no language
   model.** The module has no reference to the
   pipeline, the fusion engine, the serve layer, or
   the cascade. Nothing in this slice wires it into a
   live path, and the SDTG proxy in the cascade is
   left in place. The module exposes
   `requested_action_active`, which reports whether
   the current action is harmful at a stated
   probability threshold, so a later slice can replace
   the cascade proxy without changing the cascade
   interface. TDD blends the lexicon floor with a
   small learned text encoder, hash pinned; no such
   encoder exists in the repo and none is invented
   here. The emissions are the hash-locked lexicons
   only, and the ADR states that plainly. No large
   language model runs in the decision path, in this
   slice or any planned one.

9. **No new dependency.** The module uses NumPy only,
   matching the cascade module. The TF-IDF baseline
   implements its own tokenisation, idf weighting, and
   logistic regression in NumPy rather than depending
   on scikit-learn, so the language stream stays
   importable wherever NumPy is present.

10. **Follow-ups, in order.** (a) Pipeline shadow
    wiring: run the factorial model beside the SDTG in
    the scorer, recording its state beside CSCF verdicts
    without changing them. (b) Cascade proxy
    replacement: feed `requested_action_active` into
    the cascade trigger in place of the latched SDTG
    stage. (c) Stable-prefix gating and PII masking
    (TDD sections 7.1): consume only the stable ASR
    prefix, mask digit and card-like strings before
    scoring. (d) Learned transitions, durations, and
    the small text encoder, fitted on labelled
    transcripts with the lexicon floor held fixed.
    (e) Real-call evaluation on consented and public
    corpora, with the lead-time kill criterion applied
    for real. (f) Calibration of the fraud score
    (ADR-005 maps) before any live use. Nothing in this
    slice pre-empts any of these, and none of them is
    claimed done.

## Consequences

The language stream can now be exercised offline as a
factorial state: stage, tactics, and requested action
are reported per turn, the hazard anticipates a harmful
request by up to k turns, benign mentions are guarded
rather than scored as requests, and lead time can be
measured against the TF-IDF baseline at a fixed
false-alert budget. What cannot yet be claimed: any lead
time on real calls, any tactic or action accuracy on
real transcripts, any calibration of the hazard, or any
effect on live verdicts. The 9.5 percent clean-text EER
in TDD section 7.5 remains a whole-dialogue offline
result on LLM-generated dialogues, and the factorial
model inherits that caveat in full.

## Verification

Tests in `tests/test_factorial.py` use synthetic turns
with known ground truth, following the stage 2a, 2b, and
cascade test pattern. They cover: each harmful action
classified from a far-end turn, and personal data
classified as not harmful; near-end requests never
counting as actions; reported-speech and negation
down-weighting on the exact TDD example sentences;
user-initiated suppression; the stage dwell floor
(one alarming turn does not move the committed stage,
two consecutive turns do); tactic persistence and
decay; hazard arithmetic including the horizon and the
max with the current harmful probability; lead-time
cases with hand-computed values including a missed
warning and a late warning; false-alert counting per
100 benign calls; the TF-IDF baseline fitting and
prefix scoring on a toy split; the equal-budget
threshold comparison; and the lexicon lock hashes.
The full core suite must stay green. No licensed
ASVspoof data is touched, and no live call is scored.
