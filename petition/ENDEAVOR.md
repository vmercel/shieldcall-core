# Proposed endeavor (≤180 words)

I will invent, evaluate, and transfer into U.S. telephone infrastructure a
telephony-native closed-loop defense against AI-enabled voice fraud. The
system treats a live call as a dual-stream, partially observed process: a
linguistic trajectory of social-engineering stages under automatic speech
recognition error, and an acoustic production process that may be bona fide,
fully synthetic, or switched mid-call at a credential-harvest boundary. It
fuses those streams under the operational rule that a call is a threat if
either stream is hostile, with distribution-free uncertainty so the system
can abstain. A privacy-preserving agent — which never observes waveforms or
transcripts — spends a legally constrained interruption budget as sequential
experiments (challenge, warn, escalate, adapt) and refuses to waste a
liveness challenge on a live social engineer. The detector sits beside the
media path, not on it: RTP never hairpins, and detector failure cannot drop
a U.S. telephone call. A coverage-debt controller enrolls new synthesizer
families without a full retrain. The research outputs are (1) an open
telephony evaluation protocol for disjunctive and mid-call threats, (2)
peer-reviewed measurements against published neural baselines and independent
corpora, including negative results, and (3) a sidecar that U.S. CPaaS
providers, contact centres, and carriers can trial. The public beneficiaries
are U.S. consumers, especially older adults targeted by government- and
family-impersonation calls, in a setting where STIR/SHAKEN authenticates the
number and not the voice or the script.
