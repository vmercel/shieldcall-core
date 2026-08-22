# ADR-003: Sidecar runtime (scale unit = call, fail-open on media)

## Status

Accepted

## Context

v0.5 implemented sensors and a belief-state agent as a library. That is not a system. A telephone product has to state how it scales, what happens when a worker dies, what it costs, and whether a detector outage drops the call. An LLM gateway or a GPU anti-spoofing service would answer those questions differently (and more expensively). We need an architecture that matches the science: CPU-only scoring, per-call state, actuation that is allowed to be silent.

## Decision

1. **Sidecar, not hairpin.** RTP stays on the session border controller. ShieldCall consumes a forked 8 kHz copy. Detector downtime must not reduce call availability.
2. **Scale unit is the call.** One `CallSession` owns one `ShieldCallPipeline` and one `DefenseAgent`. Workers pin `call_id` (affinity). No shared in-memory belief across calls.
3. **Shed is success.** At `max_calls`, `open_call` returns a shed session that spends no CPU and emits only MONITOR (fail-open on the call, fail-closed on actuation).
4. **ASR is a breaker-guarded dependency.** Acoustic path continues when the breaker is open.
5. **Cost is vCPU, not GPU.** Capacity is `calls_per_core = (hop_ms / ms_per_frame) * utilization`, measured on the host, not assumed.

Implemented in `shieldcall/runtime/`. Tests in `tests/test_runtime.py`. This is not a Kubernetes manifest and not a carrier integration.

## Alternatives considered

**In-line media mixer.** Rejected: makes ShieldCall a five-nines component. We do not have that evidence.

**One shared pipeline for all calls.** Rejected: CUSUM, discourse state, and belief would leak across callers.

**GPU batching of residual nets.** Rejected for v0.6: the current scorer is CPU numpy; a GPU encoder would change the cost model and is not measured.

**Replicate in-call belief.** Rejected: RPO of detector state is "lose the rest of this call's belief." The telephone call itself is unaffected.

## Consequences

### Positive
- NFRs are code and tests, not slides.
- Operator can size cores from a measured millisecond number.
- Privacy: waveform never has to be stored; the agent still sees only scores.

### Negative
- Full-rate processing is about one call per core on the evaluation host (~6.3 ms/frame vs a 10 ms hop). Horizontal scale is mandatory.
- Shed calls are undefended for the life of that `call_id`.
- No multi-region story yet; single-worker-pool design.
