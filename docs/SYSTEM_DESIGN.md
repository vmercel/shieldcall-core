# ShieldCall system design

**Status:** designed and implemented as in-process contracts (`shieldcall.runtime`). Not a production deployment, not a carrier certification.

The detector and the agent (v0.5) answer *what to compute*. This document answers *how that computation sits on a telephone path* so that scalability, reliability, availability, and cost are explicit.

Figures: [deployment](figures/deployment_sidecar.png) · [reliability](figures/reliability_failopen.png) · [capacity](figures/capacity_cost.png)

## Requirements

### Functional
- Score 8 kHz audio and optional transcript fragments on a shared timeline.
- Fuse under the operational OR-label (scam language **or** vocoded voice).
- Choose at most one liveness challenge per call; warn on human social engineering.
- Emit an audit trace of every decision.

### Non-functional (design targets)

| NFR | Target | How it is met |
|-----|--------|----------------|
| **Call availability** | Must not be reduced by ShieldCall | Sidecar: RTP does not traverse the worker |
| **Detector availability** | Best-effort | Shed new calls rather than block media |
| **Latency** | p95 frame time ≤ 8 ms on a 10 ms hop | Measured; ready probe goes false if EWMA exceeds SLO |
| **Scalability** | Horizontal by `call_id` | One session per call; affinity pin |
| **Reliability** | Worker crash ≠ dropped call | Ephemeral session state; media path independent |
| **Cost** | CPU only; no GPU in the v0.6 path | `plan_capacity(ms_per_frame)` |
| **Privacy** | No waveform required at the agent | Sufficient statistics only (ADR-002) |
| **Observability** | Health, breaker state, shed count, EWMA ms/frame | `SidecarRuntime.health()` |

### Constraints
- Research prototype; Python/numpy scorer.
- No production ASR in-repo (pluggable `ASRBridge`).
- Channel twin is **eval-only** and must be off in live workers.

## High-level architecture

```
Caller ──SIP/RTP── SBC / CPaaS ──SIP/RTP── Callee     ← data plane (always on)
                       │
                       │ media fork (8 kHz PCM copy)
                       ▼
              Call-affinity ingress
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Worker       Worker       Worker               ← compute plane
       session      session      session
       (pipeline + agent, in memory)
          │
          ├─ control: prototypes, YAML, SLO
          └─ audit: Decision records → operator
```

Code mapping:

| Box | Module |
|-----|--------|
| Session | `runtime/session.py` `CallSession` |
| Worker | `runtime/runtime.py` `SidecarRuntime` |
| Sensor | `pipeline.py` |
| Agent | `agent/agent.py` |
| ASR gate | `runtime/asr_gate.py` |
| Capacity | `runtime/capacity.py` |
| SLO | `runtime/slo.py` |

## Scaling

**Unit of scale is the call**, not the HTTP request. Belief, CUSUM, discourse stage, and challenge budget are per-call. Sharing a pipeline across callers is a correctness bug; isolation is tested.

Admission:

```
if active_sessions >= max_calls:
    return shed session   # MONITOR only, zero CPU
```

Readiness is false when free capacity < 10% of `max_calls` or when EWMA frame time exceeds 8 ms, so a load balancer drains the worker instead of overloading it.

**Formula** (implemented, unit-tested):

```
calls_per_core = (hop_ms / ms_per_frame) * utilization
```

with `hop_ms = 10`, `utilization = 0.70`.

On the evaluation host, `python scripts/run_capacity.py` measured about **6.3 ms/frame** (realtime × ~1.6). That is **~1.1 full-rate calls per core**. 1 000 concurrent calls need on the order of 900 cores plus spare. That number is why VAD skip, `fuse_every_n_frames`, and *not* running the channel twin live are cost features, not style.

Vertical scaling (bigger box) does not remove affinity: two calls on one process still need two sessions.

## Reliability and availability

The product that must stay up is the **telephone call**. ShieldCall is an analytics sidecar.

| Failure | Call | Detector | Mitigation in code |
|---------|------|----------|--------------------|
| Worker at capacity | continues | new call undefended | shed session, `last_action=MONITOR` |
| Worker process dies | continues | that call's belief lost | pin new calls to other workers |
| ASR timeouts | continues | fraud score freezes | `CircuitBreaker` opens; acoustic remains |
| Frame time > 8 ms | continues | worker unready | `health().ready is False` |
| Control-plane config missing | continues | do not start ready | fail to listen, do not hairpin |

RPO for detector state: the rest of the current call. We do not replicate belief. RTO for the *call* is zero (SBC). RTO for *new* detector coverage is "spawn a worker."

Fail-open vs fail-closed is **split**:

- **Media:** fail-open (never hold RTP).
- **Actuation:** fail-closed (when uncertain or shed, do not challenge or tear down).

That matches the agent policy: a human social engineer is warned, not nonce-challenged.

## Cost-efficiency

No GPU is required for the residual + HMM + rule-fusion path. The bill is:

1. **vCPU-hours** of workers (dominant if every call is fully scored).
2. **ASR** (optional, out of process, often the larger bill if enabled).
3. **Audit storage** (decisions, not waveforms).

Illustrative CPU only, not a quote: at $0.04 / vCPU-hour, 1 000 concurrent full-rate calls, N+1 spare, ~$29k / month. Operators who cannot pay that must *sample* (score 1-in-N calls), skip silence, or fuse less often. A neural vocoder detector would add accelerator rent.

Bandwidth: 8 kHz 16-bit PCM is 16 kB/s per forked call (~1.1 kbps is *not* the number; 128 kbit/s). Cheap compared to CPU.

## Security and privacy

- Agent never sees waveform or full transcript (ADR-002).
- Sidecar can run in the same trust domain as the SBC; it does not need internet egress except audit export.
- Challenge nonce stays inside `Toolbelt`; do not log it.
- PII in transcripts is the ASR vendor's problem; this repo injects text in experiments.

Compliance (TCPA, recording consent, GDPR) is **out of scope** for the prototype and is not claimed.

## Observability

`SidecarRuntime.health()` returns liveness, readiness, active calls, shed count, breaker state, EWMA ms/frame. Decision traces are the audit log (`DefenseAgent.trace`). There is no Prometheus exporter in v0.6; the struct is the contract.

## What is implemented vs not

| Item | v0.6 |
|------|------|
| Session isolation | yes, tested |
| Shed / fail-open | yes, tested |
| ASR circuit breaker | yes, tested |
| Capacity formula | yes, tested |
| Measured ms/frame | `scripts/run_capacity.py` |
| Kubernetes / Envoy / SIPREC | **not** in this repo |
| Multi-region | **not** |
| Live ASR | **not** |
| Carrier trial | **not** |

## Reproduce

```bash
pytest -q tests/test_runtime.py
python scripts/run_capacity.py
python scripts/render_diagrams.py
```
