# Hosted inference design: ShieldCall Core detector as a service

Status: **design doc** (P2-4). Nothing here is deployed; there is no
production traffic, no vendor quotes, and no customer commitments.
All prices are labelled *illustrative*. Numbers grounded in the repo
are marked with their source; proposed targets are marked
`(target)`, measurements `(measured)`.

## 1. What exists today

`shieldcall/serve/http_app.py` is a FastAPI worker around
`SidecarRuntime` (`shieldcall/runtime/`). Current properties that the
design keeps:

- **In-memory session state.** One `CallSession` per `call_id`, held in
  the worker process. Audio is processed per 10 ms hop and is not
  persisted; only per-call decision traces and score history live in
  the session.
- **Recommend-only, fail-open.** The detector never hangs up, never
  joins the carrier media path, and when the runtime is shedding or a
  call is unknown the API returns a MONITOR verdict rather than an
  error (`application/detector.py`).
- **Circuit breaker on ASR.** Streaming ASR runs on a background
  worker thread (`runtime/asr_gate.py`); after 5 consecutive failures
  the breaker opens for 30 s and the pipeline continues on the
  acoustic path alone (`runtime/breaker.py`, constants in
  `runtime/slo.py`).
- **Liveness/readiness probes.** `GET /health` reports
  `active_calls`, `max_calls`, `shed_total`, `asr_breaker`,
  `ms_per_frame_ewma`; `GET /ready` returns 503 when the worker will
  not accept a new call (`runtime/health.py`, `application/detector.py`).
- **Bearer token auth** on every `/v1/` route and on the websocket
  (websocket rejects unauthenticated upgrades with close code 4401).

## 2. Authenticated endpoint (production hardening)

Gap found while writing this doc: `_check_token` in `http_app.py`
**fails open when `SHIELDCALL_SIDECAR_TOKEN` is unset** — any
deployment that forgets the variable ships unauthenticated. The
production contract is:

1. **Fail closed at startup.** If the token variable is unset in a
   production config profile, `create_app` must raise instead of
   serving. (Code change for P2-5.)
2. **Rotatable credentials.** Support a small list of active token IDs
   (e.g. `SHIELDCALL_SIDECAR_TOKENS`, comma-separated `id:value`) so a
   new token can be rolled out with a grace window before the old one
   is revoked; log which token ID authenticated each request.
3. **Per-client quota.** Reuse the `consume_ai_quota` RPC pattern
   already live on the Supabase edge functions (P0-3): a per-client,
   per-minute budget on ingest routes, 429 with `Retry-After` past the
   budget, counters exported to metrics. Never fail open on the quota
   store: if the store is unreachable, keep serving audio but freeze
   new-call opens.
4. **App-origin traffic** may use Supabase Auth JWTs (validated against
   the project JWKS) instead of the static bearer token; the static
   token remains for SBC/sidecar integrations.
5. **CORS.** `SHIELDCALL_CORS` defaults to `*` today. Production must
   pin it to the app origin; wildcard CORS stays dev-only.
6. TLS terminates at the load balancer; the worker binds to
   localhost/loopback only.

## 3. Latency SLO

### 3.1 Measured components (sources cited)

Frame processing, measured 2026-09-14 on the engineering VM with the
production-equivalent pipeline config (`channel=None`,
`use_conformal=True`, `fuse_every_n_frames=5`, PassthroughASR), 500
hops of 10 ms audio:

| percentile | ms per 10 ms frame |
|---|---|
| p50 | 4.334 |
| p95 | 4.691 |
| p99 | 5.418 |
| max | 6.318 |

The in-repo frame budget is 8 ms p95 (`runtime/slo.py`), so the current
pipeline is inside budget with ~3.3 ms of headroom per hop on this
hardware. Real hardware and real codecs will differ; this number must
be re-measured on the deployment target.

Other measured components:

- Fusion runs every 5th frame, so a fresh acoustic/linguistic signal
  is fused within **50 ms** of arriving.
- Streaming ASR adds **RTT + ~55 ms pump quantization** of
  transcript delay and ~0.04 ms of in-worker per-frame cost
  (P2-1, `scripts/bench_asr_latency.py`). ASR latency does **not** add
  to per-frame processing cost because it runs on its own thread.
- Linguistic scoring windows 45 s of discourse (`config.py`
  defaults); early-call verdicts therefore lean acoustic until enough
  transcript accumulates. This is a product property, not a bug: the
  tiered output (SAFE → WATCH → HIGH_RISK) is designed to start
  conservative.

### 3.2 Proposed hosted SLOs (targets)

- **Frame processing:** p95 <= 8 ms per 10 ms hop (already the in-repo
  SLO constant).
- **Ingest-to-event:** p95 <= 120 ms from audio chunk arrival to the
  JSON event on the websocket/REST response, on the ASR-free path
  (50 ms fusion cadence + processing + headroom).
- **ASR path:** p95 transcript-to-updated-risk <= ASR-provider RTT +
  250 ms.
- **Availability:** call-path availability stays owned by the SBC; the
  detector is best-effort. Shedding load is a *success* signal, not an
  outage (`runtime/slo.py`).
- **Readiness gate:** refuse new calls below 10% headroom
  (`detector_ready_min_free`), so the load balancer drains a hot
  worker before it starts shedding.

## 4. Cost per call

Method: `plan_capacity()` in `shieldcall/runtime/capacity.py`, fed the
measured p95 of 4.691 ms, utilization 0.70, and an *illustrative*
parameter of $0.04/vCPU-hour (a parameter of the arithmetic, **not** a
vendor quote). Excludes ASR provider bills, network egress, and
storage. The ASR bill is provider-owned and will dominate compute at
any realistic scale; verify the current vendor list price before
quoting it.

| concurrent calls | calls per vCPU | vCPUs (N+1) | illustrative $/month |
|---|---|---|---|
| 100 | 1.49 | 74 | $2,161 |
| 1,000 | 1.49 | 738 | $21,550 |
| 10,000 | 1.49 | 7,372 | $215,262 |

Worked unit cost: **$0.0268 of compute per concurrent call-hour**
(= 671/1000 vCPUs x $0.04), i.e. roughly **$0.00045 per call-minute of
analyzed audio** for compute alone.

Honest caveats: (a) 1.49 calls/vCPU is low because the current
pipeline spends ~4.7 ms of each 10 ms hop; optimizing the acoustic
feature path is the cheapest way to cut this number. (b) Bandwidth is
16 KB/s per call (8 kHz x 16-bit); at 1,000 concurrent calls that is
16 MB/s of sustained ingest, which the LB and NIC must carry. (c) The
measurement machine is not the deployment machine; re-run
`plan_capacity` with target-hardware numbers before signing anything.

## 5. Scaling plan

### Phase 0: single worker (today)

Raise `max_calls` per worker only with measured headroom; the `/ready`
gate (503 at capacity) already lets a balancer drain a hot worker.
Default stays conservative.

### Phase 1: sticky worker pool

Session state is in-process, so horizontal scaling needs **sticky
routing**:

- Load balancer with consistent hashing on `call_id` (REST) and on the
  websocket path parameter, so every frame of a call lands on the
  worker holding its session.
- Kubernetes-style probes: `/health` for liveness, `/ready` (with its
  existing 503 semantics) for readiness.
- Autoscale signals, in order of importance: growth in `shed_total`,
  fraction of workers reporting unready, `asr_breaker=open` fraction,
  `ms_per_frame_ewma` trending toward the 8 ms budget.
- Capacity headroom: N+1 with +10% or at least one spare worker
  (already the `plan_capacity` rule).

### Phase 2 (optional): stateless workers

Move session state to an external store (e.g. Redis) so any worker
can serve any frame and the pool can scale without affinity. Trade-off:
adds a store round-trip to the ingest path and a new failure domain.
Not needed until Phase 1 affinity becomes an operational burden;
keep the in-process default until then.

### Rollout

Behind the P2-5 feature flag. Canary a small fraction of traffic,
watch `shed_total` and `ms_per_frame_ewma` per worker, then ramp. The
fail-open guarantee (media path never blocked by the detector) is a
release gate for every phase.

## 6. Observability and alerts

Export from `/health` as metrics: `active_calls`, `shed_total` (rate),
`asr_breaker` state, `ms_per_frame_ewma`, plus per-route latency
histograms and 4xx/5xx rates. Alert on: shed rate rising, breaker open
for > N minutes, p95 frame time > 8 ms, token-auth failure spikes
(possible credential leak or misconfigured client).

## 7. What this doc deliberately does not decide

- Vendor selection for compute, ASR provider pricing, or region
  placement (needs Mercel: budget and data-residency call).
- Whether call audio may be retained for eval (currently never
  persisted by the worker; any change needs a privacy review and the
  P3-1 consent work).
- App Store / Play billing integration (P1-1..P1-3, needs Mercel's
  developer accounts).
