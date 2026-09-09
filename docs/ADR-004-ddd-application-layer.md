# ADR-004: Domain-driven application layer for the detector API

## Status

Accepted

## Context

shieldcall-core already has scientific bounded contexts (linguistic, acoustic, fusion, agent, runtime). The HTTP sidecar was a thin FastAPI wrapper around `SidecarRuntime`. The phone client (`ShieldCall`) scored live calls with on-device SENTINEL and Claude, and only the Lab screen called the sidecar. The product architecture says the detector is core, fail-open, recommend-only.

## Decision

1. Keep scientific packages where they are. Do not rewrite scorers as DDD entities.
2. Add `shieldcall/domain` (language of Call, Decision, Capabilities) and `shieldcall/application/DetectorApplication` (use cases the UI calls).
3. Publish every use case as an HTTP endpoint under `/v1`, plus `/health`, `/ready`, `/docs`, `/openapi.json`.
4. New `POST /v1/calls/{id}/chunk` is the live-call window: transcript plus optional PCM.
5. Stateless `POST /v1/score/{linguistic,acoustic,fuse}` for Lab and diagnostics.
6. The phone client talks to this API and **fail-opens** (local SENTINEL / Claude) if core is down.

## Alternatives considered

**Move all numpy code into domain entities.** Rejected: would churn research tests for no product gain.

**gRPC / FFI into the app.** Deferred: Expo Go cannot load a Python runtime. HTTP sidecar on LAN is the contract that already exists.

**Score only in JS.** Rejected: that abandons dual-stream fusion and the fail-open sidecar doctrine.

## Consequences

- OpenAPI is the contract the UI is written against.
- Core remains optional at runtime. A missing sidecar must not drop the call.
- Adding a use case means: domain DTO if needed, application method, HTTP route, TypeScript client method, UI call site.
