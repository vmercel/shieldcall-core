# ADR-008: Twilio Media Streams adapter (first Media Ingest Fabric adapter)

## Status

Accepted (2026-11-17). Implements the first slice of TDD v2.0 §4
(Media Ingest Fabric) and roadmap stage 4 ("Integration"), which
orders Twilio and Asterisk first (lowest barrier), then SIPREC. The
fabric table lists eight adapters; until now only the generic REST
and WebSocket ingest were BUILT. This ADR adds Twilio as the first
provider adapter and defines the canonical frame envelope every
later adapter must produce.

## Context

ShieldCall scores live calls, but a live call has to reach the
pipeline somehow. The sidecar already accepts PCM over REST and over
a generic WebSocket (`/v1/calls/{id}/stream`), which is enough for
the lab and the Expo loopback. A real deployment sits behind a
telephony provider that forks call audio to a WebSocket, and each
provider speaks its own protocol: Twilio sends JSON messages with
base64 mu-law payloads, Asterisk will send something else, SIPREC
something else again. If each provider's quirks leak into the
pipeline, every new provider rewrites the detector. The fabric
fixes that: one canonical envelope, one adapter per provider, and
the pipeline never learns a provider's name.

Twilio goes first because it is the lowest barrier of the three
named in the roadmap: no PBX to run, no SIPREC session to negotiate,
just a TwiML `<Stream>` pointing at a WebSocket URL. The Telnyx
thread (lab-only review of Telnyx Voice API media streaming) is a
separate adapter in the same fabric and is not touched here; the
roadmap order (Twilio, Asterisk, then SIPREC) is kept, and Telnyx
stays PLANNED until the lab review produces a decision.

## Decision

1. **Canonical frame envelope (`shieldcall/ingest/frame.py`).**
   `MediaFrame` is one 10 ms frame of mono float32 PCM in [-1, 1]
   with its provenance: `call_id`, `stream_sid`, `track`, per-track
   `seq`, `t_sec` (provider clock, seconds from stream start),
   `sample_rate`, `discontinuous` (True when a sequence gap preceded
   the frame), and `source` (adapter name, audit only). Ten
   milliseconds at 8 kHz is 80 samples; the envelope is defined at
   the provider's native rate, so a future wideband adapter emits
   160-sample frames at 16 kHz under the same shape. Nothing
   downstream branches on `source`.

2. **Adapter is pure (`shieldcall/ingest/twilio.py`).**
   `TwilioMediaStream` takes parsed JSON messages and returns
   frames. It opens no socket, holds no credential, and imports
   nothing from the serving layer, so the protocol translation is
   testable without a network and reusable by any host (the sidecar
   HTTP app today, a plain script tomorrow). The serving layer owns
   auth and the socket; see item 6.

3. **Protocol, verified against Twilio's docs** ("Media Streams -
   WebSocket Messages", read 2026-11-17), not from memory. Events:
   `connected` (protocol "Call"), `start` (streamSid, callSid,
   tracks, mediaFormat, customParameters), `media` (track, chunk,
   timestamp, base64 payload), `stop`, `dtmf`, `mark`. The format is
   pinned by `start.mediaFormat`: encoding must be `audio/x-mulaw`,
   sampleRate 8000, channels 1. Anything else is a `TwilioError`,
   never a silent resample: the adapter does not guess what a codec
   it was not told to expect sounds like.

4. **Exact G.711 mu-law decode, no new dependency.** A 256-entry
   lookup table built from the ITU-T G.711 algorithm (verified
   against the known vectors 0xFF -> 0.0, 0x00 -> -32124/32768,
   0x80 -> +32124/32768, plus an encode/decode round trip within a
   half quantisation step). Twilio payloads are 20 ms (160 bytes);
   each message yields two 10 ms frames, and a short final payload
   buffers its remainder for the next message.

5. **Continuity is judged from `media.timestamp`, not `chunk`.**
   The docs number `chunk` from 1 per message but their example
   numbers it globally across tracks, so a chunk jump across tracks
   is not a gap. A timestamp that lands past the end of the previous
   payload is a gap: the next frame is flagged `discontinuous`, the
   lost samples are counted, and any partial buffered frame is
   discarded rather than spliced across the two timelines. A
   repeated chunk on the same track at the same or an earlier
   timestamp is a duplicate and is dropped; a timestamp that rewinds
   is dropped as out of order. No gap is ever filled with invented
   audio: the pipeline sees real frames with a flag, not padding.
   A partial trailing frame at `stop` is discarded and counted, for
   the same reason.

6. **Tracks are never mixed.** `analyzed_track` defaults to
   `inbound` (the remote caller, the leg ShieldCall scores). The
   other leg's messages are validated and counted but never decoded.
   `start` must offer the analyzed track, or the stream is refused:
   silently scoring the wrong leg would be worse than refusing.

7. **State machine, fail closed.** `connected` -> `start` ->
   `media`* -> `stop`. Media before start, a second start, a
   `streamSid` that does not match the started stream, a `callSid`
   that does not match the pre-opened call, an unknown event, bad
   base64, or any message after `stop` all raise `TwilioError`. The
   adapter never recovers by guessing; the caller closes the stream.
   DTMF digits and mark names are recorded (they arrive on the same
   socket and an operator will want them in the log) but produce no
   frames.

8. **Serving route (`WS /v1/ingest/twilio`).** Twilio cannot send
   an Authorization header when it opens the Stream WebSocket, so
   the credential travels as a `token` query parameter, the mechanism
   Twilio documents for Stream URLs. The token is a single-purpose,
   per-integration credential from `SHIELDCALL_TWILIO_TOKENS`
   ("id:value,...", same shape as the sidecar tokens), never the
   sidecar bearer token, and the route only serves over wss in
   production. With no Twilio tokens configured the route refuses
   connections unless the local-dev escape hatch is on. Auth is
   checked before the call gate (4401, then 4404), and the call must
   already exist via POST /v1/calls (P2-6b): the operator opens it
   with the Twilio Call SID as the call_id, and the adapter rejects
   a `start.callSid` that does not match. A protocol error sends an
   error JSON and closes with 4400; `stop` sends the final event and
   closes with 1000. Frames are pushed into the same session and
   pipeline as every other ingest path, so the route adds no new
   decision path: CSCF stays the production fusion, and the sidecar
   stays recommend-only.

9. **What this does not do.** No audio is sent back to Twilio: the
   stream is unidirectional (`<Start><Stream>`), and the adapter
   has no `clear`/`mark` playback path. That is deliberate: this
   session's scope is ingest, and a playback path would need the
   VOI challenge gate (TDD §11) decided first. No TwiML endpoint is
   shipped: the operator's telephony server returns the `<Stream>`
   verb with the sidecar's wss URL, the pre-opened call_id, and the
   per-integration token. No live Twilio call has been run against
   this adapter yet; that needs a Twilio account, a number, and the
   operator's TwiML, all Mercel-side.

## Consequences

- The next adapter (Asterisk) reuses `MediaFrame` and the route
  pattern, and only writes its own wire translation. SIPREC then
  follows the same shape.
- The Channel Passport (ADR-007) can now be fed from a real
  provider: a Twilio stream declares `audio/x-mulaw` at 8 kHz, so
  the passport's declared-codec input is exact for these calls
  instead of absent.
- The per-integration token appears in the Stream URL and therefore
  in Twilio's logs. Operators must treat it as a revocable
  credential: one token per integration, rotated on any suspicion of
  exposure, never the sidecar bearer token, never reused across
  environments. This is documented here rather than hidden, because
  the URL mechanism is Twilio's, not ours.
- The `discontinuous` flag is advisory until a consumer uses it:
  today the pipeline receives the frames and the flag is visible in
  stats, but no scoring path branches on it. Wiring the flag into
  the acoustic scorer's trust accounting is future work.
- Kill-criterion note (TDD §16, Table 31): this adapter has no
  kill criterion of its own; it is plumbing. Its success condition
  is the stage-4 gate: time to warning measured end to end on a
  real Twilio call, which is still blocked on a live number and
  TwiML (Mercel).

## Verification

- 26 tests in `tests/test_twilio_ingest.py`: mu-law known vectors
  and round trip, lifecycle and format rejection, framing and
  buffering, gap/duplicate/out-of-order handling, track separation,
  stop/DTMF/mark.
- 11 tests in `tests/test_twilio_route.py`: token auth (required,
  wrong, absent-with-no-tokens), pre-opened-call gate, full
  connected/start/media/stop flow into a live session, callSid
  mismatch and malformed media closing with 4400, capabilities
  listing.
- Full core suite green after the change (see the session log).
  Three real catches during testing: the gap check first used the
  current payload's duration instead of the previous payload's
  (wrong whenever payloads differ in length); a partial frame was
  spliced across a timestamp gap instead of discarded; and the
  test's own mu-law encoder used a hand-derived segment table, so
  it was replaced by a nearest-match over the decode table.
