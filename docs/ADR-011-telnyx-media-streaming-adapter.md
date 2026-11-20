# ADR-011: Telnyx Media Streaming adapter (fourth Media Ingest Fabric adapter)

## Status

Accepted (2026-11-20). Implements the fourth slice of TDD v2.0 §4
(Media Ingest Fabric) and roadmap stage 4 ("Integration"). The
fabric table lists eight adapters; Twilio, Asterisk and SIPREC are
built, Telnyx was PLANNED. This ADR adds Telnyx and leaves Vonage,
Amazon Connect, FreeSWITCH and in-app VoIP/WebRTC planned.

## Context

Telnyx is the provider Mercel already has a thread with: a lab-only
review of the Telnyx Voice API with media streaming and SIP
Trunking, ShieldCall as a fail-open sidecar on test numbers, pilot
once lab results are in (Quinn Taylor, quinn@telnyx.com; see the
RFE packet's Telnyx correspondence exhibit). That thread is a sales
routing reply, not a technical assessment, a pilot, or an
endorsement, and nothing in this adapter changes that. What the
adapter does is make the lab review concrete: when Mercel points a
Telnyx test number at the sidecar, the audio arrives as canonical
frames instead of needing a new integration.

Telnyx's media streaming looks like Twilio's at first glance (a
WebSocket, JSON messages, base64 payloads) and differs in every
detail that matters: snake_case fields, no `tracks` list in `start`,
the call identifier is `call_control_id` (a `v2:...` id) rather than
a Call SID, the media `timestamp` is an RTP timestamp in codec
clock units rather than milliseconds, and `chunk` is one counter
shared by both tracks rather than one per track. Copying the Twilio
adapter and renaming the events would silently misread the
timeline. The adapter below is written from Telnyx's own doc.

## Decision

1. **Canonical frame envelope, unchanged.** Telnyx media becomes
   `MediaFrame` (ADR-008): 10 ms mono float32 at 8 kHz, with
   `call_id`, `stream_sid`, `track`, per-track `seq`, `t_sec` from
   the provider clock, `discontinuous`, and `source="telnyx"`.
   Nothing downstream branches on `source`.

2. **Adapter is pure (`shieldcall/ingest/telnyx.py`).**
   `TelnyxMediaStream` takes parsed JSON messages and returns
   frames. It opens no socket, holds no credential, and imports
   nothing from the serving layer, so the protocol translation is
   testable without a network. The serving layer owns auth and the
   socket; see item 9.

3. **Protocol, verified against Telnyx's docs** ("Media Streaming
   over Websockets", developers.telnyx.com, read 2026-11-20), not
   from memory. Events:
   - `connected`: `{"event": "connected", "version": "1.0.0"}`.
     No `protocol` field, no `stream_id`, no `sequence_number`.
   - `start`: `{"event": "start", "sequence_number": "1",
     "start": {"user_id", "call_control_id", "call_session_id",
     "from", "to", "tags", "client_state",
     "media_format": {"encoding": "PCMU", "sample_rate": 8000,
     "channels": 1}}, "stream_id": "..."}`. The `stream_id` is
     top-level, after the `start` object, not inside it. There is
     no `tracks` list and no custom-parameters object: track
     selection is a stream-request parameter (`stream_track`:
     `inbound_track` (default), `outbound_track`, `both_tracks`),
     and the per-message track is `media.track`.
   - `media`: `{"event": "media", "sequence_number": "4",
     "media": {"track": "inbound"|"outbound", "chunk": "2",
     "timestamp": "5", "payload": "<base64>"}, "stream_id": ...}`.
     The payload is a base64-encoded RTP payload with no RTP
     headers. `timestamp` is the RTP timestamp of the media
     packet, in the codec's RTP clock units; it is not
     wall-clock time and not milliseconds.
   - `stop`: `{"event": "stop", "sequence_number": "5",
     "stop": {"user_id", "call_control_id"}, "stream_id": ...}`.
   - `dtmf`: `{"event": "dtmf", "stream_id": ...,
     "occurred_at": "<ISO-8601>", "sequence_number": "5",
     "dtmf": {"digit": "1"}}`. DTMF may arrive out of order;
     `occurred_at` is the telephony-engine-side timestamp.
   - `mark` (received): `{"event": "mark", "stream_id": ...,
     "sequence_number": "5", "mark": {"name": "..."}}`. The sent
     form has no `stream_id` or `sequence_number`.
   - `error`: `{"event": "error", "payload": {"code": <int>,
     "title": <str>, "detail": <str>}, "stream_id": ...}`. No
     `sequence_number`. Codes: 100002 `unknown_error`, 100003
     `malformed_frame`, 100004 `invalid_media`, 100005
     `rate_limit_reached`.
   - `clear`: `{"event": "clear"}` is sent only; the adapter
     never sends it (item 10).

4. **Format is pinned by `start.media_format`, and both G.711
   codecs are supported.** The `start` example declares
   `encoding: "PCMU"`, sample rate 8000, channels 1. PCMU is
   G.711 mu-law, decoded by the shared `mulaw_decode` from
   `shieldcall.ingest.g711`. The `<Stream>` codec attribute also
   allows `PCMA`, so `encoding: "PCMA"` is accepted and decoded
   by the shared `alaw_decode`. Anything else (`G722`, `OPUS`,
   `AMR-WB`, `L16`, `default`, or an unknown string) is a
   `TelnyxError`, never a silent resample, and a sample rate
   other than 8000 or channels other than 1 is likewise refused.
   Telnyx's bidirectional codecs are a separate list and are out
   of scope here (item 10).

5. **Continuity is judged from the RTP timestamp, not from
   `chunk`.** At 8 kHz the RTP clock runs at 8000 units per
   second, so one timestamp unit is one sample: a payload of N
   samples advances the timestamp by N, and the next timestamp
   should land exactly at `last_timestamp + last_payload_samples`.
   A timestamp past that point is a gap: the next frame is
   flagged `discontinuous`, the lost samples are counted, and
   any partial buffered frame is discarded rather than spliced
   across the two timelines. A timestamp that rewinds is dropped
   as out of order. No gap is ever filled with invented audio.
   A partial trailing frame at `stop` is discarded and counted.
   Frame presentation time is `t_sec = rtp_timestamp / 8000`,
   seconds from stream start on Telnyx's clock.

6. **`chunk` is one counter for both tracks, so it is a loss
   signal, not a per-track clock.** The docs say a gap in `chunk`
   means media was lost, but consecutive values do not prove
   none was (media dropped during a reconnect is never
   numbered). The adapter therefore tracks the highest chunk
   seen across both tracks: a jump past it increments a
   `chunk_gaps` stat for visibility, but does not by itself flag
   the analysed track discontinuous, because the lost message
   may have belonged to the other leg. The timestamp rule in
   item 5 is what flags discontinuity. A duplicate is a message
   repeating a chunk already seen on the same track at the same
   or an earlier timestamp, and is dropped.

7. **Event order is not guaranteed; the adapter does not
   reorder.** The docs say to use `sequence_number` or `chunk`
   to reorder events. `sequence_number` counts the frames Telnyx
   sends and can be skipped, and two frames can share a number,
   so it orders but does not detect loss. This slice records the
   last `sequence_number` seen for diagnostics and emits frames
   in arrival order, dropping timestamp rewinds (item 5) rather
   than buffering for a reorder window. Buffering out-of-order
   media to re-sequence it would add latency to the warning path
   and is future work; the choice is recorded here rather than
   hidden, because a reordered media message today is dropped
   and counted, not silently played out of order.

8. **Tracks are never mixed, and `start` cannot verify the
   track.** Telnyx declares no track list in `start`, so unlike
   Twilio the adapter cannot refuse a stream that did not offer
   the analysed track. `analyzed_track` defaults to `inbound`
   (the remote caller, the leg ShieldCall scores). Messages on
   the other leg are validated and counted but never decoded.
   The operator must request the right leg with `stream_track`
   when starting the stream; the adapter's per-message filter is
   the backstop, not the control.

9. **State machine, fail closed.** `connected` -> `start` ->
   `media`* -> `stop`. Media before start, a second start, a
   `stream_id` that does not match the started stream, a
   `call_control_id` that does not match the pre-opened call, an
   unknown event, bad base64, a Telnyx `error` frame, or any
   message after `stop` all raise `TelnyxError`. The adapter
   never recovers by guessing; the caller closes the stream.
   DTMF digits are recorded with their `occurred_at`, and mark
   names are recorded; neither produces frames. A reconnect is a
   new WebSocket connection (new `connected`, then a new `start`
   if streaming had already started; `sequence_number` and
   `chunk` continue, they do not reset), so a reconnect is a
   fresh adapter instance on the route, not a second `start` on
   the old one. Telnyx may close without a `stop` frame when the
   connection is already failing; the route treats the `stop`
   frame or the socket closing, whichever comes first, as the
   end of the stream, exactly as the docs direct.

10. **Serving route (`WS /v1/ingest/telnyx`).** Telnyx opens the
    stream WebSocket from its own side, so the credential
    travels as a `token` query parameter in the `stream_url`,
    the same shape as the Twilio route. The token is a
    single-purpose, per-integration credential from
    `SHIELDCALL_TELNYX_TOKENS` ("id:value,..."), never the
    sidecar bearer token. With no Telnyx tokens configured the
    route refuses connections unless the local-dev escape hatch
    is on. Auth is checked before the call gate (4401, then
    4404), and the call must already exist via POST /v1/calls
    (P2-6b): the operator opens it with the Telnyx
    `call_control_id` as the call_id, and the adapter rejects a
    `start.call_control_id` that does not match. A protocol
    error sends an error JSON and closes with 4400; `stop`
    sends the final event and closes with 1000. Frames are
    pushed into the same session and pipeline as every other
    ingest path, so the route adds no new decision path: CSCF
    stays the production fusion, and the sidecar stays
    recommend-only.

11. **What this does not do.** No audio is sent back to Telnyx:
    no bidirectional RTP, no MP3 playback, no `mark` or `clear`
    messages sent. That is deliberate: this session's scope is
    ingest, and a playback path would need the VOI challenge
    gate (TDD §11) decided first. No TeXML endpoint, call-control
    webhook, or `streaming_start` caller is shipped: the
    operator's telephony server starts the stream with the
    sidecar's wss URL, the pre-opened call_id, and the
    per-integration token. No live Telnyx call has been run
    against this adapter yet; that needs a Telnyx account, a
    test number, and the operator's stream request, all
    Mercel-side. Telnyx also signs every stream WebSocket
    connection (`telnyx-signature-ed25519` /
    `telnyx-timestamp` headers, Ed25519 over
    `{timestamp}|{stream_url}` with the account public key);
    signature verification is NOT implemented in this slice, it
    needs the account public key configured plus an Ed25519
    implementation (the standard library has none), and it is
    logged as a follow-up below rather than silently skipped.
    The query token is the auth mechanism in this slice, the
    same as the Twilio route.

## Consequences

- The Telnyx lab thread now has a concrete next step: point a
  Telnyx test number's stream at `WS /v1/ingest/telnyx` and the
  sidecar scores it with the same pipeline as Twilio, Asterisk
  and SIPREC. The adapter proves the fabric's claim that a new
  provider is a wire translation, not a pipeline change.
- The shared `g711` module now has its second consumer for
  A-law: Asterisk used `alaw_decode` first, Telnyx is the first
  adapter to select the codec from a declared `media_format`
  rather than a fixed payload type.
- The per-integration token appears in the `stream_url` and
  therefore in Telnyx's logs and in the Ed25519-signed string.
  Operators must treat it as a revocable credential: one token
  per integration, rotated on any suspicion of exposure, never
  the sidecar bearer token, never reused across environments.
- The `discontinuous` flag is advisory until a consumer uses
  it: today the pipeline receives the frames and the flag is
  visible in stats, but no scoring path branches on it. Wiring
  the flag into the acoustic scorer's trust accounting is future
  work.
- The `chunk_gaps` stat is a Telnyx-specific visibility signal
  with no Twilio equivalent: a nonzero count means Telnyx lost
  media somewhere on the stream, on either leg.
- Kill-criterion note (TDD §16, Table 31): this adapter has no
  kill criterion of its own; it is plumbing. Its success
  condition is the stage-4 gate: time to warning measured end
  to end on a real Telnyx call, which is still blocked on a
  live number and stream request (Mercel).

## Follow-ups (not in this slice)

- Ed25519 signature verification on the WebSocket handshake
  (`telnyx-signature-ed25519` / `telnyx-timestamp` over
  `{timestamp}|{stream_url}`), once the account public key is
  configured and an Ed25519 implementation is available.
- Out-of-order re-sequencing by `sequence_number` if live
  traffic shows media arriving out of order often enough to
  matter; measure first, buffer second.
- Bidirectional RTP / playback only after the VOI challenge
  gate (TDD §11) is decided.

## Verification

- Tests in `tests/test_telnyx_ingest.py`: G.711 PCMU/PCMA
  known vectors and round trips, lifecycle and format
  rejection, framing and buffering, RTP-timestamp gap /
  duplicate / out-of-order handling, the shared-chunk
  counter, track separation, stop / DTMF / mark / Telnyx
  error frames, reconnect-shape tolerance (a fresh instance
  accepts a new start; a second start on one instance is an
  error).
- Tests in `tests/test_telnyx_route.py`: token auth
  (required, wrong, absent-with-no-tokens), pre-opened-call
  gate, full connected / start / media / stop flow into a live
  session (PCMU and PCMA), call_control_id mismatch and
  malformed media closing with 4400, capabilities listing.
- Full core suite green after the change (see the session log).
