# ADR-012: Vonage Voice API WebSocket adapter (fifth Media Ingest Fabric adapter)

## Status

Accepted (2026-11-21). Implements the fifth slice of TDD v2.0 §4
(Media Ingest Fabric) and roadmap stage 4 ("Integration"). The
fabric table lists eight adapters; Twilio, Asterisk, SIPREC and
Telnyx are built, Vonage was PLANNED. This ADR adds Vonage and
leaves Amazon Connect, FreeSWITCH and in-app VoIP/WebRTC planned.

## Context

Vonage is the next adapter in the fabric table after Telnyx, and
its protocol is unlike every adapter built so far. Twilio and
Telnyx both speak JSON over the WebSocket: a `connected`/`start`
handshake, base64 payloads, a per-message track label, a sequence
or chunk counter, and a timestamp or clock the adapter can check
for gaps. Vonage speaks two kinds of WebSocket message and neither
carries any of that. Binary messages are raw PCM audio. Text
messages are JSON events. There is no stream identifier, no
sequence number, no timestamp, and no per-message track label
anywhere in the protocol. An adapter written by renaming the
Telnyx events would wait forever for a `start` message that never
comes and would try to base64-decode raw PCM. The adapter below is
written from Vonage's own doc, and the ways it cannot do what the
other adapters do are stated, not papered over.

## Decision

1. **Canonical frame envelope, unchanged.** Vonage audio becomes
   `MediaFrame` (ADR-008): 10 ms mono float32 at 8 kHz, with
   `call_id`, `stream_sid`, `track`, per-track `seq`, `t_sec`,
   `discontinuous`, and `source="vonage"`. Vonage has no stream
   identifier, so `stream_sid` is "". Nothing downstream branches
   on `source`.

2. **Adapter is pure (`shieldcall/ingest/vonage.py`).**
   `VonageMediaStream` takes parsed JSON text messages
   (`handle_message`) and raw binary audio (`handle_audio`) and
   returns frames. It opens no socket, holds no credential, and
   imports nothing from the serving layer, so the protocol
   translation is testable without a network. The serving layer
   owns auth and the socket; see item 9. The split into two entry
   points is forced by the protocol: the route cannot hand a
   binary message to a JSON handler, and the adapter must not
   guess a message's kind from its bytes.

3. **Protocol, verified against Vonage's docs** ("WebSocket
   Voice Chat API Guide", developer.vonage.com, read
   2026-11-21), not from memory. The operator connects a call
   to the sidecar with an NCCO `connect` action whose endpoint
   is `{"type": "websocket", "uri": "wss://.../v1/ingest/vonage",
   "content-type": "audio/l16;rate=8000", "headers": {...}}`.
   What Vonage then sends:
   - Text, first: `{"event": "websocket:connected",
     "content-type": "audio/l16;rate=16000", ...}`. The example
     in the doc shows the rate the operator chose (16000 in the
     example) and the NCCO `headers` flattened into top-level
     fields alongside `event` and `content-type` (the doc's
     example shows `"prop1": "value1"`). A reconnection sends
     the same message again with `"reconnect_attempt": 1`.
   - Binary: raw PCM audio captured from the caller, 16-bit
     signed little-endian, at the negotiated rate, each packet
     about 20 ms.
   - Text, interleaved: `{"event": "websocket:dtmf", "digit":
     "5", "duration": 260}` (one event per keypress, digit
     `0`-`9`, `*` or `#`, duration in milliseconds).
   - Text, only in reply to outbound commands this adapter
     never sends: `{"event": "websocket:cleared"}` (ack for a
     `clear` command) and `{"event": "websocket:notify",
     "payload": {...}}` (ack for a `notify` command). They are
     recorded and counted if they arrive; they produce no
     frames.
   - No `stop` event exists over the WebSocket. The end of the
     stream is the socket closing. Call status (including
     `disconnected`, `failed`, `unanswered`) arrives on the
     operator's HTTP `eventUrl` webhook, not on the socket,
     and no webhook receiver is shipped in this slice (item 10).
   - From the application to Vonage: binary audio to play to
     the caller, and text commands (`clear`, `notify`). This
     adapter sends nothing back (item 10).

4. **Format is pinned by `websocket:connected`, and only 8 kHz
   L16 is accepted.** The `content-type` must parse as
   `audio/l16` with `rate=8000`. The doc also offers 16000
   (its recommended rate for speech recognition) and 24000.
   Those are refused with a `VonageError`, never silently
   resampled: the canonical envelope and the pipeline run at
   8 kHz, and a 16 kHz to 8 kHz decimation is a DSP claim that
   needs its own tested filter, not a one-line slice of every
   second sample. The operator chooses 8000 in the NCCO; the
   TDD fabric table lists Vonage as "Linear PCM 8 or 16 kHz",
   so 8 kHz is within the specified range. Accepting 16 kHz
   with a tested decimator is a follow-up, not a silent
   fallback. A missing `content-type`, a non-L16 encoding, or
   a rate other than 8000 all refuse to start, because the
   adapter cannot know what the bytes mean without it.

5. **There is no clock, so there is no gap detection, and that
   limit is recorded here.** Twilio and Telnyx give the adapter
   a timestamp or RTP clock to check; Vonage gives neither a
   sequence number nor a timestamp, so a lost packet is
   indistinguishable from silence that was never sent, and a
   reordered packet cannot be recognised at all. Frames are
   therefore emitted in arrival order with `t_sec` counted
   from the samples received (sample count / 8000, seconds
   from the first audio), and `discontinuous` is False except
   in the one case the protocol does reveal: a reconnection
   (item 6). No gap is ever filled with invented audio, and
   no `lost_samples` stat is kept, because there is nothing to
   count it from. A consumer that needs loss accounting must
   not assume this adapter provides it; the stat block counts
   packets, bytes, samples and frames received, which is all
   the wire offers.

6. **Reconnection is a fresh stream, and the gap it implies is
   flagged.** The doc says audio and DTMF events during a
   reconnection gap are not delivered, queued playback is
   discarded, and the reconnected session sends a fresh
   `websocket:connected` with `reconnect_attempt` counting up.
   A reconnect is therefore a fresh adapter instance on the
   route, exactly as for Telnyx. But a fresh instance whose
   first `connected` carries `reconnect_attempt` of 1 or more
   knows audio was lost in the gap, so its first frame is
   flagged `discontinuous`. A second `connected` on one
   instance is a `VonageError` unless it is a reconnect
   announcement, and the route never sends one: the route
   creates a new instance per connection. If the operator
   configures Vonage `reconnect` with `cleanCloseCodes`, a
   server close with one of those codes is treated by Vonage
   as clean and does not reconnect; the route's own close
   with code 1000 is likewise clean and ends the leg.

7. **One leg, no track separation.** The binary stream has no
   track label, and a plain `connect` action forks the leg it
   connects, not two labelled legs. Every frame is therefore
   `track="inbound"` (the remote caller, the leg ShieldCall
   scores), and there is no outbound leg to count. When only a
   mixed track exists the TDD (§4.1) calls for two-speaker
   segmentation at the evidence plane; no such segmentation
   exists yet, so a multiparty call forked through one
   `connect` action arrives mixed and is scored as inbound.
   That is a stated limit of this slice, not a property of the
   protocol: the operator isolates the caller with a
   `conversation` action and `canHear`/`canSpeak` (the doc's
   selective-listening pattern) when the call has more than
   one participant. The adapter must not claim a track
   separation the wire does not provide.

8. **State machine, fail closed.** `websocket:connected` ->
   audio* / events* -> socket close. Audio before the
   `connected` message, a `connected` with an unsupported
   format, an unknown event, a DTMF digit outside `0`-`9`,
   `*`, `#`, an empty binary packet, or a binary packet with
   an odd byte count (a truncated 16-bit sample) all raise
   `VonageError`. The adapter never recovers by guessing; the
   caller closes the stream. A partial trailing frame (fewer
   than 80 buffered samples) is discarded when the stream
   ends, not padded: padding would invent audio the caller
   never sent. The count is kept in stats as
   `discarded_on_close` so a short final packet is visible,
   not silent. Ending the stream is `finish()`, called by the
   route when the socket closes; it is idempotent, because a
   close is cleanup, not a protocol event, and a second call
   must not raise. Messages after `finish()` raise.

9. **Call binding relies entirely on the connection URI, and
   that is the weakest part of this adapter.** Vonage sends no
   call identifier over the WebSocket: no uuid, no call id, no
   stream id. The `websocket:connected` message carries only
   the negotiated format and the operator's own NCCO headers,
   which the adapter records for audit (`headers`) but does
   not treat as an identity, because the doc does not promise
   any particular header. The route therefore binds the stream
   the way the Twilio and Telnyx routes do: the operator
   pre-opens the call with POST /v1/calls (P2-6b) using the
   Vonage call uuid as the call_id, and puts that call_id in
   the NCCO websocket URI as a query parameter. There is
   nothing in-band for the adapter to check it against, so
   the operator must use a distinct URI per call; a reused
   URI would route one call's audio into another call's
   session, and no code in this slice could tell. This is
   stated here rather than hidden, because it is an operator
   obligation the other adapters do not have.

10. **Serving route (`WS /v1/ingest/vonage`).** Vonage opens
    the stream WebSocket from its own side, so the credential
    travels as a `token` query parameter in the NCCO `uri`,
    the same shape as the Twilio and Telnyx routes. The token
    is a single-purpose, per-integration credential from
    `SHIELDCALL_VONAGE_TOKENS` ("id:value,..."), never the
    sidecar bearer token. With no Vonage tokens configured the
    route refuses connections unless the local-dev escape hatch
    is on. Auth is checked before the call gate (4401, then
    4404), and the call must already exist via POST /v1/calls
    (P2-6b). The route reads both text and binary WebSocket
    messages (`ws.receive()`), where the Twilio and Telnyx
    routes read JSON only; a text message that is not JSON, or
    a protocol error from the adapter, sends an error JSON and
    closes with 4400. Frames are pushed into the same session
    and pipeline as every other ingest path, so the route adds
    no new decision path: CSCF stays the production fusion,
    and the sidecar stays recommend-only. Vonage's `vonage`
    authorization mode (a signed JWT in the handshake
    `Authorization` header, validated against Vonage's signing
    keys) is NOT verified in this slice; the query token is
    the auth mechanism here, the same as the Twilio route,
    and JWT validation is a follow-up once the verification
    keys are configured. The operator should not rely on the
    handshake JWT alone until then.

11. **What this does not do.** No audio is sent back to
    Vonage: no playback, no `clear` or `notify` commands. That
    is deliberate: this session's scope is ingest, and a
    playback path would need the VOI challenge gate (TDD §11)
    decided first. No NCCO endpoint, answer/event webhook
    receiver, or call-control caller is shipped: the
    operator's telephony server returns the NCCO pointing at
    the sidecar's wss URL with the pre-opened call_id and the
    per-integration token. No live Vonage call has been run
    against this adapter yet; that needs a Vonage account, a
    number, a Voice application, and the operator's NCCO, all
    Mercel-side. The `websocket:cleared` / `websocket:notify`
    events are only ever acks to commands this adapter never
    sends, so their arrival is recorded, not acted on.

## Consequences

- The fabric now covers the fifth of eight providers, and the
  first whose wire format is raw binary rather than framed
  JSON. The route's move from `receive_json()` to
  `receive()` is confined to the Vonage route; the Twilio and
  Telnyx routes are untouched.
- The `discontinuous` flag is advisory until a consumer uses
  it: today the pipeline receives the frames and the flag is
  visible in stats, but no scoring path branches on it. For
  Vonage the flag can only ever mean "a reconnection gap
  preceded this frame"; ordinary packet loss is invisible to
  this adapter and must not be read into a False flag.
- The per-integration token appears in the NCCO `uri` and
  therefore in Vonage's logs and event webhooks. Operators
  must treat it as a revocable credential: one token per
  integration, rotated on any suspicion of exposure, never
  the sidecar bearer token, never reused across
  environments.
- Kill-criterion note (TDD §16, Table 31): this adapter has no
  kill criterion of its own; it is plumbing. Its success
  condition is the stage-4 gate: time to warning measured end
  to end on a real Vonage call, which is still blocked on a
  live account, number and NCCO (Mercel).

## Follow-ups (not in this slice)

- Accept `audio/l16;rate=16000` (and 24000) with a tested
  anti-aliased decimator to 8 kHz, if an operator needs the
  doc's recommended 16 kHz rate; measure the filter against
  known vectors before trusting it on the warning path.
- Verify the `vonage` authorization JWT on the WebSocket
  handshake once the signing keys are configured, in addition
  to (not instead of) the per-integration query token.
- An answer/event webhook receiver, if the sidecar is ever to
  own the NCCO instead of the operator's telephony server;
  until then the `disconnected`/`failed`/`unanswered` webhook
  events are the operator's fallback signal, not the
  sidecar's.
- Two-speaker segmentation for mixed tracks (TDD §4.1), which
  would let a mixed Vonage fork be split before scoring
  instead of scored as one inbound leg.
- Bidirectional audio / playback only after the VOI challenge
  gate (TDD §11) is decided.

## Verification

- Tests in `tests/test_vonage_ingest.py`: L16 known vectors
  and round trips (including negative full-scale), lifecycle
  and format rejection (missing content-type, 16 kHz, 24 kHz,
  non-L16), framing and buffering across packet boundaries
  (20 ms packets, odd-sized packets, partial trailing frame
  discarded on finish), audio before `connected`, unknown
  events, DTMF digits and duration, `cleared`/`notify`
  recording, reconnect flagging, custom headers recorded,
  messages after finish rejected.
- Tests in `tests/test_vonage_route.py`: token auth
  (required, wrong, absent-with-no-tokens), pre-opened-call
  gate, full connected / binary / close flow into a live
  session, format mismatch and malformed JSON closing with
  4400, capabilities listing.
- Full core suite green after the change (see the session log).
