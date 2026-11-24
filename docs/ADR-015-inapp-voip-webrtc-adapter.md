# ADR-015: In-app VoIP/WebRTC adapter (eighth Media Ingest Fabric adapter)

## Status

Accepted (2026-11-24). Implements the eighth and last slice of
TDD v2.0 section 4 (Media Ingest Fabric) and roadmap stage 4
("Integration"). The fabric table lists eight adapters; Twilio,
Asterisk, SIPREC, Telnyx, Vonage, Amazon Connect and FreeSWITCH
are built, and in-app VoIP/WebRTC was the last PLANNED entry.
This ADR adds it and closes the fabric table. No adapter
remains planned.

## Context

The in-app path is unlike every adapter built so far in two
specific ways, and both decide the shape of this slice.

First, there is no vendor protocol to verify against. Twilio,
Telnyx, Vonage, Amazon Connect, Asterisk and FreeSWITCH each
own a wire format documented by someone else, and each ADR
before this one was written from that document. The in-app
sender is ShieldCall's own app, and the app has no WebRTC
stack at all today: no peer connection, no Opus encoder in
the client, no Encoded Transform, no fork of any kind. An
adapter written by renaming a provider protocol would wait
for messages no client sends, and an adapter that claimed
WebRTC interop would be claiming a peer connection, SRTP
and DTLS that do not exist in this slice. What exists is the
TDD's one-row specification (Table 8): the ShieldCallAI call
screen, Opus at 8 to 48 kHz, Encoded Transform for
app-originated media, cited to the W3C WebRTC Encoded
Transform specification, and the section 4.1 envelope rules:
adapters reorder on sequence number within a 60 ms window,
map sender timestamps to a monotonic session clock, flag
concealment so the acoustic stream can down-weight it, and
preserve codec identity for the Channel Passport. This ADR
therefore defines the fork protocol itself, in item 3, and
says plainly in item 11 what it does not do.

Second, the media is Opus, and Opus is the first codec in
the fabric that Python cannot decode from its own standard
library or from a pinned dependency. Every adapter before
this one decoded G.711 by table lookup or L16 by byte
unpacking. Opus needs a real decoder. The system already
ships one (libopus), and item 4 uses it directly, with no
new package and no subprocess, and refuses the stream on a
host that has none, rather than pretending.

The Encoded Transform grounding, checked against the W3C
specification (WebRTC Encoded Transform, W3C Working Drafts,
2021-2023) on 2026-11-24: the transform is set on an
`RTCRtpSender` or `RTCRtpReceiver` (`transform` attribute,
`RTCRtpScriptTransform`), and the script sees encoded
frames (`RTCEncodedAudioFrame`: a `timestamp` and a `data`
buffer, with metadata carrying the synchronization source).
Two consequences follow, and both are in the protocol.
The transform exposes the encoded frame's timestamp but no
RTP sequence number, so the fork must assign sequence
numbers itself, in send order, and this protocol does that
(item 3). And the frame the transform sees is the encoder's
output, an Opus packet, so the fork carries Opus packets,
not decoded PCM: decoding is the sidecar's job, and item 4
is that job.

## Decision

1. **Canonical frame envelope, unchanged, with codec
   provenance added.** In-app audio becomes `MediaFrame`
   (ADR-008): 10 ms mono float32 at 8 kHz, with `call_id`,
   `stream_sid`, `track`, per-track `seq`, `t_sec`,
   `discontinuous`, and `source="inapp"`. The analysed leg
   is `track="inbound"` (the far end). Two fields are added
   to `MediaFrame` in this slice, both defaulted so every
   existing construction is untouched: `source_codec`
   ("opus" here) and `source_rate` (the encoded rate, 8000
   to 48000). TDD section 4.1 requires codec identity to be
   preserved for the Channel Passport rather than discarded
   at decode, and until now no adapter recorded it; the
   in-app adapter is the first to do so, and the older
   adapters leave the fields empty, which means unknown,
   never a guess. Nothing downstream branches on `source`,
   `source_codec` or `source_rate` today; the passport
   still estimates the channel from audio, and may use the
   recorded identity later.

2. **Adapter is pure (`shieldcall/ingest/inapp.py`).**
   `InAppMediaStream` takes parsed JSON messages
   (`handle_message`) and returns frames. It opens no
   socket, holds no credential, and imports nothing from
   the serving layer, so the protocol translation is
   testable without a network. The serving layer owns auth
   and the socket; see item 10. The decoder is injected
   (an object with `decode(bytes)` returning float32
   samples at 8 kHz, plus `close()`); production passes
   none and the stream builds a real `OpusDecoder` at
   hello time, while tests inject a stub for protocol
   cases and use the real decoder for round-trip cases.
   The stream takes ownership of the decoder and closes it
   on `stop` or `finish()`: one decoder per stream, never
   shared, because libopus decoder state is per-stream.

3. **Protocol, defined by this ADR, because no vendor
   defines it.** The app's future Encoded Transform fork
   sends JSON text messages over one WebSocket, in this
   order:
   - `hello`, first: `{"type": "hello", "codec": "opus",
     "sample_rate": 8000|16000|24000|48000, "channels": 1,
     "track": "far-end"|"near-end", "stream_id": "...",
     "call_id": "...", "consent_ref": "..."}`. The hello
     pins the format before any audio is decoded, the way
     Twilio's `start` and Vonage's `websocket:connected`
     do. `stream_id`, `call_id` and `consent_ref` are
     optional; `call_id`, when present, is checked
     against the route's binding (item 9), and
     `consent_ref` is recorded only (item 8).
   - `media`: `{"type": "media", "seq": 0..65535,
     "timestamp": 0..2**32-1, "payload": "<base64 Opus
     packet>"}`. One Opus packet per message. `seq`
     starts at 0 on every fork and increments per packet
     in send order; it is fork-assigned, because the
     Encoded Transform API exposes no RTP sequence
     number (see Context). `timestamp` is the encoded
     frame's timestamp on the 48 kHz Opus RTP clock
     (RFC 7587: the Opus RTP clock is always 48 kHz,
     whatever rate was encoded), carried across
     unchanged.
   - `stop`: `{"type": "stop"}`. The socket closing ends
     the stream the same way.
   Binary WebSocket frames are not read: an Opus packet
   with no seq or timestamp cannot be ordered, so the
   route reads JSON only, like Twilio and Telnyx. A
   second hello is refused: a reconnect is a new
   connection with a fresh hello, never a re-pin
   mid-stream, because a re-pin could silently change
   the rate or the leg being scored.

4. **Opus is decoded by the system libopus, through
   ctypes, at 8 kHz output.** `shieldcall/ingest/opus.py`
   loads libopus once (`ctypes.util.find_library`, with a
   `libopus.so.0` fallback), configures the three decoder
   calls it uses, and exposes `OpusDecoder`, which
   decodes one packet to float32 mono PCM. The decoder
   is created at an output rate of 8000 Hz, so libopus
   performs the rate conversion internally: a 48 kHz
   Opus packet decoded to 8 kHz yields genuine 8 kHz
   audio, and the recorded `source_rate` of 48000 says
   what was encoded, so nothing is ever upsampled to
   fake bandwidth (TDD section 4.1). No new dependency
   is pinned and no subprocess is spawned; the test
   suite's encoder lives in the tests, not in production
   code, because encoding belongs to the app's WebRTC
   stack, never to the sidecar. A host with no libopus
   fails closed: `OpusDecoder` raises
   `OpusDecoderUnavailable`, the hello is refused with
   that reason in plain English, and no stream is ever
   half-decoded. Packet-loss concealment is never
   requested (`decode_fec=0`): a missing packet is a
   gap the adapter flags (item 7), never audio the
   decoder invents. A packet libopus rejects, an empty
   packet, or a packet over the 4096-byte cap is a
   protocol error, not silence.

5. **Only Opus, only mono, only 8 to 48 kHz.** A hello
   declaring another codec, two channels, or a rate
   outside 8000/16000/24000/48000 is refused before any
   audio is decoded, with an error naming the supported
   values. The TDD fabric table lists Opus 8 to 48 kHz
   for this adapter, and those four rates are exactly
   the rates Opus defines; nothing is silently
   resampled, and a codec this adapter cannot decode
   (G.722, AMR-WB, or a future codec) is a new adapter
   decision, not a fallback.

6. **Only the far end is scored; the near end is
   counted, never decoded.** A `track` of `"far-end"`
   is the remote party's voice and becomes frames as
   `inbound`. A `track` of `"near-end"` is the
   protected user's own voice: its messages are
   counted, their sequence numbers are accounted for
   (gaps included), and their payloads are never
   decoded and never framed, following the far-end-only
   rule of TDD section 4.1 (the protected user's voice
   is never analysed acoustically). There is no mixed
   track: a hello naming one is refused, following the
   SIPREC precedent (ADR-010), because one summed
   channel cannot be split into legs and scoring it
   would score the protected user's voice. One fork
   carries one leg; the app opens a second fork if it
   wants the other leg counted. No two-speaker
   segmentation exists yet to make `mixed` safe.

7. **There is a clock, so there is gap detection, and
   no concealment is synthesised.** The wire carries a
   sequence number and a 48 kHz timestamp, so this
   adapter can do what Vonage and FreeSWITCH cannot:
   reorder within the 60 ms window TDD section 4.1
   requires (2880 RTP ticks), drop packets that arrive
   after the window, and flag the first frame after a
   gap `discontinuous`. Packets are held in a reorder
   buffer keyed by sequence number and emitted in
   sequence order; a missing sequence is counted lost
   once more than 60 ms of audio past it has arrived,
   with `lost_packets` and an estimated `lost_samples`
   from the timestamp difference. A sender-clock jump
   with contiguous sequence numbers (the timestamp
   advancing further than the audio appended so far)
   is the same kind of gap and is flagged the same
   way, with `timestamp_jumps` counted separately so
   the two causes are distinguishable in stats. A
   partial frame buffered before a gap is discarded,
   not spliced across it. No gap is ever filled: TDD
   section 4.1 asks for concealment-flagged frames, and
   this slice delivers the flag (`discontinuous`) on
   the next real frame rather than synthesised audio,
   because packet-loss concealment from libopus would
   be invented audio the acoustic stream cannot tell
   from real speech. The `concealed` marker of the TDD
   envelope is therefore a follow-up, not a claim made
   here. Timestamps are unwrapped against the first
   raw value seen (32-bit wraparound, over a day at
   48 kHz), and presentation time is anchored to the
   first packet emitted in sequence order, so a first
   arrival that is out of order cannot shift the
   stream's clock. A forward sequence jump over 512
   packets with no hello in between is refused: a fork
   that restarts its sequence reconnects with a fresh
   hello.

8. **Consent is recorded, not validated, and that
   limit is stated here.** The hello may carry
   `consent_ref`, and the adapter records it on the
   stream for audit. No consent ledger exists in the
   codebase to validate it against (TDD section 14.2
   is stage-5 work), so validating it would be a
   fiction; recording it is not a claim that it was
   checked. The controls that do exist today are the
   app's own consent gate (the consent banner and
   consent service in ShieldCallAI, which the user
   must pass before any fork starts), the pre-opened
   call (P2-6b), and the per-integration token
   (item 10). Ledger validation of `consent_ref` is a
   follow-up, listed below, and until it lands the
   consent_ref on a stream proves only that the fork
   declared one.

9. **Call binding rests on the connection URI, with
   the hello check the other raw adapters do not
   have.** The app pre-opens the call with
   POST /v1/calls (P2-6b) and puts that call_id in the
   WebSocket URI as a query parameter, using a
   distinct URI per call. When the hello carries
   `call_id`, the adapter refuses the stream unless it
   equals the route's call_id, so a fork started with
   the wrong call's hello cannot be routed into
   another call's session. When the hello carries no
   `call_id`, the binding rests on the URI alone, and
   a reused URI would route one call's audio into
   another call's session, which no code in this slice
   could tell; the app must use a distinct URI per
   call and include `call_id` in the hello so the
   check runs. This is stated here rather than hidden,
   because it is an operator obligation the JSON
   provider adapters share and the app must honour.

10. **Serving route (`WS /v1/ingest/inapp`).** The
    route follows the Twilio/Telnyx shape, not the
    FreeSWITCH shape: the sender is ShieldCall's own
    client, so the route answers with session events
    after each batch of frames (the app is the one
    consumer that can use them), sends an error JSON
    and closes with 4400 on a protocol error, and
    closes with 1000 after `stop`. Auth is a
    per-integration query token from
    `SHIELDCALL_INAPP_TOKENS` ("id:value,..."), never
    the sidecar bearer token, failing closed when no
    tokens are configured and the lab escape hatch is
    off, exactly like the other ingest routes. The
    query token is used because a browser or React
    Native WebSocket cannot send a custom
    Authorization header. Accepting the user's
    Supabase access token over this socket (the P2-6d
    mechanism the REST routes already use) is a
    follow-up: no WebSocket JWT verification exists
    yet, and inventing one in this slice would have
    meant a second auth mechanism to maintain. The
    route enforces the pre-opened-call gate (P2-6b)
    and reads JSON only. The stream is
    recommend-only like every other ingest path:
    audio flows into the same pipeline and session,
    nothing hangs up. Capabilities lists the new
    endpoint.

11. **What this does not do.** No WebRTC peer
    connection is built, no RTP/SRTP/DTLS is spoken,
    and no Encoded Transform runs in the app: the
    transport is the app's own WebSocket fork, and
    the peer-connection side is follow-up work in
    ShieldCallAI, which today has no WebRTC stack.
    No app code changes in this slice. No consent
    ledger validation (item 8). No packet-loss
    concealment synthesis (item 7). No live call has
    been run against this adapter: that needs the app
    client, a device, and a consented test call
    (Mercel), and the first live fork must confirm
    the far-end/near-end leg mapping with a
    known-speaker test, the way the FreeSWITCH
    stereo order still awaits confirmation.

## Consequences

- The fabric table is now complete: eight of eight
  adapters built, none planned. Stage 4's remaining
  work is no longer adapters: Tier 1 live behind the
  cascade, the factorial discourse model, and
  time-to-warning measured end to end.
- The `discontinuous` flag is advisory until a
  consumer uses it: today the pipeline receives the
  frames and the flag is visible in stats, but no
  scoring path branches on it. Unlike FreeSWITCH,
  where the flag is always False because loss is
  invisible, a True here means a real, counted gap;
  a False here means no gap was detected within the
  60 ms window, not that no packet was ever lost
  beyond it.
- `MediaFrame` gains `source_codec` and
  `source_rate`, both defaulted. All existing
  adapters and tests are untouched by the change;
  only this adapter sets the fields. Any code that
  constructs a `MediaFrame` positionally with more
  than eight fields would break, but every
  construction in the tree uses keywords, verified
  before this change landed.
- The in-app route's token travels in the WebSocket
  URL, like every other ingest route's, and will
  appear in logs on both ends. It must be treated
  as a revocable credential: one token per
  integration, rotated on any suspicion of
  exposure, never the sidecar bearer token, never
  reused across environments. The app-build token
  and the Supabase JWT path (follow-up) must not be
  confused: the query token authorises the fork,
  the JWT authorises the user, and only the second
  can say who is speaking.
- The sidecar now depends on the system libopus at
  runtime for this one adapter. A deployment host
  without it still runs every other adapter and the
  REST ingest; only in-app forks are refused, with
  a plain-English reason, at hello time. The
  dependency is a system library, not a pinned
  package, so the deployment image must carry it;
  that is a packaging note, not a code change.
- Kill-criterion note (TDD section 16, Table 31):
  this adapter has no kill criterion of its own; it
  is plumbing. Its success condition is the stage-4
  gate: time to warning measured end to end on a
  real in-app call, which is still blocked on the
  app client and a device (Mercel).

## Follow-ups (not in this slice)

- ShieldCallAI WebRTC client: a peer connection with
  an `RTCRtpScriptTransform` on the receiver (and
  sender, for the near-end leg) that forwards
  `RTCEncodedAudioFrame` data with fork-assigned
  sequence numbers over the `WS /v1/ingest/inapp`
  socket, honouring the app's consent gate before
  any fork starts. This is a project of its own, not
  a single session: the app has no WebRTC dependency
  today.
- App-origin JWT auth over the ingest WebSocket
  (P2-6d for websockets), replacing or supplementing
  the per-integration query token for app-originated
  forks.
- Consent-ledger validation of `consent_ref`
  (TDD section 14.2, stage 5), so the recorded
  reference becomes a checked one.
- Packet-loss concealment: emit `concealed`-flagged
  frames (TDD section 4.1 envelope field) for lost
  packets, down-weighted by the acoustic stream,
  instead of flagging only the next real frame.
  Needs the envelope field, the down-weighting, and
  a measured decision that concealment helps rather
  than invents evidence.
- Wider-band output: decode at 16 kHz with a tested
  decimator for the Channel Passport's use, if the
  passport ever consumes `source_rate` and wants
  more than the 8 kHz pipeline band. Today the
  pipeline runs at 8 kHz and the decoder matches it.
- Stereo / two-leg forks, if the app ever forks
  both legs on one socket; today one fork is one
  leg, by design.
- First live fork: a consented, known-speaker test
  call on a physical device confirming the
  far-end/near-end mapping, the 60 ms window under
  real jitter, and the end-to-end time to warning.

## Verification

- `tests/test_inapp_ingest.py` (54 cases collected
  with parametrization) and
  `tests/test_inapp_route.py` (10 cases): 64 new
  tests, all passing. Full core suite: 654 passed,
  3 skipped (was 590 passed, 3 skipped).
- Protocol cases use a stub decoder, so they run
  without libopus. Real decode is covered by
  round-trip tests that encode sine tones with a
  ctypes Opus encoder in the test file and decode
  them through the real `OpusDecoder` at all four
  encoded rates, asserting frame counts, timing,
  finite samples and non-silent output; they skip
  where libopus is absent. A corrupt Opus packet
  is refused, not decoded to silence.
- Three real catches during testing, all fixed in
  the adapter, not papered over in the tests.
  First, the initial design let the first packet
  set the expected sequence number to itself, so a
  leading out-of-order packet was accepted as the
  start of the stream; sequence numbers now start
  at 0 on every fork, and the first packet waits
  for seq 0. Second, the reorder-window anchor
  first called the clock conversion on a held
  packet, anchoring the whole stream's presentation
  time to an out-of-order arrival; the flush check
  now works in RTP ticks and the base is set only
  by the first packet emitted in sequence order.
  Third, a retransmitted buffered packet was
  decoded and re-buffered instead of dropped,
  because the duplicate check ran only on the
  behind-sequence path; buffered duplicates are now
  checked first. Each catch has a regression test.
- No live decision path changes: CSCF stays the
  production fusion, no sidecar/app threshold
  moves. No live in-app call run yet: that needs
  the app client and a device (Mercel). Nothing
  fabricated, nothing backdated.
