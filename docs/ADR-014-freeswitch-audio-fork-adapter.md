# ADR-014: FreeSWITCH mod_audio_fork adapter (seventh Media Ingest Fabric adapter)

## Status

Accepted (2026-11-23). Implements the seventh slice of TDD v2.0
section 4 (Media Ingest Fabric) and roadmap stage 4
("Integration"). The fabric table lists eight adapters; Twilio,
Asterisk, SIPREC, Telnyx, Vonage and Amazon Connect are built,
FreeSWITCH was PLANNED. This ADR adds FreeSWITCH and leaves
in-app VoIP/WebRTC planned.

## Context

FreeSWITCH is the self-hosted PBX case in the fabric table, and
its wire format is unlike every adapter built so far in one
specific way: there is no handshake message defined by the
protocol. Twilio sends `connected` and `start`, Telnyx sends
`start`, Vonage sends `websocket:connected`, and each of those
declares the format before any audio arrives. mod_audio_fork
sends exactly one text frame and then raw audio, and the text
frame is whatever the operator typed into the `start` command.
The format of the audio that follows, mono or stereo, 8 kHz or
16 kHz, exists only in that operator-chosen text, or nowhere at
all. An adapter written by renaming the Vonage events would wait
for a `websocket:connected` that never comes, and an adapter
that assumed 8 kHz mono would silently mis-decode a 16 kHz
stereo fork: interleaved channels read as one channel, samples
at half speed, and a warning path fed garbage it cannot detect.
The adapter below is written from the module's own README, and
the format is pinned from the one frame the protocol does
provide before any audio is decoded.

mod_audio_fork is a third-party module, not part of FreeSWITCH
itself. The original drachtio repository is gone; community
mirrors carry the source, and secondary reports say building it
against FreeSWITCH 1.10 and later needs patches, where 1.6 and
1.8 worked natively. That is an operator build problem, not an
adapter problem: the adapter depends only on the wire format
the README defines, whichever mirror the operator builds from.
No live FreeSWITCH fork has been run against this adapter yet;
that needs a FreeSWITCH server with the module loaded, which is
Mercel-side (item 11). A different, newer module,
mod_audio_stream, speaks a different protocol (JSON
`streamAudio` messages) and is not covered here; if an operator
uses it, this adapter does not apply.

## Decision

1. **Canonical frame envelope, unchanged.** FreeSWITCH audio
   becomes `MediaFrame` (ADR-008): 10 ms mono float32 at 8 kHz,
   with `call_id`, `stream_sid`, `track`, per-track `seq`,
   `t_sec`, `discontinuous`, and `source="freeswitch"`. The
   analysed leg is `track="inbound"`. Nothing downstream
   branches on `source`.

2. **Adapter is pure (`shieldcall/ingest/freeswitch.py`).**
   `FreeSwitchMediaStream` takes raw text frames
   (`handle_text`) and raw binary audio (`handle_audio`) and
   returns frames. It opens no socket, holds no credential,
   and imports nothing from the serving layer, so the
   protocol translation is testable without a network. The
   serving layer owns auth and the socket; see item 10. The
   split into two entry points is forced by the protocol:
   text frames are operator metadata, binary frames are raw
   L16, and the adapter must not guess a message's kind from
   its bytes. Unlike the Vonage route, the text frame is not
   parsed into JSON by the route: only the first text frame
   has a defined shape, later text frames are arbitrary
   operator data, and the distinction lives in the adapter.

3. **Protocol, verified against the module's own README**
   (andrewvmail/freeswitch-modules, `mod_audio_fork/README.md`,
   read 2026-11-23), corroborated by two independent
   deployment write-ups (AMDY's FreeSWITCH integration guide
   and a mod_audio_fork streaming guide, both read the same
   day). The operator starts a fork with:
   `uuid_audio_fork <uuid> start <wss-url> <mix-type>
   <sampling-rate> <streamID> <accID> <callSid> <track>
   <metadata>`.
   - `mix-type`: `mono` (one channel, the caller's audio),
     `mixed` (one channel, both legs summed), or `stereo`
     (two channels, caller audio in one, callee audio in the
     other). The README does not state which stereo channel
     is which; see item 6.
   - `sampling-rate`: `8k` or `16k`. FreeSWITCH resamples
     the leg to the requested rate.
   - `streamID`, `accID`, `callSid`: operator-chosen
     identifiers for the stream, the account, and the call.
   - `track`: `both_tracks`, `inbound_track` or
     `outbound_track`, marked experimental in the README. It
     selects which legs the module forks; it is not part of
     this adapter's contract, which reads the wire shape
     from `mix-type` in the metadata (item 4).
   - `metadata`: a text frame of arbitrary data, sent
     immediately upon connecting. Once it has been sent, the
     incoming audio arrives in binary frames. The README's
     drachtio example passes caller data as JSON; AMDY's
     integration passes `{"config": {"sample_rate": 8000,
     "VID": "..."}}`. Both confirm the first frame is
     operator-defined JSON in practice, and that no other
     start message exists.
   - `uuid_audio_fork <uuid> send_text <metadata>` sends a
     text frame mid-call, for example to notify the server
     of a DTMF event. There is no defined shape for these
     frames.
   - `uuid_audio_fork <uuid> stop <metadata>` sends a final
     text frame and then closes the connection. There is no
     stop event distinct from any other text frame; the end
     of the stream is the socket closing.
   - From the server to the module: JSON text frames with
     `event` set to `audio` (playback), `killAudio`,
     `transcription`, `transfer`, `disconnect` or `error`.
     This adapter sends none of them (item 10).
   - Binary frames are raw L16: 16-bit signed little-endian
     PCM, no header, no container. Deployment write-ups
     agree on 20 ms chunks: 320 bytes at 8 kHz mono, 640
     bytes at 16 kHz mono.
   - The module advertises the WebSocket sub-protocol
     `audio.drachtio.org` by default, overridable with the
     `MOD_AUDIO_FORK_SUBPROTOCOL_NAME` environment variable.
     The sub-protocol is not used for authentication in this
     slice; the query token is (item 10).
   - The module sends no custom HTTP headers on the WebSocket
     upgrade (AMDY's guide states this plainly, and
     authorizes by IP allowlist for that reason). A
     header-based credential is therefore impossible, and
     the token rides in the URL query string (item 10).

4. **Format is pinned by the first text frame, under a
   contract this ADR defines, because the protocol defines
   none.** The operator's metadata must be a JSON object
   declaring the wire shape:
   `{"sample_rate": 8000, "mix_type": "mono"}`. It may also
   carry `"call_id"` (checked against the route's binding,
   item 9) and `"stream_id"` (recorded as `stream_sid`), and
   any other fields, which are recorded for audit and
   otherwise ignored. `sample_rate` accepts `8000`, `"8000"`
   or `"8k"`; `mix_type` accepts `"mono"` or `"stereo"`.
   A first frame that is not JSON, not an object, or missing
   either field refuses the stream with a `FreeSwitchError`
   naming the missing piece, before any audio is decoded.
   The adapter cannot know what the bytes mean without this
   declaration, and guessing 8 kHz mono would mis-decode
   every other configuration silently. The operator writes
   the metadata; the ADR gives the exact shape, and the
   error message repeats it. Later text frames never re-pin
   the format: they are recorded verbatim as text events and
   produce no frames, whatever JSON they happen to contain.

5. **Only 8 kHz is accepted, and `mixed` is refused.** A
   `sample_rate` of 16000 (or 24000, which no fork command
   offers) is refused with a `FreeSwitchError`, never
   silently resampled: the canonical envelope and the
   pipeline run at 8 kHz, and a 16 kHz to 8 kHz decimation
   is a DSP claim that needs its own tested filter, not a
   one-line slice of every second sample. The operator
   chooses `8k` in the start command; the TDD fabric table
   lists FreeSWITCH as "L16 over WebSocket" without pinning
   a rate, so 8 kHz is within the specified range.
   Accepting 16 kHz with a tested decimator is a follow-up,
   not a silent fallback. `mix_type` `mixed` is refused at
   the metadata frame: one summed channel cannot be split
   into legs, and scoring it as inbound would score the
   protected user's own voice acoustically, which TDD
   section 4.1 forbids (only the far end is scored; the
   protected user's voice is never analysed acoustically).
   That follows the SIPREC precedent (ADR-010), where mixed
   streams are refused, not misattributed. The operator
   uses `mono` or `stereo`. No two-speaker segmentation
   exists yet to make `mixed` safe.

6. **Stereo is de-interleaved; channel 0 is the analysed
   leg, and the order is flagged unverified.** A stereo
   packet is interleaved L16 pairs: sample 0 of channel 0,
   sample 0 of channel 1, and so on. The adapter splits
   the pairs, emits frames for channel 0 as `inbound`, and
   counts channel 1's samples without ever turning them
   into frames, following the Telnyx pattern for the
   unanalysed leg. The README says stereo carries caller
   audio in one channel and callee audio in the other
   without stating the order; a deployment write-up reports
   channel 0 as the caller and channel 1 as the agent.
   Channel 0 as inbound is taken from that report, and it
   is flagged here as unverified against a live fork: the
   first live run must confirm it with a known-speaker test
   call, and if the order is reversed the mapping is a
   one-line change in the de-interleave step. Both channels'
   sample counts are kept in stats, so a reversal is
   detectable from the record, not from a hunch. `mono`
   has one channel and it is the analysed leg: the operator
   attaches the fork to the leg to be scored, and the ADR's
   operator instructions say so.

7. **There is no clock, so there is no gap detection, and
   that limit is recorded here.** The wire carries no
   sequence number, no timestamp, and no reconnect
   announcement: a new connection sends the same metadata
   frame as a first connection, and nothing distinguishes
   a reconnect from a first connect. A lost packet is
   therefore indistinguishable from silence that was never
   sent, and a reordered packet cannot be recognised at
   all. Frames are emitted in arrival order with `t_sec`
   counted from the samples received (sample count / 8000,
   seconds from the first audio), and `discontinuous` is
   always False, because no signal on this wire ever earns
   a True. No gap is ever filled with invented audio, and
   no `lost_samples` stat is kept, because there is nothing
   to count it from. A consumer that needs loss accounting
   must not assume this adapter provides it. This is the
   same limit as Vonage (ADR-012), with one difference:
   Vonage at least announces reconnections, so its first
   frame after a reconnect can be flagged; FreeSWITCH
   cannot, and the flag stays False even there.

8. **State machine, fail closed.** Metadata text frame ->
   audio* / text events* -> socket close. Audio before the
   metadata frame, a metadata frame that is not JSON or
   lacks the format fields, an unsupported rate, a `mixed`
   fork, a metadata `call_id` that disagrees with the
   route's binding, an empty binary packet, a mono packet
   with an odd byte count (a truncated 16-bit sample), or
   a stereo packet whose byte count is not a multiple of
   four (a truncated interleaved pair) all raise
   `FreeSwitchError`. The adapter never recovers by
   guessing; the caller closes the stream. A partial
   trailing frame (fewer than 80 buffered samples) is
   discarded when the stream ends, not padded: padding
   would invent audio the caller never sent. The count is
   kept in stats as `discarded_on_close` so a short final
   packet is visible, not silent. Ending the stream is
   `finish()`, called by the route when the socket closes;
   it is idempotent, because a close is cleanup, not a
   protocol event, and a second call must not raise.
   Messages after `finish()` raise. Text frames that
   arrive after the metadata frame, including the final
   frame a `stop` command sends, are recorded verbatim in
   `text_events` and counted; they are never parsed as
   protocol, because the module defines no shape for them,
   and a DTMF notification in one must not be mistaken for
   a format change.

9. **Call binding rests on the connection URI, with one
   in-band check the other raw-binary adapter does not
   have.** The module sends no call identifier of its own:
   `streamID`, `accID` and `callSid` are values the
   operator chose when starting the fork, and they travel
   only inside the operator's metadata, if the operator
   put them there. The route therefore binds the stream
   the way the Twilio, Telnyx and Vonage routes do: the
   operator pre-opens the call with POST /v1/calls (P2-6b)
   and puts that call_id in the WebSocket URI as a query
   parameter, using a distinct URI per call. Where
   FreeSWITCH is stronger than Vonage is the metadata
   check: when the metadata carries `call_id`, the adapter
   refuses the stream unless it equals the route's
   call_id, so a fork started with the wrong call's
   metadata cannot be routed into another call's session.
   When the metadata carries no `call_id`, the binding
   rests on the URI alone, and a reused URI would route
   one call's audio into another call's session, which no
   code in this slice could tell. The operator must use a
   distinct URI per call and, where the dialplan allows
   it, include `call_id` in the metadata so the check
   runs. This is stated here rather than hidden, because
   it is an operator obligation the JSON adapters do not
   have.

10. **Serving route (`WS /v1/ingest/freeswitch`).** The
    module opens the WebSocket from its own side, so the
    credential travels as a `token` query parameter in the
    `wss-url` passed to the `start` command, the same
    shape as the Twilio, Telnyx and Vonage routes. The
    token is a single-purpose, per-integration credential
    from `SHIELDCALL_FREESWITCH_TOKENS` ("id:value,..."),
    never the sidecar bearer token. With no FreeSWITCH
    tokens configured the route refuses connections unless
    the local-dev escape hatch is on. Auth is checked
    before the call gate (4401, then 4404), and the call
    must already exist via POST /v1/calls (P2-6b). The
    route reads both text and binary WebSocket messages
    (`ws.receive()`), as the Vonage route does. It sends
    nothing back over the socket: no session events, no
    error JSON. That is deliberate and differs from the
    other routes on purpose. mod_audio_fork treats JSON
    text frames from the server as control events
    (playback, kill, transfer, disconnect), so anything
    ShieldCall sent back could be read as an instruction
    to play audio to the caller or move the call. On a
    protocol error the route closes with code 4400 and the
    reason in the close reason, which carries the message
    without sending a frame the module could act on. The
    stream is recommend-only like every other ingest path:
    audio flows into the same pipeline and session,
    nothing hangs up, and no playback, transfer or
    disconnect command is ever issued. The operator
    should firewall the route to the FreeSWITCH host in
    addition to the token, since the module cannot send
    headers and the token appears in the dialplan and in
    FreeSWITCH logs: one token per integration, rotated
    on any suspicion of exposure.

11. **What this does not do.** No audio is sent back to
    FreeSWITCH: no playback, no transcription, transfer or
    disconnect events. That is deliberate: this session's
    scope is ingest, and a playback path would need the VOI
    challenge gate (TDD section 11) decided first. No
    dialplan, ESL client, or call-control caller is
    shipped: the operator writes the `uuid_audio_fork
    start` command with the metadata JSON in item 4, the
    pre-opened call_id and the per-integration token in
    the URL. No live FreeSWITCH call has been run against
    this adapter yet; that needs a FreeSWITCH server with
    mod_audio_fork loaded, a number or SIP trunk, and the
    operator's dialplan, all Mercel-side. The
    `MOD_AUDIO_FORK_SUBPROTOCOL_NAME` sub-protocol is not
    verified in this slice; it identifies the module, not
    the caller, and verifying it would add nothing the
    token does not already provide.

## Consequences

- The fabric now covers seven of eight providers, and the
  second whose wire format is raw binary rather than
  framed JSON. The route's move from `receive_json()` to
  `receive()` is confined to the Vonage and FreeSWITCH
  routes; the Twilio and Telnyx routes are untouched.
- The `discontinuous` flag is advisory until a consumer
  uses it: today the pipeline receives the frames and the
  flag is visible in stats, but no scoring path branches
  on it. For FreeSWITCH the flag is always False, and
  that must not be read as "no loss occurred": loss on
  this wire is invisible by construction.
- The per-integration token appears in the `wss-url` in
  the dialplan and therefore in FreeSWITCH logs. Operators
  must treat it as a revocable credential: one token per
  integration, rotated on any suspicion of exposure, never
  the sidecar bearer token, never reused across
  environments. Firewalling the route to the FreeSWITCH
  host is the second factor the module's lack of headers
  makes necessary.
- Kill-criterion note (TDD section 16, Table 31): this
  adapter has no kill criterion of its own; it is plumbing.
  Its success condition is the stage-4 gate: time to
  warning measured end to end on a real FreeSWITCH call,
  which is still blocked on a live server and dialplan
  (Mercel).

## Follow-ups (not in this slice)

- Accept `audio` at 16 kHz with a tested anti-aliased
  decimator to 8 kHz, if an operator needs the `16k`
  fork rate; measure the filter against known vectors
  before trusting it on the warning path.
- Verify the stereo channel order on the first live fork
  with a known-speaker test call (item 6); swap the
  de-interleave mapping if channel 0 proves to be the
  callee leg.
- Two-speaker segmentation for mixed tracks (TDD section
  4.1), which would let a `mixed` fork be split before
  scoring instead of refused.
- A mod_audio_stream adapter, if an operator standardises
  on that module instead of mod_audio_fork; it is a
  different protocol and needs its own ADR.
- Bidirectional audio / playback only after the VOI
  challenge gate (TDD section 11) is decided.
- Sub-protocol verification, if an operator deploys
  behind a proxy that requires it; the token remains the
  authentication mechanism either way.

## Verification

- Tests in `tests/test_freeswitch_ingest.py`: L16 known
  vectors and round trips (including negative full-scale),
  metadata contract (valid, non-JSON, non-object, missing
  fields, rate spellings, 16 kHz refusal, `mixed`
  refusal, unknown mix type, `call_id` match and mismatch,
  extra fields recorded), framing and buffering across
  packet boundaries (20 ms packets, split frames, partial
  trailing frame discarded on finish), stereo
  de-interleaving with the unanalysed leg counted and
  never framed, truncated pair and odd byte refusals,
  audio before metadata, text events recorded verbatim
  without re-pinning the format, messages after finish
  rejected.
- Tests in `tests/test_freeswitch_route.py`: token auth
  (required, wrong, absent-with-no-tokens), pre-opened-call
  gate, full metadata / binary / close flow into a live
  session with nothing sent back over the socket, metadata
  mismatch and malformed metadata closing with 4400,
  capabilities listing.
- Full core suite green after the change (see the session log).
