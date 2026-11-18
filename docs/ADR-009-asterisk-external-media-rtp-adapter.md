# ADR-009: Asterisk External Media (RTP) adapter (second Media Ingest Fabric adapter)

## Status

Accepted (2026-11-18). Implements the second slice of TDD v2.0 §4
(Media Ingest Fabric) and roadmap stage 4 ("Integration"), which
orders Twilio and Asterisk first (lowest barrier), then SIPREC.
Asterisk is the first adapter whose audio arrives over UDP/RTP
rather than a WebSocket, so this ADR also defines how the fabric
handles an unauthenticated datagram transport.

## Context

The Twilio slice (ADR-008) gave the fabric its canonical frame
envelope and its first provider adapter. Asterisk is the second
named adapter in the roadmap order. Where Twilio forks audio to a
WebSocket the sidecar can authenticate, Asterisk's External Media
forks audio as plain RTP over UDP: no handshake, no credential, no
call identifier in the packets. The adapter therefore cannot do
what the Twilio adapter does (match a streamSid/callSid inside the
protocol); the binding between a UDP flow and a ShieldCall call has
to be made outside the protocol, by the port the operator points
Asterisk at.

## Protocol, verified against Asterisk's docs

- `POST /channels/externalMedia` (Asterisk REST API, "Channels",
  since 16.6.0/17.1.0, read 2026-11-18): creates a channel to an
  external media source/sink. The transport + encapsulation pair
  selects the channel driver: `udp` + `rtp` selects chan_rtp.
  Parameters: `channelId` (unique id assigned on creation),
  `app` (required, Stasis application), `external_host`
  (hostname/ip:port of the external host), `format` (required,
  format to encode audio in), `transport` (default `udp`),
  `encapsulation` (default `rtp`), `connection_type` (default
  `client`), `direction` (default `both`; the only allowed value).
- Wire format (Asterisk community reports of production External
  Media use, consistent with RFC 3550): RTP v2, one fixed SSRC per
  call, sequence numbers starting at 0 and incrementing by 1 per
  packet, timestamps starting at 0 and incrementing by the payload
  size in samples (160 per 20 ms packet at 8 kHz), 160-byte payloads
  for ulaw/alaw (20 ms at 8 kHz).
- Only codecs with a static RTP payload type work over RTP
  external media (ulaw PT 0, alaw PT 8, gsm PT 3): there is no SDP
  exchange in which to negotiate a dynamic type, so slin16 (dynamic
  PT 118) is not supported over RTP at all. Wideband over Asterisk
  needs the WebSocket transport instead, which is out of scope
  here. This slice therefore pins `format` to `ulaw` or `alaw` at
  8 kHz and refuses anything else.
- Symmetric RTP: the external host must send any return audio from
  the same IP:port it listens on (Asterisk's `strictrtp` learns the
  peer from the first packet). This slice is receive-only: the
  sidecar never sends RTP back to Asterisk, so no return path is
  built. The ARI `direction` is still `both` (the API allows no
  other value); Asterisk tolerates a peer that only listens.

## Decision

1. **Shared G.711 codecs (`shieldcall/ingest/g711.py`).** The
   mu-law table lived inside `twilio.py`; both adapters need it,
   and Asterisk adds A-law. Both tables now live in one module
   with `mulaw_decode` and `alaw_decode` (exact ITU-T G.711,
   float32 in [-1, 1]). `twilio.py` re-exports `mulaw_decode` and
   `_MULAW_TABLE` so existing imports keep working; no behaviour
   change to the Twilio adapter. A-law known vectors pinned by
   tests: 0xD5 -> +8/32768, 0x55 -> -8/32768, 0xA5 ->
   +16896/32768, 0x25 -> -16896/32768.

2. **Adapter is pure (`shieldcall/ingest/asterisk.py`).**
   `AsteriskRtpStream` takes raw UDP datagrams and returns
   canonical frames. It opens no socket, holds no credential, and
   knows nothing about HTTP or ARI. Construction binds the
   stream to a call: `call_id` is the pre-opened ShieldCall call
   id, which the operator sets to the externalMedia `channelId`
   (the same pattern as Twilio's Call SID). `channel_id` is the
   externalMedia channel id and becomes the frame `stream_sid`;
   it defaults to `call_id`. `format` is `ulaw` or `alaw`;
   `track` defaults to `inbound` (one externalMedia channel
   carries one leg; the operator creates one listener per leg,
   typically by snooping the PJSIP channel with `spy: in` for the
   caller leg).

3. **RTP parsing follows RFC 3550, strictly.** Version must be
   2; CSRC count, extension header, and padding are honoured
   (payload starts after 12 + 4*CC + extension bytes; trailing
   padding bytes are stripped). Payload type must match the
   configured format (0 for ulaw, 8 for alaw); anything else,
   including GSM (PT 3), is an `AsteriskError`, never a silent
   transcode. The first packet pins the SSRC; a different SSRC
   later is an error, because a new SSRC means a different
   stream, not a continuation. A zero-length payload after
   stripping is an error.

4. **Continuity is judged from sequence number and timestamp,
   both with wraparound arithmetic.** The RTP sequence number is
   the loss detector (expected = last + 1 mod 65536); the
   timestamp is the presentation clock (expected = last + samples
   in the last payload, mod 2^32). A forward jump in either is a
   gap: the next frame is flagged `discontinuous`, the lost
   samples are counted (from the sequence jump, or from the
   timestamp jump when the sequence is continuous but the clock
   jumped), and any partial buffered frame is discarded rather
   than spliced across the two timelines. A duplicate sequence
   number or a timestamp behind the previous payload's end is
   dropped, never replayed. `t_sec` is measured from the first
   packet's timestamp, so an arbitrary start timestamp is fine.
   Payloads are 20 ms (160 bytes) in practice; any whole number
   of 10 ms frames is accepted and a trailing partial frame
   buffers for the next packet, exactly like the Twilio adapter.
   No gap is ever filled with invented audio.

5. **State machine, fail closed.** Not-started -> streaming ->
   stopped. The stream starts on the first valid packet (RTP has
   no control messages). `stop()` discards a partial trailing
   frame, counts it, and marks the stream stopped; any datagram
   after `stop()` raises `AsteriskError`. A malformed packet, a
   version or payload-type mismatch, or an SSRC change raises;
   the caller closes the stream. The marker bit is recorded in
   stats and otherwise ignored: it marks a talkspurt boundary,
   not a discontinuity.

6. **Serving layer (`shieldcall/serve/asterisk_udp.py`).** RTP
   arrives on UDP, which does not fit the FastAPI app, so the
   listener is a standalone asyncio `DatagramProtocol`:
   `AsteriskUdpListener` binds one UDP port per call, feeds each
   datagram to its `AsteriskRtpStream`, and hands frames to an
   `on_frames` callback. `attach_asterisk_listener` wires that
   callback to a `SidecarRuntime` session exactly like the Twilio
   route (concatenate frames, `sess.push_audio(samples, 8000)`),
   and refuses to start when the call does not already exist
   (P2-6b: no implicit call creation). A protocol error stops
   the listener rather than guessing past it.

7. **RTP has no authentication, and the ADR says so plainly.**
   Anyone who can reach the UDP port can inject audio, and a
   datagram carries no call identifier, so the binding is the
   port itself. The controls are network-level, not protocol:
   the listener binds a per-call port, the operator firewalls
   that port to the Asterisk host only, and the listener accepts
   an optional `allowed_peer` (Asterisk's ip:port) from which
   datagrams from any other source are dropped and counted.
   Without `allowed_peer` or a firewall rule, the port must be
   treated as untrusted. This is a property of RTP, not a gap
   the adapter can close, and it is recorded here and in the listener's
   docstring, where the operator will actually read it.

## Consequences

- The fabric's second transport is covered: WebSocket (Twilio)
  and UDP/RTP (Asterisk). SIPREC is next; it reuses the frame
  envelope and the gap/buffer discipline, with its own session
  negotiation.
- No live decision path changes: CSCF stays the production
  fusion, frames flow into the same session and pipeline as
  every other ingest path, and the sidecar is listen-only on
  this path (no audio is ever sent back to Asterisk).
- No live Asterisk run yet: that needs an Asterisk server with
  ARI enabled, a Stasis app that creates the externalMedia
  channel and snoops the call leg, and the operator's
  `external_host` pointing at the listener port (Mercel).
  The listener itself is tested over loopback UDP.
- The A-law table is shared with nothing else yet; the Telnyx
  adapter (PLANNED) will reuse `g711.py` when its turn comes.
