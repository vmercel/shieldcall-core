# ADR-010: SIPREC adapter (third Media Ingest Fabric adapter)

## Status

Accepted (2026-11-19). Implements the third slice of TDD v2.0 §4
(Media Ingest Fabric) and roadmap stage 4 ("Integration"), which
orders Twilio and Asterisk first (lowest barrier), then SIPREC.
SIPREC is the first adapter whose media is negotiated in-band by
signalling the sidecar itself must speak: the Session Recording
Client (SRC, e.g. a PBX or SBC) opens a Recording Session (RS)
with the sidecar as Session Recording Server (SRS) over SIP, and
the metadata that says who is speaking on which stream arrives
in the INVITE body. This ADR therefore also defines the fabric's
first signalling listener.

## Protocol, verified against the RFCs (read 2026-11-19)

- RFC 7865 (Session Recording Protocol) and RFC 7866 (Session
  Recording Metadata). The metadata model and the complete example
  in RFC 7866 §8.1 were checked against
  draft-ietf-siprec-metadata-20 (the draft that became RFC 7866);
  the wire shapes below match that example exactly.
- A Recording Session is established by a SIP INVITE from the
  SRC to the SRS (RFC 7865 §6). The SRC offers SDP (RFC 3264
  offer/answer); the SRS MUST answer with the same number of
  m-lines as the offer. The SRC uses `a=sendonly` (it does not
  expect media back); the SRS answers `a=recvonly`. Every m-line
  carries `a=label` (RFC 4574) identifying the stream, and the
  metadata refers to that label. An INVITE is a recording session
  only when it carries the `siprec` option tag and the `+sip.srs`
  feature tag (RFC 7865 §6.1).
- Metadata (RFC 7866, namespace
  `urn:ietf:params:xml:ns:recording:1`, content type
  `application/rs-metadata+xml`) is carried in the INVITE body,
  usually as one part of a `multipart/mixed` body beside the SDP
  offer, and updated later by re-INVITE or UPDATE. Root element
  `<recording>`; `<datamode>complete</datamode>` for a full
  snapshot or `partial` for an update carrying only what changed
  (RFC 7866 §8.2: a participant plus a `participantsessionassoc`
  carrying a `disassociate-time`). `<session session_id>` is the
  recording session; `<participant participant_id>` carries
  `<nameID aor="sip:...">` and a display `<name>`;
  `<stream stream_id session_id>` carries a `<label>`;
  `<participantstreamassoc participant_id>` carries `<send>` and
  `<recv>` stream ids and associate/disassociate times. All ids
  are base64Binary in the schema and are treated here as opaque
  strings, never decoded. Unknown elements in other namespaces
  are ignored (the schema allows `xs:any`), so a vendor
  extension never breaks parsing.
- RTP (RFC 7865 §8, RFC 7866 §8): symmetric RTP is REQUIRED;
  RTCP support is REQUIRED. The SSRC MAY change if a conflict is
  discovered or a program restarts, and receivers SHOULD use the
  RTCP CNAME to keep track of a participant across such a change.
  The SRC maps each communication-session CNAME/SSRC to a unique
  recording-session CNAME/SSRC. This slice does not parse RTCP:
  it re-pins the SSRC on change (see decision 4) and does not
  need the CNAME, because one label is one RTP session and the
  label, not the SSRC, is the stream identity.

## Decision

1. **Shared RTP machinery (`shieldcall/ingest/rtp.py`).** The
   RTP parsing (`_parse_rtp`), wraparound arithmetic
   (`_seq_diff`, `_ts_diff`), and the continuity/buffer/drain
   state machine move out of `asterisk.py` into a generic
   `RtpStream` base class, parameterised by payload type,
   decode function, frame source name, and SSRC-change policy.
   `AsteriskRtpStream` becomes a thin subclass with the same
   constructor and behaviour, and `asterisk.py` re-exports the
   moved names, so no existing import changes. The Twilio
   adapter keeps its own JSON framing and does not use this
   module. No behaviour change to the Asterisk adapter: the
   full suite must pass unchanged.

2. **Metadata is pure (`shieldcall/ingest/siprec.py`).**
   `parse_siprec_metadata` turns an
   `application/rs-metadata+xml` document into a
   `SiprecMetadata` value: `datamode` (`complete` or `partial`,
   default `complete`), sessions, participants (aor, display
   name), streams (label, session), and per-participant send /
   recv stream sets from `participantstreamassoc`. Fail-closed
   validation: the root must be `recording` in the recording
   namespace (or with no namespace at all); `datamode` must be
   one of the two values; a stream with no `<label>` is
   rejected, because a labelless stream cannot be bound to an
   SDP m-line; duplicate labels are rejected; an association
   naming an unknown stream or participant is rejected; a
   stream whose `session_id` names no declared session is
   rejected. A stream with no `session_id` binds to the
   document's single session, and is rejected as ambiguous
   when the document carries more than one.

3. **One recording session is one ShieldCall call.**
   `SiprecRecordingSession` is keyed by the recording
   `session_id`, which becomes the ShieldCall `call_id`, the
   same pattern as the Twilio Call SID and the Asterisk
   channelId. The operator names the scored participant by
   AOR (`scored_aor`, e.g. the remote caller): a stream whose
   senders are exactly that participant is the `inbound`
   track; a stream whose senders are all other participants
   is `outbound`. A stream with no senders, or whose senders
   mix the scored participant with others, is a mixed or
   unattributable stream: the adapter records it and emits no
   frames from it, because attributing mixed audio to one leg
   would lie about whose voice was scored. The operator must
   configure the SRC for separate streams; a mixed-only
   recording is visible in stats, not silently scored.
   `apply_metadata` handles both snapshot modes: `complete`
   replaces the snapshot, `partial` merges it, and a
   `disassociate-time` on a participant or association ends
   that participant's streams (their RTP stream is stopped;
   later datagrams on its label are rejected).

4. **RTP per label, SSRC re-pin.** Each stream is one
   `SiprecRtpStream` (an `RtpStream` with source `"siprec"`),
   driven by `handle_rtp(label, datagram)`. The label, not
   the SSRC, is the stream identity, so an SSRC change
   re-pins the stream (continuity state reset, next frame
   flagged `discontinuous`, counted in stats) instead of
   raising, as RFC 7865 §8 anticipates. Everything else
   follows the Asterisk discipline exactly: sequence number
   is the loss detector, timestamp the presentation clock,
   both with wraparound arithmetic; gaps are flagged
   `discontinuous` and counted, never filled; duplicates and
   out-of-order packets are dropped; a partial frame is
   discarded, not spliced, across a gap; `stop()` discards a
   partial trailing frame. Only PCMU (PT 0) and PCMA (PT 8)
   at 8 kHz are accepted in this slice; a dynamic payload
   type is a `SiprecError`, never a silent transcode.
   Datagrams on an unknown label raise `SiprecError`; the
   caller closes the session. Datagrams on a refused (mixed)
   stream are dropped and counted, not raised: the refusal
   is a configured policy, not a protocol violation.

5. **Signalling is a minimal SIP server, not a SIP stack
   (`SiprecSipHandler`, pure; `shieldcall/serve/siprec_sip.py`
   owns the socket).** The handler takes raw SIP messages as
   text and returns response text, so it is testable without
   a socket. It handles INVITE, ACK, BYE, CANCEL, and
   OPTIONS, and answers 405 to anything else, 400 to a
   malformed request. An INVITE without the `siprec` option
   tag (in `Require` or `Supported`) is not a recording
   session and gets 488 Not Acceptable Here. The body is
   parsed as `multipart/mixed` (SDP + metadata parts) or as
   a single SDP or metadata body. The SDP offer is parsed
   (`m=audio`, `a=label`, `a=rtpmap`, `a=sendonly`); every
   m-line must carry a label, and every SDP label must have a
   metadata stream, or the INVITE gets 488. A metadata stream
   with no SDP m-line is fine: it simply never receives media.
   The 200 OK answers with the same number of m-lines,
   `a=recvonly`, the listener's RTP ports, the echoed labels
   and rtpmap lines, and `Supported: siprec`. A non-audio
   m-line is answered with port 0 (rejected per RFC 3264),
   never silently accepted. A re-INVITE or UPDATE carrying
   new metadata merges it into the live session. BYE stops
   the session and all its RTP streams and answers 200 OK.
   ACK needs no response. The handler allocates RTP ports
   through an injected allocator, so tests never bind a
   socket.

6. **Serving layer (`shieldcall/serve/siprec_sip.py`).**
   `SiprecSipListener` is a standalone asyncio
   `DatagramProtocol` on the SIP port (default 5060,
   configurable): it feeds each datagram to the handler and
   sends the response back to the sender. Each label gets
   its own `SiprecRtpListener` on an allocated UDP port,
   feeding `SiprecRecordingSession.handle_rtp`. Frames are
   pushed into the runtime session exactly like the Twilio
   and Asterisk paths (concatenate, `push_audio` at 8 kHz).
   The INVITE is the call-opening act: unlike Twilio and
   Asterisk, the operator cannot pre-open the call, because
   the recording `session_id` is generated by the SRC. The
   listener therefore opens the runtime call itself, with
   `rt.open_call(session_id)`, when a valid INVITE arrives.
   The gate that replaces the pre-open rule is the network:
   the listener accepts SIP only from a configured
   `allowed_srcs` set (the SRC's addresses); anything else
   is dropped and counted, and SIP Digest authentication is
   NOT implemented in this slice (logged as a follow-up).
   The RTP listeners likewise drop datagrams from any peer
   other than the SRC that sent the INVITE. Without
   `allowed_srcs` or a firewall rule, both ports must be
   treated as untrusted. This is a property of SIP/RTP, not
   a gap the adapter can close, and it is recorded here and
   in both listeners' docstrings.

## Consequences

- The fabric's third transport is covered: WebSocket
  (Twilio), UDP/RTP (Asterisk), and SIP + RTP (SIPREC).
  SIPREC reuses the frame envelope, the gap/buffer
  discipline, and the push-audio serving pattern; only the
  session negotiation is new.
- No live decision path changes: CSCF stays the production
  fusion, frames flow into the same session and pipeline as
  every other ingest path, and the sidecar is listen-only
  (it never sends RTP back to the SRC, and never sends SIP
  except responses to the SRC's requests).
- No live SIPREC run yet: that needs an SRC (PBX/SBC) the
  operator can point at the listener, configured for
  separate streams with the scored participant's AOR known
  to the operator, plus the `allowed_srcs` allowlist (Mercel).
  The handler and the RTP routing are tested without
  sockets; the UDP listeners are tested by driving
  `datagram_received` directly, as the Asterisk tests do.
- Follow-ups, not in this slice: SIP Digest authentication
  (RFC 3261 §22) on the listener; RTCP parsing for CNAME
  tracking across SSRC changes; wideband codecs via dynamic
  payload types; Telnyx (next adapter in the fabric table).
