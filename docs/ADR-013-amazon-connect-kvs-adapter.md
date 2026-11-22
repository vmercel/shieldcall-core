# ADR-013: Amazon Connect live media streaming adapter (sixth Media Ingest Fabric adapter)

## Status

Accepted (2026-11-22). Implements the sixth slice of TDD v2.0 section 4
(Media Ingest Fabric) and roadmap stage 4 ("Integration"). The
fabric table lists eight adapters; Twilio, Asterisk, SIPREC,
Telnyx and Vonage are built, Amazon Connect was PLANNED. This
ADR adds Amazon Connect and leaves FreeSWITCH and in-app
VoIP/WebRTC planned.

## Context

Amazon Connect is unlike every adapter built so far, and the
difference is architectural, not cosmetic. Twilio, Telnyx and
Vonage all push audio to the sidecar over a WebSocket the
provider opens. Amazon Connect pushes nothing to the sidecar.
It streams call audio into Amazon Kinesis Video Streams (KVS),
and a consumer pulls it back out with the KVS GetMedia API.
An adapter written as another WebSocket route would wait
forever for a connection AWS never makes. The adapter below
is written from Amazon's own docs, and the pull side that
cannot be built without AWS credentials is stated as a
follow-up, not faked.

## Decision

1. **Canonical frame envelope, unchanged.** Connect audio
   becomes `MediaFrame` (ADR-008): 10 ms mono float32 at
   8 kHz, with `call_id`, `stream_sid`, `track`, per-track
   `seq`, `t_sec`, `discontinuous`, and `source="amazon_connect"`.
   Nothing downstream branches on `source`.

2. **Adapter is pure (`shieldcall/ingest/amazon_connect.py`).**
   `AmazonConnectStream` takes one complete MKV fragment at a
   time (`feed_fragment`) and returns frames. It opens no
   socket, holds no credential, makes no AWS call, and
   imports nothing from the serving layer, so the protocol
   translation is testable without a network or an AWS
   account. The serving layer owns auth and the HTTP route;
   see item 9. The KVS consumer loop that fetches fragments
   from GetMedia and hands them to this adapter is NOT
   shipped in this slice (item 10).

3. **Protocol, verified against Amazon's docs** (read
   2026-11-22), not from memory:
   - "Enable live media streaming in your Connect Customer
     instance" (docs.aws.amazon.com/connect, Admin Guide):
     live media streaming is off by default; the operator
     enables it under Data storage, sets a KVS stream prefix
     and a KMS key, and adds Start/Stop media streaming
     blocks to the contact flow, choosing audio from the
     customer, to the customer, or both.
   - "Contact attributes for live media streaming in Kinesis
     Video Streams": the flow exposes
     `$.MediaStreams.Customer.Audio.StreamARN`,
     `StartTimestamp`, `StopTimestamp` and
     `StartFragmentNumber` for the customer audio stream.
     The operator passes the stream ARN and the start
     fragment number to the consumer, typically through an
     Invoke Lambda block.
   - "Develop live media streaming in Connect Customer":
     Amazon's own consumer example is the
     `connect-kvs-consumer-demo` repository, which reads
     the `AUDIO_TO_CUSTOMER` and `AUDIO_FROM_CUSTOMER`
     tracks published by Connect and stores them as raw
     PCM files.
   - The demo README: the two raw files are listened to by
     importing them into Audacity as Signed 16-bit PCM at
     8000 Hz. The payloads in the fragments are L16 PCM,
     whatever the track header claims (item 5).
   - The fragment structure itself is the Matroska (MKV)
     format KVS uses for all of its media, confirmed by the
     fragment dump in the independent Python consumer
     write-up (Neudeck, 2024): one EBML header, one
     Segment with Info (`TimecodeScale` 1000000),
     Tracks, Tags and Clusters; each Cluster has a
     `Timecode` and repeated `SimpleBlock` elements; Tags
     carry `AWS_KINESISVIDEO_FRAGMENT_NUMBER`,
     `ContactId`, `InstanceId` and a `MimeType` of
     `audio/L16;rate=8000;channels=1;`. The TDD fabric
     table (Table 8, ref [43]) says the same in one line:
     "PCM in KVS fragments", "Separate customer and agent
     tracks".

4. **One complete fragment per call, and that boundary is
   the caller's job.** GetMedia returns a byte stream, not
   fragment-sized messages, and KVS fragment boundaries
   have to be found by scanning EBML. This adapter does
   not do that scan: `feed_fragment` requires bytes that
   start with an EBML header and contain exactly one
   Segment, and trailing bytes after the Segment are an
   error. The consumer loop (item 10) owns reassembly.
   Splitting inside the adapter would mean the adapter
   silently held half a fragment with no way to say so.

5. **The payload format is pinned by the fragment's
   `MimeType` tag, because the track header lies.** The
   observed fragment dump shows `CodecID` `A_AAC` on both
   audio tracks, yet the `SimpleBlock` payloads are raw
   L16 PCM (the demo's Audacity import, and the Python
   consumer's `pcm_s16le` decode at 8000 Hz, both prove
   it). Trusting `CodecID` would send AAC frames to an
   L16 decoder. The `MimeType` tag in the same fragment
   says `audio/L16;rate=8000;channels=1;`, which is the
   format the bytes actually are. So: the first fragment
   must carry a `MimeType` tag parsing to L16 at 8000 Hz,
   mono; any other encoding, rate or channel count is
   refused, never silently resampled or re-decoded; a
   later fragment whose `MimeType` differs from the first
   is refused as a format change. A fragment with no
   `MimeType` on the first fragment is refused: the
   adapter cannot know what the bytes mean without it.
   `CodecID` is recorded for audit only and is never
   used to choose a decoder. If live traffic ever shows
   fragments without the tag, this rule must be revisited
   against a real fragment before the adapter is trusted,
   not relaxed by guessing.

6. **Track mapping comes from track names, never from
   track numbers.** Each `TrackEntry` carries a `Name`:
   `AUDIO_FROM_CUSTOMER` is the customer (the remote
   caller, the leg ShieldCall scores, `track="inbound"`)
   and `AUDIO_TO_CUSTOMER` is the agent side
   (`track="outbound"`, counted only, like Twilio).
   Track numbers are read from the same entries and are
   not assumed to be 1 and 2; the observed dump happens
   to use 1 and 2, but the mapping is the name. A track
   with no name, a track with an unknown name, or a
   fragment with no `AUDIO_FROM_CUSTOMER` track is
   refused: misattributing agent audio as customer audio
   would score the wrong leg. `AUDIO_TO_CUSTOMER` is
   optional, because the flow may stream the customer
   leg only. A later fragment whose track map differs
   from the first is refused. `SimpleBlock`s for an
   undeclared track number are refused, not dropped.

7. **Time comes from Cluster Timecodes, and gaps are
   flagged per track.** Each Cluster has an absolute
   `Timecode`; each `SimpleBlock` carries a signed
   16-bit relative timecode. The block start is
   `(cluster + relative) * TimecodeScale`, in
   milliseconds, and `t_sec` is seconds from the first
   block seen on any track. Each track keeps its own
   expected next time (previous block start plus its
   duration from its payload size). A block starting
   later than expected flags its first frame
   `discontinuous` and counts the missing samples in
   `lost_samples`; the gap is never filled. A block
   starting earlier than expected (overlap, duplicate
   or out-of-order delivery) is refused: the adapter
   will not silently re-order or double-count audio.
   `stream_sid` on each frame is the fragment number
   from the fragment's Tags, or "" when the tag is
   absent.

8. **State machine, fail closed.** A fragment must be a
   well-formed EBML document: EBML header first, one
   Segment, known sizes except on master elements (where
   an unknown size means "to the end of the enclosing
   data", the standard EBML rule); truncation, a bad
   vint, an unknown size on a leaf element, or trailing
   bytes after the Segment all raise
   `AmazonConnectError`. `SimpleBlock`s with lacing are
   refused: KVS audio blocks are unlaced, and lace
   parsing is a second framing protocol that must not
   be guessed. Empty payloads and odd byte counts (a
   truncated 16-bit sample) are refused. A `BlockGroup`
   is refused: Connect fragments use `SimpleBlock`
   only. A partial trailing frame (fewer than 80
   buffered samples) is carried across fragments and
   discarded only by `finish()`, which is idempotent
   and counts the discarded samples. Messages after
   `finish()` raise. The `ContactId` tag, when present,
   must equal the `call_id` the stream was opened with:
   it is the one in-band binding check this protocol
   offers, and a mismatch is refused rather than routed
   into the wrong call's session. When the tag is
   absent the stream still runs, and the operator's
   binding (item 9) is the only check.

9. **Serving route (`POST /v1/ingest/amazon-connect`).**
   AWS never calls this route; the operator's KVS
   consumer does. The consumer runs GetMedia with the
   stream ARN and start fragment number from the flow
   attributes, reassembles complete fragments, and POSTs
   each fragment body (raw bytes,
   `application/octet-stream`) to
   `/v1/ingest/amazon-connect?call_id=<ContactId>&token=<token>`.
   The token is a single-purpose, per-integration
   credential from `SHIELDCALL_AMAZON_CONNECT_TOKENS`
   ("id:value,..."), never the sidecar bearer token, and
   never an AWS credential: AWS credentials must never
   reach this route. With no tokens configured the route
   refuses unless the local-dev escape hatch is on, the
   same fail-closed shape as the other ingest routes.
   Auth is checked before the call gate (401, then 404).
   The call must already exist via POST /v1/calls
   (P2-6b), opened with the ContactId as the call_id.
   Fragment state (buffers, clocks, track map) is kept
   per call on the app and dropped when the call is
   closed. Only inbound frames reach the pipeline; the
   response reports inbound and outbound frame counts
   plus the usual last-event dict. A protocol error is
   a 400 with the detail, not a silent drop. The route
   adds no new decision path: CSCF stays the production
   fusion, and the sidecar stays recommend-only.

10. **What this does not do.** No KVS consumer loop is
    shipped: no GetMedia calls, no AWS SDK, no IAM
    credentials, no fragment-boundary scanning of the
    GetMedia byte stream, and no Lambda trigger. That
    loop is a follow-up that needs an AWS account, a
    Connect instance with live media streaming enabled,
    and IAM credentials the operator supplies; it must
    also decide the pull cadence and the lag budget,
    because a polling consumer cannot warn as fast as
    a push stream and the 2 s time-to-warning budget
    (TDD section 15) has to be measured on that path,
    not assumed. No AAC decoding: if a future Connect
    configuration ever delivers real AAC payloads, the
    `MimeType` rule in item 5 refuses them loudly
    instead of misdecoding them, and AAC support would
    be its own tested slice. No playback or barge-in:
    audio flows one way, into the pipeline. No live
    Connect call has been run against this adapter yet;
    that needs the operator's AWS account, instance,
    flow and consumer (Mercel-side). The adapter is
    verified against synthetic fragments built to the
    observed structure, with known ground truth, until
    a real fragment can be checked against it.

## Consequences

- The fabric now covers six of eight providers, and the
  first whose audio arrives by pull from cloud storage
  rather than by push over a socket. The POST route is
  confined to this adapter; the Twilio, Telnyx and
  Vonage WebSocket routes are untouched.
- The `discontinuous` flag is advisory until a consumer
  uses it: today the pipeline receives the frames and
  the flag is visible in stats, but no scoring path
  branches on it. For Connect the flag means a
  timecode gap between blocks on one track; ordinary
  loss inside one block is invisible, because a block
  either arrives whole or not at all.
- The per-integration token travels in the consumer's
  POST URL and therefore in its logs. Operators must
  treat it as a revocable credential: one token per
  integration, rotated on any suspicion of exposure,
  never the sidecar bearer token, never an AWS key.
- Kill-criterion note (TDD section 16, Table 31): this
  adapter has no kill criterion of its own; it is
  plumbing. Its success condition is the stage-4 gate:
  time to warning measured end to end on a real
  Connect call, which is still blocked on a live AWS
  account, instance, flow and consumer (Mercel).

## Follow-ups (not in this slice)

- The KVS GetMedia consumer loop: GetDataEndpoint plus
  GetMedia from the start fragment number, fragment
  reassembly from the byte stream, serial POSTs to
  this route, and backpressure when the sidecar is
  slow. Needs AWS credentials and a live instance.
- Verify the `MimeType`-tag and track-name rules
  against a real KVS fragment from a live Connect
  call, including whether every fragment repeats the
  Tags and Tracks headers or only the first.
- AAC decoding, only if a Connect configuration is
  ever observed delivering real AAC payloads; the
  current rule refuses those loudly.
- Two-speaker segmentation is not needed here: the
  tracks already arrive separated, unlike Vonage.
- Bidirectional audio / playback only after the VOI
  challenge gate (TDD section 11) is decided, and
  only through a mechanism Connect actually offers
  for it.

## Verification

- Tests in `tests/test_amazon_connect_ingest.py`:
  EBML parsing (known and unknown master sizes,
  truncation, trailing bytes, bad vints, skipped Void
  elements), track mapping by name with swapped track
  numbers, L16 known vectors and round trips,
  framing and buffering across fragment boundaries,
  timecode-derived `t_sec`, gap flagging and
  `lost_samples`, overlap refusal, `MimeType` pinning
  and change refusal, `ContactId` binding, laced and
  odd-sized block refusal, partial trailing frame
  discarded on finish, Audio element checks.
- Tests in `tests/test_amazon_connect_route.py`: token
  auth (required, wrong, absent-with-no-tokens),
  pre-opened-call gate, full fragment POST flow into
  a live session, malformed fragment closing with 400,
  capabilities listing.
- Full core suite green after the change (see the
  session log).
