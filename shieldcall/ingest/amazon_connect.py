"""Amazon Connect live media streaming adapter (TDD v2.0 section 4,
Media Ingest Fabric).

Translates Amazon Connect Kinesis Video Streams (KVS) fragments
into the canonical 10 ms frame envelope. Pure and
transport-agnostic: it takes one complete MKV fragment at a time
and returns frames. It opens no socket, holds no credential,
makes no AWS call, and knows nothing about HTTP. The serving
layer owns auth and the POST route; see
shieldcall.serve.http_app for the /v1/ingest/amazon-connect
route that drives this class. The KVS consumer loop that
fetches fragments from GetMedia is not shipped here (it needs
AWS credentials); see ADR-013 item 10.

Protocol (Amazon Connect docs, read 2026-11-22):
- The operator enables live media streaming on the Connect
  instance and adds Start/Stop media streaming blocks to the
  contact flow, choosing audio from the customer, to the
  customer, or both. Audio lands in Kinesis Video Streams.
- The flow exposes the customer audio stream ARN, start
  timestamp and start fragment number as contact attributes;
  a consumer calls GetMedia from the start fragment number
  and receives MKV fragments.
- Each fragment is one EBML document: an EBML header, one
  Segment with Info (TimecodeScale), Tracks, Tags and
  Clusters. Tracks are named AUDIO_FROM_CUSTOMER and
  AUDIO_TO_CUSTOMER. Tags carry AWS_KINESISVIDEO_FRAGMENT_
  NUMBER, ContactId, InstanceId and a MimeType of
  "audio/L16;rate=8000;channels=1;". Each Cluster has a
  Timecode and repeated SimpleBlock elements carrying the
  audio payloads.

What the fragment header claims is not what the bytes are:
the observed fragment dump shows CodecID "A_AAC" on both
tracks, yet the SimpleBlock payloads are raw L16 PCM (the
official consumer demo stores them as raw files read back
as Signed 16-bit PCM at 8000 Hz, and the independent Python
consumer decodes them as pcm_s16le). The payload format is
therefore pinned by the fragment's MimeType tag, never by
CodecID (see ADR-013 item 5). A fragment whose MimeType
does not say L16 at 8000 Hz mono is refused.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .frame import FRAME_SAMPLES_8K, MediaFrame

__all__ = ["AmazonConnectError", "AmazonConnectStream"]

SAMPLE_RATE = 8000
"""The only sample rate this adapter accepts (see ADR-013)."""

TRACK_INBOUND = "inbound"
"""AUDIO_FROM_CUSTOMER: the remote caller, the leg ShieldCall scores."""

TRACK_OUTBOUND = "outbound"
"""AUDIO_TO_CUSTOMER: the agent side, counted only, never scored."""

TRACK_FROM_CUSTOMER = "AUDIO_FROM_CUSTOMER"
TRACK_TO_CUSTOMER = "AUDIO_TO_CUSTOMER"

# Matroska element IDs used by KVS fragments (Matroska specification).
ID_EBML = 0x1A45DFA3
ID_SEGMENT = 0x18538067
ID_SEEK_HEAD = 0x114D9B74
ID_INFO = 0x1549A966
ID_TIMECODE_SCALE = 0x2AD7B1
ID_TRACKS = 0x1654AE6B
ID_TRACK_ENTRY = 0xAE
ID_TRACK_NUMBER = 0xD7
ID_TRACK_TYPE = 0x83
ID_TRACK_NAME = 0x536E
ID_CODEC_ID = 0x86
ID_AUDIO = 0xE1
ID_SAMPLING_FREQUENCY = 0xB5
ID_CHANNELS = 0x9F
ID_CLUSTER = 0x1F43B675
ID_CLUSTER_TIMECODE = 0xE7
ID_SIMPLE_BLOCK = 0xA3
ID_BLOCK_GROUP = 0xA0
ID_TAGS = 0x1254C367
ID_TAG = 0x7373
ID_SIMPLE_TAG = 0x67C8
ID_TAG_NAME = 0x45A3
ID_TAG_STRING = 0x4487
ID_VOID = 0xEC
ID_CRC32 = 0xBF

# Elements that contain other elements. Everything else is a leaf
# whose payload is read as bytes and decoded by its parent.
MASTER_IDS = frozenset(
    {
        ID_EBML,
        ID_SEGMENT,
        ID_SEEK_HEAD,
        ID_INFO,
        ID_TRACKS,
        ID_TRACK_ENTRY,
        ID_AUDIO,
        ID_CLUSTER,
        ID_BLOCK_GROUP,
        ID_TAGS,
        ID_TAG,
        ID_SIMPLE_TAG,
    }
)

# Element IDs that live directly under Segment. A Cluster with an
# unknown size runs to the end of its parent, but a Segment-level
# element after it (Tags, in the observed fragments) is a sibling,
# not a Cluster child: the Cluster parse stops when one appears.
SEGMENT_CHILD_IDS = frozenset(
    {
        ID_SEEK_HEAD,
        ID_INFO,
        ID_TRACKS,
        ID_TAGS,
        ID_CLUSTER,
        0x1043A770,  # Chapters
    }
)

# TimecodeScale units are nanoseconds; KVS fragments use
# 1,000,000 (1 ms), the Matroska default.
_NS_PER_MS = 1_000_000


class AmazonConnectError(ValueError):
    """An Amazon Connect fragment violation: malformed EBML, an
    unsupported or undeterminable payload format, an unmapped
    track, a timecode overlap, or a binding mismatch. The caller
    should reject the fragment; the adapter never guesses its
    way past one."""


class _Element:
    """One parsed EBML element: its ID, its payload bytes, and its
    parsed children when it is a master element."""

    __slots__ = ("element_id", "payload", "children")

    def __init__(
        self,
        element_id: int,
        payload: bytes,
        children: Optional[List["_Element"]] = None,
    ) -> None:
        self.element_id = element_id
        self.payload = payload
        self.children = children

    def child(self, element_id: int) -> Optional["_Element"]:
        for item in self.children or ():
            if item.element_id == element_id:
                return item
        return None

    def children_of(self, element_id: int) -> List["_Element"]:
        return [item for item in self.children or () if item.element_id == element_id]


def _read_vint(data: bytes, pos: int, end: int, for_id: bool) -> Tuple[int, int]:
    """Read one EBML variable-length integer starting at pos.

    Returns (value, length). For IDs the marker bit stays in the
    value; for sizes it is masked off, and an all-ones value
    means "unknown size" (returned as None). Raises
    AmazonConnectError on truncation or an over-long vint.
    """
    if pos >= end:
        raise AmazonConnectError("truncated EBML element")
    first = data[pos]
    length = 1
    mask = 0x80
    while length <= 8 and not (first & mask):
        length += 1
        mask >>= 1
    if length > 8:
        raise AmazonConnectError("EBML vint longer than 8 bytes")
    if pos + length > end:
        raise AmazonConnectError("truncated EBML vint")
    if for_id:
        value = int.from_bytes(data[pos : pos + length], "big")
        return value, length
    raw = int.from_bytes(data[pos : pos + length], "big")
    value = raw & ((1 << (7 * length)) - 1)
    if value == (1 << (7 * length)) - 1:
        return None, length  # type: ignore[return-value]
    return value, length


def _parse_elements(
    data: bytes, start: int, end: int, stop_ids: frozenset = frozenset()
) -> Tuple[List[_Element], int]:
    """Parse the EBML elements in data[start:end].

    Returns (elements, next_pos). A master element with an
    unknown size extends to `end`, except that parsing stops
    early when an element ID in `stop_ids` appears: that ID
    belongs to the enclosing level, not to this master (the
    Cluster/Tags case in KVS fragments). Unknown sizes on leaf
    elements are refused. Trailing bytes that do not form a
    complete element are refused by the caller checking
    next_pos against the expected end.
    """
    elements: List[_Element] = []
    pos = start
    while pos < end:
        element_id, id_len = _read_vint(data, pos, end, for_id=True)
        if stop_ids and element_id in stop_ids:
            break
        size, size_len = _read_vint(data, pos + id_len, end, for_id=False)
        header_end = pos + id_len + size_len
        if size is None:
            if element_id not in MASTER_IDS:
                raise AmazonConnectError(
                    f"EBML element 0x{element_id:X} has unknown size "
                    "but is not a master element"
                )
            data_end = end
        else:
            data_end = header_end + size
            if data_end > end:
                raise AmazonConnectError(
                    f"truncated EBML element 0x{element_id:X}: "
                    f"declares {size} payload bytes, "
                    f"{end - header_end} available"
                )
        if element_id in MASTER_IDS:
            child_stop = (
                SEGMENT_CHILD_IDS if element_id == ID_CLUSTER else frozenset()
            )
            children, child_end = _parse_elements(
                data, header_end, data_end, child_stop
            )
            if size is not None and child_end != data_end:
                raise AmazonConnectError(
                    f"unparsed bytes inside master element 0x{element_id:X}"
                )
            if size is None:
                # An unknown-size master ends where its children
                # stopped (a sibling at the enclosing level), not
                # at the end of the enclosing data: a Cluster
                # followed by Segment-level Tags must not swallow
                # those Tags.
                data_end = child_end
            elements.append(_Element(element_id, data[header_end:data_end], children))
        else:
            elements.append(_Element(element_id, data[header_end:data_end]))
        pos = data_end
    return elements, pos


def _as_uint(element: _Element, what: str) -> int:
    if not element.payload:
        raise AmazonConnectError(f"{what} has an empty value")
    return int.from_bytes(element.payload, "big")


def _as_float(element: _Element, what: str) -> float:
    if len(element.payload) == 4:
        return float(struct.unpack(">f", element.payload)[0])
    if len(element.payload) == 8:
        return float(struct.unpack(">d", element.payload)[0])
    raise AmazonConnectError(
        f"{what} is {len(element.payload)} bytes, not a 4- or 8-byte float"
    )


def _as_str(element: _Element, what: str) -> str:
    try:
        return element.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AmazonConnectError(f"{what} is not valid UTF-8") from exc


def _parse_mime_type(value: str) -> Tuple[str, int, int]:
    """Parse a MimeType tag value such as
    "audio/L16;rate=8000;channels=1;". Returns
    (encoding, rate, channels)."""
    parts = [part.strip() for part in value.split(";") if part.strip()]
    if not parts:
        raise AmazonConnectError("MimeType tag is empty")
    encoding = parts[0].lower()
    params: Dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, _, val = part.partition("=")
            params[key.strip().lower()] = val.strip()
    try:
        rate = int(params.get("rate", ""))
    except ValueError as exc:
        raise AmazonConnectError(f"MimeType rate is not an integer: {value!r}") from exc
    try:
        channels = int(params.get("channels", ""))
    except ValueError as exc:
        raise AmazonConnectError(
            f"MimeType channels is not an integer: {value!r}"
        ) from exc
    return encoding, rate, channels


class _TrackState:
    """Per-track decode state: the partial-frame buffer, the
    presentation clock, and the frame counter."""

    __slots__ = ("buffer", "expected_next_ms", "seq", "frames", "pending_gap", "clock_sec")

    def __init__(self) -> None:
        self.buffer = np.empty(0, dtype=np.float32)
        self.expected_next_ms: Optional[float] = None
        self.seq = 0
        self.frames = 0
        self.pending_gap = False
        # Stream-relative seconds of the next frame to emit.
        # Set from the block timecode on the first block and
        # after a gap; otherwise it advances 10 ms per frame,
        # so a frame completed from two blocks keeps the time
        # of the block its first sample came from.
        self.clock_sec: Optional[float] = None


class AmazonConnectStream:
    """One Amazon Connect call's KVS fragment stream, from the
    first fragment to the end of the call.

    - call_id: the pre-opened ShieldCall call id. The operator
      opens the call with the Connect ContactId, because the
      fragment Tags carry that ContactId and the adapter
      checks the binding when the tag is present (ADR-013
      item 8). The route owns the binding; the adapter only
      refuses a fragment whose ContactId says otherwise.
    - Fragments must be fed in order, one complete fragment
      per call, as reassembled by the operator's KVS
      consumer from the GetMedia byte stream.
    """

    def __init__(self, call_id: str):
        if not call_id:
            raise AmazonConnectError("call_id is required")
        self.call_id = call_id
        self.finished = False
        self.mime_type: Optional[str] = None
        self.track_map: Dict[str, int] = {}
        self.contact_id: Optional[str] = None
        self.instance_id: Optional[str] = None
        self.fragment_number: str = ""
        # Presentation origin: the absolute start time (ms) of
        # the first block seen on any track. t_sec counts from
        # there, because KVS timecodes are wall-clock epochs.
        self._origin_ms: Optional[float] = None
        self._states: Dict[str, _TrackState] = {}
        self._stats: Dict[str, int] = {
            "fragments": 0,
            "blocks": 0,
            "frames": 0,
            "inbound_frames": 0,
            "outbound_frames": 0,
            "lost_samples": 0,
            "discarded_on_close": 0,
        }

    # -- public ---------------------------------------------------------

    @property
    def is_finished(self) -> bool:
        return self.finished

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def feed_fragment(self, data: bytes) -> List[MediaFrame]:
        """Translate one complete KVS fragment into canonical
        frames. Returns frames for both tracks, in block order;
        the caller scores the inbound track only. Raises
        AmazonConnectError on any protocol violation."""
        if self.finished:
            raise AmazonConnectError("fragment received after the stream finished")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise AmazonConnectError("fragment must be bytes")
        fragment = bytes(data)
        if not fragment:
            raise AmazonConnectError("empty fragment")
        elements, pos = _parse_elements(fragment, 0, len(fragment))
        if pos != len(fragment):
            raise AmazonConnectError("trailing bytes after the last EBML element")
        if len(elements) != 2:
            raise AmazonConnectError(
                f"fragment must contain exactly an EBML header and one "
                f"Segment, found {len(elements)} top-level elements"
            )
        ebml, segment = elements
        if ebml.element_id != ID_EBML or segment.element_id != ID_SEGMENT:
            raise AmazonConnectError(
                "fragment must start with an EBML header followed by a Segment"
            )
        self._check_doc_type(ebml)
        timecode_scale_ns = self._timecode_scale(segment)
        self._read_tracks(segment)
        self._read_tags(segment)
        frames: List[MediaFrame] = []
        for cluster in segment.children_of(ID_CLUSTER):
            frames.extend(self._read_cluster(cluster, timecode_scale_ns))
        self._stats["fragments"] += 1
        return frames

    def finish(self) -> None:
        """End the stream when the call ends. Idempotent: a
        second call must not raise. A partial trailing frame
        per track is discarded, not padded: padding would
        invent audio the caller never sent. The count is
        kept in stats so a short final block is visible,
        not silent."""
        if self.finished:
            return
        for state in self._states.values():
            self._stats["discarded_on_close"] += int(state.buffer.size)
            state.buffer = np.empty(0, dtype=np.float32)
        self.finished = True

    # -- fragment structure ---------------------------------------------

    def _check_doc_type(self, ebml: _Element) -> None:
        doc_type = ebml.child(0x4282)  # DocType
        if doc_type is None:
            raise AmazonConnectError("EBML header has no DocType")
        if _as_str(doc_type, "DocType") != "matroska":
            raise AmazonConnectError("EBML DocType is not 'matroska'")

    def _timecode_scale(self, segment: _Element) -> int:
        info = segment.child(ID_INFO)
        if info is None:
            return _NS_PER_MS
        scale_el = info.child(ID_TIMECODE_SCALE)
        if scale_el is None:
            return _NS_PER_MS
        scale = _as_uint(scale_el, "TimecodeScale")
        if scale <= 0:
            raise AmazonConnectError(f"TimecodeScale is {scale}")
        return scale

    def _read_tracks(self, segment: _Element) -> None:
        tracks = segment.child(ID_TRACKS)
        if tracks is None:
            raise AmazonConnectError("fragment Segment has no Tracks element")
        mapping: Dict[str, int] = {}
        for entry in tracks.children_of(ID_TRACK_ENTRY):
            number_el = entry.child(ID_TRACK_NUMBER)
            if number_el is None:
                raise AmazonConnectError("TrackEntry has no TrackNumber")
            name_el = entry.child(ID_TRACK_NAME)
            if name_el is None:
                raise AmazonConnectError(
                    "TrackEntry has no Name: the track cannot be mapped "
                    "to a leg without its name"
                )
            name = _as_str(name_el, "Track Name")
            if name not in (TRACK_FROM_CUSTOMER, TRACK_TO_CUSTOMER):
                raise AmazonConnectError(f"unknown KVS track name {name!r}")
            if name in mapping:
                raise AmazonConnectError(f"duplicate KVS track name {name!r}")
            track_type = entry.child(ID_TRACK_TYPE)
            if track_type is not None and _as_uint(track_type, "TrackType") != 2:
                raise AmazonConnectError(
                    f"track {name!r} is not an audio track (TrackType != 2)"
                )
            audio = entry.child(ID_AUDIO)
            if audio is not None:
                rate_el = audio.child(ID_SAMPLING_FREQUENCY)
                if rate_el is not None:
                    rate = _as_float(rate_el, "SamplingFrequency")
                    if rate != SAMPLE_RATE:
                        raise AmazonConnectError(
                            f"track {name!r} sample rate is {rate}, "
                            f"expected {SAMPLE_RATE}"
                        )
                channels_el = audio.child(ID_CHANNELS)
                if channels_el is not None:
                    channels = _as_uint(channels_el, "Channels")
                    if channels != 1:
                        raise AmazonConnectError(
                            f"track {name!r} has {channels} channels, expected 1"
                        )
            mapping[name] = _as_uint(number_el, "TrackNumber")
        if TRACK_FROM_CUSTOMER not in mapping:
            raise AmazonConnectError(
                "fragment has no AUDIO_FROM_CUSTOMER track: "
                "the customer leg is missing"
            )
        if self.track_map:
            if mapping != self.track_map:
                raise AmazonConnectError(
                    f"track map changed mid-stream: {self.track_map} -> {mapping}"
                )
        else:
            self.track_map = mapping
            self._states = {
                name: _TrackState() for name in mapping
            }

    def _read_tags(self, segment: _Element) -> None:
        tags: Dict[str, str] = {}
        for tags_el in segment.children_of(ID_TAGS):
            for tag in tags_el.children_of(ID_TAG):
                for simple in tag.children_of(ID_SIMPLE_TAG):
                    name_el = simple.child(ID_TAG_NAME)
                    value_el = simple.child(ID_TAG_STRING)
                    if name_el is None or value_el is None:
                        continue
                    name = _as_str(name_el, "TagName")
                    value = _as_str(value_el, "TagString")
                    if name in tags and tags[name] != value:
                        raise AmazonConnectError(
                            f"tag {name!r} has conflicting values in one fragment"
                        )
                    tags[name] = value
        mime = tags.get("MimeType")
        if mime is not None:
            encoding, rate, channels = _parse_mime_type(mime)
            if encoding != "audio/l16" or rate != SAMPLE_RATE or channels != 1:
                raise AmazonConnectError(
                    f"unsupported KVS MimeType {mime!r} "
                    "(supported: audio/L16;rate=8000;channels=1)"
                )
            if self.mime_type is None:
                self.mime_type = mime
            elif self.mime_type != mime:
                raise AmazonConnectError(
                    f"MimeType changed mid-stream: {self.mime_type!r} -> {mime!r}"
                )
        elif self.mime_type is None:
            raise AmazonConnectError(
                "fragment has no MimeType tag: the payload format "
                "cannot be determined from the CodecID alone"
            )
        contact = tags.get("ContactId")
        if contact is not None:
            if contact != self.call_id:
                raise AmazonConnectError(
                    "fragment ContactId does not match the pre-opened call_id"
                )
            self.contact_id = contact
        instance = tags.get("InstanceId")
        if instance is not None:
            self.instance_id = instance
        fragment_number = tags.get("AWS_KINESISVIDEO_FRAGMENT_NUMBER")
        if fragment_number is not None:
            self.fragment_number = fragment_number

    # -- clusters and blocks --------------------------------------------

    def _read_cluster(self, cluster: _Element, scale_ns: int) -> List[MediaFrame]:
        timecode_el = cluster.child(ID_CLUSTER_TIMECODE)
        if timecode_el is None:
            raise AmazonConnectError("Cluster has no Timecode")
        cluster_ms = _as_uint(timecode_el, "Cluster Timecode") * scale_ns / _NS_PER_MS
        if cluster.child(ID_BLOCK_GROUP) is not None:
            raise AmazonConnectError(
                "BlockGroup is not expected in Connect fragments: "
                "only SimpleBlock is supported"
            )
        frames: List[MediaFrame] = []
        for block in cluster.children_of(ID_SIMPLE_BLOCK):
            frames.extend(self._read_block(block.payload, cluster_ms))
        return frames

    def _read_block(self, payload: bytes, cluster_ms: float) -> List[MediaFrame]:
        if len(payload) < 4:
            raise AmazonConnectError(
                f"SimpleBlock is {len(payload)} bytes, too short for a header"
            )
        track_number, vint_len = _read_vint(payload, 0, len(payload), for_id=False)
        if track_number is None:
            raise AmazonConnectError("SimpleBlock track number has unknown size")
        header_end = vint_len + 2 + 1
        if len(payload) < header_end:
            raise AmazonConnectError("SimpleBlock is truncated in its header")
        relative_ms = struct.unpack(">h", payload[vint_len : vint_len + 2])[0]
        flags = payload[vint_len + 2]
        if (flags >> 1) & 0x3:
            raise AmazonConnectError(
                "laced SimpleBlock is not supported: Connect audio "
                "blocks are unlaced"
            )
        audio = payload[header_end:]
        if not audio:
            raise AmazonConnectError("SimpleBlock has an empty audio payload")
        if len(audio) % 2:
            raise AmazonConnectError(
                f"SimpleBlock payload has odd byte count {len(audio)}: "
                "a truncated 16-bit sample"
            )
        name = self._track_name(track_number)
        state = self._states[name]
        start_ms = cluster_ms + relative_ms
        gap_detected = False
        if start_ms < 0:
            raise AmazonConnectError(f"SimpleBlock timecode is negative: {start_ms}")
        duration_ms = len(audio) / 2 / SAMPLE_RATE * 1000.0
        if state.expected_next_ms is not None:
            if start_ms < state.expected_next_ms - 0.5:
                raise AmazonConnectError(
                    f"SimpleBlock for {name} starts at {start_ms} ms, "
                    f"before the expected {state.expected_next_ms} ms: "
                    "overlap or out-of-order delivery"
                )
            if start_ms > state.expected_next_ms + 0.5:
                gap_ms = start_ms - state.expected_next_ms
                self._stats["lost_samples"] += round(gap_ms * SAMPLE_RATE / 1000.0)
                state.pending_gap = True
                gap_detected = True
        state.expected_next_ms = start_ms + duration_ms
        if self._origin_ms is None:
            self._origin_ms = start_ms
        block_start_sec = (start_ms - self._origin_ms) / 1000.0
        # Only a gap detected in THIS block re-bases the clock.
        # A pending gap carried over from an earlier block that
        # emitted no frame must not re-base it again: the frame
        # being completed started at that earlier block's time.
        if state.clock_sec is None or gap_detected:
            state.clock_sec = block_start_sec
        samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
        state.buffer = (
            np.concatenate([state.buffer, samples]) if state.buffer.size else samples
        )
        self._stats["blocks"] += 1
        return self._drain_frames(name, state)

    def _track_name(self, track_number: int) -> str:
        for name, number in self.track_map.items():
            if number == track_number:
                return name
        raise AmazonConnectError(
            f"SimpleBlock references undeclared track number {track_number}"
        )

    def _drain_frames(self, name: str, state: _TrackState) -> List[MediaFrame]:
        frames: List[MediaFrame] = []
        track = TRACK_INBOUND if name == TRACK_FROM_CUSTOMER else TRACK_OUTBOUND
        drained = 0
        while state.buffer.size >= FRAME_SAMPLES_8K:
            chunk = state.buffer[:FRAME_SAMPLES_8K]
            state.buffer = state.buffer[FRAME_SAMPLES_8K:]
            frames.append(
                MediaFrame(
                    call_id=self.call_id,
                    stream_sid=self.fragment_number,
                    track=track,
                    seq=state.seq,
                    t_sec=state.clock_sec,  # type: ignore[arg-type]
                    samples=chunk.copy(),
                    sample_rate=SAMPLE_RATE,
                    discontinuous=state.pending_gap and drained == 0,
                    source="amazon_connect",
                )
            )
            state.seq += 1
            state.frames += 1
            state.clock_sec = state.clock_sec + 0.01  # type: ignore[operator]
            self._stats["frames"] += 1
            if track == TRACK_INBOUND:
                self._stats["inbound_frames"] += 1
            else:
                self._stats["outbound_frames"] += 1
            drained += 1
        if frames:
            state.pending_gap = False
        return frames
