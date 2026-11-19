"""SIPREC adapter (TDD v2.0 §4, ADR-010).

Translates a SIPREC Recording Session into the canonical 10 ms
frame envelope. The Session Recording Client (SRC, e.g. a PBX or
SBC) opens a Recording Session (RS) with this sidecar as the
Session Recording Server (SRS) over SIP (RFC 7865); the metadata
that says who is speaking on which stream arrives in the INVITE
body (RFC 7866), and the media arrives as RTP on one UDP port
per stream label.

This module is pure: it takes SIP messages as text and RTP
datagrams as bytes, and returns frames and SIP responses. It
opens no socket and holds no credential. The serving layer owns
the sockets; see shieldcall.serve.siprec_sip for the listener
that drives this module.

Protocol (RFC 7865 / RFC 7866, read 2026-11-19; the metadata
model and the complete example in RFC 7866 §8.1 were checked
against draft-ietf-siprec-metadata-20):
- An INVITE is a recording session only when it carries the
  `siprec` option tag (Require or Supported) and the `+sip.srs`
  feature tag. The SRC offers SDP; the SRS answers with the
  same number of m-lines, `a=recvonly`, its own RTP ports.
  Every m-line carries `a=label` (RFC 4574), and the metadata
  refers to that label.
- Metadata namespace `urn:ietf:params:xml:ns:recording:1`,
  content type `application/rs-metadata+xml`, usually one part
  of a `multipart/mixed` body beside the SDP offer. Root
  `<recording>`; `<datamode>complete</datamode>` for a full
  snapshot, `partial` for an update carrying only what changed.
  `<session session_id>`; `<participant participant_id>` with
  `<nameID aor>`; `<stream stream_id session_id>` with
  `<label>`; `<participantstreamassoc participant_id>` with
  `<send>` / `<recv>` stream ids and associate/disassociate
  times. All ids are base64Binary and are treated here as
  opaque strings, never decoded.
- RTP (RFC 7865 §8): symmetric RTP is REQUIRED. The SSRC MAY
  change on conflict or restart, so an SSRC change re-pins
  the stream instead of raising (unlike the Asterisk adapter).
  Only PCMU (PT 0) and PCMA (PT 8) at 8 kHz are accepted in
  this slice; a dynamic payload type is a SiprecError, never
  a silent transcode.

Two deliberate limits of this slice, both fail-closed:
- The initial INVITE must carry the metadata snapshot. An
  INVITE with SDP but no metadata is answered 488: without
  metadata there is no label binding, so no stream can be
  attributed to a participant. (RFC 7865 §9 says metadata
  SHOULD be in the initial INVITE; this slice requires it.)
- SIP Digest authentication (RFC 3261 §22) is NOT
  implemented. The listener gates on the SRC allowlist
  instead; see shieldcall.serve.siprec_sip.
"""

from __future__ import annotations

import re
import secrets
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .frame import MediaFrame
from .g711 import alaw_decode, mulaw_decode
from .rtp import RtpStream

RECORDING_NS = "urn:ietf:params:xml:ns:recording:1"
"""RFC 7866 metadata namespace."""

METADATA_TYPES = ("application/rs-metadata+xml", "application/rs-metadata")
"""Content types accepted as recording metadata."""

SDP_TYPE = "application/sdp"

# Static RTP payload types (RFC 3551) accepted in this slice.
CODECS = {"PCMU": (0, mulaw_decode), "PCMA": (8, alaw_decode)}


class SiprecError(ValueError):
    """A SIPREC protocol violation: bad metadata, bad SDP, an
    unknown label, an unsupported codec, or a request this slice
    refuses. The caller should close the session; the adapter
    never guesses its way past one."""


# -- metadata ----------------------------------------------------------


@dataclass(frozen=True)
class SiprecParticipant:
    """One participant in the communication session."""

    participant_id: str
    aor: str = ""
    name: str = ""
    associated: bool = True
    """False once a partial update carries a disassociate-time
    for this participant (they left the call)."""


@dataclass(frozen=True)
class SiprecStream:
    """One media stream in the recording session."""

    stream_id: str
    session_id: str
    label: str


@dataclass
class SiprecMetadata:
    """One parsed recording-metadata document."""

    datamode: str = "complete"
    sessions: Dict[str, Dict[str, str]] = field(default_factory=dict)
    """session_id -> {"start_time": ..., "stop_time": ...,
    "sip_session_id": ...} (keys present only when given)."""
    participants: Dict[str, SiprecParticipant] = field(default_factory=dict)
    streams: Dict[str, SiprecStream] = field(default_factory=dict)
    """stream_id -> stream."""
    send: Dict[str, Set[str]] = field(default_factory=dict)
    """participant_id -> stream ids that participant speaks on."""
    recv: Dict[str, Set[str]] = field(default_factory=dict)
    """participant_id -> stream ids that participant hears."""
    disassociated_participants: Set[str] = field(default_factory=set)
    """participant_ids a partial update disassociated."""
    disassociated_streams: Set[str] = field(default_factory=set)
    """stream ids a partial update disassociated (via a
    participantstreamassoc carrying a disassociate-time)."""


def _local(tag: str) -> str:
    """Local name of an ElementTree tag, whatever namespace it
    carries."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _ns(tag: str) -> str:
    """Namespace of an ElementTree tag, or "" when unqualified."""
    if tag.startswith("{"):
        return tag[1:].split("}", 1)[0]
    return ""


def _children(elem: ET.Element, name: str) -> List[ET.Element]:
    return [c for c in elem if _local(c.tag) == name]


def _child_text(elem: ET.Element, name: str) -> Optional[str]:
    for c in elem:
        if _local(c.tag) == name:
            return (c.text or "").strip() or None
    return None


def parse_siprec_metadata(xml: str | bytes) -> SiprecMetadata:
    """Parse one RFC 7866 recording-metadata document.

    Fail-closed: the root must be `recording` in the recording
    namespace (or with no namespace); `datamode` must be
    `complete` or `partial`; a stream with no `<label>` is
    rejected, because a labelless stream cannot be bound to an
    SDP m-line; duplicate labels are rejected; an association
    naming an unknown stream or participant is rejected; a
    stream whose `session_id` names no declared session is
    rejected. Unknown elements in other namespaces are ignored,
    as the schema allows, so a vendor extension never breaks
    parsing."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise SiprecError(f"metadata is not well-formed XML: {exc}") from exc
    if _local(root.tag) != "recording":
        raise SiprecError(
            f"metadata root must be <recording>, got <{_local(root.tag)}>"
        )
    ns = _ns(root.tag)
    if ns and ns != RECORDING_NS:
        raise SiprecError(f"metadata namespace {ns!r} is not {RECORDING_NS!r}")
    meta = SiprecMetadata()
    datamode = _child_text(root, "datamode")
    if datamode is not None:
        datamode = datamode.lower()
        if datamode not in ("complete", "partial"):
            raise SiprecError(f"unknown datamode {datamode!r}")
        meta.datamode = datamode
    for sess in _children(root, "session"):
        session_id = sess.get("session_id")
        if not session_id:
            raise SiprecError("session element without session_id")
        info: Dict[str, str] = {}
        for key in ("start-time", "stop-time"):
            value = _child_text(sess, key)
            if value:
                info[key] = value
        sip_id = _child_text(sess, "sipSessionID")
        if sip_id:
            info["sip_session_id"] = " ".join(sip_id.split())
        meta.sessions[session_id] = info
    for part in _children(root, "participant"):
        participant_id = part.get("participant_id")
        if not participant_id:
            raise SiprecError("participant element without participant_id")
        aor = ""
        name = ""
        for name_id in _children(part, "nameID"):
            aor = name_id.get("aor") or ""
            name = _child_text(name_id, "name") or ""
        meta.participants[participant_id] = SiprecParticipant(
            participant_id=participant_id, aor=aor, name=name
        )
    labels: Dict[str, str] = {}
    for stream in _children(root, "stream"):
        stream_id = stream.get("stream_id")
        if not stream_id:
            raise SiprecError("stream element without stream_id")
        label = _child_text(stream, "label")
        if not label:
            raise SiprecError(
                f"stream {stream_id!r} has no <label>; it cannot be "
                "bound to an SDP m-line"
            )
        if label in labels:
            raise SiprecError(
                f"duplicate label {label!r} on streams "
                f"{labels[label]!r} and {stream_id!r}"
            )
        labels[label] = stream_id
        session_id = stream.get("session_id") or ""
        if session_id and session_id not in meta.sessions:
            raise SiprecError(
                f"stream {stream_id!r} names unknown session {session_id!r}"
            )
        if not session_id:
            if len(meta.sessions) == 1:
                session_id = next(iter(meta.sessions))
            else:
                raise SiprecError(
                    f"stream {stream_id!r} has no session_id and the "
                    "document does not carry exactly one session"
                )
        meta.streams[stream_id] = SiprecStream(
            stream_id=stream_id, session_id=session_id, label=label
        )
    for assoc in _children(root, "participantsessionassoc"):
        participant_id = assoc.get("participant_id") or ""
        session_id = assoc.get("session_id") or ""
        if participant_id not in meta.participants and meta.datamode == "complete":
            raise SiprecError(
                f"participantsessionassoc names unknown participant "
                f"{participant_id!r}"
            )
        if _child_text(assoc, "disassociate-time") is not None:
            meta.disassociated_participants.add(participant_id)
    for assoc in _children(root, "participantstreamassoc"):
        participant_id = assoc.get("participant_id") or ""
        if participant_id not in meta.participants and meta.datamode == "complete":
            raise SiprecError(
                f"participantstreamassoc names unknown participant "
                f"{participant_id!r}"
            )
        sends = {
            (s.text or "").strip()
            for s in _children(assoc, "send")
            if (s.text or "").strip()
        }
        recvs = {
            (r.text or "").strip()
            for r in _children(assoc, "recv")
            if (r.text or "").strip()
        }
        for stream_id in sends | recvs:
            if stream_id not in meta.streams and meta.datamode == "complete":
                raise SiprecError(
                    f"participantstreamassoc names unknown stream {stream_id!r}"
                )
        if _child_text(assoc, "disassociate-time") is not None:
            meta.disassociated_streams |= sends | recvs
        else:
            meta.send.setdefault(participant_id, set()).update(sends)
            meta.recv.setdefault(participant_id, set()).update(recvs)
    return meta


# -- SDP ---------------------------------------------------------------


@dataclass(frozen=True)
class SdpMedia:
    """One m-line of an SDP offer (RFC 4566)."""

    media: str
    port: int
    proto: str
    formats: Tuple[int, ...]
    label: str
    rtpmap: Dict[int, str] = field(default_factory=dict)
    """payload type -> "CODEC/RATE" (e.g. "PCMU/8000")."""
    direction: str = "sendrecv"
    connection: str = ""
    """Connection address from the nearest c= line (session-level
    when the m-line has none): where the SRC sends its RTP from
    under symmetric RTP (RFC 7865 §8, RFC 4961)."""


def parse_sdp_offer(sdp: str) -> List[SdpMedia]:
    """Parse the m-lines of an SDP offer.

    Every m-line must carry an `a=label` (RFC 7865 requires it);
    a labelless m-line is a SiprecError, because the metadata
    cannot refer to it."""
    lines = [ln.strip() for ln in re.split(r"\r\n|\n", sdp) if ln.strip()]
    medias: List[SdpMedia] = []
    current: Optional[dict] = None
    session_connection = ""

    def _flush() -> None:
        nonlocal current
        if current is None:
            return
        if not current["label"]:
            raise SiprecError(
                f"SDP m={current['media']} line has no a=label; "
                "SIPREC requires a label on every media stream"
            )
        medias.append(
            SdpMedia(
                media=current["media"],
                port=current["port"],
                proto=current["proto"],
                formats=tuple(current["formats"]),
                label=current["label"],
                rtpmap=dict(current["rtpmap"]),
                direction=current["direction"],
                connection=current["connection"] or session_connection,
            )
        )
        current = None

    for line in lines:
        if line.startswith("c=") and current is None:
            parts = line[2:].split()
            if len(parts) == 3:
                session_connection = parts[2]
        elif line.startswith("m="):
            _flush()
            parts = line[2:].split()
            if len(parts) < 4:
                raise SiprecError(f"malformed SDP m-line: {line!r}")
            try:
                port = int(parts[1])
                formats = tuple(int(f) for f in parts[3:])
            except ValueError as exc:
                raise SiprecError(f"malformed SDP m-line: {line!r}") from exc
            current = {
                "media": parts[0],
                "port": port,
                "proto": parts[2],
                "formats": formats,
                "label": "",
                "rtpmap": {},
                "direction": "sendrecv",
                "connection": "",
            }
        elif line.startswith("c=") and current is not None:
            parts = line[2:].split()
            if len(parts) == 3:
                current["connection"] = parts[2]
        elif line.startswith("a=") and current is not None:
            attr = line[2:]
            if attr.startswith("label:"):
                current["label"] = attr[len("label:"):].strip()
            elif attr.startswith("rtpmap:"):
                rest = attr[len("rtpmap:"):]
                pt, _, codec = rest.partition(" ")
                try:
                    current["rtpmap"][int(pt)] = codec.strip().upper()
                except ValueError as exc:
                    raise SiprecError(f"malformed SDP rtpmap: {line!r}") from exc
            elif attr in ("sendonly", "recvonly", "sendrecv", "inactive"):
                current["direction"] = attr
    _flush()
    return medias


def _pick_codec(media: SdpMedia) -> Optional[Tuple[int, str]]:
    """First offered payload type this slice decodes: PCMU (0)
    or PCMA (8) at 8 kHz. Returns (payload_type, codec) or None
    when the m-line offers nothing supported."""
    for pt in media.formats:
        codec = media.rtpmap.get(pt, "")
        name = codec.split("/")[0].upper() if codec else ""
        if name in CODECS and CODECS[name][0] == pt:
            rate = codec.split("/")[1] if "/" in codec else ""
            if rate and rate != "8000":
                continue
            return pt, name
        if not codec and pt in (0, 8):
            # No rtpmap line: a static payload type (RFC 3551)
            # names its codec by number alone.
            name = "PCMU" if pt == 0 else "PCMA"
            return pt, name
    return None


def build_sdp_answer(
    offer: List[SdpMedia], host: str, ports: Dict[str, int]
) -> str:
    """Build the SDP answer for an offer (RFC 3264).

    Same number of m-lines as the offer, in the same order.
    Accepted audio m-lines answer `a=recvonly` on the listener's
    port with the chosen codec; anything else (non-audio, or no
    supported codec) is answered with port 0, the RFC 3264 way
    to reject an m-line, never silently accepted."""
    out = [
        "v=0",
        f"o=siprec 0 0 IN IP4 {host}",
        "s=ShieldCall SIPREC",
        f"c=IN IP4 {host}",
        "t=0 0",
    ]
    for media in offer:
        port = ports.get(media.label, 0)
        if media.media == "audio" and port:
            picked = _pick_codec(media)
            pt = picked[0] if picked else media.formats[0]
            out.append(f"m=audio {port} RTP/AVP {pt}")
            codec = media.rtpmap.get(pt, "")
            if codec:
                out.append(f"a=rtpmap:{pt} {codec}")
            out.append(f"a=label:{media.label}")
            out.append("a=recvonly")
        else:
            out.append(f"m={media.media} 0 RTP/AVP 0")
    return "\r\n".join(out) + "\r\n"


# -- recording session ---------------------------------------------------


class SiprecRtpStream(RtpStream):
    """One SIPREC media stream: one label, one RTP session.

    The label, not the SSRC, is the stream identity, so an SSRC
    change re-pins the stream (RFC 7865 §8 anticipates it)
    instead of raising.
    """

    def __init__(
        self,
        call_id: str,
        label: str,
        track: str,
        payload_type: int,
        stream_id: str = "",
    ):
        codec = "PCMU" if payload_type == 0 else "PCMA"
        super().__init__(
            call_id,
            stream_sid=stream_id or label,
            track=track,
            payload_type=payload_type,
            decode=CODECS[codec][1],
            source="siprec",
            ssrc_change="repin",
            error_cls=SiprecError,
        )
        self.label = label
        self.stream_id = stream_id


class SiprecRecordingSession:
    """One recording session: one ShieldCall call.

    - call_id: the ShieldCall call id; the recording
      session_id from the metadata (same pattern as the Twilio
      Call SID and the Asterisk channelId).
    - scored_aor: the AOR of the participant whose voice is
      scored (the remote caller). A stream whose senders are
      exactly that participant is the `inbound` track; a stream
      whose senders are all other participants is `outbound`.
      A stream with no senders, or whose senders mix the scored
      participant with others, is refused: no frames are ever
      emitted from it, because mixed audio cannot be attributed
      to one leg.
    """

    def __init__(self, call_id: str, scored_aor: str):
        if not call_id:
            raise SiprecError("call_id is required")
        self.call_id = call_id
        self.scored_aor = scored_aor
        self.session_id: str = call_id
        self.metadata: Optional[SiprecMetadata] = None
        self.streams: Dict[str, SiprecRtpStream] = {}
        """label -> live RTP stream (only for accepted streams)."""
        self.refused_labels: Dict[str, str] = {}
        """label -> why no frames are emitted from it."""
        self.rtp_ports: Dict[str, int] = {}
        """label -> UDP port the listener bound for that stream."""
        self.src_media_addr: Dict[str, Tuple[str, int]] = {}
        """label -> (host, port) the SRC sends RTP from, from its
        SDP offer under symmetric RTP. The serving layer uses it
        as the RTP allowed_peer; empty when the offer carried no
        usable connection address."""
        self.stopped = False
        self.remote_addr: str = ""
        """Address the SRC's SIP requests arrive from; the serving
        layer uses it as the RTP allowed_peer. Set by the SIP
        handler, not by the adapter's caller."""
        self._stats: Dict[str, int] = {
            "refused_stream_datagrams": 0,
            "frames": 0,
        }

    # -- metadata + SDP -------------------------------------------------

    def apply_metadata(self, meta: SiprecMetadata) -> None:
        """Apply a metadata snapshot.

        `complete` replaces the snapshot; `partial` merges it,
        and a disassociate-time ends the named participant's or
        stream's association (their RTP stream is stopped).
        Stream bindings are rebuilt; a live RTP stream survives
        when its label keeps the same stream id and track, so a
        re-INVITE carrying a full snapshot does not lose media
        in flight."""
        if self.stopped:
            raise SiprecError("metadata received after stop")
        if meta.datamode == "complete" or self.metadata is None:
            merged = meta
        else:
            merged = self._merge_partial(meta)
        self.metadata = merged
        if merged.sessions:
            self.session_id = next(iter(merged.sessions))
        self._rebuild_streams()

    def _merge_partial(self, update: SiprecMetadata) -> SiprecMetadata:
        base = self.metadata
        assert base is not None
        sessions = dict(base.sessions)
        sessions.update(update.sessions)
        participants = dict(base.participants)
        participants.update(update.participants)
        streams = dict(base.streams)
        streams.update(update.streams)
        send = {k: set(v) for k, v in base.send.items()}
        recv = {k: set(v) for k, v in base.recv.items()}
        for pid, ids in update.send.items():
            send.setdefault(pid, set()).update(ids)
        for pid, ids in update.recv.items():
            recv.setdefault(pid, set()).update(ids)
        for pid in update.disassociated_participants:
            if pid in participants:
                old = participants[pid]
                participants[pid] = SiprecParticipant(
                    participant_id=old.participant_id,
                    aor=old.aor,
                    name=old.name,
                    associated=False,
                )
            send.pop(pid, None)
            recv.pop(pid, None)
        for stream_id in update.disassociated_streams:
            for pid in list(send):
                send[pid].discard(stream_id)
            for pid in list(recv):
                recv[pid].discard(stream_id)
        return SiprecMetadata(
            datamode="complete",
            sessions=sessions,
            participants=participants,
            streams=streams,
            send=send,
            recv=recv,
            disassociated_participants=set(base.disassociated_participants)
            | set(update.disassociated_participants),
            disassociated_streams=set(base.disassociated_streams)
            | set(update.disassociated_streams),
        )

    def _scored_participant_id(self) -> Optional[str]:
        # Match by AOR whether or not the participant is still
        # associated: association ends their streams (their send
        # entry is dropped on disassociation), but it must not
        # un-identify the other leg, which stays outbound.
        if self.metadata is None:
            return None
        for pid, part in self.metadata.participants.items():
            if part.aor == self.scored_aor:
                return pid
        return None

    def _track_for(self, stream_id: str) -> Optional[str]:
        """inbound / outbound for a stream, or None when the
        stream is refused (mixed or unattributable)."""
        if self.metadata is None:
            return None
        scored = self._scored_participant_id()
        senders = {
            pid
            for pid, ids in self.metadata.send.items()
            if stream_id in ids
        }
        if not senders:
            return None
        if scored is not None and senders == {scored}:
            return "inbound"
        if scored is not None and scored not in senders:
            return "outbound"
        return None

    def _rebuild_streams(self) -> None:
        if self.metadata is None:
            return
        by_label = {s.label: s for s in self.metadata.streams.values()}
        # Drop streams whose label vanished from the metadata.
        for label in list(self.streams):
            if label not in by_label:
                self.streams[label].stop()
                del self.streams[label]
        self.refused_labels = {}
        for label, stream in by_label.items():
            track = self._track_for(stream.stream_id)
            existing = self.streams.get(label)
            if track is None:
                # Mixed or unattributable: no frames from this
                # label. Stop a live stream that used to be
                # attributable (e.g. its sender disassociated).
                if existing is not None:
                    existing.stop()
                    del self.streams[label]
                self.refused_labels[label] = (
                    "mixed or unattributable stream: senders do not "
                    "identify exactly one leg"
                )
                continue
            if existing is not None:
                # Keep the live stream when its identity and
                # track are unchanged, so media in flight is not
                # lost on a re-INVITE with a full snapshot.
                if (
                    existing.stream_id == stream.stream_id
                    and existing.track == track
                ):
                    continue
                existing.stop()
                del self.streams[label]
            # Payload type is fixed when the SDP offer is bound;
            # a stream with no SDP binding yet waits for it.
            pending = getattr(self, "_pending_codecs", {}).get(label)
            if pending is None:
                continue
            self.streams[label] = SiprecRtpStream(
                self.call_id,
                label=label,
                track=track,
                payload_type=pending,
                stream_id=stream.stream_id,
            )

    def bind_sdp(self, offer: List[SdpMedia], ports: Dict[str, int]) -> None:
        """Bind the SDP offer: pick a codec per audio m-line and
        create the RTP streams whose metadata is already known.
        Ports are the listener's RTP ports per label, already
        allocated by the caller. Returns nothing; the answer's
        accepted labels are the keys of `ports`."""
        if self.stopped:
            raise SiprecError("SDP received after stop")
        self._pending_codecs: Dict[str, int] = {}
        self.rtp_ports = dict(ports)
        self.src_media_addr = {}
        for media in offer:
            if media.media != "audio":
                continue
            if media.label not in ports:
                continue
            host = media.connection or ""
            self.src_media_addr[media.label] = (host, media.port)
            picked = _pick_codec(media)
            if picked is None:
                continue
            self._pending_codecs[media.label] = picked[0]
        self._rebuild_streams()

    # -- RTP --------------------------------------------------------------

    def handle_rtp(self, label: str, datagram: bytes) -> List[MediaFrame]:
        """Translate one RTP datagram on the named stream into
        canonical frames. Unknown label raises SiprecError;
        datagrams on a refused (mixed) stream are dropped and
        counted, not raised: the refusal is a configured policy,
        not a protocol violation."""
        if self.stopped:
            raise SiprecError("RTP received after stop")
        stream = self.streams.get(label)
        if stream is None:
            if label in self.refused_labels:
                self._stats["refused_stream_datagrams"] += 1
                return []
            raise SiprecError(
                f"RTP on unknown SIPREC label {label!r}; no stream "
                "was negotiated for it"
            )
        frames = stream.handle_datagram(datagram)
        self._stats["frames"] += len(frames)
        return frames

    def stop(self) -> None:
        """End the session. Every live stream is stopped; a
        partial trailing frame is discarded, not padded."""
        if self.stopped:
            return
        for stream in self.streams.values():
            if not stream.is_stopped:
                stream.stop()
        self.stopped = True

    def stats(self) -> Dict[str, int]:
        out = dict(self._stats)
        for label, stream in self.streams.items():
            for key, value in stream.stats().items():
                out[f"{label}.{key}"] = value
        out["streams"] = len(self.streams)
        out["refused_streams"] = len(self.refused_labels)
        return out


# -- minimal SIP ---------------------------------------------------------


@dataclass
class _Dialog:
    """One SIP dialog (one recording session)."""

    call_id: str  # SIP Call-ID header
    to_tag: str
    session: SiprecRecordingSession
    sdp_answer: str
    remote_addr: str


class SiprecSipHandler:
    """A minimal SIP user-agent server for SIPREC (RFC 3261
    subset, RFC 7865). Pure: it takes raw SIP messages as text
    and returns response text, so it is testable without a
    socket. It handles INVITE, ACK, BYE, CANCEL, OPTIONS, and
    UPDATE; anything else gets 405, a malformed request 400.

    - host: the IP address advertised in SDP answers and the
      Contact header. Must be reachable by the SRC; "0.0.0.0"
      is refused.
    - sip_port: the UDP port the listener binds, echoed in the
      Contact header.
    - scored_aor: the AOR of the participant whose voice is
      scored, passed to every recording session.
    - port_allocator: returns the next RTP port per accepted
      audio m-line. The caller owns the range.
    - on_session_started / on_session_ended: called with the
      recording session when an INVITE creates it / a BYE ends
      it, so the serving layer can start and stop RTP
      listeners and open and close the runtime call.
    """

    def __init__(
        self,
        host: str,
        sip_port: int,
        scored_aor: str,
        port_allocator: Callable[[], int],
        on_session_started: Optional[Callable[[SiprecRecordingSession], None]] = None,
        on_session_ended: Optional[Callable[[SiprecRecordingSession], None]] = None,
    ):
        if not host or host == "0.0.0.0":
            raise SiprecError(
                "host must be an address the SRC can reach, not 0.0.0.0"
            )
        self.host = host
        self.sip_port = sip_port
        self.scored_aor = scored_aor
        self.port_allocator = port_allocator
        self.on_session_started = on_session_started
        self.on_session_ended = on_session_ended
        self._dialogs: Dict[str, _Dialog] = {}

    # -- entry point ----------------------------------------------------

    def handle_request(self, raw: str | bytes, remote_addr: str) -> Optional[str]:
        """Handle one SIP request. Returns the response text, or
        None when the request needs no response (ACK)."""
        try:
            text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            head, _, body = text.partition("\r\n\r\n")
            if not body and "\n\n" in text:
                head, _, body = text.partition("\n\n")
            lines = head.splitlines()
            if not lines:
                return self._response(None, 400, "Bad Request")
            request_line = lines[0].split()
            if len(request_line) != 3 or not request_line[2].startswith("SIP/"):
                return self._response(None, 400, "Bad Request")
            method = request_line[0].upper()
            headers = self._parse_headers(lines[1:])
            return self._dispatch(method, headers, body, remote_addr)
        except SiprecError as exc:
            return self._response(None, 400, f"Bad Request ({exc})")

    # -- dispatch ---------------------------------------------------------

    def _dispatch(
        self, method: str, headers: Dict[str, str], body: str, remote_addr: str
    ) -> Optional[str]:
        if method == "ACK":
            return None
        if method == "OPTIONS":
            return self._response(headers, 200, "OK", extra={
                "Allow": "INVITE, ACK, BYE, CANCEL, OPTIONS, UPDATE",
                "Supported": "siprec",
            })
        if method == "CANCEL":
            # The INVITE transaction is not tracked: CANCEL is
            # acknowledged and any live dialog for the Call-ID is
            # left alone (the SRC will BYE it). Documented
            # simplification, not a silent success.
            return self._response(headers, 200, "OK")
        if method == "BYE":
            return self._bye(headers)
        if method in ("INVITE", "UPDATE"):
            return self._invite_or_update(method, headers, body, remote_addr)
        return self._response(headers, 405, "Method Not Allowed", extra={
            "Allow": "INVITE, ACK, BYE, CANCEL, OPTIONS, UPDATE",
        })

    def _bye(self, headers: Dict[str, str]) -> str:
        call_id = headers.get("call-id", "")
        dialog = self._dialogs.pop(call_id, None)
        if dialog is None:
            return self._response(headers, 481, "Call/Transaction Does Not Exist")
        dialog.session.stop()
        if self.on_session_ended is not None:
            self.on_session_ended(dialog.session)
        return self._response(headers, 200, "OK", to_tag=dialog.to_tag)

    def _invite_or_update(
        self, method: str, headers: Dict[str, str], body: str, remote_addr: str
    ) -> str:
        call_id = headers.get("call-id", "")
        if not call_id:
            return self._response(headers, 400, "Bad Request")
        if method == "INVITE" and not self._has_siprec_tag(headers):
            return self._response(headers, 488, "Not Acceptable Here")
        dialog = self._dialogs.get(call_id)
        is_reinvite = dialog is not None and method == "INVITE"
        sdp_text, metadata_text = self._split_body(headers, body)
        offer: List[SdpMedia] = []
        if sdp_text:
            try:
                offer = parse_sdp_offer(sdp_text)
            except SiprecError as exc:
                return self._response(headers, 488, f"Not Acceptable Here ({exc})")
        meta: Optional[SiprecMetadata] = None
        if metadata_text:
            try:
                meta = parse_siprec_metadata(metadata_text)
            except SiprecError as exc:
                return self._response(headers, 488, f"Not Acceptable Here ({exc})")
        if method == "INVITE" and dialog is None:
            # New recording session: the initial INVITE must carry
            # the metadata snapshot (ADR-010 decision 5).
            if meta is None:
                return self._response(
                    headers, 488,
                    "Not Acceptable Here (initial INVITE without metadata)",
                )
            if not meta.sessions:
                return self._response(
                    headers, 488,
                    "Not Acceptable Here (metadata without a session)",
                )
            session_id = next(iter(meta.sessions))
            session = SiprecRecordingSession(session_id, self.scored_aor)
            session.remote_addr = remote_addr
            try:
                session.apply_metadata(meta)
            except SiprecError as exc:
                return self._response(headers, 488, f"Not Acceptable Here ({exc})")
            ports, error = self._allocate(offer, meta)
            if error is not None:
                return self._response(headers, 488, f"Not Acceptable Here ({error})")
            session.bind_sdp(offer, ports)
            answer = build_sdp_answer(offer, self.host, ports)
            to_tag = secrets.token_hex(4)
            self._dialogs[call_id] = _Dialog(
                call_id=call_id,
                to_tag=to_tag,
                session=session,
                sdp_answer=answer,
                remote_addr=remote_addr,
            )
            if self.on_session_started is not None:
                self.on_session_started(session)
            return self._response(
                headers, 200, "OK", to_tag=to_tag, body=answer,
                content_type=SDP_TYPE,
                extra={"Supported": "siprec"},
            )
        # Re-INVITE or UPDATE on an existing dialog.
        assert dialog is not None
        dialog.remote_addr = remote_addr
        dialog.session.remote_addr = remote_addr
        if meta is not None:
            try:
                dialog.session.apply_metadata(meta)
            except SiprecError as exc:
                return self._response(headers, 488, f"Not Acceptable Here ({exc})")
        if offer:
            ports, error = self._allocate(offer, dialog.session.metadata)
            if error is not None:
                return self._response(headers, 488, f"Not Acceptable Here ({error})")
            dialog.session.bind_sdp(offer, ports)
            dialog.sdp_answer = build_sdp_answer(offer, self.host, ports)
        return self._response(
            headers, 200, "OK", to_tag=dialog.to_tag, body=dialog.sdp_answer,
            content_type=SDP_TYPE,
            extra={"Supported": "siprec"},
        )

    def _allocate(
        self, offer: List[SdpMedia], meta: Optional[SiprecMetadata]
    ) -> Tuple[Dict[str, int], Optional[str]]:
        """Allocate one RTP port per accepted audio m-line, after
        checking every SDP label has a metadata stream. A metadata
        stream with no SDP m-line is fine: it simply never
        receives media (the SRC may describe more streams than it
        offers). An SDP label with no metadata stream is a 488:
        media on it could not be attributed to a participant."""
        if meta is not None:
            meta_labels = {s.label for s in meta.streams.values()}
            sdp_labels = {m.label for m in offer}
            orphaned = sdp_labels - meta_labels
            if orphaned:
                return {}, (
                    "SDP labels with no metadata stream: "
                    f"{sorted(orphaned)}"
                )
        ports: Dict[str, int] = {}
        for media in offer:
            if media.media != "audio":
                continue
            if _pick_codec(media) is None:
                continue
            try:
                ports[media.label] = self.port_allocator()
            except (SiprecError, RuntimeError, StopIteration) as exc:
                return {}, f"no RTP port available for {media.label!r} ({exc})"
        return ports, None

    # -- SIP text plumbing ------------------------------------------------

    @staticmethod
    def _parse_headers(lines: List[str]) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        current: Optional[str] = None
        for line in lines:
            if line[:1] in (" ", "\t") and current is not None:
                headers[current] += " " + line.strip()
                continue
            if ":" not in line:
                continue
            name, _, value = line.partition(":")
            current = name.strip().lower()
            headers[current] = value.strip()
        return headers

    @staticmethod
    def _has_siprec_tag(headers: Dict[str, str]) -> bool:
        for name in ("require", "supported"):
            tags = headers.get(name, "")
            if any(t.strip().lower() == "siprec" for t in tags.split(",")):
                return True
        return False

    @staticmethod
    def _split_body(
        headers: Dict[str, str], body: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Split a request body into (SDP, metadata) parts. Handles
        multipart/mixed (SDP + metadata parts) and single-part
        bodies of either type."""
        content_type = headers.get("content-type", "").lower()
        sdp_text: Optional[str] = None
        metadata_text: Optional[str] = None
        if "multipart/mixed" in content_type:
            match = re.search(r"boundary=([^;]+)", headers.get("content-type", ""), re.I)
            if match is None:
                raise SiprecError("multipart body without a boundary")
            boundary = match.group(1).strip().strip('"')
            for part in body.split("--" + boundary):
                part = part.strip("\r\n")
                if not part or part == "--":
                    continue
                part_head, _, part_body = part.partition("\r\n\r\n")
                if not part_body and "\n\n" in part:
                    part_head, _, part_body = part.partition("\n\n")
                part_headers = SiprecSipHandler._parse_headers(
                    part_head.splitlines()
                )
                part_type = part_headers.get("content-type", "").lower()
                if SDP_TYPE in part_type:
                    sdp_text = part_body.strip()
                elif any(t in part_type for t in METADATA_TYPES):
                    metadata_text = part_body.strip()
            return sdp_text, metadata_text
        if SDP_TYPE in content_type:
            return body.strip() or None, None
        if any(t in content_type for t in METADATA_TYPES):
            return None, body.strip() or None
        if body.strip():
            raise SiprecError(
                f"unsupported INVITE body content type {content_type!r}"
            )
        return None, None

    def _response(
        self,
        headers: Optional[Dict[str, str]],
        code: int,
        reason: str,
        to_tag: Optional[str] = None,
        body: Optional[str] = None,
        content_type: Optional[str] = None,
        extra: Optional[Dict[str, str]] = None,
    ) -> str:
        lines = [f"SIP/2.0 {code} {reason}"]
        if headers:
            for name in ("via", "from", "to", "call-id", "cseq"):
                value = headers.get(name)
                if value is None:
                    continue
                if name == "to" and to_tag is not None and "tag=" not in value.lower():
                    value = f"{value};tag={to_tag}"
                lines.append(f"{name.title()}: {value}")
        if extra:
            for name, value in extra.items():
                lines.append(f"{name}: {value}")
        lines.append(f"Contact: <sip:siprec@{self.host}:{self.sip_port}>")
        if body is not None:
            lines.append(f"Content-Type: {content_type or SDP_TYPE}")
            lines.append(f"Content-Length: {len(body.encode())}")
        else:
            lines.append("Content-Length: 0")
        lines.append("")
        text = "\r\n".join(lines)
        if body is not None:
            text += body
        return text
