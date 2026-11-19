"""SIPREC signalling + serving layer (ADR-010).

Covers shieldcall.ingest.siprec.SiprecSipHandler (the minimal SIP
server, pure) and shieldcall.serve.siprec_sip (the UDP listeners
and the runtime attach). SIP vectors are hand-built RFC 3261 /
RFC 7865 messages; the metadata and SDP bodies reuse the
fixtures from test_siprec.py.

Most listener tests drive `datagram_received` directly: this
sandbox blocks UDP sends, so the socket path is covered by a
single end-to-end test that skips itself where UDP loopback is
unavailable, mirroring test_asterisk_udp.py.
"""

from __future__ import annotations

import asyncio
import socket
import struct

import numpy as np
import pytest

from shieldcall.ingest.g711 import _MULAW_TABLE
from shieldcall.ingest.siprec import (
    SiprecError,
    SiprecRecordingSession,
    SiprecSipHandler,
)
from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve.siprec_sip import (
    SiprecRtpListener,
    SiprecSipListener,
    attach_siprec_listener,
)
from test_siprec import METADATA, SDP_OFFER, SESSION_ID, _packet

SRC_IP = "192.0.2.10"
SRC_PORT = 5060
SRS_IP = "192.0.2.20"
BOB_AOR = "sip:bob@biloxi.com"


def _sip_request(
    method: str = "INVITE",
    call_id: str = "rs-call-1",
    cseq: int = 1,
    sdp: str | None = SDP_OFFER,
    metadata: str | None = METADATA,
    require_siprec: bool = True,
    extra_headers: str = "",
) -> str:
    """Build one SIP request with a multipart or single-part body."""
    parts = []
    if sdp is not None:
        parts.append("Content-Type: application/sdp\r\n\r\n" + sdp)
    if metadata is not None:
        parts.append("Content-Type: application/rs-metadata+xml\r\n\r\n" + metadata)
    if len(parts) > 1:
        body = "".join(f"--boundary42\r\n{p}\r\n" for p in parts)
        body += "--boundary42--\r\n"
        content_type = "multipart/mixed; boundary=boundary42"
    elif parts:
        body = parts[0].split("\r\n\r\n", 1)[1]
        content_type = (
            "application/sdp" if sdp is not None else "application/rs-metadata+xml"
        )
    else:
        body = ""
        content_type = ""
    lines = [
        f"{method} sip:siprec@{SRS_IP}:5060 SIP/2.0",
        f"Via: SIP/2.0/UDP {SRC_IP}:{SRC_PORT};branch=z9hG4bK1234",
        f"From: <sip:src@{SRC_IP}>;tag=abc",
        f"To: <sip:siprec@{SRS_IP}>",
        f"Call-ID: {call_id}",
        f"CSeq: {cseq} {method}",
        f"Contact: <sip:src@{SRC_IP}:{SRC_PORT}>",
    ]
    if require_siprec:
        lines.append("Require: siprec")
    if extra_headers:
        lines.append(extra_headers)
    if content_type:
        lines.append(f"Content-Type: {content_type}")
    lines.append(f"Content-Length: {len(body.encode())}")
    return "\r\n".join(lines) + "\r\n\r\n" + body


def _handler(**kwargs) -> SiprecSipHandler:
    ports = iter([40000, 40002, 40004, 40006])
    started: list[SiprecRecordingSession] = []
    ended: list[SiprecRecordingSession] = []
    handler = SiprecSipHandler(
        host=SRS_IP,
        sip_port=5060,
        scored_aor=BOB_AOR,
        port_allocator=lambda: next(ports),
        on_session_started=started.append,
        on_session_ended=ended.append,
    )
    handler.started = started  # type: ignore[attr-defined]
    handler.ended = ended  # type: ignore[attr-defined]
    return handler


# -- handler: INVITE -----------------------------------------------------


def test_invite_returns_200_with_matching_sdp_answer():
    handler = _handler()
    response = handler.handle_request(_sip_request(), SRC_IP)
    assert response is not None
    assert response.startswith("SIP/2.0 200 OK")
    assert "Supported: siprec" in response
    assert "To: <sip:siprec@192.0.2.20>;tag=" in response
    assert "m=audio 40000 RTP/AVP 0" in response
    assert "m=audio 40002 RTP/AVP 8" in response
    assert "a=recvonly" in response
    assert "a=label:1" in response and "a=label:2" in response
    assert "c=IN IP4 192.0.2.20" in response
    assert len(handler.started) == 1
    assert handler.started[0].call_id == SESSION_ID
    assert set(handler.started[0].streams) == {"1", "2"}


def test_invite_without_siprec_tag_gets_488():
    handler = _handler()
    response = handler.handle_request(_sip_request(require_siprec=False), SRC_IP)
    assert response is not None
    assert response.startswith("SIP/2.0 488")
    assert handler.started == []


def test_invite_without_metadata_gets_488():
    handler = _handler()
    response = handler.handle_request(_sip_request(metadata=None), SRC_IP)
    assert response is not None
    assert response.startswith("SIP/2.0 488")
    assert handler.started == []


def test_invite_with_label_mismatch_gets_488():
    handler = _handler()
    bad_sdp = SDP_OFFER.replace("a=label:2", "a=label:3")
    response = handler.handle_request(_sip_request(sdp=bad_sdp), SRC_IP)
    assert response is not None
    assert response.startswith("SIP/2.0 488")


def test_invite_with_unsupported_codec_rejects_mline():
    handler = _handler()
    g722 = (
        "v=0\r\no=src 1 1 IN IP4 192.0.2.10\r\ns=-\r\n"
        "c=IN IP4 192.0.2.10\r\nt=0 0\r\n"
        "m=audio 50000 RTP/AVP 9\r\na=rtpmap:9 G722/8000\r\n"
        "a=label:1\r\na=sendonly\r\n"
    )
    response = handler.handle_request(_sip_request(sdp=g722), SRC_IP)
    assert response is not None
    assert response.startswith("SIP/2.0 200 OK")
    assert "m=audio 0 RTP/AVP 0" in response
    assert handler.started[0].streams == {}


def test_invite_with_bad_metadata_gets_488():
    handler = _handler()
    response = handler.handle_request(
        _sip_request(metadata="<recording><datamode>nope</datamode></recording>"),
        SRC_IP,
    )
    assert response is not None
    assert response.startswith("SIP/2.0 488")


def test_handler_refuses_unreachable_host():
    with pytest.raises(SiprecError):
        SiprecSipHandler(
            host="0.0.0.0", sip_port=5060, scored_aor=BOB_AOR,
            port_allocator=lambda: 40000,
        )


# -- handler: other methods ----------------------------------------------


def test_options_answers_capabilities():
    handler = _handler()
    response = handler.handle_request(_sip_request(method="OPTIONS", sdp=None, metadata=None), SRC_IP)
    assert response is not None
    assert response.startswith("SIP/2.0 200 OK")
    assert "Supported: siprec" in response
    assert "Allow: INVITE, ACK, BYE, CANCEL, OPTIONS, UPDATE" in response


def test_unknown_method_gets_405():
    handler = _handler()
    response = handler.handle_request(_sip_request(method="MESSAGE", sdp=None, metadata=None), SRC_IP)
    assert response is not None
    assert response.startswith("SIP/2.0 405")


def test_malformed_request_gets_400():
    handler = _handler()
    response = handler.handle_request("NOT A SIP MESSAGE", SRC_IP)
    assert response is not None
    assert response.startswith("SIP/2.0 400")


def test_ack_needs_no_response():
    handler = _handler()
    handler.handle_request(_sip_request(), SRC_IP)
    response = handler.handle_request(_sip_request(method="ACK", sdp=None, metadata=None), SRC_IP)
    assert response is None


def test_cancel_is_acknowledged():
    handler = _handler()
    response = handler.handle_request(_sip_request(method="CANCEL", sdp=None, metadata=None), SRC_IP)
    assert response is not None
    assert response.startswith("SIP/2.0 200 OK")


def test_bye_ends_session_and_unknown_bye_gets_481():
    handler = _handler()
    handler.handle_request(_sip_request(), SRC_IP)
    session = handler.started[0]
    assert not session.stopped
    response = handler.handle_request(_sip_request(method="BYE", cseq=2, sdp=None, metadata=None), SRC_IP)
    assert response is not None
    assert response.startswith("SIP/2.0 200 OK")
    assert session.stopped
    assert handler.ended == [session]
    again = handler.handle_request(_sip_request(method="BYE", cseq=3, sdp=None, metadata=None), SRC_IP)
    assert again is not None
    assert again.startswith("SIP/2.0 481")


def test_reinvite_partial_update_disassociates():
    handler = _handler()
    handler.handle_request(_sip_request(), SRC_IP)
    session = handler.started[0]
    partial = """<recording xmlns='urn:ietf:params:xml:ns:recording:1'>
  <datamode>partial</datamode>
  <participantsessionassoc participant_id="cGFydC1ib2I=" session_id="c2Vzc2lvbi0x">
    <disassociate-time>2026-11-19T12:05:00Z</disassociate-time>
  </participantsessionassoc>
</recording>"""
    response = handler.handle_request(
        _sip_request(method="INVITE", cseq=2, sdp=None, metadata=partial), SRC_IP
    )
    assert response is not None
    assert response.startswith("SIP/2.0 200 OK")
    assert "1" in session.refused_labels
    assert "1" not in session.streams
    assert "2" in session.streams


def test_update_without_body_keeps_dialog():
    handler = _handler()
    first = handler.handle_request(_sip_request(), SRC_IP)
    assert first is not None
    response = handler.handle_request(
        _sip_request(method="UPDATE", cseq=2, sdp=None, metadata=None), SRC_IP
    )
    assert response is not None
    assert response.startswith("SIP/2.0 200 OK")
    # The re-answer is the original SDP answer.
    assert "m=audio 40000 RTP/AVP 0" in response


# -- serving layer: SIP listener -----------------------------------------


class _FakeTransport:
    """Captures sendto instead of using a real UDP socket."""

    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        self.sent.append((data, addr))

    def close(self) -> None:
        pass

    def is_closing(self) -> bool:
        return False

    def get_extra_info(self, name: str):
        if name == "sockname":
            return ("127.0.0.1", 5060)
        return None


def _single_label_metadata() -> str:
    """Metadata with one stream (label 1, Bob's send) so the
    listener tests need only one RTP port (ephemeral)."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<recording xmlns='urn:ietf:params:xml:ns:recording:1'>
  <datamode>complete</datamode>
  <session session_id="{SESSION_ID}">
    <start-time>2026-11-19T12:00:00Z</start-time>
  </session>
  <participant participant_id="cGFydC1ib2I=">
    <nameID aor="sip:bob@biloxi.com"><name>Bob</name></nameID>
  </participant>
  <stream stream_id="c3RyZWFtLWJvYg==" session_id="{SESSION_ID}">
    <label>1</label>
  </stream>
  <participantstreamassoc participant_id="cGFydC1ib2I=">
    <send>c3RyZWFtLWJvYg==</send>
    <associate-time>2026-11-19T12:00:00Z</associate-time>
  </participantstreamassoc>
</recording>
"""


def _single_label_invite(sdp: str | None = None) -> str:
    """INVITE with one audio m-line (label 1) so the listener test
    needs only one RTP port (ephemeral)."""
    if sdp is None:
        sdp = (
            "v=0\r\no=src 1 1 IN IP4 192.0.2.10\r\ns=-\r\n"
            "c=IN IP4 192.0.2.10\r\nt=0 0\r\n"
            "m=audio 50000 RTP/AVP 0\r\na=rtpmap:0 PCMU/8000\r\n"
            "a=label:1\r\na=sendonly\r\n"
        )
    return _sip_request(sdp=sdp, metadata=_single_label_metadata())


def test_sip_listener_answers_invite_and_starts_media():
    received: list = []

    async def run():
        listener = SiprecSipListener(
            scored_aor=BOB_AOR,
            advertise_host=SRS_IP,
            host="127.0.0.1",
            sip_port=0,
            rtp_port_range=(40000, 40009),
            allowed_srcs={SRC_IP},
            on_frames=lambda call_id, frames: received.extend(frames),
        )
        await listener.start()
        fake = _FakeTransport()
        listener.transport = fake  # type: ignore[assignment]
        try:
            listener.datagram_received(
                _single_label_invite().encode(), (SRC_IP, SRC_PORT)
            )
            # Let the scheduled RTP-listener start run.
            await asyncio.sleep(0.05)
            # Capture the RTP listener state before stop() clears
            # it: stop() closes the sockets and empties the map.
            rtp_listener = next(iter(listener._rtp_listeners.values()))
            bound_port = rtp_listener.bound_port
            was_running = rtp_listener.is_running
            session_count = len(listener._sessions)
            # Media from the SRC's offered address flows to the sink.
            rtp_listener.datagram_received(_packet(0, 0), (SRC_IP, 50000))
            rtp_listener.datagram_received(_packet(1, 160), (SRC_IP, 50000))
            return listener, fake, bound_port, was_running, session_count
        finally:
            listener.stop()

    listener, fake, bound_port, was_running, session_count = asyncio.run(run())
    assert fake.sent, "no SIP response was sent"
    response = fake.sent[0][0].decode()
    assert response.startswith("SIP/2.0 200 OK")
    assert "m=audio 0 RTP/AVP 0" not in response
    assert session_count == 1
    assert bound_port == 40000
    assert was_running
    assert len(received) == 4
    assert all(f.source == "siprec" for f in received)
    assert all(f.track == "inbound" for f in received)


def test_sip_listener_drops_unlisted_src():
    async def run():
        listener = SiprecSipListener(
            scored_aor=BOB_AOR,
            advertise_host=SRS_IP,
            host="127.0.0.1",
            sip_port=0,
            rtp_port_range=(40100, 40109),
            allowed_srcs={SRC_IP},
        )
        await listener.start()
        fake = _FakeTransport()
        listener.transport = fake  # type: ignore[assignment]
        try:
            listener.datagram_received(
                _single_label_invite().encode(), ("198.51.100.7", 5060)
            )
            await asyncio.sleep(0.01)
            return listener, fake
        finally:
            listener.stop()

    listener, fake = asyncio.run(run())
    assert fake.sent == []
    assert listener.stats()["dropped_wrong_src"] == 1
    assert listener._sessions == {}


def test_rtp_listener_drops_wrong_peer():
    session = SiprecRecordingSession(SESSION_ID, BOB_AOR)
    from shieldcall.ingest.siprec import parse_siprec_metadata, parse_sdp_offer

    session.apply_metadata(parse_siprec_metadata(METADATA))
    session.bind_sdp(parse_sdp_offer(SDP_OFFER), {"1": 40000, "2": 40002})
    received: list = []

    async def run():
        listener = SiprecRtpListener(
            session,
            "1",
            host="127.0.0.1",
            port=0,
            on_frames=lambda call_id, frames: received.extend(frames),
            allowed_peer=(SRC_IP, 50000),
        )
        await listener.start()
        try:
            listener.datagram_received(_packet(0, 0), (SRC_IP, 50000))
            listener.datagram_received(_packet(1, 160), ("198.51.100.7", 50000))
        finally:
            listener.stop()
        return listener

    listener = asyncio.run(run())
    assert len(received) == 2
    assert listener.stats()["dropped_wrong_peer"] == 1


def test_rtp_listener_protocol_error_stops_it():
    session = SiprecRecordingSession(SESSION_ID, BOB_AOR)
    from shieldcall.ingest.siprec import parse_siprec_metadata, parse_sdp_offer

    session.apply_metadata(parse_siprec_metadata(METADATA))
    session.bind_sdp(parse_sdp_offer(SDP_OFFER), {"1": 40000, "2": 40002})

    async def run():
        listener = SiprecRtpListener(session, "1", host="127.0.0.1", port=0)
        await listener.start()
        try:
            listener.datagram_received(_packet(0, 0), (SRC_IP, 50000))
            # RTP version 0: not a valid packet.
            bad = struct.pack(">BBHII", 0x00, 0, 1, 160, 0x1234) + b"\xff" * 160
            listener.datagram_received(bad, (SRC_IP, 50000))
            listener.datagram_received(_packet(2, 320), (SRC_IP, 50000))
        finally:
            listener.stop()
        return listener

    listener = asyncio.run(run())
    assert isinstance(listener.error, SiprecError)
    assert not listener.is_running


# -- serving layer: attach against a live runtime ------------------------


def test_attach_opens_call_on_invite_and_pushes_audio():
    rt = SidecarRuntime(max_calls=4, pipeline_config=PipelineConfig(channel=None))
    pushed: list = []

    async def run():
        listener = await attach_siprec_listener(
            rt,
            scored_aor=BOB_AOR,
            advertise_host=SRS_IP,
            host="127.0.0.1",
            sip_port=0,
            rtp_port_range=(40200, 40209),
            allowed_srcs={SRC_IP},
        )
        fake = _FakeTransport()
        listener.transport = fake  # type: ignore[assignment]
        try:
            listener.datagram_received(
                _single_label_invite().encode(), (SRC_IP, SRC_PORT)
            )
            await asyncio.sleep(0.05)
            sess = rt.get_call(SESSION_ID)
            assert sess is not None, "INVITE did not open the runtime call"
            original_push = sess.push_audio

            def recording_push(samples, sample_rate):
                pushed.append((samples.copy(), sample_rate))
                return original_push(samples, sample_rate)

            sess.push_audio = recording_push  # type: ignore[method-assign]
            rtp_listener = next(iter(listener._rtp_listeners.values()))
            rtp_listener.datagram_received(_packet(0, 0), (SRC_IP, 50000))
            rtp_listener.datagram_received(_packet(1, 160), (SRC_IP, 50000))
            # BYE closes the runtime call.
            listener.datagram_received(
                _sip_request(method="BYE", cseq=2, sdp=None, metadata=None).encode(),
                (SRC_IP, SRC_PORT),
            )
            await asyncio.sleep(0.01)
            return listener, fake
        finally:
            listener.stop()

    listener, fake = asyncio.run(run())
    assert fake.sent[0][0].decode().startswith("SIP/2.0 200 OK")
    assert len(pushed) == 2
    assert all(sr == 8000 for _, sr in pushed)
    assert all(samples.size == 160 for samples, _ in pushed)
    assert rt.get_call(SESSION_ID) is None


# -- real-socket end-to-end test ---------------------------------------


def _udp_loopback_available() -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        with sock:
            sock.bind(("127.0.0.1", 0))
            sock.sendto(b"probe", ("127.0.0.1", sock.getsockname()[1]))
        return True
    except OSError:
        return False


def test_sip_listener_answers_over_loopback():
    if not _udp_loopback_available():
        pytest.skip("UDP loopback is blocked in this environment")
    received: list = []

    async def run():
        listener = SiprecSipListener(
            scored_aor=BOB_AOR,
            advertise_host="127.0.0.1",
            host="127.0.0.1",
            sip_port=0,
            rtp_port_range=(40300, 40309),
            allowed_srcs={"127.0.0.1"},
            on_frames=lambda call_id, frames: received.extend(frames),
        )
        await listener.start()
        try:
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            with sender:
                sender.bind(("127.0.0.1", 0))
                sender.settimeout(2)
                # The SRC sends RTP from its offered address under
                # symmetric RTP, so the offer must name this
                # socket's real port, or the RTP listener drops
                # the datagrams as the wrong peer.
                src_port = sender.getsockname()[1]
                sdp = (
                    "v=0\r\no=src 1 1 IN IP4 127.0.0.1\r\ns=-\r\n"
                    "c=IN IP4 127.0.0.1\r\nt=0 0\r\n"
                    f"m=audio {src_port} RTP/AVP 0\r\na=rtpmap:0 PCMU/8000\r\n"
                    "a=label:1\r\na=sendonly\r\n"
                )
                sender.sendto(
                    _single_label_invite(sdp=sdp).encode(),
                    ("127.0.0.1", listener.bound_port),
                )
                data, _ = sender.recvfrom(65535)
                await asyncio.sleep(0.05)
                rtp_listener = next(iter(listener._rtp_listeners.values()))
                sender.sendto(
                    _packet(0, 0), ("127.0.0.1", rtp_listener.bound_port)
                )
                sender.sendto(
                    _packet(1, 160), ("127.0.0.1", rtp_listener.bound_port)
                )
                await asyncio.sleep(0.05)
            return data
        finally:
            listener.stop()

    data = asyncio.run(run())
    assert data.decode().startswith("SIP/2.0 200 OK")
    assert len(received) == 4
