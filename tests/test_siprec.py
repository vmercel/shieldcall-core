"""SIPREC adapter (ADR-010).

Metadata vectors come from RFC 7866 §8.1 (the complete example)
and §8.2 (the partial update), read 2026-11-19: namespace
urn:ietf:params:xml:ns:recording:1, <datamode>, session /
participant / stream / participantstreamassoc elements, base64
ids, <label> binding streams to SDP m-lines. RTP vectors follow
the Asterisk pattern (RFC 3550, G.711 at 8 kHz).
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from shieldcall.ingest.g711 import _MULAW_TABLE, alaw_decode
from shieldcall.ingest.siprec import (
    SiprecError,
    SiprecRecordingSession,
    build_sdp_answer,
    parse_sdp_offer,
    parse_siprec_metadata,
)

SESSION_ID = "c2Vzc2lvbi0x"
BOB = "cGFydC1ib2I="
PAUL = "cGFydC1wYXVs"
BOB_STREAM = "c3RyZWFtLWJvYg=="
PAUL_STREAM = "c3RyZWFtLXBhdWw="
BOB_AOR = "sip:bob@biloxi.com"
PAUL_AOR = "sip:paul@biloxi.com"
SSRC = 0x0BADF00D

# The complete RFC 7866 §8.1 example, verbatim ids and labels,
# trimmed of the vendor-extension elements (which the parser must
# ignore, covered separately by test_parse_ignores_extensions).
RFC_EXAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<recording xmlns='urn:ietf:params:xml:ns:recording:1'>
  <datamode>complete</datamode>
  <group group_id="7+OTCyoxTmqmqyA/1weDAg==">
    <associate-time>2010-12-16T23:41:07Z</associate-time>
  </group>
  <session session_id="hVpd7YQgRW2nD22h7q60JQ==">
    <sipSessionID>ab30317f1a784dc48ff824d0d3715d86;
      remote=47755a9de7794ba387653f2099600ef2</sipSessionID>
    <group-ref>7+OTCyoxTmqmqyA/1weDAg==</group-ref>
  </session>
  <participant participant_id="srfBElmCRp2QB23b7Mpk0w==">
    <nameID aor="sip:bob@biloxi.com">
      <name xml:lang="it">Bob B</name>
    </nameID>
  </participant>
  <participant participant_id="zSfPoSvdSDCmU3A3TRDxAw==">
    <nameID aor="sip:Paul@biloxy.com">
      <name xml:lang="it">Paul</name>
    </nameID>
  </participant>
  <stream stream_id="UAAMm5GRQKSCMVvLyl4rFw=="
     session_id="hVpd7YQgRW2nD22h7q60JQ==">
    <label>96</label>
  </stream>
  <stream stream_id="i1Pz3to5hGk8fuXl+PbwCw=="
     session_id="hVpd7YQgRW2nD22h7q60JQ==">
    <label>97</label>
  </stream>
  <stream stream_id="8zc6e0lYTlWIINA6GR+3ag=="
     session_id="hVpd7YQgRW2nD22h7q60JQ==">
    <label>98</label>
  </stream>
  <stream stream_id="EiXGlc+4TruqqoDaNE76ag=="
     session_id="hVpd7YQgRW2nD22h7q60JQ==">
    <label>99</label>
  </stream>
  <sessionrecordingassoc session_id="hVpd7YQgRW2nD22h7q60JQ==">
    <associate-time>2010-12-16T23:41:07Z</associate-time>
  </sessionrecordingassoc>
  <participantsessionassoc
    participant_id="srfBElmCRp2QB23b7Mpk0w=="
    session_id="hVpd7YQgRW2nD22h7q60JQ==">
    <associate-time>2010-12-16T23:41:07Z</associate-time>
  </participantsessionassoc>
  <participantsessionassoc
     participant_id="zSfPoSvdSDCmU3A3TRDxAw=="
     session_id="hVpd7YQgRW2nD22h7q60JQ==">
    <associate-time>2010-12-16T23:41:07Z</associate-time>
  </participantsessionassoc>
  <participantstreamassoc
    participant_id="srfBElmCRp2QB23b7Mpk0w==">
    <send>i1Pz3to5hGk8fuXl+PbwCw==</send>
    <send>UAAMm5GRQKSCMVvLyl4rFw==</send>
    <recv>8zc6e0lYTlWIINA6GR+3ag==</recv>
    <recv>EiXGlc+4TruqqoDaNE76ag==</recv>
  </participantstreamassoc>
  <participantstreamassoc
     participant_id="zSfPoSvdSDCmU3A3TRDxAw==">
    <send>8zc6e0lYTlWIINA6GR+3ag==</send>
    <send>EiXGlc+4TruqqoDaNE76ag==</send>
    <recv>UAAMm5GRQKSCMVvLyl4rFw==</recv>
    <recv>i1Pz3to5hGk8fuXl+PbwCw==</recv>
  </participantstreamassoc>
</recording>
"""

# The RFC 7866 §8.2 partial update: Bob leaves the session.
RFC_PARTIAL = """<?xml version="1.0" encoding="UTF-8"?>
<recording xmlns='urn:ietf:params:xml:ns:recording:1'>
  <datamode>partial</datamode>
  <participant participant_id="srfBElmCRp2QB23b7Mpk0w==">
    <nameID aor="sip:bob@biloxi.com">
      <name xml:lang="it">Bob R</name>
    </nameID>
  </participant>
  <participantsessionassoc
     participant_id="srfBElmCRp2QB23b7Mpk0w=="
     session_id="hVpd7YQgRW2nD22h7q60JQ==">
    <disassociate-time>2010-12-16T23:41:07Z</disassociate-time>
  </participantsessionassoc>
</recording>
"""

# A compact two-participant, two-stream fixture for the session
# tests: Bob is the scored caller (label 1, inbound), Paul the
# other leg (label 2, outbound).
METADATA = f"""<?xml version="1.0" encoding="UTF-8"?>
<recording xmlns='urn:ietf:params:xml:ns:recording:1'>
  <datamode>complete</datamode>
  <session session_id="{SESSION_ID}">
    <start-time>2026-11-19T12:00:00Z</start-time>
  </session>
  <participant participant_id="{BOB}">
    <nameID aor="{BOB_AOR}"><name>Bob</name></nameID>
  </participant>
  <participant participant_id="{PAUL}">
    <nameID aor="{PAUL_AOR}"><name>Paul</name></nameID>
  </participant>
  <stream stream_id="{BOB_STREAM}" session_id="{SESSION_ID}">
    <label>1</label>
  </stream>
  <stream stream_id="{PAUL_STREAM}" session_id="{SESSION_ID}">
    <label>2</label>
  </stream>
  <participantstreamassoc participant_id="{BOB}">
    <send>{BOB_STREAM}</send>
    <recv>{PAUL_STREAM}</recv>
    <associate-time>2026-11-19T12:00:00Z</associate-time>
  </participantstreamassoc>
  <participantstreamassoc participant_id="{PAUL}">
    <send>{PAUL_STREAM}</send>
    <recv>{BOB_STREAM}</recv>
    <associate-time>2026-11-19T12:00:00Z</associate-time>
  </participantstreamassoc>
</recording>
"""

SDP_OFFER = """v=0
o=src 1 1 IN IP4 192.0.2.10
s=-
c=IN IP4 192.0.2.10
t=0 0
m=audio 50000 RTP/AVP 0 8
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=label:1
a=sendonly
m=audio 50002 RTP/AVP 8
a=rtpmap:8 PCMA/8000
a=label:2
a=sendonly
"""


def _mulaw_encode(samples: np.ndarray) -> bytes:
    flat = np.asarray(samples, dtype=np.float64).ravel()
    idx = np.argmin(
        np.abs(_MULAW_TABLE.astype(np.float64)[None, :] - flat[:, None]), axis=1
    )
    return idx.astype(np.uint8).tobytes()


def _rtp(seq: int, ts: int, payload: bytes, ssrc: int = SSRC, pt: int = 0) -> bytes:
    b0 = 0x80
    b1 = pt & 0x7F
    return struct.pack(">BBHII", b0, b1, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc) + payload


def _packet(seq: int, ts: int, n: int = 160, **kwargs) -> bytes:
    t = np.arange(n) / 8000.0
    samples = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return _rtp(seq, ts, _mulaw_encode(samples), **kwargs)


def _session(scored_aor: str = BOB_AOR) -> SiprecRecordingSession:
    session = SiprecRecordingSession(SESSION_ID, scored_aor)
    session.apply_metadata(parse_siprec_metadata(METADATA))
    offer = parse_sdp_offer(SDP_OFFER)
    session.bind_sdp(offer, {"1": 40000, "2": 40002})
    return session


# -- metadata parsing ----------------------------------------------------


def test_parse_rfc_example_complete():
    meta = parse_siprec_metadata(RFC_EXAMPLE)
    assert meta.datamode == "complete"
    assert list(meta.sessions) == ["hVpd7YQgRW2nD22h7q60JQ=="]
    assert meta.sessions["hVpd7YQgRW2nD22h7q60JQ=="]["sip_session_id"] == (
        "ab30317f1a784dc48ff824d0d3715d86; "
        "remote=47755a9de7794ba387653f2099600ef2"
    )
    assert meta.participants["srfBElmCRp2QB23b7Mpk0w=="].aor == "sip:bob@biloxi.com"
    assert meta.participants["srfBElmCRp2QB23b7Mpk0w=="].name == "Bob B"
    assert meta.participants["zSfPoSvdSDCmU3A3TRDxAw=="].aor == "sip:Paul@biloxy.com"
    labels = {s.label for s in meta.streams.values()}
    assert labels == {"96", "97", "98", "99"}
    bob = meta.send["srfBElmCRp2QB23b7Mpk0w=="]
    assert bob == {"i1Pz3to5hGk8fuXl+PbwCw==", "UAAMm5GRQKSCMVvLyl4rFw=="}
    paul = meta.send["zSfPoSvdSDCmU3A3TRDxAw=="]
    assert paul == {"8zc6e0lYTlWIINA6GR+3ag==", "EiXGlc+4TruqqoDaNE76ag=="}
    assert meta.recv["srfBElmCRp2QB23b7Mpk0w=="] == paul
    assert meta.recv["zSfPoSvdSDCmU3A3TRDxAw=="] == bob


def test_parse_rfc_partial_update():
    meta = parse_siprec_metadata(RFC_PARTIAL)
    assert meta.datamode == "partial"
    assert "srfBElmCRp2QB23b7Mpk0w==" in meta.disassociated_participants
    assert meta.participants["srfBElmCRp2QB23b7Mpk0w=="].name == "Bob R"


def test_parse_ignores_extensions():
    # Vendor extensions in other namespaces must not break parsing
    # (the RFC 7866 schema allows xs:any); the complete example
    # carries <call-center> and <mydata> extensions.
    xml = RFC_EXAMPLE.replace(
        "</group>",
        "<call-center xmlns='urn:ietf:params:xml:ns:callcenter'>"
        "<supervisor>sip:alice@atlanta.com</supervisor>"
        "</call-center></group>",
    )
    meta = parse_siprec_metadata(xml)
    assert len(meta.participants) == 2


def test_parse_rejects_wrong_namespace():
    with pytest.raises(SiprecError):
        parse_siprec_metadata(
            "<recording xmlns='urn:example:not-recording'>"
            "<datamode>complete</datamode></recording>"
        )


def test_parse_rejects_wrong_root():
    with pytest.raises(SiprecError):
        parse_siprec_metadata("<notrecording/>")
    with pytest.raises(SiprecError):
        parse_siprec_metadata("this is not xml")


def test_parse_rejects_bad_datamode():
    with pytest.raises(SiprecError):
        parse_siprec_metadata(
            "<recording><datamode>everything</datamode></recording>"
        )


def test_parse_rejects_stream_without_label():
    xml = METADATA.replace("<label>1</label>", "")
    with pytest.raises(SiprecError):
        parse_siprec_metadata(xml)


def test_parse_rejects_duplicate_label():
    xml = METADATA.replace("<label>2</label>", "<label>1</label>")
    with pytest.raises(SiprecError):
        parse_siprec_metadata(xml)


def test_parse_rejects_unknown_stream_ref():
    # Change only the reference in Bob's <send>, not the stream
    # definition: the association must fail closed.
    xml = METADATA.replace(
        f"<send>{BOB_STREAM}</send>", "<send>bm8ta25vd24tc3RyZWFt</send>"
    )
    with pytest.raises(SiprecError):
        parse_siprec_metadata(xml)


def test_parse_rejects_unknown_session_ref():
    xml = METADATA.replace(
        f'session_id="{SESSION_ID}">\n    <label>1</label>',
        'session_id="bm8tc2Vzc2lvbg==">\n    <label>1</label>',
    )
    with pytest.raises(SiprecError):
        parse_siprec_metadata(xml)


def test_parse_stream_without_session_id_binds_single_session():
    xml = METADATA.replace(
        f'<stream stream_id="{BOB_STREAM}" session_id="{SESSION_ID}">',
        f'<stream stream_id="{BOB_STREAM}">',
    )
    meta = parse_siprec_metadata(xml)
    assert meta.streams[BOB_STREAM].session_id == SESSION_ID


# -- SDP parsing ---------------------------------------------------------


def test_parse_sdp_offer():
    offer = parse_sdp_offer(SDP_OFFER)
    assert len(offer) == 2
    assert offer[0].label == "1"
    assert offer[0].formats == (0, 8)
    assert offer[0].rtpmap[0] == "PCMU/8000"
    assert offer[0].direction == "sendonly"
    assert offer[0].connection == "192.0.2.10"
    assert offer[0].port == 50000
    assert offer[1].label == "2"
    assert offer[1].rtpmap[8] == "PCMA/8000"


def test_parse_sdp_rejects_labelless_mline():
    bad = SDP_OFFER.replace("a=label:2\n", "")
    with pytest.raises(SiprecError):
        parse_sdp_offer(bad)


def test_parse_sdp_rejects_malformed_mline():
    with pytest.raises(SiprecError):
        parse_sdp_offer("m=audio notaport RTP/AVP 0\na=label:1\n")


def test_sdp_answer_matches_offer_shape():
    offer = parse_sdp_offer(SDP_OFFER)
    answer = build_sdp_answer(offer, "192.0.2.20", {"1": 40000, "2": 40002})
    lines = answer.splitlines()
    m_lines = [ln for ln in lines if ln.startswith("m=")]
    assert len(m_lines) == 2
    assert m_lines[0] == "m=audio 40000 RTP/AVP 0"
    assert m_lines[1] == "m=audio 40002 RTP/AVP 8"
    assert "a=recvonly" in lines
    assert "a=label:1" in lines and "a=label:2" in lines
    assert "c=IN IP4 192.0.2.20" in lines


def test_sdp_answer_rejects_unsupported_and_non_audio():
    offer = parse_sdp_offer(
        "v=0\r\no=x 1 1 IN IP4 192.0.2.10\r\ns=-\r\n"
        "c=IN IP4 192.0.2.10\r\nt=0 0\r\n"
        "m=audio 50000 RTP/AVP 9\r\na=rtpmap:9 G722/8000\r\n"
        "a=label:1\r\na=sendonly\r\n"
        "m=video 50004 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n"
        "a=label:3\r\na=sendonly\r\n"
    )
    answer = build_sdp_answer(offer, "192.0.2.20", {})
    m_lines = [ln for ln in answer.splitlines() if ln.startswith("m=")]
    assert m_lines == ["m=audio 0 RTP/AVP 0", "m=video 0 RTP/AVP 0"]


# -- recording session ---------------------------------------------------


def test_session_binds_tracks_from_metadata():
    session = _session()
    assert set(session.streams) == {"1", "2"}
    assert session.streams["1"].track == "inbound"
    assert session.streams["2"].track == "outbound"
    assert session.streams["1"].source == "siprec"
    assert session.call_id == SESSION_ID
    assert session.rtp_ports == {"1": 40000, "2": 40002}
    assert session.src_media_addr["1"] == ("192.0.2.10", 50000)


def test_session_routes_rtp_to_frames():
    session = _session()
    frames = session.handle_rtp("1", _packet(0, 0))
    frames += session.handle_rtp("1", _packet(1, 160))
    assert len(frames) == 4
    assert all(f.call_id == SESSION_ID for f in frames)
    assert all(f.track == "inbound" for f in frames)
    assert all(f.source == "siprec" for f in frames)
    assert all(f.samples.size == 80 for f in frames)
    assert [f.seq for f in frames] == [0, 1, 2, 3]
    out = session.handle_rtp("2", _packet(0, 0, pt=8))
    assert out and all(f.track == "outbound" for f in out)


def test_session_rejects_unknown_label():
    session = _session()
    with pytest.raises(SiprecError):
        session.handle_rtp("99", _packet(0, 0))


def test_session_refuses_mixed_stream():
    # One stream carrying both participants' send: mixed, so no
    # frames may be attributed to either leg.
    mixed = METADATA.replace(
        f"<send>{PAUL_STREAM}</send>",
        f"<send>{BOB_STREAM}</send>",
    )
    session = SiprecRecordingSession(SESSION_ID, BOB_AOR)
    session.apply_metadata(parse_siprec_metadata(mixed))
    session.bind_sdp(parse_sdp_offer(SDP_OFFER), {"1": 40000, "2": 40002})
    assert "1" in session.refused_labels
    assert "1" not in session.streams
    assert session.handle_rtp("1", _packet(0, 0)) == []
    assert session.stats()["refused_stream_datagrams"] == 1


def test_session_refuses_stream_with_no_senders():
    no_send = METADATA.replace(f"<send>{BOB_STREAM}</send>\n    ", "")
    session = SiprecRecordingSession(SESSION_ID, BOB_AOR)
    session.apply_metadata(parse_siprec_metadata(no_send))
    session.bind_sdp(parse_sdp_offer(SDP_OFFER), {"1": 40000, "2": 40002})
    assert "1" in session.refused_labels
    assert session.handle_rtp("1", _packet(0, 0)) == []


def test_session_gap_flagged_and_counted():
    session = _session()
    session.handle_rtp("1", _packet(0, 0))
    # Sequence jumps 0 -> 3: packets 1 and 2 are lost (2 x 160
    # samples); the timestamp jump agrees.
    frames = session.handle_rtp("1", _packet(3, 480))
    assert frames and frames[0].discontinuous
    assert session.stats()["1.lost_samples"] == 320
    assert session.stats()["1.gaps"] == 1


def test_session_duplicate_dropped():
    session = _session()
    session.handle_rtp("1", _packet(0, 0))
    assert session.handle_rtp("1", _packet(0, 0)) == []
    assert session.stats()["1.duplicates_dropped"] == 1


def test_session_ssrc_change_repins():
    # RFC 7865 §8 anticipates an SSRC change: the stream re-pins
    # instead of raising, and the next frame is discontinuous.
    session = _session()
    session.handle_rtp("1", _packet(0, 0, ssrc=SSRC))
    frames = session.handle_rtp("1", _packet(0, 0, ssrc=0xDEADBEEF))
    assert frames and frames[0].discontinuous
    assert session.stats()["1.ssrc_repinned"] == 1
    assert session.stats()["1.ssrc_rejected"] == 0
    # Media continues on the new SSRC.
    more = session.handle_rtp("1", _packet(1, 160, ssrc=0xDEADBEEF))
    assert more


def test_session_wrong_payload_type_raises():
    session = _session()
    with pytest.raises(SiprecError):
        session.handle_rtp("1", _packet(0, 0, pt=8))


def test_session_partial_update_disassociates_participant():
    session = _session()
    session.handle_rtp("1", _packet(0, 0))
    partial = f"""<recording xmlns='urn:ietf:params:xml:ns:recording:1'>
  <datamode>partial</datamode>
  <participantsessionassoc participant_id="{BOB}" session_id="{SESSION_ID}">
    <disassociate-time>2026-11-19T12:05:00Z</disassociate-time>
  </participantsessionassoc>
</recording>"""
    session.apply_metadata(parse_siprec_metadata(partial))
    # Bob left: his stream is stopped and refused; Paul's keeps going.
    assert "1" in session.refused_labels
    assert "1" not in session.streams
    assert session.handle_rtp("1", _packet(1, 160)) == []
    assert session.handle_rtp("2", _packet(0, 0, pt=8))


def test_session_complete_snapshot_survives_reinvite():
    session = _session()
    session.handle_rtp("1", _packet(0, 0))
    stream_before = session.streams["1"]
    session.apply_metadata(parse_siprec_metadata(METADATA))
    # Same label, same stream id, same track: the live stream is
    # kept, so media in flight is not lost on a re-INVITE.
    assert session.streams["1"] is stream_before
    frames = session.handle_rtp("1", _packet(1, 160))
    assert frames and frames[0].seq == 2


def test_session_stop_discards_and_refuses():
    session = _session()
    # 120 samples: one frame emitted, 40 buffered as a partial.
    session.handle_rtp("1", _packet(0, 0, n=120))
    session.stop()
    assert session.streams["1"].is_stopped
    assert session.stats()["1.discarded_on_stop"] == 40
    with pytest.raises(SiprecError):
        session.handle_rtp("1", _packet(1, 80))
    with pytest.raises(SiprecError):
        session.apply_metadata(parse_siprec_metadata(METADATA))


def test_session_stats_aggregate():
    session = _session()
    session.handle_rtp("1", _packet(0, 0))
    stats = session.stats()
    assert stats["streams"] == 2
    assert stats["refused_streams"] == 0
    assert stats["1.packets"] == 1
    assert stats["1.frames"] == 2
