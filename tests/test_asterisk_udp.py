"""Asterisk External Media UDP listener (ADR-009).

Covers shieldcall.serve.asterisk_udp: the listener binds a per-call
port, decodes datagrams into frames, hands them to the sink, drops
datagrams from unlisted peers, and stops fail-closed on a protocol
error. `attach_asterisk_listener` is covered against a live
SidecarRuntime session.

Most tests drive `datagram_received` directly: this sandbox blocks
UDP sends (PermissionError on sendto), so the socket path is covered
by a single end-to-end test that skips itself where UDP loopback is
unavailable. The direct-drive tests still exercise the real listener
object, including its start/stop lifecycle and the fail-closed
close on a protocol error.
"""

from __future__ import annotations

import asyncio
import socket
import struct

import numpy as np
import pytest

from shieldcall.ingest.asterisk import AsteriskError
from shieldcall.ingest.g711 import _MULAW_TABLE
from shieldcall.pipeline import PipelineConfig
from shieldcall.runtime.runtime import SidecarRuntime
from shieldcall.serve.asterisk_udp import (
    AsteriskUdpListener,
    attach_asterisk_listener,
)

CALL_ID = "ext-13094"
SSRC = 0x12345678


def _mulaw_encode(samples: np.ndarray) -> bytes:
    flat = np.asarray(samples, dtype=np.float64).ravel()
    idx = np.argmin(
        np.abs(_MULAW_TABLE.astype(np.float64)[None, :] - flat[:, None]), axis=1
    )
    return idx.astype(np.uint8).tobytes()


def _rtp(seq: int, ts: int, payload: bytes, ssrc: int = SSRC, pt: int = 0,
         version: int = 2) -> bytes:
    b0 = (version << 6)
    b1 = pt & 0x7F
    return struct.pack(">BBHII", b0, b1, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc) + payload


def _packet(seq: int, ts: int, n: int = 160) -> bytes:
    t = np.arange(n) / 8000.0
    samples = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return _rtp(seq, ts, _mulaw_encode(samples))


def _udp_loopback_available() -> bool:
    """True when this environment lets a UDP socket send to itself.
    The agent sandbox blocks UDP sends; the listener logic is then
    covered by the direct-drive tests and this probe skips the one
    real-socket test instead of failing on the environment."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        with sock:
            sock.bind(("127.0.0.1", 0))
            sock.sendto(b"probe", ("127.0.0.1", sock.getsockname()[1]))
        return True
    except OSError:
        return False


# -- direct-drive tests (no UDP sends) ---------------------------------


def test_listener_delivers_frames():
    received = []

    async def run():
        listener = AsteriskUdpListener(
            CALL_ID, host="127.0.0.1", port=0,
            on_frames=lambda call_id, frames: received.extend(frames),
        )
        await listener.start()
        try:
            listener.datagram_received(_packet(0, 0), ("127.0.0.1", 50000))
            listener.datagram_received(_packet(1, 160), ("127.0.0.1", 50000))
        finally:
            listener.stop()
        return listener

    listener = asyncio.run(run())
    assert len(received) == 4
    assert all(f.call_id == CALL_ID for f in received)
    assert all(f.source == "asterisk" for f in received)
    assert all(f.samples.size == 80 for f in received)
    assert [f.seq for f in received] == [0, 1, 2, 3]
    assert listener.stats()["packets"] == 2


def test_allowed_peer_drops_other_senders():
    received = []
    good_peer = ("127.0.0.1", 50000)
    bad_peer = ("127.0.0.1", 50001)

    async def run():
        listener = AsteriskUdpListener(
            CALL_ID, host="127.0.0.1", port=0,
            on_frames=lambda call_id, frames: received.extend(frames),
            allowed_peer=good_peer,
        )
        await listener.start()
        try:
            listener.datagram_received(_packet(0, 0), good_peer)
            listener.datagram_received(_packet(1, 160), bad_peer)
        finally:
            listener.stop()
        return listener

    listener = asyncio.run(run())
    assert len(received) == 2
    assert listener.stats()["dropped_wrong_peer"] == 1
    assert listener.stats()["packets"] == 1


def test_protocol_error_stops_listener():
    received = []

    async def run():
        listener = AsteriskUdpListener(
            CALL_ID, host="127.0.0.1", port=0,
            on_frames=lambda call_id, frames: received.extend(frames),
        )
        await listener.start()
        try:
            listener.datagram_received(_packet(0, 0), ("127.0.0.1", 50000))
            # RTP version 0: not a valid packet.
            listener.datagram_received(
                _rtp(1, 160, b"\xff" * 160, version=0), ("127.0.0.1", 50000)
            )
            # A good packet after the error must not be processed.
            listener.datagram_received(_packet(2, 320), ("127.0.0.1", 50000))
        finally:
            listener.stop()
        return listener

    listener = asyncio.run(run())
    assert isinstance(listener.error, AsteriskError)
    assert not listener.is_running
    assert len(received) == 2
    assert listener.stats()["packets"] == 1


def test_attach_refuses_unknown_call():
    rt = SidecarRuntime(max_calls=4, pipeline_config=PipelineConfig(channel=None))

    async def run():
        with pytest.raises(ValueError):
            await attach_asterisk_listener(rt, "no-such-call", host="127.0.0.1")

    asyncio.run(run())


def test_attach_pushes_into_session():
    rt = SidecarRuntime(max_calls=4, pipeline_config=PipelineConfig(channel=None))
    sess = rt.open_call(CALL_ID)
    pushed = []
    original_push = sess.push_audio

    def recording_push(samples, sample_rate):
        pushed.append((samples.copy(), sample_rate))
        return original_push(samples, sample_rate)

    sess.push_audio = recording_push  # type: ignore[method-assign]

    async def run():
        listener = await attach_asterisk_listener(
            rt, CALL_ID, host="127.0.0.1", port=0
        )
        try:
            listener.datagram_received(_packet(0, 0), ("127.0.0.1", 50000))
            listener.datagram_received(_packet(1, 160), ("127.0.0.1", 50000))
        finally:
            listener.stop()
        return listener

    listener = asyncio.run(run())
    assert listener.stats()["frames"] == 4
    # Two packets of 20 ms each, concatenated per packet by the sink.
    assert len(pushed) == 2
    assert all(sr == 8000 for _, sr in pushed)
    assert all(samples.size == 160 for samples, _ in pushed)
    assert rt.get_call(CALL_ID) is sess


# -- real-socket end-to-end test ---------------------------------------


def test_listener_delivers_frames_over_loopback():
    if not _udp_loopback_available():
        pytest.skip("UDP loopback is blocked in this environment")
    received = []

    async def run():
        listener = AsteriskUdpListener(
            CALL_ID, host="127.0.0.1", port=0,
            on_frames=lambda call_id, frames: received.extend(frames),
        )
        await listener.start()
        try:
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            with sender:
                sender.bind(("127.0.0.1", 0))
                for datagram in (_packet(0, 0), _packet(1, 160)):
                    sender.sendto(datagram, ("127.0.0.1", listener.bound_port))
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.05)
        finally:
            listener.stop()
        return listener

    listener = asyncio.run(run())
    assert len(received) == 4
    assert [f.seq for f in received] == [0, 1, 2, 3]
    assert listener.stats()["packets"] == 2
