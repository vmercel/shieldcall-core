"""Asterisk External Media UDP listener (TDD v2.0 §4, ADR-009).

Asterisk sends External Media audio as plain RTP over UDP, which
does not fit the FastAPI app, so the listener is a standalone
asyncio DatagramProtocol: one `AsteriskUdpListener` binds one UDP
port per call, feeds each datagram to its `AsteriskRtpStream`, and
hands the resulting frames to an `on_frames` callback.
`attach_asterisk_listener` wires that callback to a SidecarRuntime
session exactly like the Twilio websocket route (concatenate
frames, `sess.push_audio(samples, 8000)`), and refuses to start
when the call does not already exist (P2-6b: no implicit call
creation).

RTP has no authentication, and this module says so plainly. Anyone
who can reach the UDP port can inject audio, and a datagram carries
no call identifier, so the binding between a UDP flow and a
ShieldCall call is the port itself. The controls are network-level,
not protocol: the listener binds a per-call port, the operator
firewalls that port to the Asterisk host only, and the listener
accepts an optional `allowed_peer` (Asterisk's ip:port) from which
datagrams from any other source are dropped and counted. Without
`allowed_peer` or a firewall rule, the port must be treated as
untrusted. This is a property of RTP, not a gap this module can
close.

The listener is receive-only: it never sends RTP back to Asterisk.
The ARI `direction` is still `both` (the API allows no other
value); Asterisk tolerates a peer that only listens.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..ingest.asterisk import AsteriskError, AsteriskRtpStream
from ..ingest.frame import MediaFrame
from ..runtime.runtime import SidecarRuntime

log = logging.getLogger(__name__)

FrameSink = Callable[[str, List[MediaFrame]], None]
"""Called with (call_id, frames) for every batch of frames the
adapter produces. Runs in the event loop; keep it fast."""


class AsteriskUdpListener(asyncio.DatagramProtocol):
    """One UDP listener for one Asterisk External Media stream.

    - call_id: the pre-opened ShieldCall call id (the operator
      opens the call with the externalMedia channelId).
    - format: "ulaw" or "alaw" (the format the ARI channel was
      created with).
    - host/port: the UDP socket to bind. One listener per call:
      the port is what binds a datagram flow to this call.
    - on_frames: sink for decoded frames. If None, frames are
      decoded and counted but dropped (useful for a smoke test).
    - allowed_peer: if set, only datagrams from this (host, port)
      are accepted; anything else is dropped and counted. Set it
      to the Asterisk host's address whenever it is known.
    """

    def __init__(
        self,
        call_id: str,
        format: str = "ulaw",
        host: str = "0.0.0.0",
        port: int = 0,
        on_frames: Optional[FrameSink] = None,
        track: str = "inbound",
        channel_id: Optional[str] = None,
        allowed_peer: Optional[Tuple[str, int]] = None,
    ):
        self.call_id = call_id
        self.format = format
        self.host = host
        self.port = port
        self.on_frames = on_frames
        self.allowed_peer = allowed_peer
        self.stream = AsteriskRtpStream(
            call_id, format=format, track=track, channel_id=channel_id
        )
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.error: Optional[AsteriskError] = None
        self._dropped_wrong_peer = 0

    # -- lifecycle ------------------------------------------------------

    async def start(self) -> "AsteriskUdpListener":
        """Bind the UDP socket. Raises if already started."""
        if self.transport is not None:
            raise RuntimeError("AsteriskUdpListener already started")
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: self, local_addr=(self.host, self.port)
        )
        self.transport = transport
        sockname = transport.get_extra_info("sockname")
        log.info("asterisk listener bound for call %s on %s", self.call_id, sockname)
        return self

    def stop(self) -> None:
        """Close the UDP socket. Idempotent."""
        if self.transport is not None:
            self.transport.close()
            self.transport = None

    @property
    def is_running(self) -> bool:
        return self.transport is not None and not self.transport.is_closing()

    @property
    def bound_port(self) -> Optional[int]:
        if self.transport is None:
            return None
        sockname = self.transport.get_extra_info("sockname")
        return sockname[1] if sockname else None

    def stats(self) -> Dict[str, int]:
        stats = self.stream.stats()
        stats["dropped_wrong_peer"] = self._dropped_wrong_peer
        return stats

    # -- DatagramProtocol ----------------------------------------------

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        if self.error is not None:
            return
        if self.allowed_peer is not None and addr != self.allowed_peer:
            self._dropped_wrong_peer += 1
            return
        try:
            frames = self.stream.handle_datagram(data)
        except AsteriskError as exc:
            # Fail closed, like the Twilio route: a protocol
            # violation ends the stream rather than being guessed
            # past. The socket is closed so no further audio from
            # this flow is accepted.
            self.error = exc
            log.warning(
                "asterisk stream for call %s stopped on protocol error: %s",
                self.call_id,
                exc,
            )
            self.stop()
            return
        if frames and self.on_frames is not None:
            self.on_frames(self.call_id, frames)

    def error_received(self, exc: Exception) -> None:
        log.warning("asterisk listener for call %s socket error: %s", self.call_id, exc)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        self.transport = None


async def attach_asterisk_listener(
    rt: SidecarRuntime,
    call_id: str,
    format: str = "ulaw",
    host: str = "0.0.0.0",
    port: int = 0,
    track: str = "inbound",
    channel_id: Optional[str] = None,
    allowed_peer: Optional[Tuple[str, int]] = None,
) -> AsteriskUdpListener:
    """Start a UDP listener feeding one pre-opened call's session.

    Mirrors the Twilio route: frames are concatenated and pushed
    into the session's pipeline at 8 kHz, and the listener refuses
    to start when the call does not already exist. Returns the
    started listener; the caller owns stopping it.
    """
    sess = rt.get_call(call_id)
    if sess is None:
        raise ValueError(f"unknown call_id {call_id!r}: open the call first")

    def _push(_call_id: str, frames: List[MediaFrame]) -> None:
        live = rt.get_call(call_id)
        if live is None:
            return
        samples = np.concatenate([f.samples for f in frames])
        t0 = time.perf_counter()
        live.push_audio(samples, 8000)
        rt.observe_frame_ms((time.perf_counter() - t0) * 1000.0)

    listener = AsteriskUdpListener(
        call_id,
        format=format,
        host=host,
        port=port,
        on_frames=_push,
        track=track,
        channel_id=channel_id,
        allowed_peer=allowed_peer,
    )
    await listener.start()
    return listener
