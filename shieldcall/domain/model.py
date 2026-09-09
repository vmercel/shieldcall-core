"""Bounded-context language for the telephone detector.

Call, linguistic stream, acoustic stream, fusion, and recommended action
are the aggregates the application service speaks. Science stays in
linguistic/, acoustic/, fusion/, agent/. This module is the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


ACTIONS = ("monitor", "warn", "challenge", "escalate", "abstain", "adapt")


@dataclass(frozen=True)
class DetectorAction:
    value: str

    def __post_init__(self) -> None:
        if self.value not in ACTIONS:
            object.__setattr__(self, "value", "monitor")


@dataclass
class DetectorDecision:
    """Recommend-only decision for one analyzed chunk. Never hangs up."""

    call_id: str
    action: str = "monitor"
    shed: bool = False
    t: float = 0.0
    synth: float = 0.0
    fraud: float = 0.0
    risk: float = 0.0
    regime: str = ""
    tier: str = "SAFE"
    stage: str = ""
    n_turns: int = 0
    last_text: str = ""
    explanation: str = ""
    source: str = "shieldcall-core"

    def as_event(self) -> Dict[str, Any]:
        return {
            "type": "event",
            "call_id": self.call_id,
            "action": self.action,
            "shed": self.shed,
            "t": self.t,
            "synth": self.synth,
            "fraud": self.fraud,
            "risk": self.risk,
            "regime": self.regime,
            "tier": self.tier,
            "stage": self.stage,
            "n_turns": self.n_turns,
            "last_text": self.last_text,
            "explanation": self.explanation,
            "source": self.source,
        }


@dataclass
class CallSnapshot:
    call_id: str
    shed: bool
    closed: bool
    n_turns: int
    frames: int
    last_action: str
    last_text: str
    last: Optional[DetectorDecision] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "shed": self.shed,
            "closed": self.closed,
            "n_turns": self.n_turns,
            "frames": self.frames,
            "last_action": self.last_action,
            "last_text": self.last_text,
            "last": None if self.last is None else self.last.as_event(),
        }


@dataclass(frozen=True)
class Capabilities:
    """What this worker will do. Honest: recommend-only, fail-open."""

    actuation: str = "recommend_only"
    hang_up: bool = False
    join_carrier: bool = False
    fail_open: bool = True
    channel_twin: bool = False
    streams: tuple = ("linguistic", "acoustic", "fusion", "agent")
    endpoints: tuple = (
        "GET /health",
        "GET /ready",
        "GET /v1/capabilities",
        "GET /v1/calls",
        "POST /v1/calls",
        "GET /v1/calls/{id}",
        "DELETE /v1/calls/{id}",
        "GET /v1/calls/{id}/trace",
        "GET /v1/calls/{id}/decision",
        "POST /v1/calls/{id}/transcript",
        "POST /v1/calls/{id}/audio",
        "POST /v1/calls/{id}/chunk",
        "POST /v1/calls/{id}/inject",
        "GET /v1/scripts",
        "POST /v1/score/linguistic",
        "POST /v1/score/acoustic",
        "POST /v1/score/fuse",
        "WS /v1/calls/{id}/stream",
        "GET /docs",
        "GET /openapi.json",
    )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "actuation": self.actuation,
            "hang_up": self.hang_up,
            "join_carrier": self.join_carrier,
            "fail_open": self.fail_open,
            "channel_twin": self.channel_twin,
            "streams": list(self.streams),
            "endpoints": list(self.endpoints),
        }
