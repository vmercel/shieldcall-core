"""Tools the agent may invoke. Side effects stay outside the belief update.

Challenge verification reuses the existing nonce protocol and the
acoustic-gated check: a TTS that reads the nonce back still fails if
synthetic_prob is high.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..adaptation.hooks import ChallengeResponseProtocol
from .hypotheses import Action


@dataclass
class ToolResult:
    action: Action
    ok: bool
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)


class Toolbelt:
    def __init__(self) -> None:
        self.challenge = ChallengeResponseProtocol(seed=7)
        self._pending: Optional[dict] = None

    def reset(self) -> None:
        self.challenge = ChallengeResponseProtocol(seed=7)
        self._pending = None

    def execute(
        self,
        action: Action,
        *,
        synth: float = 0.0,
        transcript: str = "",
        family: str = "unknown",
    ) -> ToolResult:
        if action == Action.MONITOR:
            return ToolResult(action, True, "continue listening")
        if action == Action.ABSTAIN:
            return ToolResult(action, True, "interval spans SAFE and threat; no block")
        if action == Action.WARN:
            return ToolResult(action, True, "user-facing caution without tearing down the call")
        if action == Action.ESCALATE:
            return ToolResult(action, True, "handed to a human operator")
        if action == Action.ADAPT:
            return ToolResult(
                action, True, "queue acoustic example for prototype memory", data={"family": family}
            )
        if action == Action.CHALLENGE:
            if self._pending is None:
                self._pending = self.challenge.issue()
                return ToolResult(
                    action,
                    True,
                    "issued liveness challenge",
                    data={"prompt": self._pending["prompt"], "nonce": self._pending["nonce"]},
                )
            passed = self.challenge.verify(transcript or "", acoustic_synth_prob=synth)
            self._pending = None
            return ToolResult(
                action, passed, "challenge verified" if passed else "challenge failed", data={"passed": passed}
            )
        return ToolResult(action, False, "unknown action")
