"""Call-defense agent.

Percepts in, a decision out. The detector stack is a sensor. The agent
owns belief, budget, and the audit trace.

It does not import an LLM. Rationale strings are filled from the belief.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..fusion.engine import FusedRisk
from .belief import Belief, Perception, update
from .hypotheses import Action, Hypothesis
from .planner import Plan, select_action
from .tools import ToolResult, Toolbelt


@dataclass
class Decision:
    timestamp_sec: float
    action: Action
    plan: Plan
    belief: Dict[str, float]
    tool: ToolResult
    perception: Dict[str, Any]


class DefenseAgent:
    """One agent per call."""

    def __init__(self, max_challenges: int = 1):
        self.max_challenges = max_challenges
        self.belief = Belief()
        self.tools = Toolbelt()
        self.challenges_used = 0
        self.trace: List[Decision] = []
        self._last_risk: Optional[FusedRisk] = None

    def reset(self) -> None:
        self.belief = Belief()
        self.tools.reset()
        self.challenges_used = 0
        self.trace = []
        self._last_risk = None

    def perceive_risk(self, risk: FusedRisk, coverage_gap: float = 0.0) -> Perception:
        width = abs(risk.conformal_upper - risk.conformal_lower)
        return Perception(
            timestamp_sec=risk.timestamp_sec,
            synth=risk.acoustic_synth_prob,
            fraud=risk.linguistic_fraud_prob,
            handoff_score=risk.handoff_score,
            handoff_pvalue=risk.handoff_pvalue,
            coverage_gap=coverage_gap,
            regime=risk.regime,
            risk=risk.risk_score,
            abstain_band=risk.abstain or (width >= 0.45 and risk.tier != "HIGH_RISK"),
        )

    def step(self, perc: Perception, *, transcript: str = "") -> Decision:
        self.belief = update(self.belief, perc)
        plan = select_action(
            self.belief,
            challenges_used=self.challenges_used,
            max_challenges=self.max_challenges,
            abstain_band=perc.abstain_band,
        )
        if plan.action == Action.CHALLENGE:
            # First call issues; tests may step again with a transcript to verify.
            result = self.tools.execute(plan.action, synth=perc.synth, transcript=transcript)
            if result.data.get("nonce") is not None:
                self.challenges_used += 1
        else:
            result = self.tools.execute(plan.action, synth=perc.synth, transcript=transcript)
        decision = Decision(
            timestamp_sec=perc.timestamp_sec,
            action=plan.action,
            plan=plan,
            belief=self.belief.as_dict(),
            tool=result,
            perception={
                "synth": perc.synth,
                "fraud": perc.fraud,
                "handoff": perc.handoff_score,
                "gap": perc.coverage_gap,
                "regime": perc.regime,
            },
        )
        self.trace.append(decision)
        return decision

    def trace_dicts(self) -> List[Dict[str, Any]]:
        out = []
        for d in self.trace:
            out.append(
                {
                    "t": d.timestamp_sec,
                    "action": d.action.value,
                    "rationale": d.plan.rationale,
                    "utility": d.plan.utility,
                    "belief": d.belief,
                    "tool": {"ok": d.tool.ok, "detail": d.tool.detail},
                    "perception": d.perception,
                }
            )
        return out
