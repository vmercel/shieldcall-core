"""Information-gain planner with an interruption budget.

For each candidate action we predict a tiny observation model, take the
expected posterior entropy, and score

    IG − λ_cost * cost − λ_delay * threat_mass * 1[action is passive]

Likelihoods here are also heuristic. The claim is: the *policy shape*
is active testing, not that the λ values are optimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .belief import Belief, entropy
from .hypotheses import ALL, Action, Hypothesis


# P(challenge_fail | hypothesis) — human-like voices usually pass a nonce;
# vocoded voices should fail our acoustic-gated verifier.
P_CHALLENGE_FAIL = {
    Hypothesis.BENIGN: 0.06,
    Hypothesis.SOCIAL_ENGINEERING: 0.10,
    Hypothesis.SYNTHETIC_FULL: 0.82,
    Hypothesis.HANDOFF: 0.55,
    Hypothesis.UNKNOWN_FAMILY: 0.40,
}

COST = {
    Action.MONITOR: 0.00,
    Action.ABSTAIN: 0.08,
    Action.ADAPT: 0.10,
    Action.WARN: 0.22,
    Action.CHALLENGE: 0.40,
    Action.ESCALATE: 0.55,
}

LAMBDA_COST = 0.45
LAMBDA_DELAY = 0.70
THREAT = (
    Hypothesis.SOCIAL_ENGINEERING,
    Hypothesis.SYNTHETIC_FULL,
    Hypothesis.HANDOFF,
)


@dataclass(frozen=True)
class Plan:
    action: Action
    utility: float
    info_gain: float
    cost: float
    rationale: str
    ranked: List[Tuple[Action, float]]


def _renorm(p: Dict[Hypothesis, float]) -> Dict[Hypothesis, float]:
    z = sum(p.values()) + 1e-12
    return {h: p[h] / z for h in ALL}


def _posterior_after_challenge(p: Dict[Hypothesis, float], failed: bool) -> Dict[Hypothesis, float]:
    post: Dict[Hypothesis, float] = {}
    for h in ALL:
        pf = P_CHALLENGE_FAIL[h]
        lik = pf if failed else (1.0 - pf)
        post[h] = p[h] * lik
    return _renorm(post)


def expected_challenge_ig(belief: Belief) -> float:
    h0 = belief.entropy()
    p_fail = sum(belief.p[h] * P_CHALLENGE_FAIL[h] for h in ALL)
    h_fail = entropy(_posterior_after_challenge(belief.p, True))
    h_pass = entropy(_posterior_after_challenge(belief.p, False))
    return float(h0 - (p_fail * h_fail + (1 - p_fail) * h_pass))


def select_action(
    belief: Belief,
    *,
    challenges_used: int = 0,
    max_challenges: int = 1,
    abstain_band: bool = False,
) -> Plan:
    threat = belief.mass(*THREAT)
    h0 = belief.entropy()
    ranked: List[Tuple[Action, float]] = []

    def util(action: Action, ig: float) -> float:
        delay = LAMBDA_DELAY * threat if action in (Action.MONITOR, Action.ABSTAIN) else 0.0
        # Once threat is sharp, escalate beats more probing.
        if action == Action.ESCALATE:
            ig = ig + 0.35 * threat
        if action == Action.WARN:
            ig = ig + 0.20 * threat
        if action == Action.ADAPT:
            ig = ig + 0.25 * belief.p[Hypothesis.UNKNOWN_FAMILY]
        return ig - LAMBDA_COST * COST[action] - delay

    candidates = {
        Action.MONITOR: 0.0,
        Action.ABSTAIN: 0.05 if abstain_band or h0 > 1.8 else 0.0,
        Action.WARN: 0.12 * threat,
        Action.ESCALATE: 0.08 * threat,
        Action.ADAPT: (
            0.10 * belief.p[Hypothesis.UNKNOWN_FAMILY]
            if belief.p[Hypothesis.UNKNOWN_FAMILY] >= 0.25
            else -1.0
        ),
        Action.CHALLENGE: expected_challenge_ig(belief) if challenges_used < max_challenges else -1.0,
    }
    # Hard constraints
    if challenges_used >= max_challenges:
        candidates[Action.CHALLENGE] = -10.0
    if belief.mode() == Hypothesis.BENIGN and threat < 0.30:
        candidates[Action.CHALLENGE] = -10.0
        candidates[Action.ESCALATE] -= 0.4
        candidates[Action.WARN] -= 0.25

    for a, ig in candidates.items():
        ranked.append((a, util(a, ig)))
    ranked.sort(key=lambda t: t[1], reverse=True)
    best, u = ranked[0]
    ig_best = candidates[best]
    rationale = (
        f"mode={belief.mode().value} p={belief.p[belief.mode()]:.2f} "
        f"threat={threat:.2f} H={h0:.2f} → {best.value} "
        f"(U={u:.3f} IG={ig_best:.3f} cost={COST[best]:.2f})"
    )
    return Plan(action=best, utility=u, info_gain=ig_best, cost=COST[best], rationale=rationale, ranked=ranked)
