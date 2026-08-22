"""Generative call simulator that emits *sensor* scores, not typed beliefs.

Each hypothesis class produces a trajectory of (synth, fraud, handoff, gap)
with calibrated noise around class-conditional means. The agent and a
threshold baseline are scored on missed-harvest, false-challenge, false-warn,
and mean interruption cost.

This is still a simulator. Closed-loop audio+text lives in
``eval.agent_closed_loop``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .agent import DefenseAgent
from .belief import Perception
from .hypotheses import Action, Hypothesis
from .planner import COST


CLASS_MEANS = {
    Hypothesis.BENIGN: dict(synth=0.12, fraud=0.08, handoff=0.05, gap=0.1, risk=0.12),
    Hypothesis.SOCIAL_ENGINEERING: dict(synth=0.16, fraud=0.82, handoff=0.10, gap=0.15, risk=0.78),
    Hypothesis.SYNTHETIC_FULL: dict(synth=0.84, fraud=0.18, handoff=0.12, gap=0.2, risk=0.72),
    Hypothesis.HANDOFF: dict(synth=0.58, fraud=0.70, handoff=0.82, gap=0.25, risk=0.80),
    Hypothesis.UNKNOWN_FAMILY: dict(synth=0.48, fraud=0.20, handoff=0.15, gap=0.88, risk=0.42),
}


@dataclass
class SimMetrics:
    hypothesis: str
    n: int
    missed_harvest: float
    false_challenge: float
    false_warn: float
    mean_cost: float
    challenge_rate: float
    final_actions: List[str]


def _trajectory(h: Hypothesis, n_steps: int, rng: np.random.RandomState) -> List[Perception]:
    m = CLASS_MEANS[h]
    out = []
    for i in range(n_steps):
        t = 0.5 + i * 0.8
        # handoff ramps synth in the second half
        synth = m["synth"]
        if h == Hypothesis.HANDOFF:
            synth = 0.18 + 0.55 * min(1.0, i / max(n_steps / 2, 1))
        perc = Perception(
            timestamp_sec=t,
            synth=float(np.clip(synth + rng.normal(0, 0.04), 0, 1)),
            fraud=float(np.clip(m["fraud"] + rng.normal(0, 0.05), 0, 1)),
            handoff_score=float(np.clip(m["handoff"] + rng.normal(0, 0.05), 0, 1)),
            coverage_gap=float(np.clip(m["gap"] + rng.normal(0, 0.03), 0, 1)),
            risk=float(np.clip(m["risk"] + rng.normal(0, 0.05), 0, 1)),
            regime=(
                "social_engineering"
                if h == Hypothesis.SOCIAL_ENGINEERING
                else "deepfake_probe"
                if h == Hypothesis.SYNTHETIC_FULL
                else "agreement"
            ),
        )
        out.append(perc)
    return out


def threshold_policy(perc: Perception, challenged: bool) -> Action:
    """Open-loop baseline: challenge if synth high, warn if fraud high."""
    if perc.synth >= 0.65 and not challenged:
        return Action.CHALLENGE
    if perc.fraud >= 0.55:
        return Action.WARN
    if perc.risk >= 0.75:
        return Action.ESCALATE
    return Action.MONITOR


def run_agent_on_traj(traj: List[Perception]) -> Tuple[DefenseAgent, List[Action]]:
    ag = DefenseAgent()
    acts = []
    for p in traj:
        d = ag.step(p)
        acts.append(d.action)
    return ag, acts


def run_threshold_on_traj(traj: List[Perception]) -> List[Action]:
    challenged = False
    acts = []
    for p in traj:
        a = threshold_policy(p, challenged)
        if a == Action.CHALLENGE:
            challenged = True
        acts.append(a)
    return acts


def _metrics(h: Hypothesis, action_lists: List[List[Action]]) -> SimMetrics:
    n = len(action_lists)
    miss = fa_ch = fa_w = cost = ch = 0.0
    harvest_classes = (Hypothesis.SOCIAL_ENGINEERING, Hypothesis.HANDOFF, Hypothesis.SYNTHETIC_FULL)
    finals = []
    for acts in action_lists:
        finals.append(acts[-1].value)
        ch += float(Action.CHALLENGE in acts)
        cost += float(sum(COST.get(a, 0.0) for a in acts))
        interrupted = any(a in (Action.WARN, Action.CHALLENGE, Action.ESCALATE) for a in acts)
        if h in harvest_classes and not interrupted:
            miss += 1.0
        if h == Hypothesis.BENIGN and Action.CHALLENGE in acts:
            fa_ch += 1.0
        if h == Hypothesis.BENIGN and Action.WARN in acts:
            fa_w += 1.0
        if h == Hypothesis.SOCIAL_ENGINEERING and Action.CHALLENGE in acts:
            fa_ch += 1.0  # wasted nonce
    return SimMetrics(
        hypothesis=h.value,
        n=n,
        missed_harvest=miss / n,
        false_challenge=fa_ch / n,
        false_warn=fa_w / n,
        mean_cost=cost / n,
        challenge_rate=ch / n,
        final_actions=finals[:8],
    )


def compare_policies(n_per_class: int = 30, n_steps: int = 5, seed: int = 0) -> Dict[str, Dict[str, SimMetrics]]:
    rng = np.random.RandomState(seed)
    out: Dict[str, Dict[str, SimMetrics]] = {"agent": {}, "threshold": {}}
    for h in Hypothesis:
        agent_runs = []
        thr_runs = []
        for _ in range(n_per_class):
            traj = _trajectory(h, n_steps, rng)
            _, acts_a = run_agent_on_traj(traj)
            acts_t = run_threshold_on_traj(traj)
            agent_runs.append(acts_a)
            thr_runs.append(acts_t)
        out["agent"][h.value] = _metrics(h, agent_runs)
        out["threshold"][h.value] = _metrics(h, thr_runs)
    return out
