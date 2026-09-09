"""Closed-loop agent evaluation on pipeline scores (audio + injected text).

Percepts come from ShieldCallPipeline sufficient statistics, not from
hand-typed regime strings. Likelihoods in the agent remain heuristic.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..agent.agent import DefenseAgent
from ..agent.belief import Perception
from ..agent.hypotheses import Action, Hypothesis
from ..agent.planner import COST
from ..agent.simulator import SimMetrics, threshold_policy
from ..linguistic.asr_bridge import ScheduledTranscriptASR
from ..pipeline import PipelineConfig, ShieldCallPipeline
from .corpora.independent_scripts import independent_scripts
from .corpora.vishing_scripts import CallScript
from .speech_data import SpeechClip


def _script_schedule(script: CallScript, audio_seconds: float) -> List[Tuple[float, str]]:
    turns = list(script.turns)
    if not turns:
        return []
    span = max(audio_seconds - 0.4, 0.8)
    step = span / max(len(turns), 1)
    return [(0.15 + i * step, text) for i, (_, text) in enumerate(turns)]


def run_pipeline_agent(
    audio: np.ndarray,
    sr: int,
    script: CallScript,
    *,
    acoustic_scorer=None,
    chunk_ms: float = 250.0,
) -> Tuple[DefenseAgent, List[Action]]:
    asr = ScheduledTranscriptASR(_script_schedule(script, len(audio) / float(sr)))
    pipe = ShieldCallPipeline(PipelineConfig(channel=None, fuse_every_n_frames=10), asr=asr)
    if acoustic_scorer is not None:
        pipe.acoustic = acoustic_scorer
    agent = DefenseAgent()
    actions: List[Action] = []
    for ev in pipe.stream(audio, sr, chunk_ms=chunk_ms):
        if ev.risk is None:
            continue
        decision = agent.step(agent.perceive_risk(ev.risk, coverage_gap=0.0))
        actions.append(decision.action)
    return agent, actions


def _metrics(h: Hypothesis, action_lists: List[List[Action]]) -> SimMetrics:
    n = max(len(action_lists), 1)
    miss = fa_ch = fa_w = cost = ch = 0.0
    harvest = (Hypothesis.SOCIAL_ENGINEERING, Hypothesis.HANDOFF, Hypothesis.SYNTHETIC_FULL)
    finals = []
    for acts in action_lists:
        if not acts:
            acts = [Action.MONITOR]
        finals.append(acts[-1].value)
        ch += float(Action.CHALLENGE in acts)
        cost += float(sum(COST.get(a, 0.0) for a in acts))
        interrupted = any(a in (Action.WARN, Action.CHALLENGE, Action.ESCALATE) for a in acts)
        if h in harvest and not interrupted:
            miss += 1.0
        if h == Hypothesis.BENIGN and Action.CHALLENGE in acts:
            fa_ch += 1.0
        if h == Hypothesis.BENIGN and Action.WARN in acts:
            fa_w += 1.0
        if h == Hypothesis.SOCIAL_ENGINEERING and Action.CHALLENGE in acts:
            fa_ch += 1.0
    return SimMetrics(
        hypothesis=h.value,
        n=len(action_lists),
        missed_harvest=miss / n,
        false_challenge=fa_ch / n,
        false_warn=fa_w / n,
        mean_cost=cost / n,
        challenge_rate=ch / n,
        final_actions=finals[:8],
    )


def _threshold_from_agent(ag: DefenseAgent) -> List[Action]:
    challenged = False
    acts: List[Action] = []
    for d in ag.trace:
        p = Perception(
            timestamp_sec=d.timestamp_sec,
            synth=float(d.perception.get("synth", 0.0)),
            fraud=float(d.perception.get("fraud", 0.0)),
            handoff_score=float(d.perception.get("handoff", 0.0)),
            coverage_gap=float(d.perception.get("gap", 0.0)),
            risk=0.4 * float(d.perception.get("synth", 0.0))
            + 0.6 * float(d.perception.get("fraud", 0.0)),
            regime=str(d.perception.get("regime", "agreement")),
        )
        a = threshold_policy(p, challenged)
        if a == Action.CHALLENGE:
            challenged = True
        acts.append(a)
    return acts or [Action.MONITOR]


def compare_closed_loop(
    bona: Sequence[SpeechClip],
    spoof: Sequence[np.ndarray],
    *,
    n_per_class: int = 5,
    acoustic_scorer=None,
) -> Dict[str, Dict[str, dict]]:
    """Map scripts+audio to hypotheses and score agent vs threshold."""
    scripts = independent_scripts()
    scam = [s for s in scripts if s.is_scam]
    benign = [s for s in scripts if not s.is_scam]
    if not bona:
        raise ValueError("closed-loop needs bona fide clips")

    def _bona(i: int) -> Tuple[np.ndarray, int]:
        c = bona[i % len(bona)]
        n = min(len(c.audio), int(c.sample_rate * 2.5))
        return c.audio[:n], c.sample_rate

    def _spoof(i: int, sr: int) -> np.ndarray:
        if spoof:
            y = spoof[i % len(spoof)]
            return y[: min(len(y), int(sr * 2.5))]
        x, _ = _bona(i)
        return x

    plans = {
        Hypothesis.BENIGN: lambda i: ("bona", benign[i % len(benign)]),
        Hypothesis.SOCIAL_ENGINEERING: lambda i: ("bona", scam[i % len(scam)]),
        Hypothesis.SYNTHETIC_FULL: lambda i: ("spoof", benign[i % len(benign)]),
        Hypothesis.HANDOFF: lambda i: ("spoof", scam[i % len(scam)]),
    }

    out: Dict[str, Dict[str, dict]] = {"agent": {}, "threshold": {}}
    for h, maker in plans.items():
        agent_runs: List[List[Action]] = []
        thr_runs: List[List[Action]] = []
        for k in range(n_per_class):
            kind, script = maker(k)
            audio, sr = _bona(k)
            if kind == "spoof":
                audio = _spoof(k, sr)
            ag, acts = run_pipeline_agent(audio, sr, script, acoustic_scorer=acoustic_scorer)
            agent_runs.append(acts or [Action.MONITOR])
            thr_runs.append(_threshold_from_agent(ag))
        am = _metrics(h, agent_runs)
        tm = _metrics(h, thr_runs)
        out["agent"][h.value] = {
            "hypothesis": am.hypothesis,
            "n": am.n,
            "missed_harvest": am.missed_harvest,
            "false_challenge": am.false_challenge,
            "false_warn": am.false_warn,
            "mean_cost": am.mean_cost,
            "challenge_rate": am.challenge_rate,
        }
        out["threshold"][h.value] = {
            "hypothesis": tm.hypothesis,
            "n": tm.n,
            "missed_harvest": tm.missed_harvest,
            "false_challenge": tm.false_challenge,
            "false_warn": tm.false_warn,
            "mean_cost": tm.mean_cost,
            "challenge_rate": tm.challenge_rate,
        }
    return out
