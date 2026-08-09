"""
Streaming linguistic fraud-intent scorer.

Fuses:
  - Weighted multi-group pattern hits (tactical cues)
  - Scam Discourse Trajectory Graph path score (strategic structure)
  - Temporal escalation trajectory inside a sliding window

This dual tactical+strategic design is what separates ShieldCall from
simple keyword filters used in call-center speech analytics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
import re

import numpy as np

from .discourse import ScamDiscourseGraph, DiscourseState


PATTERN_GROUPS: Dict[str, Dict] = {
    "government_impersonation": {
        "weight": 0.85,
        "patterns": [
            r"\birs\b", r"\bsocial security\b", r"\bssa\b", r"\binternal revenue\b",
            r"\bwarrant\b", r"\barrest\b", r"\bfederal agent\b", r"\blaw enforcement\b",
            r"\bmedicare\b", r"\bdepartment of justice\b",
        ],
    },
    "payment_urgency": {
        "weight": 0.80,
        "patterns": [
            r"\bgift card\b", r"\bwire transfer\b", r"\bcryptocurrency\b", r"\bbitcoin\b",
            r"\bimmediately\b", r"\bwithin \d+ hours\b", r"\bact now\b", r"\bright now\b",
            r"\bmust pay\b", r"\bsend money\b", r"\bwestern union\b", r"\bmoneygram\b",
        ],
    },
    "secrecy_pressure": {
        "weight": 0.75,
        "patterns": [
            r"\bdon'?t tell\b", r"\bkeep (this )?confidential\b", r"\bdo not inform\b",
            r"\bsecret\b", r"\bbetween us\b", r"\bstay on the (line|phone)\b",
        ],
    },
    "identity_harvest": {
        "weight": 0.70,
        "patterns": [
            r"\bssn\b", r"\bsocial security number\b", r"\bdate of birth\b",
            r"\bbank account\b", r"\brouting number\b", r"\bpin\b", r"\bpassword\b",
            r"\bverification code\b", r"\bone[- ]time (code|password|pin)\b",
            r"\bcard number\b", r"\bcvv\b",
        ],
    },
    "tech_support": {
        "weight": 0.65,
        "patterns": [
            r"\bvirus\b", r"\bmalware\b", r"\bremote access\b", r"\bteamviewer\b",
            r"\byour computer\b", r"\bmicrosoft support\b", r"\bapple support\b",
            r"\banydesk\b",
        ],
    },
    "family_emergency": {
        "weight": 0.80,
        "patterns": [
            r"\bgrandson\b", r"\bgranddaughter\b", r"\bin trouble\b", r"\baccident\b",
            r"\bhospital\b", r"\bjailed\b", r"\barrested\b", r"\bbail\b",
        ],
    },
}


@dataclass
class LinguisticScore:
    timestamp_sec: float
    fraud_prob: float
    confidence: float
    active_groups: List[str] = field(default_factory=list)
    escalation_factor: float = 1.0
    raw_hits: int = 0
    discourse_stage: str = "BENIGN"
    discourse_path_score: float = 0.0
    progression_depth: int = 0
    stages_visited: List[str] = field(default_factory=list)


class LinguisticFraudScorer:
    """
    Streaming fraud-intent scorer over transcript fragments.

    fraud_prob = blend(pattern_score, discourse_path_score) * escalation
    """

    def __init__(
        self,
        window_seconds: float = 45.0,
        escalation_window: int = 5,
        pattern_weight: float = 0.55,
        discourse_weight: float = 0.45,
    ):
        self.window_seconds = window_seconds
        self.escalation_window = escalation_window
        self.pattern_weight = pattern_weight
        self.discourse_weight = discourse_weight
        self._events: Deque[tuple[float, float, List[str]]] = deque()
        self._compiled = {
            name: [re.compile(p, re.IGNORECASE) for p in group["patterns"]]
            for name, group in PATTERN_GROUPS.items()
        }
        self.discourse = ScamDiscourseGraph()

    def reset(self) -> None:
        self._events.clear()
        self.discourse.reset()

    def _score_text(self, text: str) -> tuple[float, List[str], int]:
        if not text or not text.strip():
            return 0.0, [], 0

        active = []
        total_weight = 0.0
        hits = 0
        for name, patterns in self._compiled.items():
            group_hit = False
            for pat in patterns:
                if pat.search(text):
                    group_hit = True
                    hits += 1
            if group_hit:
                active.append(name)
                total_weight += PATTERN_GROUPS[name]["weight"]

        raw = 1.0 - np.exp(-0.9 * total_weight)
        return float(raw), active, hits

    def update(self, text_fragment: str, timestamp_sec: float) -> LinguisticScore:
        score, active, hits = self._score_text(text_fragment)
        disc: DiscourseState = self.discourse.update(text_fragment, timestamp_sec)

        self._events.append((timestamp_sec, score, active))
        cutoff = timestamp_sec - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

        if not self._events:
            return LinguisticScore(timestamp_sec=timestamp_sec, fraud_prob=0.0, confidence=0.0)

        scores = [e[1] for e in self._events]
        recent = scores[-self.escalation_window :]
        earlier = scores[: -self.escalation_window] or scores[:1]
        recent_mean = float(np.mean(recent))
        earlier_mean = float(np.mean(earlier))
        escalation = 1.0
        if recent_mean > earlier_mean + 0.08:
            escalation = 1.0 + 2.5 * min(recent_mean - earlier_mean, 0.4)

        pattern_base = float(np.max(scores[-3:])) if len(scores) >= 3 else recent_mean
        blended = (
            self.pattern_weight * pattern_base
            + self.discourse_weight * disc.path_score
        )
        fraud_prob = float(np.clip(blended * escalation, 0.0, 1.0))

        confidence = min(0.3 + 0.12 * len(self._events) + 0.08 * hits + 0.05 * disc.progression_depth, 0.95)

        recent_groups = set()
        for e in list(self._events)[-self.escalation_window :]:
            recent_groups.update(e[2])

        return LinguisticScore(
            timestamp_sec=timestamp_sec,
            fraud_prob=fraud_prob,
            confidence=confidence,
            active_groups=sorted(recent_groups),
            escalation_factor=escalation,
            raw_hits=hits,
            discourse_stage=disc.stage,
            discourse_path_score=disc.path_score,
            progression_depth=disc.progression_depth,
            stages_visited=list(disc.stages_visited),
        )
