"""Call-level hypotheses the defense agent entertains.

These are mutually exclusive *for the current call*. They are not
vocoder taxonomies. Likelihoods that update them are heuristic; see
``belief.py``.
"""

from __future__ import annotations

from enum import Enum


class Hypothesis(str, Enum):
    BENIGN = "benign"
    SOCIAL_ENGINEERING = "social_engineering"  # human voice, scam script
    SYNTHETIC_FULL = "synthetic_full"  # vocoded/synthetic throughout
    HANDOFF = "handoff"  # production change timed to a harvest-class stage
    UNKNOWN_FAMILY = "unknown_family"  # far from both acoustic manifolds


ALL = tuple(Hypothesis)


class Action(str, Enum):
    MONITOR = "monitor"
    CHALLENGE = "challenge"
    WARN = "warn"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"
    ADAPT = "adapt"
