"""Factorial discourse model (TDD v2.0 sections 7.3, 7.4, 7.5; ADR-017).

A single SDTG stage index conflates three things a warning must
separate: how far along a script the call has got, which
persuasion tactics are active, and what the caller is asking
the user to do. This module tracks all three per turn:

  z = (stage, tactics, action)

Filtering factorizes approximately (TDD eq. 15): the stage
factor is the existing SDTG forward recursion, reused rather
than reimplemented, and the tactic and action factors are
independent Bayes filters over their own explicit lexicons.
No factor reads another factor's posterior during filtering.
The hazard (TDD eq. 16) is the only place the factors meet.

This module is shadow-only: it has no reference to the
pipeline, the fusion engine, the serve layer, or the cascade.
CSCF stays the production fusion and the SDTG stays the live
language stream. The emissions are the hash-locked lexicons
only. TDD blends the lexicon floor with a small learned text
encoder; no such encoder exists in the repo and none is
invented here. No language model runs in the decision path.

The action set A is completed from the TDD text, whose
section 7.3 line is cut off at the page edge after "remote
acc". Remote access is the only reading consistent with the
rest of the document (TDD Table 16 gives guidance for a
remote-access request), and the completion is stated in
ADR-017. The harmful set follows TDD eq. 16 exactly: every
action except none and personal data. Personal data is
therefore not harmful in this model, even though asking
for it is serious; the TDD definition is followed, not
silently corrected.

Speaker roles follow TDD section 7.4: only far-end turns
can constitute a requested action. A near-end (protected
user) turn never updates the action belief and reports
action none for that turn, while still updating stage and
tactic evidence. When no speaker is supplied, a turn is
treated as far-end, the only safe default for a transcript
feed that does not diarise.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .discourse import LEXICON_LOCK, STAGES, ScamDiscourseGraph


# ---------------------------------------------------------------------------
# Tactics (TDD 7.3): authority, urgency, fear or threat, reward,
# liking or rapport, secrecy or isolation.
# ---------------------------------------------------------------------------

TACTICS: List[str] = [
    "authority",
    "urgency",
    "fear_threat",
    "reward",
    "liking_rapport",
    "secrecy_isolation",
]

TACTIC_EMISSIONS: Dict[str, List[str]] = {
    "authority": [
        r"\birs\b", r"\bsocial security\b", r"\bssa\b", r"\bmedicare\b",
        r"\bbank\b", r"\bfederal\b", r"\bmicrosoft\b", r"\bapple support\b",
        r"\blaw enforcement\b", r"\bdepartment of\b", r"\bagent\b",
        r"\bcourt\b", r"\bpolice\b", r"\bgovernment\b", r"\btax office\b",
        r"\bsecurity department\b",
    ],
    "urgency": [
        r"\bimmediately\b", r"\bright now\b", r"\bact now\b",
        r"\bwithin \d+ (hour|minute|day)s?\b", r"\burgent\b",
        r"\bbefore (it|we|they|noon)\b", r"\bexpires?\b", r"\blast chance\b",
        r"\bsame[- ]day\b", r"\bcannot wait\b", r"\bshort window\b",
        r"\btime-sensitive\b", r"\bwindow closes\b", r"\bthirty minutes\b",
    ],
    "fear_threat": [
        r"\bwarrant\b", r"\barrest\b", r"\bjail\b", r"\blegal action\b",
        r"\bprosecut\b", r"\bfined?\b", r"\bdeported\b",
        r"\blose (your )?(home|benefits)\b", r"\bdetain\b",
        r"\bofficers are already\b", r"\bcollections\b",
    ],
    "reward": [
        r"\bwon\b", r"\bprize\b", r"\breward\b", r"\bbonus\b",
        r"\brefund\b", r"\bcashback\b", r"\bfree\b", r"\blottery\b",
        r"\bgrant\b", r"\bcongratulations\b", r"\byou (have )?qualified\b",
    ],
    "liking_rapport": [
        r"\btrust me\b", r"\bmy friend\b", r"\bdear\b",
        r"\bi care\b", r"\bhere to help\b", r"\bgood news\b",
        r"\byou'?re in good hands\b", r"\bi understand\b",
    ],
    "secrecy_isolation": [
        r"\bdon'?t tell\b", r"\bkeep (this )?confidential\b",
        r"\bdo not inform\b", r"\bsecret\b", r"\bbetween us\b",
        r"\bdo not call (anyone|the bank)\b", r"\bstay on the (line|phone)\b",
        r"\bdo not discuss\b", r"\bkeep this off\b",
    ],
}

TACTIC_LEXICON_LOCK = hashlib.sha256(
    json.dumps(TACTIC_EMISSIONS, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Requested actions (TDD 7.3). A = {none, personal data, one-time
# code or credential, payment or gift card, wire or crypto
# transfer, remote access}. HARMFUL_ACTIONS = A minus {none,
# personal data} (TDD eq. 16).
# ---------------------------------------------------------------------------

ACTION_NONE = "none"
ACTION_PERSONAL_DATA = "personal_data"
ACTION_OTP_CREDENTIAL = "one_time_code_or_credential"
ACTION_PAYMENT_GIFT_CARD = "payment_or_gift_card"
ACTION_WIRE_CRYPTO = "wire_or_crypto_transfer"
ACTION_REMOTE_ACCESS = "remote_access"

ACTIONS: List[str] = [
    ACTION_NONE,
    ACTION_PERSONAL_DATA,
    ACTION_OTP_CREDENTIAL,
    ACTION_PAYMENT_GIFT_CARD,
    ACTION_WIRE_CRYPTO,
    ACTION_REMOTE_ACCESS,
]

HARMFUL_ACTIONS = frozenset(
    {
        ACTION_OTP_CREDENTIAL,
        ACTION_PAYMENT_GIFT_CARD,
        ACTION_WIRE_CRYPTO,
        ACTION_REMOTE_ACCESS,
    }
)

ACTION_EMISSIONS: Dict[str, List[str]] = {
    ACTION_PERSONAL_DATA: [
        r"\bssn\b", r"\bsocial security number\b", r"\bdate of birth\b",
        r"\bbank account\b", r"\brouting number\b", r"\bhome address\b",
        r"\bmother'?s maiden\b", r"\baccount number\b",
    ],
    ACTION_OTP_CREDENTIAL: [
        r"\bverification code\b", r"\bone[- ]time (code|password|pin)\b",
        r"\bpassword\b", r"\bpin\b", r"\bpasscode\b", r"\bsecurity code\b",
        r"\bcard number\b", r"\bcvv\b", r"\bthree digits\b",
        r"\bread (me )?the code\b", r"\bcode (you )?(just )?(got|received)\b",
    ],
    ACTION_PAYMENT_GIFT_CARD: [
        r"\bgift cards?\b", r"\biTunes\b", r"\bsteam cards?\b",
        r"\bprepaid (store )?cards?\b", r"\bsend money\b",
        r"\bmust pay\b", r"\bpay (now|today|immediately)\b",
        r"\bpayment\b", r"\bstore credit\b", r"\bbuy (a )?gift cards?",
    ],
    ACTION_WIRE_CRYPTO: [
        r"\bwire transfer\b", r"\bbitcoin\b", r"\bcrypto\b",
        r"\bwestern union\b", r"\bmoneygram\b", r"\bcrypto (voucher|address|transfer)\b",
        r"\btransfer (the )?money\b", r"\bsend (it )?(by|via) wire\b",
    ],
    ACTION_REMOTE_ACCESS: [
        r"\bremote access\b", r"\bteamviewer\b", r"\banydesk\b",
        r"\binstall\b", r"\bdownload (the )?app\b",
        r"\bshare your screen\b", r"\ballow (me |us )?access\b",
        r"\bgrant (me |us )?access\b", r"\bopen (the )?link\b",
    ],
}

ACTION_LEXICON_LOCK = hashlib.sha256(
    json.dumps(ACTION_EMISSIONS, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Benign-mention guard (TDD 7.4): reported speech and negation of
# a request. A guarded far-end turn has its action emission
# blended toward uniform; it is down-weighted, not deleted.
# The example sentences are TDD's own.
# ---------------------------------------------------------------------------

REPORTED_SPEECH_PATTERNS: List[str] = [
    r"\bthey told me\b", r"\bsomeone (called|told|said)\b",
    r"\bhe (said|told)\b", r"\bshe (said|told)\b",
    r"\bthe caller (said|told|asked)\b",
    r"\bmy (mother|father|relative|friend) (said|told)\b",
    r"\bi (got|received) (a )?(call|message|text)\b",
    r"\breported\b", r"\ba message (said|claimed)\b",
]

NEGATION_PATTERNS: List[str] = [
    r"\bnever ask\b", r"\bwill never ask\b", r"\bwe (do )?not ask\b",
    r"\bno legitimate\b", r"\bnever (give|share|send|tell)\b",
    r"\bdon'?t (give|share|send) (them |him |her )?(your|the|a)\b",
    r"\bdo not (give|share|send)\b",
    r"\bwill not ask (you )?for\b",
]

_FAR_SPEAKERS = {"far", "far_end", "far-end", "caller", "agent", "scammer"}
_NEAR_SPEAKERS = {"near", "near_end", "near-end", "user", "callee", "protected"}


def _normalise_speaker(speaker: str) -> str:
    lowered = str(speaker).strip().lower()
    if lowered in _FAR_SPEAKERS:
        return "far"
    if lowered in _NEAR_SPEAKERS:
        return "near"
    raise ValueError(f"unknown speaker {speaker!r}: expected a far-end or near-end label")


@dataclass
class FactorialConfig:
    """Configuration for the factorial discourse model (ADR-017).

    Every numeric value is a prior constant, stated here and in
    the ADR, not a fitted value: no labelled tactic or action
    data exists in the repo to fit them from.

    ``horizon_turns`` is k in TDD eq. 16. ``min_stage_turns``
    is the HSMM dwell floor: a new stage must be the SDTG
    argmax for that many consecutive turns before the
    committed stage moves. ``guard_emission_weight`` is the
    weight a guarded turn's action emission keeps; the rest
    is uniform, so 0.25 means 75 percent down-weighted.
    """

    horizon_turns: int = 3
    min_stage_turns: int = 2
    tactic_prior: float = 0.10
    tactic_persist_on: float = 0.85
    tactic_enter: float = 0.10
    tactic_hit_active: float = 0.70
    tactic_hit_inactive: float = 0.08
    action_stay: float = 0.60
    action_prior_none: float = 0.70
    guard_emission_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.horizon_turns < 1:
            raise ValueError("horizon_turns must be at least 1")
        if self.min_stage_turns < 1:
            raise ValueError("min_stage_turns must be at least 1")
        for name in (
            "tactic_prior",
            "tactic_persist_on",
            "tactic_enter",
            "tactic_hit_active",
            "tactic_hit_inactive",
            "action_stay",
            "action_prior_none",
            "guard_emission_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.tactic_hit_active <= self.tactic_hit_inactive:
            raise ValueError("a hit must be likelier under the active tactic")


@dataclass
class FactorialState:
    """Factorial state after one turn (TDD eq. 14, 16)."""

    timestamp_sec: float
    speaker: str
    # Stage factor: committed stage (dwell floor applied), the
    # SDTG argmax, and the full SDTG posterior.
    stage: str
    stage_map: str
    stage_probs: Dict[str, float]
    path_score: float
    stages_visited: List[str] = field(default_factory=list)
    progression_depth: int = 0
    # Tactic factor: posterior probability per tactic.
    tactic_probs: Dict[str, float] = field(default_factory=dict)
    # Action factor: the action requested by this turn (none for
    # near-end turns), and the action belief.
    action: str = ACTION_NONE
    action_probs: Dict[str, float] = field(default_factory=dict)
    harm_probability: float = 0.0
    # Hazard (eq. 16) and the fraud score passed to fusion.
    hazard: float = 0.0
    fraud_score: float = 0.0
    # Benign-mention guard flags for this turn.
    guard_reported_speech: bool = False
    guard_negated_request: bool = False
    guard_user_initiated: bool = False
    guard_speaker_role: bool = False

    @property
    def guard_applied(self) -> bool:
        return (
            self.guard_reported_speech
            or self.guard_negated_request
            or self.guard_user_initiated
            or self.guard_speaker_role
        )

    @property
    def active_tactics(self) -> List[str]:
        return [t for t in TACTICS if self.tactic_probs.get(t, 0.0) >= 0.5]


def harmful_probability(action_probs: Dict[str, float]) -> float:
    """Posterior probability that the requested action is harmful."""
    return float(sum(action_probs.get(a, 0.0) for a in HARMFUL_ACTIONS))


def requested_action_active(state: FactorialState, threshold: float = 0.5) -> bool:
    """True when the current turn requests a harmful action.

    This is the exact factorial form that replaces the SDTG-stage
    proxy in the cascade trigger (ADR-016 follow-up). A near-end
    turn reports action none, so it never counts here.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return state.action in HARMFUL_ACTIONS and state.action_probs.get(state.action, 0.0) >= threshold


class FactorialDiscourseModel:
    """Streaming factorial discourse tracker (TDD 7.3, guard 7.4).

    One instance serves one call. Feed turns in order with the
    speaker of each turn. The model consumes turns as given: it
    does not gate on stable ASR prefixes and does not mask PII
    (TDD 7.1), both follow-ups owned by the ASR bridge.
    """

    def __init__(self, config: Optional[FactorialConfig] = None):
        self.config = config or FactorialConfig()
        self._sdtg = ScamDiscourseGraph()
        self._compiled_tactics = {
            tactic: [re.compile(p, re.IGNORECASE) for p in pats]
            for tactic, pats in TACTIC_EMISSIONS.items()
        }
        self._compiled_actions = {
            action: [re.compile(p, re.IGNORECASE) for p in pats]
            for action, pats in ACTION_EMISSIONS.items()
        }
        self._compiled_reported = [re.compile(p, re.IGNORECASE) for p in REPORTED_SPEECH_PATTERNS]
        self._compiled_negation = [re.compile(p, re.IGNORECASE) for p in NEGATION_PATTERNS]
        self.reset()

    def reset(self) -> None:
        self._sdtg.reset()
        self._committed_stage = "BENIGN"
        self._pending_stage: Optional[str] = None
        self._pending_run = 0
        self._tactic_belief = {t: self.config.tactic_prior for t in TACTICS}
        rest = (1.0 - self.config.action_prior_none) / (len(ACTIONS) - 1)
        self._action_belief = {
            a: (self.config.action_prior_none if a == ACTION_NONE else rest) for a in ACTIONS
        }
        self._last_user_actions: Optional[frozenset] = None

    # -- stage factor: SDTG belief plus the dwell floor -------------

    def _commit_stage(self, stage_map: str) -> str:
        if stage_map == self._committed_stage:
            self._pending_stage = None
            self._pending_run = 0
            return self._committed_stage
        if stage_map == self._pending_stage:
            self._pending_run += 1
        else:
            self._pending_stage = stage_map
            self._pending_run = 1
        if self._pending_run >= self.config.min_stage_turns:
            self._committed_stage = stage_map
            self._pending_stage = None
            self._pending_run = 0
        return self._committed_stage

    # -- tactic factor: independent Bernoulli chains ------------------

    def _update_tactics(self, text: str) -> None:
        cfg = self.config
        for tactic in TACTICS:
            hit = any(p.search(text) for p in self._compiled_tactics[tactic])
            prior = self._tactic_belief[tactic]
            predicted = prior * cfg.tactic_persist_on + (1.0 - prior) * cfg.tactic_enter
            if hit:
                lik_active, lik_inactive = cfg.tactic_hit_active, cfg.tactic_hit_inactive
            else:
                lik_active, lik_inactive = 1.0 - cfg.tactic_hit_active, 1.0 - cfg.tactic_hit_inactive
            numerator = predicted * lik_active
            denominator = numerator + (1.0 - predicted) * lik_inactive
            self._tactic_belief[tactic] = numerator / denominator if denominator > 0 else predicted

    # -- action factor: per-turn emission, far-end only ---------------

    def _action_hits(self, text: str) -> Dict[str, int]:
        return {
            action: sum(1 for p in pats if p.search(text))
            for action, pats in self._compiled_actions.items()
        }

    def _emission_probs(self, hits: Dict[str, int]) -> Dict[str, float]:
        # One hit on a specific request phrase is strong evidence:
        # the non-hit base is 0.05, so a single hit dominates the
        # emission while a turn with no hits still favours none.
        if sum(hits.values()) == 0:
            raw = {a: (1.0 if a == ACTION_NONE else 0.05) for a in ACTIONS}
        else:
            raw = {a: (0.05 + float(hits.get(a, 0))) for a in ACTIONS}
        total = sum(raw.values())
        return {a: raw[a] / total for a in ACTIONS}

    def _update_action(self, text: str, *, guarded: bool) -> str:
        hits = self._action_hits(text)
        emission = self._emission_probs(hits)
        if guarded:
            weight = self.config.guard_emission_weight
            uniform = 1.0 / len(ACTIONS)
            emission = {a: weight * emission[a] + (1.0 - weight) * uniform for a in ACTIONS}
        stay = self.config.action_stay
        uniform = 1.0 / len(ACTIONS)
        prior_mix = {a: stay * self._action_belief[a] + (1.0 - stay) * uniform for a in ACTIONS}
        unnorm = {a: emission[a] * prior_mix[a] for a in ACTIONS}
        total = sum(unnorm.values())
        self._action_belief = {a: unnorm[a] / total for a in ACTIONS} if total > 0 else dict(prior_mix)
        return max(ACTIONS, key=lambda a: self._action_belief[a])

    # -- main update ---------------------------------------------------

    def update(self, text: str, timestamp_sec: float, speaker: str = "far") -> FactorialState:
        role = _normalise_speaker(speaker)
        if not math.isfinite(float(timestamp_sec)):
            raise ValueError("timestamp_sec must be finite")
        if not text or not text.strip():
            return self._state(timestamp_sec, role, action=ACTION_NONE, harm_probability=0.0)

        # Stage and tactic evidence comes from both speakers (7.4(i)).
        sdtg_state = self._sdtg.update(text, float(timestamp_sec))
        committed = self._commit_stage(sdtg_state.stage)
        self._update_tactics(text)

        if role == "near":
            # The user's words never constitute a request (7.4(i)).
            # Record what the user asked for, so a far-end echo of
            # the same action next turn is recognised as
            # user-initiated (7.4(iii)).
            hits = self._action_hits(text)
            self._last_user_actions = frozenset(a for a, n in hits.items() if n > 0)
            return self._state(
                timestamp_sec, role, action=ACTION_NONE, harm_probability=0.0, speaker_guard=True
            )

        reported = any(p.search(text) for p in self._compiled_reported)
        negated = any(p.search(text) for p in self._compiled_negation)
        hits = self._action_hits(text)
        top_action = max(hits, key=lambda a: hits[a]) if sum(hits.values()) else ACTION_NONE
        user_initiated = bool(
            self._last_user_actions and top_action in self._last_user_actions
        )
        guarded = reported or negated or user_initiated
        action = self._update_action(text, guarded=guarded)
        self._last_user_actions = None
        harm = harmful_probability(self._action_belief)
        return self._state(
            timestamp_sec,
            role,
            action=action,
            harm_probability=harm,
            reported=reported,
            negated=negated,
            user_initiated=user_initiated,
        )

    def _state(
        self,
        timestamp_sec: float,
        role: str,
        *,
        action: str,
        harm_probability: float,
        reported: bool = False,
        negated: bool = False,
        user_initiated: bool = False,
        speaker_guard: bool = False,
    ) -> FactorialState:
        # The hazard projects the current harmful probability
        # forward over the horizon under an independence
        # approximation, stated in ADR-017 item 6. For a
        # near-end turn the belief is unchanged, so the hazard
        # still reflects the harmful request already made.
        belief_harm = harmful_probability(self._action_belief)
        hazard = 1.0 - (1.0 - belief_harm) ** self.config.horizon_turns
        fraud = max(hazard, float(harm_probability))
        probs = {s: float(np.exp(self._sdtg._log_belief[i])) for i, s in enumerate(STAGES)}
        return FactorialState(
            timestamp_sec=float(timestamp_sec),
            speaker=role,
            stage=self._committed_stage,
            stage_map=self._sdtg._last_stage,
            stage_probs=probs,
            path_score=float(self._sdtg._path_to_score()),
            stages_visited=list(self._sdtg._visited),
            progression_depth=self._sdtg._progression_depth(),
            tactic_probs=dict(self._tactic_belief),
            action=action,
            action_probs=dict(self._action_belief),
            harm_probability=float(harm_probability),
            hazard=float(hazard),
            fraud_score=float(fraud),
            guard_reported_speech=bool(reported),
            guard_negated_request=bool(negated),
            guard_user_initiated=bool(user_initiated),
            guard_speaker_role=bool(speaker_guard),
        )


# ---------------------------------------------------------------------------
# Lead-time evaluation (TDD 15.4, 7.3) and the mandatory TF-IDF
# prefix baseline (TDD 7.5).
# ---------------------------------------------------------------------------


@dataclass
class DialogueTurn:
    text: str
    timestamp_sec: float
    speaker: str = "far"


@dataclass
class CallTranscript:
    """One call for lead-time evaluation.

    ``label`` is 1 for a scam call, 0 for benign. For a scam
    call, ``first_harm_index`` is the turn index of the first
    harmful request; None means the call contains no harmful
    request, and it is counted separately, not scored for lead
    time.
    """

    turns: List[DialogueTurn]
    label: int
    first_harm_index: Optional[int] = None

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("label must be 0 (benign) or 1 (scam)")
        if self.first_harm_index is not None and not (
            0 <= self.first_harm_index < len(self.turns)
        ):
            raise ValueError("first_harm_index must index a turn in the call")


# A scorer maps a call's turns to one fraud score per turn, in
# order. Both the factorial model and the TF-IDF baseline are
# used through this shape, so the comparison is at equal
# false-alert rate on identical inputs.
TurnScorer = Callable[[List[DialogueTurn]], List[float]]


@dataclass(frozen=True)
class LeadTimeReport:
    """Lead-time measurement at one threshold (TDD 15.4)."""

    n_calls: int
    n_scam: int
    n_benign: int
    n_scam_with_harm: int
    threshold: float
    false_alerts: int
    false_alerts_per_100_benign: float
    warned_before_harm: int
    missed: int
    lead_times_sec: Tuple[float, ...]
    median_lead_time_sec: Optional[float]
    mean_lead_time_sec: Optional[float]

    @property
    def fraction_warned_before_harm(self) -> float:
        if not self.n_scam_with_harm:
            return 0.0
        return self.warned_before_harm / self.n_scam_with_harm


def evaluate_lead_time(
    calls: Sequence[CallTranscript],
    threshold: float,
    scorer: TurnScorer,
) -> LeadTimeReport:
    """Measure lead time at a fixed threshold.

    A scam call is warned when a turn at or before the harmful
    request scores at or above the threshold; the lead time is
    the harmful request time minus the first warning time. A
    warning after the harmful request is late, not a lead: it
    counts as missed. A benign call false-alerts when any turn
    scores at or above the threshold.
    """
    calls = list(calls)
    if not calls:
        raise ValueError("evaluate_lead_time needs at least one call")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    lead_times: List[float] = []
    false_alerts = 0
    warned = 0
    missed = 0
    n_scam = 0
    n_benign = 0
    n_scam_with_harm = 0
    for call in calls:
        scores = scorer(list(call.turns))
        if len(scores) != len(call.turns):
            raise ValueError("scorer must return one score per turn")
        if call.label == 0:
            n_benign += 1
            if any(float(s) >= threshold for s in scores):
                false_alerts += 1
            continue
        n_scam += 1
        if call.first_harm_index is None:
            continue
        n_scam_with_harm += 1
        harm_turn = call.turns[call.first_harm_index]
        warned_at: Optional[float] = None
        for idx, score in enumerate(scores):
            if idx > call.first_harm_index:
                break
            if float(score) >= threshold:
                warned_at = float(call.turns[idx].timestamp_sec)
                break
        if warned_at is None:
            missed += 1
        else:
            warned += 1
            lead_times.append(float(harm_turn.timestamp_sec) - warned_at)
    return LeadTimeReport(
        n_calls=len(calls),
        n_scam=n_scam,
        n_benign=n_benign,
        n_scam_with_harm=n_scam_with_harm,
        threshold=float(threshold),
        false_alerts=false_alerts,
        false_alerts_per_100_benign=100.0 * false_alerts / n_benign if n_benign else 0.0,
        warned_before_harm=warned,
        missed=missed,
        lead_times_sec=tuple(lead_times),
        median_lead_time_sec=float(np.median(lead_times)) if lead_times else None,
        mean_lead_time_sec=float(np.mean(lead_times)) if lead_times else None,
    )


def threshold_at_false_alert_budget(
    benign_scores: Sequence[Sequence[float]], budget_per_100: float
) -> float:
    """Most sensitive threshold whose false-alert rate is at or below the budget.

    False-alert rate is measured per call: a benign call
    false-alerts when any of its scores reaches the threshold.
    The scan starts at the lowest observed score and rises:
    the first threshold whose false-alert rate meets the
    budget is the most sensitive one that does so. The
    candidate above every observed score (zero false alerts)
    is last, so a zero budget still returns a usable
    threshold instead of the trivial never-warn value.
    """
    benign = [list(map(float, scores)) for scores in benign_scores]
    if not benign:
        raise ValueError("threshold_at_false_alert_budget needs benign calls")
    if not math.isfinite(float(budget_per_100)) or not 0.0 <= budget_per_100 <= 100.0:
        raise ValueError("budget_per_100 must be in [0, 100]")
    observed = sorted({s for scores in benign for s in scores})
    candidates = observed + ([observed[-1] + 1e-9] if observed else [1.0])
    for threshold in candidates:
        false_alerts = sum(1 for scores in benign if any(s >= threshold for s in scores))
        if 100.0 * false_alerts / len(benign) <= budget_per_100:
            return float(threshold)
    return float(candidates[-1])


@dataclass(frozen=True)
class LeadTimeComparison:
    """Factorial model versus TF-IDF baseline at equal false-alert rate."""

    budget_per_100_benign: float
    factorial: LeadTimeReport
    baseline: LeadTimeReport
    factorial_wins_on_lead_time: bool


def compare_lead_time(
    calls: Sequence[CallTranscript],
    factorial_scorer: TurnScorer,
    baseline_scorer: TurnScorer,
    budget_per_100: float = 10.0,
) -> LeadTimeComparison:
    """Compare lead times at the same false-alert budget (TDD Table 31).

    Each model's threshold is set from the benign calls alone,
    at the stated budget, before either model is scored on the
    scam calls. The factorial model wins only when its median
    lead time is strictly greater than the baseline's. When
    neither model warns before harm, or the medians tie, it
    does not win: the kill criterion then says to ship the
    baseline plus the SDTG stage trace.
    """
    calls = list(calls)
    benign_scores = {
        "factorial": [factorial_scorer(list(c.turns)) for c in calls if c.label == 0],
        "baseline": [baseline_scorer(list(c.turns)) for c in calls if c.label == 0],
    }
    factorial_threshold = threshold_at_false_alert_budget(
        benign_scores["factorial"], budget_per_100
    )
    baseline_threshold = threshold_at_false_alert_budget(
        benign_scores["baseline"], budget_per_100
    )
    factorial_report = evaluate_lead_time(calls, factorial_threshold, factorial_scorer)
    baseline_report = evaluate_lead_time(calls, baseline_threshold, baseline_scorer)
    wins = bool(
        factorial_report.median_lead_time_sec is not None
        and (
            baseline_report.median_lead_time_sec is None
            or factorial_report.median_lead_time_sec > baseline_report.median_lead_time_sec
        )
    )
    return LeadTimeComparison(
        budget_per_100_benign=float(budget_per_100),
        factorial=factorial_report,
        baseline=baseline_report,
        factorial_wins_on_lead_time=wins,
    )


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfPrefixBaseline:
    """TF-IDF plus logistic regression, scored on prefixes (TDD 7.5).

    Trained on whole-call texts from the same split as the model
    under test, scored on cumulative turn prefixes. The logistic
    regression is plain full-batch gradient descent in NumPy
    with fixed settings and zero initialisation, so training is
    deterministic. This is the mandatory baseline every language
    result is reported beside: it is deliberately plain, and it
    exists to be beaten on lead time, not on full-call AUC.
    """

    def __init__(
        self,
        max_features: int = 500,
        iterations: int = 200,
        learning_rate: float = 0.5,
        l2: float = 0.01,
    ):
        if max_features < 1:
            raise ValueError("max_features must be at least 1")
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        self.max_features = int(max_features)
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self._vocab: List[str] = []
        self._index: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self._bias = 0.0

    @property
    def is_fitted(self) -> bool:
        return self._weights is not None

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TfidfPrefixBaseline":
        texts = list(texts)
        labels = list(labels)
        if len(texts) != len(labels) or not texts:
            raise ValueError("fit needs a non-empty set of texts with matching labels")
        if any(label not in (0, 1) for label in labels):
            raise ValueError("labels must be 0 (benign) or 1 (scam)")
        if len(set(labels)) != 2:
            raise ValueError("fit needs both classes present")
        tokenised = [_tokenise(t) for t in texts]
        doc_freq: Dict[str, int] = {}
        for tokens in tokenised:
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        ranked = sorted(doc_freq.items(), key=lambda kv: (-kv[1], kv[0]))
        self._vocab = [token for token, _ in ranked[: self.max_features]]
        self._index = {token: i for i, token in enumerate(self._vocab)}
        n_docs = len(texts)
        self._idf = np.array(
            [math.log((n_docs + 1.0) / (doc_freq[token] + 1.0)) + 1.0 for token in self._vocab],
            dtype=np.float64,
        )
        matrix = self._matrix(tokenised)
        y = np.asarray(labels, dtype=np.float64)
        weights = np.zeros(len(self._vocab), dtype=np.float64)
        bias = 0.0
        for _ in range(self.iterations):
            logits = matrix @ weights + bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            error = probs - y
            gradient = matrix.T @ error / n_docs + self.l2 * weights
            weights -= self.learning_rate * gradient
            bias -= self.learning_rate * float(error.mean())
        self._weights = weights
        self._bias = float(bias)
        return self

    def _matrix(self, tokenised: Sequence[Sequence[str]]) -> np.ndarray:
        assert self._idf is not None
        rows = np.zeros((len(tokenised), len(self._vocab)), dtype=np.float64)
        for row, tokens in enumerate(tokenised):
            counts: Dict[int, float] = {}
            for token in tokens:
                idx = self._index.get(token)
                if idx is not None:
                    counts[idx] = counts.get(idx, 0.0) + 1.0
            for idx, count in counts.items():
                rows[row, idx] = count * self._idf[idx]
            norm = float(np.linalg.norm(rows[row]))
            if norm > 0:
                rows[row] /= norm
        return rows

    def score_text(self, text: str) -> float:
        if not self.is_fitted:
            raise ValueError("baseline must be fitted before scoring")
        vector = self._matrix([_tokenise(text)])[0]
        logit = float(vector @ self._weights + self._bias)
        return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit)))))

    def score_prefixes(self, turns: Sequence[DialogueTurn]) -> List[float]:
        """One score per turn, each on the text accumulated so far."""
        accumulated: List[str] = []
        scores: List[float] = []
        for turn in turns:
            accumulated.append(turn.text)
            scores.append(self.score_text(" ".join(accumulated)))
        return scores


def factorial_scorer_for(model: FactorialDiscourseModel) -> TurnScorer:
    """Adapt a factorial model instance to the turn-scorer shape.

    The instance is reset before each call in the returned
    scorer, so one model object scores many calls independently.
    For evaluation, prefer a fresh model per call via a factory;
    this adapter exists for the common single-model case.
    """

    def score(turns: List[DialogueTurn]) -> List[float]:
        model.reset()
        return [
            model.update(turn.text, turn.timestamp_sec, turn.speaker).fraud_score
            for turn in turns
        ]

    return score


__all__ = [
    "ACTION_EMISSIONS",
    "ACTION_LEXICON_LOCK",
    "ACTION_NONE",
    "ACTION_OTP_CREDENTIAL",
    "ACTION_PAYMENT_GIFT_CARD",
    "ACTION_PERSONAL_DATA",
    "ACTION_REMOTE_ACCESS",
    "ACTION_WIRE_CRYPTO",
    "ACTIONS",
    "CallTranscript",
    "DialogueTurn",
    "FactorialConfig",
    "FactorialDiscourseModel",
    "FactorialState",
    "HARMFUL_ACTIONS",
    "LEXICON_LOCK",
    "LeadTimeComparison",
    "LeadTimeReport",
    "NEGATION_PATTERNS",
    "REPORTED_SPEECH_PATTERNS",
    "TACTIC_EMISSIONS",
    "TACTIC_LEXICON_LOCK",
    "TACTICS",
    "TfidfPrefixBaseline",
    "compare_lead_time",
    "evaluate_lead_time",
    "factorial_scorer_for",
    "harmful_probability",
    "requested_action_active",
    "threshold_at_false_alert_budget",
]
