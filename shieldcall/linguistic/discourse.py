"""
Scam Discourse Trajectory Graph (SDTG)
======================================

Keyword lists miss the *script structure* of social-engineering calls.
Real scams walk a progressive path:

  GREETING  ->  AUTHORITY  ->  PROBLEM  ->  URGENCY  ->  HARVEST  ->  PAYMENT  ->  SECRECY  ->  THREAT

SDTG models this as a probabilistic stage machine. Emissions come from
pattern groups; transitions encode how professional scammers escalate.
The path-likelihood of the observed stage sequence is a signal no
static bag-of-words detector can produce  -  and it fires earlier when
the *trajectory* is scam-like even before the payment ask.

Novelty: continuous, streaming stage inference with path log-likelihood
as a first-class fraud feature, not post-hoc explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re

import numpy as np


# Ordered stages of a classic vishing / impersonation script
STAGES: List[str] = [
    "GREETING",
    "AUTHORITY",
    "PROBLEM",
    "URGENCY",
    "HARVEST",
    "PAYMENT",
    "SECRECY",
    "THREAT",
    "BENIGN",
]

# Emission patterns: stage -> regex list
STAGE_EMISSIONS: Dict[str, List[str]] = {
    "GREETING": [
        r"\bhello\b", r"\bgood (morning|afternoon|evening)\b",
        r"\bthis is (a )?(courtesy )?call\b", r"\bcalling from\b",
        r"\bon the line\b", r"\bdesk here\b", r"\bi am with\b",
    ],
    "AUTHORITY": [
        r"\birs\b", r"\bsocial security\b", r"\bssa\b", r"\bmedicare\b",
        r"\bbank\b", r"\bfederal\b", r"\bmicrosoft\b", r"\bapple support\b",
        r"\blaw enforcement\b", r"\bdepartment of\b", r"\bagent\b",
        r"\btax (bureau|office|desk|assessment)\b", r"\bbenefits integrity\b",
        r"\bprosecutor\b", r"\bcard services\b", r"\bhelpdesk\b",
        r"\benrollment desk\b", r"\bcloud account\b", r"\bpower cooperative\b",
        r"\bparcel claims\b", r"\bnational assessment\b", r"\bpublic defender\b",
        r"\bcourt clerk\b", r"\bawards clearing\b", r"\bintake line\b",
        r"\bstatus review\b", r"\bdevice security\b",
    ],
    "PROBLEM": [
        r"\bunusual activity\b", r"\bsuspicious\b", r"\bcompromised\b",
        r"\bvirus\b", r"\bmalware\b", r"\breach(ed)?\b", r"\bfraud(ulent)?\b",
        r"\bsuspended\b", r"\blocked\b", r"\bin trouble\b", r"\baccident\b",
        r"\bfroz(e|en)\b", r"\bfreeze\b", r"\bdiscrepanc\w+\b", r"\bdrain\b",
        r"\bintrusion\b", r"\bworm\b", r"\bcrash\b", r"\btamper\w*\b",
        r"\bhostile login\b", r"\bcollision\b", r"\bmismatch\b",
        r"\bflagged\b", r"\bhold on the\b",
    ],
    "URGENCY": [
        r"\bimmediately\b", r"\bright now\b", r"\bact now\b",
        r"\bwithin \d+ (hour|minute|day)s?\b", r"\burgent\b",
        r"\bbefore (it|we|they|noon)\b", r"\bexpires?\b", r"\blast chance\b",
        r"\bthis morning\b", r"\bsame[- ]day\b", r"\bcannot wait\b",
        r"\bshort window\b", r"\bclose of business\b", r"\bwithin the hour\b",
        r"\btime-sensitive\b", r"\bwindow closes\b", r"\bthirty minutes\b",
    ],
    "HARVEST": [
        r"\bssn\b", r"\bsocial security number\b", r"\bdate of birth\b",
        r"\bbank account\b", r"\brouting number\b", r"\bpin\b", r"\bpassword\b",
        r"\bverification code\b", r"\bone[- ]time (code|password|pin)\b",
        r"\bcard number\b", r"\bcvv\b", r"\bmother'?s maiden\b",
        r"\btaxpayer identification\b", r"\bwage (statements?|slips?|number|identifier)\b",
        r"\blong number\b", r"\bthree digits\b", r"\bdeposit account\b",
        r"\bidentifier from your card\b", r"\bpasskey\b", r"\bback digits\b",
        r"\bdeposit coordinates\b",
    ],
    "PAYMENT": [
        r"\bgift card\b", r"\bwire transfer\b", r"\bbitcoin\b", r"\bcrypto\b",
        r"\bsend money\b", r"\bmust pay\b", r"\biTunes\b", r"\bsteam card\b",
        r"\bwestern union\b", r"\bmoneygram\b", r"\bpaypal\b",
        r"\bprepaid (store )?cards?\b", r"\bstore credit\b", r"\bcrypto (voucher|address|transfer)\b",
        r"\bmoney transfer counter\b", r"\bpharmacy prepaid\b",
        r"\btransfer counter\b",
    ],
    "SECRECY": [
        r"\bdon'?t tell\b", r"\bkeep (this )?confidential\b", r"\bdo not inform\b",
        r"\bsecret\b", r"\bbetween us\b", r"\bdo not call (anyone|the bank)\b",
        r"\bstay on the (line|phone)\b",
        r"\bdo not discuss\b", r"\bnot for your (spouse|employer)\b",
        r"\bkeep this off\b", r"\bnot public\b", r"\bdo not ring the printed\b",
        r"\bshould not hear\b", r"\bdo not loop in\b", r"\btipping off\b",
        r"\boff social media\b",
    ],
    "THREAT": [
        r"\bwarrant\b", r"\barrest\b", r"\bjail\b", r"\blegal action\b",
        r"\bprosecut\b", r"\bfined?\b", r"\bdeported\b", r"\blose (your )?(home|benefits)\b",
        r"\bmarshals? will detain\b", r"\bofficers are already\b", r"\bobstruction\b",
        r"\bpickup\b", r"\bcourthouse\b", r"\bholding\b", r"\bcollections\b",
        r"\bcomplaint draft\b", r"\bdetain you\b", r"\bstays? in holding\b",
    ],
    "BENIGN": [
        r"\bhow are you\b", r"\bthank you\b", r"\bhave a (nice|good) day\b",
        r"\bappointment\b", r"\breminder\b", r"\bsurvey\b",
        r"\bno action (is )?required\b", r"\bno payment\b", r"\bhave a good\b",
    ],
}

# Log-space transition bonuses: progressing forward through scam stages
# is more "script-like" than random jumps.
def _default_transition_matrix() -> np.ndarray:
    n = len(STAGES)
    idx = {s: i for i, s in enumerate(STAGES)}
    # Base: small self-loop + uniform leak
    T = np.full((n, n), 0.02, dtype=np.float64)
    for i in range(n):
        T[i, i] = 0.35
    # Scam progression chain
    chain = ["GREETING", "AUTHORITY", "PROBLEM", "URGENCY", "HARVEST", "PAYMENT", "SECRECY", "THREAT"]
    for a, b in zip(chain, chain[1:]):
        T[idx[a], idx[b]] += 0.35
        T[idx[a], idx[a]] += 0.05
    # Skip-ahead (aggressive scammers)
    for i, a in enumerate(chain):
        for b in chain[i + 2 : i + 4]:
            T[idx[a], idx[b]] += 0.08
    # BENIGN self
    T[idx["BENIGN"], idx["BENIGN"]] = 0.7
    # Normalize rows
    T = T / T.sum(axis=1, keepdims=True)
    return np.log(T + 1e-12)


@dataclass
class DiscourseState:
    timestamp_sec: float
    stage: str
    stage_probs: Dict[str, float]
    path_loglik: float
    path_score: float  # calibrated 0..1 scam-path probability
    stages_visited: List[str] = field(default_factory=list)
    progression_depth: int = 0


class ScamDiscourseGraph:
    """Streaming HMM-style stage tracker for scam scripts."""

    def __init__(self):
        self._compiled = {
            stage: [re.compile(p, re.IGNORECASE) for p in pats]
            for stage, pats in STAGE_EMISSIONS.items()
        }
        self._log_trans = _default_transition_matrix()
        self._stage_index = {s: i for i, s in enumerate(STAGES)}
        self._log_belief = np.log(np.ones(len(STAGES)) / len(STAGES))
        self._path_loglik = 0.0
        self._visited: List[str] = []
        self._last_stage = "BENIGN"

    def reset(self) -> None:
        self._log_belief = np.log(np.ones(len(STAGES)) / len(STAGES))
        self._path_loglik = 0.0
        self._visited = []
        self._last_stage = "BENIGN"

    def _emission_logprobs(self, text: str) -> np.ndarray:
        hits = np.zeros(len(STAGES), dtype=np.float64)
        for stage, patterns in self._compiled.items():
            c = sum(1 for p in patterns if p.search(text))
            hits[self._stage_index[stage]] = c
        # Laplace-smoothed multinomial emission
        scores = hits + 0.15
        # Boost BENIGN slightly when nothing hits
        if hits.sum() == 0:
            scores[self._stage_index["BENIGN"]] += 1.0
        probs = scores / scores.sum()
        return np.log(probs + 1e-12)

    def update(self, text: str, timestamp_sec: float) -> DiscourseState:
        if not text or not text.strip():
            probs = {s: float(np.exp(self._log_belief[i])) for i, s in enumerate(STAGES)}
            return DiscourseState(
                timestamp_sec=timestamp_sec,
                stage=self._last_stage,
                stage_probs=probs,
                path_loglik=self._path_loglik,
                path_score=self._path_to_score(),
                stages_visited=list(self._visited),
                progression_depth=self._progression_depth(),
            )

        log_em = self._emission_logprobs(text)
        # Predict
        # log_belief'[j] = logsumexp_i(log_belief[i] + log_trans[i,j])
        pred = np.zeros(len(STAGES), dtype=np.float64)
        for j in range(len(STAGES)):
            pred[j] = np.logaddexp.reduce(self._log_belief + self._log_trans[:, j])
        # Update
        unnorm = pred + log_em
        unnorm -= np.max(unnorm)
        belief = np.exp(unnorm)
        belief /= belief.sum()
        self._log_belief = np.log(belief + 1e-12)

        stage_i = int(np.argmax(belief))
        stage = STAGES[stage_i]

        # Path log-likelihood contribution
        trans_ll = self._log_trans[self._stage_index[self._last_stage], stage_i]
        self._path_loglik += float(trans_ll + log_em[stage_i])
        self._last_stage = stage
        if not self._visited or self._visited[-1] != stage:
            self._visited.append(stage)

        probs = {s: float(belief[i]) for i, s in enumerate(STAGES)}
        return DiscourseState(
            timestamp_sec=timestamp_sec,
            stage=stage,
            stage_probs=probs,
            path_loglik=self._path_loglik,
            path_score=self._path_to_score(),
            stages_visited=list(self._visited),
            progression_depth=self._progression_depth(),
        )

    def _progression_depth(self) -> int:
        chain = ["GREETING", "AUTHORITY", "PROBLEM", "URGENCY", "HARVEST", "PAYMENT", "SECRECY", "THREAT"]
        depth = 0
        visited = set(self._visited)
        for i, s in enumerate(chain):
            if s in visited:
                depth = i + 1
        return depth

    def _path_to_score(self) -> float:
        """Map stage structure and path log-likelihood to [0, 1].

        Structure (depth, diversity, terminal stages) is the main term.
        Mean per-step path log-likelihood is a secondary term so the
        transition matrix is not dead code. Benign-only paths are capped.
        """
        depth = self._progression_depth()
        scam_stages = [s for s in self._visited if s not in ("BENIGN",)]
        if not scam_stages and depth == 0:
            return 0.0
        diversity = len(set(scam_stages))
        struct = 0.12 * depth + 0.10 * diversity
        if "PAYMENT" in self._visited:
            struct += 0.20
        if "THREAT" in self._visited:
            struct += 0.15
        if "HARVEST" in self._visited:
            struct += 0.12
        if "SECRECY" in self._visited:
            struct += 0.10
        n = max(len(self._visited), 1)
        ll = self._path_loglik / n
        ll_score = float(np.clip((ll + 3.5) / 3.0, 0.0, 1.0))
        score = 0.75 * struct + 0.25 * ll_score
        if depth <= 1 and "PAYMENT" not in self._visited and "THREAT" not in self._visited:
            score = min(score, 0.35)
        return float(np.clip(score, 0.0, 1.0))
