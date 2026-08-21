"""
Adaptation interface  -  confirmed examples for online acoustic updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import time

import numpy as np


@dataclass
class AdaptationExample:
    features: np.ndarray
    is_synthetic: bool
    source: str = "unknown"  # human_review | challenge_response | verified_liveness
    timestamp_sec: float = 0.0
    family: str = "unknown"  # synthesizer family tag when known
    meta: dict = field(default_factory=dict)


class AdaptationBuffer:
    """
    Bounded buffer of confirmed examples for few-shot / online updates.

    Feeds Prototype Memory Adaptation (PMA) on the acoustic scorer and
    the coverage-debt tracker.
    """

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._examples: List[AdaptationExample] = []

    def add(self, example: AdaptationExample) -> None:
        self._examples.append(example)
        if len(self._examples) > self.max_size:
            self._examples = self._examples[-self.max_size :]

    def get_recent(self, n: int = 50) -> List[AdaptationExample]:
        return self._examples[-n:]

    def by_family(self, family: str) -> List[AdaptationExample]:
        return [e for e in self._examples if e.family == family]

    def clear(self) -> None:
        self._examples.clear()

    def __len__(self) -> int:
        return len(self._examples)


class ChallengeResponseProtocol:
    """
    Interactive liveness challenge protocol.

    Novelty: instead of passive detection only, ShieldCall can request
    a challenge (repeat phrase, change speaking rate, answer a nonce)
    and use the acoustic response as a high-trust adaptation / verify
    signal  -  closing the loop between detection and enrollment of new
    synthesizer failure modes.
    """

    CHALLENGES = [
        "Please say the numbers {nonce} clearly.",
        "Please count from {a} to {b} at a normal pace.",
        "Please repeat: 'the blue sky has silver linings' quickly then slowly.",
    ]

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._pending: Optional[dict] = None

    def issue(self) -> dict:
        nonce = self._rng.randint(1000, 9999)
        a = self._rng.randint(1, 5)
        template = self.CHALLENGES[self._rng.randint(0, len(self.CHALLENGES))]
        prompt = template.format(nonce=nonce, a=a, b=a + 4)
        self._pending = {
            "prompt": prompt,
            "nonce": nonce,
            "issued_at": time.time(),
            "expected_tokens": [str(nonce), "blue sky", "silver"],
        }
        return dict(self._pending)

    def verify(
        self,
        transcript: str,
        acoustic_synth_prob: Optional[float] = None,
        synth_max: float = 0.55,
        timeout_sec: float = 30.0,
    ) -> bool:
        """Pass only if the nonce/phrase matches *and* the voice is not synthetic."""
        if not self.verify_transcript(transcript, timeout_sec=timeout_sec):
            return False
        if acoustic_synth_prob is not None and acoustic_synth_prob >= synth_max:
            return False
        return True

    def verify_transcript(self, transcript: str, timeout_sec: float = 30.0) -> bool:
        if not self._pending:
            return False
        if time.time() - self._pending["issued_at"] > timeout_sec:
            self._pending = None
            return False
        text = transcript.lower()
        # Soft match: any expected token is a weak pass; nonce is strong
        if str(self._pending["nonce"]) in text:
            self._pending = None
            return True
        hits = sum(1 for t in self._pending["expected_tokens"] if t.lower() in text)
        ok = hits >= 1
        if ok:
            self._pending = None
        return ok
