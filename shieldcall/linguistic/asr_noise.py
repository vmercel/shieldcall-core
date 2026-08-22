"""ASR-in-the-loop surrogate when a production recognizer is absent.

Whisper is optional. The confirmatory paper path uses a documented
character/phonetic noise model with a target WER so linguistic scores
can be measured under recognition error without claiming a deployed ASR.

If ``whisper`` is importable, ``transcribe_whisper`` is available.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


# Common telephony ASR substitutions (not fitted to STAGE_EMISSIONS).
_SUBS = (
    ("social security", "so shall security"),
    ("gift card", "gift cart"),
    ("routing number", "rooting number"),
    ("internal revenue", "intern revenue"),
    ("warrant", "warren"),
    ("password", "pass word"),
    ("verification", "varification"),
    ("immediately", "in mediately"),
    ("prepaid", "pre paid"),
    ("bitcoin", "bit coin"),
)


def degrade_text(text: str, wer: float = 0.25, seed: int = 0) -> str:
    """Approximate a target word-error rate with substitutions and drops."""
    acc = 0
    for ch in text:
        acc = (acc * 31 + ord(ch)) % 100000
    rng = np.random.RandomState(seed + acc)
    out = text
    for a, b in _SUBS:
        if a in out.lower() and rng.rand() < min(1.0, wer * 2):
            # case-insensitive replace of first occurrence
            idx = out.lower().find(a)
            out = out[:idx] + b + out[idx + len(a) :]
    words = out.split()
    kept: List[str] = []
    for w in words:
        r = rng.rand()
        if r < wer * 0.35:
            continue  # deletion
        if r < wer * 0.55:
            kept.append(w[::-1][: max(2, len(w) - 1)])  # scramble
        else:
            kept.append(w)
    return " ".join(kept) if kept else out


def degrade_turns(turns: List[str], wer: float = 0.25, seed: int = 0) -> List[str]:
    return [degrade_text(t, wer=wer, seed=seed + i) for i, t in enumerate(turns)]


def transcribe_whisper(audio: np.ndarray, sr: int) -> Optional[str]:
    """Optional Whisper path. Returns None if the extra is not installed."""
    try:
        import whisper  # type: ignore
    except Exception:
        return None
    model = whisper.load_model("tiny")
    # whisper expects 16 kHz float
    if sr != 16000:
        import math

        n = int(math.ceil(len(audio) * 16000 / sr))
        t_old = np.linspace(0, 1, len(audio), endpoint=False)
        t_new = np.linspace(0, 1, n, endpoint=False)
        audio = np.interp(t_new, t_old, audio.astype(np.float64)).astype(np.float32)
    result = model.transcribe(audio, fp16=False)
    return str(result.get("text") or "")
