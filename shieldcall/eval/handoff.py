"""Handoff protocol: same acoustic mix, only *timing* relative to stages differs.

Aligned: vocoder starts at a harvest/payment stage.
Unaligned: same vocoded fraction, cut time far from that stage.

Utterance-mean synthetic probability is matched by construction.
If SAPC cannot rank aligned above unaligned, the timing claim is false.

Audio is Mini LibriSpeech with a pulse-formant tail — not PartialSpoof,
not neural TTS. That limit is stated in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..acoustic.changepoint import estimate_mean_shift_time
from ..acoustic.scorer import AcousticDeepfakeScorer
from ..audio.preprocessor import TelephonyPreprocessor
from ..fusion.coupling import CouplingResult, StageAlignedCoupling
from ..linguistic.scorer import LinguisticFraudScorer
from .corpora.vishing_scripts import CallScript, heldout_scripts
from .metrics import auc_roc
from .protocols import fit_acoustic_from_clips
from .speech_data import SpeechClip, load_speaker_disjoint, speech_available
from .vocoders import vocode


@dataclass
class HandoffTrial:
    clip_id: str
    condition: str  # aligned | unaligned
    cut_frac: float
    harvest_frac: float
    mean_synth: float
    coupling: CouplingResult
    n_frames: int


def _scale_turns(script: CallScript, duration: float) -> List[Tuple[float, str]]:
    if not script.turns:
        return []
    tmax = max(script.turns[-1][0] + 0.8, 1e-3)
    return [(float(t) / tmax * duration, text) for t, text in script.turns]


def _harvest_frac(script: CallScript) -> float:
    """Fraction along the script of the first harvest/payment-like turn."""
    keys = (
        "prepaid", "card", "transfer", "crypto", "store credit",
        "gift", "numbers on the", "identifier", "wage", "passkey",
        "ssn", "social security", "routing", "cvv",
    )
    n = max(len(script.turns), 1)
    for i, (_, text) in enumerate(script.turns):
        low = text.lower()
        if any(k in low for k in keys):
            return (i + 0.5) / n
    return 0.55


WINDOW = 0.34


def splice_window(
    audio: np.ndarray,
    sr: int,
    start_frac: float,
    width: float = WINDOW,
    vocoder: str = "pulse_formant",
) -> np.ndarray:
    """Replace a fixed-duration interior window with a vocoded copy (matched mix)."""
    n = len(audio)
    w = max(8, int(n * width))
    start = int(np.clip(start_frac, 0.0, 1.0 - width) * n)
    start = min(start, n - w)
    out = audio.astype(np.float32).copy()
    out[start : start + w] = vocode(audio[start : start + w], sr, vocoder)  # type: ignore[arg-type]
    return out


def splice_handoff(audio: np.ndarray, sr: int, cut_frac: float, vocoder: str = "pulse_formant") -> np.ndarray:
    return splice_window(audio, sr, start_frac=cut_frac, vocoder=vocoder)


def score_handoff_call(
    audio: np.ndarray,
    sr: int,
    turns: Sequence[Tuple[float, str]],
    scorer: AcousticDeepfakeScorer,
) -> Tuple[CouplingResult, float, int]:
    sapc = StageAlignedCoupling(n_perm=99, seed=0)
    li = LinguisticFraudScorer()
    pre = TelephonyPreprocessor(target_sr=sr)
    scorer.reset()
    synths: List[float] = []
    stimes: List[float] = []
    ti = 0
    for frame in pre.stream_from_array(audio, sr, chunk_ms=100.0):
        while ti < len(turns) and turns[ti][0] <= frame.timestamp_sec + 1e-9:
            ls = li.update(turns[ti][1], turns[ti][0])
            sapc.observe_stage(ls.discourse_stage, turns[ti][0])
            ti += 1
        a = scorer.score_frame(frame)
        if a.is_speech:
            synths.append(a.synthetic_prob)
            stimes.append(frame.timestamp_sec)
    while ti < len(turns):
        ls = li.update(turns[ti][1], turns[ti][0])
        sapc.observe_stage(ls.discourse_stage, turns[ti][0])
        ti += 1
    loc = estimate_mean_shift_time(synths, stimes, win=16)
    if loc is not None:
        sapc.observe_alarm(loc[0])
    mean_s = float(np.mean(synths)) if synths else 0.0
    return sapc.evaluate(), mean_s, len(synths)


def aligned_start(harvest_frac: float, width: float = WINDOW) -> float:
    """Vocoder *onset* at the harvest stage (clipped so the window fits)."""
    return float(np.clip(harvest_frac, 0.0, 1.0 - width))


def unaligned_start(harvest_frac: float, width: float = WINDOW) -> float:
    """Same window length, placed at the opposite end of the call."""
    a = aligned_start(harvest_frac, width)
    other = 0.02 if a > 0.35 else (1.0 - width - 0.02)
    return float(np.clip(other, 0.0, 1.0 - width))


def run_handoff_protocol(
    n_test_speakers: int = 5,
    utt_per_speaker: int = 1,
    seed: int = 0,
    vocoder: str = "pulse_formant",
) -> Dict[str, float]:
    if not speech_available():
        raise FileNotFoundError("Mini LibriSpeech required")
    train, test = load_speaker_disjoint(
        n_train_speakers=4,
        n_test_speakers=n_test_speakers,
        utt_per_speaker=max(utt_per_speaker, 3),
        min_seconds=1.8,
        max_seconds=5.0,
        seed=seed,
    )
    sr = train[0].sample_rate
    spoof_train = [vocode(c.audio, sr, vocoder) for c in train]  # type: ignore[arg-type]
    scorer = fit_acoustic_from_clips(train, spoof_train, sr)
    scam = next(s for s in heldout_scripts() if s.script_id == "h01")
    harvest = _harvest_frac(scam)
    # Short LibriSpeech clips (~4s) pack three severe stages into 1.5s, which
    # makes any change-point look "close" to a stage. Stitch three clips so
    # the script is spread over ~10s and aligned vs unaligned are separable.
    by_spk: Dict[str, List[SpeechClip]] = {}
    for c in test:
        by_spk.setdefault(c.speaker_id, []).append(c)
    stitched: List[SpeechClip] = []
    for spk, clips in by_spk.items():
        if len(clips) < 2:
            continue
        audio = np.concatenate([c.audio for c in clips[:3]])
        stitched.append(
            SpeechClip(
                path=clips[0].path,
                speaker_id=spk,
                chapter_id=clips[0].chapter_id,
                utt_id=f"{clips[0].utt_id}+",
                audio=audio,
                sample_rate=sr,
                transcript="",
            )
        )
    trials: List[HandoffTrial] = []
    for clip in stitched:
        d = len(clip.audio) / float(sr)
        turns = _scale_turns(scam, d)
        for name, cut in (
            ("aligned", aligned_start(harvest)),
            ("unaligned", unaligned_start(harvest)),
        ):
            audio = splice_handoff(clip.audio, sr, cut, vocoder=vocoder)
            coup, mean_s, nfr = score_handoff_call(audio, sr, turns, scorer)
            trials.append(
                HandoffTrial(
                    clip_id=clip.utt_id,
                    condition=name,
                    cut_frac=cut,
                    harvest_frac=harvest,
                    mean_synth=mean_s,
                    coupling=coup,
                    n_frames=nfr,
                )
            )

    aligned = [t for t in trials if t.condition == "aligned"]
    unaligned = [t for t in trials if t.condition == "unaligned"]
    # Rank aligned vs unaligned using coupling score vs mean synth
    y = [1] * len(aligned) + [0] * len(unaligned)
    coup_scores = [t.coupling.score for t in aligned] + [t.coupling.score for t in unaligned]
    mean_scores = [t.mean_synth for t in aligned] + [t.mean_synth for t in unaligned]
    stat_scores = [t.coupling.statistic for t in aligned] + [t.coupling.statistic for t in unaligned]
    pair_wins = 0
    pair_n = 0
    for a, u in zip(aligned, unaligned):
        pair_n += 1
        if a.coupling.score > u.coupling.score + 1e-9:
            pair_wins += 1
    return {
        "n_pairs": float(pair_n),
        "pair_win_rate": float(pair_wins / max(pair_n, 1)),
        "auc_coupling_score": auc_roc(y, coup_scores),
        "auc_coupling_stat": auc_roc(y, stat_scores),
        "auc_mean_synth": auc_roc(y, mean_scores),
        "mean_synth_aligned": float(np.mean([t.mean_synth for t in aligned])),
        "mean_synth_unaligned": float(np.mean([t.mean_synth for t in unaligned])),
        "mean_stat_aligned": float(np.mean([t.coupling.statistic for t in aligned])),
        "mean_stat_unaligned": float(np.mean([t.coupling.statistic for t in unaligned])),
        "mean_p_aligned": float(np.mean([t.coupling.p_value for t in aligned])),
        "mean_p_unaligned": float(np.mean([t.coupling.p_value for t in unaligned])),
    }
