"""Evaluation protocols with correct labels.

Acoustic: bona fide vs vocoded real speech, speaker-disjoint.
Linguistic: held-out call scripts vs a keyword-only baseline.
Operational fusion: a call is positive if it is a *threat*
(scam language OR vocoded voice), not if it is merely synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..acoustic.scorer import AcousticDeepfakeScorer, AcousticScore
from ..audio.channel import ChannelConfig, CodecProfile, TelephonyChannelTwin
from ..audio.preprocessor import TelephonyPreprocessor
from ..fusion.baselines import LogisticLateFusion
from ..fusion.engine import FusionEngine
from ..linguistic.scorer import LinguisticFraudScorer, LinguisticScore
from .corpora.vishing_scripts import CallScript, heldout_scripts, train_scripts
from .harness import EvalResult
from .metrics import auc_roc, equal_error_rate
from .speech_data import SpeechClip, load_speaker_disjoint, speech_available
from .vocoders import vocode, VocoderName


@dataclass
class PairScore:
    synth: float
    fraud: float
    cscf: float
    naive: float
    logreg: float
    y: int
    cell: str


def clip_mean_embedding(audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
    pre = TelephonyPreprocessor(target_sr=sr)
    scorer = AcousticDeepfakeScorer(seed=0)
    embs = []
    for frame in pre.stream_from_array(audio, sr, chunk_ms=100.0):
        s = scorer.score_frame(frame)
        if s.is_speech and s.embedding is not None:
            embs.append(s.embedding)
    if not embs:
        return None
    return np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)


def score_acoustic_clip(
    audio: np.ndarray,
    sr: int,
    scorer: AcousticDeepfakeScorer,
    profile: CodecProfile = CodecProfile.CLEAN,
    seed: int = 0,
) -> float:
    if profile != CodecProfile.CLEAN:
        audio = TelephonyChannelTwin(ChannelConfig(profile=profile, seed=seed)).apply(audio, sr)
    pre = TelephonyPreprocessor(target_sr=sr)
    scorer.reset()
    scores = []
    for frame in pre.stream_from_array(audio, sr, chunk_ms=100.0):
        s = scorer.score_frame(frame)
        if s.is_speech:
            scores.append(s.synthetic_prob)
    return float(np.mean(scores)) if scores else 0.0


def score_script(script: CallScript, use_discourse: bool = True) -> LinguisticScore:
    scorer = LinguisticFraudScorer(
        discourse_weight=0.45 if use_discourse else 0.0,
        pattern_weight=0.55 if use_discourse else 1.0,
    )
    last = LinguisticScore(timestamp_sec=0.0, fraud_prob=0.0, confidence=0.0)
    for t, text in script.turns:
        last = scorer.update(text, t)
    return last


def fit_acoustic_from_clips(
    bona: Sequence[SpeechClip],
    spoof_audio: Sequence[np.ndarray],
    sr: int,
) -> AcousticDeepfakeScorer:
    embs: List[np.ndarray] = []
    labels: List[bool] = []
    for clip in bona:
        e = clip_mean_embedding(clip.audio, sr)
        if e is not None:
            embs.append(e)
            labels.append(False)
    for audio in spoof_audio:
        e = clip_mean_embedding(audio, sr)
        if e is not None:
            embs.append(e)
            labels.append(True)
    scorer = AcousticDeepfakeScorer(seed=0)
    if embs:
        scorer.fit(embs, labels)
    return scorer


def linguistic_protocol() -> Dict[str, EvalResult]:
    """Held-out scripts: patterns-only vs patterns+stage tracker."""
    results: Dict[str, EvalResult] = {}
    for split, scripts in (("train", train_scripts()), ("heldout", heldout_scripts())):
        for use_d, name in ((False, "patterns"), (True, "patterns_sdtg")):
            labels, scores = [], []
            for s in scripts:
                sc = score_script(s, use_discourse=use_d)
                labels.append(1 if s.is_scam else 0)
                scores.append(sc.fraud_prob)
            results[f"ling_{split}_{name}"] = EvalResult(
                condition=f"{split}/{name}",
                n_samples=len(scripts),
                eer_estimate=equal_error_rate(labels, scores),
                mean_latency_ms=0.0,
                auc=auc_roc(labels, scores),
                notes=f"linguistic {split} {name}",
            )
        # traps: benign with isolated keywords should stay low under SDTG
        traps = [s for s in scripts if s.trap == "isolated_keyword"]
        if traps:
            trap_scores = [score_script(s, True).fraud_prob for s in traps]
            results[f"ling_{split}_trap_mean"] = EvalResult(
                condition=f"{split}/traps",
                n_samples=len(traps),
                eer_estimate=float(np.mean(trap_scores)),
                mean_latency_ms=0.0,
                auc=0.0,
                notes="mean fraud_prob on isolated-keyword benign (lower is better)",
            )
    return results


def acoustic_protocol(
    n_train_speakers: int = 8,
    n_test_speakers: int = 8,
    utt_per_speaker: int = 4,
    vocoders: Sequence[VocoderName] = ("pulse_formant",),
    profiles: Sequence[CodecProfile] = (CodecProfile.CLEAN, CodecProfile.NARROWBAND),
    seed: int = 0,
) -> Tuple[Dict[str, EvalResult], AcousticDeepfakeScorer, List[SpeechClip], List[SpeechClip], Dict[str, List[np.ndarray]]]:
    if not speech_available():
        raise FileNotFoundError("Mini LibriSpeech not found. Run scripts/download_speech.py")
    train, test = load_speaker_disjoint(
        n_train_speakers=n_train_speakers,
        n_test_speakers=n_test_speakers,
        utt_per_speaker=utt_per_speaker,
        min_seconds=1.2,
        max_seconds=4.0,
        seed=seed,
    )
    sr = train[0].sample_rate
    train_spoof: Dict[str, List[np.ndarray]] = {v: [] for v in vocoders}
    test_spoof: Dict[str, List[np.ndarray]] = {v: [] for v in vocoders}
    for clip in train:
        for v in vocoders:
            train_spoof[v].append(vocode(clip.audio, sr, v))
    for clip in test:
        for v in vocoders:
            test_spoof[v].append(vocode(clip.audio, sr, v))

    # Fit PMA on LPC (primary spoof family)
    fit_spoof = train_spoof.get("lpc") or next(iter(train_spoof.values()))
    scorer = fit_acoustic_from_clips(train, fit_spoof, sr)

    results: Dict[str, EvalResult] = {}
    for v in vocoders:
        for profile in profiles:
            labels, scores = [], []
            for clip in test:
                scores.append(score_acoustic_clip(clip.audio, sr, scorer, profile, seed))
                labels.append(0)
            for audio in test_spoof[v]:
                scores.append(score_acoustic_clip(audio, sr, scorer, profile, seed))
                labels.append(1)
            results[f"ac_{v}_{profile.value}"] = EvalResult(
                condition=f"acoustic/{v}/{profile.value}",
                n_samples=len(labels),
                eer_estimate=equal_error_rate(labels, scores),
                mean_latency_ms=0.0,
                auc=auc_roc(labels, scores),
                notes="speaker-disjoint bona fide vs vocoded",
            )
    return results, scorer, train, test, test_spoof


def operational_fusion_protocol(
    scorer: AcousticDeepfakeScorer,
    test_bona: Sequence[SpeechClip],
    test_spoof: Sequence[np.ndarray],
    profile: CodecProfile = CodecProfile.NARROWBAND,
    seed: int = 0,
) -> Tuple[EvalResult, List[PairScore], Dict[str, EvalResult]]:
    """Threat label = scam language OR vocoded audio."""
    sr = test_bona[0].sample_rate
    scam = [s for s in heldout_scripts() if s.is_scam]
    benign = [s for s in heldout_scripts() if not s.is_scam]
    train_scam = [s for s in train_scripts() if s.is_scam]
    train_benign = [s for s in train_scripts() if not s.is_scam]

    def cells(bona, spoof, scam_s, ben_s, n_pair: int) -> List[Tuple[np.ndarray, CallScript, int, str]]:
        out = []
        n = min(n_pair, len(bona), len(spoof), len(scam_s), len(ben_s))
        for i in range(n):
            out.append((bona[i].audio, ben_s[i % len(ben_s)], 0, "safe"))
            out.append((bona[i].audio, scam_s[i % len(scam_s)], 1, "social_engineering"))
            out.append((spoof[i], ben_s[i % len(ben_s)], 1, "spoof_probe"))
            out.append((spoof[i], scam_s[i % len(scam_s)], 1, "dual"))
        return out

    # Train logreg on train speakers' LPC if available: use test set only for
    # reporting; logreg is fit on a disjoint pairing of train scripts with
    # the first half of test? Better: use train_scripts with train audio.
    # We only have test_bona here. Fit logreg on a held-in half of the pairs
    # would leak. Use train_scripts + score_acoustic on test is wrong.
    # Fit logreg on synthetic stream scores from train scripts + acoustic
    # scores collected on bona/spoof *train* would be correct; we pass
    # only test here. Fit logreg on train scripts' (0.2, fraud) and (0.8, fraud)
    # proxy acoustics — too fake.
    # Practical: fit logreg on pairing of train scripts with *test* audio
    # would leak acoustics. Skip using test audio for logreg train.
    # Use linguistic scores from train scripts with placeholder synth
    # taken from a small train-like split: first 2 bona as "train-like"
    # is still test speaker leak.
    #
    # Fit logreg only on linguistic extremes with assumed synth:
    #   (low synth, high fraud) -> 1, (high synth, low fraud) -> 1,
    #   (low, low) -> 0, (high, high) -> 1
    # That is essentially encoding the operational label. It is a fair
    # *late fusion* baseline if those four corners are the training set
    # from train scripts' actual fraud scores and actual train acoustics.
    #
    # We'll compute train acoustics from test_bona[0:0] — not available.
    # Require caller to pass nothing extra: fit logreg on train scripts
    # with synth in {0.15, 0.75} grid. Honest: this is a weak supervised
    # fusion, not speaker-leak free acoustics.

    logreg = LogisticLateFusion()
    gx, gy, gl = [], [], []
    for s in train_scam + train_benign:
        fraud = score_script(s, True).fraud_prob
        y_lang = 1 if s.is_scam else 0
        for synth in (0.15, 0.35, 0.65, 0.85):
            y = 1 if (y_lang == 1 or synth >= 0.55) else 0
            gx.append(synth)
            gy.append(fraud)
            gl.append(y)
    logreg.fit(gx, gy, gl)

    items = cells(test_bona, test_spoof, scam, benign, n_pair=min(12, len(test_bona), len(test_spoof)))
    pairs: List[PairScore] = []
    by_mode = {m: ([], []) for m in ("cscf", "naive", "acoustic", "linguistic", "logreg")}

    for audio, script, y, cell in items:
        synth = score_acoustic_clip(audio, sr, scorer, profile, seed)
        li = score_script(script, True)
        fraud = li.fraud_prob
        fe = FusionEngine(use_conformal=False)
        for _ in range(5):
            fe.update_acoustic(
                AcousticScore(
                    timestamp_sec=1.0,
                    frame_index=1,
                    synthetic_prob=synth,
                    confidence=0.8,
                    is_speech=True,
                    features=np.zeros(64, dtype=np.float32),
                    residual_cue=synth,
                )
            )
            fe.update_linguistic(li)
        cscf = fe.fuse(1.0).risk_score
        naive = FusionEngine.naive_sum(synth, fraud)
        lr = logreg.predict_proba(synth, fraud)
        pairs.append(PairScore(synth, fraud, cscf, naive, lr, y, cell))
        by_mode["cscf"][0].append(y)
        by_mode["cscf"][1].append(cscf)
        by_mode["naive"][0].append(y)
        by_mode["naive"][1].append(naive)
        by_mode["acoustic"][0].append(y)
        by_mode["acoustic"][1].append(synth)
        by_mode["linguistic"][0].append(y)
        by_mode["linguistic"][1].append(fraud)
        by_mode["logreg"][0].append(y)
        by_mode["logreg"][1].append(lr)

    extras: Dict[str, EvalResult] = {}
    for mode, (labs, scs) in by_mode.items():
        extras[f"op_{mode}"] = EvalResult(
            condition=f"operational/{mode}",
            n_samples=len(labs),
            eer_estimate=equal_error_rate(labs, scs),
            mean_latency_ms=0.0,
            auc=auc_roc(labs, scs),
            notes="threat = scam OR vocoded",
        )
        # disagreement recall at 0.5
        recs = []
        for p in pairs:
            if p.cell in ("social_engineering", "spoof_probe"):
                score = {"cscf": p.cscf, "naive": p.naive, "acoustic": p.synth, "linguistic": p.fraud, "logreg": p.logreg}[mode]
                recs.append(1.0 if score >= 0.5 else 0.0)
        extras[f"op_{mode}"].extras["disagreement_recall@0.5"] = float(np.mean(recs)) if recs else 0.0
        safe_fp = [1.0 if {"cscf": p.cscf, "naive": p.naive, "acoustic": p.synth, "linguistic": p.fraud, "logreg": p.logreg}[mode] >= 0.5 else 0.0 for p in pairs if p.cell == "safe"]
        extras[f"op_{mode}"].extras["safe_fpr@0.5"] = float(np.mean(safe_fp)) if safe_fp else 0.0

    return extras["op_cscf"], pairs, extras


def adaptation_protocol(
    train_bona: Sequence[SpeechClip],
    test_bona: Sequence[SpeechClip],
    n_shots: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    """Fit on pulse-formant, measure unseen LPC EER before/after k-shot."""
    sr = train_bona[0].sample_rate
    pulse_train = [vocode(c.audio, sr, "pulse_formant") for c in train_bona]
    lpc_train = [vocode(c.audio, sr, "lpc") for c in train_bona]
    lpc_test = [vocode(c.audio, sr, "lpc") for c in test_bona]
    scorer = fit_acoustic_from_clips(train_bona, pulse_train, sr)

    def eer_unseen(sc: AcousticDeepfakeScorer) -> float:
        labels, scores = [], []
        for c in test_bona:
            scores.append(score_acoustic_clip(c.audio, sr, sc, CodecProfile.CLEAN, seed))
            labels.append(0)
        for a in lpc_test:
            scores.append(score_acoustic_clip(a, sr, sc, CodecProfile.CLEAN, seed))
            labels.append(1)
        return equal_error_rate(labels, scores)

    eer_before = eer_unseen(scorer)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(lpc_train), size=min(n_shots, len(lpc_train)), replace=False)
    for i in idx:
        emb = clip_mean_embedding(lpc_train[int(i)], sr)
        if emb is not None:
            scorer.adapt(emb, is_synthetic=True)
    eer_after = eer_unseen(scorer)
    return {
        "eer_before": float(eer_before),
        "eer_after": float(eer_after),
        "eer_reduction": float(eer_before - eer_after),
        "n_shots": float(n_shots),
    }


def _script_turns(script: CallScript) -> List[str]:
    return [t[1] for t in script.turns]


def linguistic_ablation_protocol(
    *,
    confirmatory: bool = True,
    asr_wer: float = 0.0,
    seed: int = 0,
) -> Dict[str, EvalResult]:
    """Locked-lexicon confirmatory linguistic table.

    Systems: narrow keywords (PATTERN_GROUPS), wide lexicon (STAGE_EMISSIONS
    bag), SDTG (HMM path), NTM (fit on train scripts only).
    Population: independent_scripts if confirmatory else author held-out.
    """
    from ..linguistic.asr_noise import degrade_turns
    from ..linguistic.discourse import emission_only_score, path_only_score, wide_lexicon_score
    from ..linguistic.ntm import NeuralTrajectoryModel
    from ..linguistic.scorer import LinguisticFraudScorer
    from .corpora.independent_scripts import independent_scripts
    from .metrics import bootstrap_ci

    train = train_scripts()
    test = independent_scripts() if confirmatory else heldout_scripts()
    ntm = NeuralTrajectoryModel(seed=seed)
    ntm.fit(([t[1] for t in s.turns] for s in train), (1 if s.is_scam else 0 for s in train))

    def narrow_score(turns: List[str]) -> float:
        scorer = LinguisticFraudScorer(discourse_weight=0.0, pattern_weight=1.0)
        last = None
        for i, t in enumerate(turns):
            last = scorer.update(t, float(i))
        return float(last.fraud_prob if last else 0.0)

    systems = {
        "narrow_keywords": lambda turns: narrow_score(turns),
        "wide_lexicon": lambda turns: emission_only_score(turns),
        "sdtg": lambda turns: path_only_score(turns),
        "ntm": lambda turns: ntm.score(turns),
    }
    results: Dict[str, EvalResult] = {}
    for name, fn in systems.items():
        labels, scores = [], []
        for s in test:
            turns = _script_turns(s)
            if asr_wer > 0:
                turns = degrade_turns(turns, wer=asr_wer, seed=seed)
            labels.append(1 if s.is_scam else 0)
            scores.append(float(fn(turns)))
        auc, lo, hi = bootstrap_ci(labels, scores, auc_roc, n_boot=400, seed=seed)
        traps = [sc for s, sc in zip(test, scores) if s.trap == "isolated_keyword" and not s.is_scam]
        results[f"ling_{name}{'_asr' if asr_wer else ''}"] = EvalResult(
            condition=f"ling/{name}/wer={asr_wer}",
            n_samples=len(test),
            eer_estimate=equal_error_rate(labels, scores),
            mean_latency_ms=0.0,
            auc=auc,
            notes=f"confirmatory={confirmatory} wer={asr_wer}",
            extras={
                "auc_lo": lo,
                "auc_hi": hi,
                "trap_mean": float(np.mean(traps)) if traps else 0.0,
            },
        )
    return results


def fusion_ablation_from_pairs(pairs: Sequence[PairScore]) -> Dict[str, EvalResult]:
    """Attribute complementary-cell recall to floors vs blend vs OR."""
    from .metrics import bootstrap_ci, fpr_at_threshold, recall_at_threshold

    results: Dict[str, EvalResult] = {}
    modes = {
        "naive": lambda p: p.naive,
        "logreg": lambda p: p.logreg,
        "cscf": lambda p: p.cscf,
        "floors": lambda p: FusionEngine.floors_only(p.synth, p.fraud),
        "calibrated_or": lambda p: FusionEngine.calibrated_or(p.synth, p.fraud),
    }
    for name, fn in modes.items():
        labels = [p.y for p in pairs]
        scores = [float(fn(p)) for p in pairs]
        disc_y, disc_s, safe_s = [], [], []
        for p, sc in zip(pairs, scores):
            if p.cell in ("social_engineering", "spoof_probe"):
                disc_y.append(1)
                disc_s.append(sc)
            if p.cell == "safe":
                safe_s.append(sc)
        auc, lo, hi = bootstrap_ci(labels, scores, auc_roc, n_boot=400, seed=0)
        rec = float(np.mean([1.0 if s >= 0.5 else 0.0 for s in disc_s])) if disc_s else 0.0
        fpr = float(np.mean([1.0 if s >= 0.5 else 0.0 for s in safe_s])) if safe_s else 0.0
        results[f"fuse_{name}"] = EvalResult(
            condition=f"fusion/{name}",
            n_samples=len(pairs),
            eer_estimate=equal_error_rate(labels, scores),
            mean_latency_ms=0.0,
            auc=auc,
            extras={"auc_lo": lo, "auc_hi": hi, "disc_recall@0.5": rec, "safe_fpr@0.5": fpr},
        )
    return results
