#!/usr/bin/env python3
"""
Scientific ablations: prove (or falsify) each novelty claim with numbers.

Outputs a markdown-friendly table. Commit results under docs/results/ when
filing or publishing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import time

import numpy as np

from shieldcall.audio.channel import CodecProfile
from shieldcall.audio.preprocessor import TelephonyPreprocessor
from shieldcall.acoustic.scorer import AcousticDeepfakeScorer
from shieldcall.linguistic.scorer import LinguisticFraudScorer, LinguisticScore
from shieldcall.linguistic.discourse import ScamDiscourseGraph
from shieldcall.fusion.engine import FusionEngine
from shieldcall.eval.harness import (
    generate_synthetic_benchmark,
    evaluate_acoustic_channel,
    evaluate_adaptation_recovery,
)
from shieldcall.eval.metrics import equal_error_rate, auc_roc


SCAM_SCRIPT = [
    (0.0, "Hello, this is a courtesy call from the IRS and your bank."),
    (1.0, "We detected unusual activity and your account is compromised."),
    (2.0, "You must act immediately within 2 hours."),
    (3.0, "Verify your social security number and bank account now."),
    (4.0, "Purchase gift cards and read the numbers. Do not tell anyone."),
    (5.0, "There is a warrant for your arrest if you refuse."),
]

BENIGN_SCRIPT = [
    (0.0, "Hello, this is a reminder about your dentist appointment tomorrow."),
    (1.0, "Please arrive ten minutes early and bring your insurance card."),
    (2.0, "Thank you and have a nice day."),
]


@dataclass
class Row:
    name: str
    metric: str
    value: float
    notes: str = ""


def _linguistic_scores(script: List[Tuple[float, str]], use_discourse: bool = True) -> float:
    scorer = LinguisticFraudScorer(
        discourse_weight=0.45 if use_discourse else 0.0,
        pattern_weight=1.0 if not use_discourse else 0.55,
    )
    last = 0.0
    for t, text in script:
        s = scorer.update(text, t)
        last = s.fraud_prob
    return last


def ablation_linguistic() -> List[Row]:
    rows = []
    scam_full = _linguistic_scores(SCAM_SCRIPT, use_discourse=True)
    scam_pat = _linguistic_scores(SCAM_SCRIPT, use_discourse=False)
    ben_full = _linguistic_scores(BENIGN_SCRIPT, use_discourse=True)
    ben_pat = _linguistic_scores(BENIGN_SCRIPT, use_discourse=False)
    rows.append(Row("linguistic_scam_full", "fraud_prob", scam_full, "patterns+SDTG"))
    rows.append(Row("linguistic_scam_patterns_only", "fraud_prob", scam_pat, "patterns only"))
    rows.append(Row("linguistic_benign_full", "fraud_prob", ben_full, "patterns+SDTG"))
    rows.append(Row("linguistic_benign_patterns_only", "fraud_prob", ben_pat, "patterns only"))
    rows.append(
        Row(
            "linguistic_margin_full",
            "scam-benign",
            scam_full - ben_full,
            "higher is better separation",
        )
    )
    rows.append(
        Row(
            "linguistic_margin_patterns",
            "scam-benign",
            scam_pat - ben_pat,
            "baseline without SDTG weight",
        )
    )
    g = ScamDiscourseGraph()
    st = g.update("", 0.0)
    for t, text in SCAM_SCRIPT:
        st = g.update(text, t)
    rows.append(
        Row("sdtg_scam_path_score", "path_score", st.path_score, f"depth={st.progression_depth}")
    )
    g.reset()
    for t, text in BENIGN_SCRIPT:
        st = g.update(text, t)
    rows.append(
        Row("sdtg_benign_path_score", "path_score", st.path_score, f"depth={st.progression_depth}")
    )
    return rows


def _fuse_call(
    audio: np.ndarray,
    sr: int,
    script: List[Tuple[float, str]],
    mode: str,
) -> float:
    """
    mode: cscf | naive_sum | acoustic_only | linguistic_only
    """
    pre = TelephonyPreprocessor(target_sr=sr)
    ac = AcousticDeepfakeScorer(seed=0)
    li = LinguisticFraudScorer()
    fe = FusionEngine(use_conformal=False)
    li_idx = 0
    last_risk = 0.0
    last_li = LinguisticScore(timestamp_sec=0.0, fraud_prob=0.0, confidence=0.0)

    for frame in pre.stream_from_array(audio, sr, chunk_ms=100.0):
        a = ac.score_frame(frame)
        while li_idx < len(script) and script[li_idx][0] <= frame.timestamp_sec + 1e-9:
            last_li = li.update(script[li_idx][1], script[li_idx][0])
            li_idx += 1

        if mode == "cscf":
            fe.update_acoustic(a)
            fe.update_linguistic(last_li)
            if frame.frame_index % 5 == 0:
                last_risk = fe.fuse(frame.timestamp_sec).risk_score
        elif mode == "naive_sum":
            synth = a.synthetic_prob if a.is_speech else 0.0
            last_risk = float(np.clip(0.4 * synth + 0.6 * last_li.fraud_prob, 0, 1))
        elif mode == "acoustic_only":
            if a.is_speech:
                last_risk = a.synthetic_prob
        elif mode == "linguistic_only":
            last_risk = last_li.fraud_prob
        else:
            raise ValueError(mode)
    return last_risk


def ablation_fusion() -> List[Row]:
    samples = generate_synthetic_benchmark(n_human=8, n_synth=8, seed=1)
    rows = []
    modes = ["cscf", "naive_sum", "acoustic_only", "linguistic_only"]
    for mode in modes:
        labels, scores = [], []
        for s in samples:
            if s.is_synthetic:
                script = SCAM_SCRIPT
            else:
                script = [(0.3, s.transcript)] if s.transcript else BENIGN_SCRIPT
            r = _fuse_call(s.audio, s.sample_rate, script, mode)
            scores.append(r)
            labels.append(1 if s.is_synthetic else 0)
        rows.append(Row(f"fusion_{mode}", "EER", equal_error_rate(labels, scores)))
        rows.append(Row(f"fusion_{mode}", "AUC", auc_roc(labels, scores)))
    return rows


def ablation_channel() -> List[Row]:
    samples = generate_synthetic_benchmark(n_human=10, n_synth=10, seed=2)
    rows = []
    for profile in [CodecProfile.CLEAN, CodecProfile.NARROWBAND, CodecProfile.HARSH_VOIP]:
        r = evaluate_acoustic_channel(samples, profile, seed=2)
        rows.append(Row(f"acoustic_{profile.value}", "EER", r.eer_estimate, f"AUC={r.auc:.3f}"))
        rows.append(
            Row(f"acoustic_{profile.value}", "AUC", r.auc, f"lat_ms={r.mean_latency_ms:.2f}")
        )
    return rows


def ablation_adaptation() -> List[Row]:
    m = evaluate_adaptation_recovery(n_shots=5, seed=0)
    return [
        Row("pma_gap_before", "gap", m["mean_gap_before"]),
        Row("pma_gap_after_5shot", "gap", m["mean_gap_after"]),
        Row("pma_gap_reduction", "delta", m["gap_reduction"], "positive = adaptation helps"),
    ]


def ablation_regimes() -> List[Row]:
    """
    Hard cases where modalities disagree  -  this is where CSCF must
    catch threats that unimodal detectors miss.
    """
    from shieldcall.eval.harness import _synth_tone, _synth_vocoder_like
    from shieldcall.fusion.explain import classify_regime

    sr = 8000
    # Scripts compressed into audio duration so linguistic path fully fires
    scam_fit = [
        (0.2, "Hello from the IRS and your bank regarding unusual activity."),
        (0.6, "You must verify your social security number immediately."),
        (1.0, "Purchase gift cards and do not tell anyone."),
        (1.4, "There is a warrant for your arrest if you refuse."),
    ]
    benign_fit = [
        (0.3, "Hello, reminder about your dentist appointment tomorrow."),
        (0.9, "Please arrive early. Thank you and have a nice day."),
    ]
    cases = [
        # (label_high_risk, audio, script, name)
        (1, _synth_tone(sr, 2.0, 160, seed=1), scam_fit, "human_voice_scam_language"),
        (1, _synth_vocoder_like(sr, 2.0, 160, seed=2), benign_fit, "synth_voice_benign_language"),
        (0, _synth_tone(sr, 2.0, 160, seed=3), benign_fit, "human_voice_benign_language"),
        (1, _synth_vocoder_like(sr, 2.0, 160, seed=4), scam_fit, "synth_voice_scam_language"),
    ]
    rows = []
    per_case_cscf = {}
    for mode in ["cscf", "naive_sum", "acoustic_only", "linguistic_only"]:
        labels, scores = [], []
        for y, audio, script, name in cases:
            sc = _fuse_call(audio, sr, script, mode)
            scores.append(sc)
            labels.append(y)
            if mode == "cscf":
                per_case_cscf[name] = sc
        rows.append(Row(f"regime_{mode}", "EER", equal_error_rate(labels, scores)))
        rows.append(Row(f"regime_{mode}", "AUC", auc_roc(labels, scores)))

    # Operational gates: CSCF must flag social-engineering & deepfake-probe
    rows.append(
        Row(
            "cscf_flags_social_engineering",
            "risk",
            per_case_cscf["human_voice_scam_language"],
            "must be high even if voice looks human",
        )
    )
    rows.append(
        Row(
            "cscf_flags_deepfake_probe",
            "risk",
            per_case_cscf["synth_voice_benign_language"],
            "must be elevated even if language is mild",
        )
    )
    rows.append(
        Row(
            "cscf_safe_on_benign_human",
            "risk",
            per_case_cscf["human_voice_benign_language"],
            "must stay low",
        )
    )
    # Acoustic-only misses pure social engineering if voice is human-like
    ac_se = _fuse_call(cases[0][1], sr, cases[0][2], "acoustic_only")
    rows.append(
        Row(
            "acoustic_only_on_social_eng",
            "risk",
            ac_se,
            "often low  -  shows why fusion is needed",
        )
    )

    rows.append(Row("regime_tag_social_eng", "match", float(classify_regime(0.2, 0.85) == "social_engineering")))
    rows.append(Row("regime_tag_deepfake_probe", "match", float(classify_regime(0.85, 0.15) == "deepfake_probe")))
    rows.append(Row("regime_tag_dual", "match", float(classify_regime(0.85, 0.85) == "dual_threat")))
    return rows


def main() -> None:
    print("ShieldCall Core  -  Research Ablations")
    print("=" * 72)
    all_rows: List[Row] = []
    sections = [
        ("1. Linguistic / SDTG", ablation_linguistic),
        ("2. Channel acoustic", ablation_channel),
        ("3. Fusion CSCF vs baselines", ablation_fusion),
        ("4. Disagreement regimes (hard cases)", ablation_regimes),
        ("5. PMA coverage debt", ablation_adaptation),
    ]
    for title, fn in sections:
        print(f"\n{title}")
        print("-" * 72)
        t0 = time.perf_counter()
        rows = fn()
        dt = time.perf_counter() - t0
        for r in rows:
            print(f"  {r.name:40s}  {r.metric:12s}  {r.value:8.4f}  {r.notes}")
        print(f"  ({dt:.1f}s)")
        all_rows.extend(rows)

    print("\n" + "=" * 72)
    print("Falsification gates (must pass for claim language):")
    by_name_metric = {(r.name, r.metric): r.value for r in all_rows}

    def get(name: str, metric: str, default: float = 0.0) -> float:
        return by_name_metric.get((name, metric), default)

    cscf_auc = get("fusion_cscf", "AUC")
    naive_auc = get("fusion_naive_sum", "AUC")
    checks = [
        (
            "SDTG scam margin >= patterns-only margin (approx)",
            get("linguistic_margin_full", "scam-benign")
            >= get("linguistic_margin_patterns", "scam-benign") - 0.05,
        ),
        (
            "Scam path_score > benign path_score",
            get("sdtg_scam_path_score", "path_score") > get("sdtg_benign_path_score", "path_score"),
        ),
        (
            "CSCF AUC >= naive_sum AUC - 0.05",
            cscf_auc + 1e-6 >= naive_auc - 0.05,
        ),
        (
            "CSCF flags social engineering (human voice + scam script) risk>=0.55",
            get("cscf_flags_social_engineering", "risk") >= 0.55,
        ),
        (
            "CSCF flags deepfake probe (synth voice + mild language) risk>=0.40",
            get("cscf_flags_deepfake_probe", "risk") >= 0.40,
        ),
        (
            "CSCF stays low on benign human risk<=0.35",
            get("cscf_safe_on_benign_human", "risk") <= 0.35,
        ),
        (
            "Regime tags social_engineering / deepfake_probe / dual",
            get("regime_tag_social_eng", "match") == 1.0
            and get("regime_tag_deepfake_probe", "match") == 1.0
            and get("regime_tag_dual", "match") == 1.0,
        ),
        (
            "PMA gap reduces after shots",
            get("pma_gap_reduction", "delta") > 0,
        ),
        (
            "Acoustic AUC under narrowband > 0.5 (better than chance)",
            get("acoustic_narrowband", "AUC") > 0.5,
        ),
    ]
    failed = 0
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failed += 1
    print("=" * 72)
    if failed:
        print(f"{failed} gate(s) FAILED  -  do not freeze patent claim language on those claims.")
        raise SystemExit(1)
    print("All gates passed on current synthetic suite.")


if __name__ == "__main__":
    main()
