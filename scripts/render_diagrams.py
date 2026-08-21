#!/usr/bin/env python3
"""
Render high-quality architecture and process diagrams as PNG figures.

Output directory: docs/figures/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"

# Light, print-friendly palette (docs, slides, patent exhibits)
C = {
    "bg": "#ffffff",
    "panel": "#f7f9fc",
    "panel2": "#eef3f9",
    "border": "#8aa0b8",
    "accent": "#0b6bcb",
    "accent2": "#c6286a",
    "accent3": "#5b2d8e",
    "accent4": "#2f4fbf",
    "green": "#0f7a66",
    "amber": "#b36b00",
    "text": "#1a2332",
    "muted": "#5a6b7d",
    "white": "#ffffff",
    "soft": "#334155",
    "danger": "#c62828",
    "safe": "#1b7f5a",
    "quad_safe": "#e8f6f0",
    "quad_probe": "#e8f1fb",
    "quad_dual": "#fce8f1",
    "quad_se": "#fff4e5",
}


def _setup(w=14, h=9, facecolor=None):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    facecolor = facecolor or C["bg"]
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, text, fc, ec=None, fs=10, radius=0.02, bold=True):
    ec = ec or C["border"]
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius * 40}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.6,
        mutation_aspect=1,
        zorder=3,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=C["text"],
        fontweight=weight,
        zorder=4,
        linespacing=1.35,
    )
    return box


def _arrow(ax, x1, y1, x2, y2, color=None, lw=1.8, style="-|>", rad=0.0):
    color = color or C["accent"]
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arr)


def _title(ax, text, y=96):
    ax.text(
        50,
        y,
        text,
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=C["text"],
    )


def _subtitle(ax, text, y=91):
    ax.text(50, y, text, ha="center", va="top", fontsize=10, color=C["muted"])


def render_architecture_pipeline():
    fig, ax = _setup(14, 10)
    _title(ax, "ShieldCall: sensor stack and defense agent")
    _subtitle(
        ax,
        "Detectors emit sufficient statistics only. The agent never sees waveforms or transcripts.",
    )

    _box(
        ax,
        32,
        80,
        36,
        7,
        "Incoming audio stream\n(8 kHz / 16 kHz mono RTP or file)",
        C["panel2"],
        C["accent"],
        fs=10,
    )
    _box(
        ax,
        28,
        66,
        44,
        9,
        "Telephony Preprocessor + Channel Twin (TCT)\nResample | Bandlimit | VAD | Framing | optional codec/PLC",
        C["panel"],
        C["green"],
        fs=9.5,
    )
    _arrow(ax, 50, 80, 50, 75.2, C["green"])
    _arrow(ax, 40, 66, 22, 58, C["accent4"])
    _arrow(ax, 60, 66, 78, 58, C["accent2"])
    _box(
        ax,
        6,
        40,
        32,
        16,
        "Acoustic stream\n\nSTRF residual fingerprint\nPrototype Memory (PMA)\nSynthetic-voice score",
        C["panel"],
        C["accent4"],
        fs=9.5,
    )
    _box(
        ax,
        62,
        40,
        32,
        16,
        "Linguistic stream\n\nASR bridge fragments\nPattern groups + SDTG\nFraud-intent score",
        C["panel"],
        C["accent2"],
        fs=9.5,
    )
    _arrow(ax, 22, 40, 40, 32, C["accent4"])
    _arrow(ax, 78, 40, 60, 32, C["accent2"])
    _box(
        ax,
        26,
        18,
        48,
        13,
        "Score fusion (disagreement regimes)\nSAPC timing statistic  |  ACI uncertainty\nSufficient statistics only",
        C["panel"],
        C["amber"],
        fs=9.5,
    )
    _arrow(ax, 50, 18, 50, 12.5, C["amber"])
    _box(
        ax,
        22,
        4,
        56,
        8,
        "Belief-state agent: monitor | challenge | warn | escalate | adapt | abstain",
        C["panel2"],
        C["accent"],
        fs=10,
    )
    ax.text(8, 30, "Telephony-robust\nacoustic cues", fontsize=8, color=C["muted"], ha="left")
    ax.text(92, 30, "Script structure\n+ intent", fontsize=8, color=C["muted"], ha="right")
    fig.tight_layout(pad=0.6)
    path = OUT / "architecture_pipeline.png"
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def render_sdtg_stages():
    fig, ax = _setup(14, 6.5)
    _title(ax, "Scam Discourse Trajectory Graph (SDTG)")
    _subtitle(
        ax,
        "Streaming stage machine: fraud as progressive script structure, not bag-of-words",
    )
    stages = [
        ("GREETING", C["panel2"]),
        ("AUTHORITY", C["accent4"]),
        ("PROBLEM", C["accent"]),
        ("URGENCY", C["amber"]),
        ("HARVEST", C["accent3"]),
        ("PAYMENT", C["accent2"]),
        ("SECRECY", C["danger"]),
        ("THREAT", C["danger"]),
    ]
    n = len(stages)
    y = 48
    w, h = 10.2, 14
    gap = 1.8
    total = n * w + (n - 1) * gap
    x0 = (100 - total) / 2
    for i, (name, color) in enumerate(stages):
        x = x0 + i * (w + gap)
        _box(ax, x, y, w, h, name, C["panel"], color, fs=8.5)
        if i < n - 1:
            _arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2, color, lw=1.5)
    _box(ax, 40, 18, 20, 10, "BENIGN\n(non-scam path)", C["panel"], C["safe"], fs=9)
    ax.annotate(
        "",
        xy=(50, 32),
        xytext=(50, 42),
        arrowprops=dict(arrowstyle="<->", color=C["muted"], lw=1.4),
    )
    ax.text(52, 36, "escape / reset", fontsize=8, color=C["muted"])
    ax.text(
        50,
        8,
        "Path log-likelihood and progression depth feed linguistic fraud probability",
        ha="center",
        fontsize=10,
        color=C["soft"],
    )
    fig.tight_layout(pad=0.6)
    path = OUT / "sdtg_stages.png"
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def render_cscf_regimes():
    fig, ax = _setup(12, 10)
    _title(ax, "Fusion regimes (not causal inference)")
    _subtitle(ax, "Operational threat is a disjunction: flag if either stream is hostile")
    ax.add_patch(
        FancyBboxPatch(
            (12, 12),
            76,
            70,
            boxstyle="round,pad=0.02,rounding_size=2",
            facecolor=C["panel"],
            edgecolor=C["border"],
            lw=1.5,
            zorder=1,
        )
    )
    ax.add_patch(plt.Rectangle((12, 47), 38, 35, facecolor=C["quad_probe"], alpha=1.0, zorder=1))
    ax.add_patch(plt.Rectangle((50, 47), 38, 35, facecolor=C["quad_dual"], alpha=1.0, zorder=1))
    ax.add_patch(plt.Rectangle((12, 12), 38, 35, facecolor=C["quad_safe"], alpha=1.0, zorder=1))
    ax.add_patch(plt.Rectangle((50, 12), 38, 35, facecolor=C["quad_se"], alpha=1.0, zorder=1))
    ax.text(31, 72, "Deepfake probe", ha="center", fontsize=13, fontweight="bold", color=C["accent"], zorder=5)
    ax.text(31, 64, "High synth\nLow fraud language", ha="center", fontsize=9, color=C["soft"], zorder=5)
    ax.text(69, 72, "Dual threat", ha="center", fontsize=13, fontweight="bold", color=C["accent2"], zorder=5)
    ax.text(69, 64, "High synth\nHigh fraud language", ha="center", fontsize=9, color=C["soft"], zorder=5)
    ax.text(31, 30, "Agreement (safe)", ha="center", fontsize=13, fontweight="bold", color=C["safe"], zorder=5)
    ax.text(31, 22, "Low synth\nLow fraud language", ha="center", fontsize=9, color=C["soft"], zorder=5)
    ax.text(69, 30, "Social engineering", ha="center", fontsize=13, fontweight="bold", color=C["amber"], zorder=5)
    ax.text(69, 22, "Low synth (human voice)\nHigh fraud script", ha="center", fontsize=9, color=C["soft"], zorder=5)
    ax.text(50, 6, "Linguistic fraud probability (increasing right)", ha="center", fontsize=11, color=C["muted"])
    ax.text(
        5,
        47,
        "Acoustic synthetic\nprobability (up)",
        ha="center",
        va="center",
        fontsize=10,
        color=C["muted"],
        rotation=90,
    )
    ax.plot([50, 50], [12, 82], color=C["border"], lw=1.2, zorder=2)
    ax.plot([12, 88], [47, 47], color=C["border"], lw=1.2, zorder=2)
    ax.text(
        50,
        2,
        "A weighted sum can bury a vocoded probe under mild language, or a live vishing call under a human voice score",
        ha="center",
        fontsize=9,
        color=C["muted"],
    )
    fig.tight_layout(pad=0.6)
    path = OUT / "cscf_regimes.png"
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def render_adaptation_loop():
    fig, ax = _setup(13, 8.5)
    _title(ax, "Prototype Memory Adaptation and Coverage Debt")
    _subtitle(ax, "Closed loop: detect out-of-distribution synthesizers, enroll few-shot, measure recovery")
    nodes = [
        (50, 72, 28, 10, "Streaming acoustic\nembedding (STRF)", C["accent4"]),
        (18, 42, 28, 12, "Coverage-gap score\n(distance to both\nhuman and synth manifolds)", C["amber"]),
        (50, 42, 28, 12, "Prototype Memory\n(PMA)\nMahalanobis score", C["accent"]),
        (82, 42, 28, 12, "Challenge-response\nor human review\nlabel", C["accent2"]),
        (50, 14, 34, 10, "Debt index + recovery metric\n(gap before vs after k-shot)", C["green"]),
    ]
    for x, y, w, h, t, ec in nodes:
        _box(ax, x - w / 2, y - h / 2, w, h, t, C["panel"], ec, fs=9)
    _arrow(ax, 50, 67, 32, 49, C["amber"])
    _arrow(ax, 50, 67, 50, 48.5, C["accent"])
    _arrow(ax, 64, 42, 68, 42, C["accent2"])
    _arrow(ax, 82, 36, 60, 19, C["green"], rad=0.15)
    _arrow(ax, 32, 36, 40, 19, C["green"], rad=-0.1)
    _arrow(ax, 50, 19, 50, 28, C["muted"], style="<|-")
    ax.text(
        50,
        4,
        "Few-shot updates reduce coverage debt without waiting for a full retrain cycle",
        ha="center",
        fontsize=10,
        color=C["soft"],
    )
    fig.tight_layout(pad=0.6)
    path = OUT / "adaptation_loop.png"
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def render_patent_pathway():
    fig, ax = _setup(14, 8)
    _title(ax, "Patent Filing Pathway (U.S. and PCT)")
    _subtitle(ax, "Official portals: USPTO Patent Center and WIPO ePCT")
    steps = [
        ("1. Prepare\ndisclosure", "Spec, drawings,\ninventors, prior art"),
        ("2. File U.S.\nprovisional", "patentcenter.uspto.gov\nUtility Provisional"),
        ("3. Patent\nPending", "12-month window\nrun experiments"),
        ("4. Nonprovisional\n(+ optional PCT)", "Claims examined\nePCT if foreign"),
        ("5. Prosecution\nand grants", "Office actions\nnational phases"),
    ]
    y = 52
    w, h = 15, 22
    gap = 3.5
    total = len(steps) * w + (len(steps) - 1) * gap
    x0 = (100 - total) / 2
    colors = [C["accent4"], C["accent"], C["green"], C["amber"], C["accent2"]]
    for i, ((title, detail), col) in enumerate(zip(steps, colors)):
        x = x0 + i * (w + gap)
        _box(ax, x, y, w, h * 0.55, title, C["panel"], col, fs=10)
        _box(ax, x, y - 14, w, 12, detail, C["panel2"], C["border"], fs=8, bold=False)
        if i < len(steps) - 1:
            _arrow(ax, x + w, y + h * 0.28, x + w + gap, y + h * 0.28, col, lw=2)
    ax.text(
        50,
        18,
        "Primary filing entry: https://patentcenter.uspto.gov/\nInternational PCT: https://pct.wipo.int/ePCT/",
        ha="center",
        fontsize=11,
        color=C["soft"],
        linespacing=1.5,
    )
    ax.text(
        50,
        6,
        "This diagram is procedural guidance, not legal advice. Use a registered patent attorney for claims.",
        ha="center",
        fontsize=9,
        color=C["muted"],
    )
    fig.tight_layout(pad=0.6)
    path = OUT / "patent_pathway.png"
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def render_package_map():
    fig, ax = _setup(13, 9)
    _title(ax, "ShieldCall Core Package Map")
    _subtitle(ax, "Research-grade modules with stable public interfaces")
    packages = [
        (18, 68, "audio", "TCT channel twin\nVAD, preprocessor", C["green"]),
        (50, 68, "acoustic", "STRF features\nresidual, PMA scorer", C["accent4"]),
        (82, 68, "linguistic", "Patterns, SDTG\nASR bridge", C["accent2"]),
        (18, 38, "fusion", "Regimes, SAPC\nACI", C["amber"]),
        (50, 38, "agent", "Belief, planner\ntools, traces", C["accent3"]),
        (82, 38, "eval", "Protocols, handoff\nspeech, scripts", C["accent"]),
        (34, 12, "pipeline.py", "Sensor entrypoint", C["safe"]),
        (66, 12, "adaptation", "Challenge, PMA, debt", C["border"]),
    ]
    for x, y, name, detail, col in packages:
        label = f"shieldcall.{name}" if name not in ("pipeline.py", "config + demo") else name
        _box(ax, x - 12, y, 24, 8, label, C["panel"], col, fs=9)
        ax.text(x, y - 4, detail, ha="center", va="top", fontsize=8, color=C["muted"])
    _box(ax, 38, 48, 24, 7, "ShieldCallPipeline", C["panel2"], C["accent4"], fs=10)
    fig.tight_layout(pad=0.6)
    path = OUT / "package_map.png"
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def render_tct_strf():
    fig, ax = _setup(13, 8)
    _title(ax, "Telephony Channel Twin and STRF Residual Path")
    _subtitle(ax, "Channel is first-class: residual cues are evaluated under realistic phone distortion")
    stages = [
        (14, "Source\nwaveform"),
        (32, "Bandlimit\n300-3400 Hz"),
        (50, "G.711-like\nmu-law"),
        (68, "Noise / SNR\n+ packet loss"),
        (86, "PLC\nconcealment"),
    ]
    for i, (x, t) in enumerate(stages):
        _box(ax, x - 8, 62, 16, 14, t, C["panel"], C["green"] if i else C["accent"], fs=8.5)
        if i < len(stages) - 1:
            _arrow(ax, x + 8, 69, stages[i + 1][0] - 8, 69, C["green"])
    ax.text(50, 52, "Telephony Channel Twin (TCT)", ha="center", fontsize=11, color=C["green"], fontweight="bold")
    _arrow(ax, 86, 62, 50, 42, C["accent4"], rad=0.2)
    _box(
        ax,
        28,
        22,
        44,
        18,
        "STRF residual analysis\n\nHarmonic model  |  residual energy / flatness\nGrid artifact score  |  phase irregularity\n64-D streaming feature vector",
        C["panel"],
        C["accent4"],
        fs=9,
    )
    _arrow(ax, 50, 22, 50, 12, C["amber"])
    _box(ax, 30, 4, 40, 7, "Acoustic authenticity score (PMA + residual)", C["panel2"], C["amber"], fs=9.5)
    fig.tight_layout(pad=0.6)
    path = OUT / "tct_strf.png"
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def render_claim_code_map():
    fig, ax = _setup(13, 8.5)
    _title(ax, "Claim to Code to Experiment Map")
    _subtitle(ax, "Novelty is only asserted where implementation and measurement exist")
    rows = [
        ("TCT channel twin", "audio/channel.py", "channel ablations"),
        ("STRF residual FP", "acoustic/residual.py", "acoustic EER / AUC"),
        ("SDTG path score", "linguistic/discourse.py", "scam vs benign margin"),
        ("Fusion regimes", "fusion/engine.py", "operational OR-label table"),
        ("SAPC timing", "fusion/coupling.py", "synthetic yes; audio no"),
        ("Belief agent", "agent/*.py", "scripted action traces"),
    ]
    headers = ["Research claim", "Primary code", "Evidence"]
    xs = [10, 40, 72]
    widths = [26, 28, 22]
    y = 78
    for x, w, htxt in zip(xs, widths, headers):
        _box(ax, x, y, w, 7, htxt, C["panel2"], C["accent"], fs=10)
    for i, (claim, code, evid) in enumerate(rows):
        yy = 66 - i * 10
        vals = [claim, code, evid]
        ecs = [C["green"], C["accent4"], C["amber"]]
        for x, w, t, ec in zip(xs, widths, vals, ecs):
            _box(ax, x, yy, w, 8, t, C["panel"], ec, fs=8.5, bold=False)
    ax.text(
        50,
        4,
        "Run: pytest -q  |  python scripts/run_paper_experiments.py  |  python scripts/run_agent_demo.py",
        ha="center",
        fontsize=9,
        color=C["muted"],
    )
    fig.tight_layout(pad=0.6)
    path = OUT / "claim_code_map.png"
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def render_agent_loop():
    fig, ax = _setup(14, 9)
    _title(ax, "Belief-state call-defense agent")
    _subtitle(ax, "Not an LLM. Detectors are sensors. Actions are experiments with an interruption budget.")
    _box(ax, 6, 70, 28, 14, "Perception\n(synth, fraud, SAPC,\ngap, regime, ACI band)", C["panel"], C["accent4"], fs=9)
    _box(ax, 36, 70, 28, 14, "Belief over five hypotheses\nbenign | social eng.\nsynthetic | handoff | unknown", C["panel"], C["accent"], fs=9)
    _box(ax, 66, 70, 28, 14, "Planner\nIG − λ cost − delay risk\nmax 1 challenge / call", C["panel"], C["amber"], fs=9)
    _arrow(ax, 34, 77, 36, 77, C["accent"])
    _arrow(ax, 64, 77, 66, 77, C["amber"])
    actions = [
        (10, "monitor"),
        (26, "challenge"),
        (42, "warn"),
        (58, "escalate"),
        (74, "adapt / abstain"),
    ]
    for x, name in actions:
        _box(ax, x, 42, 14, 10, name, C["panel2"], C["green"], fs=8.5)
    _arrow(ax, 80, 70, 80, 53, C["green"])
    _box(
        ax,
        12,
        12,
        76,
        22,
        "Policy shape (heuristic likelihoods, not fitted Bayes)\n\n"
        "Human social engineer  →  warn/escalate (a live speaker will pass a nonce)\n"
        "Synthetic / unknown family  →  challenge is informative (TTS fails the acoustic gate)\n"
        "Handoff mass high  →  treat as dual-stream timing threat, not a keyword hit\n"
        "Benign mode  →  never interrupt",
        C["panel"],
        C["accent3"],
        fs=9,
        bold=False,
    )
    fig.tight_layout(pad=0.6)
    path = OUT / "agent_loop.png"
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def render_sapc_timing():
    fig, ax = _setup(14, 8)
    _title(ax, "Stage-aligned production change (SAPC)")
    _subtitle(
        ax,
        "Same vocoded fraction; only timing relative to a harvest stage differs. Utterance-mean scores cannot separate these.",
    )
    # two timelines
    ax.plot([8, 92], [62, 62], color=C["border"], lw=2)
    ax.plot([8, 92], [32, 32], color=C["border"], lw=2)
    ax.text(8, 70, "Aligned (hypothesis)", fontsize=11, fontweight="bold", color=C["accent"])
    ax.text(8, 40, "Unaligned (matched mix)", fontsize=11, fontweight="bold", color=C["muted"])
    # aligned vocode window
    ax.add_patch(plt.Rectangle((48, 56), 28, 12, facecolor=C["quad_dual"], ec=C["accent2"], lw=1.5, zorder=3))
    ax.text(62, 62, "vocoded", ha="center", va="center", fontsize=9, color=C["accent2"], zorder=4)
    ax.plot([50, 50], [52, 72], color=C["amber"], lw=2, ls="--")
    ax.text(50, 74, "harvest stage", ha="center", fontsize=8, color=C["amber"])
    # unaligned
    ax.add_patch(plt.Rectangle((10, 26), 28, 12, facecolor=C["quad_probe"], ec=C["accent"], lw=1.5, zorder=3))
    ax.text(24, 32, "vocoded", ha="center", va="center", fontsize=9, color=C["accent"], zorder=4)
    ax.plot([50, 50], [22, 42], color=C["amber"], lw=2, ls="--")
    ax.text(
        50,
        8,
        "Measured on Mini LibriSpeech splices: mix matched, ranking AUC 0.47. Claim not supported.\n"
        "On synthetic point processes the same statistic ranks at AUC 1.00.",
        ha="center",
        fontsize=9,
        color=C["soft"],
    )
    fig.tight_layout(pad=0.6)
    path = OUT / "sapc_timing.png"
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def render_graphical_abstract():
    fig, ax = _setup(14, 8)
    _title(ax, "Graphical abstract")
    _subtitle(ax, "ShieldCall: dual-stream telephony sensing plus a budgeted defense agent")
    _box(ax, 4, 48, 22, 28, "Call\n\naudio + ASR\nfragments", C["panel2"], C["border"], fs=10)
    _box(ax, 30, 55, 20, 14, "Acoustic\nresidual + PMA", C["panel"], C["accent4"], fs=9)
    _box(ax, 30, 36, 20, 14, "Linguistic\nkeywords + stages", C["panel"], C["accent2"], fs=9)
    _box(ax, 54, 42, 20, 24, "Fusion\nOR-label risk\nSAPC  |  ACI", C["panel"], C["amber"], fs=9)
    _box(ax, 78, 42, 18, 24, "Agent\nbelief\naction\ntrace", C["panel"], C["accent3"], fs=9)
    _arrow(ax, 26, 62, 30, 62, C["accent4"])
    _arrow(ax, 26, 43, 30, 43, C["accent2"])
    _arrow(ax, 50, 62, 54, 58, C["accent4"])
    _arrow(ax, 50, 43, 54, 50, C["accent2"])
    _arrow(ax, 74, 54, 78, 54, C["accent3"])
    ax.text(
        50,
        12,
        "What is measured: stage tracker beats keywords on paraphrases; disagreement fusion raises recall at a false-alarm cost;\n"
        "the agent warns on human vishing and does not waste a nonce; SAPC fails on vocoded LibriSpeech splices.",
        ha="center",
        fontsize=9,
        color=C["soft"],
    )
    fig.tight_layout(pad=0.6)
    path = OUT / "graphical_abstract.png"
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [
        render_graphical_abstract(),
        render_architecture_pipeline(),
        render_sdtg_stages(),
        render_cscf_regimes(),
        render_adaptation_loop(),
        render_patent_pathway(),
        render_package_map(),
        render_tct_strf(),
        render_claim_code_map(),
        render_agent_loop(),
        render_sapc_timing(),
    ]
    print("Rendered figures:")
    for p in paths:
        print(f"  {p}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
