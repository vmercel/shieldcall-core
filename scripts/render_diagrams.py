#!/usr/bin/env python3
"""Publication figures at journal print size.

~7.16 in wide (Elsevier two-column span). Type is true print points.
Nodes pair a line-icon with a short label. Line art is PDF + PNG @ 600 dpi.

The defense-agent figure is a closed perceive-act cycle (PEAS, sensors,
actuators, belief, utility, tools, budget, audit) — not a pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
PAPER = ROOT / "paper" / "figures"
ICON = OUT / "icons"

NAVY = "#1B365D"
INK = "#1A1A1A"
MUTED = "#3D4A5C"
LINE = "#2C5282"
FILL = "#F7FAFC"
FILL2 = "#EDF2F7"
SE = "#C05621"
SAFE = "#007A5E"
PROBE = "#0072B2"
DUAL = "#9B2C2C"
GOLD = "#B7791F"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

_ICON_ARR = {}


def _setup(w=7.16, h=5.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.subplots_adjust(left=0.025, right=0.975, top=0.97, bottom=0.03)
    return fig, ax


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    png = OUT / name
    pdf = OUT / name.replace(".png", ".pdf")
    fig.savefig(png, dpi=600, facecolor="white", pad_inches=0.12)
    fig.savefig(pdf, facecolor="white", pad_inches=0.12)
    for dest in (PAPER / name, PAPER / pdf.name):
        dest.write_bytes((OUT / dest.name).read_bytes())
    plt.close(fig)
    return png


def _box(ax, x, y, w, h, fc=FILL, ec=LINE, lw=1.15, r=0.7):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.04,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(p)
    return p


def _label(ax, x, y, text, size=9, weight="normal", color=INK, ha="center", va="center", z=7, **kw):
    ax.text(
        x, y, text,
        fontsize=size, fontweight=weight, color=color,
        ha=ha, va=va, zorder=z, linespacing=1.28, **kw,
    )


def _arrow(ax, x1, y1, x2, y2, color=LINE, lw=1.2, rad=0.0, style="-|>", ms=12):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=ms, linewidth=lw, color=color,
            connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=0.5, shrinkB=0.5,
        )
    )


def _title(ax, text, y=98.4):
    ax.text(50, y, text, ha="center", va="top", fontsize=11.5, fontweight="bold", color=INK, zorder=10)


def _disk(ax, x, y, r, fc, ec="white", lw=0.8, zorder=8):
    """Visually circular disk, correcting for non-square axes."""
    fig_w, fig_h = ax.figure.get_size_inches()
    height = 2 * r
    width = height * (fig_h / fig_w)
    ax.add_patch(Ellipse((x, y), width, height, fc=fc, ec=ec, lw=lw, zorder=zorder))


def _badge(ax, x, y, n, color=NAVY, r=2.0):
    _disk(ax, x, y, r, color, "white", 0.7, 8)
    _label(ax, x, y, str(n), 8, "bold", "white", z=9)


def _icon(ax, name: str, x: float, y: float, zoom: float = 0.046):
    if name not in _ICON_ARR:
        png = ICON / f"{name}.png"
        jpg = ICON / f"{name}.jpg"
        path = png if png.exists() else jpg
        if not path.exists():
            return
        _ICON_ARR[name] = np.asarray(Image.open(path).convert("RGBA"))
    im = OffsetImage(_ICON_ARR[name], zoom=zoom)
    ab = AnnotationBbox(im, (x, y), frameon=False, zorder=6, pad=0.0)
    ax.add_artist(ab)


def _node(ax, x, y, w, h, icon, title, sub="", ec=LINE, fc=FILL, ts=9.2, ss=7.6, zoom=0.046):
    _box(ax, x, y, w, h, fc, ec)
    cx = x + w / 2
    if sub:
        _icon(ax, icon, cx, y + h * 0.70, zoom)
        _label(ax, cx, y + h * 0.36, title, ts, "bold")
        _label(ax, cx, y + h * 0.15, sub, ss, color=MUTED)
    else:
        _icon(ax, icon, cx, y + h * 0.64, zoom)
        _label(ax, cx, y + h * 0.20, title, ts, "bold")


def _chip(ax, x, y, w, h, text, ec, fc="white"):
    _box(ax, x, y, w, h, fc, ec, lw=0.95, r=0.5)
    _label(ax, x + w / 2, y + h / 2, text, 7.8, color=ec)


# ---------------------------------------------------------------------------
# Agent: PEAS + closed perceive-act cycle
# ---------------------------------------------------------------------------
def render_agent_loop():
    fig, ax = _setup(7.16, 8.15)
    _title(ax, "Call-defense agent: perceive-act cycle", y=99.4)

    # PEAS (Russell & Norvig). One strip, four cells, no overlapping boxes.
    _box(ax, 1.6, 88.6, 96.8, 7.8, FILL2, NAVY, lw=1.05, r=0.5)
    peas = [
        (1.6, "P", "Performance", "1 challenge cap", GOLD),
        (25.8, "E", "Environment", "live 8 kHz call", NAVY),
        (50.0, "A", "Actuators", "six tools", SE),
        (74.2, "S", "Sensors", "scores only", PROBE),
    ]
    for i, (x, letter, name, sub, col) in enumerate(peas):
        if i:
            ax.plot([x, x], [89.2, 95.8], color="#C5D4E8", lw=0.9, zorder=4)
        _disk(ax, x + 3.6, 92.5, 1.35, col, col, 0.0, 8)
        _label(ax, x + 3.6, 92.5, letter, 8.0, "bold", "white", z=9)
        _label(ax, x + 14.6, 93.4, name, 8.2, "bold")
        _label(ax, x + 14.6, 90.6, sub, 7.8, color=MUTED)

    # Environment
    _box(ax, 35, 79.0, 30, 8.8, FILL2, NAVY, lw=1.35)
    _icon(ax, "phone", 41.2, 83.4, 0.038)
    _label(ax, 55.4, 84.2, "Environment", 9.5, "bold")
    _label(ax, 55.4, 81.2, "live telephone call", 8.0, color=MUTED)

    # Sensors
    _box(ax, 1.8, 46.5, 22.4, 26.0, FILL, PROBE, lw=1.2)
    _icon(ax, "ear", 13.0, 67.4, 0.048)
    _label(ax, 13.0, 59.4, "Sensors", 10.0, "bold")
    _label(ax, 13.0, 52.6, "synth, fraud,\nSAPC, coverage gap", 7.6, color=MUTED)

    # Actuators + tools
    _box(ax, 75.8, 46.5, 22.4, 26.0, FILL, SE, lw=1.2)
    _icon(ax, "shield", 87.0, 68.0, 0.040)
    _label(ax, 87.0, 61.6, "Actuators", 10.0, "bold")
    _label(ax, 87.0, 56.8, "monitor   warn", 8.0, color=SE)
    _label(ax, 87.0, 53.2, "challenge  escalate", 8.0, color=SE)
    _label(ax, 87.0, 49.6, "adapt   abstain", 8.0, color=SE)

    # Agent interior
    _box(ax, 26.6, 33.5, 46.8, 43.2, "#F8FBFE", NAVY, lw=1.45, r=1.0)
    _label(ax, 50.0, 74.6, "Agent interior", 9.2, "bold", NAVY)

    # Center belief
    _disk(ax, 50.0, 54.2, 6.4, "white", "#C5D4E8", 1.05, 4)
    _label(ax, 50.0, 56.0, "p(H)", 9.0, "bold", NAVY, z=7)
    _label(ax, 50.0, 52.2, "belief", 7.4, color=MUTED, z=7)

    # Four cycle stations (N E S W) — boxes do not sit on the arrows
    #   1 Perceive (north)  2 Update (east)  3 Decide (south)  4 Act (west)
    bw, bh = 13.8, 11.2
    stations = [
        (50.0, 67.0, "ear", "Perceive", PROBE, 1, 0.040),
        (64.6, 54.2, "brain", "Update", GOLD, 2, 0.038),
        (50.0, 41.4, "scale", "Decide", SAFE, 3, 0.038),
        (35.4, 54.2, "shield", "Act", SE, 4, 0.034),
    ]
    for px, py, ic, title, col, n, z in stations:
        _box(ax, px - bw / 2, py - bh / 2, bw, bh, "white", col, lw=1.25)
        _icon(ax, ic, px, py + 1.8, z)
        _label(ax, px, py - 3.35, title, 8.2, "bold")
        if n == 1:
            _badge(ax, px, py + bh / 2 + 0.2, n, col, 1.7)
        elif n == 2:
            _badge(ax, px + bw / 2 + 0.2, py, n, col, 1.7)
        elif n == 3:
            _badge(ax, px, py - bh / 2 - 0.2, n, col, 1.7)
        else:
            _badge(ax, px - bw / 2 - 0.2, py, n, col, 1.7)

    # Cycle arrows in the corner gaps (not through boxes)
    _arrow(ax, 57.0, 64.4, 57.8, 59.6, PROBE, lw=1.2, rad=-0.35)   # 1 -> 2
    _arrow(ax, 61.4, 48.8, 56.8, 46.8, GOLD, lw=1.2, rad=-0.35)    # 2 -> 3
    _arrow(ax, 43.0, 46.8, 38.6, 48.8, SAFE, lw=1.2, rad=-0.35)    # 3 -> 4
    _arrow(ax, 42.2, 59.6, 43.0, 64.4, SE, lw=1.2, rad=-0.35)      # 4 -> 1

    # Outer world loop
    _arrow(ax, 42.0, 79.0, 16.0, 72.6, PROBE, lw=1.25, rad=-0.08)
    _label(ax, 24.5, 77.6, "percepts", 7.6, color=PROBE)
    _arrow(ax, 24.2, 58.0, 26.6, 58.0, PROBE, lw=1.15)
    _arrow(ax, 73.4, 54.2, 75.8, 58.0, SE, lw=1.15)
    _arrow(ax, 87.0, 72.5, 64.0, 79.0, SE, lw=1.25, rad=-0.08)
    _label(ax, 79.8, 77.6, "actions", 7.6, color=SE)

    # Outcome path sits in the 1.5-unit band under the agent box
    _arrow(ax, 87.0, 46.5, 87.0, 31.6, MUTED, lw=1.05)
    _arrow(ax, 87.0, 31.6, 50.0, 31.6, MUTED, lw=1.05)
    _arrow(ax, 50.0, 31.6, 50.0, 33.5, MUTED, lw=1.05)
    _label(ax, 68.8, 33.4, "outcome feeds next percept", 7.3, color=MUTED)

    # Bottom row: memory, hypotheses, budget, human
    _node(ax, 1.8, 14.8, 18.4, 14.8, "memory", "Memory", "prototypes", GOLD, FILL2, 8.4, 7.2, 0.036)

    _box(ax, 21.6, 14.8, 34.0, 14.8, FILL2, NAVY, lw=1.05)
    _label(ax, 38.6, 26.8, "Hypotheses H  (mutually exclusive)", 8.0, "bold")
    hyps = [
        (22.4, "benign", SAFE),
        (28.9, "SE", SE),
        (35.4, "synth", PROBE),
        (41.9, "handoff", GOLD),
        (48.4, "unk.", MUTED),
    ]
    for hx, name, col in hyps:
        _chip(ax, hx, 16.6, 6.1, 6.6, name, col)

    _node(ax, 57.0, 14.8, 19.0, 14.8, "loop", "Budget", "max 1 nonce", GOLD, FILL2, 8.4, 7.2, 0.034)
    _node(ax, 77.4, 14.8, 20.8, 14.8, "operator", "Human", "on escalate", DUAL, FILL2, 8.4, 7.2, 0.034)

    _box(ax, 1.8, 2.2, 18.4, 10.6, FILL2, MUTED, lw=1.0)
    _icon(ax, "document", 6.6, 7.6, 0.030)
    _label(ax, 14.4, 8.8, "Audit", 8.4, "bold")
    _label(ax, 14.4, 5.4, "every Decision", 7.2, color=MUTED)

    _label(
        ax, 60.5, 9.8,
        "Decide:  argmax   IG(a)  -  lam_c cost(a)  -  lam_d 1[passive]",
        8.0, color=MUTED,
    )
    _label(
        ax, 60.5, 5.4,
        "Human social engineer  ->  warn (a nonce would pass).    Synthetic / unknown  ->  challenge.",
        8.0, color=MUTED,
    )
    return _save(fig, "agent_loop.png")


def render_graphical_abstract():
    fig, ax = _setup(7.16, 4.65)
    _title(ax, "Graphical abstract")

    _node(ax, 2.4, 46, 17.6, 38, "phone", "Call", "8 kHz audio", NAVY, FILL2, 10.0, 8.0, 0.048)
    _node(ax, 26.4, 64, 20.2, 22, "wave", "Acoustic", "", PROBE, FILL, 9.6, 7.6, 0.042)
    _node(ax, 26.4, 38, 20.2, 22, "document", "Linguistic", "", DUAL, FILL, 9.6, 7.6, 0.040)
    _node(ax, 53.0, 46, 20.2, 38, "scale", "Fusion", "OR-label  |  SAPC", GOLD, FILL, 10.0, 8.0, 0.046)
    _box(ax, 79.2, 46, 18.4, 38, FILL, SE, lw=1.25)
    _icon(ax, "brain", 88.4, 74.8, 0.042)
    _icon(ax, "loop", 88.4, 62.8, 0.038)
    _label(ax, 88.4, 52.4, "Agent\nloop", 10.0, "bold")

    _arrow(ax, 20.0, 72, 26.4, 75, PROBE)
    _arrow(ax, 20.0, 56, 26.4, 49, DUAL)
    _arrow(ax, 46.6, 75, 53.0, 70, PROBE)
    _arrow(ax, 46.6, 49, 53.0, 56, DUAL)
    _arrow(ax, 73.2, 65, 79.2, 65, SE)

    _arrow(ax, 88.4, 46.0, 88.4, 24.5, SE, lw=1.2)
    _arrow(ax, 88.4, 24.5, 11.2, 24.5, SE, lw=1.2)
    _arrow(ax, 11.2, 24.5, 11.2, 46.0, SE, lw=1.2)
    _label(ax, 50, 27.8, "actions close the loop on the live call", 8.2, color=SE)

    _label(
        ax, 50, 8.8,
        "Measured: stages beat keywords on paraphrases; fusion raises recall at an FPR cost;\n"
        "SAPC fails on vocoded LibriSpeech splices; the agent warns on human vishing.",
        8.0, color=MUTED,
    )
    return _save(fig, "graphical_abstract.png")


def render_architecture_pipeline():
    fig, ax = _setup(7.16, 6.2)
    _title(ax, "Sensor stack  (feeds the agent)")

    _box(ax, 24, 84.0, 52, 10.2, FILL2, NAVY, lw=1.25)
    _icon(ax, "phone", 32.2, 89.2, 0.038)
    _label(ax, 54.5, 89.2, "Incoming call  |  8 kHz", 10.0, "bold")

    _box(ax, 18, 70.2, 64, 9.4, FILL, SAFE, lw=1.1)
    _label(ax, 50, 74.9, "Preprocessor  |  telephone-band channel  |  VAD", 9.2, "bold")
    _arrow(ax, 50, 84.0, 50, 79.7, SAFE)

    _node(ax, 3.5, 40.5, 43, 24.5, "wave", "Acoustic stream",
          "residual  |  prototypes  |  CUSUM", PROBE, FILL, 10.0, 8.0, 0.046)
    _node(ax, 53.5, 40.5, 43, 24.5, "document", "Linguistic stream",
          "keywords  |  stage tracker\n(no production ASR here)", DUAL, FILL, 10.0, 7.8, 0.042)

    _arrow(ax, 36, 70.2, 25, 65.1, PROBE)
    _arrow(ax, 64, 70.2, 75, 65.1, DUAL)

    _box(ax, 16, 21.8, 68, 13.4, FILL, GOLD, lw=1.2)
    _icon(ax, "scale", 25.0, 28.6, 0.038)
    _label(ax, 56.5, 30.6, "Fusion  |  regimes  |  SAPC  |  ACI", 10.0, "bold")
    _label(ax, 56.5, 25.2, "sufficient statistics only   (no waveform, no transcript)", 8.0, color=MUTED)
    _arrow(ax, 25, 40.5, 38, 35.3, PROBE)
    _arrow(ax, 75, 40.5, 62, 35.3, DUAL)
    _arrow(ax, 50, 21.8, 50, 16.4, NAVY)

    _box(ax, 16, 3.0, 68, 12.4, FILL2, SE, lw=1.35)
    _icon(ax, "brain", 27.2, 9.3, 0.036)
    _icon(ax, "loop", 36.2, 9.3, 0.032)
    _label(ax, 58.8, 9.3, "Defense agent  (closed loop, Fig. 2)", 10.0, "bold")
    return _save(fig, "architecture_pipeline.png")


def render_cscf_regimes():
    fig, ax = _setup(7.16, 5.7)
    _title(ax, "Fusion regimes  (disjunction, not averaging)")

    quads = [
        (12, 51, "#EBF4FF", PROBE, "wave", "Spoof probe", "high synth  |  low fraud"),
        (54, 51, "#FCE8EC", DUAL, "shield", "Dual threat", "high synth  |  high fraud"),
        (12, 12, "#E6F4EA", SAFE, "phone", "Agreement", "low synth  |  low fraud"),
        (54, 12, "#FEF3E8", SE, "megaphone", "Social engineering", "human voice  |  hostile script"),
    ]
    for x, y, fc, ec, ic, title, sub in quads:
        ax.add_patch(Rectangle((x, y), 34, 33, facecolor=fc, edgecolor=ec, lw=1.25, zorder=2))
        _icon(ax, ic, x + 7.0, y + 25.0, 0.040)
        _label(ax, x + 21.2, y + 25.0, title, 10.2, "bold", ec)
        _label(ax, x + 17.0, y + 12.5, sub, 8.5, color=MUTED)

    ax.plot([50, 50], [12, 84], color="#CBD5E0", lw=1.05, zorder=3)
    ax.plot([12, 88], [45.5, 45.5], color="#CBD5E0", lw=1.05, zorder=3)

    _arrow(ax, 12, 5.8, 46, 5.8, MUTED, lw=1.05, ms=10)
    _label(ax, 48, 5.8, "Linguistic fraud", 8.2, color=MUTED, ha="left")
    _arrow(ax, 6.4, 12, 6.4, 82, MUTED, lw=1.05, ms=10)
    _label(ax, 6.4, 86.6, "Acoustic\nsynthetic", 7.6, color=MUTED)
    return _save(fig, "cscf_regimes.png")


def render_sdtg_stages():
    fig, ax = _setup(7.16, 3.7)
    _title(ax, "Scam discourse as a stage path")
    _icon(ax, "document", 8.0, 88.5, 0.036)

    stages = [
        ("Greeting", PROBE), ("Authority", PROBE), ("Problem", GOLD), ("Urgency", SE),
        ("Harvest", DUAL), ("Payment", DUAL), ("Secrecy", DUAL), ("Threat", DUAL),
    ]
    n = len(stages)
    bw, bh, gap = 9.2, 18.0, 2.6
    total = n * bw + (n - 1) * gap
    x0 = (100 - total) / 2
    y = 42
    for i, (name, col) in enumerate(stages):
        x = x0 + i * (bw + gap)
        _box(ax, x, y, bw, bh, FILL, col, lw=1.15)
        _disk(ax, x + bw / 2, y + bh * 0.68, 2.15, col, col, 0.0, 8)
        _label(ax, x + bw / 2, y + bh * 0.68, str(i + 1), 8.5, "bold", "white", z=9)
        _label(ax, x + bw / 2, y + bh * 0.28, name, 7.8, "bold")
        if i < n - 1:
            _arrow(ax, x + bw, y + bh / 2, x + bw + gap, y + bh / 2, col, lw=1.05, ms=9)

    _label(ax, 50, 16, "Path score uses progression, not a bag of keywords.", 8.6, color=MUTED)
    return _save(fig, "sdtg_stages.png")


def render_tct_strf():
    fig, ax = _setup(7.16, 4.6)
    _title(ax, "Telephone-band path and residual score")
    steps = [
        ("phone", "Source", NAVY),
        ("wave", "Bandlimit", SAFE),
        ("memory", "mu-law", SAFE),
        ("ear", "Loss + PLC", SAFE),
    ]
    for i, (ic, name, col) in enumerate(steps):
        x = 4.5 + i * 24.2
        _node(ax, x, 58, 21.4, 26.5, ic, name, "", col, FILL, 9.6, 7.6, 0.042)
        if i < 3:
            _arrow(ax, x + 21.4, 71.2, x + 24.2, 71.2, SAFE, lw=1.15)

    _box(ax, 16, 10, 68, 36, FILL, PROBE, lw=1.2)
    _icon(ax, "wave", 27.5, 28.2, 0.048)
    _label(ax, 56.5, 32.4, "Residual fingerprint + prototypes", 10.0, "bold")
    _label(ax, 56.5, 22.4, "Evaluated under the channel, not on clean-lab audio.", 8.2, color=MUTED)
    _arrow(ax, 88.5, 58.0, 72.0, 46.0, PROBE, rad=0.12)
    return _save(fig, "tct_strf.png")


def render_adaptation_loop():
    fig, ax = _setup(7.16, 4.75)
    _title(ax, "Few-shot memory and coverage gap")
    _node(ax, 5, 42, 24.5, 34, "wave", "Embedding", "", PROBE, FILL, 10.0, 7.6, 0.048)
    _node(ax, 37.75, 42, 24.5, 34, "brain", "Prototypes", "", GOLD, FILL, 10.0, 7.6, 0.046)
    _node(ax, 70.5, 42, 24.5, 34, "shield", "Challenge\nlabel", "", SE, FILL, 9.4, 7.6, 0.044)
    _arrow(ax, 29.5, 59, 37.75, 59, GOLD)
    _arrow(ax, 62.25, 59, 70.5, 59, SE)
    _box(ax, 16, 8, 68, 20.5, FILL2, SAFE, lw=1.15)
    _label(ax, 50, 22.0, "Coverage gap before vs after k labelled shots", 9.0, "bold", SAFE)
    _label(ax, 50, 13.8, "LPC recovery was small  (EER  0.50  ->  0.45)", 8.2, color=MUTED)
    _arrow(ax, 82.5, 42.0, 66.0, 28.6, SAFE, rad=0.18)
    return _save(fig, "adaptation_loop.png")


def render_package_map():
    fig, ax = _setup(7.16, 5.0)
    _title(ax, "Package map")
    pkgs = [
        (4.5, 54, "wave", "audio", PROBE),
        (28.0, 54, "wave", "acoustic", PROBE),
        (51.5, 54, "document", "linguistic", DUAL),
        (75.0, 54, "scale", "fusion", GOLD),
        (4.5, 12, "brain", "agent", SE),
        (28.0, 12, "shield", "adaptation", SE),
        (51.5, 12, "ear", "eval", SAFE),
        (75.0, 12, "loop", "pipeline", NAVY),
    ]
    for x, y, ic, name, col in pkgs:
        _node(ax, x, y, 20.5, 32, ic, name, "", col, FILL, 10.5, 7.6, 0.046)
    return _save(fig, "package_map.png")


def render_sapc_timing():
    fig, ax = _setup(7.16, 4.9)
    _title(ax, "SAPC: same mix, different timing")

    _label(ax, 6, 83.5, "Aligned", 10.0, "bold", PROBE, ha="left")
    ax.plot([6, 94], [70.5, 70.5], color="#CBD5E0", lw=2.2, solid_capstyle="round")
    ax.add_patch(Rectangle((48, 63.5), 32, 14, facecolor="#FCE8EC", ec=DUAL, lw=1.25, zorder=3))
    _icon(ax, "wave", 54.2, 70.5, 0.028)
    _label(ax, 70.5, 70.5, "vocoded window", 8.6, color=DUAL)
    ax.plot([52, 52], [58, 85], color=GOLD, lw=1.45, ls="--")
    _label(ax, 52, 87.0, "harvest", 8.2, color=GOLD)

    _label(ax, 6, 49.0, "Unaligned  (matched duration)", 10.0, "bold", MUTED, ha="left")
    ax.plot([6, 94], [34.5, 34.5], color="#CBD5E0", lw=2.2, solid_capstyle="round")
    ax.add_patch(Rectangle((8, 27.5), 32, 14, facecolor="#EBF4FF", ec=PROBE, lw=1.25, zorder=3))
    _icon(ax, "wave", 14.2, 34.5, 0.028)
    _label(ax, 30.5, 34.5, "vocoded window", 8.6, color=PROBE)
    ax.plot([52, 52], [24, 51], color=GOLD, lw=1.45, ls="--")

    _label(
        ax, 50, 11.0,
        "Utterance-mean scores cannot separate these two.\n"
        "On Mini LibriSpeech splices the ranking AUC was 0.47  -  claim not supported.",
        8.4, color=MUTED,
    )
    return _save(fig, "sapc_timing.png")


def render_claim_code_map():
    fig, ax = _setup(7.16, 5.2)
    _title(ax, "Claim  |  code  |  evidence")
    rows = [
        ("Stage tracker vs keywords", "linguistic/", "held-out AUC 0.88 vs 0.42"),
        ("OR-label fusion", "fusion/engine.py", "recall 1.00 vs 0.30; FPR 0.40"),
        ("Pulse-formant / LPC", "acoustic/ + eval/", "EER 0  /  AUC 0.49"),
        ("SAPC timing", "fusion/coupling.py", "synthetic 1.00; audio 0.47"),
        ("Defense agent", "agent/", "scripted traces, not EER"),
    ]
    for x, t in ((4, "Claim"), (36, "Code"), (68, "Evidence")):
        _box(ax, x, 81.5, 28, 8.6, FILL2, NAVY)
        _label(ax, x + 14, 85.8, t, 9.6, "bold")
    for i, (a, b, c) in enumerate(rows):
        y = 66.5 - i * 13.4
        _box(ax, 4, y, 28, 11.2, FILL, PROBE)
        _box(ax, 36, y, 28, 11.2, FILL, GOLD)
        _box(ax, 68, y, 28, 11.2, FILL, SAFE)
        _label(ax, 18, y + 5.6, a, 8.3)
        _label(ax, 50, y + 5.6, b, 8.3)
        _label(ax, 82, y + 5.6, c, 8.3)
    return _save(fig, "claim_code_map.png")


def render_patent_pathway():
    fig, ax = _setup(7.16, 3.85)
    _title(ax, "Filing path  (procedural, not legal advice)")
    steps = ["Disclosure", "U.S.\nprovisional", "Pending", "Non-\nprovisional", "Prosecution"]
    for i, name in enumerate(steps):
        x = 3.2 + i * 19.4
        col = PROBE if i < 2 else GOLD
        _box(ax, x, 28, 16.8, 44, FILL, col, lw=1.15)
        _disk(ax, x + 8.4, 64.2, 2.7, col, "white", 0.6, 8)
        _label(ax, x + 8.4, 64.2, str(i + 1), 10, "bold", "white", z=9)
        _label(ax, x + 8.4, 44.0, name, 8.6, "bold")
        if i < 4:
            _arrow(ax, x + 16.8, 50, x + 19.4, 50, MUTED, lw=1.1, ms=10)
    _label(ax, 50, 12.5, "patentcenter.uspto.gov      pct.wipo.int/ePCT", 8.2, color=MUTED)
    return _save(fig, "patent_pathway.png")


def render_linguistic_auc():
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=150)
    fig.patch.set_facecolor("white")
    labels = ["Keywords\n(train)", "Keywords+stages\n(train)", "Keywords\n(held-out)", "Keywords+stages\n(held-out)"]
    vals = [0.929, 1.000, 0.417, 0.883]
    colors = ["#94A3B8", PROBE, "#94A3B8", PROBE]
    bars = ax.bar(labels, vals, color=colors, width=0.62, edgecolor=INK, linewidth=0.4, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.axhline(0.5, color="#94A3B8", ls="--", lw=0.9, zorder=2)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("AUC", fontsize=10.5)
    ax.set_title("Linguistic detection on author-written scripts", fontsize=11, pad=8)
    ax.tick_params(labelsize=8.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, ls=":", color="#E2E8F0", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    name = "linguistic_auc.png"
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=600, facecolor="white")
    fig.savefig(OUT / "linguistic_auc.pdf", facecolor="white")
    for dest in (PAPER / name, PAPER / "linguistic_auc.pdf"):
        dest.write_bytes((OUT / dest.name).read_bytes())
    plt.close(fig)
    return OUT / name


def render_fusion_operational():
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=150)
    fig.patch.set_facecolor("white")
    methods = ["Acoustic", "Linguistic", "Naive sum", "Logistic", "CSCF"]
    auc = [0.833, 0.827, 0.973, 0.993, 1.000]
    rec = [0.90, 0.05, 0.30, 1.00, 1.00]
    fpr = [0.80, 0.00, 0.00, 0.50, 0.40]
    x = np.arange(len(methods))
    w = 0.26
    ax.bar(x - w, auc, w, label="AUC", color=PROBE, edgecolor=INK, lw=0.35, zorder=3)
    ax.bar(x, rec, w, label="Disagreement recall @ 0.5", color="#E69F00", edgecolor=INK, lw=0.35, zorder=3)
    ax.bar(x + w, fpr, w, label="Safe-cell FPR @ 0.5", color="#CC79A7", edgecolor=INK, lw=0.35, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9.2)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Score", fontsize=10.5)
    ax.set_title("Operational fusion  (threat = scam language or vocoded voice)", fontsize=10.5, pad=8)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, ls=":", color="#E2E8F0", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.8)
    fig.tight_layout()
    name = "fusion_operational.png"
    fig.savefig(OUT / name, dpi=600, facecolor="white")
    fig.savefig(OUT / "fusion_operational.pdf", facecolor="white")
    for dest in (PAPER / name, PAPER / "fusion_operational.pdf"):
        dest.write_bytes((OUT / dest.name).read_bytes())
    plt.close(fig)
    return OUT / name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (ICON / "phone.png").exists() and not (ICON / "phone.jpg").exists():
        raise SystemExit(f"Missing icons in {ICON}")
    paths = [
        render_graphical_abstract(),
        render_architecture_pipeline(),
        render_agent_loop(),
        render_cscf_regimes(),
        render_sdtg_stages(),
        render_tct_strf(),
        render_adaptation_loop(),
        render_package_map(),
        render_sapc_timing(),
        render_claim_code_map(),
        render_patent_pathway(),
        render_linguistic_auc(),
        render_fusion_operational(),
    ]
    print("Rendered at journal width (vector PDF + 600 dpi PNG):")
    for p in paths:
        pdf = p.with_suffix(".pdf")
        print(f"  {p.name:28s}  png {p.stat().st_size // 1024:4d} KB   pdf {pdf.stat().st_size // 1024:3d} KB")


if __name__ == "__main__":
    main()
