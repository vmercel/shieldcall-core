#!/usr/bin/env python3
"""Walk three scripted calls through the defense agent and print traces."""

from __future__ import annotations

from shieldcall.agent import DefenseAgent, Perception


def play(name: str, frames: list[Perception]) -> None:
    print(f"\n=== {name} ===")
    ag = DefenseAgent()
    for p in frames:
        d = ag.step(p)
        mode = max(d.belief, key=d.belief.get)
        print(
            f"  t={p.timestamp_sec:4.1f}s  {d.action.value:10s}  "
            f"mode={mode:20s} {d.belief[mode]:.2f}  {d.plan.rationale}"
        )


def main() -> None:
    play(
        "benign dentist reminder",
        [Perception(t, synth=0.12, fraud=0.04, risk=0.08) for t in (0.5, 1.5, 3.0, 5.0)],
    )
    play(
        "human IRS vishing (social engineering)",
        [
            Perception(0.5, synth=0.14, fraud=0.15, risk=0.18),
            Perception(2.0, synth=0.16, fraud=0.62, risk=0.58, regime="social_engineering"),
            Perception(4.0, synth=0.15, fraud=0.91, risk=0.86, regime="social_engineering"),
            Perception(6.0, synth=0.17, fraud=0.93, risk=0.90, regime="social_engineering"),
        ],
    )
    play(
        "handoff: human opener then vocoded harvest",
        [
            Perception(1.0, synth=0.18, fraud=0.22, handoff_score=0.05, risk=0.20),
            Perception(4.0, synth=0.35, fraud=0.55, handoff_score=0.4, handoff_pvalue=0.2, risk=0.50),
            Perception(6.5, synth=0.62, fraud=0.80, handoff_score=0.88, handoff_pvalue=0.04, risk=0.84),
            Perception(8.0, synth=0.65, fraud=0.85, handoff_score=0.90, handoff_pvalue=0.03, risk=0.88),
        ],
    )


if __name__ == "__main__":
    main()
