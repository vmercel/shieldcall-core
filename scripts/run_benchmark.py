#!/usr/bin/env python3
"""Run the synthetic telephony benchmark and print a summary table."""

from __future__ import annotations

from shieldcall.eval.harness import run_full_benchmark


def main() -> None:
    print("ShieldCall Core — full synthetic benchmark")
    print("=" * 72)
    results = run_full_benchmark(seed=0)
    for name, r in results.items():
        extra = ""
        if r.extras:
            extra = " | " + ", ".join(f"{k}={v:.3f}" for k, v in r.extras.items())
        print(
            f"{name:28s}  n={r.n_samples:3d}  EER={r.eer_estimate:.3f}  "
            f"AUC={r.auc:.3f}  lat={r.mean_latency_ms:.2f}ms  {r.notes}{extra}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
