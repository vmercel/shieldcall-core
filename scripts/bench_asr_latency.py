#!/usr/bin/env python3
"""
P2-1 latency bench: cost of the streaming-ASR hook on the telephone frame path.

The sidecar calls push_audio once per 25 ms frame (~100 fps). This bench
measures, per frame:

  baseline        PassthroughASR (what the sidecar used before)
  gated-null      GatedASR(PassthroughASR) -> circuit-breaker overhead
  gated-stream    GatedASR(DeepgramStreamingASR + LoopbackTransport)
                  at simulated provider RTTs -> proves provider RTT stays
                  OFF the frame path (background thread + queues)

plus end-to-end transcript delay (audio chunk in -> fragment out) at each
simulated RTT, which is where provider latency honestly shows up.

No network, no API key: LoopbackTransport simulates the provider.

Regression guard: mean frame-path overhead of gated-stream vs baseline must
stay under 1 ms, otherwise exit 1.
"""

from __future__ import annotations

import statistics
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from shieldcall.linguistic.asr_bridge import PassthroughASR
from shieldcall.linguistic.deepgram_stream import DeepgramStreamingASR, LoopbackTransport
from shieldcall.runtime.asr_gate import GatedASR
from shieldcall.runtime.breaker import CircuitBreaker

SR = 8000
FRAME = (np.random.RandomState(7).rand(200).astype(np.float32) * 2 - 1) * 0.1
N = 400
WARMUP = 50


def bench_push(asr, n=N) -> tuple[float, float]:
    for i in range(WARMUP):
        asr.push_audio(FRAME, SR, i * 0.025)
    dts = []
    for i in range(n):
        t0 = time.perf_counter()
        asr.push_audio(FRAME, SR, (WARMUP + i) * 0.025)
        dts.append((time.perf_counter() - t0) * 1000.0)
    dts.sort()
    mean = statistics.fmean(dts)
    p95 = dts[int(0.95 * len(dts))]
    return mean, p95


def e2e_delay(rtt_ms: float, reps: int = 6) -> float:
    """Fresh adapter per rep, exactly one audio chunk: no cross-talk."""
    ds = []
    for _ in range(reps):
        asr = DeepgramStreamingASR(
            transport_factory=lambda: LoopbackTransport(latency_ms=rtt_ms)
        )
        try:
            t0 = time.monotonic()
            asr.push_audio(FRAME, SR, 0.0)
            frags: list = []
            while time.monotonic() - t0 < 10.0 and not frags:
                time.sleep(0.005)
                # Empty push: drains fragments without queueing new audio.
                frags = asr.push_audio(np.zeros(0, dtype=np.float32), SR, 0.0)
            assert frags, "no fragment arrived within 10 s"
            ds.append((time.monotonic() - t0) * 1000.0)
        finally:
            asr.close()
    return statistics.fmean(ds)


def main() -> int:
    rows = []
    base = PassthroughASR()
    m, p = bench_push(base)
    rows.append(("baseline PassthroughASR", m, p, None))

    gated_null = GatedASR(PassthroughASR(), CircuitBreaker())
    m, p = bench_push(gated_null)
    rows.append(("gated-null (breaker overhead)", m, p, None))

    for rtt in (0, 50, 150, 300):
        asr = DeepgramStreamingASR(
            transport_factory=lambda r=rtt: LoopbackTransport(latency_ms=r)
        )
        gated = GatedASR(asr, CircuitBreaker())
        m, p = bench_push(gated)
        rows.append((f"gated-stream RTT={rtt}ms", m, p, None))
        asr.close()

    e2e = {}
    for rtt in (0, 50, 150, 300):
        e2e[rtt] = e2e_delay(rtt)

    print(f"{'config':<34}{'mean ms/frame':>14}{'p95 ms/frame':>14}")
    for name, mean, p95, _ in rows:
        print(f"{name:<34}{mean:>14.4f}{p95:>14.4f}")
    print("\nend-to-end transcript delay (audio in -> fragment out):")
    for rtt, ms in e2e.items():
        print(f"  simulated RTT {rtt:>3} ms -> {ms:7.1f} ms")

    base_mean = rows[0][1]
    worst_stream = max(r[1] for r in rows if r[0].startswith("gated-stream"))
    overhead = worst_stream - base_mean
    print(f"\nworst frame-path overhead vs baseline: {overhead:.4f} ms")
    if overhead >= 1.0:
        print("REGRESSION: streaming hook added >= 1 ms to the frame path")
        return 1
    print("OK: provider RTT stays off the frame path (< 1 ms overhead)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
