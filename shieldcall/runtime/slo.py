"""Service objectives for a ShieldCall sidecar worker.

These are design targets, not measured production SLOs. Frame time
is compared against the 10 ms hop so a worker can stay real-time.
Call availability is *not* ShieldCall's uptime: the media path
bypasses the detector (fail-open).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SLO:
    hop_ms: float = 10.0
    frame_budget_ms: float = 8.0          # p95 process time vs 10 ms hop
    realtime_min_x: float = 1.25          # audio_seconds / wall_seconds
    max_interrupt_per_call: int = 1       # challenges
    detector_ready_min_free: float = 0.10 # refuse new calls below 10% headroom
    asr_fail_threshold: int = 5
    asr_reset_sec: float = 30.0
    # Call path (PSTN) target if the operator already has five-nines.
    # Detector path is best-effort; shedding is success, not outage.
    call_availability: str = "owned by SBC; detector must not reduce it"
    detector_availability: str = "best-effort; shed rather than block media"

    def within_frame_budget(self, ms_per_frame: float) -> bool:
        return ms_per_frame <= self.frame_budget_ms
