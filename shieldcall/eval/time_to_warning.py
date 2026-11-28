"""Time-to-warning measurement (TDD v2.0 section 10.4; ADR-019).

The budget is at most 2 s at p95, measured from the
moment discriminative evidence ends (the last frame of
a cloned utterance, or the end of a harmful request)
to the moment the warning is rendered. Figure 21
budgets the critical path at 1.2 s: ingest, decode
and jitter at most 100 ms, acoustic evidence
accumulation at most 700 ms, Tier 1 inference at
most 200 ms, stable-prefix ASR and discourse update
at most 700 ms in parallel, and fusion, agent, signed
event and render at most 200 ms.

This module measures only the in-process part. A
laboratory harness feeds audio faster than real time,
so two quantities are reported separately and must
not be conflated:

* accumulation, in audio seconds: the frame timestamp
  of the first HIGH_RISK event at or after the
  evidence end, minus the evidence end time. This is
  how much post-evidence audio the pipeline consumed
  before warning. In a live call that audio arrives
  at 1x speed, so accumulation seconds are also wall
  seconds of waiting.
* processing, in wall seconds: how long the deciding
  push took, from ingest of its chunk to the verdict
  being returned.

ASR lag, render time, and network time are
caller-supplied and default to absent. They are
never treated as zero: the estimated live warning
time is labelled measured-only while any stage is
absent, and the full 2 s budget verdict is then
incomplete. The warning itself is the first CSCF
HIGH_RISK event; the provisional cue is the first
SUSPICIOUS or HIGH_RISK event, per the TDD
requirement to show a cue early and upgrade it.

Nothing here changes a live decision path. The
harness drives a pipeline through its public
``push_audio`` / ``push_transcript`` calls and reads
verdicts off the returned events. No model is loaded
and no network is touched.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# Figure 21 component budgets, in seconds (TDD section 10.4).
BUDGET_ACCUMULATION_SEC = 0.7
BUDGET_TIER1_SEC = 0.2
BUDGET_DECIDING_CHUNK_SEC = 0.2
BUDGET_WARNING_P95_SEC = 2.0

# A clock returns wall seconds. The default is the
# monotonic clock; a fake clock makes the arithmetic
# deterministic in tests.
Clock = Callable[[], float]


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank percentile, stated so runs are comparable.

    The p95 of a handful of runs is a coarse statistic;
    nearest-rank does not pretend otherwise by
    interpolating between two measured values.
    """
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    rank = math.ceil(q * len(ordered))
    return ordered[max(0, min(len(ordered) - 1, rank - 1))]


@dataclass(frozen=True)
class Turn:
    """One transcript turn, at its time in audio seconds."""

    time_sec: float
    text: str


@dataclass
class WarningMeasurement:
    """One measured run over a synthetic call.

    ``warned`` means a HIGH_RISK event was returned at
    or after the chunk containing ``evidence_end_sec``.
    A warning that only occurred before the evidence
    end sets ``warned_before_evidence_end`` and does
    not count: the evidence that sufficed ended
    earlier, so the declared evidence end was
    mis-specified. Accumulation is never negative.
    """

    evidence_end_sec: float
    audio_duration_sec: float
    chunk_ms: float
    n_chunks: int
    warned: bool
    warned_before_evidence_end: bool
    first_high_risk_audio_sec: Optional[float]
    accumulation_audio_sec: Optional[float]
    provisional_cue: bool
    first_provisional_audio_sec: Optional[float]
    provisional_accumulation_audio_sec: Optional[float]
    deciding_chunk_index: Optional[int]
    deciding_chunk_processing_sec: Optional[float]
    chunk_processing_sec: List[float] = field(default_factory=list)
    turn_processing_sec: List[float] = field(default_factory=list)
    turn_processing_before_evidence_sec: float = 0.0
    total_wall_sec: float = 0.0
    tier1_latency_sec: List[float] = field(default_factory=list)
    tier1_scored_windows: int = 0
    shadow_windows_total: int = 0
    shadow_drained: bool = True
    asr_lag_sec: Optional[float] = None
    render_sec: Optional[float] = None
    network_sec: Optional[float] = None

    @property
    def chunk_processing_p50_sec(self) -> Optional[float]:
        return _percentile(self.chunk_processing_sec, 0.50)

    @property
    def chunk_processing_p95_sec(self) -> Optional[float]:
        return _percentile(self.chunk_processing_sec, 0.95)

    @property
    def chunk_processing_max_sec(self) -> Optional[float]:
        return _percentile(self.chunk_processing_sec, 1.0)

    @property
    def steady_chunk_processing_p95_sec(self) -> Optional[float]:
        """p95 over all chunks except the first (warm-up)."""
        return _percentile(self.chunk_processing_sec[1:], 0.95)

    @property
    def tier1_latency_p95_sec(self) -> Optional[float]:
        return _percentile(self.tier1_latency_sec, 0.95)

    @property
    def measured_only(self) -> bool:
        """True while any out-of-process stage is unmeasured."""
        return (
            self.asr_lag_sec is None
            or self.render_sec is None
            or self.network_sec is None
        )

    @property
    def estimated_live_warning_sec(self) -> Optional[float]:
        """Accumulation plus processing plus supplied stages.

        None when the run never warned. When
        ``measured_only`` is true this is the measured
        part only, not the full end-to-end figure.
        """
        if not self.warned:
            return None
        total = (
            (self.accumulation_audio_sec or 0.0)
            + (self.deciding_chunk_processing_sec or 0.0)
            + self.turn_processing_before_evidence_sec
            + (self.asr_lag_sec or 0.0)
            + (self.render_sec or 0.0)
            + (self.network_sec or 0.0)
        )
        return float(total)

    @property
    def accumulation_within_budget(self) -> Optional[bool]:
        if self.accumulation_audio_sec is None:
            return None
        return self.accumulation_audio_sec <= BUDGET_ACCUMULATION_SEC

    @property
    def deciding_chunk_within_budget(self) -> Optional[bool]:
        if self.deciding_chunk_processing_sec is None:
            return None
        return self.deciding_chunk_processing_sec <= BUDGET_DECIDING_CHUNK_SEC

    @property
    def tier1_within_budget(self) -> Optional[bool]:
        p95 = self.tier1_latency_p95_sec
        if p95 is None:
            return None
        return p95 <= BUDGET_TIER1_SEC

    def to_dict(self) -> dict:
        return {
            "evidence_end_sec": self.evidence_end_sec,
            "audio_duration_sec": self.audio_duration_sec,
            "chunk_ms": self.chunk_ms,
            "n_chunks": self.n_chunks,
            "warned": self.warned,
            "warned_before_evidence_end": self.warned_before_evidence_end,
            "first_high_risk_audio_sec": self.first_high_risk_audio_sec,
            "accumulation_audio_sec": self.accumulation_audio_sec,
            "accumulation_within_budget": self.accumulation_within_budget,
            "provisional_cue": self.provisional_cue,
            "first_provisional_audio_sec": self.first_provisional_audio_sec,
            "provisional_accumulation_audio_sec": self.provisional_accumulation_audio_sec,
            "deciding_chunk_index": self.deciding_chunk_index,
            "deciding_chunk_processing_sec": self.deciding_chunk_processing_sec,
            "deciding_chunk_within_budget": self.deciding_chunk_within_budget,
            "chunk_processing_sec": list(self.chunk_processing_sec),
            "chunk_processing_p50_sec": self.chunk_processing_p50_sec,
            "chunk_processing_p95_sec": self.chunk_processing_p95_sec,
            "chunk_processing_max_sec": self.chunk_processing_max_sec,
            "steady_chunk_processing_p95_sec": self.steady_chunk_processing_p95_sec,
            "turn_processing_sec": list(self.turn_processing_sec),
            "turn_processing_before_evidence_sec": self.turn_processing_before_evidence_sec,
            "total_wall_sec": self.total_wall_sec,
            "tier1_latency_sec": list(self.tier1_latency_sec),
            "tier1_latency_p95_sec": self.tier1_latency_p95_sec,
            "tier1_within_budget": self.tier1_within_budget,
            "tier1_scored_windows": self.tier1_scored_windows,
            "shadow_windows_total": self.shadow_windows_total,
            "shadow_drained": self.shadow_drained,
            "asr_lag_sec": self.asr_lag_sec,
            "render_sec": self.render_sec,
            "network_sec": self.network_sec,
            "measured_only": self.measured_only,
            "estimated_live_warning_sec": self.estimated_live_warning_sec,
        }


@dataclass
class RunSummary:
    """p50/p95 over repeated runs of the same call.

    Percentiles are over warned runs only. A single
    run never yields a p95 claim: use at least two
    runs, and treat a handful of runs as coarse.
    """

    n_runs: int
    warned_runs: int
    accumulation_p50_sec: Optional[float]
    accumulation_p95_sec: Optional[float]
    deciding_chunk_p95_sec: Optional[float]
    tier1_latency_p95_sec: Optional[float]
    estimated_live_p50_sec: Optional[float]
    estimated_live_p95_sec: Optional[float]
    budget_verdict: str  # "met" | "not met" | "incomplete"
    budget_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_runs": self.n_runs,
            "warned_runs": self.warned_runs,
            "accumulation_p50_sec": self.accumulation_p50_sec,
            "accumulation_p95_sec": self.accumulation_p95_sec,
            "deciding_chunk_p95_sec": self.deciding_chunk_p95_sec,
            "tier1_latency_p95_sec": self.tier1_latency_p95_sec,
            "estimated_live_p50_sec": self.estimated_live_p50_sec,
            "estimated_live_p95_sec": self.estimated_live_p95_sec,
            "budget_verdict": self.budget_verdict,
            "budget_notes": list(self.budget_notes),
        }


def measure_time_to_warning(
    pipeline,
    audio: np.ndarray,
    sample_rate: int,
    *,
    evidence_end_sec: float,
    turns: Sequence[Turn] = (),
    chunk_ms: float = 100.0,
    clock: Optional[Clock] = None,
    drain_timeout_sec: float = 5.0,
    asr_lag_sec: Optional[float] = None,
    render_sec: Optional[float] = None,
    network_sec: Optional[float] = None,
) -> WarningMeasurement:
    """Measure one synthetic call end to end, in process.

    The pipeline is reset first, so a factory-built
    pipeline and a reused one behave the same; a
    scorer installed on the shadow observer survives
    reset, as ADR-018 specifies. Audio is fed in
    ``chunk_ms`` chunks. Turns are delivered through
    ``push_transcript`` before the chunk whose audio
    interval contains the turn time. The evidence end
    time must fall inside the audio. Turns may also
    arrive through the pipeline's own ASR; in that
    case pass no turns and their discourse update
    time sits inside the chunk processing times.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise ValueError("audio must not be empty")
    duration_sec = samples.size / float(sample_rate)
    if evidence_end_sec < 0.0 or evidence_end_sec > duration_sec:
        raise ValueError("evidence_end_sec must fall inside the audio")
    for stage, value in (
        ("asr_lag_sec", asr_lag_sec),
        ("render_sec", render_sec),
        ("network_sec", network_sec),
    ):
        if value is not None and value < 0.0:
            raise ValueError(f"{stage} must not be negative")

    tick = clock or time.monotonic
    chunk_len = max(1, int(sample_rate * float(chunk_ms) / 1000.0))
    ordered_turns = sorted(turns, key=lambda t: t.time_sec)
    turn_idx = 0

    pipeline.reset()
    chunk_processing: List[float] = []
    turn_processing: List[float] = []
    turn_before_evidence = 0.0

    first_warning: Optional[Tuple[int, float]] = None
    first_warning_any: Optional[Tuple[int, float]] = None
    first_provisional: Optional[Tuple[int, float]] = None
    evidence_chunk_index: Optional[int] = None

    feed_start = tick()
    n_chunks = 0
    for start in range(0, samples.size, chunk_len):
        chunk_index = n_chunks
        n_chunks += 1
        chunk = samples[start : start + chunk_len]
        chunk_start_sec = start / float(sample_rate)
        chunk_end_sec = (start + len(chunk)) / float(sample_rate)
        if evidence_chunk_index is None and chunk_end_sec + 1e-9 >= evidence_end_sec:
            evidence_chunk_index = chunk_index
        # Deliver every turn whose time falls in this
        # chunk before the chunk is pushed, so the
        # discourse state for the chunk includes it.
        while turn_idx < len(ordered_turns) and ordered_turns[turn_idx].time_sec <= chunk_end_sec + 1e-9:
            turn = ordered_turns[turn_idx]
            turn_idx += 1
            before = tick()
            pipeline.push_transcript(turn.text, float(turn.time_sec))
            elapsed = tick() - before
            turn_processing.append(float(elapsed))
            if turn.time_sec <= evidence_end_sec + 1e-9:
                turn_before_evidence += float(elapsed)
        before = tick()
        events = pipeline.push_audio(chunk, int(sample_rate))
        elapsed = tick() - before
        chunk_processing.append(float(elapsed))
        for event in events:
            risk = getattr(event, "risk", None)
            if risk is None:
                continue
            frame = getattr(event, "frame", None)
            frame_ts = float(frame.timestamp_sec) if frame is not None else chunk_start_sec
            tier = str(getattr(risk, "tier", ""))
            if tier == "HIGH_RISK" and first_warning_any is None:
                first_warning_any = (chunk_index, frame_ts)
            if evidence_chunk_index is not None and chunk_index >= evidence_chunk_index:
                if tier == "HIGH_RISK" and first_warning is None:
                    first_warning = (chunk_index, frame_ts)
                if tier in ("SUSPICIOUS", "HIGH_RISK") and first_provisional is None:
                    first_provisional = (chunk_index, frame_ts)
    feed_end = tick()
    if evidence_chunk_index is None:  # pragma: no cover - guarded above
        evidence_chunk_index = n_chunks - 1

    # Any turn left over falls after the last chunk;
    # deliver it so its processing is counted, but it
    # cannot affect a warning this run produced.
    while turn_idx < len(ordered_turns):
        turn = ordered_turns[turn_idx]
        turn_idx += 1
        before = tick()
        pipeline.push_transcript(turn.text, float(turn.time_sec))
        turn_processing.append(float(tick() - before))

    drained = True
    tier1_latencies: List[float] = []
    scored_windows = 0
    windows_total = 0
    drain = getattr(pipeline, "drain_shadow", None)
    if callable(drain):
        drained = bool(drain(float(drain_timeout_sec)))
    records = getattr(pipeline, "shadow_windows", None)
    if records is not None:
        windows = list(records)
        windows_total = len(windows)
        for record in windows:
            if getattr(record, "tier1_score", None) is not None:
                scored_windows += 1
            decision_stamp = getattr(record, "decision_monotonic_sec", None)
            tier1_stamp = getattr(record, "tier1_monotonic_sec", None)
            if decision_stamp is not None and tier1_stamp is not None:
                tier1_latencies.append(float(tier1_stamp) - float(decision_stamp))

    warned = first_warning is not None
    warned_before = (
        first_warning_any is not None
        and first_warning_any[0] < evidence_chunk_index
    )
    if warned:
        assert first_warning is not None
        first_ts = first_warning[1]
        accumulation = max(0.0, first_ts - float(evidence_end_sec))
    else:
        first_ts = None
        accumulation = None
    if first_provisional is not None:
        provisional_ts = first_provisional[1]
        provisional_accumulation = max(0.0, provisional_ts - float(evidence_end_sec))
        provisional = True
    else:
        provisional_ts = None
        provisional_accumulation = None
        provisional = False

    return WarningMeasurement(
        evidence_end_sec=float(evidence_end_sec),
        audio_duration_sec=float(duration_sec),
        chunk_ms=float(chunk_ms),
        n_chunks=n_chunks,
        warned=warned,
        warned_before_evidence_end=bool(warned_before),
        first_high_risk_audio_sec=first_ts,
        accumulation_audio_sec=accumulation,
        provisional_cue=provisional,
        first_provisional_audio_sec=provisional_ts,
        provisional_accumulation_audio_sec=provisional_accumulation,
        deciding_chunk_index=first_warning[0] if warned else None,
        deciding_chunk_processing_sec=(
            chunk_processing[first_warning[0]] if warned else None
        ),
        chunk_processing_sec=chunk_processing,
        turn_processing_sec=turn_processing,
        turn_processing_before_evidence_sec=float(turn_before_evidence),
        total_wall_sec=float(feed_end - feed_start),
        tier1_latency_sec=tier1_latencies,
        tier1_scored_windows=scored_windows,
        shadow_windows_total=windows_total,
        shadow_drained=drained,
        asr_lag_sec=asr_lag_sec,
        render_sec=render_sec,
        network_sec=network_sec,
    )


def run_repeated(
    factory: Callable[[], object],
    audio: np.ndarray,
    sample_rate: int,
    n_runs: int,
    **kwargs,
) -> Tuple[List[WarningMeasurement], RunSummary]:
    """Run the same call ``n_runs`` times, fresh pipeline each run."""
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    measurements = [
        measure_time_to_warning(factory(), audio, sample_rate, **kwargs)
        for _ in range(n_runs)
    ]
    return measurements, summarize_runs(measurements)


def summarize_runs(measurements: Sequence[WarningMeasurement]) -> RunSummary:
    """Summarise repeated runs; the 2 s verdict needs full stages."""
    runs = list(measurements)
    warned = [m for m in runs if m.warned]
    notes: List[str] = []
    if not runs:
        return RunSummary(0, 0, None, None, None, None, None, None, "incomplete", ["no runs"])
    if not warned:
        notes.append("no run reached HIGH_RISK; no warning time exists")
        return RunSummary(
            len(runs), 0, None, None, None, None, None, None, "incomplete", notes
        )
    accumulation = [m.accumulation_audio_sec for m in warned if m.accumulation_audio_sec is not None]
    deciding = [m.deciding_chunk_processing_sec for m in warned if m.deciding_chunk_processing_sec is not None]
    tier1_all: List[float] = []
    for m in warned:
        tier1_all.extend(m.tier1_latency_sec)
    estimated = [m.estimated_live_warning_sec for m in warned if m.estimated_live_warning_sec is not None]
    accumulation_p95 = _percentile(accumulation, 0.95)
    if accumulation_p95 is not None and accumulation_p95 > BUDGET_ACCUMULATION_SEC:
        notes.append("accumulation p95 exceeds the 700 ms budget")
    tier1_p95 = _percentile(tier1_all, 0.95)
    if tier1_p95 is not None and tier1_p95 > BUDGET_TIER1_SEC:
        notes.append("Tier 1 latency p95 exceeds the 200 ms budget")
    deciding_p95 = _percentile(deciding, 0.95)
    if deciding_p95 is not None and deciding_p95 > BUDGET_DECIDING_CHUNK_SEC:
        notes.append("deciding-chunk processing p95 exceeds the 200 ms budget")
    if any(m.measured_only for m in warned):
        notes.append("ASR lag, render, or network time unmeasured; full budget verdict incomplete")
        verdict = "incomplete"
    else:
        estimated_p95 = _percentile(estimated, 0.95)
        verdict = "met" if (estimated_p95 is not None and estimated_p95 <= BUDGET_WARNING_P95_SEC) else "not met"
        if verdict == "not met":
            notes.append("estimated live warning p95 exceeds 2 s")
    return RunSummary(
        n_runs=len(runs),
        warned_runs=len(warned),
        accumulation_p50_sec=_percentile(accumulation, 0.50),
        accumulation_p95_sec=accumulation_p95,
        deciding_chunk_p95_sec=deciding_p95,
        tier1_latency_p95_sec=tier1_p95,
        estimated_live_p50_sec=_percentile(estimated, 0.50),
        estimated_live_p95_sec=_percentile(estimated, 0.95),
        budget_verdict=verdict,
        budget_notes=notes,
    )


__all__ = [
    "BUDGET_ACCUMULATION_SEC",
    "BUDGET_DECIDING_CHUNK_SEC",
    "BUDGET_TIER1_SEC",
    "BUDGET_WARNING_P95_SEC",
    "RunSummary",
    "Turn",
    "WarningMeasurement",
    "measure_time_to_warning",
    "run_repeated",
    "summarize_runs",
]
