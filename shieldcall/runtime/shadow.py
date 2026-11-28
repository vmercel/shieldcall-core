"""Pipeline shadow wiring (TDD v2.0 section 10.3; ADR-018).

The cascade (ADR-016) and the factorial discourse model
(ADR-017) run beside the live pipeline in shadow: trigger
decisions and Tier 1 scores are recorded beside CSCF
verdicts, and factorial states are recorded beside SDTG
scores. Nothing here touches a live decision path. CSCF
stays the production fusion, and the factorial fraud
score is recorded, never fused.

Tier 1 runs on one background worker thread, so a slow
encoder can never delay the 10 ms acoustic cadence. The
calling thread decides the trigger and hands a triggered
window to the worker; the worker fills the Tier 1 fields
of the window record in place. A record read before the
worker finishes honestly shows no Tier 1 score yet.

A Tier 1 outage is recorded, never hidden and never
retried on every window: once scoring fails with
``Tier1UnavailableError``, or the scorer factory raises,
the outage is latched and later windows are recorded
with the same reason. No score is ever invented. Shadow
code must never fail a call, so every scorer exception
is caught and recorded.

Speaker labels: the factorial model needs a speaker
because only far-end turns can constitute a requested
action (ADR-017 item 4). ASR fragments carry ``remote``,
``local``, or ``unknown``; ``remote`` maps to far,
``local`` to near, and ``unknown`` to far, the safe
default ADR-017 states for a feed that does not
diarise. Turns with no speaker at all are far-end.
Unrecognised labels also default to far-end rather
than raising into the call path.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..acoustic.cascade import (
    CascadeConfig,
    CascadeDecision,
    CascadeTrigger,
    Phase8Tier1Scorer,
    Tier1Result,
    Tier1UnavailableError,
    WindowBuilder,
    prob_to_llr,
)
from ..fusion.engine import FusedRisk
from ..linguistic.factorial import (
    FactorialDiscourseModel,
    FactorialState,
    requested_action_active,
)

# Bounded so a stalled Tier 1 scorer cannot grow the queue
# without bound. A full queue drops the oldest unstarted
# window and counts it, rather than blocking the call.
DEFAULT_QUEUE_SIZE = 64

REASON_NO_SCORER = "no scorer configured"
REASON_TIER1_UNAVAILABLE = "tier1 unavailable"
REASON_TIER1_ERROR = "tier1 error"
REASON_QUEUE_FULL = "queue full; window dropped"
REASON_SHADOW_CLOSED = "shadow closed"

_FAR_SPEAKERS = {
    "far", "far_end", "far-end", "caller", "agent", "scammer", "remote",
}
_NEAR_SPEAKERS = {
    "near", "near_end", "near-end", "user", "callee", "protected", "local",
}


def map_speaker(speaker: str) -> str:
    """Map an ASR/pipeline speaker label to the factorial ``far``/``near`` role.

    Unrecognised labels default to far-end, the safe default
    ADR-017 states for a feed that does not diarise: a
    requested action must not be missed because a label
    was spelled differently.
    """
    lowered = str(speaker).strip().lower()
    if lowered in _NEAR_SPEAKERS:
        return "near"
    return "far"


@dataclass
class ShadowConfig:
    """Configuration for the shadow observer (ADR-018).

    ``seed`` controls the cascade audit draws. ``None``
    seeds from operating system entropy and is the only
    correct choice for production: a fixed seed would let
    a prober learn exactly which windows are audited
    (ADR-016 item 3). Fixed seeds are for tests and
    reproducible offline runs only.
    """

    window_sec: float = 2.0
    sample_rate: int = 8000
    tau0: float = 0.0
    seed: Optional[int] = None
    requested_action_threshold: float = 0.5
    queue_size: int = DEFAULT_QUEUE_SIZE

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0.0 <= self.requested_action_threshold <= 1.0:
            raise ValueError("requested_action_threshold must be in [0, 1]")
        if self.queue_size < 1:
            raise ValueError("queue_size must be at least 1")


@dataclass
class ShadowWindowRecord:
    """One completed window, with its shadow evidence.

    Created at decision time with the Tier 1 fields empty
    and the most recent CSCF risk attached. The background
    worker fills ``tier1_score``, ``tier1_llr``,
    ``tier1_calibrated``, and ``tier1_model_sha256`` in
    place when scoring succeeds, or
    ``tier1_unavailable_reason`` when it does not. A
    record read before the worker finishes shows no Tier 1
    score and no unavailability reason yet; that is the
    honest state, not a missing field.
    """

    window_index: int
    start_sec: float
    end_sec: float
    decision: CascadeDecision
    tier0_llr: float
    changepoint_latched: bool
    requested_action_latched: bool
    factorial_action: Optional[str]
    factorial_hazard: Optional[float]
    sdtg_stage: Optional[str]
    cscf_risk_score: Optional[float]
    cscf_tier: Optional[str]
    tier1_score: Optional[float] = None
    tier1_llr: Optional[float] = None
    tier1_calibrated: bool = False
    tier1_model_sha256: Optional[str] = None
    tier1_unavailable_reason: Optional[str] = None
    tier1_dropped: bool = False
    # Wall-clock stamps for time-to-warning measurement
    # (ADR-019). Set from time.monotonic, never from audio
    # time. The Tier 1 stamp stays None on any record that
    # was never scored, so no latency is computed from it.
    decision_monotonic_sec: Optional[float] = None
    tier1_monotonic_sec: Optional[float] = None


# A scorer scores one window and returns a Tier1Result.
# Phase8Tier1Scorer matches this shape.
Tier1ScorerFn = Callable[[np.ndarray, int], Tier1Result]
ScorerFactory = Callable[[], object]


class ShadowObserver:
    """Shadow cascade + factorial state for one call.

    One instance serves one pipeline, which serves one
    call at a time (``reset`` starts a new call). Feed
    frames through :meth:`observe_frame` and transcript
    turns through :meth:`observe_turn`, in call order.
    Read completed windows from :attr:`windows` and
    factorial states from :attr:`factorial_states`.
    """

    def __init__(
        self,
        config: Optional[ShadowConfig] = None,
        *,
        scorer_factory: Optional[ScorerFactory] = None,
        tier1_scorer: Optional[object] = None,
    ):
        self.config = config or ShadowConfig()
        cascade_config = CascadeConfig(
            window_sec=self.config.window_sec,
            sample_rate=self.config.sample_rate,
            tau0=self.config.tau0,
            seed=self.config.seed,
        )
        self._builder = WindowBuilder(cascade_config)
        self._trigger = CascadeTrigger(cascade_config)
        self._factorial = FactorialDiscourseModel()
        self._scorer_factory = scorer_factory
        self._scorer = tier1_scorer
        self._scorer_tried = tier1_scorer is not None
        self._tier1_unavailable_reason: Optional[str] = None
        self._llr_sum = 0.0
        self._llr_samples = 0
        self._changepoint_latched = False
        self._requested_action_latched = False
        self._last_factorial: Optional[FactorialState] = None
        self._last_sdtg_stage: Optional[str] = None
        self._last_cscf: Optional[FusedRisk] = None
        self._windows: List[ShadowWindowRecord] = []
        self._factorial_states: List[FactorialState] = []
        self._lock = threading.Lock()
        self._generation = 0
        self.dropped_windows = 0
        self._queue: "queue.Queue[Optional[Tuple[int, object, ShadowWindowRecord]]]" = queue.Queue(
            maxsize=self.config.queue_size
        )
        self._worker: Optional[threading.Thread] = None
        self._closed = False

    def set_tier1_scorer(self, scorer: object) -> None:
        """Install a Tier 1 scorer directly (tests and offline runs).

        Clears a latched outage: an operator who replaces a
        missing scorer expects scoring to resume. The scorer
        must expose ``score_window(samples, sample_rate)``
        returning a ``Tier1Result``.
        """
        self._scorer = scorer
        self._scorer_tried = True
        self._tier1_unavailable_reason = None

    # -- feeding ------------------------------------------------------

    def observe_turn(
        self,
        text: str,
        timestamp_sec: float,
        speaker: str = "far",
        sdtg_stage: Optional[str] = None,
    ) -> FactorialState:
        """Feed one transcript turn; return the factorial state after it."""
        state = self._factorial.update(text, float(timestamp_sec), map_speaker(speaker))
        self._last_factorial = state
        self._factorial_states.append(state)
        if sdtg_stage is not None:
            self._last_sdtg_stage = str(sdtg_stage)
        if requested_action_active(state, self.config.requested_action_threshold):
            self._requested_action_latched = True
        return state

    def observe_frame(
        self,
        hop_samples: np.ndarray,
        is_speech: bool,
        t_sec: float,
        tier0_llr: float,
        changepoint_alarm: bool,
        cscf_risk: Optional[FusedRisk] = None,
    ) -> Optional[ShadowWindowRecord]:
        """Feed one hop of audio; return a record when a window completes.

        ``hop_samples`` is the newest hop of one pipeline
        frame (ADR-018: frames overlap, so only the newest
        hop advances the window). ``tier0_llr`` is the
        frame's Tier 0 LLR, from ``prob_to_llr`` on the
        STRF synthetic probability; it is accumulated over
        the speech hops of the window, weighted by sample
        count, and the window LLR is the mean. Non-speech
        hops contribute nothing to the window or the LLR.
        """
        if cscf_risk is not None:
            self._last_cscf = cscf_risk
        if changepoint_alarm:
            self._changepoint_latched = True
        arr = np.asarray(hop_samples, dtype=np.float32)
        if bool(is_speech) and arr.size:
            self._llr_sum += float(tier0_llr) * int(arr.size)
            self._llr_samples += int(arr.size)
        window = self._builder.add(arr, bool(is_speech), float(t_sec))
        if window is None:
            return None
        mean_llr = self._llr_sum / self._llr_samples if self._llr_samples else 0.0
        decision = self._trigger.decide(
            mean_llr,
            changepoint=self._changepoint_latched,
            requested_action=self._requested_action_latched,
        )
        factorial = self._last_factorial
        cscf = self._last_cscf
        record = ShadowWindowRecord(
            window_index=window.window_index,
            start_sec=float(window.start_sec),
            end_sec=float(window.end_sec),
            decision=decision,
            tier0_llr=float(mean_llr),
            changepoint_latched=self._changepoint_latched,
            requested_action_latched=self._requested_action_latched,
            factorial_action=factorial.action if factorial is not None else None,
            factorial_hazard=factorial.hazard if factorial is not None else None,
            sdtg_stage=self._last_sdtg_stage,
            cscf_risk_score=float(cscf.risk_score) if cscf is not None else None,
            cscf_tier=str(cscf.tier) if cscf is not None else None,
            decision_monotonic_sec=time.monotonic(),
        )
        self._llr_sum = 0.0
        self._llr_samples = 0
        self._changepoint_latched = False
        self._requested_action_latched = False
        with self._lock:
            self._windows.append(record)
        if decision.triggered:
            self._submit(window, record)
        else:
            record.tier1_unavailable_reason = "not triggered"
        return record

    # -- background Tier 1 --------------------------------------------

    def _submit(self, window: object, record: ShadowWindowRecord) -> None:
        if self._closed:
            record.tier1_unavailable_reason = REASON_SHADOW_CLOSED
            record.tier1_dropped = True
            self.dropped_windows += 1
            return
        if self._scorer is None and self._scorer_factory is None:
            record.tier1_unavailable_reason = REASON_NO_SCORER
            return
        if self._tier1_unavailable_reason is not None:
            record.tier1_unavailable_reason = self._tier1_unavailable_reason
            return
        if self._worker is None:
            self._worker = threading.Thread(
                target=self._worker_loop, name="shieldcall-shadow-tier1", daemon=True
            )
            self._worker.start()
        item: Tuple[int, object, ShadowWindowRecord] = (self._generation, window, record)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Drop the oldest unstarted window, not the new one,
            # and count it. The dropped record keeps its
            # decision; only its Tier 1 fields say it was
            # never scored, and why.
            try:
                _gen, _win, old = self._queue.get_nowait()
                with self._lock:
                    old.tier1_unavailable_reason = REASON_QUEUE_FULL
                    old.tier1_dropped = True
                self.dropped_windows += 1
                self._queue.task_done()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:  # pragma: no cover - single-threaded race only
                record.tier1_unavailable_reason = REASON_QUEUE_FULL
                record.tier1_dropped = True
                self.dropped_windows += 1

    def _get_scorer(self) -> Optional[object]:
        if self._scorer is not None:
            return self._scorer
        if self._scorer_tried:
            return None
        self._scorer_tried = True
        factory = self._scorer_factory
        if factory is None:
            return None
        try:
            self._scorer = factory()
        except Tier1UnavailableError as exc:
            self._tier1_unavailable_reason = REASON_TIER1_UNAVAILABLE
            self._scorer = None
            return None
        except Exception:
            self._tier1_unavailable_reason = REASON_TIER1_ERROR
            self._scorer = None
            return None
        return self._scorer

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                generation, window, record = item
                if generation != self._generation:
                    # Stale call: the record belongs to a
                    # discarded call's list. Never fill it.
                    continue
                self._score_window(window, record, generation)
            finally:
                self._queue.task_done()

    def _score_window(
        self, window: object, record: ShadowWindowRecord, generation: int
    ) -> None:
        if self._tier1_unavailable_reason is not None:
            with self._lock:
                record.tier1_unavailable_reason = self._tier1_unavailable_reason
            return
        scorer = self._get_scorer()
        if scorer is None:
            with self._lock:
                record.tier1_unavailable_reason = (
                    self._tier1_unavailable_reason or REASON_NO_SCORER
                )
            return
        try:
            result = scorer.score_window(window.samples, self.config.sample_rate)  # type: ignore[attr-defined]
        except Tier1UnavailableError:
            self._tier1_unavailable_reason = REASON_TIER1_UNAVAILABLE
            with self._lock:
                record.tier1_unavailable_reason = REASON_TIER1_UNAVAILABLE
            return
        except Exception:
            with self._lock:
                record.tier1_unavailable_reason = REASON_TIER1_ERROR
            return
        if generation != self._generation:
            # The call ended while Tier 1 was scoring. The
            # result belongs to the discarded call: drop it
            # rather than filling a record no one reads.
            return
        with self._lock:
            record.tier1_score = float(result.score)
            record.tier1_llr = result.llr
            record.tier1_calibrated = bool(result.calibrated)
            record.tier1_model_sha256 = result.model_sha256
            record.tier1_monotonic_sec = time.monotonic()

    # -- reading and lifecycle ----------------------------------------

    @property
    def windows(self) -> List[ShadowWindowRecord]:
        with self._lock:
            return list(self._windows)

    @property
    def factorial_states(self) -> List[FactorialState]:
        return list(self._factorial_states)

    @property
    def last_factorial(self) -> Optional[FactorialState]:
        return self._last_factorial

    def drain_shadow(self, timeout_sec: float = 5.0) -> bool:
        """Wait until every queued window has been scored or dropped.

        Returns True when the queue is empty, False on
        timeout. Callers that need complete shadow records
        (tests, offline runs) use this instead of sleeping.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while self._queue.unfinished_tasks > 0:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        return True

    def reset(self) -> None:
        """Start a new call: fresh windows, trigger, and factorial state.

        Queued windows from the previous call are dropped by
        generation, never scored into the new call's records.
        A latched Tier 1 outage and a constructed scorer are
        kept: a missing backbone stays missing across calls,
        and reloading it per call would repeat a doomed load.
        The worker thread is kept alive.
        """
        self._generation += 1
        self._builder.reset()
        self._trigger.reset()
        self._factorial.reset()
        self._llr_sum = 0.0
        self._llr_samples = 0
        self._changepoint_latched = False
        self._requested_action_latched = False
        self._last_factorial = None
        self._last_sdtg_stage = None
        self._last_cscf = None
        with self._lock:
            self._windows = []
        self._factorial_states = []
        self.dropped_windows = 0

    def close(self) -> None:
        """Stop the background worker. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout=5.0)
            self._worker = None


def scorer_factory_for_model(model_path: str) -> ScorerFactory:
    """Build a lazy Phase 8 Tier 1 scorer factory for a model file.

    The factory is called at most once, by the background
    worker, on the first triggered window. Construction
    reads the model file and raises
    ``Tier1UnavailableError`` when it is missing or
    unreadable, which the observer latches as an outage.
    The backbone loads from the local cache only, exactly
    as ADR-016 specifies; scoring never downloads.
    """

    def factory() -> Phase8Tier1Scorer:
        return Phase8Tier1Scorer(str(model_path))

    return factory


__all__ = [
    "DEFAULT_QUEUE_SIZE",
    "REASON_NO_SCORER",
    "REASON_QUEUE_FULL",
    "REASON_SHADOW_CLOSED",
    "REASON_TIER1_ERROR",
    "REASON_TIER1_UNAVAILABLE",
    "ShadowConfig",
    "ShadowObserver",
    "ShadowWindowRecord",
    "map_speaker",
    "prob_to_llr",
    "scorer_factory_for_model",
]
