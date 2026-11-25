"""Risk-triggered acoustic cascade (TDD v2.0 section 6.5; ADR-016).

STRF (Tier 0) runs on every frame. The Phase 8 encoder (Tier 1)
runs on a 2 s window of far-end speech only when a trigger fires.
This module is the decision logic, the window assembly, the Tier 1
scorer, and the offline kill-criteria evaluator. It is shadow-only:
nothing here touches the pipeline, the fusion engine, or any live
decision path. CSCF stays the production fusion.

Trigger proxies, stated plainly because TDD specifies the exact
forms and they are not built yet: the change-point trigger uses a
latched StreamingCUSUM alarm, not the BOCPD posterior of TDD
section 6.7; the requested-action trigger uses a latched SDTG
stage in HARVEST or PAYMENT, not the factorial requested-action
state of TDD section 7.3. Both proxies are replaced by the exact
forms in follow-up slices without changing this interface.

Labels follow the repo convention: 1 = spoof, 0 = bona fide.
Tier 1 scores are spoof scores: higher means more spoof-like.
The Phase 8 head file stores the opposite sign (a bona fide
logit), and the scorer negates it; see ``Phase8Tier1Scorer``.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Callable, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

# TDD section 6.5: random audit probability per window.
DEFAULT_AUDIT_PROBABILITY = 0.02
# TDD section 6.5: first window of far-end speech in every call.
FIRST_WINDOW_INDEX = 0
# TDD Table 31: cascade kill criteria.
MAX_BENIGN_TRIGGER_RATE = 0.15
MAX_CASCADE_MISS_GAP = 0.01
# TDD section 6.5: Tier 1 costs at most about four Tier 0 windows.
# This is the TDD bound, not a measurement; callers pass a measured
# cost when one exists.
DEFAULT_TIER1_COST = 4.0
DEFAULT_TIER0_COST = 1.0

_EPS = 1e-6


@dataclass(frozen=True)
class CascadeConfig:
    """Configuration for the cascade (ADR-016).

    ``tau0`` is the Tier 0 LLR trigger threshold. Its default of 0.0
    is the neutral point of a calibrated LLR, not a fitted operating
    point; TDD section 6.5 requires it to be set low (high recall)
    from benign shadow traffic before any live use.

    ``seed`` controls the audit draws. ``None`` seeds from operating
    system entropy and is the only correct choice for production:
    a fixed seed would let a prober learn exactly which windows are
    audited. Fixed seeds are for tests and reproducible offline runs.
    """

    window_sec: float = 2.0
    sample_rate: int = 8000
    tau0: float = 0.0
    audit_probability: float = DEFAULT_AUDIT_PROBABILITY
    seed: Optional[int] = None
    requested_action_stages: Tuple[str, ...] = ("HARVEST", "PAYMENT")

    def __post_init__(self) -> None:
        if not math.isfinite(self.window_sec) or self.window_sec <= 0:
            raise ValueError("window_sec must be a positive finite number")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not math.isfinite(self.tau0):
            raise ValueError("tau0 must be finite")
        if not (0.0 <= self.audit_probability <= 1.0):
            raise ValueError("audit_probability must be in [0, 1]")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be an int or None")

    @property
    def window_samples(self) -> int:
        """Speech samples per window."""
        return int(round(self.window_sec * self.sample_rate))


def prob_to_llr(probability: float) -> float:
    """Clipped logit of a Tier 0 synthetic probability.

    This is an LLR only to the extent the input probability is
    calibrated. With no fitted calibrator (ADR-005) it is an
    uncalibrated LLR, and callers must treat it as such. The
    clipping keeps perfectly confident inputs finite.
    """
    p = float(probability)
    if not math.isfinite(p):
        raise ValueError("probability must be finite")
    p = min(max(p, _EPS), 1.0 - _EPS)
    return float(math.log(p / (1.0 - p)))


@dataclass
class CascadeWindow:
    """One 2 s window of far-end speech, ready for Tier 1."""

    samples: np.ndarray
    start_sec: float
    end_sec: float
    window_index: int

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


class WindowBuilder:
    """Accumulate far-end speech into fixed 2 s windows.

    Non-speech frames contribute nothing, so a window is 2 s of
    far-end speech rather than 2 s of wall time. A partial window
    held at reset is discarded, never scored short.
    """

    def __init__(self, config: Optional[CascadeConfig] = None):
        self.config = config or CascadeConfig()
        self._buffer: List[np.ndarray] = []
        self._buffered = 0
        self._start_sec: Optional[float] = None
        self._windows_completed = 0
        self.dropped_partial_samples = 0

    @property
    def pending_samples(self) -> int:
        return self._buffered

    @property
    def windows_completed(self) -> int:
        return self._windows_completed

    def add(
        self, samples: np.ndarray, is_speech: bool, t_sec: float
    ) -> Optional[CascadeWindow]:
        """Add one frame; return a completed window, or None.

        ``samples`` is one frame of mono audio at the configured
        sample rate. ``t_sec`` is the frame timestamp. Only speech
        frames advance the window.
        """
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("samples must be a one-dimensional array")
        if arr.size and not bool(np.isfinite(arr).all()):
            raise ValueError("samples must be finite")
        if not math.isfinite(float(t_sec)):
            raise ValueError("t_sec must be finite")
        if not is_speech or arr.size == 0:
            return None
        if self._start_sec is None:
            self._start_sec = float(t_sec)
        self._buffer.append(np.array(arr, copy=True))
        self._buffered += int(arr.size)
        if self._buffered < self.config.window_samples:
            return None
        window = np.concatenate(self._buffer)[: self.config.window_samples]
        leftover = np.concatenate(self._buffer)[self.config.window_samples :]
        self._buffer = [leftover] if leftover.size else []
        self._buffered = int(leftover.size)
        start = self._start_sec
        self._start_sec = float(t_sec) if self._buffered else None
        index = self._windows_completed
        self._windows_completed += 1
        return CascadeWindow(
            samples=window,
            start_sec=float(start),
            end_sec=float(t_sec),
            window_index=index,
        )

    def reset(self) -> int:
        """Discard any partial window; return dropped sample count."""
        dropped = self._buffered
        self.dropped_partial_samples += dropped
        self._buffer = []
        self._buffered = 0
        self._start_sec = None
        self._windows_completed = 0
        return dropped


@dataclass(frozen=True)
class CascadeDecision:
    """Trigger decision for one window, with every reason recorded."""

    window_index: int
    triggered: bool
    reasons: FrozenSet[str]
    tier0_llr: float
    changepoint: bool
    requested_action: bool
    audit_draw: float
    audit_triggered: bool


class CascadeTrigger:
    """Decide, per window, whether Tier 1 should look (TDD 6.5).

    The trigger holds the window counter and the audit generator,
    so one instance serves one call. The caller supplies the
    window Tier 0 LLR (mean over the window's speech frames) and
    latched flags for the change-point alarm and the
    requested-action state observed during the window.
    """

    REASON_FIRST_WINDOW = "first_window"
    REASON_TIER0_LLR = "tier0_llr"
    REASON_CHANGEPOINT = "changepoint"
    REASON_REQUESTED_ACTION = "requested_action"
    REASON_RANDOM_AUDIT = "random_audit"

    def __init__(self, config: Optional[CascadeConfig] = None):
        self.config = config or CascadeConfig()
        self._rng = random.Random(self.config.seed)
        self._window_index = 0

    @property
    def window_index(self) -> int:
        return self._window_index

    def is_requested_action(self, discourse_stage: str) -> bool:
        """True when an SDTG stage is a requested-action stage.

        This is the ADR-016 proxy for the factorial requested-action
        state (TDD section 7.3), which is not built yet.
        """
        return str(discourse_stage) in self.config.requested_action_stages

    def decide(
        self,
        tier0_llr: float,
        *,
        changepoint: bool = False,
        requested_action: bool = False,
    ) -> CascadeDecision:
        llr = float(tier0_llr)
        if not math.isfinite(llr):
            raise ValueError("tier0_llr must be finite")
        # One draw per window, including the first, so the draw
        # sequence never depends on which reasons held.
        draw = float(self._rng.random())
        index = self._window_index
        self._window_index += 1
        reasons = set()
        if index == FIRST_WINDOW_INDEX:
            reasons.add(self.REASON_FIRST_WINDOW)
        if llr > self.config.tau0:
            reasons.add(self.REASON_TIER0_LLR)
        if changepoint:
            reasons.add(self.REASON_CHANGEPOINT)
        if requested_action:
            reasons.add(self.REASON_REQUESTED_ACTION)
        audit = index != FIRST_WINDOW_INDEX and draw < self.config.audit_probability
        if audit:
            reasons.add(self.REASON_RANDOM_AUDIT)
        return CascadeDecision(
            window_index=index,
            triggered=bool(reasons),
            reasons=frozenset(reasons),
            tier0_llr=llr,
            changepoint=bool(changepoint),
            requested_action=bool(requested_action),
            audit_draw=draw,
            audit_triggered=bool(audit),
        )

    def reset(self) -> None:
        """Start a new call: fresh counter, fresh audit stream."""
        self._rng = random.Random(self.config.seed)
        self._window_index = 0


class Tier1UnavailableError(RuntimeError):
    """Tier 1 cannot score: a required model, library, or cache is missing.

    Raised instead of any fallback score. A cascade that cannot run
    Tier 1 must record the outage, never pretend Tier 1 looked.
    """


@dataclass(frozen=True)
class Tier1Result:
    """Tier 1 score for one window.

    ``score`` is the uncalibrated spoof logit (higher means more
    spoof-like). ``llr`` is present only when the caller supplied a
    fitted calibrator; otherwise it is None and ``calibrated`` is
    False, and no caller may treat ``score`` as an LLR.
    """

    score: float
    llr: Optional[float]
    calibrated: bool
    window_sec: float
    model_sha256: str
    backbone: str
    layers: Tuple[int, ...]


# Embedding function type: 16 kHz window samples -> hidden-state
# statistics, one row per hidden state (row 0 is the embedding
# output, rows 1..N are transformer layers), each row holding the
# mean and standard deviation over time concatenated. This matches
# the Phase 8 harness layout, so an injected function can stand in
# for the real backbone in tests and offline runs.
EmbedFn = Callable[[np.ndarray], np.ndarray]


class Phase8Tier1Scorer:
    """Tier 1: truncated wav2vec2 encoder plus the Phase 8 head.

    The head is the logistic regression fitted on ASVspoof 2019 LA
    train and dev only (``phases/phase8/model8.npz``). Its stored
    score is a bona fide logit, so this scorer negates it into a
    spoof score. The 11.78% EER measured on whole clips does not
    transfer to 2 s windows: windowed accuracy is unmeasured until
    the shadow gate re-measures it (ADR-016 item 11).

    The backbone is loaded lazily on the first scored window, from
    the local Hugging Face cache only. Importing this module never
    imports torch or transformers, and constructing the scorer
    never downloads anything.
    """

    BACKBONE_SAMPLE_RATE = 16000

    def __init__(
        self,
        model_path: str,
        *,
        backbone_id: str = "facebook/wav2vec2-base",
        cache_dir: Optional[str] = None,
        embed_fn: Optional[EmbedFn] = None,
        calibrator: object = None,
        cell: str = "unknown",
    ):
        self.model_path = str(model_path)
        self.backbone_id = str(backbone_id)
        self.cache_dir = cache_dir
        self._embed_fn = embed_fn
        self._calibrator = calibrator
        self.cell = str(cell)
        self._backbone = None
        try:
            with open(self.model_path, "rb") as handle:
                raw = handle.read()
        except OSError as exc:
            raise Tier1UnavailableError(
                f"Phase 8 model file is missing or unreadable: {self.model_path}"
            ) from exc
        self.model_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            with np.load(self.model_path, allow_pickle=False) as data:
                kind = str(data["m_kind"])
                layers = tuple(int(v) for v in np.asarray(data["m_layers"]).ravel())
                mu = np.asarray(data["m_mu"], dtype=np.float64)
                sd = np.asarray(data["m_sd"], dtype=np.float64)
                weights = np.asarray(data["m_w"], dtype=np.float64)
                bias = float(np.asarray(data["m_b"]).ravel()[0])
                meta_raw = data["meta"]
                meta = str(meta_raw.ravel()[0]) if meta_raw.size else ""
        except (KeyError, ValueError, OSError) as exc:
            raise Tier1UnavailableError(
                f"Phase 8 model file is unreadable: {self.model_path}"
            ) from exc
        if kind != "lr":
            raise Tier1UnavailableError(
                f"unsupported Phase 8 head kind {kind!r}: only 'lr' is scored"
            )
        if not layers or len(weights) % len(layers) != 0:
            raise Tier1UnavailableError("Phase 8 head layers and weights disagree")
        if not (len(mu) == len(sd) == len(weights)):
            raise Tier1UnavailableError("Phase 8 head vectors disagree in length")
        self.layers = layers
        self._mu = mu
        self._sd = sd
        self._weights = weights
        self._bias = bias
        self.meta = meta

    def score_window(self, samples: np.ndarray, sample_rate: int) -> Tier1Result:
        """Score one 2 s window; raise Tier1UnavailableError if Tier 1 is absent."""
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("samples must be a non-empty one-dimensional array")
        if not bool(np.isfinite(arr).all()):
            raise ValueError("samples must be finite")
        if int(sample_rate) not in (8000, self.BACKBONE_SAMPLE_RATE):
            raise ValueError(
                "sample_rate must be 8000 or 16000; other rates are refused, "
                "never silently resampled"
            )
        if arr.size < self.BACKBONE_SAMPLE_RATE // 2:
            raise ValueError("window is too short to score: need at least 0.5 s")
        samples16k = self._to_16k(arr, int(sample_rate))
        features = self._embed(samples16k)
        flat = np.asarray(features, dtype=np.float64)[list(self.layers), :].reshape(-1)
        if flat.size != self._weights.size:
            raise Tier1UnavailableError(
                "backbone feature width does not match the Phase 8 head"
            )
        bona_fide_logit = float(((flat - self._mu) / self._sd) @ self._weights + self._bias)
        score = -bona_fide_logit
        llr: Optional[float] = None
        calibrated = False
        if self._calibrator is not None:
            llr = float(self._calibrator.llr(self.cell, score))  # type: ignore[attr-defined]
            calibrated = True
        return Tier1Result(
            score=float(score),
            llr=llr,
            calibrated=calibrated,
            window_sec=float(arr.size / int(sample_rate)),
            model_sha256=self.model_sha256,
            backbone=self.backbone_id,
            layers=self.layers,
        )

    def _to_16k(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == self.BACKBONE_SAMPLE_RATE:
            return np.array(samples, dtype=np.float32, copy=True)
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:  # pragma: no cover - scipy is declared
            raise Tier1UnavailableError("scipy is required to upsample to 16 kHz") from exc
        return resample_poly(samples, 2, 1).astype(np.float32)

    def _embed(self, samples16k: np.ndarray) -> np.ndarray:
        if self._embed_fn is not None:
            features = np.asarray(self._embed_fn(samples16k), dtype=np.float64)
            if features.ndim != 2 or features.shape[0] != 1 + max(self.layers):
                raise Tier1UnavailableError(
                    "injected embedding has the wrong hidden-state layout"
                )
            return features
        if self._backbone is None:
            self._backbone = self._load_backbone()
        return self._backbone(samples16k)

    def _load_backbone(self) -> EmbedFn:
        try:
            import torch
            from transformers import Wav2Vec2Model
        except ImportError as exc:
            raise Tier1UnavailableError(
                "torch and transformers are required to run Tier 1"
            ) from exc
        n_layers = int(max(self.layers))
        try:
            model = Wav2Vec2Model.from_pretrained(
                self.backbone_id, local_files_only=True, cache_dir=self.cache_dir
            ).eval()
        except (OSError, ValueError) as exc:
            raise Tier1UnavailableError(
                "wav2vec2 backbone is not in the local cache; Tier 1 never "
                "downloads a model at scoring time"
            ) from exc
        model.encoder.layers = model.encoder.layers[:n_layers]

        def embed(samples16k: np.ndarray) -> np.ndarray:
            # Per-window normalisation, matching the Phase 8 harness
            # applied at clip scale. No silence trim: windows are
            # fixed 2 s slices of speech (ADR-016 item 5).
            x = np.asarray(samples16k, dtype=np.float32)
            x = (x - x.mean()) / (x.std() + 1e-6)
            with torch.no_grad():
                hidden = model(
                    torch.from_numpy(x.astype(np.float32))[None],
                    output_hidden_states=True,
                ).hidden_states
            return np.concatenate(
                [np.r_[h[0].mean(0).numpy(), h[0].std(0).numpy()] for h in hidden]
            ).astype(np.float64)

        return embed


@dataclass(frozen=True)
class CascadeRow:
    """One window for offline kill-criteria evaluation.

    ``tier1_score`` must be present for every row, including
    windows the cascade did not trigger: the evaluator runs Tier 1
    on all windows offline so the cascade miss rate can be
    measured against always-on Tier 1. A live cascade never
    computes the scores this row needs.
    """

    label: int
    triggered: bool
    tier1_score: float


@dataclass(frozen=True)
class CascadeReport:
    """Kill-criteria measurement for one operating point (TDD Table 31)."""

    n_windows: int
    n_spoof: int
    n_bona: int
    n_triggered: int
    trigger_rate: float
    benign_trigger_rate: float
    fnr_trigger: float
    always_on_miss_rate: float
    cascade_miss_rate: float
    miss_gap: float
    expected_cost: float
    cost_ratio: float
    passes_kill_criteria: bool
    tier1_threshold: float
    tier0_cost: float
    tier1_cost: float


def evaluate_cascade(
    rows: Sequence[CascadeRow],
    tier1_threshold: float,
    *,
    tier0_cost: float = DEFAULT_TIER0_COST,
    tier1_cost: float = DEFAULT_TIER1_COST,
) -> CascadeReport:
    """Measure the cascade against always-on Tier 1 (TDD eq. 8, Table 31).

    A spoof window is missed by always-on Tier 1 when its Tier 1
    score is at or below the threshold. It is missed by the cascade
    when it was not triggered, or when it was triggered and Tier 1
    still missed it. The kill check passes only when the benign
    trigger rate is at most 0.15 and the cascade miss rate is
    within 1 point of the always-on miss rate.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("evaluate_cascade needs at least one row")
    if not math.isfinite(float(tier1_threshold)):
        raise ValueError("tier1_threshold must be finite")
    if not (math.isfinite(tier0_cost) and tier0_cost > 0):
        raise ValueError("tier0_cost must be positive and finite")
    if not (math.isfinite(tier1_cost) and tier1_cost > 0):
        raise ValueError("tier1_cost must be positive and finite")
    for row in rows:
        if row.label not in (0, 1):
            raise ValueError("label must be 0 (bona fide) or 1 (spoof)")
        if not math.isfinite(float(row.tier1_score)):
            raise ValueError("tier1_score must be finite")
    n_windows = len(rows)
    spoof = [r for r in rows if r.label == 1]
    bona = [r for r in rows if r.label == 0]
    n_triggered = sum(1 for r in rows if r.triggered)
    trigger_rate = n_triggered / n_windows
    benign_trigger_rate = (
        sum(1 for r in bona if r.triggered) / len(bona) if bona else 0.0
    )
    fnr_trigger = (
        sum(1 for r in spoof if not r.triggered) / len(spoof) if spoof else 0.0
    )
    always_on_miss_rate = (
        sum(1 for r in spoof if r.tier1_score <= tier1_threshold) / len(spoof)
        if spoof
        else 0.0
    )
    cascade_miss_rate = (
        sum(
            1
            for r in spoof
            if (not r.triggered) or r.tier1_score <= tier1_threshold
        )
        / len(spoof)
        if spoof
        else 0.0
    )
    miss_gap = cascade_miss_rate - always_on_miss_rate
    expected_cost = tier0_cost + trigger_rate * tier1_cost
    return CascadeReport(
        n_windows=n_windows,
        n_spoof=len(spoof),
        n_bona=len(bona),
        n_triggered=n_triggered,
        trigger_rate=float(trigger_rate),
        benign_trigger_rate=float(benign_trigger_rate),
        fnr_trigger=float(fnr_trigger),
        always_on_miss_rate=float(always_on_miss_rate),
        cascade_miss_rate=float(cascade_miss_rate),
        miss_gap=float(miss_gap),
        expected_cost=float(expected_cost),
        cost_ratio=float(expected_cost / tier0_cost),
        passes_kill_criteria=bool(
            benign_trigger_rate <= MAX_BENIGN_TRIGGER_RATE
            and miss_gap <= MAX_CASCADE_MISS_GAP
        ),
        tier1_threshold=float(tier1_threshold),
        tier0_cost=float(tier0_cost),
        tier1_cost=float(tier1_cost),
    )


__all__ = [
    "CascadeConfig",
    "CascadeDecision",
    "CascadeReport",
    "CascadeRow",
    "CascadeTrigger",
    "CascadeWindow",
    "DEFAULT_AUDIT_PROBABILITY",
    "DEFAULT_TIER0_COST",
    "DEFAULT_TIER1_COST",
    "MAX_BENIGN_TRIGGER_RATE",
    "MAX_CASCADE_MISS_GAP",
    "Phase8Tier1Scorer",
    "Tier1Result",
    "Tier1UnavailableError",
    "WindowBuilder",
    "evaluate_cascade",
    "prob_to_llr",
]
