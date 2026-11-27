"""
ShieldCall streaming pipeline  -  single entrypoint for dual-stream detection.

Wires preprocessor  ->  acoustic + linguistic  ->  CSCF fusion  ->  adaptation hooks
into one low-latency call loop suitable for embedding in a local service
or mobile FFI later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, List, Optional

import numpy as np

from .audio.preprocessor import TelephonyPreprocessor, Frame
from .audio.channel import ChannelConfig
from .acoustic.scorer import AcousticDeepfakeScorer, AcousticScore
from .acoustic.cascade import prob_to_llr
from .linguistic.factorial import FactorialState

if TYPE_CHECKING:
    # Imported lazily in __init__: importing the runtime package
    # at module level would cycle (runtime imports the pipeline).
    from .runtime.shadow import ShadowObserver, ShadowWindowRecord
from .linguistic.scorer import LinguisticFraudScorer, LinguisticScore
from .linguistic.asr_bridge import ASRBridge, PassthroughASR, TranscriptFragment
from .fusion.engine import FusionEngine, FusedRisk
from .fusion.coupling import StageAlignedCoupling
from .acoustic.changepoint import StreamingCUSUM
from .adaptation.hooks import AdaptationBuffer, AdaptationExample
from .adaptation.coverage import CoverageDebtTracker


@dataclass
class PipelineConfig:
    target_sr: int = 8000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    acoustic_weight: float = 0.40
    linguistic_weight: float = 0.60
    suspicious_threshold: float = 0.35
    high_risk_threshold: float = 0.62
    use_conformal: bool = True
    channel: Optional[ChannelConfig] = None
    fuse_every_n_frames: int = 5
    # Shadow wiring (ADR-018): the cascade and the factorial
    # discourse model run beside CSCF and are recorded, never
    # fused. False restores the pre-ADR-018 pipeline exactly.
    shadow_enabled: bool = True
    # Neutral LLR point, not a fitted operating point (ADR-016).
    # Set from benign shadow traffic before any live use.
    shadow_tau0: float = 0.0
    # Audit seed: None seeds from OS entropy, the only correct
    # choice for production. Fixed seeds are for tests only.
    shadow_audit_seed: Optional[int] = None
    # Phase 8 model file for Tier 1. None means no scorer is
    # configured: triggered windows are recorded with the
    # reason, and no score is invented.
    shadow_tier1_model_path: Optional[str] = None
    shadow_requested_action_threshold: float = 0.5


@dataclass
class PipelineEvent:
    frame: Optional[Frame]
    acoustic: Optional[AcousticScore]
    linguistic: Optional[LinguisticScore]
    risk: Optional[FusedRisk]
    transcripts: List[TranscriptFragment] = field(default_factory=list)
    # Shadow-only (ADR-018): factorial state after this event's
    # turns, and the cascade window record when a window
    # completed on this frame. Neither affects the risk.
    factorial: Optional[FactorialState] = None
    shadow_window: Optional[ShadowWindowRecord] = None


class ShieldCallPipeline:
    """End-to-end streaming dual-stream detection engine."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        asr: Optional[ASRBridge] = None,
    ):
        self.config = config or PipelineConfig()
        self.preprocessor = TelephonyPreprocessor(
            target_sr=self.config.target_sr,
            frame_ms=self.config.frame_ms,
            hop_ms=self.config.hop_ms,
            channel_config=self.config.channel,
        )
        self.acoustic = AcousticDeepfakeScorer()
        self.linguistic = LinguisticFraudScorer()
        self.fusion = FusionEngine(
            acoustic_weight=self.config.acoustic_weight,
            linguistic_weight=self.config.linguistic_weight,
            suspicious_threshold=self.config.suspicious_threshold,
            high_risk_threshold=self.config.high_risk_threshold,
            use_conformal=self.config.use_conformal,
        )
        self.asr: ASRBridge = asr or PassthroughASR()
        self.adaptation_buffer = AdaptationBuffer()
        self.coverage = CoverageDebtTracker()
        self.cusum = StreamingCUSUM()
        self.sapc = StageAlignedCoupling()
        self._last_risk: Optional[FusedRisk] = None
        self.shadow: Optional[ShadowObserver] = None
        if self.config.shadow_enabled:
            # Lazy import: the runtime package imports this
            # module, so a top-level import would cycle.
            from .runtime.shadow import (
                ShadowConfig,
                ShadowObserver,
                scorer_factory_for_model,
            )

            factory = None
            if self.config.shadow_tier1_model_path is not None:
                factory = scorer_factory_for_model(
                    self.config.shadow_tier1_model_path
                )
            self.shadow = ShadowObserver(
                ShadowConfig(
                    window_sec=2.0,
                    sample_rate=self.config.target_sr,
                    tau0=self.config.shadow_tau0,
                    seed=self.config.shadow_audit_seed,
                    requested_action_threshold=self.config.shadow_requested_action_threshold,
                ),
                scorer_factory=factory,
            )

    @property
    def shadow_windows(self) -> List[ShadowWindowRecord]:
        """Completed cascade windows, in order (empty when shadow is off)."""
        return self.shadow.windows if self.shadow is not None else []

    @property
    def last_factorial(self) -> Optional[FactorialState]:
        """Latest factorial state, or None when shadow is off / no turn yet."""
        return self.shadow.last_factorial if self.shadow is not None else None

    def drain_shadow(self, timeout_sec: float = 5.0) -> bool:
        """Wait for queued Tier 1 windows to finish; True when drained."""
        return self.shadow.drain_shadow(timeout_sec) if self.shadow is not None else True

    def close(self) -> None:
        """Stop the shadow worker thread, if one was started. Idempotent."""
        if self.shadow is not None:
            self.shadow.close()

    def reset(self) -> None:
        self.preprocessor.reset()
        self.acoustic.reset()
        self.linguistic.reset()
        self.fusion.reset()
        self.asr.reset()
        self.coverage.reset()
        self.cusum.reset()
        self.sapc.reset()
        self._last_risk = None
        if self.shadow is not None:
            self.shadow.reset()

    def push_audio(self, samples: np.ndarray, sample_rate: int) -> List[PipelineEvent]:
        events: List[PipelineEvent] = []
        frames = self.preprocessor.push(samples, sample_rate)
        for frame in frames:
            events.append(self._process_frame(frame))
        return events

    def push_transcript(self, text: str, timestamp_sec: float) -> LinguisticScore:
        frag = self.asr.push_text(text, timestamp_sec)
        score = self.linguistic.update(frag.text, frag.timestamp_sec)
        self.fusion.update_linguistic(score)
        if self.shadow is not None:
            # No speaker is supplied on this path; far-end is
            # the safe default (ADR-018).
            self.shadow.observe_turn(
                frag.text, frag.timestamp_sec, "far", score.discourse_stage
            )
        return score

    def _attach_handoff(self, risk: FusedRisk) -> FusedRisk:
        coup = self.sapc.evaluate()
        risk.handoff_statistic = coup.statistic
        risk.handoff_pvalue = coup.p_value
        risk.handoff_score = coup.score
        # Timing evidence is extra, not a replacement for CSCF.
        # Only raise risk when the permutation test is in the tail.
        if coup.p_value < 0.15 and coup.statistic >= 0.35:
            risk.risk_score = max(risk.risk_score, coup.score)
            if risk.risk_score >= self.config.high_risk_threshold:
                risk.tier = "HIGH_RISK"
            elif risk.risk_score >= self.config.suspicious_threshold and risk.tier == "SAFE":
                risk.tier = "SUSPICIOUS"
        return risk

    def _process_frame(self, frame: Frame) -> PipelineEvent:
        ac = self.acoustic.score_frame(frame)
        self.fusion.update_acoustic(ac)
        alarm = False
        if ac.is_speech:
            cusum_alarm = self.cusum.update(ac.synthetic_prob, frame.timestamp_sec)
            if cusum_alarm is not None:
                alarm = True
                self.sapc.observe_alarm(cusum_alarm.timestamp_sec)

        if ac.embedding is not None and ac.is_speech:
            gap = self.acoustic.memory.coverage_gap(ac.embedding)
            self.coverage.observe_gap(gap)

        transcripts = self.asr.push_audio(frame.samples, frame.sample_rate, frame.timestamp_sec)
        li_score: Optional[LinguisticScore] = None
        factorial: Optional[FactorialState] = None
        for frag in transcripts:
            li_score = self.linguistic.update(frag.text, frag.timestamp_sec)
            self.fusion.update_linguistic(li_score)
            self.sapc.observe_stage(li_score.discourse_stage, frag.timestamp_sec)
            if self.shadow is not None:
                factorial = self.shadow.observe_turn(
                    frag.text, frag.timestamp_sec, frag.speaker, li_score.discourse_stage
                )

        risk: Optional[FusedRisk] = None
        if frame.frame_index % self.config.fuse_every_n_frames == 0:
            risk = self._attach_handoff(self.fusion.fuse(frame.timestamp_sec))
            self._last_risk = risk

        shadow_window: Optional[ShadowWindowRecord] = None
        if self.shadow is not None:
            # Frames overlap (25 ms frame, 10 ms hop), so only
            # the newest hop advances the shadow window; the
            # whole frame would double-count audio (ADR-018).
            hop_len = max(1, self.preprocessor.hop_len)
            hop = frame.samples[-hop_len:] if len(frame.samples) > hop_len else frame.samples
            hop_t = frame.timestamp_sec + (len(frame.samples) - len(hop)) / frame.sample_rate
            shadow_window = self.shadow.observe_frame(
                hop,
                ac.is_speech,
                hop_t,
                prob_to_llr(ac.synthetic_prob),
                alarm,
                risk,
            )

        return PipelineEvent(
            frame=frame,
            acoustic=ac,
            linguistic=li_score,
            risk=risk,
            transcripts=transcripts,
            factorial=factorial,
            shadow_window=shadow_window,
        )

    def stream(
        self, samples: np.ndarray, sample_rate: int, chunk_ms: float = 100.0
    ) -> Iterator[PipelineEvent]:
        self.reset()
        chunk_len = max(1, int(sample_rate * chunk_ms / 1000.0))
        for start in range(0, len(samples), chunk_len):
            chunk = samples[start : start + chunk_len]
            for ev in self.push_audio(chunk, sample_rate):
                yield ev

    def adapt(
        self,
        features: np.ndarray,
        is_synthetic: bool,
        source: str = "human_review",
        family: str = "unknown",
    ) -> None:
        gap_before = self.acoustic.memory.coverage_gap(features)
        self.acoustic.adapt(features, is_synthetic)
        gap_after = self.acoustic.memory.coverage_gap(features)
        self.adaptation_buffer.add(
            AdaptationExample(
                features=features,
                is_synthetic=is_synthetic,
                source=source,
                family=family,
            )
        )
        self.coverage.register_adaptation(family, is_synthetic, gap_before, gap_after)

    @property
    def last_risk(self) -> Optional[FusedRisk]:
        return self._last_risk
