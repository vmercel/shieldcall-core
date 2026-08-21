"""
ShieldCall streaming pipeline  -  single entrypoint for dual-stream detection.

Wires preprocessor  ->  acoustic + linguistic  ->  CSCF fusion  ->  adaptation hooks
into one low-latency call loop suitable for embedding in a local service
or mobile FFI later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np

from .audio.preprocessor import TelephonyPreprocessor, Frame
from .audio.channel import ChannelConfig
from .acoustic.scorer import AcousticDeepfakeScorer, AcousticScore
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


@dataclass
class PipelineEvent:
    frame: Optional[Frame]
    acoustic: Optional[AcousticScore]
    linguistic: Optional[LinguisticScore]
    risk: Optional[FusedRisk]
    transcripts: List[TranscriptFragment] = field(default_factory=list)


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
        if ac.is_speech:
            alarm = self.cusum.update(ac.synthetic_prob, frame.timestamp_sec)
            if alarm is not None:
                self.sapc.observe_alarm(alarm.timestamp_sec)

        if ac.embedding is not None and ac.is_speech:
            gap = self.acoustic.memory.coverage_gap(ac.embedding)
            self.coverage.observe_gap(gap)

        transcripts = self.asr.push_audio(frame.samples, frame.sample_rate, frame.timestamp_sec)
        li_score: Optional[LinguisticScore] = None
        for frag in transcripts:
            li_score = self.linguistic.update(frag.text, frag.timestamp_sec)
            self.fusion.update_linguistic(li_score)
            self.sapc.observe_stage(li_score.discourse_stage, frag.timestamp_sec)

        risk: Optional[FusedRisk] = None
        if frame.frame_index % self.config.fuse_every_n_frames == 0:
            risk = self._attach_handoff(self.fusion.fuse(frame.timestamp_sec))
            self._last_risk = risk

        return PipelineEvent(
            frame=frame,
            acoustic=ac,
            linguistic=li_score,
            risk=risk,
            transcripts=transcripts,
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
