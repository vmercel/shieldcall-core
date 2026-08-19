"""Consent-gated MVP inference orchestration.

This class is intentionally transport-agnostic. A VoIP or enterprise integration owns
audio capture, obtains consent, creates a CallSession, and passes short-lived chunks
for scoring. The service returns warnings and redacted audit metadata only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .eval.governance import HeldOutRiskCalibrator
from .integrations.contracts import (
    AudioChunk,
    CallSession,
    RiskEvidence,
    UserWarning,
    WarningPolicy,
    redactable_audit_event,
)
from .pipeline import PipelineConfig, ShieldCallPipeline


@dataclass(frozen=True)
class AnalysisOutcome:
    warning: Optional[UserWarning]
    audit_event: Optional[dict]
    emitted_events: int


class ShieldCallMVPService:
    """Applies production policy around the research pipeline.

    The service does not treat an uncalibrated score as a confirmed fraud finding.
    When a held-out calibrator has not been fitted for the active model release, it
    emits a deliberately wide interval that results in an uncertainty-oriented
    warning policy.
    """

    def __init__(
        self,
        model_version: str,
        pipeline: Optional[ShieldCallPipeline] = None,
        calibrator: Optional[HeldOutRiskCalibrator] = None,
        warning_policy: Optional[WarningPolicy] = None,
    ) -> None:
        self.model_version = model_version
        self.pipeline = pipeline or ShieldCallPipeline(config=PipelineConfig(use_conformal=False))
        self.calibrator = calibrator
        self.warning_policy = warning_policy or WarningPolicy()

    def analyze_chunk(self, session: CallSession, chunk: AudioChunk) -> AnalysisOutcome:
        session.authorize_audio_analysis()
        if chunk.session_id != session.session_id:
            raise PermissionError("Audio chunk session does not match the authorized call session")
        chunk.validate()
        events = self.pipeline.push_audio(chunk.samples, chunk.sample_rate)
        risk_event = next((event.risk for event in reversed(events) if event.risk is not None), None)
        if risk_event is None:
            return AnalysisOutcome(warning=None, audit_event=None, emitted_events=len(events))

        if self.calibrator is not None and self.calibrator.fitted:
            lower, upper = self.calibrator.interval_for(risk_event.risk_score)
        else:
            lower, upper = 0.0, 1.0
        drivers: List[str] = list(risk_event.active_linguistic_groups)
        if risk_event.regime:
            drivers.append(risk_event.regime)
        evidence = RiskEvidence(
            model_version=self.model_version,
            acoustic_probability=risk_event.acoustic_synth_prob,
            linguistic_probability=risk_event.linguistic_fraud_prob,
            risk_score=risk_event.risk_score,
            calibration_lower=lower,
            calibration_upper=upper,
            regime=risk_event.regime,
            drivers=tuple(drivers),
        )
        warning = self.warning_policy.decide(session, evidence)
        return AnalysisOutcome(
            warning=warning,
            audit_event=redactable_audit_event(session, warning),
            emitted_events=len(events),
        )
