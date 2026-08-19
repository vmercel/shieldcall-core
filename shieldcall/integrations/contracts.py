"""Production integration contracts for authorized, privacy-preserving call analysis.

These contracts intentionally do not implement native carrier-call capture. ShieldCall
may analyze audio only when a supported integration supplies it and a valid consent
record covers the requested purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional
from uuid import uuid4

import numpy as np


class IntegrationChannel(str, Enum):
    VOIP = "voip"
    ENTERPRISE_CONTACT_CENTER = "enterprise_contact_center"
    ANDROID_CALL_SCREENING = "android_call_screening"


class WarningTier(str, Enum):
    NO_WARNING = "no_warning"
    VERIFY_BEFORE_ACTING = "verify_before_acting"
    HIGH_RISK_PATTERN = "high_risk_pattern"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class AnalysisConsent:
    consent_id: str
    subject_id: str
    purpose: str
    captured_at: str
    active: bool = True
    expires_at: Optional[str] = None

    def permits(self, requested_purpose: str, now: Optional[datetime] = None) -> bool:
        if not self.active or self.purpose != requested_purpose:
            return False
        if self.expires_at is None:
            return True
        parsed_expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        return (now or datetime.now(timezone.utc)) < parsed_expiry


@dataclass(frozen=True)
class CallSession:
    session_id: str
    channel: IntegrationChannel
    consent: AnalysisConsent
    created_at: str
    tenant_id: str
    participant_reference: Optional[str] = None

    @classmethod
    def create(
        cls,
        channel: IntegrationChannel,
        consent: AnalysisConsent,
        tenant_id: str,
        participant_reference: Optional[str] = None,
    ) -> "CallSession":
        return cls(
            session_id=str(uuid4()),
            channel=channel,
            consent=consent,
            created_at=datetime.now(timezone.utc).isoformat(),
            tenant_id=tenant_id,
            participant_reference=participant_reference,
        )

    @property
    def audio_analysis_supported(self) -> bool:
        return self.channel in {IntegrationChannel.VOIP, IntegrationChannel.ENTERPRISE_CONTACT_CENTER}

    def authorize_audio_analysis(self) -> None:
        if not self.audio_analysis_supported:
            raise PermissionError("This integration channel does not provide audio analysis authority")
        if not self.consent.permits("fraud_risk_analysis"):
            raise PermissionError("A current consent record for fraud_risk_analysis is required")


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray
    sample_rate: int
    timestamp_sec: float
    session_id: str

    def validate(self) -> None:
        if self.sample_rate not in {8000, 16000, 24000, 48000}:
            raise ValueError("Unsupported sample rate for the MVP contract")
        if self.samples.ndim != 1 or self.samples.size == 0:
            raise ValueError("Audio chunks must be one-dimensional and non-empty")
        if not np.isfinite(self.samples).all():
            raise ValueError("Audio chunks must contain finite samples")


@dataclass(frozen=True)
class RiskEvidence:
    model_version: str
    acoustic_probability: float
    linguistic_probability: float
    risk_score: float
    calibration_lower: float
    calibration_upper: float
    regime: str
    drivers: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class UserWarning:
    tier: WarningTier
    title: str
    guidance: str
    evidence: RiskEvidence
    session_id: str
    created_at: str


class WarningPolicy:
    """Deterministic policy that avoids accusing callers or auto-blocking calls."""

    def decide(self, session: CallSession, evidence: RiskEvidence) -> UserWarning:
        if evidence.calibration_upper - evidence.calibration_lower >= 0.45:
            tier = WarningTier.UNCERTAIN
            title = "Unable to verify this call with confidence"
            guidance = "Avoid sharing sensitive information. Verify the caller through an official number before acting."
        elif evidence.risk_score >= 0.62:
            tier = WarningTier.HIGH_RISK_PATTERN
            title = "High-risk fraud pattern detected"
            guidance = "Pause the call. Do not send money or share codes. Verify the request through an official channel."
        elif evidence.risk_score >= 0.35:
            tier = WarningTier.VERIFY_BEFORE_ACTING
            title = "Verify before acting"
            guidance = "The call includes risk signals. Independently verify the caller before sharing information or making a payment."
        else:
            tier = WarningTier.NO_WARNING
            title = "No high-risk pattern detected"
            guidance = "This result is not a guarantee of safety. Continue to use normal verification practices."
        return UserWarning(
            tier=tier,
            title=title,
            guidance=guidance,
            evidence=evidence,
            session_id=session.session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


def redactable_audit_event(session: CallSession, warning: UserWarning) -> Dict[str, object]:
    """Return metadata suitable for encrypted audit logging without raw audio."""

    return {
        "event_id": str(uuid4()),
        "session_id": session.session_id,
        "tenant_id": session.tenant_id,
        "channel": session.channel.value,
        "consent_id": session.consent.consent_id,
        "warning_tier": warning.tier.value,
        "risk_score": warning.evidence.risk_score,
        "model_version": warning.evidence.model_version,
        "created_at": warning.created_at,
    }
