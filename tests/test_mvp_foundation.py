from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from shieldcall.eval.governance import (
    DatasetManifest,
    EvaluationSampleRecord,
    HeldOutRiskCalibrator,
)
from shieldcall.governance import AdaptationRegistry, AdaptationStatus, ConsentRecord
from shieldcall.integrations import (
    AnalysisConsent,
    CallSession,
    IntegrationChannel,
    RiskEvidence,
    WarningPolicy,
    WarningTier,
)
from shieldcall.mvp_service import ShieldCallMVPService


def _consent() -> AnalysisConsent:
    return AnalysisConsent(
        consent_id="consent-1",
        subject_id="subject-1",
        purpose="fraud_risk_analysis",
        captured_at=datetime.now(timezone.utc).isoformat(),
        expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    )


def test_native_call_screening_does_not_authorize_audio_analysis():
    session = CallSession.create(IntegrationChannel.ANDROID_CALL_SCREENING, _consent(), tenant_id="tenant-1")
    with pytest.raises(PermissionError):
        session.authorize_audio_analysis()


def test_authorized_voip_session_can_receive_safe_warning():
    session = CallSession.create(IntegrationChannel.VOIP, _consent(), tenant_id="tenant-1")
    evidence = RiskEvidence(
        model_version="model-1",
        acoustic_probability=0.1,
        linguistic_probability=0.8,
        risk_score=0.8,
        calibration_lower=0.7,
        calibration_upper=0.9,
        regime="social_engineering",
    )
    warning = WarningPolicy().decide(session, evidence)
    assert warning.tier == WarningTier.HIGH_RISK_PATTERN
    assert "official channel" in warning.guidance


def test_held_out_calibrator_requires_evidence_and_returns_intervals():
    calibrator = HeldOutRiskCalibrator(alpha=0.1, minimum_examples=20)
    scores = np.linspace(0.0, 1.0, 20)
    labels = np.array([0] * 10 + [1] * 10)
    report = calibrator.fit(scores, labels)
    lower, upper = calibrator.interval_for(0.55)
    assert report.calibration_size == 20
    assert 0.0 <= lower <= upper <= 1.0


def test_dataset_manifest_rejects_train_test_speaker_leakage():
    records = (
        EvaluationSampleRecord("a", "train", 0, "ds", "speaker-1", "human", "clean", "en", "license-a"),
        EvaluationSampleRecord("b", "calibration", 1, "ds", "speaker-2", "tts-a", "clean", "en", "license-a"),
        EvaluationSampleRecord("c", "test", 1, "ds", "speaker-1", "tts-b", "clean", "en", "license-a"),
    )
    manifest = DatasetManifest("fixture", "1", records)
    with pytest.raises(ValueError):
        manifest.validate()


def test_adaptation_requires_consent_provenance_review_and_rollback():
    registry = AdaptationRegistry()
    consent = ConsentRecord(
        consent_id="consent-1",
        purpose="model_improvement",
        captured_at=datetime.now(timezone.utc).isoformat(),
    )
    candidate = registry.submit(
        feature_vector=[0.1, 0.2, 0.3],
        label_is_synthetic=True,
        source="human_review",
        family="tts-a",
        consent=consent,
        quality_score=0.95,
        provenance={"label_method": "two_person_review", "collection_channel": "authorized_pilot"},
        model_version="model-1",
    )
    with pytest.raises(ValueError):
        registry.create_release("model-1", [candidate.candidate_id], "release-manager")
    registry.approve(candidate.candidate_id, "reviewer-1", "Verified against case evidence", "approval-set-1")
    release = registry.create_release("model-1", [candidate.candidate_id], "release-manager")
    applied = registry.apply_release(release.release_id)
    assert applied.status == AdaptationStatus.APPLIED
    rolled_back = registry.rollback_release(release.release_id, "pilot error analysis")
    assert rolled_back.status == AdaptationStatus.ROLLED_BACK


def test_mvp_service_never_proceeds_without_authorized_channel():
    service = ShieldCallMVPService(model_version="model-1")
    session = CallSession.create(IntegrationChannel.ANDROID_CALL_SCREENING, _consent(), tenant_id="tenant-1")
    from shieldcall.integrations import AudioChunk

    chunk = AudioChunk(np.ones(80, dtype=np.float32), 8000, 0.0, session.session_id)
    with pytest.raises(PermissionError):
        service.analyze_chunk(session, chunk)
