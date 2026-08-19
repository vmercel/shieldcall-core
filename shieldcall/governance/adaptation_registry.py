"""Governed adaptation intake for ShieldCall model updates.

The research prototype exposed direct online adaptation. This module replaces that
unsafe path in production-facing code with a reviewable workflow: a candidate
must carry consent, provenance, quality metadata, and an approval decision
before it can be used by a model-training or adaptation job.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Dict, Iterable, List, Mapping, Optional, Sequence
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray


class AdaptationStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class ConsentRecord:
    """Evidence that a sample may be used for the stated purpose."""

    consent_id: str
    purpose: str
    captured_at: str
    expires_at: Optional[str] = None


@dataclass(frozen=True)
class AdaptationCandidate:
    """A reviewable, feature-only adaptation candidate.

    Raw audio is deliberately excluded. Callers should persist raw communications
    only through a separate, consented evidence system with its own retention
    policy. The candidate stores a stable feature fingerprint for audit and
    deduplication, not the original recording.
    """

    candidate_id: str
    feature_vector: Sequence[float]
    label_is_synthetic: bool
    source: str
    family: str
    consent: ConsentRecord
    quality_score: float
    provenance: Mapping[str, str]
    created_at: str
    model_version: str
    status: AdaptationStatus = AdaptationStatus.PENDING
    reviewer: Optional[str] = None
    review_note: Optional[str] = None

    @property
    def feature_fingerprint(self) -> str:
        array: NDArray[np.float32] = np.asarray(self.feature_vector, dtype=np.float32)
        return sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True)
class AdaptationApproval:
    candidate_id: str
    reviewer: str
    approved_at: str
    release_id: str
    rationale: str


@dataclass
class AdaptationRelease:
    """An immutable, reviewable group of approved adaptation samples."""

    release_id: str
    base_model_version: str
    candidate_ids: List[str]
    created_at: str
    created_by: str
    status: AdaptationStatus = AdaptationStatus.APPROVED
    metadata: Dict[str, str] = field(default_factory=dict)


class AdaptationRegistry:
    """In-memory reference implementation of a governed update workflow.

    Production deployments should implement the same contract with encrypted,
    access-controlled durable storage and immutable audit events.
    """

    MIN_QUALITY_SCORE = 0.80
    ALLOWED_SOURCES = {"human_review", "verified_liveness", "authorized_pilot"}

    def __init__(self) -> None:
        self._candidates: Dict[str, AdaptationCandidate] = {}
        self._approvals: Dict[str, AdaptationApproval] = {}
        self._releases: Dict[str, AdaptationRelease] = {}
        self._applied_release_ids: List[str] = []

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def submit(
        self,
        feature_vector: Sequence[float],
        label_is_synthetic: bool,
        source: str,
        family: str,
        consent: ConsentRecord,
        quality_score: float,
        provenance: Mapping[str, str],
        model_version: str,
    ) -> AdaptationCandidate:
        if source not in self.ALLOWED_SOURCES:
            raise ValueError("Adaptation source is not approved for governed intake")
        if not consent.consent_id or not consent.purpose:
            raise ValueError("A consent record with purpose is required")
        if quality_score < 0.0 or quality_score > 1.0:
            raise ValueError("quality_score must be between 0 and 1")
        if not provenance.get("label_method") or not provenance.get("collection_channel"):
            raise ValueError("Provenance must identify label_method and collection_channel")
        vector: NDArray[np.float32] = np.asarray(feature_vector, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("feature_vector must be a finite, non-empty one-dimensional vector")

        candidate = AdaptationCandidate(
            candidate_id=str(uuid4()),
            feature_vector=tuple(float(item) for item in vector),
            label_is_synthetic=bool(label_is_synthetic),
            source=source,
            family=family or "unknown",
            consent=consent,
            quality_score=float(quality_score),
            provenance=dict(provenance),
            created_at=self.utc_now(),
            model_version=model_version,
        )
        if any(existing.feature_fingerprint == candidate.feature_fingerprint for existing in self._candidates.values()):
            raise ValueError("Duplicate adaptation candidate")
        self._candidates[candidate.candidate_id] = candidate
        return candidate

    def approve(self, candidate_id: str, reviewer: str, rationale: str, release_id: str) -> AdaptationApproval:
        candidate = self._get_candidate(candidate_id)
        if candidate.status != AdaptationStatus.PENDING:
            raise ValueError("Only pending candidates may be approved")
        if candidate.quality_score < self.MIN_QUALITY_SCORE:
            raise ValueError("Candidate quality is below the governed adaptation threshold")
        if not reviewer or not rationale or not release_id:
            raise ValueError("Reviewer, rationale, and release_id are required")

        approval = AdaptationApproval(
            candidate_id=candidate_id,
            reviewer=reviewer,
            approved_at=self.utc_now(),
            release_id=release_id,
            rationale=rationale,
        )
        self._approvals[candidate_id] = approval
        self._candidates[candidate_id] = AdaptationCandidate(
            **{**asdict(candidate), "status": AdaptationStatus.APPROVED, "reviewer": reviewer, "review_note": rationale}
        )
        return approval

    def create_release(
        self,
        base_model_version: str,
        candidate_ids: Iterable[str],
        created_by: str,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> AdaptationRelease:
        ids = list(candidate_ids)
        if not ids:
            raise ValueError("An adaptation release requires at least one approved candidate")
        for candidate_id in ids:
            candidate = self._get_candidate(candidate_id)
            approval = self._approvals.get(candidate_id)
            if candidate.status != AdaptationStatus.APPROVED or approval is None:
                raise ValueError("All release candidates require recorded approval")
        release_id = str(uuid4())
        release = AdaptationRelease(
            release_id=release_id,
            base_model_version=base_model_version,
            candidate_ids=ids,
            created_at=self.utc_now(),
            created_by=created_by,
            metadata=dict(metadata or {}),
        )
        self._releases[release_id] = release
        return release

    def apply_release(self, release_id: str) -> AdaptationRelease:
        release = self._get_release(release_id)
        if release.status != AdaptationStatus.APPROVED:
            raise ValueError("Only approved releases may be applied")
        self._releases[release_id] = AdaptationRelease(**{**asdict(release), "status": AdaptationStatus.APPLIED})
        for candidate_id in release.candidate_ids:
            candidate = self._get_candidate(candidate_id)
            self._candidates[candidate_id] = AdaptationCandidate(**{**asdict(candidate), "status": AdaptationStatus.APPLIED})
        self._applied_release_ids.append(release_id)
        return self._releases[release_id]

    def rollback_release(self, release_id: str, reason: str) -> AdaptationRelease:
        release = self._get_release(release_id)
        if release.status != AdaptationStatus.APPLIED:
            raise ValueError("Only an applied release may be rolled back")
        metadata = dict(release.metadata)
        metadata["rollback_reason"] = reason
        metadata["rolled_back_at"] = self.utc_now()
        self._releases[release_id] = AdaptationRelease(
            **{**asdict(release), "status": AdaptationStatus.ROLLED_BACK, "metadata": metadata}
        )
        for candidate_id in release.candidate_ids:
            candidate = self._get_candidate(candidate_id)
            self._candidates[candidate_id] = AdaptationCandidate(**{**asdict(candidate), "status": AdaptationStatus.ROLLED_BACK})
        return self._releases[release_id]

    def approved_feature_batch(self, release_id: str) -> tuple[np.ndarray, np.ndarray]:
        release = self._get_release(release_id)
        if release.status not in {AdaptationStatus.APPROVED, AdaptationStatus.APPLIED}:
            raise ValueError("Feature batches are available only for approved or applied releases")
        candidates = [self._get_candidate(candidate_id) for candidate_id in release.candidate_ids]
        features: NDArray[np.float32] = np.asarray(
            [candidate.feature_vector for candidate in candidates], dtype=np.float32
        )
        labels: NDArray[np.bool_] = np.asarray(
            [candidate.label_is_synthetic for candidate in candidates], dtype=bool
        )
        return features, labels

    def _get_candidate(self, candidate_id: str) -> AdaptationCandidate:
        if candidate_id not in self._candidates:
            raise KeyError(f"Unknown adaptation candidate: {candidate_id}")
        return self._candidates[candidate_id]

    def _get_release(self, release_id: str) -> AdaptationRelease:
        if release_id not in self._releases:
            raise KeyError(f"Unknown adaptation release: {release_id}")
        return self._releases[release_id]
