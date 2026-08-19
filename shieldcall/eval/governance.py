"""Reproducibility and calibration primitives for ShieldCall model releases."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import ceil
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EvaluationSampleRecord:
    """Metadata only. Audio remains in a governed dataset store."""

    sample_id: str
    split: str
    label: int
    source_dataset: str
    speaker_id: str
    attack_family: str
    codec_condition: str
    language: str
    consent_or_license_ref: str


@dataclass(frozen=True)
class DatasetManifest:
    dataset_name: str
    dataset_version: str
    records: Tuple[EvaluationSampleRecord, ...]
    locked_test_split: str = "test"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.dataset_name or not self.dataset_version:
            raise ValueError("Dataset name and version are required")
        if not self.records:
            raise ValueError("A dataset manifest requires records")
        identifiers = [record.sample_id for record in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate sample IDs in dataset manifest")
        splits = {record.split for record in self.records}
        required_splits = {"train", "calibration", self.locked_test_split}
        if not required_splits.issubset(splits):
            raise ValueError("Manifest must include train, calibration, and locked test splits")
        train_speakers = {record.speaker_id for record in self.records if record.split == "train"}
        test_speakers = {record.speaker_id for record in self.records if record.split == self.locked_test_split}
        overlap = train_speakers.intersection(test_speakers) - {"unknown"}
        if overlap:
            raise ValueError("Speaker overlap between train and locked test splits is not permitted")
        for record in self.records:
            if record.label not in {0, 1}:
                raise ValueError("Labels must be 0 or 1")
            if not record.consent_or_license_ref:
                raise ValueError("Every sample must include a consent or license reference")

    @property
    def fingerprint(self) -> str:
        rows = "\n".join(
            "|".join(
                [
                    record.sample_id,
                    record.split,
                    str(record.label),
                    record.source_dataset,
                    record.speaker_id,
                    record.attack_family,
                    record.codec_condition,
                    record.language,
                    record.consent_or_license_ref,
                ]
            )
            for record in sorted(self.records, key=lambda item: item.sample_id)
        )
        return sha256(rows.encode("utf-8")).hexdigest()

    def split_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.records:
            counts[record.split] = counts.get(record.split, 0) + 1
        return counts


@dataclass(frozen=True)
class CalibrationReport:
    calibration_size: int
    alpha: float
    residual_quantile: float
    empirical_coverage: float
    mean_interval_width: float


class HeldOutRiskCalibrator:
    """Split-conformal-style interval builder from a locked calibration split.

    The calibrator requires labeled holdout examples and never updates itself from
    unlabeled live scores. Coverage still depends on the validation distribution
    matching the deployment population and must be re-evaluated per release.
    """

    def __init__(self, alpha: float = 0.1, minimum_examples: int = 20) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.minimum_examples = minimum_examples
        self._quantile: Optional[float] = None
        self._calibration_size = 0

    @property
    def fitted(self) -> bool:
        return self._quantile is not None

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> CalibrationReport:
        score_array: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
        label_array: NDArray[np.float64] = np.asarray(labels, dtype=np.float64)
        if score_array.ndim != 1 or label_array.ndim != 1 or score_array.size != label_array.size:
            raise ValueError("Scores and labels must be aligned one-dimensional sequences")
        if score_array.size < self.minimum_examples:
            raise ValueError("Insufficient held-out calibration examples")
        if np.any(score_array < 0.0) or np.any(score_array > 1.0):
            raise ValueError("Scores must be probabilities in [0, 1]")
        if not np.isin(label_array, [0.0, 1.0]).all():
            raise ValueError("Labels must be binary")
        residuals = np.abs(score_array - label_array)
        level = min(1.0, ceil((score_array.size + 1) * (1.0 - self.alpha)) / score_array.size)
        self._quantile = float(np.quantile(residuals, level, method="higher"))
        self._calibration_size = int(score_array.size)
        lower, upper = self.intervals(score_array.tolist())
        coverage = float(np.mean((label_array >= lower) & (label_array <= upper)))
        return CalibrationReport(
            calibration_size=self._calibration_size,
            alpha=self.alpha,
            residual_quantile=self._quantile,
            empirical_coverage=coverage,
            mean_interval_width=float(np.mean(upper - lower)),
        )

    def intervals(self, scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        if self._quantile is None:
            raise RuntimeError("Fit the calibrator on a held-out split before requesting intervals")
        score_array: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
        return (
            np.clip(score_array - self._quantile, 0.0, 1.0),
            np.clip(score_array + self._quantile, 0.0, 1.0),
        )

    def interval_for(self, score: float) -> tuple[float, float]:
        lower, upper = self.intervals([score])
        return float(lower[0]), float(upper[0])


def brier_score(scores: Sequence[float], labels: Sequence[int]) -> float:
    score_array: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
    label_array: NDArray[np.float64] = np.asarray(labels, dtype=np.float64)
    if score_array.shape != label_array.shape:
        raise ValueError("Scores and labels must have the same shape")
    return float(np.mean((score_array - label_array) ** 2))


def expected_calibration_error(scores: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    score_array: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
    label_array: NDArray[np.float64] = np.asarray(labels, dtype=np.float64)
    if score_array.shape != label_array.shape:
        raise ValueError("Scores and labels must have the same shape")
    if bins <= 1:
        raise ValueError("bins must be greater than one")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (score_array >= lower) & (score_array < upper if upper < 1.0 else score_array <= upper)
        if not np.any(mask):
            continue
        confidence = float(np.mean(score_array[mask]))
        accuracy = float(np.mean(label_array[mask]))
        error += float(np.mean(mask)) * abs(confidence - accuracy)
    return float(error)
