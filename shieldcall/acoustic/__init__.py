from .scorer import AcousticDeepfakeScorer, AcousticScore, PrototypeMemory
from .features import extract_frame_features, FEATURE_DIM
from .residual import extract_residual_fingerprint, ResidualFingerprint
from .cascade import (
    CascadeConfig,
    CascadeDecision,
    CascadeReport,
    CascadeRow,
    CascadeTrigger,
    CascadeWindow,
    Phase8Tier1Scorer,
    Tier1Result,
    Tier1UnavailableError,
    WindowBuilder,
    evaluate_cascade,
    prob_to_llr,
)
from .passport import (
    ChannelPassport,
    PassportVector,
    calibrated_llr,
    estimate_passport,
    gate_weights,
)

__all__ = [
    "AcousticDeepfakeScorer",
    "AcousticScore",
    "PrototypeMemory",
    "CascadeConfig",
    "CascadeDecision",
    "CascadeReport",
    "CascadeRow",
    "CascadeTrigger",
    "CascadeWindow",
    "Phase8Tier1Scorer",
    "Tier1Result",
    "Tier1UnavailableError",
    "WindowBuilder",
    "evaluate_cascade",
    "prob_to_llr",
    "extract_frame_features",
    "FEATURE_DIM",
    "extract_residual_fingerprint",
    "ResidualFingerprint",
    "ChannelPassport",
    "PassportVector",
    "calibrated_llr",
    "estimate_passport",
    "gate_weights",
]
