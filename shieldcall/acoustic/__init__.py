from .scorer import AcousticDeepfakeScorer, AcousticScore, PrototypeMemory
from .features import extract_frame_features, FEATURE_DIM
from .residual import extract_residual_fingerprint, ResidualFingerprint

__all__ = [
    "AcousticDeepfakeScorer",
    "AcousticScore",
    "PrototypeMemory",
    "extract_frame_features",
    "FEATURE_DIM",
    "extract_residual_fingerprint",
    "ResidualFingerprint",
]
