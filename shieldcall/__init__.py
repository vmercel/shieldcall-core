"""
ShieldCall Core  -  Streaming dual-stream detection engine.

Joint linguistic fraud-intent and acoustic synthesis-artifact analysis
designed for real telephone conditions and rapid adaptation to new synthesizers.

Novel subsystems
----------------
- STRF: Spectral-Temporal Residual Fingerprinting
- SDTG: Scam Discourse Trajectory Graph
- CSCF: Cross-Stream Causal Fusion
- PMA:  Prototype Memory Adaptation + Coverage-Debt tracking
- TCT:  Telephony Channel Twin
- CSR:  Conformal Streaming Risk
- CTE:  Counterfactual Threat Explanations
"""

__version__ = "0.3.0"

from .acoustic.scorer import AcousticDeepfakeScorer, AcousticScore
from .audio.channel import ChannelConfig, CodecProfile, TelephonyChannelTwin
from .fusion.engine import FusedRisk, FusionEngine
from .linguistic.scorer import LinguisticFraudScorer, LinguisticScore
from .mvp_service import AnalysisOutcome, ShieldCallMVPService
from .pipeline import PipelineConfig, PipelineEvent, ShieldCallPipeline

__all__ = [
    "__version__",
    "FusionEngine",
    "FusedRisk",
    "AcousticDeepfakeScorer",
    "AcousticScore",
    "LinguisticFraudScorer",
    "LinguisticScore",
    "ShieldCallPipeline",
    "PipelineConfig",
    "PipelineEvent",
    "TelephonyChannelTwin",
    "ChannelConfig",
    "CodecProfile",
    "AnalysisOutcome",
    "ShieldCallMVPService",
]
