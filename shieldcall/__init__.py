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

__version__ = "0.2.0"

from .fusion.engine import FusionEngine, FusedRisk
from .acoustic.scorer import AcousticDeepfakeScorer, AcousticScore
from .linguistic.scorer import LinguisticFraudScorer, LinguisticScore
from .pipeline import ShieldCallPipeline, PipelineConfig, PipelineEvent
from .audio.channel import TelephonyChannelTwin, ChannelConfig, CodecProfile

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
]
