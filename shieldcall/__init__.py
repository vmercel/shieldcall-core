"""
ShieldCall Core — streaming dual-stream detection engine.

Joint scoring of (1) linguistic fraud-intent on transcripts and
(2) residual/vocoder artifacts on telephone-bandwidth audio.

Components (descriptive names; acronyms are shorthand only)
------------------------------------------------------------
- Telephone channel simulator (TCT)
- Residual fingerprint + prototype memory (STRF, PMA)
- Scam-script stage tracker (SDTG)
- Rule-based cross-stream fusion (CSCF)
- Heuristic uncertainty band (CSR)
- Input-space counterfactuals (CTE)
"""

__version__ = "0.5.0"

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
