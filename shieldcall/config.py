"""YAML/dict configuration loader for ShieldCall Core."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .audio.channel import ChannelConfig, CodecProfile
from .pipeline import PipelineConfig

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULTS: Dict[str, Any] = {
    "audio": {
        "target_sr": 8000,
        "frame_ms": 25.0,
        "hop_ms": 10.0,
        "channel_profile": "narrowband",
    },
    "fusion": {
        "acoustic_weight": 0.40,
        "linguistic_weight": 0.60,
        "suspicious_threshold": 0.35,
        "high_risk_threshold": 0.62,
        "use_conformal": True,
        "conformal_alpha": 0.1,
    },
    "acoustic": {
        "strf_weight": 0.45,
        "prototype_weight": 0.55,
        "history_frames": 30,
    },
    "linguistic": {
        "window_seconds": 45.0,
        "pattern_weight": 0.55,
        "discourse_weight": 0.45,
    },
}


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    # deep copy nested
    import copy
    cfg = copy.deepcopy(DEFAULTS)
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if yaml is None:
        raise ImportError("PyYAML is required to load config files")
    with path.open("r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    for section, values in user.items():
        if section in cfg and isinstance(cfg[section], dict) and isinstance(values, dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg


def pipeline_config_from_dict(cfg: Dict[str, Any]) -> PipelineConfig:
    audio = cfg.get("audio", {})
    fusion = cfg.get("fusion", {})
    profile_name = audio.get("channel_profile", "narrowband")
    try:
        profile = CodecProfile(profile_name)
    except ValueError:
        profile = CodecProfile.NARROWBAND
    channel = ChannelConfig(profile=profile)
    return PipelineConfig(
        target_sr=int(audio.get("target_sr", 8000)),
        frame_ms=float(audio.get("frame_ms", 25.0)),
        hop_ms=float(audio.get("hop_ms", 10.0)),
        acoustic_weight=float(fusion.get("acoustic_weight", 0.40)),
        linguistic_weight=float(fusion.get("linguistic_weight", 0.60)),
        suspicious_threshold=float(fusion.get("suspicious_threshold", 0.35)),
        high_risk_threshold=float(fusion.get("high_risk_threshold", 0.62)),
        use_conformal=bool(fusion.get("use_conformal", True)),
        channel=channel,
    )
