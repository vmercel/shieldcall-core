"""Combined lexicon lock: stage emissions + narrow keyword groups."""

from __future__ import annotations

import hashlib
import json


def current_lexicon_lock() -> str:
    """sha256[:16] of frozen STAGE_EMISSIONS and PATTERN_GROUPS."""
    from .discourse import STAGE_EMISSIONS
    from .scorer import PATTERN_GROUPS

    payload = {
        "stage_emissions": STAGE_EMISSIONS,
        "pattern_groups": PATTERN_GROUPS,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def stage_emissions_lock() -> str:
    from .discourse import STAGE_EMISSIONS

    blob = json.dumps(STAGE_EMISSIONS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
