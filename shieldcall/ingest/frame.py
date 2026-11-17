"""Canonical media frame envelope for the Media Ingest Fabric (TDD v2.0 §4).

Every adapter translates its provider's wire format into this envelope
before the audio reaches the pipeline. One frame is 10 ms of mono audio
at the provider's native sample rate (8000 Hz for telephony adapters),
so downstream code never has to know which provider a call came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_MS = 10
"""One canonical frame is 10 ms of audio."""

FRAME_SAMPLES_8K = 80
"""Samples per 10 ms frame at the 8 kHz telephony rate."""


@dataclass(frozen=True)
class MediaFrame:
    """One 10 ms frame of decoded mono audio, with its provenance.

    - call_id: the ShieldCall call this frame belongs to (the
      pre-opened call id, which the operator sets to the provider's
      call identifier, e.g. the Twilio Call SID).
    - stream_sid: the provider's stream identifier, or "" when the
      provider has none.
    - track: which leg the audio came from ("inbound" is the remote
      caller; "outbound" is the local leg). Tracks are never mixed:
      an adapter emits frames per track and the caller picks one.
    - seq: per-track frame counter, starting at 0, in emission order.
    - t_sec: presentation time of the frame's first sample, in seconds
      from the start of the stream (the provider's own clock).
    - samples: float32 mono PCM in [-1, 1], exactly FRAME_MS long.
    - sample_rate: native rate of the samples (8000 for telephony).
    - discontinuous: True when a sequence gap preceded this frame
      (lost packets upstream). The audio is real; the flag says the
      timeline jumped, so a consumer that assumes continuity must not
      silently bridge it.
    - source: adapter that produced the frame ("twilio", "asterisk",
      ...). Recorded for audit, never used for branching.
    """

    call_id: str
    stream_sid: str
    track: str
    seq: int
    t_sec: float
    samples: np.ndarray
    sample_rate: int = 8000
    discontinuous: bool = False
    source: str = ""
