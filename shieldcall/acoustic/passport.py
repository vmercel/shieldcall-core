"""Channel Passport: blind channel estimation (TDD v2.0 section 5; ADR-007).

Every 500 ms the passport emits one :class:`PassportVector` describing
how the audio travelled: the declared codec from the envelope (exact
when present), the effective bandwidth (which catches upstream
transcoding the declared codec hides), a companding signature, the loss
and concealment rate from the envelope, an SNR from
voice-activity-separated frame energy, a clipping fraction, and a
processing cue from the noise floor's modulation in pauses.

The vector maps to a discrete channel cell plus a continuous support
score ``rho`` in [0, 1], the density of the vector under the training
distribution of channels. The passport feeds three consumers (TDD
section 5.2): per-cell calibration (ADR-005) through
:func:`calibrated_llr`, the channel-expert mixture through
:func:`gate_weights`, and abstention through :meth:`PassportVector.is_out_of_support`.

Honest limits, stated where they bite rather than hidden:

* Loss is observable only from the envelope. Without packet counts the
  rate is 0.0 and ``loss_observed`` is False.
* The SNR is an energy-percentile estimate, not a true VAD measurement.
* Echo-canceller residue needs a far-end reference signal, which a
  single-channel passport does not have; it is out of scope. The
  noise-floor modulation cue is the observable proxy TDD section 5.1
  names for noise-suppressor gating.
* The support reference points are nominal values from the TDD and the
  Phase 8 conditions, not fitted to data. Fitting them on real channel
  traffic is future work; see ADR-007.
* Nothing here touches a live decision path: the passport reports a
  cell, a support score and an abstention flag. Callers decide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cadence at which the passport emits a vector (TDD section 5.1).
WINDOW_SECONDS = 0.5

#: Default abstention threshold (TDD section 5.2 item 3). A placeholder
#: pending calibration on real channel traffic; see ADR-007.
DEFAULT_RHO_MIN = 0.25

#: Loss rate at or above which a call is a high-loss call first.
HIGH_LOSS_THRESHOLD = 0.05

#: SNR below which a call is a low-SNR call first.
LOW_SNR_THRESHOLD_DB = 10.0

#: Bandwidth boundaries for the cell rules (Hz).
NARROWBAND_BANDWIDTH_HZ = 4200.0
WIDEBAND_BANDWIDTH_HZ = 6000.0

#: Samples with |x| above this count as active for the companding test;
#: near-silent samples carry no quantisation evidence either way.
COMPANDING_ACTIVE_FLOOR = 0.02

#: Minimum active samples for a companding verdict; fewer means "no
#: evidence", not a guessed law.
COMPANDING_MIN_SAMPLES = 100

#: Tolerance for "the round trip reproduced this sample exactly".
COMPANDING_EXACT_TOL = 1e-6

#: Companding score at or above which the audio counts as companded
#: for the cell rules.
COMPANDED_SCORE_THRESHOLD = 0.5

#: Frame length for the spectral / energy measurements.
FRAME_SECONDS = 0.02

#: Clipping threshold on the [-1, 1] full-scale range.
CLIPPING_THRESHOLD = 0.999

# Cells (TDD section 5 example set).
CELL_CLEAN_WIDEBAND = "clean_wideband"
CELL_OPUS_WIDEBAND = "opus_wideband"
CELL_NARROWBAND_G711 = "narrowband_g711"
CELL_PSTN_DEGRADED = "pstn_degraded"
CELL_AMR_NB = "amr_nb"
CELL_LOW_SNR = "low_snr"
CELL_HIGH_LOSS = "high_loss"
CELL_UNKNOWN = "unknown"

KNOWN_CELLS = (
    CELL_CLEAN_WIDEBAND,
    CELL_OPUS_WIDEBAND,
    CELL_NARROWBAND_G711,
    CELL_PSTN_DEGRADED,
    CELL_AMR_NB,
    CELL_LOW_SNR,
    CELL_HIGH_LOSS,
)

#: Nominal reference points per known cell, as feature vectors
#: ``(log10(effective_bandwidth_hz), snr_db, loss_rate,
#: companding_score)``. Bandwidth is logged to match the feature
#: vector (see ``_feature_vector``). Nominal values from the TDD and
#: the Phase 8 conditions, NOT fitted to data. A cell may hold several
#: points spanning its plausible range (e.g. high loss at 5%, 15% and
#: 30%).
def _ref(bw_hz: float, snr: float, loss: float, comp: float) -> Tuple[float, float, float, float]:
    return (math.log10(bw_hz), snr, loss, comp)


REFERENCE_POINTS: Dict[str, List[Tuple[float, float, float, float]]] = {
    CELL_CLEAN_WIDEBAND: [_ref(7500.0, 35.0, 0.0, 0.0)],
    CELL_OPUS_WIDEBAND: [_ref(7000.0, 30.0, 0.01, 0.0), _ref(6500.0, 25.0, 0.03, 0.0)],
    CELL_NARROWBAND_G711: [_ref(3400.0, 30.0, 0.0, 0.95), _ref(3400.0, 35.0, 0.0, 0.9)],
    CELL_PSTN_DEGRADED: [
        _ref(3400.0, 12.0, 0.10, 0.95),
        _ref(3400.0, 15.0, 0.15, 0.9),
        _ref(3400.0, 20.0, 0.05, 0.9),
    ],
    CELL_AMR_NB: [_ref(3400.0, 25.0, 0.01, 0.2)],
    CELL_LOW_SNR: [
        _ref(3400.0, 5.0, 0.01, 0.5),
        _ref(3400.0, 8.0, 0.02, 0.5),
        _ref(7000.0, 5.0, 0.01, 0.0),
        _ref(7000.0, 8.0, 0.02, 0.0),
    ],
    CELL_HIGH_LOSS: [
        _ref(3400.0, 25.0, 0.05, 0.5),
        _ref(3400.0, 25.0, 0.15, 0.5),
        _ref(3400.0, 20.0, 0.30, 0.5),
        _ref(7000.0, 25.0, 0.15, 0.0),
    ],
}

#: Kernel scales for the support score, in feature units:
#: (log10 bandwidth dex, SNR dB, loss rate, companding score).
#: Companding is deliberately weak (scale 1.0): post-companding line
#: noise moves samples off the quantisation grid, so the exactness the
#: signature relies on is erased on noisy lines. It still matters for
#: the cell rule (a threshold), but it must not push an otherwise
#: ordinary call out of support.
SUPPORT_SCALES = (0.12, 8.0, 0.05, 1.0)

#: Declared-codec spellings normalised to cells. Anything unlisted is
#: not mapped to a cell: the passport never claims a codec it did not see.
_DECLARED_CODEC_CELLS = {
    "opus": CELL_OPUS_WIDEBAND,
    "opus_wb": CELL_OPUS_WIDEBAND,
    "opus-wb": CELL_OPUS_WIDEBAND,
    "opus_nb": CELL_UNKNOWN,  # narrowband Opus is not the wideband cell
    "g711": CELL_NARROWBAND_G711,
    "g.711": CELL_NARROWBAND_G711,
    "pcmu": CELL_NARROWBAND_G711,
    "pcma": CELL_NARROWBAND_G711,
    "ulaw": CELL_NARROWBAND_G711,
    "alaw": CELL_NARROWBAND_G711,
    "amr": CELL_AMR_NB,
    "amr_nb": CELL_AMR_NB,
    "amr-nb": CELL_AMR_NB,
    "pstn": CELL_PSTN_DEGRADED,
}


# ---------------------------------------------------------------------------
# Low-level measurements
# ---------------------------------------------------------------------------

def _frame_energies(x: np.ndarray, frame_len: int) -> np.ndarray:
    """Mean squared energy per non-overlapping frame (last partial dropped)."""
    n_frames = len(x) // frame_len
    if n_frames == 0:
        return np.empty(0)
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.mean(frames ** 2, axis=1)


def effective_bandwidth(x: np.ndarray, sr: int) -> float:
    """B-hat: the frequency below which 99% of long-term spectral energy lies.

    TDD eq. 1. The long-term spectrum is the mean periodogram over
    20 ms Hann-windowed frames.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0 or sr <= 0:
        return 0.0
    frame_len = max(1, int(round(FRAME_SECONDS * sr)))
    n_frames = len(x) // frame_len
    if n_frames == 0:
        frame_len = len(x)
        n_frames = 1
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    window = np.hanning(frame_len)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mean_spec = np.mean(spec, axis=0)
    total = float(np.sum(mean_spec))
    if total <= 0.0:
        return 0.0
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
    cum = np.cumsum(mean_spec) / total
    idx = int(np.searchsorted(cum, 0.99))
    return float(freqs[min(idx, len(freqs) - 1)])


def _mulaw_roundtrip(x: np.ndarray) -> np.ndarray:
    """G.711 mu-law compress -> 8-bit quantise -> expand (mirrors
    ``shieldcall.audio.channel._ulaw_encode_decode``)."""
    x = np.clip(x.astype(np.float64), -1.0, 1.0)
    mu = 255.0
    sign = np.sign(x)
    compressed = sign * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((compressed + 1.0) * 0.5 * 255.0)
    q = np.clip(q, 0, 255)
    dequant = (q / 255.0) * 2.0 - 1.0
    return sign * (1.0 / mu) * np.expm1(np.abs(dequant) * np.log1p(mu))


def _alaw_roundtrip(x: np.ndarray) -> np.ndarray:
    """G.711 A-law compress -> 8-bit quantise -> expand (ITU-T G.711)."""
    x = np.clip(x.astype(np.float64), -1.0, 1.0)
    a_law = 87.6
    sign = np.sign(x)
    ax = np.abs(x)
    compressed = np.where(
        ax < 1.0 / a_law,
        a_law * ax / (1.0 + math.log(a_law)),
        (1.0 + np.log(a_law * np.maximum(ax, 1e-12))) / (1.0 + math.log(a_law)),
    )
    compressed = sign * compressed
    q = np.round((compressed + 1.0) * 0.5 * 255.0)
    q = np.clip(q, 0, 255)
    dequant = (q / 255.0) * 2.0 - 1.0
    expanded = np.where(
        np.abs(dequant) < 1.0 / (1.0 + math.log(a_law)),
        np.abs(dequant) * (1.0 + math.log(a_law)) / a_law,
        np.exp(np.abs(dequant) * (1.0 + math.log(a_law)) - 1.0) / a_law,
    )
    return sign * expanded


def companding_signature(x: np.ndarray) -> Tuple[str, float]:
    """Companding law and score for ``x``.

    Returns ``(law, score)`` where ``law`` is ``"mulaw"``, ``"alaw"``
    or ``"none"`` and ``score`` is the fraction of active samples the
    winning law's round trip reproduces exactly. A signal that was
    already companded upstream re-quantises losslessly (score near 1);
    a clean float signal is moved by up to half a quantisation step
    (score near 0). Fewer than ``COMPANDING_MIN_SAMPLES`` active
    samples means no evidence: ``("none", 0.0)``.
    """
    x = np.asarray(x, dtype=np.float64)
    active = x[np.abs(x) > COMPANDING_ACTIVE_FLOOR]
    if len(active) < COMPANDING_MIN_SAMPLES:
        return "none", 0.0
    scores = {}
    for law, fn in (("mulaw", _mulaw_roundtrip), ("alaw", _alaw_roundtrip)):
        y = fn(active)
        exact = np.mean(np.abs(active - y) <= COMPANDING_EXACT_TOL)
        scores[law] = float(exact)
    law = max(scores, key=scores.get)
    if scores[law] < COMPANDED_SCORE_THRESHOLD:
        return "none", float(scores[law])
    return law, float(scores[law])


def estimate_snr(x: np.ndarray, sr: int) -> float:
    """SNR in dB from voice-activity-separated frame energy (TDD eq. 1 form).

    Noise is the mean of the lowest quintile of 20 ms frame energies;
    speech is the mean of frames more than 6 dB above the noise floor,
    falling back to the top quintile when no frame clears it. An
    energy-percentile estimate, not a true VAD measurement. Clipped to
    [-10, 60] dB; pure silence (zero noise energy) reports 60 dB.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 60.0
    frame_len = max(1, int(round(FRAME_SECONDS * sr)))
    energies = _frame_energies(x, frame_len)
    if len(energies) == 0:
        return 60.0
    ordered = np.sort(energies)
    n_noise = max(1, len(ordered) // 5)
    noise_energy = float(np.mean(ordered[:n_noise]))
    if noise_energy <= 0.0:
        return 60.0
    speech = energies[energies > 4.0 * noise_energy]
    if len(speech) < 3:
        n_speech = max(1, len(ordered) // 5)
        speech = ordered[-n_speech:]
    speech_energy = float(np.mean(speech))
    snr = 10.0 * math.log10(max(speech_energy - noise_energy, 1e-12) / noise_energy)
    return float(min(max(snr, -10.0), 60.0))


def noise_floor_modulation(x: np.ndarray, sr: int) -> float:
    """Coefficient of variation of the pause-frame energies.

    Pause frames are the 20 ms frames within 6 dB of the estimated
    noise floor (the mean of the lowest quintile of frame energies).
    The floor-relative cut keeps louder speech frames out of the set:
    a plain below-median cut mixes quiet speech with pauses and
    measures the speech envelope instead of the floor. A steady line
    noise floor scores near 0; a gated noise suppressor makes the
    pause floor bursty and scores high. 0.0 when there is no
    measurable noise floor.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    frame_len = max(1, int(round(FRAME_SECONDS * sr)))
    energies = _frame_energies(x, frame_len)
    if len(energies) < 5:
        return 0.0
    ordered = np.sort(energies)
    n_noise = max(1, len(ordered) // 5)
    noise_floor = float(np.mean(ordered[:n_noise]))
    if noise_floor <= 0.0:
        return 0.0
    pause = energies[energies < 4.0 * noise_floor]
    if len(pause) < 5:
        return 0.0
    mean = float(np.mean(pause))
    if mean <= 0.0:
        return 0.0
    return float(np.std(pause) / mean)


# ---------------------------------------------------------------------------
# Passport vector
# ---------------------------------------------------------------------------

@dataclass
class PassportVector:
    """One 500 ms Channel Passport emission (TDD section 5.1)."""

    declared_codec: Optional[str] = None
    declared_rate: Optional[int] = None
    effective_bandwidth_hz: float = 0.0
    companding_law: str = "none"
    companding_score: float = 0.0
    loss_rate: float = 0.0
    loss_observed: bool = False
    snr_db: float = 60.0
    clipping_fraction: float = 0.0
    noise_floor_modulation: float = 0.0
    cell: str = CELL_UNKNOWN
    rho: float = 0.0

    def is_out_of_support(self, rho_min: float = DEFAULT_RHO_MIN) -> bool:
        """True when the acoustic stream should abstain (TDD 5.2 item 3)."""
        return self.rho < rho_min

    def to_dict(self) -> Dict[str, object]:
        """JSON-safe record for fleet channel-mix telemetry (TDD 5.2 item 4)."""
        return {
            "declared_codec": self.declared_codec,
            "declared_rate": self.declared_rate,
            "effective_bandwidth_hz": self.effective_bandwidth_hz,
            "companding_law": self.companding_law,
            "companding_score": self.companding_score,
            "loss_rate": self.loss_rate,
            "loss_observed": self.loss_observed,
            "snr_db": self.snr_db,
            "clipping_fraction": self.clipping_fraction,
            "noise_floor_modulation": self.noise_floor_modulation,
            "cell": self.cell,
            "rho": self.rho,
            "out_of_support": self.is_out_of_support(),
        }


# ---------------------------------------------------------------------------
# Cell mapping and support
# ---------------------------------------------------------------------------

def _feature_vector(passport: PassportVector) -> Tuple[float, float, float, float]:
    return (
        math.log10(max(passport.effective_bandwidth_hz, 1.0)),
        float(passport.snr_db),
        float(passport.loss_rate),
        float(passport.companding_score),
    )


def _kernel_distance(features: Sequence[float], reference: Sequence[float]) -> float:
    """Squared normalised distance between a feature vector and a reference."""
    total = 0.0
    for value, ref, scale in zip(features, reference, SUPPORT_SCALES):
        total += ((value - ref) / scale) ** 2
    return total


def support_score(passport: PassportVector) -> float:
    """rho in [0, 1]: max Gaussian kernel value against the reference points.

    1.0 on a reference point, near 0 far from every reference point.
    The reference points are nominal (see ADR-007), not fitted.
    """
    features = _feature_vector(passport)
    best = 0.0
    for points in REFERENCE_POINTS.values():
        for ref in points:
            value = math.exp(-0.5 * _kernel_distance(features, ref))
            if value > best:
                best = value
    return float(best)


def map_cell(passport: PassportVector) -> str:
    """Discrete channel cell for a passport, by documented precedence.

    1. high loss (>= 5%), 2. low SNR (< 10 dB), 3. declared codec,
    4. bandwidth plus companding evidence, 5. unknown.
    """
    if passport.loss_rate >= HIGH_LOSS_THRESHOLD:
        return CELL_HIGH_LOSS
    if passport.snr_db < LOW_SNR_THRESHOLD_DB:
        return CELL_LOW_SNR
    if passport.declared_codec:
        cell = _DECLARED_CODEC_CELLS.get(passport.declared_codec.strip().lower())
        if cell is not None:
            return cell
    if (
        passport.effective_bandwidth_hz <= NARROWBAND_BANDWIDTH_HZ
        and passport.companding_score >= COMPANDED_SCORE_THRESHOLD
    ):
        return CELL_PSTN_DEGRADED
    if passport.effective_bandwidth_hz > WIDEBAND_BANDWIDTH_HZ:
        return CELL_CLEAN_WIDEBAND
    return CELL_UNKNOWN


def gate_weights(passport: PassportVector) -> Dict[str, float]:
    """Softmax gate weights over the known cells (TDD section 5.2 item 2).

    Weight of a cell is proportional to exp(-d^2 / 2) where d is the
    normalised distance to the cell's nearest reference point. Weights
    sum to 1; the mapped cell gets the largest weight when the passport
    sits near its reference. The experts themselves are not implemented
    here; this only supplies the gate signal the mixture needs.
    """
    features = _feature_vector(passport)
    raw: Dict[str, float] = {}
    for cell, points in REFERENCE_POINTS.items():
        nearest = min(_kernel_distance(features, ref) for ref in points)
        raw[cell] = math.exp(-0.5 * nearest)
    total = sum(raw.values())
    if total <= 0.0:
        equal = 1.0 / len(raw)
        return {cell: equal for cell in raw}
    return {cell: value / total for cell, value in raw.items()}


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def estimate_passport(
    audio: np.ndarray,
    sr: int,
    declared_codec: Optional[str] = None,
    declared_rate: Optional[int] = None,
    packets_total: Optional[int] = None,
    packets_lost: int = 0,
    packets_concealed: int = 0,
) -> PassportVector:
    """Estimate one passport vector from ``audio`` (one ~500 ms window).

    ``audio`` is mono float samples in [-1, 1]. Packet counts come from
    the frame envelope; when ``packets_total`` is None the loss rate is
    reported as unobserved (0.0, ``loss_observed`` False) rather than
    guessed from the waveform.
    """
    x = np.asarray(audio, dtype=np.float64).ravel()
    bandwidth = effective_bandwidth(x, sr)
    law, comp_score = companding_signature(x)
    snr = estimate_snr(x, sr)
    modulation = noise_floor_modulation(x, sr)
    clipping = float(np.mean(np.abs(x) >= CLIPPING_THRESHOLD)) if len(x) else 0.0

    loss_observed = packets_total is not None and packets_total > 0
    loss_rate = 0.0
    if loss_observed:
        loss_rate = float(min((packets_lost + packets_concealed) / packets_total, 1.0))

    passport = PassportVector(
        declared_codec=declared_codec,
        declared_rate=declared_rate,
        effective_bandwidth_hz=bandwidth,
        companding_law=law,
        companding_score=comp_score,
        loss_rate=loss_rate,
        loss_observed=loss_observed,
        snr_db=snr,
        clipping_fraction=clipping,
        noise_floor_modulation=modulation,
    )
    passport.cell = map_cell(passport)
    passport.rho = support_score(passport)
    return passport


def calibrated_llr(passport: PassportVector, calibrator, score: float) -> float:
    """Calibrated LLR of one raw acoustic score in the passport's cell.

    Delegates to a ``PerCellCalibrator`` (ADR-005) with
    ``passport.cell``; an unknown cell falls back to the pooled map
    through the calibrator's existing rule.
    """
    return float(calibrator.llr(passport.cell, score))


class ChannelPassport:
    """Streaming passport: accumulates audio and emits one vector per 500 ms.

    Envelope data is fed in through :meth:`note_envelope` and
    :meth:`note_packets`; the estimator never assumes it. :meth:`update`
    returns a :class:`PassportVector` when a full window is ready, else
    ``None``. Leftover audio shorter than a window is carried over.
    """

    def __init__(
        self,
        declared_codec: Optional[str] = None,
        declared_rate: Optional[int] = None,
        window_seconds: float = WINDOW_SECONDS,
    ):
        self.declared_codec = declared_codec
        self.declared_rate = declared_rate
        self.window_seconds = float(window_seconds)
        self._buffer: List[np.ndarray] = []
        self._buffered = 0
        self._packets_total = 0
        self._packets_lost = 0
        self._packets_concealed = 0
        self._packets_observed = False

    def note_envelope(
        self, declared_codec: Optional[str] = None, declared_rate: Optional[int] = None
    ) -> None:
        """Record (or update) the declared codec and rate from the envelope."""
        if declared_codec is not None:
            self.declared_codec = declared_codec
        if declared_rate is not None:
            self.declared_rate = declared_rate

    def note_packets(self, total: int, lost: int = 0, concealed: int = 0) -> None:
        """Accumulate envelope packet counts for the loss rate."""
        self._packets_total += max(int(total), 0)
        self._packets_lost += max(int(lost), 0)
        self._packets_concealed += max(int(concealed), 0)
        self._packets_observed = True

    def update(self, audio: np.ndarray, sr: int) -> Optional[PassportVector]:
        """Feed audio; emit a passport once a full window is buffered."""
        x = np.asarray(audio, dtype=np.float64).ravel()
        if len(x) == 0:
            return None
        self._buffer.append(x)
        self._buffered += len(x)
        window_len = int(round(self.window_seconds * sr))
        if self._buffered < window_len:
            return None
        joined = np.concatenate(self._buffer)
        window = joined[:window_len]
        rest = joined[window_len:]
        self._buffer = [rest] if len(rest) else []
        self._buffered = len(rest)
        packets_total = self._packets_total if self._packets_observed else None
        passport = estimate_passport(
            window,
            sr,
            declared_codec=self.declared_codec,
            declared_rate=self.declared_rate,
            packets_total=packets_total,
            packets_lost=self._packets_lost,
            packets_concealed=self._packets_concealed,
        )
        self._packets_total = 0
        self._packets_lost = 0
        self._packets_concealed = 0
        self._packets_observed = False
        return passport
