"""Channel Passport (TDD v2.0 section 5; ADR-007).

Ground truth is synthetic audio with known properties, so every
expected value below is derived, not asserted from the code under test.
Nothing here touches licensed ASVspoof data.

The test signal is a burst-voiced signal: 200 ms harmonic bursts
separated by 100 ms gaps over a low noise floor, which is what the
passport's energy-percentile SNR estimate needs (real calls have
pauses; a pause-free tone measures dynamic range, not SNR).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from shieldcall.acoustic.passport import (
    ChannelPassport,
    PassportVector,
    calibrated_llr,
    companding_signature,
    effective_bandwidth,
    estimate_passport,
    estimate_snr,
    gate_weights,
    noise_floor_modulation,
    support_score,
)
from shieldcall.audio.channel import (
    _add_noise,
    _bandlimit,
    _ulaw_encode_decode,
)
from shieldcall.fusion.calibration import PerCellCalibrator

SR = 16000


def _burst_voiced(seed: int = 0, seconds: float = 2.0) -> np.ndarray:
    """Harmonic bursts (200 ms on, 100 ms off) over a -40 dB noise floor."""
    t = np.arange(int(seconds * SR)) / SR
    burst = ((t % 0.3) < 0.2).astype(float)
    voiced = sum(np.sin(2.0 * np.pi * 120 * k * t) / (k ** 0.7) for k in range(1, 60))
    voiced = voiced / np.max(np.abs(voiced)) * 0.5
    rng = np.random.RandomState(seed)
    floor = rng.randn(len(t)) * 0.005
    return (voiced * burst + floor).astype(np.float32)


# ---------------------------------------------------------------------------
# Individual measurements
# ---------------------------------------------------------------------------

def test_effective_bandwidth_wideband_vs_bandlimited():
    sig = _burst_voiced()
    assert effective_bandwidth(sig, SR) > 6000.0
    narrow = _bandlimit(sig, SR)
    assert 3000.0 < effective_bandwidth(narrow, SR) < 3800.0


def test_companding_signature_mulaw_roundtrip():
    sig = _burst_voiced()
    law, score = companding_signature(sig)
    assert law == "none"
    assert score < 0.01
    ulaw = _ulaw_encode_decode(sig)
    law, score = companding_signature(ulaw)
    assert law == "mulaw"
    assert score > 0.99


def test_companding_signature_alaw_roundtrip():
    from shieldcall.acoustic.passport import _alaw_roundtrip

    sig = _burst_voiced()
    alaw = _alaw_roundtrip(sig).astype(np.float32)
    law, score = companding_signature(alaw)
    assert law == "alaw"
    assert score > 0.99


def test_companding_signature_no_evidence_on_silence():
    law, score = companding_signature(np.zeros(SR, dtype=np.float32))
    assert law == "none"
    assert score == 0.0


def test_snr_tracks_added_noise():
    sig = _burst_voiced()
    assert estimate_snr(sig, SR) > 20.0
    for target in (20.0, 10.0, 5.0):
        noisy = _add_noise(sig, target, np.random.RandomState(1))
        assert estimate_snr(noisy, SR) == pytest.approx(target, abs=4.0)


def test_clipping_fraction():
    sig = _burst_voiced()
    assert estimate_passport(sig, SR).clipping_fraction == 0.0
    clipped = np.clip(sig * 3.0, -1.0, 1.0).astype(np.float32)
    assert estimate_passport(clipped, SR).clipping_fraction > 0.0


def test_noise_floor_modulation_steady_vs_gated():
    t = np.arange(2 * SR) / SR
    burst = ((t % 0.3) < 0.2).astype(float)
    voiced = sum(np.sin(2.0 * np.pi * 120 * k * t) / (k ** 0.7) for k in range(1, 60))
    voiced = voiced / np.max(np.abs(voiced)) * 0.5
    rng = np.random.RandomState(0)
    steady = (voiced * burst + rng.randn(len(t)) * 0.005).astype(np.float32)
    # Gated floor: gap noise alternates between near-zero and loud per block.
    gap_level = np.where(((t % 0.6) < 0.3), 0.001, 0.02)
    gated = (voiced * burst + rng.randn(len(t)) * gap_level).astype(np.float32)
    assert noise_floor_modulation(steady, SR) < 0.3
    assert noise_floor_modulation(gated, SR) > 0.6


def test_loss_from_envelope_counts():
    sig = _burst_voiced()
    p = estimate_passport(sig, SR, packets_total=100, packets_lost=10, packets_concealed=2)
    assert p.loss_rate == pytest.approx(0.12)
    assert p.loss_observed is True
    p = estimate_passport(sig, SR)
    assert p.loss_rate == 0.0
    assert p.loss_observed is False


# ---------------------------------------------------------------------------
# Cell mapping
# ---------------------------------------------------------------------------

def test_cell_declared_opus_wideband():
    p = estimate_passport(_burst_voiced(), SR, declared_codec="opus")
    assert p.cell == "opus_wideband"


def test_cell_declared_g711():
    p = estimate_passport(_ulaw_encode_decode(_burst_voiced()), SR, declared_codec="g711")
    assert p.cell == "narrowband_g711"


def test_cell_declared_amr():
    p = estimate_passport(_burst_voiced(), SR, declared_codec="amr_nb")
    assert p.cell == "amr_nb"


def test_cell_loss_and_snr_take_precedence_over_declared_codec():
    sig = _burst_voiced()
    p = estimate_passport(sig, SR, declared_codec="opus", packets_total=100, packets_lost=12)
    assert p.cell == "high_loss"
    noisy = _add_noise(sig, 5.0, np.random.RandomState(1))
    p = estimate_passport(noisy, SR, declared_codec="opus")
    assert p.cell == "low_snr"


def test_cell_pstn_from_bandwidth_and_companding_evidence():
    # Real PSTN order: analog bandlimit first, then mu-law encode, so
    # the received audio stays on the mu-law grid.
    pstn = _ulaw_encode_decode(_bandlimit(_burst_voiced(), SR))
    p = estimate_passport(pstn, SR)
    assert p.cell == "pstn_degraded"


def test_cell_clean_wideband_undeclared():
    p = estimate_passport(_burst_voiced(), SR)
    assert p.cell == "clean_wideband"


def test_cell_unknown_when_bandlimited_without_companding():
    p = estimate_passport(_bandlimit(_burst_voiced(), SR), SR)
    assert p.cell == "unknown"


# ---------------------------------------------------------------------------
# Support, abstention, gate weights
# ---------------------------------------------------------------------------

def test_support_high_on_reference_like_passports():
    p = estimate_passport(_burst_voiced(), SR, declared_codec="opus")
    assert p.rho > 0.8
    assert not p.is_out_of_support()
    pstn = _ulaw_encode_decode(_bandlimit(_burst_voiced(), SR))
    p = estimate_passport(pstn, SR)
    assert p.rho > 0.5


def test_support_low_and_abstain_far_outside_training_channels():
    weird = PassportVector(
        effective_bandwidth_hz=200.0,
        snr_db=-5.0,
        loss_rate=0.5,
        companding_score=0.0,
    )
    weird.rho = support_score(weird)
    assert weird.rho < 0.1
    assert weird.is_out_of_support()
    # The threshold is tunable: a mid-support passport (rho between
    # 0.05 and 0.25) abstains at the default 0.25 but not at 0.05.
    mid = PassportVector(
        effective_bandwidth_hz=2200.0,
        snr_db=30.0,
        loss_rate=0.0,
        companding_score=0.0,
    )
    mid.rho = support_score(mid)
    assert 0.05 < mid.rho < 0.25
    assert mid.is_out_of_support()
    assert mid.is_out_of_support(rho_min=0.05) is False


def test_gate_weights_sum_to_one_and_argmax_matches_cell():
    p = estimate_passport(_burst_voiced(), SR, declared_codec="opus")
    weights = gate_weights(p)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert max(weights, key=weights.get) == "opus_wideband"


def test_to_dict_is_json_safe():
    p = estimate_passport(_burst_voiced(), SR, declared_codec="opus")
    record = json.loads(json.dumps(p.to_dict()))
    assert record["cell"] == "opus_wideband"
    assert record["out_of_support"] is False


# ---------------------------------------------------------------------------
# Calibration feed and streaming
# ---------------------------------------------------------------------------

def test_calibrated_llr_uses_passport_cell():
    rng = np.random.RandomState(0)
    cells = {
        "clean_wideband": (
            np.r_[rng.normal(1.0, 1.0, 100), rng.normal(-1.0, 1.0, 100)],
            np.r_[np.ones(100), np.zeros(100)],
        ),
        "pstn_degraded": (
            np.r_[rng.normal(3.0, 1.0, 100), rng.normal(1.0, 1.0, 100)],
            np.r_[np.ones(100), np.zeros(100)],
        ),
    }
    calibrator = PerCellCalibrator().fit(cells)
    clean = PassportVector(cell="clean_wideband")
    pstn = PassportVector(cell="pstn_degraded")
    llr_clean = calibrated_llr(clean, calibrator, 0.0)
    llr_pstn = calibrated_llr(pstn, calibrator, 0.0)
    assert llr_clean == pytest.approx(calibrator.llr("clean_wideband", 0.0))
    assert llr_pstn == pytest.approx(calibrator.llr("pstn_degraded", 0.0))
    # Same raw score, different cells: the PSTN map must shift the LLR
    # by more than half a bit, or per-cell calibration does nothing.
    assert abs(llr_clean - llr_pstn) > 0.5
    # An unknown cell falls back to the pooled map.
    unknown = PassportVector(cell="unknown")
    a, b = calibrator.pooled
    assert calibrated_llr(unknown, calibrator, 0.0) == pytest.approx(b)


def test_streaming_passport_emits_one_vector_per_window():
    passport = ChannelPassport(declared_codec="opus")
    sig = _burst_voiced(seconds=1.2)
    chunk = int(0.2 * SR)
    emitted = []
    for start in range(0, len(sig), chunk):
        out = passport.update(sig[start : start + chunk], SR)
        if out is not None:
            emitted.append(out)
    assert len(emitted) == 2
    assert all(p.cell == "opus_wideband" for p in emitted)
    assert all(p.declared_codec == "opus" for p in emitted)


def test_streaming_passport_carries_packet_counts():
    passport = ChannelPassport()
    passport.note_packets(total=100, lost=12)
    sig = _burst_voiced(seconds=0.5)
    out = passport.update(sig, SR)
    assert out is not None
    assert out.cell == "high_loss"
    assert out.loss_observed is True


def test_short_window_does_not_crash():
    # A 100 ms window is below the 500 ms design cadence, so the SNR
    # estimate is noisy and the cell is whatever the measurements say.
    # The contract is only that it returns a well-formed vector.
    p = estimate_passport(_burst_voiced(seconds=0.1), SR)
    assert p.cell in (
        "clean_wideband",
        "opus_wideband",
        "narrowband_g711",
        "pstn_degraded",
        "amr_nb",
        "low_snr",
        "high_loss",
        "unknown",
    )
    assert 0.0 <= p.rho <= 1.0
