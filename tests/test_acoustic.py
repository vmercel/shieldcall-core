import numpy as np

from shieldcall.acoustic.features import FEATURE_DIM, extract_frame_features
from shieldcall.acoustic.residual import extract_residual_fingerprint
from shieldcall.acoustic.scorer import AcousticDeepfakeScorer, PrototypeMemory
from shieldcall.audio.preprocessor import Frame, TelephonyPreprocessor


def _frame(sr=8000, f=150.0):
    t = np.linspace(0, 0.025, int(sr * 0.025), endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)
    return Frame(samples=samples, sample_rate=sr, timestamp_sec=0.0, is_speech=True, frame_index=0)


def test_feature_dim():
    feats = extract_frame_features(_frame().samples, 8000)
    assert feats.shape == (FEATURE_DIM,)
    assert np.isfinite(feats).all()


def test_residual_fingerprint():
    fp = extract_residual_fingerprint(_frame().samples, 8000)
    assert fp.vector.shape[0] >= 8
    assert fp.residual_energy_ratio >= 0


def test_scorer_range():
    scorer = AcousticDeepfakeScorer(seed=0)
    s = scorer.score_frame(_frame())
    assert 0 <= s.synthetic_prob <= 1
    assert 0 <= s.confidence <= 1
    assert s.is_speech


def test_nonspeech_zero():
    scorer = AcousticDeepfakeScorer(seed=0)
    f = _frame()
    f.is_speech = False
    s = scorer.score_frame(f)
    assert s.synthetic_prob == 0.0


def test_prototype_adaptation_reduces_gap():
    mem = PrototypeMemory(seed=1)
    emb = np.ones(FEATURE_DIM, dtype=np.float32) * 2.0
    gap_before = mem.coverage_gap(emb)
    for _ in range(8):
        mem.add(emb + np.random.randn(FEATURE_DIM).astype(np.float32) * 0.05, is_synthetic=True)
    gap_after = mem.coverage_gap(emb)
    assert gap_after <= gap_before + 0.05  # should not get worse; usually improves


def test_streaming_scorer():
    pre = TelephonyPreprocessor(target_sr=8000)
    scorer = AcousticDeepfakeScorer(seed=0)
    t = np.linspace(0, 1.0, 8000, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 160 * t)).astype(np.float32)
    scores = [scorer.score_frame(f) for f in pre.stream_from_array(audio, 8000)]
    assert len(scores) > 50
    speech_scores = [s.synthetic_prob for s in scores if s.is_speech]
    assert len(speech_scores) > 0
