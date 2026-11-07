import numpy as np
import pytest

from shieldcall.eval.lfcc import lfcc, trim_silence
from shieldcall.eval import real_codecs


def test_lfcc_shape_and_cmvn():
    x = (np.random.RandomState(0).randn(16000) * 0.1).astype(np.float32)
    f = lfcc(x, 16000)
    assert f.shape[1] == 60 and abs(f.mean()) < 1e-3


def test_trim_silence_removes_edges():
    x = np.zeros(16000, np.float32)
    x[6000:10000] = np.random.RandomState(1).randn(4000) * 0.2
    y = trim_silence(x, 16000)
    assert len(y) < 8000


@pytest.mark.skipif(not real_codecs.available(), reason="PyAV not installed")
def test_real_codec_roundtrip_rates():
    x = (0.2 * np.sin(2 * np.pi * 300 * np.arange(16000) / 16000)).astype(np.float32)
    y, sr = real_codecs.roundtrip(x, 16000, "g711_ulaw")
    assert sr == 8000 and 7000 < len(y) < 9000 and float(np.abs(y).max()) <= 1.0
