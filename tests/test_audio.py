import numpy as np

from shieldcall.audio.channel import ChannelConfig, CodecProfile, TelephonyChannelTwin
from shieldcall.audio.preprocessor import TelephonyPreprocessor
from shieldcall.audio.vad import StreamingVAD


def _tone(sr=8000, dur=0.5, f=180.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_preprocessor_frames():
    pre = TelephonyPreprocessor(target_sr=8000)
    frames = list(pre.stream_from_array(_tone(), 8000, chunk_ms=100))
    assert len(frames) > 10
    assert frames[0].samples.shape[0] == pre.frame_len
    assert frames[0].sample_rate == 8000


def test_resample_16k_to_8k():
    pre = TelephonyPreprocessor(target_sr=8000)
    x = _tone(sr=16000, dur=0.5)
    frames = pre.push(x, 16000)
    assert len(frames) > 0
    assert frames[0].sample_rate == 8000


def test_channel_profiles_change_signal():
    x = _tone()
    clean = TelephonyChannelTwin(ChannelConfig(profile=CodecProfile.CLEAN, seed=0)).apply(x, 8000)
    harsh = TelephonyChannelTwin(ChannelConfig(profile=CodecProfile.HARSH_VOIP, seed=0)).apply(x, 8000)
    assert clean.shape == x.shape
    assert harsh.shape == x.shape
    # Harsh should differ from clean
    assert np.mean((clean - harsh) ** 2) > 1e-6


def test_vad_detects_tone_vs_silence():
    vad = StreamingVAD()
    speech = vad.decide(_tone(dur=0.025))
    silence = vad.decide(np.zeros(200, dtype=np.float32))
    assert speech.energy > silence.energy
