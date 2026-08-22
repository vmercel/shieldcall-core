import numpy as np

from shieldcall import ShieldCallPipeline, PipelineConfig, __version__
from shieldcall.audio.channel import ChannelConfig, CodecProfile
from shieldcall.linguistic.asr_bridge import ScheduledTranscriptASR
from shieldcall.adaptation.hooks import ChallengeResponseProtocol
from shieldcall.config import load_config, pipeline_config_from_dict


def test_version():
    assert __version__ == "0.6.0"


def test_end_to_end_pipeline():
    sr = 8000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 170 * t)).astype(np.float32)
    schedule = [
        (0.5, "Hello from your bank"),
        (1.5, "Unusual activity detected, verify social security number"),
        (2.5, "Buy gift cards immediately, do not tell anyone"),
    ]
    pipe = ShieldCallPipeline(
        config=PipelineConfig(
            channel=ChannelConfig(profile=CodecProfile.NARROWBAND, seed=0),
            use_conformal=True,
            fuse_every_n_frames=5,
        ),
        asr=ScheduledTranscriptASR(schedule),
    )
    risks = []
    for ev in pipe.stream(audio, sr):
        if ev.risk is not None:
            risks.append(ev.risk)
    assert len(risks) > 5
    # Later in call should elevate
    assert max(r.risk_score for r in risks) > min(r.risk_score for r in risks[:3])
    assert any(r.linguistic_fraud_prob > 0.3 for r in risks)


def test_adaptation_hook():
    pipe = ShieldCallPipeline()
    feats = np.random.randn(64).astype(np.float32)
    pipe.adapt(feats, is_synthetic=True, family="test_family", source="unit_test")
    assert len(pipe.adaptation_buffer) == 1
    snap = pipe.coverage.snapshot()
    assert snap.families_known >= 1


def test_challenge_response():
    cr = ChallengeResponseProtocol(seed=1)
    ch = cr.issue()
    assert "prompt" in ch
    assert "nonce" in ch
    ch = cr.issue()
    assert cr.verify_transcript(f"the code is {ch['nonce']}") is True
    ch2 = cr.issue()
    assert cr.verify_transcript("unrelated filler text xyz") is False or str(ch2["nonce"]) in "unrelated filler text xyz"


def test_challenge_rejects_synthetic_voice():
    cr = ChallengeResponseProtocol(seed=2)
    ch = cr.issue()
    assert cr.verify(f"the code is {ch['nonce']}", acoustic_synth_prob=0.9) is False
    ch = cr.issue()
    assert cr.verify(f"the code is {ch['nonce']}", acoustic_synth_prob=0.1) is True


def test_config_load():
    cfg = load_config()
    assert "fusion" in cfg
    pcfg = pipeline_config_from_dict(cfg)
    assert pcfg.target_sr == 8000
