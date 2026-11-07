"""Named test conditions for the benchmark. Real codecs (PyAV/FFmpeg) where marked REAL; others are simulations."""
from __future__ import annotations
import zlib
import numpy as np
from .lfcc import resample
from . import real_codecs as rc
from ..audio.channel import TelephonyChannelTwin, ChannelConfig, CodecProfile, _bandlimit, _add_noise, _packet_loss

# name -> kind. 'seen' conditions are used to augment the telephony-trained model; 'unseen' are test-only.
CONDITIONS = {
    "clean_16k":      dict(kind="none", seen=True),
    "opus_wb_24k":    dict(kind="REAL", codec="opus_wb_24k", seen=False),
    "bandlimit_nb":   dict(kind="sim", seen=True),
    "g711_ulaw":      dict(kind="REAL", codec="g711_ulaw", seen=True),
    "g711_alaw":      dict(kind="REAL", codec="g711_alaw", seen=False),
    "amr_nb_12k2":    dict(kind="REAL", codec="amr_nb_12k2", seen=True),
    "amr_nb_4k75":    dict(kind="REAL", codec="amr_nb_4k75", seen=False),
    "opus_nb_12k":    dict(kind="REAL", codec="opus_nb_12k", seen=True),
    "ulaw_snr20":     dict(kind="REAL+noise", codec="g711_ulaw", snr=20.0, seen=True),
    "ulaw_snr10":     dict(kind="REAL+noise", codec="g711_ulaw", snr=10.0, seen=False),
    "ulaw_snr0":      dict(kind="REAL+noise", codec="g711_ulaw", snr=0.0, seen=False),
    "ulaw_loss10":    dict(kind="REAL+loss", codec="g711_ulaw", loss=0.10, seen=False),
    "sim_harsh_voip": dict(kind="sim", profile=CodecProfile.HARSH_VOIP, seen=False),
    "sim_pstn":       dict(kind="sim", profile=CodecProfile.DEGRADED_PSTN, seen=False),
}
SEEN = [k for k, v in CONDITIONS.items() if v["seen"] and k != "clean_16k"]


def apply(name: str, x16: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    c = CONDITIONS[name]
    rng = np.random.RandomState(seed)
    if c["kind"] == "none":
        return x16.astype(np.float32), 16000
    if name == "bandlimit_nb":
        x8 = resample(x16, 16000, 8000)
        return _bandlimit(x8, 8000).astype(np.float32), 8000
    if c["kind"] == "sim":
        x8 = resample(x16, 16000, 8000)
        y = TelephonyChannelTwin(ChannelConfig(profile=c["profile"], seed=seed)).apply(x8, 8000)
        return y, 8000
    y, sr = rc.roundtrip(x16, 16000, c["codec"])
    if "snr" in c:
        y = np.clip(_add_noise(y, c["snr"], rng), -1, 1).astype(np.float32)
    if "loss" in c:
        y = _packet_loss(y, sr, c["loss"], 20.0, rng).astype(np.float32)
    return y, sr


def seed_for(utt: str, cond: str) -> int:
    return zlib.crc32(f"{utt}|{cond}".encode()) & 0x7FFFFFFF
