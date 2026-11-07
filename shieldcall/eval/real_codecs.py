"""Real codec round trips via PyAV (bundled FFmpeg). Optional dependency: `pip install av`.

These are actual encoders/decoders (G.711 mu-law/A-law, AMR-NB, Opus, G.722), not the
numpy caricatures in ``audio/channel.py``. Input/Output: float32 mono in [-1, 1].
"""
from __future__ import annotations

import io
from fractions import Fraction

import numpy as np

# name -> (container, codec, codec_sr, bitrate or None)
CODECS = {
    "g711_ulaw": ("wav", "pcm_mulaw", 8000, None),
    "g711_alaw": ("wav", "pcm_alaw", 8000, None),
    "amr_nb_12k2": ("amr", "libopencore_amrnb", 8000, 12200),
    "amr_nb_4k75": ("amr", "libopencore_amrnb", 8000, 4750),
    "opus_nb_12k": ("ogg", "libopus", 8000, 12000),
    "opus_wb_24k": ("ogg", "libopus", 16000, 24000),
    "g722_64k": ("wav", "g722", 16000, 64000),
}


def available() -> bool:
    try:
        import av  # noqa: F401
        return True
    except Exception:
        return False


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def roundtrip(x: np.ndarray, sr: int, name: str) -> tuple[np.ndarray, int]:
    """Encode+decode with codec ``name``. Returns (audio, sample_rate_out)."""
    import av
    container, codec, csr, br = CODECS[name]
    x = np.clip(_resample(np.asarray(x, np.float32), sr, csr), -1, 1)
    pcm = (x * 32767).astype(np.int16)
    buf = io.BytesIO()
    out = av.open(buf, "w", format=container)
    st = out.add_stream(codec, rate=csr)
    st.layout = "mono"
    if br:
        st.bit_rate = br
    frame_size = st.codec_context.frame_size or 960
    pos = 0
    pts = 0
    while pos < len(pcm):
        chunk = pcm[pos:pos + frame_size]
        if len(chunk) < frame_size and st.codec_context.frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        fr = av.AudioFrame.from_ndarray(chunk[None, :], format="s16", layout="mono")
        fr.sample_rate = csr
        fr.pts = pts
        fr.time_base = Fraction(1, csr)
        for p in st.encode(fr):
            out.mux(p)
        pos += len(chunk)
        pts += len(chunk)
    for p in st.encode(None):
        out.mux(p)
    out.close()
    buf.seek(0)
    inp = av.open(buf, "r")
    dec = []
    for fr in inp.decode(audio=0):
        a = fr.to_ndarray()
        if a.ndim > 1:
            a = a.mean(axis=0) if a.shape[0] <= 2 else a.reshape(-1)
        if a.dtype.kind == "i":
            a = a.astype(np.float32) / float(np.iinfo(a.dtype).max + 1)
        dec.append(a.astype(np.float32))
    sr_out = inp.streams.audio[0].rate
    inp.close()
    y = np.concatenate(dec) if dec else np.zeros(1, np.float32)
    y = y.astype(np.float32)
    if float(np.abs(y).max()) > 1.5:  # some decoders return int16-valued floats
        y = y / 32768.0
    if sr_out != csr:  # decoders such as Opus emit 48 kHz; return at the codec's nominal rate
        y = _resample(y, int(sr_out), csr)
        sr_out = csr
    return y, int(sr_out)
