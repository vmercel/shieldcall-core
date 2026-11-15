"""Phase 9 shared code: frame-level SSL features, temporal head, neural-vocoder pseudo-spoofs. Reuses Phase 8 audio/channel/metric code."""
import os, sys, json, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase8"))
from phase8_lib import *            # decode, prep, channel, gl_resynth, Backbone, eer, SEEN, HELD, SR ...

CAP_S = 3.0; CAP = int(CAP_S * SR)
T_FR = 75                           # wav2vec2 gives 50 frames/s; adjacent frames are averaged -> 25 frames/s, 75 frames for 3 s
DIN = 1536                          # two layer groups: mean(L1,L2) | mean(L3,L4), 768 each
SEL_EXCLUDE = "ulaw_snr0"           # stress test only, as in Phase 8


class FrameBackbone(Backbone):
    def frames(self, x):
        x = (x - x.mean()) / (x.std() + 1e-6)
        with self.torch.no_grad():
            hs = self.m(self.torch.from_numpy(x)[None], output_hidden_states=True).hidden_states
            f = self.torch.cat([(hs[1] + hs[2]) / 2, (hs[3] + hs[4]) / 2], -1)[0]
            n = f.shape[0] // 2
            f = f.mean(0, keepdim=True) if n == 0 else f[: 2 * n].reshape(n, 2, -1).mean(1)
        f = f.numpy(); n = min(len(f), T_FR)
        out = np.zeros((T_FR, DIN), np.float16); out[:n] = f[:n]
        return out, n


_voc = None
def hifigan_resynth(x):
    """Pseudo-spoof: copy-synthesis with a pretrained neural vocoder (microsoft/speecht5_hifigan, 16 kHz, 80-band log10 mel). A published idea, not ours."""
    global _voc
    import torch, librosa
    if _voc is None:
        from transformers import SpeechT5HifiGan
        _voc = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval()
    S = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=80, fmax=7600, power=1.0)
    m = np.log10(np.maximum(S, 1e-5)).T.astype(np.float32)
    with torch.no_grad():
        y = _voc(torch.from_numpy(m)[None])[0].numpy()
    y = np.nan_to_num(y); mx = np.abs(y).max()
    if mx > 0: y = y * (np.abs(x).max() / mx)
    return np.clip(y, -1, 1).astype(np.float32)


def view_audio(x, view, utt):
    """view: 'clean_16k' | channel name | 'GL|<ch>' | 'HF|<ch>'"""
    if "|" in view:
        kind, ch = view.split("|")
        g = gl_resynth(x) if kind == "GL" else hifigan_resynth(x)
        return g if ch == "clean_16k" else channel(g, ch, utt + kind)
    return x if view == "clean_16k" else channel(x, view, utt)


# ------------------------------------------------------------------ temporal head
def build_head(cfg):
    import torch
    nn = torch.nn

    class Block(nn.Module):
        def __init__(s, h, dil, drop):
            super().__init__(); s.ln = nn.LayerNorm(h); s.c = nn.Conv1d(h, h, 3, padding=dil, dilation=dil); s.l = nn.Linear(h, h); s.d = nn.Dropout(drop)
        def forward(s, x, m):
            y = s.ln(x) * m[..., None]; y = torch.nn.functional.gelu(s.c(y.transpose(1, 2)).transpose(1, 2)); return x + s.d(s.l(y))

    class Head(nn.Module):
        def __init__(s, din=DIN, h=cfg["h"], blocks=cfg["blocks"], drop=cfg["drop"]):
            super().__init__()
            s.inp = nn.Sequential(nn.Dropout(drop), nn.Linear(din, h))
            s.blocks = nn.ModuleList([Block(h, (1, 2, 4, 8)[i], drop) for i in range(blocks)])
            s.att = nn.Sequential(nn.Linear(h, 64), nn.Tanh(), nn.Linear(64, 1))
            s.out = nn.Sequential(nn.Linear(2 * h, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))
        def forward(s, x, m):                       # x (B,T,D) normalised, m (B,T) float 1/0
            h = s.inp(x) * m[..., None]
            for b in s.blocks: h = b(h, m) * m[..., None]
            a = s.att(h)[..., 0].masked_fill(m == 0, -1e4).softmax(1)[..., None]
            mu = (a * h).sum(1); sd = ((a * (h - mu[:, None]) ** 2).sum(1) + 1e-5).sqrt()
            return s.out(torch.cat([mu, sd], 1))[:, 0]
    return Head()


def lengths_mask(T, n=T_FR):
    return (np.arange(n)[None, :] < np.asarray(T)[:, None]).astype(np.float32)


def predict(models, mu, sd, X, T, bs=256, device="cpu"):
    import torch
    out = []
    mu_t = torch.from_numpy(mu).to(device); sd_t = torch.from_numpy(sd).to(device)
    for m in models: m.eval().to(device)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = (torch.from_numpy(np.asarray(X[i:i + bs], dtype=np.float32)).to(device) - mu_t) / sd_t
            mk = torch.from_numpy(lengths_mask(T[i:i + bs])).to(device); x = x * mk[..., None]
            out.append(np.mean([m(x, mk).cpu().numpy() for m in models], 0))
    return np.concatenate(out)


def save_ckpt(path, models, cfg, mu, sd, meta):
    import torch
    torch.save({"states": [m.state_dict() for m in models], "cfg": cfg, "mu": mu, "sd": sd, "meta": meta}, path)


def load_ckpt(path):
    import torch
    z = torch.load(path, map_location="cpu", weights_only=False)
    ms = []
    for st in z["states"]:
        m = build_head(z["cfg"]); m.load_state_dict(st); ms.append(m.eval())
    return ms, z["mu"], z["sd"], z["meta"]
