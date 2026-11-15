"""Phase 10 shared code: RawBoost-style waveform distortions, a bank of six proxy-attack families, 8-layer frame features, temporal head with optional one-class loss.
Reuses Phase 8 audio, channel and metric code. Every function is seeded."""
import os, sys, json, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase8"))
from phase8_lib import *            # decode, prep, channel, Backbone, pseudo_spoof, eer, SEEN, HELD, SR ...

CAP_S = 3.0; CAP = int(CAP_S * SR)
POOL = 3                            # wav2vec2 50 fps -> 16.7 fps (3 frames averaged)
T_FR = 50                           # 3 s
GD = 768                            # one layer group = mean of two transformer layers
NG = 4                              # groups: L1-2, L3-4, L5-6, L7-8
DIN = GD * NG
FAMS = ["PS0", "PS1", "PS2", "PS3", "PS4", "HF"]       # proxy attack families: 0 Griffin-Lim, 1 coarse GL, 2 random phase, 3 LPC source-filter, 4 mu-law+lowpass, HF neural vocoder
FOLD_HELD = {"A": {"PS3", "PS4", "HF"}, "B": {"PS0", "PS1", "PS2"}}
SEL_EXCLUDE = "ulaw_snr0"


class FrameBackbone(Backbone):
    def frames(self, x):
        x = (x - x.mean()) / (x.std() + 1e-6)
        ng = (len(self.m.encoder.layers)) // 2
        with self.torch.no_grad():
            hs = self.m(self.torch.from_numpy(x)[None], output_hidden_states=True).hidden_states
            f = self.torch.cat([(hs[2 * k + 1] + hs[2 * k + 2]) / 2 for k in range(ng)], -1)[0]
            n = f.shape[0] // POOL
            f = f.mean(0, keepdim=True) if n == 0 else f[: POOL * n].reshape(n, POOL, -1).mean(1)
        f = f.numpy(); n = min(len(f), T_FR)
        out = np.zeros((T_FR, GD * ng), np.float16); out[:n] = f[:n]
        return out, n


# ------------------------------------------------------------------ RawBoost-style distortions (convolutive + non-linear, impulsive, stationary coloured noise)
def rawboost(x, seed):
    from scipy.signal import firwin2, lfilter, butter, sosfilt
    rng = np.random.RandomState(seed % (2 ** 31)); algo = rng.randint(1, 7); y = x.astype(np.float64).copy(); pk = np.abs(y).max() + 1e-8
    def conv(y):
        for _ in range(rng.randint(1, 4)):
            knots = np.linspace(0, 1, 9); g = 10 ** (rng.uniform(-12, 4, 9) / 20)
            y = lfilter(firwin2(63, knots, g), [1.0], y)
        y = y / (np.abs(y).max() + 1e-8)
        for i in (2, 3):
            y = y + rng.uniform(-0.25, 0.25) * y ** i
        return y * pk / (np.abs(y).max() + 1e-8)
    def isd(y):
        m = rng.rand(len(y)) < rng.uniform(0.0, 0.08)
        return y + m * rng.randn(len(y)) * y.std() * rng.uniform(0.5, 3.0)
    def ssi(y):
        lo, hi = np.sort(rng.uniform(80, 7500, 2)); hi = max(hi, lo + 300)
        nz = sosfilt(butter(4, [lo, min(hi, 7900)], btype="band", fs=SR, output="sos"), rng.randn(len(y)))
        snr = rng.uniform(10, 40); return y + nz * (y.std() / (nz.std() + 1e-8)) * 10 ** (-snr / 20)
    for a in {1: [conv], 2: [isd], 3: [ssi], 4: [conv, isd], 5: [isd, ssi], 6: [conv, isd, ssi]}[algo]: y = a(y)
    y = np.nan_to_num(y); m = np.abs(y).max()
    if m > 1: y = y / m
    return y.astype(np.float32)


_voc = None
def hifigan_resynth(x):
    """Copy-synthesis with a pretrained neural vocoder (microsoft/speecht5_hifigan, 16 kHz, 80-band log10 mel). Published model, used as a proxy attack."""
    global _voc
    import torch, librosa
    if _voc is None:
        from transformers import SpeechT5HifiGan
        _voc = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval()
    S = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=80, fmax=7600, power=1.0)
    m = np.log10(np.maximum(S, 1e-5)).T.astype(np.float32)
    with torch.no_grad(): y = _voc(torch.from_numpy(m)[None])[0].numpy()
    y = np.nan_to_num(y); mx = np.abs(y).max()
    if mx > 0: y = y * (np.abs(x).max() / mx)
    return np.clip(y, -1, 1).astype(np.float32)


def view_audio(x, view, utt):
    """view: 'clean_16k' | channel | 'RB|<ch>' (RawBoost then channel) | '<FAM>|<ch>' (proxy attack family then channel)"""
    if "|" in view:
        kind, ch = view.split("|")
        if kind == "RB": g = rawboost(x, zlib.crc32((utt + ch).encode()))
        elif kind == "HF": g = hifigan_resynth(x)
        else: g = pseudo_spoof(x, int(kind[2:]), zlib.crc32(utt.encode()) % 9973)
        return g if ch == "clean_16k" else channel(g, ch, utt + kind)
    return x if view == "clean_16k" else channel(x, view, utt)


# ------------------------------------------------------------------ temporal head
def build_head(cfg):
    import torch
    nn = torch.nn; din = GD * cfg["groups"]

    class Block(nn.Module):
        def __init__(s, h, dil, drop):
            super().__init__(); s.ln = nn.LayerNorm(h); s.c = nn.Conv1d(h, h, 3, padding=dil, dilation=dil); s.l = nn.Linear(h, h); s.d = nn.Dropout(drop)
        def forward(s, x, m):
            y = s.ln(x) * m[..., None]; y = torch.nn.functional.gelu(s.c(y.transpose(1, 2)).transpose(1, 2)); return x + s.d(s.l(y))

    class Head(nn.Module):
        def __init__(s):
            super().__init__(); h = cfg["h"]; drop = cfg["drop"]; s.oc = cfg["loss"] == "oc"
            s.inp = nn.Sequential(nn.Dropout(drop), nn.Linear(din, h))
            s.blocks = nn.ModuleList([Block(h, (1, 2, 4, 8)[i], drop) for i in range(cfg["blocks"])])
            s.att = nn.Sequential(nn.Linear(h, 64), nn.Tanh(), nn.Linear(64, 1))
            s.out = nn.Sequential(nn.Linear(2 * h, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 64 if s.oc else 1))
            if s.oc: s.center = nn.Parameter(torch.randn(64))
            s.pooled = nn.Sequential(nn.Dropout(0.5), nn.Linear(2 * din, 1)) if cfg.get("pooled") else None
        def forward(s, x, m):                       # x (B,T,din) normalised, m (B,T) float
            h = s.inp(x) * m[..., None]
            for b in s.blocks: h = b(h, m) * m[..., None]
            a = s.att(h)[..., 0].masked_fill(m == 0, -1e4).softmax(1)[..., None]
            mu = (a * h).sum(1); sd = ((a * (h - mu[:, None]) ** 2).sum(1) + 1e-5).sqrt(); z = s.out(torch.cat([mu, sd], 1))
            if s.oc: return torch.nn.functional.cosine_similarity(z, s.center[None], dim=1)       # one-class score in [-1, 1], higher = bona fide
            o = z[:, 0]
            if s.pooled is not None:                # equal-weight second expert: mean+std of the input features, trained jointly
                n = m.sum(1, keepdim=True).clamp(min=1); pm = (x * m[..., None]).sum(1) / n; pv = (((x - pm[:, None]) ** 2) * m[..., None]).sum(1) / n
                o = o + s.pooled(torch.cat([pm, (pv + 1e-5).sqrt()], 1))[:, 0]
            return o
    return Head()


def oc_loss(s, y, m0=0.9, m1=0.2, alpha=20.0):
    import torch
    F = torch.nn.functional
    return torch.where(y > 0.5, F.softplus(alpha * (m0 - s)), F.softplus(alpha * (s - m1)))


def lengths_mask(T, n=T_FR):
    return (np.arange(n)[None, :] < np.asarray(T)[:, None]).astype(np.float32)


def predict(models, mu, sd, X, T, bs=256, device="cpu"):
    import torch
    din = len(mu); out = []; mu_t = torch.from_numpy(mu).to(device); sd_t = torch.from_numpy(sd).to(device)
    for m in models: m.eval().to(device)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = (torch.from_numpy(np.asarray(X[i:i + bs][..., :din], dtype=np.float32)).to(device) - mu_t) / sd_t
            mk = torch.from_numpy(lengths_mask(T[i:i + bs])).to(device); x = x * mk[..., None]
            out.append(np.mean([m(x, mk).cpu().numpy() for m in models], 0))
    return np.concatenate(out)


def save_ckpt(path, models, cfg, mu, sd, meta):
    import torch
    meta = dict(meta, cfg=cfg); torch.save({"states": [m.state_dict() for m in models], "cfg": cfg, "mu": mu, "sd": sd, "meta": meta}, path)


def load_ckpt(path):
    import torch
    z = torch.load(path, map_location="cpu", weights_only=False); ms = []
    for st in z["states"]:
        m = build_head(z["cfg"]); m.load_state_dict(st); ms.append(m.eval())
    return ms, z["mu"], z["sd"], z["meta"]
