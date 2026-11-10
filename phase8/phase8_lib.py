"""Phase 8 shared code: audio, channel views, backbone features, heads, metrics. Everything is seeded and deterministic."""
import io, os, sys, zlib, random, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # repo root, for shieldcall.eval.*

BACKBONES = {                                       # name -> (hf id, n hidden states)
    "w2v2-base": ("facebook/wav2vec2-base", 13),
    "wavlm-base-plus": ("microsoft/wavlm-base-plus", 13),
    "xlsr-300m": ("facebook/wav2vec2-xls-r-300m", 25),
}
SR = 16000
SEEN = ["g711_ulaw", "g711_alaw", "amr_nb_12k2", "opus_nb_12k", "ulaw_snr20", "ulaw_snr10", "bandlimit_nb", "g722_64k", "sim_pstn", "sim_harsh_voip"]  # training channels
HELD = ["amr_nb_4k75", "opus_wb_24k", "ulaw_loss10", "ulaw_snr0"]       # never trained on; used to judge dev (ulaw_snr0 is reported as a stress test, not selected on)
ALLC = {}


def conds():
    from shieldcall.eval import conditions as C
    if "g722_64k" not in C.CONDITIONS:
        C.CONDITIONS["g722_64k"] = {"kind": "REAL", "codec": "g722_64k", "seen": True}
    return C


def sid(s, mod):
    return zlib.crc32(s.encode()) % mod


# ------------------------------------------------------------------ audio
def decode(b):
    try:
        import soundfile as sf
        x, sr = sf.read(io.BytesIO(b), dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
    except Exception:
        import av
        c = av.open(io.BytesIO(b)); st = c.streams.audio[0]; ch = []
        for fr in c.decode(st):
            a = fr.to_ndarray(); f = np.issubdtype(a.dtype, np.integer); a = a.astype(np.float32)
            if f: a /= 32768.0
            ch.append(a.mean(0) if a.ndim > 1 and a.shape[0] <= 8 else a.ravel())
        x, sr = np.concatenate(ch), st.rate
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x.astype(np.float32)


def prep(x, trim, cap):
    """The one place where silence policy and length cap are applied, for train, dev and eval alike."""
    if trim:
        from shieldcall.eval.lfcc import trim_silence
        x = trim_silence(x, SR)
    return x[:cap]


def gl_resynth(x, seed=0):
    """Pseudo-spoof: Griffin-Lim copy-synthesis of a bona fide clip (80-band mel, 1024 FFT, hop 256, 32 iterations)."""
    import librosa
    S = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=256, n_mels=80)
    y = librosa.feature.inverse.mel_to_audio(S, sr=SR, n_fft=1024, hop_length=256, n_iter=32)
    return np.clip(y, -1, 1).astype(np.float32)


def channel(x, cond, utt):
    C = conds()
    y, sr = C.apply(cond, x, C.seed_for(utt, cond))
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, SR); y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


# ------------------------------------------------------------------ backbone
class Backbone:
    def __init__(self, name, threads=0, max_layer=0):
        import torch
        from transformers import AutoModel
        self.torch = torch
        torch.set_num_threads(threads or (os.cpu_count() or 2))
        try: torch.backends.nnpack.set_flags(False)
        except Exception: pass
        self.name = name; self.m = AutoModel.from_pretrained(BACKBONES[name][0]).eval()
        if max_layer:                                   # truncate the transformer: cheaper, and early layers carry most of the spoofing cue
            self.m.encoder.layers = self.m.encoder.layers[:max_layer]

    def __call__(self, x):
        """x float32 16 kHz -> (n_layers, 2*D) float16: mean and std over time of every hidden state."""
        x = (x - x.mean()) / (x.std() + 1e-6)
        with self.torch.no_grad():
            hs = self.m(self.torch.from_numpy(x)[None], output_hidden_states=True).hidden_states
        return np.stack([np.r_[h[0].mean(0).numpy(), h[0].std(0).numpy()] for h in hs]).astype(np.float16)


def layer_sets(L):
    return {"A_every3": list(range(3, L, 3)), "B_early": list(range(1, L // 2 + 1)),
            "C_all": list(range(L)), "D_mid": list(range(max(1, L // 6), int(L * 0.65) + 1))}


# ------------------------------------------------------------------ metrics
def eer(bona, spoof):
    s = np.r_[bona, spoof]; y = np.r_[np.ones(len(bona)), np.zeros(len(spoof))]
    o = np.argsort(s, kind="stable"); y = y[o]
    frr = np.cumsum(y) / y.sum(); far = 1 - np.cumsum(1 - y) / (1 - y).sum()
    i = np.argmin(np.abs(frr - far)); return float((frr[i] + far[i]) / 2)


# ------------------------------------------------------------------ heads (numpy inference so the eval needs no sklearn)
def _std(X, mu=None, sd=None):
    if mu is None:
        mu = X.mean(0); sd = X.std(0) + 1e-6
    return (X - mu) / sd, mu, sd


def flat(F, layers):
    return F[:, layers, :].reshape(len(F), -1).astype(np.float32)


def fit_head(kind, F, y, layers, seed=0, C=0.01):
    """F: (n, L, 2D) float16, y: 1 = bona fide. Returns a dict of arrays; score_head() applies it. Higher = bona fide."""
    from sklearn.linear_model import LogisticRegression
    M = {"kind": kind, "layers": np.array(layers)}
    if kind == "lr":
        X, mu, sd = _std(flat(F, layers))
        c = LogisticRegression(C=C, max_iter=400, class_weight="balanced").fit(X, y)
        M.update(mu=mu, sd=sd, w=c.coef_[0].astype(np.float32), b=np.float32(c.intercept_[0]))
    elif kind == "committee":
        mus, sds, ws, bs = [], [], [], []
        for l in layers:
            X, mu, sd = _std(F[:, l, :].astype(np.float32))
            c = LogisticRegression(C=C, max_iter=300, class_weight="balanced").fit(X, y)
            z = X @ c.coef_[0] + c.intercept_[0]
            mus.append(mu); sds.append(sd); ws.append(c.coef_[0] / max(z.std(), 1e-6)); bs.append(c.intercept_[0] / max(z.std(), 1e-6))
        M.update(mu=np.stack(mus), sd=np.stack(sds), w=np.stack(ws).astype(np.float32), b=np.array(bs, dtype=np.float32))
    elif kind == "mlp":
        import torch
        torch.manual_seed(seed); torch.set_num_threads(os.cpu_count() or 2)
        X, mu, sd = _std(flat(F, layers)); Xt = torch.from_numpy(X); yt = torch.from_numpy(y.astype(np.float32))
        W1, B1, W2, B2 = [], [], [], []
        for s in range(5):                                           # 5-seed ensemble
            torch.manual_seed(seed * 100 + s)
            net = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(128, 1))
            opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.05)
            pw = torch.tensor((1 - y.mean()) / y.mean(), dtype=torch.float32)
            for ep in range(12):
                net.train(); perm = torch.randperm(len(Xt))
                for i in range(0, len(perm), 128):
                    j = perm[i:i + 128]; opt.zero_grad()
                    torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[j])[:, 0], yt[j], pos_weight=pw).backward(); opt.step()
            net.eval(); l1, l2 = net[1], net[4]
            W1.append(l1.weight.detach().numpy().T); B1.append(l1.bias.detach().numpy()); W2.append(l2.weight.detach().numpy()[0]); B2.append(l2.bias.detach().numpy()[0])
        M.update(mu=mu, sd=sd, W1=np.stack(W1), B1=np.stack(B1), W2=np.stack(W2), B2=np.array(B2))
    return M


def score_head(M, F):
    k = str(M["kind"]); layers = list(M["layers"])
    if k == "lr":
        return ((flat(F, layers) - M["mu"]) / M["sd"]) @ M["w"] + M["b"]
    if k == "committee":
        out = 0
        for i, l in enumerate(layers):
            out = out + ((F[:, l, :].astype(np.float32) - M["mu"][i]) / M["sd"][i]) @ M["w"][i] + M["b"][i]
        return out / len(layers)
    if k == "mlp":
        X = (flat(F, layers) - M["mu"]) / M["sd"]; o = 0
        for s in range(len(M["W2"])):
            o = o + np.maximum(X @ M["W1"][s] + M["B1"][s], 0) @ M["W2"][s] + M["B2"][s]
        return o / len(M["W2"])
    raise ValueError(k)


def save_model(path, M, meta):
    d = {f"m_{k}": v for k, v in M.items()}; d["meta"] = np.array(json.dumps(meta)); np.savez_compressed(path, **d)


def load_model(path):
    z = np.load(path, allow_pickle=False)
    M = {k[2:]: z[k] for k in z.files if k.startswith("m_")}; M["kind"] = str(M["kind"])
    return M, json.loads(str(z["meta"]))


# ------------------------------------------------------------------ cheap pseudo-spoof families (signal processing only; no extra data, no neural vocoder)
def _istft_ola(S, hop, n):
    import librosa
    return librosa.istft(S, hop_length=hop, length=n)


def pseudo_spoof(x, fam, seed=0):
    """fam 0 Griffin-Lim 80-mel (as before); 1 Griffin-Lim coarse (64 mel, hop 128, 8 iters); 2 random-phase STFT resynthesis;
    3 source-filter vocoder (LPC envelope + pulse/noise excitation); 4 low-bit mu-law quantisation + 7 kHz lowpass."""
    import librosa
    from scipy.signal import lfilter, butter, sosfilt
    rng = np.random.RandomState(seed)
    if fam == 0: return gl_resynth(x)
    if fam == 1:
        S = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=128, n_mels=64)
        y = librosa.feature.inverse.mel_to_audio(S, sr=SR, n_fft=1024, hop_length=128, n_iter=8)
    elif fam == 2:
        S = librosa.stft(x, n_fft=512, hop_length=128); ph = np.exp(1j * rng.uniform(-np.pi, np.pi, S.shape))
        y = _istft_ola(np.abs(S) * ph, 128, len(x))
    elif fam == 3:
        fl, hp = 400, 160; n = len(x); y = np.zeros(n + fl)
        f0 = librosa.yin(x, fmin=70, fmax=400, sr=SR, frame_length=1024, hop_length=hp) if n > 1024 else np.full(n // hp + 1, 0.0)
        w = np.hanning(fl); phase = 0.0
        for k, s in enumerate(range(0, max(n - fl, 1), hp)):
            fr = x[s:s + fl] * w
            if len(fr) < fl or fr.std() < 1e-5: continue
            try: a = librosa.lpc(fr + 1e-7 * rng.randn(fl), order=16)
            except Exception: continue
            res = lfilter(a, [1.0], fr); g = res.std() + 1e-8
            period = SR / max(float(f0[min(k, len(f0) - 1)]), 70.0)
            exc = np.zeros(fl); voiced = (res.std() > 0.02 * (np.abs(x).max() + 1e-8)) and 70 <= f0[min(k, len(f0) - 1)] <= 400
            if voiced:
                t = phase
                while t < fl: exc[int(t)] = 1.0; t += period
                phase = t - fl; exc *= g * np.sqrt(period)
            else: exc = rng.randn(fl) * g
            y[s:s + fl] += lfilter([1.0], a, exc) * w
        y = y[:n]
    else:
        q = 2 ** 6; mu = 255.0; xc = np.clip(x / (np.abs(x).max() + 1e-8), -1, 1)
        c = np.sign(xc) * np.log1p(mu * np.abs(xc)) / np.log1p(mu); c = np.round(c * q) / q
        y = np.sign(c) * (np.expm1(np.abs(c) * np.log1p(mu)) / mu)
        y = sosfilt(butter(6, 7000, fs=SR, output="sos"), y)
    y = np.nan_to_num(y); m = np.abs(y).max()
    if m > 0: y = y * (np.abs(x).max() / m)
    return np.clip(y, -1, 1).astype(np.float32)


def qfeat(x):
    """Cheap handcrafted features: mean and std of LFCC and delta-LFCC (80 dims). Costs milliseconds."""
    from shieldcall.eval.lfcc import lfcc
    c = np.asarray(lfcc(x, SR), dtype=np.float32)
    if c.ndim != 2 or c.shape[0] < 3: return np.zeros(c.shape[-1] * 4 if c.ndim == 2 else 80, np.float32)
    d = np.diff(c, axis=0)
    return np.concatenate([c.mean(0), c.std(0), d.mean(0), d.std(0)]).astype(np.float32)
