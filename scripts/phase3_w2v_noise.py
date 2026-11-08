"""Iteration 5: add low-SNR (0/5/10 dB) G.711 views to the wav2vec2 training set (900 views, A01/A05 still held out)."""
import _repo  # noqa
import pickle, random, time
import numpy as np, torch
torch.backends.nnpack.set_flags(False)
from transformers import Wav2Vec2Model
from phase3_lib import load, decode
from shieldcall.eval.lfcc import resample
from shieldcall.audio.channel import _add_noise
LAYERS = (3, 6, 9, 12); MAXS = 40000
m = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval(); rng = random.Random(11)
V = load("data/work/views_train.pkl")
base = [k for k, v in V.items() if v["cond"] in ("g711_ulaw", "clean_16k") and v["attack"] not in ("A01", "A05")]
g = [k for k in base if V[k]["label"] == 0]; s = [k for k in base if V[k]["label"] == 1]; rng.shuffle(g); rng.shuffle(s)
keys = g[:450] + s[:450]; out = {}; t0 = time.time()
for i, k in enumerate(keys):
    x, sr = decode(V[k]); snr = rng.choice([0.0, 5.0, 10.0])
    x = np.clip(_add_noise(x, snr, np.random.RandomState(i)), -1, 1).astype(np.float32)
    x = resample(x, sr, 16000)[:MAXS]; x = (x - x.mean()) / (x.std() + 1e-6)
    with torch.no_grad():
        hs = m(torch.from_numpy(x)[None], output_hidden_states=True).hidden_states
    f = np.concatenate([np.r_[hs[l][0].mean(0).numpy(), hs[l][0].std(0).numpy()] for l in LAYERS])
    out[(k, "n%d" % snr)] = (f.astype(np.float16), V[k]["label"], V[k]["attack"], "noise%d" % snr)
    if i % 200 == 0: print(i, len(keys), round(time.time() - t0), flush=True)
pickle.dump(out, open("data/work/w2v_trainnoise.pkl", "wb")); print("done")
