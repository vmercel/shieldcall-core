#!/usr/bin/env python3
"""Reduced wav2vec2-base feature extraction (CPU). Pools mean+std over time for layers 3,6,9,12 -> 6144 dims.
All views are decoded from the cached int16 views (so channel conditions are identical to other iterations) and upsampled to 16 kHz.
Subsets (seeded): train 2400 views (A02-A04,A06 + genuine; A01/A05 held out), val 4 conds x 600 clips, test 4 conds x 800 clips."""
import _repo  # noqa
import sys, pickle, time, random
import numpy as np, torch
torch.backends.nnpack.set_flags(False)
from transformers import Wav2Vec2Model
from phase3_lib import load, decode
from shieldcall.eval.lfcc import resample
which = sys.argv[1]; LAYERS = (3, 6, 9, 12); MAXS = 40000
m = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval(); rng = random.Random(7)
if which == "train":
    V = load("data/work/views_train.pkl")
    g = [k for k, v in V.items() if v["label"] == 0]; s = [k for k, v in V.items() if v["label"] == 1 and v["attack"] not in ("A01", "A05")]
    rng.shuffle(g); rng.shuffle(s); keys = g[:1200] + s[:1200]
elif which == "val":
    V = load("data/work/views_val.pkl"); C = ("clean_16k", "g711_ulaw", "amr_nb_12k2", "ulaw_snr0")
    utts = sorted({k[0] for k in V}); rng.shuffle(utts); keep = set(utts[:600]); keys = [k for k in V if k[1] in C and k[0] in keep]
else:
    V = load("data/work/views_test.pkl"); C = ("clean_16k", "g711_ulaw", "amr_nb_12k2", "ulaw_snr0")
    utts = sorted({k[0] for k in V}); rng.shuffle(utts); keep = set(utts[:800]); keys = [k for k in V if k[1] in C and k[0] in keep]
out = {}; t0 = time.time()
for i, k in enumerate(keys):
    x, sr = decode(V[k]); x = resample(x, sr, 16000)[:MAXS]
    x = (x - x.mean()) / (x.std() + 1e-6)
    with torch.no_grad():
        hs = m(torch.from_numpy(x)[None], output_hidden_states=True).hidden_states
    f = np.concatenate([np.r_[h[0].mean(0).numpy(), h[0].std(0).numpy()] for h in (hs[l] for l in LAYERS)])
    out[k] = (f.astype(np.float16), V[k]["label"], V[k]["attack"], V[k]["cond"])
    if i % 200 == 0:
        print(which, i, len(keys), round(time.time() - t0), flush=True)
pickle.dump(out, open(f"data/work/w2v_{which}.pkl", "wb"))
print("done", which, len(out), round(time.time() - t0))
