"""Phase 4: truncated-wav2vec2 (4 layers) features, optional spectral-gate denoise.
usage: phase4_extract.py {val|train|noise} {0|1}   (1 = denoise)"""
import _repo  # noqa
import pickle, random, sys, time
import numpy as np, torch
torch.backends.nnpack.set_flags(False)
from phase3_lib import load, decode
from shieldcall.eval.lfcc import resample
from shieldcall.audio.channel import _add_noise
from shieldcall.eval import ssl_front as S
which, dn = sys.argv[1], bool(int(sys.argv[2])); out = {}; t0 = time.time()
def emb(x, sr):
    return S.embed(resample(x, sr, 16000).astype(np.float32), 4, dn)
if which in ("val", "test"):
    V = load(f"data/work/views_{which}.pkl"); keys = list(pickle.load(open(f"data/work/w2v_{which}.pkl", "rb")))
    for i, k in enumerate(keys):
        x, sr = decode(V[k]); out[k] = (emb(x, sr), V[k]["label"], V[k]["attack"], V[k]["cond"])
        if i % 400 == 0: print(i, len(keys), round(time.time() - t0), flush=True)
elif which == "train":
    V = load("data/work/views_train.pkl"); keys = list(pickle.load(open("data/work/w2v_train.pkl", "rb")))
    for i, k in enumerate(keys):
        x, sr = decode(V[k]); out[k] = (emb(x, sr), V[k]["label"], V[k]["attack"], V[k]["cond"])
        if i % 400 == 0: print(i, len(keys), round(time.time() - t0), flush=True)
else:
    rng = random.Random(23); V = load("data/work/views_train.pkl")
    base = [k for k, v in V.items() if v["cond"] in ("g711_ulaw", "clean_16k", "amr_nb_12k2") and v["attack"] not in ("A01", "A05")]
    g = [k for k in base if V[k]["label"] == 0]; s = [k for k in base if V[k]["label"] == 1]; rng.shuffle(g); rng.shuffle(s)
    keys = g[:1500] + s[:1500]
    for i, k in enumerate(keys):
        x, sr = decode(V[k]); snr = rng.choice([0., 3., 6., 10., 15., 20.])
        x = np.clip(_add_noise(x, snr, np.random.RandomState(1000 + i)), -1, 1).astype(np.float32)
        out[(k, "m%d" % snr)] = (emb(x, sr), V[k]["label"], V[k]["attack"], "noise%d" % snr)
        if i % 400 == 0: print(i, len(keys), round(time.time() - t0), flush=True)
pickle.dump(out, open(f"data/work/p4_{which}_{int(dn)}.pkl", "wb")); print("done", which, dn)
