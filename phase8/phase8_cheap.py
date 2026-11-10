#!/usr/bin/env python3
"""Cheap-strategy tests on cached 2019 features (train/dev only). LR head for speed. Same selection score as phase8_fit."""
import sys, numpy as np
from phase8_lib import *
from phase8_fit import load, dev_report
FD = sys.argv[1] if len(sys.argv) > 1 else "/home/user/scratch/f8b"
tr = load(FD, "w2v2-base_train_t1_c4"); dv = load(FD, "w2v2-base_dev_t1_c4")
L = tr["F"].shape[1]; D = tr["F"].shape[2] // 2
def run(name, Ftr, Fdv, layers, C=0.01):
    M = fit_head("lr", Ftr, tr["y"], layers, C=C); d = dev_report(score_head(M, Fdv), dv)
    lt = ~np.isin(tr["attack"], ["A05", "A06"]); ld = ~np.isin(dv["attack"], ["A01", "A02", "A03", "A04"])
    M2 = fit_head("lr", Ftr[lt], tr["y"][lt], layers, C=C); s2 = score_head(M2, Fdv[ld]); y2 = dv["y"][ld]
    lo = eer(s2[y2 == 1], s2[y2 == 0])
    print(f"{name:34s} clean {d['clean']*100:5.2f} seen {d['seen']*100:5.2f} held {d['held']*100:5.2f} LOAO {lo*100:5.2f} score {(0.5*d['mean3']+0.5*lo)*100:5.2f}", flush=True)
B = list(range(1, 7))
run("B_early(1-6) reference", tr["F"], dv["F"], B)
for k in (2, 3, 4, 5): run(f"truncate at layer {k}", tr["F"], dv["F"], list(range(1, k + 1)))
run("layers 0-6", tr["F"], dv["F"], list(range(0, 7)))
def inorm(F, which):
    F = F.astype(np.float32).copy(); m, s = F[..., :D], F[..., D:]
    if which in ("mean", "both"): m -= m.mean(-1, keepdims=True); m /= m.std(-1, keepdims=True) + 1e-6
    if which in ("std", "both"): s -= s.mean(-1, keepdims=True); s /= s.std(-1, keepdims=True) + 1e-6
    return np.concatenate([m, s], -1)
for w in ("mean", "std", "both"): run(f"per-utt dim-z ({w})", inorm(tr["F"][:, :7], w), inorm(dv["F"][:, :7], w), B)
Fm = lambda F: F[..., :D]; Fs = lambda F: F[..., D:]
run("mean part only", Fm(tr["F"]), Fm(dv["F"]), B); run("std part only", Fs(tr["F"]), Fs(dv["F"]), B)
def cm(F):  # remove per-utterance cross-layer mean (channel offset shared across layers)
    F = F.astype(np.float32); return F - F[:, 1:7].mean(1, keepdims=True)
run("minus cross-layer mean", cm(tr["F"]), cm(dv["F"]), B)
