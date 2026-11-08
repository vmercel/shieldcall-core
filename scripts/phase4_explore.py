import _repo  # noqa
import pickle, numpy as np, json, sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from phase3_lib import roc_eer
L = lambda n: pickle.load(open(f"data/work/w2v_{n}.pkl", "rb"))
tr = L("train"); tr.update(L("trainnoise")); va = L("val")
def arr(d):
    k = list(d); return np.stack([d[i][0] for i in k]).astype(np.float32), np.array([d[i][1] for i in k]), np.array([d[i][3] for i in k])
Xt, yt, ct = arr(tr); Xv, yv, cv = arr(va)
def ev(Xt_, Xv_, C=0.01, tag=""):
    sc = StandardScaler().fit(Xt_); clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xt_), yt)
    s = clf.decision_function(sc.transform(Xv_)); r = {c: roc_eer(yv[cv == c], s[cv == c]) for c in sorted(set(cv))}
    m = float(np.mean(list(r.values()))); print(f"{tag:34s} MEAN {m:.4f}", {c[:6]: round(v, 3) for c, v in r.items()}, flush=True); return m
D = 1536  # per layer: 768 mean + 768 std ; layers 3,6,9,12
sl = lambda ls, parts=("m", "s"): np.concatenate([np.arange(i * D + (0 if p == "m" else 768), i * D + (768 if p == "m" else 1536)) for i in ls for p in parts])
ev(Xt, Xv, tag="all layers m+s (ref)")
for ls in ([0], [1], [2], [3], [0, 1], [1, 2], [2, 3]):
    idx = sl(ls); ev(Xt[:, idx], Xv[:, idx], tag=f"layers {[3,6,9,12][0] if False else [[3,6,9,12][i] for i in ls]}")
for p in ("m", "s"):
    idx = sl([0, 1, 2, 3], (p,)); ev(Xt[:, idx], Xv[:, idx], tag=f"all layers {p} only")
