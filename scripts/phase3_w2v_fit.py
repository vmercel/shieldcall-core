"""Fit on wav2vec2 pooled features (train subset, held-out attacks A01/A05 excluded), evaluate val; then (--test) score test subset once."""
import _repo  # noqa
import sys, pickle, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from phase3_lib import roc_eer
L = lambda n: pickle.load(open(f"data/work/w2v_{n}.pkl", "rb"))
tr = L("train")
import os
if "--noise" in sys.argv: tr.update(L("trainnoise"))
keys = list(tr)
X = np.stack([tr[k][0] for k in keys]).astype(np.float32); y = np.array([tr[k][1] for k in keys])
def evalset(d, sc, clf, name):
    by = {}
    for k, (f, lab, atk, cond) in d.items():
        by.setdefault(cond, ([], []))[0].append(lab); by[cond][1].append(clf.decision_function(sc.transform(f.astype(np.float32)[None]))[0])
    r = {c: roc_eer(*v) for c, v in by.items()}; r["MEAN"] = float(np.mean(list(r.values()))); return r
va = L("val")
best = None
for C in (0.001, 0.01, 0.1):
    sc = StandardScaler().fit(X); clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X), y)
    r = evalset(va, sc, clf, "val"); print("C", C, "VAL", round(r["MEAN"], 4), {c: round(v, 3) for c, v in r.items() if c != "MEAN"}, flush=True)
    if best is None or r["MEAN"] < best[0]: best = (r["MEAN"], C, sc, clf, r)
json.dump(dict(C=best[1], val=best[4]), open("data/work/w2v_val_result.json", "w"))
if "--test" in sys.argv:
    te = L("test"); r = evalset(te, best[2], best[3], "test"); print("TEST", round(r["MEAN"], 4), {c: round(v, 3) for c, v in r.items() if c != "MEAN"})
    json.dump(dict(recipe="wav2vec2-base layers 3/6/9/12 mean+std pooling, logistic regression, C=%s" % best[1], test=r), open("docs/results/phase3_test_w2v" + ("_noise" if "--noise" in sys.argv else "") + ".json", "w"), indent=1)
