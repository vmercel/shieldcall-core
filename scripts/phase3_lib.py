"""Shared helpers for Phase 3 iterations."""
import _repo  # noqa
import pickle, time, json
import numpy as np
from collections import defaultdict
from shieldcall.eval.features2 import build

HELD = ("A01", "A05")
ACC_MARGIN = 0.003   # an iteration is accepted only if validation mean EER improves by > 0.3 points


def roc_eer(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    o = np.argsort(-s, kind="mergesort"); ys = y[o]
    P, N = ys.sum(), len(ys) - ys.sum()
    tpr = np.r_[0, np.cumsum(ys) / P]; fpr = np.r_[0, np.cumsum(1 - ys) / N]
    i = np.argmin(np.abs((1 - tpr) - fpr))
    return float(((1 - tpr)[i] + fpr[i]) / 2)


def load(path):
    return pickle.load(open(path, "rb"))


def decode(v):
    return v["a"].astype(np.float32) / 32767.0, v["sr"]


def featurize(views, sets, keys=None):
    out = {}
    for k in (keys or views.keys()):
        x, sr = decode(views[k]); out[k] = build(x, sr, sets)
    return out


def val_eval(score_fn, vviews, vfeats):
    """score_fn(F)->float. Returns dict cond->EER and mean. Positive = spoof (held-out attacks A01/A05)."""
    by = defaultdict(lambda: ([], []))
    for k, v in vviews.items():
        s = score_fn(vfeats[k]); by[v["cond"]][0].append(v["label"]); by[v["cond"]][1].append(s)
    res = {c: roc_eer(*by[c]) for c in by}
    res["MEAN"] = float(np.mean([res[c] for c in by]))
    return res
