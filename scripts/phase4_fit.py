"""Phase 4 validation experiments on truncated-wav2vec2 features. Acceptance: mean val EER < reference - 0.003."""
import _repo  # noqa
import pickle, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from phase3_lib import roc_eer
def L(w, d): return pickle.load(open(f"data/work/p4_{w}_{d}.pkl", "rb"))
def arr(*ds):
    k = [i for d in ds for i in d]; m = {}; [m.update(d) for d in ds]
    return np.stack([m[i][0] for i in k]).astype(np.float32), np.array([m[i][1] for i in k]), np.array([m[i][3] for i in k])
def report(tag, s, yv, cv):
    r = {c: roc_eer(yv[cv == c], s[cv == c]) for c in sorted(set(cv))}; m = float(np.mean(list(r.values())))
    print(f"{tag:42s} MEAN {m:.4f}", {c[:6]: round(v, 3) for c, v in r.items()}, flush=True); return m
def lr(Xt, yt, Xv, C=0.01):
    sc = StandardScaler().fit(Xt); c = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xt), yt); return c.decision_function(sc.transform(Xv))
def ccc(Xt, ct, Xv, C=0.1):
    """Channel-conditional compensation: soft condition posterior from the features themselves (no oracle labels), subtract posterior-weighted condition mean."""
    cls = sorted(set(ct)); mu = np.stack([Xt[ct == c].mean(0) for c in cls])
    sc = StandardScaler().fit(Xt); clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xt), ct)
    f = lambda X: X - clf.predict_proba(sc.transform(X)) @ mu
    return f(Xt), f(Xv)
if __name__ == "__main__":
    dn = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tr, nz, va = L("train", dn), L("noise", dn), L("val", dn)
    Xv, yv, cv = arr(va)
    Xb, yb, cb = arr(tr); Xn, yn, cn = arr(nz); Xa, ya, ca = arr(tr, nz)
    report(f"[dn={dn}] base train only", lr(Xb, yb, Xv), yv, cv)
    report(f"[dn={dn}] base + 3000 noise views", lr(Xa, ya, Xv), yv, cv)
    Xc_t, Xc_v = ccc(Xa, ca, Xv)
    report(f"[dn={dn}] + channel compensation", lr(Xc_t, ya, Xc_v), yv, cv)
