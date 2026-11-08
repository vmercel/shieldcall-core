"""Phase 5 idea: paired-channel nuisance projection. Learn the subspace in which the SAME clip moves when only the channel/noise changes
(differences between views of one utterance), project it out of the features, then classify. Validation protocol unchanged."""
import _repo  # noqa
import pickle, numpy as np
from phase4_fit import *
dn = 0
tr, nz, va = L("train", dn), L("noise", dn), L("val", dn)
Xv, yv, cv = arr(va)
# group views by utterance id
def utt(k): return k[0][0] if isinstance(k[0], tuple) else k[0]
allv = {}; allv.update(tr); allv.update(nz)
keys = list(allv); X = np.stack([allv[k][0] for k in keys]).astype(np.float32); y = np.array([allv[k][1] for k in keys]); c = np.array([allv[k][3] for k in keys])
u = np.array([utt(k) for k in keys])
sc = StandardScaler().fit(X); Z = sc.transform(X)
from collections import defaultdict
g = defaultdict(list); [g[a].append(i) for i, a in enumerate(u)]
D = []
for a, idx in g.items():
    if len(idx) > 1:
        m = Z[idx].mean(0); D.append(Z[idx] - m)
D = np.concatenate(D); print("pairs/groups", len(g), "diff rows", len(D), flush=True)
U, S, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
base = report("ref noise-aug LR (Phase4.2)", lr(X, y, Xv), yv, cv)
for k in (16, 64, 128, 256):
    P = Vt[:k].T; proj = lambda A: A - (A @ P) @ P.T
    clf = LogisticRegression(C=0.01, max_iter=3000).fit(proj(Z), y)
    report(f"paired nuisance projection k={k}", clf.decision_function(proj(sc.transform(Xv))), yv, cv)
