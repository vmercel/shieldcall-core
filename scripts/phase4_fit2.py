import _repo  # noqa
import numpy as np, sys
from phase4_fit import *
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
dn = int(sys.argv[1]); tr, nz, va = L("train", dn), L("noise", dn), L("val", dn)
Xv, yv, cv = arr(va); Xa, ya, ca = arr(tr, nz)
REF = 0.0762
# 1. MLP head
sc = StandardScaler().fit(Xa); pca = PCA(256, random_state=0).fit(sc.transform(Xa))
Z = lambda X: pca.transform(sc.transform(X))
for a in (1e-1, 1.0):
    m = MLPClassifier((256,), alpha=a, max_iter=200, early_stopping=True, random_state=0).fit(Z(Xa), ya)
    report(f"MLP256 pca256 alpha={a}", m.predict_proba(Z(Xv))[:, 1], yv, cv)
# 2. SNR-gated mixture of experts: gate = condition posterior from features; experts = LR per condition group
grp = lambda c: "clean" if c in ("clean_16k", "g711_ulaw", "amr_nb_12k2") or not c.startswith("noise") else ("lo" if c in ("noise0", "noise3", "noise6") else "hi")
ga = np.array([grp(c) if c.startswith("noise") else "clean" for c in ca]); G = ["clean", "lo", "hi"]
gate = LogisticRegression(C=0.1, max_iter=2000).fit(Z(Xa), ga)
P = gate.predict_proba(Z(Xv)); S = np.zeros(len(Xv))
for j, g in enumerate(gate.classes_):
    msk = np.ones(len(Xa), bool)  # each expert sees all data but weights its group x3
    w = np.where(ga == g, 3.0, 1.0)
    e = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(Xa), ya, sample_weight=w)
    S += P[:, j] * e.decision_function(sc.transform(Xv))
report("SNR-gated mixture of experts", S, yv, cv)
