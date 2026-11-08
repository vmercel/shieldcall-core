"""Score the ASVspoof 5 sample with the FIXED Phase 3b recipe (LR C=0.01 on train + noise views). No tuning on this data."""
import _repo  # noqa
import pickle, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from phase3_lib import roc_eer
L = lambda n: pickle.load(open(f"data/work/w2v_{n}.pkl", "rb"))
tr = L("train"); tr.update(L("trainnoise")); te = L("asv5")
a = lambda d: (np.stack([v[0] for v in d.values()]).astype(np.float32), np.array([v[1] for v in d.values()]), np.array([v[2] for v in d.values()]), np.array([v[3] for v in d.values()]))
Xt, yt, _, _ = a(tr); Xe, ye, ae, ce = a(te)
sc = StandardScaler().fit(Xt); clf = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(Xt), yt); s = clf.decision_function(sc.transform(Xe))
def ci(y, s, seed=0, n=500):
    rng = np.random.RandomState(seed); bs = []
    for _ in range(n):
        j = rng.choice(len(y), len(y)); bs.append(roc_eer(y[j], s[j]))
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
res = {"n": int(len(ye)), "n_spoof": int(ye.sum()), "pooled_eer": roc_eer(ye, s), "bootstrap95": ci(ye, s), "per_codec": {}, "per_attack_vs_bonafide": {}}
for c in sorted(set(ce)):
    m = ce == c; res["per_codec"][c] = {"n": int(m.sum()), "n_bona": int((ye[m] == 0).sum()), "eer": roc_eer(ye[m], s[m])}
for at in sorted(set(ae) - {"-", "bonafide"}):
    m = (ae == at) | (ye == 0); res["per_attack_vs_bonafide"][at] = {"n": int((ae == at).sum()), "eer": roc_eer(ye[m], s[m])}
json.dump(res, open("docs/results/phase6_asv5_sample.json", "w"), indent=1); print(json.dumps(res, indent=1))
