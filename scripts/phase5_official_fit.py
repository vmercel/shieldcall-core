"""Score the official-eval sample with the FIXED Phase 3b recipe (LR C=0.01 on train + noise views). No tuning on this data."""
import _repo  # noqa
import pickle, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from phase3_lib import roc_eer
L = lambda n: pickle.load(open(f"data/work/w2v_{n}.pkl", "rb"))
tr = L("train"); tr.update(L("trainnoise")); te = L("official")
a = lambda d: (np.stack([v[0] for v in d.values()]).astype(np.float32), np.array([v[1] for v in d.values()]), np.array([v[2] for v in d.values()]))
Xt, yt, _ = a(tr); Xe, ye, ae = a(te)
sc = StandardScaler().fit(Xt); clf = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(Xt), yt); s = clf.decision_function(sc.transform(Xe))
eer = roc_eer(ye, s); rng = np.random.RandomState(0); bs = []
for _ in range(500):
    j = rng.choice(len(ye), len(ye)); bs.append(roc_eer(ye[j], s[j]))
res = {"n": int(len(ye)), "n_spoof": int(ye.sum()), "pooled_eer": eer, "bootstrap95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "per_attack_eer_vs_bonafide": {}}
for at in sorted(set(ae) - {"-"}):
    m = (ae == at) | (ye == 0); res["per_attack_eer_vs_bonafide"][at] = {"n": int((ae == at).sum()), "eer": roc_eer(ye[m], s[m])}
json.dump(res, open("docs/results/phase5_official_eval_sample.json", "w"), indent=1); print(json.dumps(res, indent=1))
