import _repo  # noqa
import sys, time, json
import numpy as np
from sklearn.mixture import GaussianMixture
from phase3_lib import *

SETS = [tuple(s.split("+")) for s in sys.argv[1].split(",")]
NC = int(sys.argv[2]) if len(sys.argv) > 2 else 32
tv = load("data/work/views_train.pkl"); vv = load("data/work/views_val.pkl")
keys = [k for k, v in tv.items() if v["attack"] not in HELD]
res = {}
for sets in SETS:
    t0 = time.time()
    F = featurize(tv, sets, keys)
    rng = np.random.RandomState(0)
    G = {}
    for lab in (0, 1):
        fr = np.concatenate([F[k] for k in keys if tv[k]["label"] == lab])
        fr = fr[rng.choice(len(fr), min(400000, len(fr)), replace=False)]
        G[lab] = GaussianMixture(NC, covariance_type="diag", max_iter=20, random_state=0, reg_covar=1e-3).fit(fr)
    vf = featurize(vv, sets)
    r = val_eval(lambda f: float(np.mean(G[1].score_samples(f) - G[0].score_samples(f))), vv, vf)
    res["+".join(sets)] = r
    print("+".join(sets), NC, "MEAN %.4f" % r["MEAN"], {c: round(v, 3) for c, v in r.items() if c != "MEAN"}, round(time.time() - t0), "s", flush=True)
json.dump(res, open("data/work/gmm_%s_%d.json" % (sys.argv[1].replace("+", "-").replace(",", "_"), NC), "w"))
