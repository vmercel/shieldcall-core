"""Fit the accepted Phase-3 recipe on ALL train attacks, score the locked test split once, report per-condition EER."""
import _repo  # noqa
import sys, json, time
import numpy as np, torch
torch.backends.nnpack.set_flags(False)
from phase3_lib import *
from shieldcall.eval.tdnn import TDNN, crop, specaug, score
SETS = ("lfcc", "mgd"); EP = 8
torch.manual_seed(0); rng = np.random.RandomState(0)
tv = load("data/work/views_train.pkl"); keys = list(tv); F = featurize(tv, SETS, keys); D = F[keys[0]].shape[1]
y = np.array([tv[k]["label"] for k in keys], np.float32)
conds = sorted({tv[k]["cond"] for k in keys}); cid = np.array([conds.index(tv[k]["cond"]) for k in keys])
model = TDNN(D, n_dom=len(conds), drop=0.3); opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=EP * (len(keys) // 64 + 1)); posw = torch.tensor((y == 0).sum() / (y == 1).sum())
t0 = time.time()
for ep in range(EP):
    model.train(); perm = rng.permutation(len(keys))
    for b in range(0, len(perm), 64):
        idx = perm[b:b + 64]
        x = specaug(torch.from_numpy(np.stack([crop(F[keys[i]], 200, rng).T for i in idx])), rng); t = torch.from_numpy(y[idx])
        lm = rng.beta(0.4, 0.4); pm = torch.from_numpy(rng.permutation(len(idx))); x = lm * x + (1 - lm) * x[pm]; t = lm * t + (1 - lm) * t[pm]
        lam = 2 / (1 + np.exp(-10 * ep / EP)) - 1
        lo, dd = model(x, lam=0.5 * lam)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lo, t, pos_weight=posw) + 0.3 * torch.nn.functional.cross_entropy(dd, torch.from_numpy(cid[idx]))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    print("ep", ep, round(time.time() - t0), flush=True)
del F
vv = load("data/work/views_val.pkl"); vf = featurize(vv, SETS)
dev = {}
for k, v in vv.items():
    dev.setdefault(v["cond"], ([], []))[0].append(v["label"]); dev[v["cond"]][1].append(score(model, vf[k]))
dev_s = np.array([s for c in dev for s in dev[c][1]]); dev_y = np.array([l for c in dev for l in dev[c][0]])
tvw = load("data/work/views_test.pkl"); out = {}
by = {}
for k, v in tvw.items():
    x, sr = decode(v); s = score(model, build(x, sr, SETS))
    by.setdefault(v["cond"], {"y": [], "s": [], "atk": []}); by[v["cond"]]["y"].append(v["label"]); by[v["cond"]]["s"].append(s); by[v["cond"]]["atk"].append(v["attack"])
res = {c: dict(eer=roc_eer(d["y"], d["s"]), n=len(d["y"])) for c, d in by.items()}
json.dump(dict(recipe="lfcc+mgd | TDNN | specaug+mixup+channel-adversarial(GRL) | 8 epochs", test=res, mean_eer=float(np.mean([r["eer"] for r in res.values()]))), open("docs/results/phase3_test_iter2.json", "w"), indent=1)
json.dump({c: dict(y=d["y"], s=d["s"], atk=d["atk"]) for c, d in by.items()}, open("data/work/test_scores_iter2.json", "w"))
print(json.dumps({c: round(r["eer"], 3) for c, r in res.items()}), "MEAN", np.mean([r["eer"] for r in res.values()]))
