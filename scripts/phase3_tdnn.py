import _repo  # noqa
import sys, time, json
import numpy as np, torch
torch.backends.nnpack.set_flags(False) if hasattr(torch.backends,"nnpack") else None
from phase3_lib import *
from shieldcall.eval.tdnn import TDNN, crop, specaug, score

sets = tuple(sys.argv[1].split("+")); EP = int(sys.argv[2]); tricks = sys.argv[3] if len(sys.argv) > 3 else ""
torch.manual_seed(0); rng = np.random.RandomState(0)
tv = load("data/work/views_train.pkl"); vv = load("data/work/views_val.pkl")
keys = [k for k, v in tv.items() if v["attack"] not in HELD]
t0 = time.time(); F = featurize(tv, sets, keys); D = F[keys[0]].shape[1]
vf = featurize(vv, sets); print("feats", round(time.time() - t0), "s", D, flush=True)
y = np.array([tv[k]["label"] for k in keys], np.float32)
conds = sorted({tv[k]['cond'] for k in keys}); cid = np.array([conds.index(tv[k]['cond']) for k in keys])
model = TDNN(D, n_dom=len(conds) if 'grl' in tricks else 0, drop=0.3 if 'reg' in tricks else 0.2); opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=EP * (len(keys) // 64 + 1))
posw = torch.tensor((y == 0).sum() / (y == 1).sum())
best = None
for ep in range(EP):
    model.train(); perm = rng.permutation(len(keys)); tl = 0
    for b in range(0, len(perm), 64):
        idx = perm[b:b + 64]
        x = torch.from_numpy(np.stack([crop(F[keys[i]], 200, rng).T for i in idx])); t = torch.from_numpy(y[idx])
        if "aug" in tricks: x = specaug(x, rng)
        if 'mix' in tricks:
            lam_m = rng.beta(0.4, 0.4); pm = torch.from_numpy(rng.permutation(len(idx)))
            x = lam_m * x + (1 - lam_m) * x[pm]; t = lam_m * t + (1 - lam_m) * t[pm]
        p_ = ep / EP; lam = 2 / (1 + np.exp(-10 * p_)) - 1
        lo, dd = model(x, lam=0.5 * lam if 'grl' in tricks else 0.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lo, t, pos_weight=posw)
        if 'grl' in tricks: loss = loss + 0.3 * torch.nn.functional.cross_entropy(dd, torch.from_numpy(cid[idx]))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(idx)
    if ep % 2 == 1 or ep == EP - 1:
        r = val_eval(lambda f: score(model, f), vv, vf)
        print(ep, "loss %.4f" % (tl / len(keys)), "VAL MEAN %.4f" % r["MEAN"], {c: round(v, 3) for c, v in r.items() if c != "MEAN"}, round(time.time() - t0), "s", flush=True)
    else:
        print(ep, "loss %.4f" % (tl / len(keys)), round(time.time() - t0), "s", flush=True)
