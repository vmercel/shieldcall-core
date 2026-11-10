#!/usr/bin/env python3
"""Cheap-strategy selection on 2019 train/dev only: 4-layer backbone, pseudo-spoof families, LFCC fusion."""
import sys, json, numpy as np
from phase8_lib import *
from phase8_fit import load, dev_report
FD = sys.argv[1]; tag = "w2v2-base-L4"
tr = load(FD, f"{tag}_train_t1_c4"); dv = load(FD, f"{tag}_dev_t1_c4")
lay = [1, 2, 3, 4]; isGL = tr["attack"] == "GL"; fam = np.array([v.split("|")[0] if a == "GL" else "" for v, a in zip(tr["view"], tr["attack"])])
print(f"train views {len(tr['y'])}  pseudo {isGL.sum()}  families {dict(zip(*np.unique(fam[isGL], return_counts=True)))}  dev views {len(dv['y'])}", flush=True)
lt = ~np.isin(tr["attack"], ["A05", "A06"]); ld = ~np.isin(dv["attack"], ["A01", "A02", "A03", "A04"])
def zs(s, ref): return (s - ref.mean()) / (ref.std() + 1e-9)
def evalscores(name, sc, sc_loao):
    d = dev_report(sc, dv); y2 = dv["y"][ld]; lo = eer(sc_loao[y2 == 1], sc_loao[y2 == 0]); r = dict(name=name, clean=d["clean"], seen=d["seen"], held=d["held"], loao=lo, score=0.5 * d["mean3"] + 0.5 * lo, per_attack=d["per_attack"], stress=d["stress_snr0"])
    print(f"{name:38s} clean {d['clean']*100:5.2f} seen {d['seen']*100:5.2f} held {d['held']*100:5.2f} LOAO {lo*100:5.2f} score {r['score']*100:5.2f} | A10~ n/a  stress0dB {d['stress_snr0']*100:4.1f}", flush=True); return r
def head(kind, key, mask, layers, C=0.01):
    F = tr[key][mask]; M = fit_head(kind, F, tr["y"][mask], layers, C=C); return M
R = {}; S = {}
cfgs = [("lr", "no pseudo", ~isGL), ("lr", "GL0 only", ~isGL | (fam == "GL0")), ("lr", "all families", np.ones(len(isGL), bool))]
for kind, nm, m in cfgs + [("mlp", "all families", np.ones(len(isGL), bool))]:
    M = head(kind, "F", m, lay); s = score_head(M, dv["F"]); M2 = head(kind, "F", m & lt, lay); s2 = score_head(M2, dv["F"][ld])
    R[f"{kind} {nm}"] = evalscores(f"{kind} L1-4 pseudo={nm}", s, s2); S[f"{kind} {nm}"] = (s, s2)
# leave-family-out check: does training without family k still beat no-pseudo on dev?  (reported only)
# LFCC fusion
Qd = np.zeros((len(tr["y"]), 1, 240), np.float32); Qd[:, 0, :] = tr["Q"]
def qhead(mask, C=0.1): return fit_head("lr", Qd[mask].astype(np.float16), tr["y"][mask], [0], C=C)
Qv = dv["Q"][:, None, :].astype(np.float16)
allm = np.ones(len(isGL), bool)
Mq = qhead(allm); sq = score_head(Mq, Qv); Mq2 = qhead(allm & lt); sq2 = score_head(Mq2, Qv[ld])
R["lfcc only"] = evalscores("LFCC stats only (LR)", sq, sq2)
base = "mlp all families" if R["mlp all families"]["score"] <= R["lr all families"]["score"] else "lr all families"
s, s2 = S[base]; print("fusion base:", base)
for w in (0.1, 0.2, 0.3, 0.5):
    f = (1 - w) * zs(s, s) + w * zs(sq, sq); f2 = (1 - w) * zs(s2, s2) + w * zs(sq2, sq2)
    R[f"fuse w={w}"] = evalscores(f"fuse SSL+LFCC w={w}", f, f2)
json.dump(R, open(FD + "/cheap2_results.json", "w"), indent=1)
