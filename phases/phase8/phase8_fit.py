#!/usr/bin/env python3
"""Phase 8 model selection and final fit. Uses ONLY ASVspoof 2019 LA train and dev features. ASVspoof 2021 is never read here.
  select : grid over head x layer set x pseudo-spoofs, judged on dev (clean / seen-codec / held-out-codec) and on a leave-attacks-out split
  final  : refit the winning config on train+dev and write model8.npz
"""
import argparse, glob, json, os, sys, time
import numpy as np
from phase8_lib import *


def load(feat_dir, tag):
    fs = sorted(glob.glob(os.path.join(feat_dir, tag + "_part*.npz")))
    if not fs: raise SystemExit(f"no feature shards for {tag} in {feat_dir}")
    Z = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in Z]) for k in ("F", "utt", "view", "y", "attack") + (("Q",) if "Q" in Z[0].files else ())}


def sub(D, m): return {k: v[m] for k, v in D.items()}


def group(view):
    if view == "clean_16k": return "clean"
    if view == "ulaw_snr0": return "stress"           # 0 dB noise: reported, never selected on
    return "seen" if view in SEEN else "held"


def dev_report(score, D):
    g = np.array([group(v) for v in D["view"]]); out = {}
    for name in ("clean", "seen", "held"):
        m = g == name; y = D["y"][m]; s = score[m]
        out[name] = eer(s[y == 1], s[y == 0]) if (y == 1).sum() > 10 and (y == 0).sum() > 10 else float("nan")
    m = g == "stress"; y = D["y"][m]; s0 = score[m]
    out["stress_snr0"] = eer(s0[y == 1], s0[y == 0]) if (y == 1).sum() > 10 and (y == 0).sum() > 10 else float("nan")
    out["mean3"] = float(np.nanmean([out["clean"], out["seen"], out["held"]]))
    out["pooled"] = eer(score[D["y"] == 1], score[D["y"] == 0])
    pa = {}
    bo = score[D["attack"] == "bonafide"]
    for a in sorted(set(D["attack"]) - {"bonafide"}): pa[a] = eer(bo, score[D["attack"] == a])
    out["per_attack"] = pa
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["select", "final"])
    ap.add_argument("--backbone", default="w2v2-base"); ap.add_argument("--max-layer", type=int, default=0)
    ap.add_argument("--trim", type=int, default=1); ap.add_argument("--cap-s", type=int, default=4)
    ap.add_argument("--feats", default="work/feats"); ap.add_argument("--out", default="work/models")
    ap.add_argument("--config", default="", help="final: path to selection json (default: best in work/models/select_<tag>.json)")
    ap.add_argument("--quick", action="store_true", help="smaller grid")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    base = f"{a.backbone}{'-L%d' % a.max_layer if a.max_layer else ''}"
    tr = load(a.feats, f"{base}_train_t{a.trim}_c{a.cap_s}"); dv = load(a.feats, f"{base}_dev_t{a.trim}_c{a.cap_s}")
    L = tr["F"].shape[1]; sets = layer_sets(L)
    print(f"{base}: train views {len(tr['y'])} ({(tr['attack']=='GL').sum()} pseudo), dev views {len(dv['y'])}, layers {L}", flush=True)
    sel_path = os.path.join(a.out, f"select_{base}_t{a.trim}.json")
    if a.cmd == "select":
        grid = []
        for kind in ("lr", "committee", "mlp"):
            for sn in sets:
                for ps in (0, 1):
                    if a.quick and (sn not in ("A_every3", "D_mid") or kind == "mlp"): continue
                    grid.append((kind, sn, ps, 0.01))
        for C in (0.003, 0.03):
            grid.append(("lr", "A_every3", 0, C))
        res = []
        for kind, sn, ps, C in grid:
            t0 = time.time(); lay = sets[sn]
            keep = (tr["attack"] != "GL") if not ps else np.ones(len(tr["y"]), bool)
            M = fit_head(kind, tr["F"][keep], tr["y"][keep], lay, C=C); d = dev_report(score_head(M, dv["F"]), dv)
            loao_tr = keep & ~np.isin(tr["attack"], ["A05", "A06"]); loao_dv = ~np.isin(dv["attack"], ["A01", "A02", "A03", "A04"])
            M2 = fit_head(kind, tr["F"][loao_tr], tr["y"][loao_tr], lay, C=C); s2 = score_head(M2, dv["F"][loao_dv])
            y2 = dv["y"][loao_dv]; loao = eer(s2[y2 == 1], s2[y2 == 0])
            r = dict(kind=kind, set=sn, layers=lay, pseudo=ps, C=C, dev=d, loao_eer=loao, score=0.5 * d["mean3"] + 0.5 * loao, secs=round(time.time() - t0))
            res.append(r)
            print(f"{kind:9s} {sn:9s} pseudo={ps} C={C:<6} dev clean {d['clean']*100:5.2f} seen {d['seen']*100:5.2f} held {d['held']*100:5.2f} | LOAO {loao*100:5.2f} | score {r['score']*100:5.2f}  ({r['secs']}s)", flush=True)
        res.sort(key=lambda r: r["score"])
        json.dump({"backbone": base, "trim": a.trim, "ranked": res}, open(sel_path, "w"), indent=1)
        print("BEST:", {k: res[0][k] for k in ("kind", "set", "pseudo", "C", "score")})
    else:
        cfg = json.load(open(a.config or sel_path))["ranked"][0]
        # fit on all of train + dev (the official 2019 training condition allows both)
        F = np.concatenate([tr["F"], dv["F"]]); y = np.concatenate([tr["y"], dv["y"]]); at = np.concatenate([tr["attack"], dv["attack"]])
        keep = (at != "GL") if not cfg["pseudo"] else np.ones(len(y), bool)
        M = fit_head(cfg["kind"], F[keep], y[keep], cfg["layers"], C=cfg["C"])
        meta = dict(backbone=a.backbone, max_layer=a.max_layer, trim=a.trim, cap_s=a.cap_s, config={k: cfg[k] for k in ("kind", "set", "pseudo", "C", "layers")},
                    selection_scores=dict(dev=cfg["dev"], loao=cfg["loao_eer"]), n_fit_views=int(keep.sum()), note="fit on ASVspoof 2019 LA train+dev only")
        p = os.path.join(a.out, "model8.npz"); save_model(p, M, meta)
        import hashlib; print("wrote", p, hashlib.sha256(open(p, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
