#!/usr/bin/env python3
"""Phase 9 head selection and final fit. Uses ASVspoof 2019 LA train/dev features only. Never touches 2021.

select : for each candidate config train (a) on all train views, judged on dev clean / seen-channel / held-channel EER, (b) on train without attacks A05,A06,
         judged on dev A05,A06 vs dev bona fide. score = 0.5*mean(clean,seen,held) + 0.5*leave-attacks-out (lower wins; ulaw_snr0 is a stress test only).
final  : refit the winning config on train+dev views (3 seeds, scores averaged) -> model9.pt
"""
import argparse, json, os, sys, time, hashlib
import numpy as np
from phase9_lib import *

CONFIGS = {   # fixed before any 2021 scoring; at most these four
    "c1_conv_nopseudo": dict(h=192, blocks=3, drop=0.2, pseudo="none"),
    "c2_conv_gl_hifigan": dict(h=192, blocks=3, drop=0.2, pseudo="all"),
    "c3_conv_wide_gl_hifigan": dict(h=256, blocks=3, drop=0.3, pseudo="all"),
    "c4_attnpool_only_gl_hifigan": dict(h=192, blocks=0, drop=0.2, pseudo="all"),
}


def load(out, split):
    meta = json.load(open(f"{out}/meta_{split}.json"))["items"]; N = len(meta)
    X = np.memmap(f"{out}/feats_{split}.f16", np.float16, "r", shape=(N, T_FR, DIN))
    T = np.array(np.memmap(f"{out}/len_{split}.i16", np.int16, "r", shape=(N,))).astype(np.int64)
    D = np.array(np.memmap(f"{out}/done_{split}.u8", np.uint8, "r", shape=(N,))).astype(bool)
    return meta, X, T, D


class Cat:
    """train+dev as one array (indices are always passed sorted)"""
    def __init__(s, A, B): s.A, s.B, s.n = A, B, len(A)
    def __getitem__(s, i):
        i = np.asarray(i); lo = i[i < s.n]; hi = i[i >= s.n] - s.n
        return np.concatenate([np.asarray(s.A[lo]), np.asarray(s.B[hi])]) if len(hi) else np.asarray(s.A[lo])


def norm_stats(X, T, idx, rng):
    s = np.sort(rng.choice(idx, min(1500, len(idx)), replace=False)); A = np.asarray(X[s], dtype=np.float32); M = lengths_mask(T[s]) > 0
    v = A[M]; return v.mean(0).astype(np.float32), (v.std(0) + 1e-5).astype(np.float32)


def train(cfg, X, T, y, idx, mu, sd, seed, epochs, bs=128, device="cpu", log=None):
    import torch
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    m = build_head(cfg).to(device); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.05)
    steps = epochs * int(np.ceil(len(idx) / bs)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.15)
    yy = y[idx]; w = np.where(yy == 1, 0.5 / max(yy.sum(), 1), 0.5 / max((1 - yy).sum(), 1)) * len(yy)         # class-balanced weights
    mu_t = torch.from_numpy(mu).to(device); sd_t = torch.from_numpy(sd).to(device); t0 = time.time()
    for ep in range(epochs):
        m.train(); perm = rng.permutation(len(idx)); tot = 0
        for i in range(0, len(perm), bs):
            j = np.sort(perm[i:i + bs]); b = idx[j]
            x = (torch.from_numpy(np.asarray(X[b], dtype=np.float32)).to(device) - mu_t) / sd_t
            mk = torch.from_numpy(lengths_mask(T[b])).to(device)
            for k in range(len(b)):                                                                              # time masking (SpecAugment-style)
                for _ in range(rng.randint(0, 3)):
                    L = rng.randint(2, 9); s0 = rng.randint(0, max(T_FR - L, 1)); x[k, s0:s0 + L] = 0
            x = (x + 0.05 * torch.randn_like(x)) * mk[..., None]
            lg = m(x, mk); wt = torch.from_numpy(w[j].astype(np.float32)).to(device)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(lg, torch.from_numpy(y[b].astype(np.float32)).to(device), reduction="none") * wt).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step(); sch.step(); tot += float(loss) * len(b)
        if log: log(f"    epoch {ep + 1}/{epochs} loss {tot / len(idx):.4f}  {(time.time() - t0) / 60:.1f} min")
    return m.eval()


def pick_idx(meta, D, cfg, drop_attacks=()):
    return np.array([i for i, it in enumerate(meta) if D[i] and it["attack"] not in drop_attacks and not (it["attack"] in ("GL", "HF") and cfg["pseudo"] == "none")])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["select", "final"]); ap.add_argument("--work", default="work9"); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--sel-frac", type=float, default=0.5, help="select: fraction of train views used per candidate (same subset for all)")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--out", default="out9"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import torch; torch.set_num_threads(os.cpu_count() or 2)
    log = lambda s: print(s, flush=True)
    mt, Xt, Tt, Dt = load(a.work, "train"); md, Xd, Td, Dd = load(a.work, "dev")
    yt = np.array([it["y"] for it in mt]); yd = np.array([it["y"] for it in md]); rng = np.random.RandomState(7)
    log(f"train views {Dt.sum()}/{len(mt)}  dev views {Dd.sum()}/{len(md)}")
    if a.mode == "select":
        res = {}; sub = rng.rand(len(mt)) < a.sel_frac
        dv = lambda f: np.array([i for i, it in enumerate(md) if Dd[i] and f(it)])
        sets = {"clean": dv(lambda it: it["view"] == "clean_16k"), "seen": dv(lambda it: it["view"] in SEEN),
                "held": dv(lambda it: it["view"] in HELD and it["view"] != SEL_EXCLUDE), "stress_0dB": dv(lambda it: it["view"] == SEL_EXCLUDE)}
        bona = {k: v[yd[v] == 1] for k, v in sets.items()}; spoof = {k: v[yd[v] == 0] for k, v in sets.items()}
        for name, cfg in CONFIGS.items():
            r = {}
            idx = pick_idx(mt, Dt & sub, cfg); mu, sd = norm_stats(Xt, Tt, idx, rng)
            log(f"== {name} (all attacks) on {len(idx)} views"); m = train(cfg, Xt, Tt, yt, idx, mu, sd, 1, a.epochs, device=a.device, log=log)
            sc = predict([m], mu, sd, Xd, Td, device=a.device)
            for k in sets:
                if len(bona[k]) > 3 and len(spoof[k]) > 3: r[k] = 100 * eer(sc[bona[k]], sc[spoof[k]])
            idx2 = pick_idx(mt, Dt & sub, cfg, ("A05", "A06")); log(f"== {name} (leave attacks A05,A06 out) on {len(idx2)} views")
            m2 = train(cfg, Xt, Tt, yt, idx2, mu, sd, 2, a.epochs, device=a.device, log=log); sc2 = predict([m2], mu, sd, Xd, Td, device=a.device)
            cl = sets["clean"]; ab = cl[yd[cl] == 1]; ar = np.array([i for i in cl if md[i]["attack"] in ("A05", "A06")])
            lao = 100 * eer(sc2[ab], sc2[ar]) if len(ar) > 3 and len(ab) > 3 else float("nan")
            r["leave_attacks_out"] = lao
            r["score"] = 0.5 * np.mean([r.get("clean", 50), r.get("seen", 50), r.get("held", 50)]) + 0.5 * (lao if lao == lao else 50)
            res[name] = r; log(json.dumps({name: r}))
            json.dump(dict(configs=CONFIGS, results=res, epochs=a.epochs, sel_frac=a.sel_frac), open(f"{a.out}/dev_selection_results.json", "w"), indent=1)
        best = min(res, key=lambda k: res[k]["score"]); log("WINNER " + best)
        json.dump(dict(configs=CONFIGS, results=res, winner=best, epochs=a.epochs, sel_frac=a.sel_frac), open(f"{a.out}/dev_selection_results.json", "w"), indent=1)
    else:
        sel = json.load(open(f"{a.out}/dev_selection_results.json")); name = sel["winner"]; cfg = CONFIGS[name]; log(f"final fit of {name} on train+dev, {a.seeds} seeds")
        XA = Cat(Xt, Xd); TA = np.r_[Tt, Td]; yA = np.r_[yt, yd]; meta = mt + md; DA = np.r_[Dt, Dd]
        idx = pick_idx(meta, DA, cfg); mu, sd = norm_stats(XA, TA, idx, rng)
        ms = [train(cfg, XA, TA, yA, idx, mu, sd, 10 + s, a.epochs, device=a.device, log=log) for s in range(a.seeds)]
        save_ckpt(f"{a.out}/model9.pt", ms, cfg, mu, sd, dict(winner=name, epochs=a.epochs, seeds=a.seeds, cap_s=CAP_S, t_fr=T_FR, views=int(len(idx))))
        h = hashlib.sha256(open(f"{a.out}/model9.pt", "rb").read()).hexdigest(); open(f"{a.out}/model9.sha256", "w").write(h + "\n"); log("model9.pt sha256 " + h)


if __name__ == "__main__":
    main()
