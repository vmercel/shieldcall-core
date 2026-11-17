"""Phase 11: train the Phase 8 feature stack on REAL modern spoofs (ASVspoof 5) and test on a different corpus (In-the-Wild).
NOT comparable to the ASVspoof 2021 organiser baselines (different training data). Fixed before any score (see PROTOCOL.md).
stages: select -> extract (resumable) -> dev (train A17-A24, validate on unseen A25-A32, speaker-disjoint) -> final (refit on all) -> external In-the-Wild, paired against the Phase 8 head."""
import os, sys, json, zlib, random, time, argparse, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "phases", "phase8"))
import phase8_lib as L
CAP = 48000; VIEWS = ["native", "g711_ulaw", "sim_pstn"]
TRAIN_ATT = [f"A{i}" for i in range(17, 25)]; VAL_ATT = [f"A{i}" for i in range(25, 33)]
MODES = {  # asv5 shards, per-attack/bona quotas train, val; itw shards, per-class per-shard
    "smoke": dict(a5=[3, 19], tr_att=6, tr_bona=24, va_att=4, va_bona=16, itw=[0], itw_n=15),
    "full": dict(a5=[i * 12 + 3 for i in range(16)], tr_att=400, tr_bona=3200, va_att=150, va_bona=1200, itw=list(range(8)), itw_n=375),
}
A5 = "SpeechAntiSpoofingBenchmarks/ASVspoof5"; ITW = "SpeechAntiSpoofingBenchmarks/InTheWild"

def eer(y, s):
    from sklearn.metrics import roc_curve
    f, t, _ = roc_curve(y, s); i = int(np.nanargmin(abs(f - (1 - t)))); return 100 * (f[i] + 1 - t[i]) / 2

def get_shard(repo, name, work):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, f"data/{name}", repo_type="dataset", cache_dir=os.path.join(work, "hf"))

def select_and_save(a, work):
    """Deterministic selection; writes audio bytes to work/aud/<group>/<id>.flac and a json manifest."""
    import pyarrow.parquet as pq
    mf = os.path.join(work, f"manifest_{a.mode}.json")
    if os.path.exists(mf): return json.load(open(mf))
    c = MODES[a.mode]; items = []; cnt = {}
    q = {"train": (c["tr_att"], c["tr_bona"], TRAIN_ATT, 0), "val": (c["va_att"], c["va_bona"], VAL_ATT, 1)}
    cap_sh = lambda n: -(-n // len(c["a5"]))
    for sh in c["a5"]:
        t = pq.read_table(get_shard(A5, "test-%05d-of-00200.parquet" % sh, work)).to_pandas()
        rng = random.Random(1100 + sh); order = list(t.index); rng.shuffle(order); per = {}
        for i in order:
            n = json.loads(t.loc[i, "notes"]); at = n["attack_id"]; key = "bona" if at == "bonafide" else at
            par = zlib.crc32(n["speaker_id"].encode()) % 2
            for g, (qa, qb, atts, p) in q.items():
                if par != p or (key != "bona" and at not in atts): continue
                lim = qb if key == "bona" else qa
                if cnt.get((g, key), 0) >= lim or per.get((g, key), 0) >= cap_sh(lim): continue
                cnt[(g, key)] = cnt.get((g, key), 0) + 1; per[(g, key)] = per.get((g, key), 0) + 1
                d = os.path.join(work, "aud", g); os.makedirs(d, exist_ok=True); p_ = os.path.join(d, n["utterance_id"] + ".flac")
                open(p_, "wb").write(t.loc[i, "audio"]["bytes"])
                items.append(dict(group=g, utt=n["utterance_id"], key=key, codec=n["codec"], label=int(key != "bona"), path=p_))
        print("asv5 shard", sh, len(items), flush=True)
    for sh in c["itw"]:
        t = pq.read_table(get_shard(ITW, "test-%05d-of-00008.parquet" % sh, work)).to_pandas(); rng = random.Random(2200 + sh)
        for lab in (0, 1):
            idx = list(t.index[t.label == lab]); rng.shuffle(idx)
            for i in idx[:c["itw_n"]]:
                u = json.loads(t.loc[i, "notes"])["utterance_id"]; d = os.path.join(work, "aud", "itw"); os.makedirs(d, exist_ok=True)
                p_ = os.path.join(d, u + ".flac"); open(p_, "wb").write(t.loc[i, "audio"]["bytes"])
                items.append(dict(group="itw", utt=u, key="spoof" if lab else "bona", codec="-", label=lab, path=p_))
        print("itw shard", sh, len(items), flush=True)
    json.dump(items, open(mf, "w")); return items

def fit_clip(x):
    if len(x) < CAP: x = np.tile(x, CAP // len(x) + 1)[:CAP]
    return x[:CAP].astype(np.float32)

def extract(items, work, threads):
    bb = L.Backbone("w2v2-base", threads, 4); fd = os.path.join(work, "feats"); os.makedirs(fd, exist_ok=True)
    CH = 250; t0 = time.time()
    for k in range(0, len(items), CH):
        fn = os.path.join(fd, f"c{k:06d}.npz")
        if os.path.exists(fn): continue
        F = []; M = []
        for s in items[k:k + CH]:
            try: x = L.prep(L.decode(open(s["path"], "rb").read()), 1, CAP)
            except Exception as e: print("skip", s["utt"], e); continue
            for v in VIEWS:
                xx = x if v == "native" else L.channel(x, v, s["utt"])
                F.append(bb(fit_clip(L.prep(xx, 0, CAP)))[1:5].reshape(-1)); M.append((s["group"], s["utt"], v, s["key"], s["codec"]))
        np.savez(fn, F=np.array(F, dtype=np.float16), M=np.array(M)); print("extracted", k + CH, "/", len(items), round(time.time() - t0), "s", flush=True)
    Fs = []; Ms = []
    for fn in sorted(os.listdir(fd)):
        z = np.load(os.path.join(fd, fn)); Fs.append(z["F"]); Ms.append(z["M"])
    return np.vstack(Fs).astype(np.float32), np.vstack(Ms)

def head(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X); m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
    return lambda Z: m.decision_function(sc.transform(Z)), sc, m          # higher = spoof

def boot_diff(y, s_a, s_b, n=1000, seed=0):
    rng = np.random.RandomState(seed); idx = np.arange(len(y)); d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx)); yb = y[b]
        if yb.min() == yb.max(): continue
        d.append(eer(yb, s_a[b]) - eer(yb, s_b[b]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="smoke"); ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--work", default=os.path.join(HERE, "work")); ap.add_argument("--out", default=os.path.join(HERE, "out")); a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    items = select_and_save(a, a.work); F, M = extract(items, a.work, a.threads)
    grp, view, key = M[:, 0], M[:, 2], M[:, 3]; y = (key != "bona").astype(int)
    m8 = np.load(os.path.join(ROOT, "phases", "phase8", "model8.npz"), allow_pickle=True)
    s8 = -(((F - m8["m_mu"]) / m8["m_sd"]) @ m8["m_w"] + float(m8["m_b"]))              # Phase 8 head, higher = spoof
    tr = grp == "train"; va = grp == "val"; it = grp == "itw"; rep = {"mode": a.mode, "clips": {g: int(((grp == g) & (view == "native")).sum()) for g in ("train", "val", "itw")}}
    # ---- dev: choose C on unseen-attack validation only (mean EER over the 3 views)
    best = None; dev = {}
    for C in (0.001, 0.003, 0.01, 0.03):
        f, _, _ = head(F[tr], y[tr], C); s = f(F); r = {v: eer(y[va & (view == v)], s[va & (view == v)]) for v in VIEWS}; r["mean"] = float(np.mean(list(r.values()))); dev[str(C)] = r
        if best is None or r["mean"] < best[0]: best = (r["mean"], C)
    C = best[1]; rep["dev_C_selection"] = dev; rep["chosen_C"] = C
    f_dev, _, _ = head(F[tr], y[tr], C); s_dev = f_dev(F)
    rep["val_unseen_attacks"] = {"phase8": {v: eer(y[va & (view == v)], s8[va & (view == v)]) for v in VIEWS}, "phase11_dev": {v: eer(y[va & (view == v)], s_dev[va & (view == v)]) for v in VIEWS}}
    # ---- final: refit on train+val (all attacks A17-A32), score In-the-Wild (never used for any choice)
    f_fin, sc, m = head(F[tr | va], y[tr | va], C); s_fin = f_fin(F)
    ext = {}
    for v in VIEWS:
        k_ = it & (view == v); yy = y[k_]
        ext[v] = {"n": int(k_.sum()), "phase8": eer(yy, s8[k_]), "phase11_dev_model": eer(yy, s_dev[k_]), "phase11_final": eer(yy, s_fin[k_]),
                  "paired_diff_final_minus_phase8_95CI": boot_diff(yy, s_fin[k_], s8[k_])}
    rep["in_the_wild"] = ext
    rep["verdict_rule"] = "PASS only if final beats Phase 8 on In-the-Wild native AND g711 with the paired 95% CI entirely below 0"
    rep["pass"] = bool(all(ext[v]["paired_diff_final_minus_phase8_95CI"][1] < 0 for v in ("native", "g711_ulaw")))
    # save model in Phase 8's format (higher = bona fide)
    w = -(m.coef_[0] / sc.scale_); b = -(m.intercept_[0] - (m.coef_[0] * sc.mean_ / sc.scale_).sum())
    np.savez(os.path.join(a.out, "model11.npz"), m_kind="lr", m_layers=np.array([1, 2, 3, 4]), m_mu=np.zeros(6144), m_sd=np.ones(6144), m_w=w, m_b=b, meta=json.dumps({"C": C, "train": "ASVspoof5 A17-A32 + G711/sim_pstn views"}))
    rep["model11_sha256"] = hashlib.sha256(open(os.path.join(a.out, "model11.npz"), "rb").read()).hexdigest()
    json.dump(rep, open(os.path.join(a.out, f"report_{a.mode}.json"), "w"), indent=1); print(json.dumps(rep, indent=1))

if __name__ == "__main__": main()
