#!/usr/bin/env python3
"""Phase 8 final evaluation: ASVspoof 2021 LA with the Phase 8 model (cross-corpus; trained on ASVspoof 2019 train+dev only).

Nothing here is tuned on 2021 data. The classifier is model.npz (committed before the run).
Features: wav2vec2-base hidden layers 3/6/9/12, mean+std over time, first 2.5 s, per-clip z-norm (same as Phases 3b-6).
Scores are written so that HIGHER = BONA FIDE (the official scorer's convention).

modes
  smoke   first N trials in tar order, streamed straight from Zenodo (a few MB). Pipeline check only.
  sample  seeded stratified sample of the official 'eval' subset (codec x key cells). Needs the 7.8 GB tarball.
  full    all 148,176 'eval' trials. Needs the tarball. Resumable.
"""
import argparse, gzip, hashlib, io, json, multiprocessing as mp, os, platform, queue, random, re, subprocess, sys, tarfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz?download=1"
SEED = 20261109


# ---------------------------------------------------------------- features (Phase 8: backbone, silence policy and head all come from model8.npz)
sys.path.insert(0, os.path.dirname(HERE))
from phase8_lib import Backbone, decode as p8_decode, prep, load_model as load_head, score_head


def worker(tq, rq, threads, model_path):
    M, meta = load_head(model_path)
    bb = Backbone(meta["backbone"], threads, meta.get("max_layer", 0)); cap = int(meta["cap_s"] * 16000)
    while True:
        item = tq.get()
        if item is None:
            break
        tid, b = item
        try:
            f = bb(prep(p8_decode(b), meta["trim"], cap))[None]
            rq.put((tid, float(score_head(M, f)[0]), None))      # higher = bona fide
        except Exception as e:                                   # keep going, report at the end
            rq.put((tid, None, repr(e)))


# ---------------------------------------------------------------- data
def read_meta(keys_dir):
    rows = {}
    for l in open(os.path.join(keys_dir, "LA", "CM", "trial_metadata.txt")):
        p = l.split()
        if p[7] == "eval":
            rows[p[1]] = dict(spk=p[0], codec=p[2], tx=p[3], attack=p[4], key=p[5])
    return rows


def pick(meta, mode, n, per_cell):
    ids = sorted(meta)
    if mode == "full":
        return set(ids)
    rng = random.Random(SEED); cells = {}
    for t in ids:
        cells.setdefault((meta[t]["codec"], meta[t]["key"]), []).append(t)
    out = set()
    for c in sorted(cells):
        v = cells[c][:]; rng.shuffle(v); out.update(v[:per_cell])
    return out


def tar_stream(path):
    if path:
        f = open(path, "rb")
        return tarfile.open(fileobj=f, mode="r|gz")
    import requests
    r = requests.get(URL, stream=True, timeout=120)
    r.raise_for_status()
    return tarfile.open(fileobj=r.raw, mode="r|gz")


# ---------------------------------------------------------------- metrics
def eer(bona, spoof):
    s = np.r_[bona, spoof]; y = np.r_[np.ones(len(bona)), np.zeros(len(spoof))]
    o = np.argsort(s); y = y[o]
    frr = np.cumsum(y) / y.sum(); far = 1 - np.cumsum(1 - y) / (1 - y).sum()
    i = np.argmin(np.abs(frr - far)); return float((frr[i] + far[i]) / 2)


def boot_ci(bona, spoof, k=300):
    rng = np.random.RandomState(0); v = []
    for _ in range(k):
        v.append(eer(rng.choice(bona, len(bona)), rng.choice(spoof, len(spoof))))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def parse_tables(txt):
    """official main.py prints two LaTeX tables (min t-DCF, then EER); return {'tdcf': {row: [..]}, 'eer': {...}}"""
    txt = re.sub(r"\\cellcolor\[rgb\]\{[^}]*\}", "", txt)
    out = {"tdcf": {}, "eer": {}}; cur = None
    for l in txt.splitlines():
        if "Table for min tDCFs" in l: cur = "tdcf"
        elif "Table for EERs" in l: cur = "eer"
        elif cur and "&" in l and not l.strip().startswith("\\"):
            p = [x.strip().rstrip("\\").strip() for x in l.split("&")]
            try: out[cur][p[0]] = [float(x) for x in p[1:]]
            except ValueError: pass
    return out


def official(ep_dir, keys_dir, ids, score_file, tmp):
    """Run the organizers' main.py on a metadata copy restricted to the scored trials (validated to reproduce the full-set numbers)."""
    import shutil
    k2 = os.path.join(tmp, "keys_scored")
    if os.path.exists(k2): shutil.rmtree(k2)
    shutil.copytree(keys_dir, k2)
    for tr in ("PA", "DF"): os.makedirs(os.path.join(k2, tr), exist_ok=True)
    p = os.path.join(k2, "LA", "CM", "trial_metadata.txt")
    L = [l for l in open(p) if l.split()[1] in ids and l.split()[7] == "eval"]; open(p, "w").writelines(L)
    r = subprocess.run([sys.executable, "main.py", "--cm-score-file", score_file, "--track", "LA", "--subset", "eval", "--metadata", k2],
                       cwd=ep_dir, capture_output=True, text=True)
    return r.stdout + r.stderr


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "sample", "full"], required=True)
    ap.add_argument("--tar", default="", help="path to ASVspoof2021_LA_eval.tar.gz (sample/full)")
    ap.add_argument("--keys", default="keys", help="dir with LA/CM/trial_metadata.txt (from LA-keys-full.tar.gz)")
    ap.add_argument("--eval-package", default="2021/eval-package")
    ap.add_argument("--model", required=True, help="model8.npz from phase8_fit.py final")
    ap.add_argument("--ref-scores", default="", help="optional Phase 7 score file (id score per line) for a paired comparison on identical trials")
    ap.add_argument("--out", default="phase8_out")
    ap.add_argument("--n-smoke", type=int, default=300)
    ap.add_argument("--per-cell", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out); os.makedirs(a.out, exist_ok=True)
    cpu = os.cpu_count() or 2
    W = a.workers or max(1, min(4, cpu // 2)); T = max(1, cpu // W)
    meta = read_meta(a.keys); print("eval trials in keys:", len(meta), flush=True)
    want = None if a.mode == "smoke" else pick(meta, a.mode, 0, a.per_cell)
    if want is not None: print("trials to score:", len(want), flush=True)
    sf = os.path.join(a.out, f"scores_{a.mode}.txt")
    done = set(l.split()[0] for l in open(sf)) if os.path.exists(sf) else set()
    if done: print("resuming, already scored:", len(done), flush=True)
    ctx = mp.get_context("spawn"); tq = ctx.Queue(maxsize=W * 6); rq = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(tq, rq, T, os.path.abspath(a.model)), daemon=True) for _ in range(W)]
    [p.start() for p in procs]
    out = open(sf, "a"); errs = []; sent = 0; got = 0; t0 = time.time()
    total = a.n_smoke if a.mode == "smoke" else len(want) - len(done & want)

    def drain(block=False):
        nonlocal got
        while True:
            try: tid, s, e = rq.get(block, 0.2 if block else 0)
            except queue.Empty: return
            got += 1
            if e: errs.append((tid, e))
            else: out.write(f"{tid} {s:.6f}\n"); done.add(tid)
            if got % 200 == 0:
                out.flush(); r = got / (time.time() - t0)
                print(f"scored {got}/{total}  {r:.1f} clips/s  eta {(total - got) / max(r, 1e-9) / 60:.0f} min", flush=True)
            block = False

    t = tar_stream(a.tar or None) if a.mode != "smoke" else tar_stream(None)
    for mem in t:
        if not mem.isfile() or not mem.name.endswith(".flac"): continue
        tid = os.path.basename(mem.name)[:-5]
        if a.mode == "smoke":
            if tid not in meta: continue
            if sent >= a.n_smoke: break
        elif tid not in want or tid in done: continue
        b = t.extractfile(mem).read(); sent += 1
        while True:
            try: tq.put((tid, b), timeout=0.5); break
            except queue.Full: drain()
        drain()
    for _ in procs: tq.put(None)
    while got < sent: drain(True)
    out.close(); [p.join(timeout=30) for p in procs]
    print(f"scoring finished: {len(done)} trials, {len(errs)} errors, {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- metrics
    S = {l.split()[0]: float(l.split()[1]) for l in open(sf)}
    ids = set(S) & set(meta)
    res = {"mode": a.mode, "n_scored": len(ids), "errors": len(errs), "minutes": round((time.time() - t0) / 60, 1), "workers": W,
           "sampling": None if a.mode == "full" else ("first %d trials in tar order" % a.n_smoke if a.mode == "smoke" else f"seed {SEED}, {a.per_cell} trials per codec x key cell")}
    bo = np.array([S[t] for t in ids if meta[t]["key"] == "bonafide"]); sp = np.array([S[t] for t in ids if meta[t]["key"] == "spoof"])
    res["n_bonafide"], res["n_spoof"] = int(len(bo)), int(len(sp))
    if len(bo) > 20 and len(sp) > 20:
        res["pooled_eer"] = eer(bo, sp); res["pooled_eer_ci95"] = boot_ci(bo, sp)
        res["per_codec_eer"] = {}
        for c in sorted({meta[t]["codec"] for t in ids}):
            b = np.array([S[t] for t in ids if meta[t]["codec"] == c and meta[t]["key"] == "bonafide"])
            s = np.array([S[t] for t in ids if meta[t]["codec"] == c and meta[t]["key"] == "spoof"])
            if len(b) > 5 and len(s) > 5: res["per_codec_eer"][c] = eer(b, s)
        res["per_attack_eer"] = {}
        for at in sorted({meta[t]["attack"] for t in ids if meta[t]["key"] == "spoof"}):
            s = np.array([S[t] for t in ids if meta[t]["attack"] == at]); res["per_attack_eer"][at] = eer(bo, s)
    # ---- paired comparison with the Phase 7 model on identical trials
    if a.ref_scores and os.path.isfile(a.ref_scores) and "pooled_eer" in res:
        R = {}
        for l in (gzip.open(a.ref_scores, "rt") if a.ref_scores.endswith(".gz") else open(a.ref_scores)):
            p = l.split()
            if len(p) == 2: R[p[0]] = float(p[1])
        com = sorted(t for t in ids if t in R)
        y = np.array([meta[t]["key"] == "bonafide" for t in com]); s_new = np.array([S[t] for t in com]); s_old = np.array([R[t] for t in com])
        e_new, e_old = eer(s_new[y], s_new[~y]), eer(s_old[y], s_old[~y]); rng = np.random.RandomState(1); dl = []
        for _ in range(200):
            i = rng.randint(0, len(com), len(com)); yy = y[i]
            if yy.sum() > 5 and (~yy).sum() > 5: dl.append(eer(s_new[i][yy], s_new[i][~yy]) - eer(s_old[i][yy], s_old[i][~yy]))
        res["paired_vs_phase7"] = dict(n=len(com), eer_phase8=e_new, eer_phase7=e_old, delta=e_new - e_old, delta_ci95=[float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))])
    # ---- official scorer (needs the eval-package + keys)
    ep = os.path.abspath(a.eval_package)
    if os.path.isfile(os.path.join(ep, "main.py")) and len(ids) > 100:
        try:
            txt = official(ep, os.path.abspath(a.keys), ids, os.path.abspath(sf), a.out)
            open(os.path.join(a.out, f"official_tables_{a.mode}.txt"), "w").write(txt)
            res["official"] = parse_tables(txt)
            base = {}
            for name in ("LFCC-GMM", "CQCC-GMM", "LFCC-LCNN", "RawNet2"):
                p = os.path.join(a.keys, "LA", "CM", name, "score.txt")
                if os.path.isfile(p):
                    q = os.path.join(a.out, f"base_{name}.txt")
                    open(q, "w").writelines(l for l in open(p) if l.split()[0] in ids)
                    base[name] = parse_tables(official(ep, os.path.abspath(a.keys), ids, os.path.abspath(q), a.out))
            res["baselines_same_trials"] = {k: {"pooled_eer": v["eer"].get("Pooled", [None])[-1], "pooled_min_tdcf": v["tdcf"].get("Pooled", [None])[-1]} for k, v in base.items()}
        except Exception as e:
            res["official_error"] = repr(e)
    res["env"] = dict(python=sys.version.split()[0], platform=platform.platform(), cpus=cpu, model_sha256=hashlib.sha256(open(a.model, "rb").read()).hexdigest())
    try:
        import torch, transformers; res["env"].update(torch=torch.__version__, transformers=transformers.__version__)
    except Exception: pass
    json.dump(res, open(os.path.join(a.out, f"results_{a.mode}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("official", "per_attack_eer")}, indent=1))


if __name__ == "__main__":
    main()
