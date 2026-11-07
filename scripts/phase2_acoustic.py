#!/usr/bin/env python3
"""Phase 2a - acoustic baselines. Fits on train/dev only, then writes one score per
(test clip, condition, model). Metrics are computed by phase2_report.py from the score file.

Models (all LFCC front end, see shieldcall/eval/lfcc.py):
  gmmA : 32-mix diagonal GMM per class, trained on CLEAN 16 kHz audio (the standard ASVspoof baseline recipe)
  gmmB : same, trained on 8 kHz audio passed through random SEEN telephone conditions
  lrA/lrB : logistic regression on utterance mean+std of LFCC (same two training regimes)
Score = higher means more likely spoof.
"""
from __future__ import annotations
import _repo  # noqa: F401
import argparse, csv, gzip, json, pickle, random, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from shieldcall.eval.lfcc import lfcc, trim_silence, resample
from shieldcall.eval import conditions as C

SEED = 20261107


def load(root: Path, rows):
    out = {}
    for r in rows:
        x, sr = sf.read(root / r["split"] / f"{r['utt']}.flac", dtype="float32")
        assert sr == 16000
        out[r["utt"]] = trim_silence(x, 16000)
    return out


def feats(x, sr, rate):
    return lfcc(resample(x, sr, rate), rate)


def stats(f):
    return np.concatenate([f.mean(0), f.std(0)])


class Model:
    def __init__(self, rate):
        self.rate = rate

    def fit(self, F_bona, F_spoof, ncomp=32):
        rng = np.random.RandomState(SEED)

        def pool(Fs, cap=300_000):
            a = np.concatenate(Fs)
            return a[rng.choice(len(a), min(cap, len(a)), replace=False)]
        self.g0 = GaussianMixture(ncomp, covariance_type="diag", max_iter=15, random_state=SEED, reg_covar=1e-3).fit(pool(F_bona))
        self.g1 = GaussianMixture(ncomp, covariance_type="diag", max_iter=15, random_state=SEED, reg_covar=1e-3).fit(pool(F_spoof))
        X = np.stack([stats(f) for f in F_bona + F_spoof]); y = np.r_[np.zeros(len(F_bona)), np.ones(len(F_spoof))]
        self.sc = StandardScaler().fit(X)
        self.lr = LogisticRegression(C=0.1, max_iter=500).fit(self.sc.transform(X), y)

    def score(self, f):
        gmm = float(np.mean(self.g1.score_samples(f) - self.g0.score_samples(f)))
        lr = float(self.lr.decision_function(self.sc.transform(stats(f)[None]))[0])
        return gmm, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/asvspoof2019_subset")
    ap.add_argument("--manifest", default="data/manifests/asvspoof2019_la_subset.csv")
    ap.add_argument("--out", default="docs/results/phase2_acoustic_scores.csv.gz")
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--max-test", type=int, default=0)
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.manifest)))
    for r in rows:
        r["label"] = int(r["label"])
    root = Path(a.data)
    t0 = time.time()
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "dev", "test")}
    if a.max_test:
        split["test"] = split["test"][:: max(1, len(split["test"]) // a.max_test)]
    aud = {s: load(root, split[s]) for s in split}
    print("loaded", {s: len(v) for s, v in aud.items()}, round(time.time() - t0), "s", flush=True)
    rng = random.Random(SEED)

    # ---- fit
    mp = work / "models.pkl"
    if mp.exists():
        mA, mB, thr_data = pickle.load(open(mp, "rb"))
    else:
        FA = {0: [], 1: []}; FB = {0: [], 1: []}
        for r in split["train"]:
            x = aud["train"][r["utt"]]
            FA[r["label"]].append(feats(x, 16000, 16000))
            for k in range(2):
                cond = rng.choice(C.SEEN)
                y, sr = C.apply(cond, x, C.seed_for(r["utt"] + str(k), cond))
                FB[r["label"]].append(feats(y, sr, 8000))
        print("train feats", round(time.time() - t0), "s", flush=True)
        mA, mB = Model(16000), Model(8000)
        mA.fit(FA[0], FA[1]); mB.fit(FB[0], FB[1])
        print("fit", round(time.time() - t0), "s", flush=True)
        # dev scores (threshold + calibration data): A on clean, B on one random seen condition
        thr_data = []
        for r in split["dev"]:
            x = aud["dev"][r["utt"]]
            cond = rng.choice(C.SEEN)
            y, sr = C.apply(cond, x, C.seed_for(r["utt"], "dev"))
            ga, la = mA.score(feats(x, 16000, 16000))
            gb, lb = mB.score(feats(y, sr, 8000))
            thr_data.append(dict(utt=r["utt"], label=r["label"], attack=r["attack"], speaker=r["speaker"], dev_cond=cond,
                                 gmmA=ga, lrA=la, gmmB=gb, lrB=lb))
        pickle.dump((mA, mB, thr_data), open(mp, "wb"))
    json.dump(thr_data, open(work / "dev_scores.json", "w"))
    print("dev done", round(time.time() - t0), "s", flush=True)

    # ---- score test
    out_rows = []
    lat = {}
    for cond in C.CONDITIONS:
        ck = work / f"test_{cond}.json"
        if ck.exists() and not a.max_test:
            out_rows += json.load(open(ck)); continue
        res = []
        tc = ts = 0.0
        for r in split["test"]:
            x = aud["test"][r["utt"]]
            t1 = time.perf_counter()
            y, sr = C.apply(cond, x, C.seed_for(r["utt"], cond))
            t2 = time.perf_counter()
            fa = feats(y, sr, 16000); fb = feats(y, sr, 8000)
            ga, la = mA.score(fa); gb, lb = mB.score(fb)
            t3 = time.perf_counter()
            tc += t2 - t1; ts += t3 - t2
            res.append(dict(utt=r["utt"], speaker=r["speaker"], attack=r["attack"], label=r["label"], cond=cond,
                            dur=len(y) / sr, gmmA=ga, lrA=la, gmmB=gb, lrB=lb))
        lat[cond] = dict(channel_ms_per_clip=1000 * tc / len(res), score_ms_per_clip_both_models=1000 * ts / len(res),
                         audio_s_per_clip=float(np.mean([q["dur"] for q in res])))
        out_rows += res
        if not a.max_test:
            json.dump(res, open(ck, "w"))
        print("cond", cond, len(res), round(time.time() - t0), "s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys = list(out_rows[0].keys())
    with gzip.open(a.out, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out_rows)
    json.dump(lat, open(work / "latency_acoustic.json", "w"), indent=1)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
