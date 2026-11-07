#!/usr/bin/env python3
"""Phase 2d - the repository's own AcousticDeepfakeScorer (STRF heuristic + prototype memory) on the same
locked splits, on a smaller stratified subset (the streaming scorer is slow on one CPU).
Prototype memory is fit on train clean audio; threshold/calibration from dev; test scored once."""
from __future__ import annotations
import _repo  # noqa: F401
import csv, json, time, sys
from pathlib import Path
import numpy as np, soundfile as sf

from shieldcall.acoustic.scorer import AcousticDeepfakeScorer
from shieldcall.eval.protocols import clip_mean_embedding, score_acoustic_clip
from shieldcall.eval.lfcc import trim_silence
from shieldcall.eval import conditions as C
sys.path.insert(0, str(Path(__file__).parent))
from phase2_report import Calib, summarize  # noqa: E402

SEED = 20261107
N_BONA_TEST, N_PER_ATK_TEST = 100, 8
N_FIT = 150
CONDS = ["clean_16k", "g711_ulaw", "amr_nb_12k2", "ulaw_snr10"]


def pick(rows, split, nb, na):
    rng = np.random.RandomState(SEED)
    out = []
    for atk in sorted({r["attack"] for r in rows if r["split"] == split}):
        g = [r for r in rows if r["split"] == split and r["attack"] == atk]
        idx = rng.permutation(len(g))[: nb if atk == "bonafide" else na]
        out += [g[i] for i in idx]
    return out


def load(r):
    x, sr = sf.read(f"data/asvspoof2019_subset/{r['split']}/{r['utt']}.flac", dtype="float32")
    return trim_silence(x, sr)


def main():
    rows = list(csv.DictReader(open("data/manifests/asvspoof2019_la_subset.csv")))
    fit_rows = pick(rows, "train", N_FIT, 25)[: 10**9]
    t0 = time.time()
    embs, labs = [], []
    for r in fit_rows:
        e = clip_mean_embedding(load(r), 16000)
        if e is not None:
            embs.append(e); labs.append(r["label"] == "1")
    scorer = AcousticDeepfakeScorer(seed=0)
    scorer.fit(embs, labs)
    print("fit", len(embs), round(time.time() - t0), "s", flush=True)
    dev_rows = pick(rows, "dev", 100, 17)
    ds = [score_acoustic_clip(load(r), 16000, scorer) for r in dev_rows]
    dy = np.array([int(r["label"]) for r in dev_rows])
    cal = Calib(dy, np.array(ds))
    print("dev", round(time.time() - t0), "s", flush=True)
    test_rows = pick(rows, "test", N_BONA_TEST, N_PER_ATK_TEST)
    res = {}; lat = []
    for cond in CONDS:
        s, y = [], []
        for r in test_rows:
            x = load(r)
            yv, sr = C.apply(cond, x, C.seed_for(r["utt"], cond))
            t1 = time.perf_counter()
            s.append(score_acoustic_clip(yv, sr, scorer)); y.append(int(r["label"]))
            lat.append((time.perf_counter() - t1, len(yv) / sr))
        res[cond] = summarize(np.array(y), np.array(s), cal)
        print(cond, res[cond]["eer"], round(time.time() - t0), "s", flush=True)
    out = dict(note="repository AcousticDeepfakeScorer; subset sizes: fit %d clips, dev %d, test %d per condition" % (len(embs), len(dev_rows), len(test_rows)),
               dev_eer=cal.dev_eer, threshold=cal.thr, by_condition=res,
               latency_s_per_audio_s=float(sum(a for a, _ in lat) / sum(b for _, b in lat)))
    Path("docs/results/phase2_repo_acoustic.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
