#!/usr/bin/env python3
"""Phase 3: cache channel-processed audio views (int16) so every feature/model iteration sees identical inputs.
  train pool : all 2580 genuine (clean + 3 random SEEN conditions) + 1000 clips per attack A01-A06 (clean + 1 SEEN)
  val        : dev sample (genuine 1500, A01 250, A05 250) under 8 conditions (several unseen)
  test       : locked test split under all 14 conditions  (built with --test)
"""
import _repo  # noqa
import argparse, csv, pickle, random, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
from shieldcall.eval.lfcc import trim_silence
from shieldcall.eval import conditions as C

SEED = 20261108
VAL_CONDS = ["clean_16k", "g711_ulaw", "amr_nb_12k2", "opus_nb_12k", "ulaw_snr10", "ulaw_snr0", "sim_pstn", "amr_nb_4k75"]
VAL_HELDOUT = ("A01", "A05")


def load(path):
    x, sr = sf.read(path, dtype="float32"); return trim_silence(x, 16000)


def q(y):
    return (np.clip(y, -1, 1) * 32767).astype(np.int16)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--which", choices=["train", "val", "test"], required=True); a = ap.parse_args()
    rng = random.Random(SEED); out = {}
    t0 = time.time()
    if a.which in ("train", "val"):
        rows = list(csv.DictReader(open("data/manifests/asvspoof2019_la_trainfull.csv")))
        root = Path("data/asvspoof2019_full")
    if a.which == "train":
        tr = [r for r in rows if r["split"] == "train"]
        sel = [r for r in tr if r["label"] == "0"]
        for atk in ["A01", "A02", "A03", "A04", "A05", "A06"]:
            g = [r for r in tr if r["attack"] == atk]; rng.shuffle(g); sel += g[:1000]
        for i, r in enumerate(sel):
            x = load(root / "train" / f"{r['utt']}.flac")
            conds = ["clean_16k"] + [rng.choice(C.SEEN) for _ in range(3 if r["label"] == "0" else 1)]
            for k, cond in enumerate(conds):
                y, sr = C.apply(cond, x, C.seed_for(r["utt"] + str(k), cond))
                out[(r["utt"], k)] = dict(a=q(y), sr=sr, cond=cond, label=int(r["label"]), attack=r["attack"], speaker=r["speaker"])
            if i % 1000 == 0: print(i, len(sel), round(time.time() - t0), flush=True)
        pickle.dump(out, open("data/work/views_train.pkl", "wb"), protocol=4)
    elif a.which == "val":
        dv = [r for r in rows if r["split"] == "dev" and (r["label"] == "0" or r["attack"] in VAL_HELDOUT)]
        for i, r in enumerate(dv):
            x = load(root / "dev" / f"{r['utt']}.flac")
            for cond in VAL_CONDS:
                y, sr = C.apply(cond, x, C.seed_for(r["utt"], cond))
                out[(r["utt"], cond)] = dict(a=q(y), sr=sr, cond=cond, label=int(r["label"]), attack=r["attack"], speaker=r["speaker"])
            if i % 500 == 0: print(i, len(dv), round(time.time() - t0), flush=True)
        pickle.dump(out, open("data/work/views_val.pkl", "wb"), protocol=4)
    else:
        rows = list(csv.DictReader(open("data/manifests/asvspoof2019_la_subset.csv")))
        te = [r for r in rows if r["split"] == "test"]
        for i, r in enumerate(te):
            x = load(Path("data/asvspoof2019_subset/test") / f"{r['utt']}.flac")
            for cond in C.CONDITIONS:
                y, sr = C.apply(cond, x, C.seed_for(r["utt"], cond))
                out[(r["utt"], cond)] = dict(a=q(y), sr=sr, cond=cond, label=int(r["label"]), attack=r["attack"], speaker=r["speaker"])
            if i % 500 == 0: print(i, len(te), round(time.time() - t0), flush=True)
        pickle.dump(out, open("data/work/views_test.pkl", "wb"), protocol=4)
    print("done", len(out), round(time.time() - t0))


if __name__ == "__main__":
    main()
