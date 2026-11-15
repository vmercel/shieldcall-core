#!/usr/bin/env python3
"""Phase 9 frame-level feature extraction from ASVspoof 2019 LA train / dev (HF mirror Bisher/ASVspoof_2019_LA, ODC-By). Nothing from 2021 is read.
Output per split: feats_<split>.f16 (N,75,1536 float16 memmap), meta_<split>.json, done_<split>.u8. Resumable (re-run continues), multi-process.
"""
import argparse, json, os, sys, time, zlib
import numpy as np
from phase9_lib import *

REPO = "Bisher/ASVspoof_2019_LA"; FILES = {"train": "data/train-00000-of-00001.parquet", "dev": "data/validation-00000-of-00001.parquet"}
SEED = 20261114
SCALES = {  # clips per class: bona, spoof per attack
    "full": {"train": (2580, 2000), "dev": (800, 270)},
    "tiny": {"train": (30, 8), "dev": (12, 4)},
}


def views_for(split, utt, y):
    h = zlib.crc32(utt.encode()); n = len(SEEN)
    if split == "dev":
        return ["clean_16k", SEEN[h % n], HELD[h % 4], HELD[(h + 1) % 4]]
    v = ["clean_16k", SEEN[h % n]]
    if y == 1: v += [SEEN[(h + 3) % n], "GL|clean_16k", "HF|clean_16k", "HF|" + SEEN[(h + 5) % n]]   # bona fide: extra channel view + 3 pseudo-spoof views
    return v


def plan(path, split, scale):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path); t = pf.read(columns=["audio_file_name", "system_id", "key"]).to_pydict()
    groups = {}
    for i, (u, s, k) in enumerate(zip(t["audio_file_name"], t["system_id"], t["key"])): groups.setdefault("bonafide" if k == 0 else s, []).append(i)
    rng = np.random.RandomState(SEED + (split == "dev")); nb, ns = SCALES[scale][split]; clips = []
    for g in sorted(groups):
        v = groups[g][:]; rng.shuffle(v)
        for i in v[: nb if g == "bonafide" else ns]:
            y = int(g == "bonafide"); clips.append(dict(i=i, utt=t["audio_file_name"][i], atk=g, y=y))
    clips.sort(key=lambda c: c["i"]); items = []
    for c in clips:
        c["views"] = views_for(split, c["utt"], c["y"]); c["base"] = len(items)
        items += [dict(utt=c["utt"], view=v, y=0 if "|" in v else c["y"], attack="GL" if v.startswith("GL") else "HF" if v.startswith("HF") else c["atk"]) for v in c["views"]]
    return pf, clips, items


def worker(path, split, clips, N, out, w, W, threads):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path); bb = FrameBackbone("w2v2-base", threads, 4)
    X = np.memmap(f"{out}/feats_{split}.f16", np.float16, "r+", shape=(N, T_FR, DIN)); D = np.memmap(f"{out}/done_{split}.u8", np.uint8, "r+", shape=(N,))
    Tn = np.memmap(f"{out}/len_{split}.i16", np.int16, "r+", shape=(N,))
    offs = []; o = 0
    for g in range(pf.num_row_groups): n = pf.metadata.row_group(g).num_rows; offs.append((o, o + n)); o += n
    t0 = time.time(); nd = 0
    for g, (lo, hi) in enumerate(offs):
        if g % W != w: continue
        cs = [c for c in clips if lo <= c["i"] < hi and not all(D[c["base"] + j] for j in range(len(c["views"])))]
        if not cs: continue
        tb = pf.read_row_group(g, columns=["audio"]).to_pydict()["audio"]
        for c in cs:
            try:
                x = prep(decode(tb[c["i"] - lo]["bytes"]), 1, CAP)
                for j, v in enumerate(c["views"]):
                    if D[c["base"] + j]: continue
                    f, n = bb.frames(prep(view_audio(x, v, c["utt"]), 0, CAP)); X[c["base"] + j] = f; Tn[c["base"] + j] = n; D[c["base"] + j] = 1
            except Exception as e:
                print("skip", c["utt"], repr(e)[:150], flush=True)
            nd += 1
            if nd % 100 == 0: X.flush(); D.flush(); Tn.flush(); print(f"[w{w}] {nd} clips {nd / (time.time() - t0):.2f}/s", flush=True)
    X.flush(); D.flush(); Tn.flush()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=["train", "dev"]); ap.add_argument("--scale", default="full", choices=list(SCALES))
    ap.add_argument("--out", default="work9"); ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, FILES[a.split], repo_type="dataset")
    pf, clips, items = plan(path, a.split, a.scale); N = len(items)
    cpu = os.cpu_count() or 2; W = a.workers or max(1, min(4, cpu // 2)); T = max(1, cpu // W)
    mp_ = f"{a.out}/meta_{a.split}.json"
    if os.path.exists(mp_) and json.load(open(mp_))["scale"] != a.scale: sys.exit(f"{a.out} holds a different scale; use a fresh --out")
    json.dump(dict(scale=a.scale, items=items), open(mp_, "w"))
    for nm, dt, shp in ((f"feats_{a.split}.f16", np.float16, (N, T_FR, DIN)), (f"done_{a.split}.u8", np.uint8, (N,)), (f"len_{a.split}.i16", np.int16, (N,))):
        p = f"{a.out}/{nm}"
        if not os.path.exists(p) or os.path.getsize(p) != int(np.prod(shp)) * np.dtype(dt).itemsize: np.memmap(p, dt, "w+", shape=shp).flush()
    D = np.memmap(f"{a.out}/done_{a.split}.u8", np.uint8, "r", shape=(N,)); print(f"{a.split}: {len(clips)} clips, {N} views, {int(D.sum())} done, {W} workers x {T} threads", flush=True)
    import multiprocessing as mp
    ctx = mp.get_context("spawn"); ps = [ctx.Process(target=worker, args=(path, a.split, clips, N, a.out, w, W, T)) for w in range(W)]
    [p.start() for p in ps]; t0 = time.time()
    while any(p.is_alive() for p in ps):
        time.sleep(30); d = int(np.memmap(f"{a.out}/done_{a.split}.u8", np.uint8, "r", shape=(N,)).sum()); print(f"{a.split}: {d}/{N} views  {(time.time() - t0) / 60:.0f} min", flush=True)
    [p.join() for p in ps]
    d = int(np.memmap(f"{a.out}/done_{a.split}.u8", np.uint8, "r", shape=(N,)).sum()); print(f"{a.split} finished: {d}/{N} views done", flush=True)
    if d < N: print("some views were skipped (see 'skip' lines); re-running continues them", flush=True)


if __name__ == "__main__":
    main()
