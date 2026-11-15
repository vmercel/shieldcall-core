#!/usr/bin/env python3
"""Phase 8 feature extraction from ASVspoof 2019 LA train / dev (HF mirror Bisher/ASVspoof_2019_LA, ODC-By).
Every clip is processed identically at train, dev and eval time (silence policy, length cap). Channel views are deterministic per clip.
Resumable: shards are written every SHARD clips; re-running continues. Selection order is a seeded, class-balanced, NESTED sequence,
so a screening run with --n 600 is a prefix of a full run with --n 8000 and nothing is recomputed.
"""
import argparse, glob, os, sys, time, zlib
import numpy as np
from phase8_lib import *

REPO = "Bisher/ASVspoof_2019_LA"; FILES = {"train": "data/train-00000-of-00001.parquet", "dev": "data/validation-00000-of-00001.parquet"}
SHARD = 300; SEED = 20261109


def plan(path, n, split):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    t = pf.read(columns=["audio_file_name", "system_id", "key"]).to_pydict()
    groups = {}
    for i, (u, s, k) in enumerate(zip(t["audio_file_name"], t["system_id"], t["key"])):
        groups.setdefault("bonafide" if k == 0 else s, []).append(i)
    rng = np.random.RandomState(SEED + (0 if split == "train" else 1))
    for g in groups.values(): rng.shuffle(g)
    atk = sorted(a for a in groups if a != "bonafide"); order = []; ptr = {g: 0 for g in groups}
    def take(g):
        if ptr[g] < len(groups[g]): ptr[g] += 1; return groups[g][ptr[g] - 1]
    while len(order) < n:
        got = 0
        for a in atk:
            for g in ("bonafide", a):
                i = take(g)
                if i is not None: order.append(i); got += 1
        if not got: break
    order = order[:n]
    names = {i: (t["audio_file_name"][i], "bonafide" if t["key"][i] == 0 else t["system_id"][i], int(t["key"][i] == 0)) for i in order}
    return pf, order, names


def views_for(split, utt, k):
    h = zlib.crc32(utt.encode())
    if split == "train":
        return ["clean_16k"] + [SEEN[(h + 3 * j) % len(SEEN)] for j in range(k - 1)]
    return (["clean_16k", SEEN[h % len(SEEN)], HELD[h % len(HELD)], HELD[(h + 1) % len(HELD)]] + [SEEN[(h + 3 + j) % len(SEEN)] for j in range(8)])[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev"], required=True); ap.add_argument("--backbone", default="w2v2-base", choices=list(BACKBONES))
    ap.add_argument("--n", type=int, required=True); ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--trim", type=int, default=1); ap.add_argument("--cap-s", type=float, default=4.0)
    ap.add_argument("--pseudo", type=int, default=0, help="train only: Griffin-Lim pseudo-spoofs made from this many bona fide clips")
    ap.add_argument("--fams", type=int, default=1, help="pseudo-spoof families to use (1 = Griffin-Lim only)"); ap.add_argument("--max-layer", type=int, default=0); ap.add_argument("--out", default="work/feats"); ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, FILES[a.split], repo_type="dataset")
    pf, order, names = plan(path, a.n, a.split)
    tag = f"{a.backbone}{'-L%d' % a.max_layer if a.max_layer else ''}_{a.split}_t{a.trim}_c{int(a.cap_s)}"
    done = set()
    for f in glob.glob(os.path.join(a.out, tag + "_part*.npz")):
        done |= set(np.load(f)["utt"].tolist())
    part = len(glob.glob(os.path.join(a.out, tag + "_part*.npz")))
    todo = [i for i in order if names[i][0] not in done]
    print(f"{tag}: {len(order)} planned, {len(done)} views-sets already done, {len(todo)} to do", flush=True)
    if not todo: return
    pseudo_ids = set([names[i][0] for i in order if names[i][2] == 1][: a.pseudo]) if (a.split == "train" and a.pseudo) else set()
    bb = Backbone(a.backbone, a.threads, a.max_layer); cap = int(a.cap_s * SR)
    want = set(todo); rg_off = []; o = 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows; rg_off.append((o, o + n)); o += n
    buf = {k: [] for k in ("F", "utt", "view", "y", "attack", "Q")}; t0 = time.time(); nd = 0
    def flush():
        nonlocal part, buf
        if buf["utt"]:
            np.savez(os.path.join(a.out, f"{tag}_part{part:04d}.npz"), F=np.stack(buf["F"]), utt=np.array(buf["utt"]), view=np.array(buf["view"]), y=np.array(buf["y"]), attack=np.array(buf["attack"]), Q=np.stack(buf["Q"]))
            part += 1; buf = {k: [] for k in buf}
    for g, (lo, hi) in enumerate(rg_off):
        idx = [i for i in range(lo, hi) if i in want]
        if not idx: continue
        tb = pf.read_row_group(g, columns=["audio"]).to_pydict()["audio"]
        for i in idx:
            utt, atk, y = names[i]
            try:
                x = prep(decode(tb[i - lo]["bytes"]), a.trim, cap)
                items = [(v, channel(x, v, utt) if v != "clean_16k" else x, y, atk) for v in views_for(a.split, utt, a.views)]
                if utt in pseudo_ids:
                    fm = [0] + [1 + (zlib.crc32(utt.encode()) + j) % 4 for j in range(a.fams - 1)] if a.fams > 1 else [0]
                    sc = SEEN[zlib.crc32(utt.encode()) % len(SEEN)]
                    for fam in fm:
                        gx = pseudo_spoof(x, fam, zlib.crc32(utt.encode()) % 9973); items += [(f"GL{fam}|clean_16k", gx, 0, "GL"), (f"GL{fam}|{sc}", channel(gx, sc, utt + "GL%d" % fam), 0, "GL")]
                for v, xx, yy, aa in items:
                    buf["F"].append(bb(prep(xx, 0, cap))); buf["utt"].append(utt); buf["view"].append(v); buf["y"].append(yy); buf["attack"].append(aa); buf["Q"].append(qfeat(prep(xx, 0, cap)))
            except Exception as e:
                print("skip", utt, repr(e)[:120], flush=True)
            nd += 1
            if nd % 50 == 0:
                r = nd / (time.time() - t0); print(f"{nd}/{len(todo)} clips  {r:.2f}/s  eta {(len(todo) - nd) / max(r, 1e-9) / 60:.0f} min", flush=True)
            if len(buf["utt"]) >= SHARD * a.views: flush()
    flush(); print("done", tag, round((time.time() - t0) / 60, 1), "min")


if __name__ == "__main__":
    main()
