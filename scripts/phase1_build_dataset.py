#!/usr/bin/env python3
"""Phase 1 - data foundation: build and LOCK the benchmark manifest.

Source: ASVspoof 2019 Logical Access (Edinburgh DataShare, handle 10283/3336, licence ODC-By 1.0)
        + BothBosu/scam-dialogue (Hugging Face, Apache-2.0, LLM-generated *synthetic* text).

What this does (deterministic, seed=20261107):
  1. Reads the official protocol files. Official splits are speaker-disjoint
     (train/dev/eval) and eval contains attack families never seen in train.
  2. Draws a stratified subset (sized for a single CPU) per split and per attack id.
  3. Extracts those FLAC files, records SHA-256 of each file.
  4. Verifies speaker-disjointness and records which eval attacks were seen in training.
  5. Writes manifests + a lock file whose hash identifies the held-out test set.
     Thresholds / calibration may only use the 'dev' split; the 'test' split is
     read by phase2 only for scoring.

Usage:  python scripts/phase1_build_dataset.py --zip /path/LA.zip --out data/asvspoof2019_subset
"""
from __future__ import annotations

import argparse, csv, hashlib, json, random, sys, zipfile, io, time
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261107
QUOTA = {  # (n_bona, n_per_attack)
    "train": (2580, 430),   # 2580 bona + 6 attacks x 430
    "dev": (1000, 170),
    "test": (1000, 100),    # official 'eval': 1000 bona + 13 attacks x 100
}
PROTO = {
    "train": "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "test": "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
}
AUDIO_DIR = {"train": "LA/ASVspoof2019_LA_train/flac", "dev": "LA/ASVspoof2019_LA_dev/flac", "test": "LA/ASVspoof2019_LA_eval/flac"}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="data/asvspoof2019_subset")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--lock", default="docs/results/phase1_lock.json")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mdir = Path(a.manifest_dir); mdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    z = zipfile.ZipFile(a.zip)
    rows = []
    zip_names = set(z.namelist())
    for split, pf in PROTO.items():
        entries = []
        for line in z.read(pf).decode().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            spk, utt, _, attack, key = p[:5]
            entries.append(dict(split=split, speaker=spk, utt=utt, attack=attack if key == "spoof" else "bonafide", label=int(key == "spoof")))
        by = defaultdict(list)
        for e in entries:
            by[e["attack"]].append(e)
        nb, npa = QUOTA[split]
        for atk, lst in sorted(by.items()):
            lst = sorted(lst, key=lambda e: e["utt"])
            rng.shuffle(lst)
            take = nb if atk == "bonafide" else npa
            rows += lst[:take]
        print(split, {k: min(len(v), nb if k == "bonafide" else npa) for k, v in sorted(by.items())})
    # extract + hash
    t0 = time.time()
    for i, r in enumerate(rows):
        name = f"{AUDIO_DIR[r['split']]}/{r['utt']}.flac"
        data = z.read(name)
        r["sha256"] = sha256(data)
        r["bytes"] = len(data)
        dst = out / r["split"] / f"{r['utt']}.flac"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        if i % 2000 == 0:
            print("extracted", i, len(rows), round(time.time() - t0), "s", flush=True)
    rows.sort(key=lambda r: (r["split"], r["utt"]))
    # checks
    spk = {s: {r["speaker"] for r in rows if r["split"] == s} for s in PROTO}
    disjoint = {f"{x}&{y}": len(spk[x] & spk[y]) for x, y in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    train_atk = {r["attack"] for r in rows if r["split"] == "train" and r["label"]}
    test_atk = {r["attack"] for r in rows if r["split"] == "test" and r["label"]}
    for r in rows:
        r["attack_seen_in_train"] = int(r["attack"] in train_atk) if r["label"] else ""
    with open(mdir / "asvspoof2019_la_subset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "speaker", "utt", "attack", "label", "attack_seen_in_train", "sha256", "bytes"])
        w.writeheader(); w.writerows(rows)
    mbytes = (mdir / "asvspoof2019_la_subset.csv").read_bytes()
    test_hash = sha256("\n".join(r["sha256"] for r in rows if r["split"] == "test").encode())
    lock = {
        "locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED, "source": "ASVspoof 2019 LA, https://datashare.ed.ac.uk/handle/10283/3336 (ODC-By 1.0)",
        "zip_sha256": None,
        "counts": {s: dict(Counter(r["attack"] for r in rows if r["split"] == s)) for s in PROTO},
        "speakers": {s: len(v) for s, v in spk.items()},
        "speaker_overlap": disjoint,
        "attacks_train": sorted(train_atk), "attacks_test": sorted(test_atk),
        "test_attacks_unseen_in_train": sorted(test_atk - train_atk),
        "manifest_sha256": sha256(mbytes), "heldout_test_audio_sha256": test_hash,
        "rule": "Thresholds, calibration and model fitting use train/dev only. 'test' is scored once per configuration.",
    }
    assert all(v == 0 for v in disjoint.values()), disjoint
    Path(a.lock).parent.mkdir(parents=True, exist_ok=True)
    Path(a.lock).write_text(json.dumps(lock, indent=2))
    print(json.dumps({k: lock[k] for k in ("speakers", "speaker_overlap", "test_attacks_unseen_in_train", "manifest_sha256", "heldout_test_audio_sha256")}, indent=1))


if __name__ == "__main__":
    main()
