#!/usr/bin/env python3
"""Phase 3 data: extract the FULL official train split and a stratified dev sample (train/dev only; the locked
test split is untouched). Writes data/manifests/asvspoof2019_la_trainfull.csv (hashes included)."""
import argparse, csv, hashlib, random, zipfile
from collections import defaultdict
from pathlib import Path
SEED = 20261108
ap = argparse.ArgumentParser(); ap.add_argument("--zip", required=True); a = ap.parse_args()
z = zipfile.ZipFile(a.zip); rng = random.Random(SEED); rows = []
P = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.%s.txt"
for split, pf, aud, quota in (("train", "train.trn", "LA/ASVspoof2019_LA_train/flac", None), ("dev", "dev.trl", "LA/ASVspoof2019_LA_dev/flac", (1500, 250))):
    by = defaultdict(list)
    for l in z.read(P % pf).decode().splitlines():
        p = l.split()
        if len(p) >= 5:
            by["bonafide" if p[4] == "bonafide" else p[3]].append(dict(split=split, speaker=p[0], utt=p[1], attack="bonafide" if p[4] == "bonafide" else p[3], label=int(p[4] == "spoof")))
    for k, v in sorted(by.items()):
        v.sort(key=lambda r: r["utt"]); rng.shuffle(v)
        rows += v if quota is None else v[: quota[0] if k == "bonafide" else quota[1]]
out = Path("data/asvspoof2019_full"); 
for r in rows:
    b = z.read(f"{'LA/ASVspoof2019_LA_train/flac' if r['split']=='train' else 'LA/ASVspoof2019_LA_dev/flac'}/{r['utt']}.flac")
    r["sha256"] = hashlib.sha256(b).hexdigest()
    d = out / r["split"]; d.mkdir(parents=True, exist_ok=True); (d / f"{r['utt']}.flac").write_bytes(b)
with open("data/manifests/asvspoof2019_la_trainfull.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["split", "speaker", "utt", "attack", "label", "sha256"]); w.writeheader(); w.writerows(rows)
print(len(rows), {s: sum(r["split"] == s for r in rows) for s in ("train", "dev")})
