#!/usr/bin/env python3
"""Phase 2b - language baselines on BothBosu/scam-dialogue (Apache-2.0; LLM-GENERATED synthetic dialogues,
NOT recordings of real calls). English only. Writes per-dialogue scores; metrics come from phase2_report.py.

Split: seeded 50/50 stratified by (type) into 'dev' (threshold/calibration/TF-IDF fit) and 'test' (scored only).
Systems:
  wide_bag : shieldcall wide_lexicon_score on the whole dialogue (lexicon locked 2026-08-22, never tuned on this data)
  repo     : LinguisticFraudScorer streamed turn by turn; score = max fraud_prob over the call (declared before running)
  tfidf_lr : TF-IDF (1-2 grams) + logistic regression fit on dev (standard text baseline)
Noise axis: simulated ASR word errors via shieldcall.linguistic.asr_noise.degrade_turns at WER 0 / 0.15 / 0.30.
"""
from __future__ import annotations
import _repo  # noqa: F401
import argparse, csv, gzip, hashlib, json, random, re, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from shieldcall.linguistic.scorer import LinguisticFraudScorer
from shieldcall.linguistic.discourse import wide_lexicon_score
from shieldcall.linguistic.asr_noise import degrade_turns

URL = "https://huggingface.co/datasets/BothBosu/scam-dialogue/resolve/main/scam-dialogue_all.csv"
SEED = 20261107
WERS = (0.0, 0.15, 0.30)


def turns_of(d: str):
    parts = re.split(r"\b(?=(?:caller|receiver):)", d.strip())
    return [p.strip() for p in parts if p.strip()]


def repo_score(turns):
    sc = LinguisticFraudScorer()
    best = 0.0
    for i, t in enumerate(turns):
        best = max(best, sc.update(t, 0.8 * i).fraud_prob)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/work/scam-dialogue_all.csv")
    ap.add_argument("--out", default="docs/results/phase2_language_scores.csv.gz")
    ap.add_argument("--manifest", default="data/manifests/scam_dialogue_manifest.csv")
    a = ap.parse_args()
    cp = Path(a.cache); cp.parent.mkdir(parents=True, exist_ok=True)
    if not cp.exists():
        urllib.request.urlretrieve(URL, cp)
    df = pd.read_csv(cp)
    file_sha = hashlib.sha256(cp.read_bytes()).hexdigest()
    rng = np.random.RandomState(SEED)
    df["id"] = [f"sd{i:04d}" for i in range(len(df))]
    df["split"] = "test"
    for t, g in df.groupby("type"):
        idx = rng.permutation(g.index.values)
        df.loc[idx[: len(idx) // 2], "split"] = "dev"
    df["sha256"] = [hashlib.sha256(s.encode()).hexdigest() for s in df.dialogue]
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    df[["id", "split", "type", "label", "sha256"]].to_csv(a.manifest, index=False)
    dev, test = df[df.split == "dev"], df[df.split == "test"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(dev.dialogue)
    lr = LogisticRegression(C=3.0, max_iter=1000).fit(vec.transform(dev.dialogue), dev.label)
    rows = []; lat = {"repo_ms_per_turn": [], "tfidf_ms_per_dialogue": []}
    for part, d in (("dev", dev), ("test", test)):
        for wer in WERS:
            for k, r in enumerate(d.itertuples()):
                tr = turns_of(r.dialogue)
                if wer > 0:
                    tr = degrade_turns(tr, wer=wer, seed=SEED + k)
                text = " ".join(tr)
                t0 = time.perf_counter(); s_repo = repo_score(tr); t1 = time.perf_counter()
                s_wide = wide_lexicon_score(text)
                t2 = time.perf_counter(); s_lr = float(lr.decision_function(vec.transform([text]))[0]); t3 = time.perf_counter()
                lat["repo_ms_per_turn"].append(1000 * (t1 - t0) / max(1, len(tr)))
                lat["tfidf_ms_per_dialogue"].append(1000 * (t3 - t2))
                rows.append(dict(id=r.id, part=part, type=r.type, label=int(r.label), wer=wer, turns=len(tr),
                                 repo=s_repo, wide_bag=s_wide, tfidf_lr=s_lr))
        print(part, "done", flush=True)
    with gzip.open(a.out, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(dict(source_sha256=file_sha, n=len(df), n_dev=len(dev), n_test=len(test),
                   repo_ms_per_turn_mean=float(np.mean(lat["repo_ms_per_turn"])),
                   repo_ms_per_turn_p95=float(np.percentile(lat["repo_ms_per_turn"], 95)),
                   tfidf_ms_per_dialogue_mean=float(np.mean(lat["tfidf_ms_per_dialogue"]))),
              open("data/work/language_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
