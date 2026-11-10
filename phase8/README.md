# Phase 8: cheaper backbone, pseudo-spoofs, one consistent channel policy

Goal: improve on Phase 7 (ASVspoof 2021 LA pooled EER 13.42%) under a small compute budget. Everything is selected on ASVspoof 2019 LA train/dev; 2021 is only scored.
See `PROTOCOL.md` for the design, the selection table and what was rejected.

Run on the 2021 evaluation (reuses the Phase 7 download if present):

    git pull && bash phase8/run.sh smoke      # a few hundred clips
    bash phase8/run.sh full                   # 148,176 trials; resumable

Optional paired comparison against your Phase 7 scores: `REF=phase7/out/scores_full.txt bash phase8/run.sh full`.
Components (Griffin-Lim pseudo-spoofs, SSL layer features, codec augmentation) are published ideas; no novelty claim. 2021 LA is anechoic VCTK speech, not real calls.
