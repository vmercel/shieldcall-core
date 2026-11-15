# Benchmark phases

All benchmark phases live here. Each folder has its own `PROTOCOL.md` (written before the run), run script, code and README. Run any phase from the repository root, for example `bash phases/phase10/run.sh smoke`. Folders `phase7` to `phase10` were moved here from the repository root; the protocols keep their original wording.

| Phase | What | ASVspoof 2021 LA pooled EER / min t-DCF |
|---|---|---|
| 7 | Fixed Phase 3b recipe (pooled wav2vec2 + logistic regression) | 13.42% / 0.533 |
| 8 | Truncated wav2vec2, channel views, Griffin-Lim pseudo-spoofs, selected on 2019 dev | 11.78% / 0.4421 (best so far) |
| 9 | Frame-level temporal head, HiFi-GAN pseudo-spoofs | 14.51% / 0.4911 (rejected, worse than Phase 8) |
| 10 | Held-out proxy-attack-family validation, RawBoost-style augmentation, one-class loss option | 12.63% / 0.4363 (EER worse than Phase 8, min t-DCF marginally lower) |

Organizer baselines on the same trials: LFCC-LCNN 9.26% / 0.345, RawNet2 9.50% / 0.426, CQCC-GMM 15.62% / 0.497, LFCC-GMM 19.30% / 0.576. The 2021 eval set has been scored once per phase; see each protocol for the disclosure. Result write-ups: `docs/results/PHASE7_ASVSPOOF2021_LA.md`, `PHASE8_ASVSPOOF2021_LA.md`, `PHASE9_ASVSPOOF2021_LA.md`, `PHASE10_ASVSPOOF2021_LA.md`.
