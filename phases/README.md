# Benchmark phases

Only recipes that improved on the previous best are kept here. Each folder has its `PROTOCOL.md` (written before the run), run script and README. Run from the repository root, for example `bash phases/phase8/run.sh smoke`.

| Phase | What | ASVspoof 2021 LA pooled EER / min t-DCF |
|---|---|---|
| 7 | Pooled wav2vec2 + logistic regression (starting point) | 13.42% / 0.533 |
| 8 | Truncated wav2vec2, channel views, Griffin-Lim pseudo-spoofs, selected on 2019 dev (best) | 11.78% / 0.4421 |

Organizer baselines on the same trials: LFCC-LCNN 9.26% / 0.345, RawNet2 9.50% / 0.426, CQCC-GMM 15.62% / 0.497, LFCC-GMM 19.30% / 0.576. Every scoring of the 2021 set, including versions that were removed for not improving, is listed in `docs/results/SCORING_LOG_ASVSPOOF2021_LA.md`.
