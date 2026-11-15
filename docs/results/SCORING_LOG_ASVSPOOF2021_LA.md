# ASVspoof 2021 LA: every scoring of the evaluation set

The 2021 LA evaluation set (148,176 trials, official keys and scorer) was scored four times, with models trained on ASVspoof 2019 LA data only. Later versions were designed after seeing earlier aggregate results, so this is a repeated look, not an untouched test. Only the best-performing recipe's code is kept in `phases/`; versions that did not improve on it were removed, and their numbers are kept here so the count of looks stays honest.

| Version | Pooled EER (95% CI) | min t-DCF | Outcome |
|---|---|---|---|
| Phase 7: pooled wav2vec2 + logistic regression | 13.42% (13.14-13.68) | 0.533 | starting point (code kept) |
| Phase 8: truncated wav2vec2, channel views, Griffin-Lim pseudo-spoofs | 11.78% (11.54-12.02) | 0.4421 | best (code kept) |
| Phase 9: frame-level temporal head, HiFi-GAN pseudo-spoofs | 14.51% (14.24-14.73) | 0.4911 | worse, removed |
| Phase 10: held-out attack-family validation, RawBoost-style augmentation, one-class option | 12.63% (12.42-12.90) | 0.4363 | EER worse than Phase 8 (+0.85, paired CI 0.57-1.14); t-DCF marginally lower, unconfirmed; removed |

Organizer baselines on the same trials: LFCC-LCNN 9.26% / 0.345, RawNet2 9.50% / 0.426, CQCC-GMM 15.62% / 0.497, LFCC-GMM 19.30% / 0.576. Best version is third of five. Source speech is anechoic VCTK, not real calls.
