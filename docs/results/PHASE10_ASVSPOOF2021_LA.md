# Phase 10: ASVspoof 2021 LA result (full set, 148,176 trials, 0 errors)

Training and selection used ASVspoof 2019 LA train and dev only. This was the fourth scoring of the 2021 LA evaluation set; it is a repeated look, not an untouched test.

| | Pooled EER (95% CI) | min t-DCF |
|---|---|---|
| Phase 7 | 13.42% (13.14-13.68) | 0.533 |
| Phase 8 | 11.78% (11.54-12.02) | 0.4421 |
| Phase 9 | 14.51% (14.24-14.73) | 0.4911 |
| **Phase 10** | **12.63% (12.42-12.90)** | **0.4363** |
| RawNet2 (organizers) | 9.50% | 0.426 |
| LFCC-LCNN (organizers) | 9.26% | 0.345 |
| CQCC-GMM / LFCC-GMM | 15.62% / 19.30% | 0.497 / 0.576 |

Paired against Phase 8: EER +0.85 points (CI +0.57 to +1.14), i.e. worse. Paired against Phase 9: -1.87 points (better). The min t-DCF is 0.0058 lower than Phase 8; no interval was computed for t-DCF, so treat that as small and unconfirmed.

Verdict: mixed. EER worse than Phase 8, min t-DCF marginally lower. Rank is still 3rd of 5 on both metrics. Not better than RawNet2 or LFCC-LCNN.

Per-codec EER: none 7.39, alaw 12.73, g722 8.16, gsm 18.45, opus 9.34, pstn 15.17, ulaw 13.38. Versus Phase 8 (9.09, 10.56, 9.29, 13.79, 10.24, 16.35, 10.66): better on none, g722, opus, pstn; worse on alaw, gsm, ulaw.
Per-attack EER: A07 11.8, A08 5.3, A09 2.5, A10 35.4, A11 24.3, A12 13.8, A13 1.2, A14 6.7, A15 15.4, A16 4.7, A17 2.4, A18 8.8, A19 0.8.
Largest remaining failures: A10 (35.4%) and A11 (24.3%); A10 was also the worst in Phases 5-9.

Dev selection (2019 only; lower is better): c1 conv+BCE L4 7.7, c2 conv+OC L4 8.2, c3 conv+OC L8 8.7, c4 conv+pooled expert+BCE L4 7.6 (winner). The one-class loss and the 8-layer features did not help on dev. Leave-family-out EER was 16-18% for every config, so the proxy attack families are hard but did not predict A10/A11.
Model sha256 c38aefad5151db6f1e8421225c7ccd1ae0c073e2d9f3f2bcfd16b9b99e8ede89. Scoring took 218.8 min.

All components are published ideas; no novelty claim is made.
