# Phase 8: ASVspoof 2021 LA, full evaluation set

Run by the user on a Mac (194.7 min, 4 workers, 0 decode errors) on all 148,176 official 'eval' trials. Model: `phase8/model8.npz` (sha256 ec4b5359...cfcaa), trained only on ASVspoof 2019 LA train+dev. Scored with the organizers' evaluation package and official keys.

| System | Pooled EER % | Pooled min t-DCF |
|---|---|---|
| ShieldCall Phase 8 | 11.78 (95% CI 11.54-12.02) | 0.4421 |
| ShieldCall Phase 7 | 13.42 | 0.533 |
| LFCC-GMM (organizer) | 19.30 | 0.576 |
| CQCC-GMM (organizer) | 15.62 | 0.497 |
| RawNet2 (organizer) | 9.50 | 0.426 |
| LFCC-LCNN (organizer) | 9.26 | 0.345 |

Per-codec EER %: none 9.09, alaw 10.56, g722 9.29, gsm 13.79, opus 10.24, pstn 16.35, ulaw 10.66.
Per-attack EER %: A10 35.8, A11 15.4, A12 13.0, A15 13.0, A07 11.8, A08 8.1, A14 7.2, A16 7.1, A18 6.1, A09 3.9, A17 3.1, A19 2.7, A13 0.7.

## Reading it honestly
- Better than Phase 7 (13.42 -> 11.78 EER, 0.533 -> 0.4421 min t-DCF) and better than LFCC-GMM and CQCC-GMM on both metrics. Still worse than RawNet2 and LFCC-LCNN: 3rd of 5 on both. Not state of the art, not better than the best organizer baselines.
- Main failure: A10 (vocoder-type), 35.8% EER. Weakest codec: PSTN, 16.35%.
- Caveats: the 2021 set was scored once before (Phase 7) and Phase 8 design choices were informed by that aggregate pattern (see phase8/PROTOCOL.md); baselines' recipes differ; 2021 LA source speech is anechoic VCTK, so this is not real-call performance.
