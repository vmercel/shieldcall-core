# Phase 9: ASVspoof 2021 LA, frame-level temporal head (negative result)

Run by the user on a Mac (157.9 min scoring, 0 decode errors) on all 148,176 official 'eval' trials. Model `model9.pt` sha256 a40630727cefa1e8ed67e758dc50de13836489832f960a3e27ce58d841b42805 (3 seeds), trained on ASVspoof 2019 LA train+dev only. Design and selection rule: `phase9/PROTOCOL.md` (committed before the run).

| System | Pooled EER % | Pooled min t-DCF |
|---|---|---|
| ShieldCall Phase 9 | **14.51** (95% CI 14.24-14.73) | 0.4911 |
| ShieldCall Phase 8 | 11.78 (11.54-12.02) | 0.4421 |
| ShieldCall Phase 7 | 13.42 | 0.533 |
| LFCC-LCNN (organizer) | 9.26 | 0.345 |
| RawNet2 (organizer) | 9.50 | 0.426 |
| CQCC-GMM (organizer) | 15.62 | 0.497 |
| LFCC-GMM (organizer) | 19.30 | 0.576 |

Paired against Phase 8 on identical trials: +2.72 EER points (95% CI +2.42 to +2.99), i.e. worse.
Per-codec EER %: none 8.83, alaw 11.38, g722 8.83, gsm 19.52, opus 9.80, pstn 20.43, ulaw 11.16.
Per-attack EER %: A07 14.5, A08 4.8, A09 6.9, A10 33.9, A11 29.0, A12 13.2, A13 1.0, A14 9.0, A15 18.8, A16 4.9, A17 4.1, A18 9.7, A19 1.4.

Dev selection (2019 only; lower is better; score = 0.5 x mean(clean, seen, held) + 0.5 x leave-attacks-out): c1 conv no pseudo-spoofs 7.75; c2 conv + GL/HiFi-GAN 5.31; **c3 wide conv + GL/HiFi-GAN 5.10 (chosen)**; c4 attentive pooling only 5.52. Phase 8's logistic-regression recipe scored 10.92 on its own dev subset (not identical subsets or views, so not strictly comparable).

## Reading it honestly
- Rejected under the project's accept rule: it does not beat the previous phase. Phase 8 (11.78%) remains the best ShieldCall result. Against the four organizer baselines alone, Phase 9 would still rank 3rd of 5 on EER (ahead of CQCC-GMM and LFCC-GMM) and on min t-DCF (0.4911 vs CQCC-GMM 0.4974), behind RawNet2 and LFCC-LCNN.
- Large dev improvement (10.9 -> 5.1) did not transfer: the temporal head fit the 2019 training attacks and channels more tightly. Clean-audio EER improved slightly (8.83% vs 9.09%), but unseen attacks A11 (29.0 vs 15.4), A15 (18.8 vs 13.0), A07 (14.5 vs 11.8) and the telephony codecs GSM (19.5 vs 13.8) and PSTN (20.4 vs 16.4) got worse. A10 did not improve (33.9 vs 35.8); the HiFi-GAN pseudo-spoofs did not fix it.
- Caveats: one run, three seeds averaged; 2021 LA was scored three times (Phases 7, 8, 9); source speech is anechoic VCTK; baselines' recipes differ.
