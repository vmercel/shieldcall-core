# Phase 7: ASVspoof 2021 LA, full evaluation set (fixed Phase 3b recipe)

Run by the author on a Mac (280 min, 4 workers, 0 decode errors), 148,176 'eval' trials (14,816 bona fide, 133,360 spoof). Model sha256 62aca8ec...05f0b7, trained only on an ASVspoof 2019 LA subset (3,300 views); nothing tuned on 2021. Scored with the organizers' eval-package and official keys. The package crashed on a newer matplotlib (cm.get_cmap), so the official tables were recomputed from the author's unchanged score file (scores_full.txt.gz, from the run report); the author's own EER code agrees (13.42%, 95% CI 13.14-13.68).

## Pooled and per-codec EER (%), official scorer, same 148,176 trials
| System | C1 none | C2 alaw | C3 pstn | C4 g722 | C5 ulaw | C6 gsm | C7 opus | Pooled EER | Pooled min t-DCF |
|---|---|---|---|---|---|---|---|---|---|
| ShieldCall Phase 3b | 11.34 | 13.31 | 11.76 | 12.29 | 13.33 | 13.80 | 12.05 | 13.42 | 0.533 |
| LFCC-GMM (organizer) | 12.72 | 21.21 | 35.55 | 15.28 | 18.76 | 18.46 | 12.73 | 19.30 | 0.576 |
| CQCC-GMM (organizer) | 10.57 | 14.76 | 20.58 | 11.61 | 13.58 | 14.01 | 11.21 | 15.62 | 0.497 |
| LFCC-LCNN (organizer) | 6.71 | 8.89 | 12.02 | 6.34 | 9.25 | 11.00 | 6.66 | 9.26 | 0.345 |
| RawNet2 (organizer) | 5.84 | 6.59 | 16.72 | 6.41 | 6.33 | 10.66 | 7.95 | 9.50 | 0.426 |

## Reading it honestly
- Pooled EER: better than LFCC-GMM and CQCC-GMM, worse than LFCC-LCNN and RawNet2 (3rd of 5). Pooled min t-DCF: 4th of 5 (worse than CQCC-GMM). This is mid-table, not state of the art.
- Channel stability is the one distinctive result: our EER moves only 11.3 -> 13.8 across the six telephony codecs and none, while LFCC-GMM goes 12.7 -> 35.6 and RawNet2 5.8 -> 16.7 on PSTN. That fits the Phase 3b design (noise/channel views, SSL features), but it is an observation from one run, not a proven cause.
- Weak spots: attack A10 33.8% EER, A11 20.7%; all others 3.3-14.7% (A13 3.3).
- Caveats: the organizer baselines' training data/recipes differ from ours (I believe 2019 LA training data; not re-verified); no seeds or confidence intervals beyond the bootstrap for pooled EER; source speech is anechoic VCTK, so this is codec robustness, not real-call performance; min t-DCF depends on the ASV system supplied with the keys.
- Not claimable: state of the art, better than organizer baselines, real-world accuracy.
