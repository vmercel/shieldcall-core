# Phase 8 protocol: can the ShieldCall acoustic stream reach the organizers' baselines on ASVspoof 2021 LA?

Written before any Phase 8 model was scored on ASVspoof 2021. Phase 7 (fixed Phase 3b recipe) gave pooled EER 13.42% / min t-DCF 0.533 against
organizer baselines of 19.30 / 15.62 / 9.26 / 9.50 (EER). The goal here is to close the gap to the best baseline by design choices
made on ASVspoof 2019 data only.

## What Phase 7 taught us (the reasons for the changes)
1. The Phase 3b model was fit on only 2,400 clips with 2.5 s of audio each. The official 2019 LA training set has 25,380 clips. Scale is the first lever.
2. Training audio had leading/trailing silence trimmed (the 2019 silence shortcut), the 2021 audio scored in Phase 7 did not. Phase 8 applies one
   identical policy (trim or not, chosen on dev) at train, dev and eval time.
3. Weakest 2021 attacks in Phase 7 were A10 and A11 (vocoder-type). Candidate remedy: Griffin-Lim copy-synthesis pseudo-spoofs made from bona fide
   training clips (a published idea, not ours). Dev cannot measure its benefit (2019 train/dev contain no vocoder-only attack), so it is only
   admitted if it does not hurt dev or the leave-attacks-out check.
4. Pooled statistics of one frozen backbone leave information unused: candidates are other backbones (WavLM-base-plus, XLS-R-300M truncated to
   12 layers), all-layer feature sets, a per-layer committee, and a small MLP ensemble.

## Data rules
- Training and selection use ASVspoof 2019 LA train and dev only (HF mirror Bisher/ASVspoof_2019_LA, ODC-By). 2021 audio, keys and scores are never read by
  `phase8_extract.py` or `phase8_fit.py`.
- Channel views: real codec round trips through PyAV (G.711 u/A-law, AMR-NB, Opus, G.722) plus noise and two simulated telephone-network profiles.
  Training channels: g711 u-law and A-law, AMR-NB 12.2, Opus-NB 12k, u-law + 20 dB and 10 dB noise, G.722, band-limit, simulated degraded PSTN, simulated harsh VoIP.
  Held out (dev judging only): AMR-NB 4.75, Opus-WB 24k, u-law with 10% packet loss, u-law + 0 dB noise.
  The 2021 codec list is public, so A-law and G.722 and PSTN/VoIP-like channels were chosen for training; no 2021 audio was used.
- Revision note: a first screen of the base model (dev only) showed that the 0 dB-noise view (EER 44%) and the simulated PSTN view (20%) dominated the held-out
  group. 0 dB noise is not a 2021 condition, so it is now a reported stress test and excluded from selection, and the PSTN/VoIP simulators moved into training
  because 2021 contains a real PSTN condition. Both changes were made on 2019 dev data only, before any Phase 8 score on 2021.

## Selection rule (fixed in code: `phase8_fit.py select`)
score = 0.5 x mean(EER clean, EER seen channels, EER held-out channels on dev; 0 dB noise excluded) + 0.5 x EER on dev attacks A05,A06 with a model that never saw A05,A06 in training.
Lowest score wins. Backbones are screened on identical subsets first; the cheapest backbone within 0.5 EER points of the best is kept.

## Final evaluation
- Model refit on 2019 train+dev, then scored on 2021 LA eval with the organizers' evaluation package: pooled and per-codec EER, min t-DCF,
  bootstrap CI, and a paired comparison with the Phase 7 scores on the identical trials.
- Disclosure: the 2021 eval set was already scored once (Phase 7). Phase 8 choices were informed by Phase 7's aggregate pattern (point 2 and 3 above), so
  this is a second look at the same set, not an untouched test. One final model, one evaluation; no tuning after seeing the 2021 numbers.
- Everything is reported whatever the outcome, including per-attack and per-codec results and the model hash.

## What this does not show
Real-call performance. 2021 LA source speech is anechoic VCTK; the numbers describe a lab benchmark only.

## Cheap-compute selection (final configuration, 2019 train/dev only)
Backbone truncated to the first 4 transformer layers of wav2vec2-base (about 3x cheaper than the 12-layer pass), training 4,000 clips x 3 channel views
plus pseudo-spoofs, dev 1,500 clips x 4 views. Selection score as above (lower is better). Full table in `dev_selection_results.json`.

| Candidate (layers 1-4) | clean | seen | held | leave-attacks-out | score |
|---|---|---|---|---|---|
| LR, no pseudo-spoofs (Phase 7-like) | 1.87 | 4.13 | 9.06 | 24.38 | 14.70 |
| **LR, Griffin-Lim pseudo-spoofs (chosen)** | 1.87 | 4.27 | 9.59 | 16.60 | **10.92** |
| LR, 5 pseudo-spoof families | 2.67 | 5.20 | 11.60 | 17.58 | 12.04 |
| MLP ensemble, 5 families | 2.00 | 4.13 | 8.79 | 16.78 | 10.88 |
| LFCC-statistics head alone | 33.3 | 37.6 | 36.6 | 45.8 | 40.8 |
| MLP + LFCC score fusion (w 0.1-0.5) | worse in every case (11.8-21.2) |

Tried and rejected on dev: score fusion with handcrafted LFCC statistics, per-utterance normalisation of layer statistics (no change), extra pseudo-spoof
families (LPC source-filter, random-phase, mu-law quantisation: no gain for LR), 0 dB-noise view (reported as stress only).
Chosen: linear head (logistic regression), simplest candidate within 0.5 points of the best. `model8.npz` was refit on 2019 LA train+dev only
(19,600 views). sha256 ec4b535916128ba7ea19fb59426b783b7297253715200bb188e3a126953cfcaa.

Run on ASVspoof 2021 LA (never used for any decision above):  `bash phase8/run.sh smoke|sample|full`  (set `REF=/path/to/phase7/scores_full.txt` for the paired comparison).
Expected result is unknown; Phase 7 pooled EER was 13.42%. Report whatever it is.
