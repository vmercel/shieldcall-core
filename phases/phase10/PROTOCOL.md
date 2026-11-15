# Phase 10 protocol: robustness to unseen attacks, tested on ASVspoof 2021 LA

Written and committed before any Phase 10 model was scored on ASVspoof 2021. Training and selection use ASVspoof 2019 LA **train and dev only** (the organizers' own training condition), so results stay comparable to the organizer baselines.

## Where we stand (same 148,176 official 2021 LA eval trials)
| System | pooled EER % | pooled min t-DCF |
|---|---|---|
| LFCC-LCNN (organizer) | 9.26 | 0.345 |
| RawNet2 (organizer) | 9.50 | 0.426 |
| ShieldCall Phase 8 (best so far) | 11.78 | 0.4421 |
| ShieldCall Phase 7 | 13.42 | 0.533 |
| ShieldCall Phase 9 | 14.51 | 0.4911 |
| CQCC-GMM (organizer) | 15.62 | 0.497 |
| LFCC-GMM (organizer) | 19.30 | 0.576 |

## What Phase 9 taught us
Phase 9 improved the 2019 dev selection score from 10.9 to 5.1 and got worse on 2021 (11.78 -> 14.51). The unseen attacks A11 (15.4 -> 29.0) and A15 (13.0 -> 18.8) and the telephone codecs GSM and PSTN degraded. The leave-attacks-out check (A05, A06) read 5.5% on dev, because A05/A06 resemble the training attacks, so it could not see this failure. Selection was rewarding fit to 2019, not robustness to new attacks. Phase 10 changes the validation first, then the training.

## Changes (all published building blocks; no novelty claim for any single one)
1. **Validation that can see unseen attack types.** Six proxy-attack families are generated from bona fide training clips: Griffin-Lim, coarse Griffin-Lim, random-phase resynthesis, LPC source-filter vocoder, mu-law quantisation + lowpass, and HiFi-GAN neural-vocoder copy-synthesis. Candidates are scored on families they never saw in training (two folds: {LPC, mu-law, HiFi-GAN} held out, then {GL, coarse GL, random phase} held out), in addition to the held-out real attacks A05/A06 and held-out channels. Selection score = 0.4 x mean(clean, seen-channel, held-channel EER) + 0.3 x leave-attacks-out EER + 0.3 x mean leave-family-out EER on dev. 0 dB noise stays a stress test only.
2. **RawBoost-style waveform distortions** (random filtering with non-linearity, impulsive noise, coloured stationary noise) added to the training views of bona fide and spoofed clips, in addition to the Phase 8/9 telephone channels. Own implementation of the published idea.
3. **One-class loss (OC-softmax) candidates**, which compact bona fide speech instead of fitting known attacks.
4. **Depth probe:** layers 1-8 (four frame-level groups) against layers 1-4 (two groups).
5. **Joint pooled-statistics expert** (mean+std of the input features) added to the temporal head in one candidate. Trained jointly, equal weight; no weights tuned on 2021.
6. Same backbone family as Phases 7-9 (wav2vec2-base, frozen, truncated to 4 or 8 layers), same silence policy and 3 s cap at train, dev and eval time, 16.7 frames/s.

Not done, on purpose: multi-crop test-time averaging (would double 2021 scoring time), distillation, training a raw-waveform network from scratch (not cheap on a laptop CPU), any use of ASVspoof 2019 eval or ASVspoof 5 audio for training.

## Candidates (fixed here; at most four)
c1 conv head + BCE, layers 1-4; c2 conv head + OC-softmax, layers 1-4; c3 conv head + OC-softmax, layers 1-8; c4 conv head + pooled expert + BCE, layers 1-4. The winner is refit on train+dev (3 seeds, scores averaged); the model hash is printed and stored before any 2021 audio is scored.

## Evaluation and disclosure
- Official 2021 LA eval package and keys; pooled and per-codec EER, min t-DCF, per-attack EER, bootstrap CI, paired comparison with the Phase 7, 8 and 9 scores on identical trials.
- ASVspoof 2021 LA eval will have been scored four times (Phases 7, 8, 9, 10). Phase 10 design choices were informed by the aggregate per-attack and per-codec results of earlier phases (see "What Phase 9 taught us"), so this is a repeated look at the same set, not an untouched test. Nothing is tuned after seeing Phase 10 numbers; one final model, one evaluation. All results are reported whatever the outcome, including if Phase 10 does not beat Phase 8.

## What this does not show
Real-call performance. 2021 LA source speech is anechoic VCTK; the numbers describe a lab benchmark only.
