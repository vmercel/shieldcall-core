# Phase 9 protocol: does a frame-level temporal head close the gap to the organizer baselines on ASVspoof 2021 LA?

Written and committed before any Phase 9 model was scored on ASVspoof 2021. All design choices below are fixed here; selection uses ASVspoof 2019 LA train/dev only.

## Where we stand (all on the same 148,176 official 2021 LA eval trials)
| System | pooled EER % | pooled min t-DCF |
|---|---|---|
| LFCC-GMM (organizer) | 19.30 | 0.576 |
| CQCC-GMM (organizer) | 15.62 | 0.497 |
| RawNet2 (organizer) | 9.50 | 0.426 |
| LFCC-LCNN (organizer) | 9.26 | 0.345 |
| ShieldCall Phase 7 | 13.42 (CI 13.14-13.68) | 0.533 |
| ShieldCall Phase 8 | 11.78 (CI 11.54-12.02) | 0.4421 |

Phase 8 per-codec EER: none 9.09, alaw 10.56, g722 9.29, gsm 13.79, opus 10.24, pstn 16.35, ulaw 10.66. Per attack: A10 35.8 (the main failure), A11 15.4, A12 13.0, A15 13.0, A07 11.8, A08 8.1, A14 7.2, A16 7.1, A18 6.1, A09 3.9, A17 3.1, A19 2.7, A13 0.7.

## What Phase 8 left on the table (the reasons for the changes)
1. Temporal structure was discarded: mean+std pooling over time keeps no ordering or local pattern. The best baseline (LFCC-LCNN) is a convolutional model over time-frequency frames. Phase 9 keeps frame-level wav2vec2 features and learns a small temporal head (dilated 1-D convolutions, attentive mean+std pooling). Using SSL frame features with a trained sequence back-end is published practice (e.g. SSL + AASIST); we claim no novelty.
2. Scale: Phase 8's final model saw about 6.5k of the 25,380 training clips. Phase 9 uses all 2,580 bona fide and 12,000 spoofed training clips (2,000 per attack), plus about 2.4k dev clips in the final refit.
3. A10 (vocoder-type attack, 35.8% EER): add copy-synthesis pseudo-spoofs from a pretrained neural vocoder (HiFi-GAN, microsoft/speecht5_hifigan) next to Griffin-Lim. Published idea. Whether it helps cannot be measured on 2019 dev (no vocoder-only attack there), so it is admitted by the same rule as Phase 8: only if it does not hurt the dev selection score.

## Fixed recipe
- Backbone: wav2vec2-base truncated to the first 4 transformer layers. Features: mean of layers 1-2 and mean of layers 3-4 (768 each, 1536 total), 50 fps averaged in pairs to 25 fps, first 3.0 s after silence trimming (same policy at train, dev and eval).
- Channel views: the Phase 8 training and held-out channel lists, unchanged. Training views per clip: bona fide clean + 2 channels + 3 pseudo-spoof views (Griffin-Lim clean, HiFi-GAN clean, HiFi-GAN + channel); spoofed clean + 1 channel. Dev views: clean, 1 seen, 2 held-out channels.
- Head: input dropout -> linear -> 0-3 residual dilated conv blocks (kernel 3, dilation 1,2,4) -> attentive mean+std pooling -> MLP -> logit. Class-balanced BCE, AdamW, one-cycle LR, time-masking and feature noise.
- Candidates (at most these four): c1 conv (no pseudo-spoofs); c2 conv + GL/HiFi-GAN pseudo-spoofs; c3 wider conv + pseudo-spoofs; c4 attentive pooling only + pseudo-spoofs.
- Selection score (identical to Phase 8): 0.5 x mean(dev EER on clean, seen-channel, held-channel views; 0 dB noise excluded) + 0.5 x EER on dev attacks A05,A06 with a model trained without A05,A06. Lowest wins. Selection trains on a fixed 50% subset of the training views, 5 epochs.
- Final model: winning config refit on 2019 train+dev, 3 seeds averaged (logit mean), 8 epochs. The model hash is printed and stored before any 2021 audio is scored.

## Evaluation and disclosure
- Official 2021 LA eval package and keys; pooled and per-codec EER, min t-DCF, per-attack EER, bootstrap CI, paired comparison against the Phase 7 and Phase 8 scores on identical trials.
- 2021 LA eval has now been scored by three different systems (Phase 7, 8, 9). Phase 8 and 9 design choices were informed by the aggregate pattern of the previous 2021 results (point 1-3 above), so this is a repeated look at the same set, not an untouched test. Nothing is tuned after seeing the Phase 9 numbers; one final model, one evaluation. All three results are reported whatever the outcome.
- Not used here, deliberately: ASVspoof 5 audio for training (kept as a possible out-of-domain test for a later phase).

## What this does not show
Real-call performance. 2021 LA source speech is anechoic VCTK; the numbers describe a lab benchmark only. The recipe is assembled from published components.
