# Phase 3 iteration log (acoustic model), 2026-11-07

Protocol. Train on ASVspoof2019 train attacks with A01 and A05 held out; validate on dev speakers with held-out attacks A01/A05; accept an iteration only if mean validation EER beats the previous accepted one by more than 0.3 points. The locked test split (A07-A19, 67 unseen speakers) is scored once per accepted recipe family. Steps 0-3 validate on 8 conditions; steps 4-5 use a cheaper 4-condition validation (clean, G.711, AMR-NB 12.2k, G.711 + 0 dB noise), so compare those against the TDNN's score on the same four (17.3%).

| # | Recipe | Val mean EER | Decision |
|---|---|---|---|
| 0 | LFCC + GMM, 8 kHz band-locked, telephone-augmented | 20.6% (8 cond) | baseline |
| 1 | + modified group-delay channel (GMM) | 20.3% | rejected |
| 1b | + harmonic-residual fine-structure channel | 24.7% | rejected |
| 2 | LFCC+MGD, TDNN, SpecAugment | 18.6% | accepted |
| 3 | + channel-adversarial (gradient reversal), mixup, dropout 0.3 | 18.0% (8 cond), 17.3% (same 4 cond) | accepted |
| 4 | wav2vec2-base, layers 3/6/9/12 mean+std pooled, logistic regression (2,400 train clips) | 12.5% (4 cond) | accepted |
| 5 | + 900 low-SNR (0/5/10 dB) training views | 7.9% (4 cond) | accepted |

Locked test (single scoring of recipe 5, 800-clip stratified-random subset of the test split, 4 conditions): mean EER **14.4%**: clean 8.7%, G.711 9.3%, AMR-NB 12.2k 11.7%, G.711 + 0 dB noise 27.9%.
For reference on the full test split: recipe 3 averaged 29.5% over 14 conditions (25.5 clean, 25.6 G.711, 28.5 AMR, 39.9 at 0 dB); the Phase 2 baseline averaged 32.2%. The recipe 5 subset is not identical to the full split, so the comparison is indicative.

Caveats. Test attacks include hard voice-conversion systems (A17/A18); only 4 of 14 test conditions were scored for recipe 5; trained on 3,300 views; one seed; no confidence intervals yet; pretrained wav2vec2 was trained on 960 h of LibriSpeech, which is unrelated to ASVspoof but is not "from scratch". Nothing here is novel by itself: group delay, gradient reversal, mixup, SSL front ends and noise augmentation are published. The element under test is the validation protocol and the telephone-channel recipe.
