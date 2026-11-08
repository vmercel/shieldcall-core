# Phase 4 iterations (cheap-compute strategy, 1 CPU)

Protocol unchanged: validation = dev speakers, attacks A01/A05 held out, 4 conditions (clean, G.711, AMR-NB 12.2k, G.711+0 dB noise); accept only if mean EER improves by more than 0.3 points over the last accepted recipe (7.91%, Phase 3b). Locked test (A07-A19, 67 unseen speakers) scored once for the accepted recipe.

| # | Change | Val mean EER | Decision |
|---|---|---|---|
| ref | wav2vec2-base, layers 3/6/9/12, mean+std, LR, +900 noise views (Phase 3b) | 7.91% | accepted |
| 4.0 | ablations: single layers 3/6/9/12; means vs stds | 8.44 / 11.4 / 9.8 / 13.9; 8.79 / 10.3 | informative; early layers dominate |
| 4.1 | Truncated encoder (CNN + first 4 layers, ~4x faster), no extra noise views | 11.05% | rejected |
| 4.2 | + 3000 noise views, SNR 0-20 dB | 7.62% (0.29 better) | rejected (margin 0.3) |
| 4.3 | + channel-conditional compensation (soft condition posterior from features, no oracle labels) | 7.53% | rejected vs 4.2 |
| 4.4 | MLP head (PCA 256) / SNR-gated mixture of experts | 7.78% / 7.64% | rejected |
| 4.5 | 4.2 + spectral-gate denoise (min-statistics + Wiener gain) on all train/val/test audio | 7.45% (0.46 better than ref) | accepted |

## Locked test, scored once for 4.5 (800 clips x 4 conditions, same subset as Phase 3b)
Mean EER 17.2% (bootstrap 95% CI 15.8-18.3): clean 12.6, G.711 13.2, AMR-NB 13.9, G.711+0 dB 29.1.
Phase 3b recipe on the same subset: 14.4% mean (clean 8.7, G.711 9.3, AMR 11.7, 0 dB 27.9).

## Reading
- The validation gain from 4.5 did NOT transfer: the cheaper recipe is worse on the locked test (17.2% vs 14.4%). Phase 3b remains the best measured recipe; 4.5's advantage is about 4x lower compute only.
- Likely cause: attacks A01/A05 are easy (layer 3 alone gets 0.1% clean EER on validation), so validation rewards early-layer features and under-weights the harder test attacks. The acceptance gate is only as good as the validation split; it should hold out harder attacks (e.g. A17/A18-like) before further iterations. The test split must not be used for that selection.
- 0 dB noise stays at 25-29% for every variant.
- Every component here is published prior art (SSL front ends, layer truncation, noise augmentation, spectral gating, mixture of experts, feature compensation). No novelty claim.
- Single seed; subset; not real-world validation.
