# Phase 11: training on real modern spoofs (ASVspoof 5), tested on a different corpus

**Not comparable to the ASVspoof 2021 organiser baselines**: the training data differs (the 2021 baselines train on 2019 data only). This phase asks a product question: does the Phase 8 feature stack improve when it sees real, recent attacks and telephone-line views in training, and does the gain carry to a corpus it never saw?

Fixed before any full-scale score:
- Features: Phase 8 (wav2vec2-base truncated to 4 layers, mean+std per layer, 3 s cap). Views per clip: native, G.711 mu-law, simulated phone line (`sim_pstn`). Head: logistic regression.
- Data: ASVspoof 5 eval-split mirror (ODC-By). Dev-train = attacks A17-A24; validation = unseen attacks A25-A32 with different speakers (speaker parity split). C is chosen on validation only.
- Final model: refit on all attacks A17-A32, then scored once on In-the-Wild (Apache-2.0; Mueller et al., Interspeech 2022) which is never used for any choice. Phase 8's head is scored on the identical clips and views; the paired bootstrap CI of the EER difference is reported.
- Rule: PASS only if the final model beats Phase 8 on In-the-Wild native AND G.711 with the paired 95% CI entirely below zero. If it does not pass, the code is removed from the repository.
- Screening (sandbox, 1,600 train clips): unseen ASVspoof 5 attacks EER 23.0/23.5/31.2 (Phase 8) vs 10.6/12.6/29.1; In-the-Wild sample (600 clips) 26.0/30.3/31.5 vs 20.3/18.3/26.3.

Caveats: 3 s crops, a sample of In-the-Wild, English speech, no real phone calls, a single seed.
