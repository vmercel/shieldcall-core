# Phase 7 protocol: ASVspoof 2021 LA (written before any 2021 score was computed)

Why: the ASVspoof organizers (X. Wang, NII) advised that, to measure telephony-channel degradation and compare with the literature, we should
evaluate on ASVspoof 2021 LA with the official keys and report EER and min t-DCF using the organizers' evaluation package.

What is fixed (no tuning on 2021 data, ever):
- Model: `model.npz` (StandardScaler + logistic regression C=0.01 on wav2vec2-base layers 3/6/9/12 mean+std, first 2.5 s).
  Trained only on the ASVspoof 2019 LA subset of this repo (3,300 views, incl. noise views). It has never seen 2021 audio.
- Data: ASVspoof 2021 LA evaluation set (ODC-By), official 'eval' subset, 148,176 trials, with the official keys.
- Metrics: pooled and per-codec EER; min t-DCF with the organizers' pre-computed C012 coefficients; bootstrap 95% interval for pooled EER.
- Comparators: the organizers' four baseline score files (LFCC-GMM, CQCC-GMM, LFCC-LCNN, RawNet2) scored by the same package on the same trials.
- Sample mode (if used instead of full): seeded stratified sample (seed 20261109), equal trials per codec x key cell.
  Sample results are NOT directly comparable to published full-set numbers, because bona fide trials are over-represented.

What is reported regardless of outcome: all numbers, including per-attack and per-codec results, and the model hash.
What this does not show: real-call performance. 2021 LA source speech is anechoic VCTK.
