# Phase 5: comparison on the official ASVspoof 2019 LA evaluation trial list

Purpose: put one number on the same footing as published ASVspoof 2019 LA results (clean 16 kHz audio, official eval attacks A07-A19, EER pooled over all trials), because the custom channel-condition benchmark cannot be compared with the literature.

Method: random sample of 6,000 trials from `ASVspoof2019.LA.cm.eval.trl.txt` (seed 20261108; 5,393 spoof, 607 bona fide), original audio, no channel simulation. Recipe fixed in advance (Phase 3b: wav2vec2-base layers 3/6/9/12 mean+std over the first 2.5 s, logistic regression C=0.01, trained on 2,400 train views + 900 noise views; A01/A05 not used for selection). No tuning on this sample.

Result: pooled EER **10.05%** (bootstrap 95% CI 8.8-11.6). Per attack vs bona fide: A10 30.2%, A11 17.0%, A07 9.4%, A08 9.7%, A12 7.8%, A15 7.6%, A09 6.6%, A16 6.6%, A18 6.1%, A19 5.8%, A14 4.6%, A17 3.9%, A13 2.6% (about 400 spoofs each). See `phase5_official_eval_sample.json`.

Published reference (full eval set, ASVspoof 2019 organisers' baselines): LFCC-GMM EER 8.09% (min t-DCF 0.2116), CQCC-GMM EER 9.57% (min t-DCF 0.2366) (e.g. Alluri et al., Interspeech 2019, Table 1).

Reading: our recipe is at roughly baseline level on the standard protocol, not better. It uses a 2.5 s crop, 3,300 training views and a sample of the eval set, so the figures are not strictly comparable. min t-DCF was not computed (needs the ASV scores and the full trial list). The hard attacks are A10 and A11, as in the literature. State this as a measured, reproducible baseline, not a state-of-the-art claim.
