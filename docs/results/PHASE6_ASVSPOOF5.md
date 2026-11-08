# Phase 6: ASVspoof 5 stratified sample (cross-corpus, built-in codec conditions)

Date: 8 November 2026. Fixed recipe, no tuning on this data. Reproduce with `scripts/phase6_asv5_extract.py 24 130` then `scripts/phase6_asv5_fit.py`.

## What was run
- **Data:** ASVspoof 5 Track 1 evaluation set (crowdsourced MLS English speech, 16 kHz FLAC; ODC-By, bona fide audio CC BY 4.0), read from the Hugging Face mirror `SpeechAntiSpoofingBenchmarks/ASVspoof5` with original audio and unmodified labels.
- **Sample (fixed before scoring):** 24 of the 200 shards, evenly spaced; from each shard the first 130 bona fide and first 130 spoof clips in file order. 6,240 clips (3,120 bona fide, 3,120 spoof), 12 codec conditions (C00 = none, C01 to C11), 16 spoofing attacks (A17 to A32). The sample is balanced by class, unlike the full set, which is mostly spoof. EER is a rate measure, so balance does not bias it, but the sample is not the official trial list.
- **Model:** the Phase 3b recipe, unchanged: wav2vec2-base layers 3/6/9/12 (mean and std), first 2.5 s, logistic regression C = 0.01, trained only on the ASVspoof 2019 LA training subset plus its noise views. This is a cross-corpus test: the model never saw ASVspoof 5 audio.

## Results (equal error rate, lower is better)
| Condition | EER | Bona fide clips |
|---|---|---|
| **Pooled** | **22.63%** (95% bootstrap 21.6 to 23.8) | 3,120 |
| No codec (C00) | 16.9% | 787 |
| Codec conditions C01 to C11 | 15.0% (C06) to 28.6% (C10); C03 26.5, C07 27.3, C08 27.3 | 180 to 231 each |

Per attack, against all bona fide clips (EER): A20 3.1, A32 3.3, A27 5.5, A19 8.1, A30 8.8, A25 9.4, A18 10.2, A31 11.7, A23 14.0, A26 18.1, A22 27.3, A21 28.1, A17 31.4, A24 35.6, A29 39.6, **A28 49.5 (chance)**. Raw numbers: `phase6_asv5_sample.json`.

## How to read this
- The same recipe scored 10.05% on the official ASVspoof 2019 LA sample (Phase 5). On ASVspoof 5 it is **22.63%**, with 16.9% even without a codec. Realistic source audio, many unseen attacks and codecs roughly double the error.
- The organizers report that their closed-condition baselines (RawNet2, AASIST, trained on ASVspoof 5 data) reach EERs no lower than 29% on the evaluation set (Wang et al., ASVspoof 5 workshop paper, 2024). Our 22.63% is lower, **but the numbers are not comparable**: different sample, balanced classes, different training data (ours is cross-corpus and uses a pretrained model, which the closed condition forbids), and the primary metric there is minDCF, which we have not computed. Top challenge systems reach EERs below 15% (closed) and about 3% (open). We make no claim of competitiveness.
- Narrowband codecs (C08 to C11) and some codecs (C03, C07, C10) are the hardest, which is the telephone-relevant finding. One attack (A28) is not detected at all.
- Limits: one seed, one fixed sample, mean-pooled 2.5 s features, a linear classifier. Not real phone calls. Not a validation of real-world performance.

## Next
Harder validation, a fine-tuned or longer-context front end trained on ASVspoof 5 training data (open condition), and minDCF.
