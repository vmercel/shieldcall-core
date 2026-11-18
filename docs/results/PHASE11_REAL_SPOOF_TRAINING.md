# Phase 11: training on real modern spoofs (ASVspoof 5), tested on In-the-Wild

Not comparable to the ASVspoof 2021 organiser baselines (different training data). Protocol: `phases/phase11/PROTOCOL.md`. Run on a 16-CPU Mac, single seed, 3 s crops, English speech.

Training: 6,400 clips (ASVspoof 5 attacks A17-A24, 3 views). Validation: 2,400 clips of unseen attacks A25-A32, different speakers (C=0.003 chosen here). Final: refit on all A17-A32. External: In-the-Wild, 6,000 clips per view, never used for any choice. Phase 8's head scored on identical clips.

| EER % | Phase 8 head | Phase 11 final | paired 95% CI (diff) |
|---|---|---|---|
| In-the-Wild, native | 28.1 | 19.1 | -10.4 to -7.6 |
| In-the-Wild, G.711 | 29.6 | 16.7 | -14.3 to -11.4 |
| In-the-Wild, simulated phone line | 35.9 | 28.6 | -9.1 to -5.9 |
| Unseen ASVspoof 5 attacks (native / G.711 / phone line) | 23.0 / 24.4 / 30.6 | 9.6 / 10.8 / 25.2 | not computed |

Pass rule (native and G.711 CIs entirely below zero): met. Limits: phone-line condition still 25-29%; validation attacks come from the same corpus as training. Published reference for In-the-Wild (Mueller et al., Interspeech 2022): best 2019-trained model 33.1% EER; training data differs, so this is not a like-for-like baseline comparison.
Model sha256: 2d2878122d05ef73a3e266f924b15bddbead84b59f4c639fed2a00a3f3014413
