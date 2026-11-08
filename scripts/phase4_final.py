"""Score the locked test subset ONCE for the accepted Phase 4 recipe (denoise + truncated w2v + noise views + LR)."""
import _repo  # noqa
import json
from phase4_fit import *
tr, nz, te = L("train", 1), L("noise", 1), L("test", 1)
Xa, ya, ca = arr(tr, nz); Xt, yt, ct = arr(te)
s = lr(Xa, ya, Xt); m = report("LOCKED TEST subset", s, yt, ct)
# bootstrap CI of the mean EER over clips (resampling within condition)
rng = np.random.RandomState(0); bs = []
for _ in range(200):
    e = []
    for c in sorted(set(ct)):
        i = np.where(ct == c)[0]; j = rng.choice(i, len(i)); e.append(roc_eer(yt[j], s[j]))
    bs.append(np.mean(e))
r = {c: roc_eer(yt[ct == c], s[ct == c]) for c in sorted(set(ct))}
json.dump({"recipe": "phase4: truncated wav2vec2 (4 layers) + spectral gate + 3000 noise views + LR(C=0.01)", "n_clips": int(len(yt)), "per_condition_eer": r, "mean_eer": m, "bootstrap95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}, open("docs/results/phase4_test.json", "w"), indent=1)
print("CI", np.percentile(bs, [2.5, 97.5]))
