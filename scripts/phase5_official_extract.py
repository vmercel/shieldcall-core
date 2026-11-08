"""Standard-protocol check: random stratified-by-nothing sample of the OFFICIAL ASVspoof 2019 LA eval trial list (cm.eval.trl.txt),
original 16 kHz audio, no channel simulation. Features = Phase 3b recipe (wav2vec2-base layers 3/6/9/12, mean+std, first 2.5 s).
usage: phase5_official_extract.py N"""
import _repo  # noqa
import sys, io, pickle, time, random, zipfile
import numpy as np, torch, soundfile as sf
torch.backends.nnpack.set_flags(False)
from transformers import Wav2Vec2Model
N = int(sys.argv[1]); LAYERS = (3, 6, 9, 12); MAXS = 40000
z = zipfile.ZipFile("/home/user/scratch/dl/LA.zip")
rows = [l.split() for l in z.read("LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt").decode().splitlines()]
rng = random.Random(20261108); rng.shuffle(rows); rows = rows[:N]   # row: spk utt - attack key
m = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval(); out = {}; t0 = time.time()
for i, r in enumerate(rows):
    utt, att, lab = r[1], r[3], r[4]
    x, sr = sf.read(io.BytesIO(z.read(f"LA/ASVspoof2019_LA_eval/flac/{utt}.flac")), dtype="float32"); x = x[:MAXS]
    x = (x - x.mean()) / (x.std() + 1e-6)
    with torch.no_grad(): hs = m(torch.from_numpy(x)[None], output_hidden_states=True).hidden_states
    f = np.concatenate([np.r_[hs[l][0].mean(0).numpy(), hs[l][0].std(0).numpy()] for l in LAYERS])
    out[utt] = (f.astype(np.float16), int(lab == "spoof"), att, "official_eval")
    if i % 250 == 0:
        print(i, N, round(time.time() - t0), flush=True)
        pickle.dump(out, open("data/work/w2v_official.pkl", "wb"))
pickle.dump(out, open("data/work/w2v_official.pkl", "wb")); print("done", len(out), round(time.time() - t0))
