"""Phase 6: ASVspoof 5 (Track 1 eval, ODC-By; bona fide CC BY 4.0) stratified sample, with its built-in codec conditions.
Source: Hugging Face mirror SpeechAntiSpoofingBenchmarks/ASVspoof5 (original FLAC bytes, unmodified labels).
Design (fixed before looking at any score): SHARDS evenly spaced parquet shards of the 200; from each shard keep the first PER_CLASS
bona fide and first PER_CLASS spoof clips in file order. Features = Phase 3b recipe (wav2vec2-base layers 3/6/9/12, mean+std, first 2.5 s).
usage: phase6_asv5_extract.py SHARDS PER_CLASS"""
import _repo  # noqa
import sys, io, json, pickle, time
import numpy as np, torch, soundfile as sf, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
torch.backends.nnpack.set_flags(False)
from transformers import Wav2Vec2Model
SHARDS = int(sys.argv[1]); PER = int(sys.argv[2]); LAYERS = (3, 6, 9, 12); MAXS = 40000
fs = HfFileSystem(); repo = "datasets/SpeechAntiSpoofingBenchmarks/ASVspoof5/data/test-%05d-of-00200.parquet"
ids = [int(round(i * 200 / SHARDS)) % 200 for i in range(SHARDS)]
m = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval(); out = {}; t0 = time.time(); n = 0
for sh in ids:
    pf = pq.ParquetFile(fs.open(repo % sh, "rb")); keep = {0: 0, 1: 0}
    for b in pf.iter_batches(batch_size=32, columns=["path", "audio", "label", "notes"]):
        for r in b.to_pylist():
            lab = int(r["label"])
            if keep[lab] >= PER: continue
            meta = json.loads(r["notes"]); keep[lab] += 1
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32"); x = x[:MAXS]
            x = (x - x.mean()) / (x.std() + 1e-6)
            with torch.no_grad(): hs = m(torch.from_numpy(x)[None], output_hidden_states=True).hidden_states
            f = np.concatenate([np.r_[hs[l][0].mean(0).numpy(), hs[l][0].std(0).numpy()] for l in LAYERS])
            out[meta["utterance_id"]] = (f.astype(np.float16), lab, meta["attack_id"], meta["codec"]); n += 1
            if n % 100 == 0:
                print(n, sh, round(time.time() - t0), flush=True); pickle.dump(out, open("data/work/w2v_asv5.pkl", "wb"))
        if keep[0] >= PER and keep[1] >= PER: break
    print("shard", sh, keep, flush=True)
pickle.dump(out, open("data/work/w2v_asv5.pkl", "wb")); print("done", len(out), round(time.time() - t0), flush=True)
