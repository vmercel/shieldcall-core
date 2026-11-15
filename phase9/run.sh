#!/usr/bin/env bash
# Phase 9 runner. Usage: bash phase9/run.sh smoke|full      (resumable: if anything stops, run the same command again)
# Optional: REF="phase7=/path/scores_full.txt,phase8=/path/scores_full.txt" for paired comparisons (defaults to the Phase 7/8 outputs if present).
set -euo pipefail
MODE="${1:-smoke}"
cd "$(dirname "$0")"
WORK="$PWD/../phase7/work"; mkdir -p "$WORK"
if [ "$MODE" = "smoke" ]; then SCALE=tiny; EP=1; SEL=1.0; SEEDS=1; else SCALE=full; EP=8; SEL=0.5; SEEDS=3; fi
W9="$PWD/work9_$SCALE"; O9="$PWD/out9_$SCALE"; mkdir -p "$W9" "$O9"
echo "== 1/6 python environment"
PYOK=""
for c in python3.11 python3.12 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)' 2>/dev/null; then PYOK="$c"; break; fi
done
VENV=.venv9
if [ -x ../phase8/.venv8/bin/python ] && ../phase8/.venv8/bin/python -c 'import torch,transformers,librosa,av,sklearn,pyarrow' 2>/dev/null; then VENV=../phase8/.venv8; fi   # reuse the working Phase 8 environment
if [ ! -x $VENV/bin/python ] || ! $VENV/bin/python -c 'import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)' 2>/dev/null; then
  rm -rf .venv9; VENV=.venv9
  if [ -n "$PYOK" ]; then "$PYOK" -m venv .venv9
  else TOOLS="$WORK/uvtool"; python3 -m venv "$TOOLS"; "$TOOLS/bin/pip" install -q uv; "$TOOLS/bin/uv" venv --python 3.11 .venv9; fi
fi
source $VENV/bin/activate
if [ "$VENV" = ".venv9" ]; then
  if python -m pip --version >/dev/null 2>&1; then python -m pip install -q --upgrade pip 2>/dev/null || true; python -m pip install -q -r requirements.txt
  else "$WORK/uvtool/bin/uv" pip install -q -r requirements.txt; fi
fi
python -c "import torch, transformers, librosa, av, sklearn, scipy, soundfile, pyarrow; print('environment OK: torch', torch.__version__, 'python', __import__('sys').version.split()[0])"
FREE=$(df -Pk . | awk 'NR==2{print int($4/1048576)}'); echo "free disk: ${FREE} GB"
if [ "$MODE" = "full" ] && [ "$FREE" -lt 30 ]; then echo "Need about 30 GB free (10 GB features + 8 GB evaluation audio + models). Free some space and re-run."; exit 1; fi
CAF=""; command -v caffeinate >/dev/null && CAF="caffeinate -i"
echo "== 2/6 frame-level features, ASVspoof 2019 LA train (resumable)"
$CAF python3 phase9_extract.py --split train --scale $SCALE --out "$W9"
echo "== 3/6 frame-level features, ASVspoof 2019 LA dev (resumable)"
$CAF python3 phase9_extract.py --split dev --scale $SCALE --out "$W9"
echo "== 4/6 head selection on 2019 dev only, then final fit on 2019 train+dev"
if [ ! -f "$O9/dev_selection_results.json" ] || ! grep -q winner "$O9/dev_selection_results.json"; then $CAF python3 phase9_train.py select --work "$W9" --out "$O9" --epochs $EP --sel-frac $SEL; fi
[ -f "$O9/model9.pt" ] || $CAF python3 phase9_train.py final --work "$W9" --out "$O9" --epochs $EP --seeds $SEEDS
echo "model hash (fixed BEFORE any 2021 audio is scored): $(cat $O9/model9.sha256)"
echo "== 5/6 organizers' evaluation package, keys and ASVspoof 2021 LA audio"
[ -d "$WORK/2021" ] || git clone -q --depth 1 https://github.com/asvspoof-challenge/2021 "$WORK/2021"
git -C "$WORK/2021" rev-parse HEAD > "$WORK/eval_package_commit.txt"
if [ ! -f "$WORK/keys/LA/CM/trial_metadata.txt" ]; then
  for i in 1 2 3 4 5 6; do
    curl -L -s --retry 5 -o "$WORK/LA-keys-full.tar.gz" https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz || true
    SUM=$(python3 -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$WORK/LA-keys-full.tar.gz" || echo none)
    [ "$SUM" = "f052cc2ed276745afa3b5198665d3b26" ] && break
    echo "keys download not ready, retry $i"; sleep 10
  done
  [ "$SUM" = "f052cc2ed276745afa3b5198665d3b26" ] || { echo "Could not fetch the keys. Download https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz into $WORK and re-run."; exit 1; }
  tar -xzf "$WORK/LA-keys-full.tar.gz" -C "$WORK"
fi
TAR=""
if [ "$MODE" = "full" ]; then
  TAR="$WORK/ASVspoof2021_LA_eval.tar.gz"
  curl -L -C - --retry 30 --retry-delay 5 -o "$TAR" "https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz?download=1"
  SUM=$(python3 -c "import hashlib,sys;h=hashlib.md5()
f=open(sys.argv[1],'rb')
for b in iter(lambda:f.read(1<<24),b''): h.update(b)
print(h.hexdigest())" "$TAR")
  [ "$SUM" = "2abee34d8b0b91159555fc4f016e4562" ] || { echo "Checksum mismatch ($SUM). Delete $TAR and re-run."; exit 1; }
fi
echo "== 6/6 scoring ASVspoof 2021 LA (resumable)"
if [ -z "${REF:-}" ]; then
  REF=""; [ -f ../phase7/out/scores_full.txt ] && REF="phase7=$PWD/../phase7/out/scores_full.txt"
  [ -f ../phase8/out/scores_full.txt ] && REF="${REF:+$REF,}phase8=$PWD/../phase8/out/scores_full.txt"
fi
$CAF python3 phase9_eval.py --mode "$([ "$MODE" = full ] && echo full || echo smoke)" --tar "$TAR" --keys "$WORK/keys" --eval-package "$WORK/2021/eval-package" --model "$O9/model9.pt" --ref-scores "$REF" --out "$O9/eval"
python3 - "$MODE" "$O9" <<'P'
import sys, os, gzip, shutil, zipfile
m, O = sys.argv[1], sys.argv[2]; e = os.path.join(O, "eval"); sc = os.path.join(e, f"scores_{m if m=='full' else 'smoke'}.txt")
with open(sc, "rb") as f, gzip.open(sc + ".gz", "wb") as g: shutil.copyfileobj(f, g)
z = f"phase9_report_{m}.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as Z:
    for n in sorted(os.listdir(e)):
        if n.endswith((".json", ".gz")) or n.startswith("official_tables"): Z.write(os.path.join(e, n), n)
    for n in ("dev_selection_results.json", "model9.sha256"): Z.write(os.path.join(O, n), n)
    Z.write("../phase7/work/eval_package_commit.txt", "eval_package_commit.txt")
print("Upload this file to the chat:", os.path.abspath(z))
P
