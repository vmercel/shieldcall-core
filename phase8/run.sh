#!/usr/bin/env bash
# Phase 8 runner. Usage: bash phase8/run.sh smoke|sample|full
set -euo pipefail
MODE="${1:-smoke}"
cd "$(dirname "$0")"
WORK="$PWD/../phase7/work"; mkdir -p "$WORK"   # shares the Phase 7 download (keys, 7.8 GB eval audio) if you already ran it
echo "== 1/5 python environment"
# torch/scipy wheels exist only for Python 3.10-3.12. Find one; if none is installed, fetch a private Python 3.11 with uv (no admin rights, no conda needed).
PYOK=""
for c in python3.11 python3.12 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)' 2>/dev/null; then PYOK="$c"; break; fi
done
if [ ! -x .venv8/bin/python ] || ! .venv8/bin/python -c 'import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)' 2>/dev/null; then
  rm -rf .venv8
  if [ -n "$PYOK" ]; then
    echo "using $($PYOK --version)"; "$PYOK" -m venv .venv8
  else
    echo "no Python 3.10-3.12 found; installing a private Python 3.11 with uv"
    TOOLS="$WORK/uvtool"; python3 -m venv "$TOOLS"; "$TOOLS/bin/pip" install -q uv
    "$TOOLS/bin/uv" venv --python 3.11 .venv8
  fi
fi
source .venv8/bin/activate
python -m pip install -q --upgrade pip 2>/dev/null || true
if python -m pip --version >/dev/null 2>&1; then python -m pip install -q -r requirements.txt
else "$WORK/uvtool/bin/uv" pip install -q -r requirements.txt; fi
python -c "import torch, transformers, librosa, av, sklearn, scipy, soundfile, pyarrow; print('environment OK: torch', torch.__version__, 'python', __import__('sys').version.split()[0])"
echo "== 2/5 organizers' evaluation package + keys"
[ -d "$WORK/2021" ] || git clone -q --depth 1 https://github.com/asvspoof-challenge/2021 "$WORK/2021"
git -C "$WORK/2021" rev-parse HEAD > "$WORK/eval_package_commit.txt"
if [ ! -f "$WORK/keys/LA/CM/trial_metadata.txt" ]; then
  for i in 1 2 3 4 5 6; do
    curl -L -s --retry 5 -o "$WORK/LA-keys-full.tar.gz" https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz || true
    SUM=$(python3 -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$WORK/LA-keys-full.tar.gz" || echo none)
    [ "$SUM" = "f052cc2ed276745afa3b5198665d3b26" ] && break
    echo "keys download not ready (server busy?), retry $i"; sleep 10
  done
  [ "$SUM" = "f052cc2ed276745afa3b5198665d3b26" ] || { echo "Could not fetch the keys. Download https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz by hand into $WORK and re-run."; exit 1; }
  tar -xzf "$WORK/LA-keys-full.tar.gz" -C "$WORK"
fi
TAR=""
if [ "$MODE" != "smoke" ]; then
  echo "== 3/5 evaluation audio (7.8 GB, resumable)"
  TAR="$WORK/ASVspoof2021_LA_eval.tar.gz"
  curl -L -C - --retry 30 --retry-delay 5 -o "$TAR" "https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz?download=1"
  SUM=$(python3 -c "import hashlib,sys;h=hashlib.md5()
f=open(sys.argv[1],'rb')
for b in iter(lambda:f.read(1<<24),b''): h.update(b)
print(h.hexdigest())" "$TAR")
  [ "$SUM" = "2abee34d8b0b91159555fc4f016e4562" ] || { echo "Checksum mismatch ($SUM). Delete $TAR and re-run."; exit 1; }
else
  echo "== 3/5 smoke mode streams a few hundred clips directly, no big download"
fi
echo "== 4/5 scoring (resumable: if it stops, run the same command again)"
CAF=""; command -v caffeinate >/dev/null && CAF="caffeinate -i"
$CAF python3 phase8_eval.py --mode "$MODE" --tar "$TAR" --keys "$WORK/keys" --eval-package "$WORK/2021/eval-package" --model "$PWD/model8.npz" --ref-scores "${REF:-}" --out "$PWD/out"
echo "== 5/5 report bundle"
python3 - "$MODE" <<'P'
import sys, os, gzip, shutil, zipfile, json
m = sys.argv[1]; o = "out"
sc = os.path.join(o, f"scores_{m}.txt")
with open(sc, "rb") as f, gzip.open(sc + ".gz", "wb") as g: shutil.copyfileobj(f, g)
z = f"phase8_report_{m}.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as Z:
    for n in sorted(os.listdir(o)):
        if n.endswith((".json", ".gz")) or n.startswith("official_tables"): Z.write(os.path.join(o, n), n)
    Z.write("../phase7/work/eval_package_commit.txt", "eval_package_commit.txt")
print("Upload this file to the chat:", os.path.abspath(z))
P
