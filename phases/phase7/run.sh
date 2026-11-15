#!/usr/bin/env bash
# Phase 7 runner. Usage: bash phase7/run.sh smoke|sample|full
set -euo pipefail
MODE="${1:-smoke}"
cd "$(dirname "$0")"
WORK="$PWD/work"; mkdir -p "$WORK"
echo "== 1/5 python environment"
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
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
$CAF python3 phase7_run.py --mode "$MODE" --tar "$TAR" --keys "$WORK/keys" --eval-package "$WORK/2021/eval-package" --out "$PWD/out"
echo "== 5/5 report bundle"
python3 - "$MODE" <<'P'
import sys, os, gzip, shutil, zipfile, json
m = sys.argv[1]; o = "out"
sc = os.path.join(o, f"scores_{m}.txt")
with open(sc, "rb") as f, gzip.open(sc + ".gz", "wb") as g: shutil.copyfileobj(f, g)
z = f"phase7_report_{m}.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as Z:
    for n in sorted(os.listdir(o)):
        if n.endswith((".json", ".gz")) or n.startswith("official_tables"): Z.write(os.path.join(o, n), n)
    Z.write("work/eval_package_commit.txt", "eval_package_commit.txt")
print("Upload this file to the chat:", os.path.abspath(z))
P
