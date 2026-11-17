#!/usr/bin/env bash
# Phase 11 runner. Usage: bash phases/phase11/run.sh smoke|full      (resumable: re-run the same command if interrupted)
set -euo pipefail
MODE="${1:-smoke}"
cd "$(dirname "$0")"
echo "== 1/3 python environment"
PYOK=""
for c in python3.11 python3.12 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)' 2>/dev/null; then PYOK="$c"; break; fi
done
V=../phase8/.venv8      # reuse the Phase 8 environment if you already built it
if [ ! -x $V/bin/python ] || ! $V/bin/python -c 'import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)' 2>/dev/null; then
  rm -rf $V
  if [ -n "$PYOK" ]; then echo "using $($PYOK --version)"; "$PYOK" -m venv $V
  else
    mkdir -p work; TOOLS="$PWD/work/uvtool"; python3 -m venv "$TOOLS"; "$TOOLS/bin/pip" install -q uv; "$TOOLS/bin/uv" venv --python 3.11 $V
  fi
fi
source $V/bin/activate
python -m pip install -q --upgrade pip 2>/dev/null || true
if python -m pip --version >/dev/null 2>&1; then python -m pip install -q -r requirements.txt; else work/uvtool/bin/uv pip install -q -r requirements.txt; fi
python -c "import torch, transformers, librosa, av, sklearn, scipy, soundfile, pyarrow, huggingface_hub; print('environment OK: torch', torch.__version__)"
echo "== 2/3 data + features + training + external test ($MODE)"
CAF=""; command -v caffeinate >/dev/null && CAF="caffeinate -i"
$CAF python3 phase11.py --mode "$MODE"
echo "== 3/3 report bundle"
python3 - "$MODE" <<'P'
import sys, os, zipfile
m = sys.argv[1]; z = f"phase11_report_{m}.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as Z:
    Z.write(f"out/report_{m}.json", f"report_{m}.json"); Z.write("out/model11.npz", "model11.npz")
print("Upload this file to the chat:", os.path.abspath(z))
P
