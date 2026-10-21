#!/usr/bin/env bash
# One-time setup on the project machine, ece028 (run from the repo root):   bash scripts/setup.sh
# Puts the Python env, the model files and the layer dumps on /scratch (fast, no AFS
# token or quota), and symlinks them into the repo. /scratch is NOT backed up and can be
# purged: everything there is regenerable from this script; your code lives in git.
set -euo pipefail
SCR=${SCR:-/scratch/$USER/645}
if ! mkdir -p "$SCR" 2>/dev/null; then
  echo "Cannot create $SCR. Check that this machine has /scratch (df -h /scratch);"
  echo "if not, rerun with a local disk path, e.g.  SCR=/tmp/$USER/645 bash scripts/setup.sh"
  exit 1
fi
mkdir -p "$SCR"/{venv,models,tensors,pipcache}
export PIP_CACHE_DIR="$SCR/pipcache"

# Ultralytics and PyTorch need Python >= 3.9. Pick one with:  PYTHON=/path/to/python3.x bash scripts/setup.sh
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then PY=$(command -v "$c"); break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "No Python >= 3.9 found on PATH (python3 here is $(python3 --version 2>&1))."
  echo "Find one with:  ls /usr/bin/python3* /usr/local/bin/python3* 2>/dev/null; module avail python 2>&1 | head"
  echo "then rerun:     PYTHON=/full/path/to/python3.X bash scripts/setup.sh"
  exit 1
fi
echo "Using $PY ($("$PY" --version))"
if [ -x "$SCR/venv/bin/python" ] && ! "$SCR/venv/bin/python" -c 'import sys; sys.exit(sys.version_info < (3, 9))'; then
  echo "Removing old venv built with $("$SCR/venv/bin/python" --version)"; rm -rf "$SCR/venv"
fi
if [ ! -x "$SCR/venv/bin/python" ]; then
  "$PY" -m venv "$SCR/venv"
fi
source "$SCR/venv/bin/activate"
pip install -q -U pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch, much smaller
pip install -q -r requirements.txt

# big, regenerable files live on scratch; the repo sees them through symlinks (gitignored)
for d in models tensors; do
  [ -e "data/$d" ] || ln -s "$SCR/$d" "data/$d"
done
mkdir -p results/raw

python scripts/export_model.py --batch 1
python scripts/dump_layers.py --batch 1
bash scripts/machine_info.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DYOLO_ENABLE_BROADWELL=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench_layer data/tensors/b1/model.4.m.0.cv1 s1_v1 -r 10
echo
echo "Done. Each new shell:  source $SCR/venv/bin/activate"
