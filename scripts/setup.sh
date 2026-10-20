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

if [ ! -x "$SCR/venv/bin/python" ]; then
  python3 -m venv "$SCR/venv"
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
