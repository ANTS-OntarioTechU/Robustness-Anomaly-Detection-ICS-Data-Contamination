#!/bin/bash
# Pre-flight check. Run on a LOGIN NODE first, then inside a GPU `salloc`
# to confirm CUDA visibility. Exits non-zero on any failure so it's safe
# to chain in front of sbatch submissions.

set -euo pipefail

: "${SWAT_PROJECT_DIR:=$(pwd)}"
: "${SWAT_DATA_DIR:=$SCRATCH/swat_data}"
: "${SWAT_OUTPUT_DIR:=$SCRATCH/swat_paper_run}"
: "${SWAT_VENV:=$SWAT_PROJECT_DIR/venv}"

echo "SWAT_PROJECT_DIR : $SWAT_PROJECT_DIR"
echo "SWAT_DATA_DIR    : $SWAT_DATA_DIR"
echo "SWAT_OUTPUT_DIR  : $SWAT_OUTPUT_DIR"
echo "SWAT_VENV        : $SWAT_VENV"
echo

# 1) Venv
[[ -d "$SWAT_VENV" ]] || { echo "FAIL: venv missing at $SWAT_VENV"; exit 1; }
echo "OK  venv present"

# 2) Data files
for f in normal.csv attack.csv; do
    p="$SWAT_DATA_DIR/$f"
    if [[ ! -s "$p" ]]; then
        echo "FAIL: $p missing or empty — upload the SWaT CSVs first"; exit 1
    fi
    sz=$(stat -c%s "$p")
    echo "OK  $f ($sz bytes)"
done

# 3) Output dir writable
mkdir -p "$SWAT_OUTPUT_DIR/checkpoints/clean"
for atk in random_flip targeted_flip feature_noise; do
    mkdir -p "$SWAT_OUTPUT_DIR/checkpoints/attacks/$atk"
done
mkdir -p "$SWAT_OUTPUT_DIR/figures" "$SWAT_OUTPUT_DIR/logs"
touch "$SWAT_OUTPUT_DIR/.write_test" && rm "$SWAT_OUTPUT_DIR/.write_test"
echo "OK  output dir writable"

# 4) Python + imports
module --force purge
module load StdEnv/2023
module load python/3.11
module load scipy-stack
module load cuda/12.2
# shellcheck disable=SC1091
source "$SWAT_VENV/bin/activate"

python - <<'PY'
import importlib, sys
mods = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch", "pyod"]
for m in mods:
    importlib.import_module(m)
    print(f"OK  import {m}")
import torch
print(f"OK  torch.cuda.is_available() = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    device = {torch.cuda.get_device_name(0)}")
PY

# 5) Task-id decoder — verify grid config
python - <<'PY'
from src.config import CONFIG, clean_task_shape, attack_task_shape, decode_clean_task
N = clean_task_shape()
assert attack_task_shape() == N
for i in range(N):
    m, s = decode_clean_task(i)
    assert m in CONFIG["MODELS"]
    assert s in CONFIG["SEEDS"]
print(f"OK  grid decoder: {N} tasks/array, "
      f"{len(CONFIG['POISON_RATES'])} rates/task for attack runs")
print(f"OK  total planned combos: {N} clean + "
      f"{N * len(CONFIG['ATTACKS']) * len(CONFIG['POISON_RATES'])} attack")
PY

echo
echo "All checks passed."
