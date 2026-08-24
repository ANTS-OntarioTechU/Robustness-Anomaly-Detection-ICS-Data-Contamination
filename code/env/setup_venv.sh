#!/bin/bash
# One-shot venv creation on Narval. Run from a LOGIN NODE (not a compute node):
#
#   cd ~/projects/def-liyang/$USER/narval_swat_run
#   bash env/setup_venv.sh
#
# Creates $SWAT_PROJECT_DIR/venv with all dependencies from the Alliance
# wheelhouse. Idempotent — safe to re-run after a module update.

set -euo pipefail

: "${SWAT_PROJECT_DIR:=$(pwd)}"
: "${SWAT_VENV:=$SWAT_PROJECT_DIR/venv}"

echo "Project dir : $SWAT_PROJECT_DIR"
echo "Venv target : $SWAT_VENV"

module --force purge
module load StdEnv/2023
module load python/3.11
module load scipy-stack
module load cuda/12.2

if [[ -d "$SWAT_VENV" ]]; then
    echo "Venv already exists at $SWAT_VENV — activating and updating."
else
    echo "Creating venv ..."
    python -m venv --system-site-packages "$SWAT_VENV"
fi

# shellcheck disable=SC1091
source "$SWAT_VENV/bin/activate"

pip install --no-index --upgrade pip

echo "Installing pinned requirements from Alliance wheelhouse ..."
pip install --no-index -r "$SWAT_PROJECT_DIR/env/requirements.txt" || {
    echo ""
    echo "WARNING: --no-index install failed for at least one package."
    echo "Falling back to online install (needed for pyod on some software stacks)."
    pip install -r "$SWAT_PROJECT_DIR/env/requirements.txt"
}

# Quick sanity check — fail loudly if anything is missing
python - <<'PY'
import sys
print("Python :", sys.version.split()[0])
import numpy, pandas, sklearn, scipy, torch, matplotlib
import pyod
print("numpy  :", numpy.__version__)
print("pandas :", pandas.__version__)
print("sklearn:", sklearn.__version__)
print("torch  :", torch.__version__, "CUDA?", torch.cuda.is_available())
print("pyod   :", pyod.__version__)
PY

echo ""
echo "Venv ready. Activate later with:"
echo "  source $SWAT_VENV/bin/activate"
