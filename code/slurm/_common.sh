#!/bin/bash
# Shared environment bootstrap for every Slurm job in this package.
# Sourced by the sbatch scripts; not meant to be executed directly.

set -euo pipefail

# ── Paths (override by exporting before sbatch) ─────────────────────
: "${SWAT_PROJECT_DIR:=$HOME/projects/def-liyang/$USER/narval_swat_run}"
: "${SWAT_DATA_DIR:=$SCRATCH/swat_data}"
: "${SWAT_OUTPUT_DIR:=$SCRATCH/swat_paper_run}"
export SWAT_PROJECT_DIR SWAT_DATA_DIR SWAT_OUTPUT_DIR

# ── Modules (Narval / StdEnv 2023) ──────────────────────────────────
module --force purge
module load StdEnv/2023
module load python/3.11
module load scipy-stack
module load cuda/12.2

# ── Virtualenv (created once by env/setup_venv.sh) ──────────────────
: "${SWAT_VENV:=$SWAT_PROJECT_DIR/venv}"
if [[ ! -d "$SWAT_VENV" ]]; then
    echo "ERROR: venv not found at $SWAT_VENV. Run env/setup_venv.sh first." >&2
    exit 2
fi
# shellcheck disable=SC1091
source "$SWAT_VENV/bin/activate"

# ── Thread hygiene (avoid BLAS oversubscription) ────────────────────
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Deterministic matmul where possible
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# ── Diagnostic header (goes to the per-task log) ────────────────────
echo "───────────────────────────────────────────────────────────────"
echo "SLURM_JOB_ID         : ${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID   : ${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID  : ${SLURM_ARRAY_TASK_ID:-}"
echo "Host                 : $(hostname)"
echo "Date                 : $(date -Iseconds)"
echo "SWAT_PROJECT_DIR     : $SWAT_PROJECT_DIR"
echo "SWAT_DATA_DIR        : $SWAT_DATA_DIR"
echo "SWAT_OUTPUT_DIR      : $SWAT_OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU visible)"
echo "───────────────────────────────────────────────────────────────"

cd "$SWAT_PROJECT_DIR"
