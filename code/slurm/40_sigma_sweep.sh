#!/bin/bash
#SBATCH --job-name=swat_sigma
#SBATCH --account=def-liyang
#SBATCH --array=0-32%12
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=logs/sigma_%A_%a.out

# Feature-noise σ sweep — 33 array tasks (11 models × 3 seeds, SOD excluded).
# One submission runs a single σ value at the fixed poison rate of 0.10.
#
# The σ to use is passed via the SWAT_SIGMA environment variable, e.g.:
#
#     SWAT_SIGMA=0.30 sbatch slurm/40_sigma_sweep.sh
#
# See scripts/submit_sigma_sweep.sh for a one-shot submission of multiple σ.

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

: "${SWAT_SIGMA:?SWAT_SIGMA env var must be set, e.g. SWAT_SIGMA=0.30 sbatch ...}"
: "${SWAT_SIGMA_RATE:=0.10}"

echo "Running σ sweep with SWAT_SIGMA=${SWAT_SIGMA} rate=${SWAT_SIGMA_RATE}"
python -m src.run_sigma_sweep \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --sigma   "$SWAT_SIGMA" \
    --rate    "$SWAT_SIGMA_RATE"
