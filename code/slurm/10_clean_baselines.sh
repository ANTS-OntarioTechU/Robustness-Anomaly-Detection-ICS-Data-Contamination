#!/bin/bash
#SBATCH --job-name=swat_clean
#SBATCH --account=def-liyang
#SBATCH --array=0-35%12
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/clean_%A_%a.out

# 36 array tasks = 12 models × 3 seeds. One clean baseline per task.
# %12 concurrency cap avoids swamping the scheduler; adjust to taste.
#
# GPU: a full A100 so autoencoder and lstm_ae tasks can train comfortably.
# PyOD tasks won't use the GPU but run fast regardless. If you care about
# GPU-hour budget, split this into CPU + GPU variants later.
#
# The in-script resume logic means resubmitting this whole array is safe:
# completed combos skip instantly.

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

python -m src.run_clean --task-id "$SLURM_ARRAY_TASK_ID"
