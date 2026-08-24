#!/bin/bash
#SBATCH --job-name=swat_atk_random
#SBATCH --account=def-liyang
#SBATCH --array=0-35%12
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/atk_random_%A_%a.out

# 36 array tasks × 4 poison rates = 144 combos for random_flip.
# Each task holds the (model, seed) fixed and loops rates internally so
# preprocessing + splits are paid once per task, not once per combo.

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

python -m src.run_attack --task-id "$SLURM_ARRAY_TASK_ID" --attack random_flip
