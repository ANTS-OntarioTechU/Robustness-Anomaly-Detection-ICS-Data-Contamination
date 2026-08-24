#!/bin/bash
#SBATCH --job-name=swat_atk_targeted
#SBATCH --account=def-liyang
#SBATCH --array=0-35%12
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/atk_targeted_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

python -m src.run_attack --task-id "$SLURM_ARRAY_TASK_ID" --attack targeted_flip
