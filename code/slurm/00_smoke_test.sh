#!/bin/bash
#SBATCH --job-name=swat_smoke
#SBATCH --account=def-liyang
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:45:00
#SBATCH --output=logs/smoke_%j.out

# Sanity check: run ONE clean baseline (task-id 0 = iforest, seed=42) and
# ONE attack combo (task-id 0, attack=random_flip) so we confirm the whole
# pipeline lights up before burning A100-hours on the full grid.

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

echo "=== Clean-baseline smoke (task 0) ==="
python -m src.run_clean  --task-id 0

echo "=== Attack smoke (task 0, random_flip) ==="
python -m src.run_attack --task-id 0 --attack random_flip

echo "=== Aggregate smoke (builds tables from whatever is done) ==="
python -m src.aggregate

echo "Smoke test finished at $(date -Iseconds)"
