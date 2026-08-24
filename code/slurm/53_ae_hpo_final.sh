#!/bin/bash
# AE HPO — Final confirmation (top-3 × 3 seeds × 13-cell grid = 9 tasks).
# Each task internally runs the full clean + 3 attacks × 4 rates poisoning grid
# for one (config, seed) pair. 13 trains × ~120 s ≈ 26 min — give 2 h headroom.
#
#   python scripts/make_hpo_manifests.py --stage ae_final \
#       --top-from $SWAT_OUTPUT_DIR/hpo/results/ae_stage3.csv

#SBATCH --job-name=swat_ae_hpo_final
#SBATCH --account=def-liyang
#SBATCH --array=0-8%9
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:30:00
#SBATCH --output=logs/ae_hpo_final_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

python -m src.hpo.runner_ae \
    --manifest "$SWAT_OUTPUT_DIR/hpo/manifests/ae_final.jsonl" \
    --task-id  "$SLURM_ARRAY_TASK_ID" \
    --stage    ae_final
