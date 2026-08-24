#!/bin/bash
# AE HPO — Stage 2 (training sweep over top-3 archs, 144 configs).
# Manifest depends on Stage 1 results — regenerate before submitting:
#
#   python scripts/make_hpo_manifests.py --stage ae_stage2 \
#       --top-from $SWAT_OUTPUT_DIR/hpo/results/ae_stage1.csv

#SBATCH --job-name=swat_ae_hpo_s2
#SBATCH --account=def-liyang
#SBATCH --array=0-143%12
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=logs/ae_hpo_s2_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

python -m src.hpo.runner_ae \
    --manifest "$SWAT_OUTPUT_DIR/hpo/manifests/ae_stage2.jsonl" \
    --task-id  "$SLURM_ARRAY_TASK_ID" \
    --stage    ae_stage2
