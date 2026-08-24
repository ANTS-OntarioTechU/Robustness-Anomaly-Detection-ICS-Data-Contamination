#!/bin/bash
# AE HPO — Stage 3 (loss × threshold × scaler over top-3 from Stage 2, 81 configs).
#
#   python scripts/make_hpo_manifests.py --stage ae_stage3 \
#       --top-from $SWAT_OUTPUT_DIR/hpo/results/ae_stage2.csv

#SBATCH --job-name=swat_ae_hpo_s3
#SBATCH --account=def-liyang
#SBATCH --array=0-80%12
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=logs/ae_hpo_s3_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

python -m src.hpo.runner_ae \
    --manifest "$SWAT_OUTPUT_DIR/hpo/manifests/ae_stage3.jsonl" \
    --task-id  "$SLURM_ARRAY_TASK_ID" \
    --stage    ae_stage3
