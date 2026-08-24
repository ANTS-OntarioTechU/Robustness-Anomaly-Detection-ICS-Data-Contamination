#!/bin/bash
# LSTM-AE HPO — Stage 2 (training sweep over top-3 archs, 108 configs).
#
#   python scripts/make_hpo_manifests.py --stage lstm_ae_stage2 \
#       --top-from $SWAT_OUTPUT_DIR/hpo/results/lstm_ae_stage1.csv

#SBATCH --job-name=swat_lstm_hpo_s2
#SBATCH --account=def-liyang
#SBATCH --array=0-107%12
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/lstm_hpo_s2_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

python -m src.hpo.runner_lstm_ae \
    --manifest "$SWAT_OUTPUT_DIR/hpo/manifests/lstm_ae_stage2.jsonl" \
    --task-id  "$SLURM_ARRAY_TASK_ID" \
    --stage    lstm_ae_stage2
