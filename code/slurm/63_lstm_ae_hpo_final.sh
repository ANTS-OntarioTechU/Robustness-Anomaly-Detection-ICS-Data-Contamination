#!/bin/bash
# LSTM-AE HPO — Final confirmation (top-3 × 3 seeds × 13-cell grid = 9 tasks).
# 13 trains × ~210 s ≈ 45 min — give 3 h headroom for the 50-window worst case.
#
#   python scripts/make_hpo_manifests.py --stage lstm_ae_final \
#       --top-from $SWAT_OUTPUT_DIR/hpo/results/lstm_ae_stage3.csv

#SBATCH --job-name=swat_lstm_hpo_final
#SBATCH --account=def-liyang
#SBATCH --array=0-8%9
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/lstm_hpo_final_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

python -m src.hpo.runner_lstm_ae \
    --manifest "$SWAT_OUTPUT_DIR/hpo/manifests/lstm_ae_final.jsonl" \
    --task-id  "$SLURM_ARRAY_TASK_ID" \
    --stage    lstm_ae_final
