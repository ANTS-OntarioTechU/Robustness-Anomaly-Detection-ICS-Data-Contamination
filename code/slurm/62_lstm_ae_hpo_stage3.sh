#!/bin/bash
# LSTM-AE HPO — Stage 3 (loss × threshold over top-3 from Stage 2, 27 configs).
# Scaler is not swept for LSTM-AE — the contiguous split is MinMax-scaled before
# windowing in src/data.py::create_lstm_ae_splits, and changing it there would
# be a separate ablation.
#
#   python scripts/make_hpo_manifests.py --stage lstm_ae_stage3 \
#       --top-from $SWAT_OUTPUT_DIR/hpo/results/lstm_ae_stage2.csv

#SBATCH --job-name=swat_lstm_hpo_s3
#SBATCH --account=def-liyang
#SBATCH --array=0-26%12
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/lstm_hpo_s3_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

python -m src.hpo.runner_lstm_ae \
    --manifest "$SWAT_OUTPUT_DIR/hpo/manifests/lstm_ae_stage3.jsonl" \
    --task-id  "$SLURM_ARRAY_TASK_ID" \
    --stage    lstm_ae_stage3
