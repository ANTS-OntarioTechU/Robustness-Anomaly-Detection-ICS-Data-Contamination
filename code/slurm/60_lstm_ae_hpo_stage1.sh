#!/bin/bash
# LSTM-AE HPO — Stage 1 (architecture sweep, 72 configs).
# Window × hidden_dim × num_layers × dropout. Larger window = longer per-task
# wall time; window=50 with hidden_dim=256 layers=2 is the worst case (~6-8 min
# per train, ~14-16 min per task with clean+poisoned). 1.5 h is safe.
#
# Manifest: $SWAT_OUTPUT_DIR/hpo/manifests/lstm_ae_stage1.jsonl

#SBATCH --job-name=swat_lstm_hpo_s1
#SBATCH --account=def-liyang
#SBATCH --array=0-71%12
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=logs/lstm_hpo_s1_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

python -m src.hpo.runner_lstm_ae \
    --manifest "$SWAT_OUTPUT_DIR/hpo/manifests/lstm_ae_stage1.jsonl" \
    --task-id  "$SLURM_ARRAY_TASK_ID" \
    --stage    lstm_ae_stage1
