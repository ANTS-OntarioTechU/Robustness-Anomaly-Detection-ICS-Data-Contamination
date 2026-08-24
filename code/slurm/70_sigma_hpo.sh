#!/bin/bash
# σ-ablation on HPO winners — 42 tasks covering 2 detectors × 3 seeds ×
# 7 (σ, rate) pairs. Each task runs clean + feature-noise-poisoned training
# for a single (detector, σ, rate, seed) combination.
#
# Manifest: $SWAT_OUTPUT_DIR/hpo/manifests/sigma_hpo.jsonl
# Outputs : $SWAT_OUTPUT_DIR/hpo/results/sigma_hpo/{task_id:04d}.json
#
# Prerequisite: both final-stage aggregations (ae_final_by_config.csv and
# lstm_ae_final_by_config.csv) must exist, and the manifest must be generated:
#
#     python scripts/aggregate_hpo.py --stage ae_final --multi-seed
#     python scripts/aggregate_hpo.py --stage lstm_ae_final --multi-seed
#     python scripts/make_hpo_manifests.py --stage sigma_hpo \\
#         --ae-from   $SWAT_OUTPUT_DIR/hpo/results/ae_final_by_config.csv \\
#         --lstm-from $SWAT_OUTPUT_DIR/hpo/results/lstm_ae_final_by_config.csv
#
# Wall-clock budget: AE tasks ~3-5 min (2 trains at batch=2048), LSTM-AE tasks
# ~8-15 min (window=30, hidden=256). 1 h per task is comfortably safe.

#SBATCH --job-name=swat_sigma_hpo
#SBATCH --account=def-liyang
#SBATCH --array=0-41%12
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/sigma_hpo_%A_%a.out

set -euo pipefail
# Use SLURM_SUBMIT_DIR (the directory where sbatch was invoked — the project
# root) instead of $(dirname "$0"), because Slurm copies the script to a
# spool dir without the sibling _common.sh.
source "$SLURM_SUBMIT_DIR/slurm/_common.sh"

python -m src.hpo.runner_sigma_hpo \
    --manifest "$SWAT_OUTPUT_DIR/hpo/manifests/sigma_hpo.jsonl" \
    --task-id  "$SLURM_ARRAY_TASK_ID"
