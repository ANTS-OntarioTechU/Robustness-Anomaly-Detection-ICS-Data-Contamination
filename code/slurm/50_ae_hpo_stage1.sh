#!/bin/bash
# AE HPO — Stage 1 (architecture sweep, 192 configs).
# Each task evaluates one config under clean + 10% targeted-flip poisoning.
#
# Manifest: $SWAT_OUTPUT_DIR/hpo/manifests/ae_stage1.jsonl
# Outputs : $SWAT_OUTPUT_DIR/hpo/results/ae_stage1/{task_id:04d}.json
#
# Concurrency cap (%N) keeps the def-liyang allocation honest. Bump to %16 or
# higher if your fairshare allows; bump down if jobs queue too long.
#
# Wall-clock budget: clean+poisoned AE ≈ 200-280 s on A100 → 1 h is comfortable.

#SBATCH --job-name=swat_ae_hpo_s1
#SBATCH --account=def-liyang
#SBATCH --array=0-191%12
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/ae_hpo_s1_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

python -m src.hpo.runner_ae \
    --manifest "$SWAT_OUTPUT_DIR/hpo/manifests/ae_stage1.jsonl" \
    --task-id  "$SLURM_ARRAY_TASK_ID" \
    --stage    ae_stage1
