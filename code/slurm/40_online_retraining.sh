#!/bin/bash
#SBATCH --job-name=swat_online
#SBATCH --account=def-liyang
#SBATCH --array=0-989%24
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/online_%A_%a.out

# Phase 1.3 — paper headline: online / retraining poisoning.
#
# 990 array tasks = 11 detectors × 2 generators × 3 T × 3 Δp × 5 seeds.
# Each task runs T+1 trainings (round 0 = clean baseline + T poisoning rounds).
# Per-round JSONs land in $SWAT_OUTPUT_DIR/checkpoints/online/ so a mid-task
# failure costs at most one round's training, not the whole trajectory.
#
# Concurrency cap %24: more parallel than the baseline grid because many
# (detector, T) cells finish well under the 8 h wallclock and the manifest
# is much larger.
#
# PRE-FLIGHT (run once on a login node before the first sbatch):
#   1) source venv/bin/activate
#   2) python scripts/make_online_manifest.py
#   3) ls $SWAT_OUTPUT_DIR/online_manifest.tsv
#
# RESUMING / partial reruns:
#   * Re-submitting the whole array is safe — completed combos skip in ms.
#   * Per-round skip-if-exists is also implemented; deleting a single round
#     JSON forces just that round's recomputation (the runner replays the
#     cumulative-pool state from the prior rounds' JSONs).
#   * For just the still-missing tasks:
#         python scripts/missing_combos.py --grid online
#         bash $SWAT_OUTPUT_DIR/resubmit_online.sh

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

python -m src.run_online --task-id "$SLURM_ARRAY_TASK_ID"
