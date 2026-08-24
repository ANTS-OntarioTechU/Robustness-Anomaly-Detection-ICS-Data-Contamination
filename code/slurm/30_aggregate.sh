#!/bin/bash
#SBATCH --job-name=swat_aggregate
#SBATCH --account=def-liyang
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/aggregate_%j.out

# Aggregation is pure pandas/matplotlib — no GPU needed.
# Produces: checkpoints/{clean_baselines,attack_checkpoint}.csv, all_results.csv,
#           tables T4/T5, compute_cost, multi_criteria_ranking, figures F3–F8,
#           run_summary.txt (missing-combo report).
#
# You can also run this on a login node interactively:
#   cd $SWAT_PROJECT_DIR && source venv/bin/activate && python -m src.aggregate

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/_common.sh"

python -m src.aggregate
