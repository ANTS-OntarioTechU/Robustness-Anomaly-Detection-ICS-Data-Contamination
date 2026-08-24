#!/bin/bash
# One-shot submission of the feature-noise σ sweep.
# Submits three Slurm arrays — one per σ value — each 33 tasks.
#
# Usage:
#     cd ~/projects/def-liyang/$USER/narval_swat_run
#     bash scripts/submit_sigma_sweep.sh
#
# Override the σ list by exporting SIGMAS before invocation:
#     SIGMAS="0.20 0.40 0.60 0.80" bash scripts/submit_sigma_sweep.sh

set -euo pipefail

: "${SIGMAS:=0.30 0.50 1.00}"

cd "$(dirname "$0")/.."

for sigma in $SIGMAS; do
    echo "Submitting σ = $sigma ..."
    SWAT_SIGMA="$sigma" sbatch slurm/40_sigma_sweep.sh
done

echo
echo "Submitted. Check queue with:"
echo "    squeue -u \$USER"
echo
echo "Outputs will land at:"
for sigma in $SIGMAS; do
    tag=$(echo "$sigma" | tr '.' 'p')
    echo "    \$SWAT_OUTPUT_DIR/checkpoints/sigma_sweep/s${tag}/"
done
