#!/bin/bash
# Gate job: runs after AE Stage 1 finishes (via afterany dependency).
# Aggregates Stage 1 → picks top 3 → generates Stage 2 manifest → submits
# Stage 2 → chains the next gate (S2→S3) with afterany on Stage 2.
#
# This is the first gate in the AE chain. The orchestrator submits it.

#SBATCH --job-name=gate_ae_s1_s2
#SBATCH --account=def-liyang
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --output=logs/gate_ae_s1_s2_%j.out

set -euo pipefail
# Slurm copies this script to a spool dir and sets $0 to that copy,
# so $(dirname $0) doesn't find _common.sh. Use SLURM_SUBMIT_DIR (the
# directory from which sbatch was invoked — the project root).
source "$SLURM_SUBMIT_DIR/slurm/_common.sh"

MIN_SUCCESS=170   # out of 192 — require 88%+ success rate to proceed

# Count successful S1 JSONs (exclude .error.json)
N_OK=$(find "$SWAT_OUTPUT_DIR/hpo/results/ae_stage1" -maxdepth 1 \
         -name '[0-9]*.json' ! -name '*.error.json' 2>/dev/null | wc -l)
N_ERR=$(find "$SWAT_OUTPUT_DIR/hpo/results/ae_stage1" -maxdepth 1 \
          -name '*.error.json' 2>/dev/null | wc -l)
echo "AE Stage 1: $N_OK successes, $N_ERR errors"

if [[ $N_OK -lt $MIN_SUCCESS ]]; then
    echo "ERROR: only $N_OK/192 successes (threshold $MIN_SUCCESS) — chain aborted"
    exit 1
fi

# Aggregate + generate next manifest
python scripts/aggregate_hpo.py --stage ae_stage1
python scripts/make_hpo_manifests.py --stage ae_stage2 \
    --top-from "$SWAT_OUTPUT_DIR/hpo/results/ae_stage1.csv"

# Submit Stage 2 array
S2_JOBID=$(sbatch --parsable slurm/51_ae_hpo_stage2.sh)
echo "Submitted AE Stage 2 array as $S2_JOBID"

# Chain the next gate (S2 → S3)
GATE_JOBID=$(sbatch --parsable --dependency=afterany:$S2_JOBID \
    slurm/92_gate_ae_s2_to_s3.sh)
echo "Chained AE S2→S3 gate as $GATE_JOBID (waits on $S2_JOBID)"
