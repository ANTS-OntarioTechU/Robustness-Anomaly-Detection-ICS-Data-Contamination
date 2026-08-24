#!/bin/bash
# Gate job: runs after LSTM-AE Stage 1 finishes.
# Aggregates → picks top 3 → generates Stage 2 manifest → submits Stage 2
# → chains the S2→S3 gate.

#SBATCH --job-name=gate_lstm_s1_s2
#SBATCH --account=def-liyang
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --output=logs/gate_lstm_s1_s2_%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/_common.sh"

MIN_SUCCESS=64   # out of 72 (~89%)

N_OK=$(find "$SWAT_OUTPUT_DIR/hpo/results/lstm_ae_stage1" -maxdepth 1 \
         -name '[0-9]*.json' ! -name '*.error.json' 2>/dev/null | wc -l)
N_ERR=$(find "$SWAT_OUTPUT_DIR/hpo/results/lstm_ae_stage1" -maxdepth 1 \
          -name '*.error.json' 2>/dev/null | wc -l)
echo "LSTM-AE Stage 1: $N_OK successes, $N_ERR errors"
if [[ $N_OK -lt $MIN_SUCCESS ]]; then
    echo "ERROR: only $N_OK/72 successes — chain aborted"
    exit 1
fi

python scripts/aggregate_hpo.py --stage lstm_ae_stage1
python scripts/make_hpo_manifests.py --stage lstm_ae_stage2 \
    --top-from "$SWAT_OUTPUT_DIR/hpo/results/lstm_ae_stage1.csv"

S2_JOBID=$(sbatch --parsable slurm/61_lstm_ae_hpo_stage2.sh)
echo "Submitted LSTM-AE Stage 2 array as $S2_JOBID"

GATE_JOBID=$(sbatch --parsable --dependency=afterany:$S2_JOBID \
    slurm/95_gate_lstm_s2_to_s3.sh)
echo "Chained LSTM-AE S2→S3 gate as $GATE_JOBID (waits on $S2_JOBID)"
