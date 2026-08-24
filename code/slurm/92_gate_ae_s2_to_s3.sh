#!/bin/bash
# Gate job: AE Stage 2 → Stage 3.

#SBATCH --job-name=gate_ae_s2_s3
#SBATCH --account=def-liyang
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --output=logs/gate_ae_s2_s3_%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/_common.sh"

MIN_SUCCESS=128   # out of 144 (~89%)

N_OK=$(find "$SWAT_OUTPUT_DIR/hpo/results/ae_stage2" -maxdepth 1 \
         -name '[0-9]*.json' ! -name '*.error.json' 2>/dev/null | wc -l)
N_ERR=$(find "$SWAT_OUTPUT_DIR/hpo/results/ae_stage2" -maxdepth 1 \
          -name '*.error.json' 2>/dev/null | wc -l)
echo "AE Stage 2: $N_OK successes, $N_ERR errors"
if [[ $N_OK -lt $MIN_SUCCESS ]]; then
    echo "ERROR: only $N_OK/144 successes — chain aborted"
    exit 1
fi

python scripts/aggregate_hpo.py --stage ae_stage2
python scripts/make_hpo_manifests.py --stage ae_stage3 \
    --top-from "$SWAT_OUTPUT_DIR/hpo/results/ae_stage2.csv"

S3_JOBID=$(sbatch --parsable slurm/52_ae_hpo_stage3.sh)
echo "Submitted AE Stage 3 array as $S3_JOBID"

GATE_JOBID=$(sbatch --parsable --dependency=afterany:$S3_JOBID \
    slurm/93_gate_ae_s3_to_final.sh)
echo "Chained AE S3→Final gate as $GATE_JOBID (waits on $S3_JOBID)"
