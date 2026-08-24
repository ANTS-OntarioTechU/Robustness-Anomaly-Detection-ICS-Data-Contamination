#!/bin/bash
# Gate job: AE Stage 3 → Final. This is the last AE gate — no downstream
# gate to chain, just submit the Final array and let it finish.

#SBATCH --job-name=gate_ae_s3_final
#SBATCH --account=def-liyang
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --output=logs/gate_ae_s3_final_%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/_common.sh"

MIN_SUCCESS=72   # out of 81 (~89%)

N_OK=$(find "$SWAT_OUTPUT_DIR/hpo/results/ae_stage3" -maxdepth 1 \
         -name '[0-9]*.json' ! -name '*.error.json' 2>/dev/null | wc -l)
N_ERR=$(find "$SWAT_OUTPUT_DIR/hpo/results/ae_stage3" -maxdepth 1 \
          -name '*.error.json' 2>/dev/null | wc -l)
echo "AE Stage 3: $N_OK successes, $N_ERR errors"
if [[ $N_OK -lt $MIN_SUCCESS ]]; then
    echo "ERROR: only $N_OK/81 successes — chain aborted"
    exit 1
fi

python scripts/aggregate_hpo.py --stage ae_stage3
python scripts/make_hpo_manifests.py --stage ae_final \
    --top-from "$SWAT_OUTPUT_DIR/hpo/results/ae_stage3.csv"

FINAL_JOBID=$(sbatch --parsable slurm/53_ae_hpo_final.sh)
echo "Submitted AE Final array as $FINAL_JOBID"
echo "AE chain complete — Final will aggregate separately once it finishes."
