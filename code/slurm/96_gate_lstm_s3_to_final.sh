#!/bin/bash
# Gate job: LSTM-AE Stage 3 → Final. End of the LSTM-AE chain.

#SBATCH --job-name=gate_lstm_s3_final
#SBATCH --account=def-liyang
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --output=logs/gate_lstm_s3_final_%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/_common.sh"

MIN_SUCCESS=24   # out of 27 (~89%)

N_OK=$(find "$SWAT_OUTPUT_DIR/hpo/results/lstm_ae_stage3" -maxdepth 1 \
         -name '[0-9]*.json' ! -name '*.error.json' 2>/dev/null | wc -l)
N_ERR=$(find "$SWAT_OUTPUT_DIR/hpo/results/lstm_ae_stage3" -maxdepth 1 \
          -name '*.error.json' 2>/dev/null | wc -l)
echo "LSTM-AE Stage 3: $N_OK successes, $N_ERR errors"
if [[ $N_OK -lt $MIN_SUCCESS ]]; then
    echo "ERROR: only $N_OK/27 successes — chain aborted"
    exit 1
fi

python scripts/aggregate_hpo.py --stage lstm_ae_stage3
python scripts/make_hpo_manifests.py --stage lstm_ae_final \
    --top-from "$SWAT_OUTPUT_DIR/hpo/results/lstm_ae_stage3.csv"

FINAL_JOBID=$(sbatch --parsable slurm/63_lstm_ae_hpo_final.sh)
echo "Submitted LSTM-AE Final array as $FINAL_JOBID"
echo "LSTM-AE chain complete — Final will aggregate separately once it finishes."
