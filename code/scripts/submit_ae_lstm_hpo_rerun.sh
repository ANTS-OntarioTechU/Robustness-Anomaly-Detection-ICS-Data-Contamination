#!/bin/bash
# Phase 1.2 helper — rerun AE + LSTM-AE rows of the existing main grid under
# the post-HPO config (paper/SWaT_config_patch.md applied to src/config.py
# and src/models.py).
#
# This script:
#   1) Backs up existing AE/LSTM-AE JSONs to checkpoints/.pre_hpo_backup/
#   2) Submits clean baselines for AE/LSTM-AE task IDs (30..35) only
#   3) Chains the three baseline-attack arrays after the clean baselines
#      finish (--dependency=afterok), also restricted to task IDs 30..35
#
# Task ID mapping (from src/config.decode_clean_task with 12 models × 3 seeds):
#   autoencoder × {42, 123, 456} → 30, 31, 32
#   lstm_ae     × {42, 123, 456} → 33, 34, 35
#
# Run on a Narval LOGIN node (not via sbatch directly):
#
#     cd ~/projects/def-liyang/$USER/narval_swat_run
#     bash scripts/submit_ae_lstm_hpo_rerun.sh
#
# After the chained jobs finish, re-run aggregation:
#
#     sbatch slurm/30_aggregate.sh
#
# This regenerates Tables VII–VIII and Figs 2/5/6/8 with HPO-winner numbers
# in the AE/LSTM-AE rows.

set -euo pipefail

: "${SWAT_PROJECT_DIR:=$HOME/projects/def-liyang/$USER/narval_swat_run}"
: "${SWAT_OUTPUT_DIR:=$SCRATCH/swat_paper_run}"
cd "$SWAT_PROJECT_DIR"

# ── 1) Backup existing AE/LSTM-AE JSONs ─────────────────────────────
BACKUP="$SWAT_OUTPUT_DIR/checkpoints/.pre_hpo_backup"
mkdir -p "$BACKUP"
moved=0
for d in clean attacks/random_flip attacks/targeted_flip attacks/feature_noise; do
    src="$SWAT_OUTPUT_DIR/checkpoints/$d"
    dst="$BACKUP/$d"
    [[ -d "$src" ]] || continue
    mkdir -p "$dst"
    for m in autoencoder lstm_ae; do
        for s in 42 123 456; do
            for f in "$src"/${m}__${s}*.json; do
                if [[ -f "$f" ]]; then
                    mv "$f" "$dst/"
                    moved=$((moved + 1))
                fi
            done
        done
    done
done
echo "[hpo_rerun] backed up $moved JSON(s) -> $BACKUP"

# ── 2) Submit fresh AE/LSTM-AE clean baselines (task IDs 30–35) ─────
JID_CLEAN=$(sbatch --parsable --array=30-35 \
    --job-name=swat_clean_hpo \
    slurm/10_clean_baselines.sh)
echo "[hpo_rerun] clean baselines  : $JID_CLEAN  (array=30-35)"

# ── 3) Chain the 3 baseline-attack arrays after clean finishes ──────
JID_RFL=$(sbatch --parsable --array=30-35 \
    --dependency=afterok:"$JID_CLEAN" \
    --job-name=swat_atk_random_hpo \
    slurm/20_attack_random_flip.sh)
echo "[hpo_rerun] random_flip      : $JID_RFL  (after $JID_CLEAN)"

JID_TFL=$(sbatch --parsable --array=30-35 \
    --dependency=afterok:"$JID_CLEAN" \
    --job-name=swat_atk_targeted_hpo \
    slurm/21_attack_targeted_flip.sh)
echo "[hpo_rerun] targeted_flip    : $JID_TFL  (after $JID_CLEAN)"

JID_FNZ=$(sbatch --parsable --array=30-35 \
    --dependency=afterok:"$JID_CLEAN" \
    --job-name=swat_atk_noise_hpo \
    slurm/22_attack_feature_noise.sh)
echo "[hpo_rerun] feature_noise    : $JID_FNZ  (after $JID_CLEAN)"

cat <<EOF

[hpo_rerun] All jobs submitted.
            Tail logs as they appear:
                tail -f logs/clean_${JID_CLEAN}_30.out

            When everything finishes, regenerate paper artifacts:
                sbatch slurm/30_aggregate.sh
EOF
