"""
Hyperparameter-optimization sub-package for AE and LSTM-AE.

Companion to the main detector zoo. The HPO pipeline evaluates every candidate
config under two conditions (clean training pool + 10 % targeted-flip poisoning)
and ships per-task JSON results suitable for Phase-1-style sensitivity analysis.

Usage
-----
1. Generate manifests (one JSONL per stage per detector):

    python scripts/make_hpo_manifests.py --stage ae_stage1
    python scripts/make_hpo_manifests.py --stage lstm_ae_stage1

2. Submit a Slurm array whose task-id indexes the manifest:

    sbatch slurm/50_ae_hpo_stage1.sh

3. Each Slurm task calls the runner:

    python -m src.hpo.runner_ae \
        --manifest $SWAT_OUTPUT_DIR/hpo/manifests/ae_stage1.jsonl \
        --task-id $SLURM_ARRAY_TASK_ID

Stages 2, 3, and final depend on the previous stage's top-3 configs;
regenerate those manifests after the preceding stage finishes.
"""
