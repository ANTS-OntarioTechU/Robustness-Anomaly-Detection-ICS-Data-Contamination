# AE / LSTM-AE Hyperparameter Optimization — Operating Manual

A 642-task package for systematically optimizing the feed-forward Autoencoder
(AE) and the LSTM-Autoencoder (LSTM-AE) on SWaT under both clean and 10 %
targeted-poisoning conditions. Output is per-config JSON suitable for the same
Phase-1-style sensitivity analysis used to justify the PCA and SVM tunings.

This package only ships the experiment harness (Slurm + Python runners + grid
generator). Aggregation/analysis scripts and paper-section text are intentionally
out of scope — they will be built once the runs come back.

---

## What gets evaluated

| Detector | Stage 1 (architecture) | Stage 2 (training) | Stage 3 (loss/threshold/scaler) | Final (3 seeds × 13-cell grid) |
|---|---|---|---|---|
| **AE** | 192 configs (6 hidden × 4 dropout × 4 act × 2 BN) | 144 (top-3 × 2 opt × 4 LR × 3 batch × 2 ep) | 81 (top-3 × 3 loss × 3 thresh × 3 scaler) | 9 tasks |
| **LSTM-AE** | 72 configs (4 window × 3 hidden × 2 layers × 3 dropout) | 108 (top-3 × 2 opt × 3 LR × 3 batch × 2 ep) | 27 (top-3 × 3 loss × 3 thresh) | 9 tasks |

**Total Slurm tasks: 642.**

Stages 1–3 evaluate each candidate under **two conditions in a single Slurm task**:
the clean training pool **plus** 10 % targeted-flip poisoning. The output JSON has
both blocks plus a `composite_f1 = mean(clean_f1, poisoned_f1)` ranking key. The
final stage re-runs the top-3 winners across 3 seeds × the full poisoning grid
(clean + 3 attacks × 4 rates).

---

## Files added in this package

```
src/hpo/
  __init__.py
  grids.py                  # All grid definitions + enumeration helpers
  configurable_models.py    # ConfigurableAEDetector, ConfigurableLSTMAEDetector
  runner_ae.py              # CLI: trains+evals one AE config under clean+poisoned
  runner_lstm_ae.py         # CLI: same for LSTM-AE
slurm/
  50_ae_hpo_stage1.sh       # 192-task array
  51_ae_hpo_stage2.sh       # 144-task array (manifest depends on Stage 1)
  52_ae_hpo_stage3.sh       # 81-task array
  53_ae_hpo_final.sh        # 9-task array (top-3 × 3 seeds, 13-cell grid each)
  60_lstm_ae_hpo_stage1.sh  # 72-task array
  61_lstm_ae_hpo_stage2.sh  # 108-task array
  62_lstm_ae_hpo_stage3.sh  # 27-task array
  63_lstm_ae_hpo_final.sh   # 9-task array
scripts/
  make_hpo_manifests.py     # Generates JSONL manifest per stage
  README_HPO.md             # this file
```

Re-uses (no edits to existing code): `src/data.py`, `src/attacks.py`,
`src/eval_utils.py`, `src/config.py`, `slurm/_common.sh`.

---

## Compute budget (rough estimates)

Per-task wall-clock (clean + 10 % targeted, single A100, 6 CPUs):

| Stage | Avg task | Worst-case task | Array size | At 12-way concurrency |
|---|---|---|---|---|
| AE Stage 1 | ~4 min | ~6 min (deepest+widest) | 192 | ~80 min |
| AE Stage 2 | ~5 min | ~8 min (bs=512, 100 ep) | 144 | ~75 min |
| AE Stage 3 | ~5 min | ~7 min | 81 | ~45 min |
| AE Final | ~28 min/task (13 trains) | ~35 min | 9 | ~28 min (one wave) |
| LSTM-AE Stage 1 | ~12 min | ~25 min (W=50, h=256, L=2) | 72 | ~150 min |
| LSTM-AE Stage 2 | ~14 min | ~22 min | 108 | ~165 min |
| LSTM-AE Stage 3 | ~14 min | ~20 min | 27 | ~50 min |
| LSTM-AE Final | ~50 min/task | ~70 min | 9 | ~50 min |

**End-to-end ≈ 12–15 wall-clock hours at 12-way concurrency**, including
human-in-the-loop pauses to regenerate manifests after Stages 1 and 2. Bumping
the `%N` concurrency cap in each Slurm script (currently 12) shortens this
linearly until you hit the `def-liyang` fairshare ceiling.

---

## End-to-end run order

```bash
# 0. From your laptop, copy this package up to Narval and cd in
rsync -av narval_swat_run/ \
    narval:projects/def-liyang/$USER/narval_swat_run/

ssh narval
cd ~/projects/def-liyang/$USER/narval_swat_run

# 1. Bootstrap a tmux session and load the env (per the master handoff)
tmux new -s swat-hpo
module load StdEnv/2023 python/3.11 scipy-stack cuda/12.2
source venv/bin/activate
export SWAT_DATA_DIR=/scratch/$USER/swat_data
export SWAT_OUTPUT_DIR=/scratch/$USER/swat_paper_run

# 2. Stage-1 manifests (no prerequisites)
python scripts/make_hpo_manifests.py --stage ae_stage1
python scripts/make_hpo_manifests.py --stage lstm_ae_stage1

# 3. Submit Stage 1 for both detectors (independent — submit in parallel)
sbatch slurm/50_ae_hpo_stage1.sh
sbatch slurm/60_lstm_ae_hpo_stage1.sh

# 4. WAIT for Stage 1 to finish. Check with:
squeue -u $USER
ls $SWAT_OUTPUT_DIR/hpo/results/ae_stage1/ | wc -l       # expect 192
ls $SWAT_OUTPUT_DIR/hpo/results/lstm_ae_stage1/ | wc -l  # expect 72

# 5. Aggregate Stage-1 JSONs into a CSV. (You'll write this aggregator
#    in the follow-up session; it should produce one row per task with at
#    least these columns: task_id, composite_f1, clean_f1, poisoned_f1,
#    delta_f1, config (JSON-encoded). Sort/deduplicate is handled by
#    make_hpo_manifests.py.)
python scripts/aggregate_hpo.py --stage ae_stage1       # NOT IN THIS PACKAGE
python scripts/aggregate_hpo.py --stage lstm_ae_stage1  # NOT IN THIS PACKAGE

# 6. Generate Stage-2 manifests using top-3 winners from Stage 1
python scripts/make_hpo_manifests.py --stage ae_stage2 \
    --top-from $SWAT_OUTPUT_DIR/hpo/results/ae_stage1.csv
python scripts/make_hpo_manifests.py --stage lstm_ae_stage2 \
    --top-from $SWAT_OUTPUT_DIR/hpo/results/lstm_ae_stage1.csv

# 7. Submit Stage 2
sbatch slurm/51_ae_hpo_stage2.sh
sbatch slurm/61_lstm_ae_hpo_stage2.sh

# 8. WAIT, aggregate, generate Stage-3 manifests, repeat the loop
python scripts/make_hpo_manifests.py --stage ae_stage3 \
    --top-from $SWAT_OUTPUT_DIR/hpo/results/ae_stage2.csv
python scripts/make_hpo_manifests.py --stage lstm_ae_stage3 \
    --top-from $SWAT_OUTPUT_DIR/hpo/results/lstm_ae_stage2.csv
sbatch slurm/52_ae_hpo_stage3.sh
sbatch slurm/62_lstm_ae_hpo_stage3.sh

# 9. Final confirmation runs (top-3 winners × 3 seeds × full poisoning grid)
python scripts/make_hpo_manifests.py --stage ae_final \
    --top-from $SWAT_OUTPUT_DIR/hpo/results/ae_stage3.csv
python scripts/make_hpo_manifests.py --stage lstm_ae_final \
    --top-from $SWAT_OUTPUT_DIR/hpo/results/lstm_ae_stage3.csv
sbatch slurm/53_ae_hpo_final.sh
sbatch slurm/63_lstm_ae_hpo_final.sh
```

The aggregator script (`scripts/aggregate_hpo.py`) is **not in this package**.
You only need it once Stage 1 finishes, and the next session can build it then.
What it must produce: a CSV with `task_id, composite_f1, clean_f1, poisoned_f1,
delta_f1, config` columns sorted by composite_f1 descending, written to
`$SWAT_OUTPUT_DIR/hpo/results/{stage}.csv`. The format is consumed by
`make_hpo_manifests.py --top-from`.

---

## Per-task output JSON layout

Each Slurm task writes one file:
`$SWAT_OUTPUT_DIR/hpo/results/{stage}/{task_id:04d}.json`

```json
{
  "stage":           "ae_stage1",
  "task_id":         17,
  "config":          { ...full config from manifest line... },
  "input_dim":       44,
  "n_train_normal":  246612,
  "results": {
    "clean": {
      "f1": 0.8702, "fnr": 0.2050, "precision": ..., "recall": ...,
      "roc_auc": ..., "pr_auc": ..., "accuracy": ...,
      "tp": ..., "tn": ..., "fp": ..., "fn": ...,
      "threshold": 0.0182, "train_time": 87.3,
      "stopped_epoch": 42, "best_val_loss": 0.00191, "n_params": 106988
    },
    "poisoned": {
      ...same fields...,
      "poison_attack": "targeted_flip",
      "poison_rate":   0.10,
      "effective_contamination": 0.0909,
      "n_injected":              22418
    }
  },
  "composite_f1": 0.7641,
  "delta_f1":    -0.2122,
  "wallclock_s": 247.1
}
```

For final-stage files, `results` has 13 keys instead of 2:
`clean`, `random_flip__r0.01`, `random_flip__r0.03`, …, `feature_noise__r0.10`.
`composite_f1` and `delta_f1` then anchor on `targeted_flip__r0.10`.

Failed tasks write to `{task_id:04d}.error.json` with the same envelope plus
`error` and `traceback` fields. Re-running the array picks them up because the
runner skips existing non-error JSONs and overwrites errors.

---

## Idempotency, restart, debug

* **Idempotent.** Each runner skips its task if the output JSON already exists.
  Re-submitting the array is safe and only re-runs missing tasks.
* **Force re-run.** Pass `--force` (also accepted by the runner) to overwrite.
* **Single-task debug.** From an interactive `salloc`:
  ```bash
  python -m src.hpo.runner_ae \
      --manifest $SWAT_OUTPUT_DIR/hpo/manifests/ae_stage1.jsonl \
      --task-id 0 --stage ae_stage1
  ```
* **Find errors.**
  ```bash
  find $SWAT_OUTPUT_DIR/hpo/results -name '*.error.json' | xargs -I{} jq '.error' {}
  ```
* **Find missing tasks.**
  ```bash
  comm -23 \
      <(seq 0 191 | xargs -I{} printf '%04d\n' {} | sort) \
      <(ls $SWAT_OUTPUT_DIR/hpo/results/ae_stage1/ | sed 's/\.json//' | sort) \
      | head
  ```

---

## What this design fixes

| Gap in the current paper | What this package adds |
|---|---|
| AE HPO existed but was clean-F1-only — could not justify the deployed config under poisoning | Every candidate evaluated under clean + 10 % targeted poisoning; final stage adds the full poisoning grid + variance |
| LSTM-AE had **no** HPO at all — Section 4.2 of the paper says its parameters came from "standard practice + small design ablations" | A full 3-stage grid (216 tasks) parallel to the AE design, ranked on composite_f1 |
| No paper-friendly sensitivity table for AE/LSTM-AE comparable to PCA's `n_components` and SVM's `nu` | Stage-1 outputs let you compute per-knob ΔF1 in the same Phase-1 format and apply the same promotion rule (ΔF1 ≥ 0.03 OR ΔFNR ≤ −0.05) |

---

## Knobs you might tune before running

All in `src/hpo/grids.py`:

* `AE_FIXED_STAGE1.poison_rate` — currently 0.10 (the paper's strongest cell).
  Drop to 0.05 if you want a lighter poisoning anchor.
* `AE_FIXED_STAGE1.poison_attack` — `targeted_flip` is the most damaging single
  attack family per the paper; switch to `random_flip` if the question is
  "which configs survive the early-rate catastrophe".
* `FINAL_RATES` / `FINAL_ATTACKS` — the full grid evaluated by Stage Final.
  Currently mirrors the main paper grid.
* `%N` concurrency in each Slurm script — bump up if your fairshare allows.
* `--time` budgets — generous by default; trim once you have a few real runs
  to measure against.

---

## Pre-flight checklist

```bash
# Same as the existing project
bash scripts/check_env.sh

# New: confirm the HPO package imports cleanly
python -c "from src.hpo.grids import enumerate_ae_stage1, enumerate_lstm_ae_stage1; \
           print('AE s1', len(enumerate_ae_stage1())); \
           print('LSTM s1', len(enumerate_lstm_ae_stage1()))"
# Expected: AE s1 192 / LSTM s1 72

# Smoke-test a single task interactively (skip if you trust it):
python -m src.hpo.runner_ae \
    --manifest $SWAT_OUTPUT_DIR/hpo/manifests/ae_stage1.jsonl \
    --task-id 0 --stage ae_stage1
```

If any import fails on Narval, it's almost always the `__sklearn_tags__` shim
issue covered in §7 of the master handoff — but the HPO runners don't import
from `src/models.py` so they shouldn't hit it. The PyOD shim is irrelevant
here because AE/LSTM-AE are pure PyTorch.
