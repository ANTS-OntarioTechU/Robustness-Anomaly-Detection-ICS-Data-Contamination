# SWaT Anomaly IDS Poisoning — Narval Paper Run

Narval-adapted package for the paper-v2 experiment. Scientific logic is
faithful to `SWaT_Anomaly_IDS_Poisoning_PAPER_RUN.ipynb` (full-data LSTM-AE
split + tuned PCA/SVM). Operational design is HPC-native: code lives in
`~/projects`, data and outputs live in `$SCRATCH`, heavy execution is driven
by Slurm arrays with per-combo JSON checkpointing so any interruption costs
at most one combo.

---

## What this package contains

```
narval_swat_run/
├── README.md                          ← this file
├── env/
│   ├── setup_venv.sh                  one-shot venv bootstrap
│   └── requirements.txt               Alliance-wheelhouse-friendly pins
├── src/                               experiment engine
│   ├── config.py                      grid, tuned params, paths, task-id decoder
│   ├── data.py                        load, preprocess, both splits
│   ├── models.py                      10 PyOD + Autoencoder + LSTM-AE
│   ├── attacks.py                     random_flip / targeted_flip / feature_noise
│   ├── eval_utils.py                  seed, metrics, F1-opt threshold, atomic IO
│   ├── run_clean.py                   CLI entry — one (model, seed)
│   ├── run_attack.py                  CLI entry — one (model, seed), 4 rates
│   └── aggregate.py                   merges JSONs → CSVs + figures + tables
├── slurm/
│   ├── _common.sh                     sourced by every sbatch (modules, venv, env vars)
│   ├── 00_smoke_test.sh               single-task sanity check
│   ├── 10_clean_baselines.sh          array 0–35, clean baselines
│   ├── 20_attack_random_flip.sh       array 0–35, random_flip × 4 rates
│   ├── 21_attack_targeted_flip.sh     array 0–35, targeted_flip × 4 rates
│   ├── 22_attack_feature_noise.sh     array 0–35, feature_noise × 4 rates
│   ├── 30_aggregate.sh                final aggregation job
│   └── 40_online_retraining.sh        array 0–989, archival online-retraining extension
├── notebook/
│   └── SWaT_Narval_Interactive.ipynb  Layer-1 inspection / diagnostic notebook
└── scripts/
    ├── check_env.sh                   pre-flight check
    ├── missing_combos.py              writes a resubmit.sh for stragglers
    ├── make_online_manifest.py        builds $SWAT_OUTPUT_DIR/online_manifest.tsv
    └── submit_ae_lstm_hpo_rerun.sh    Phase 1.2 helper (chains AE/LSTM-AE rerun)
```

**R02 source files** (added/modified during the post-pilot revision):

| File | Status | Purpose |
|---|---|---|
| `src/attacks_online.py` | NEW | Online-retraining attack-pool ranking + helpers |
| `src/run_online.py` | NEW | Per-trajectory Slurm runner (manifest-driven) |
| `src/config.py` | PATCHED + ONLINE_* added | HPO winners now in CONFIG; online grid spec |
| `src/models.py` | PATCHED | AE activation + batch-size + LSTM-AE optimizer/batch read from CONFIG |
| `src/config.py.pre_hpo`, `src/models.py.pre_hpo` | NEW (backups) | Pre-HPO snapshots for rollback |

---

## Grid

- **Models (12):** `iforest svm lof cluster knn histogram pca mcd abod sod autoencoder lstm_ae`
- **Seeds (3):** `42 123 456`
- **Attacks (3):** `random_flip targeted_flip feature_noise`
- **Poison rates (4):** `0.01 0.03 0.05 0.10`
- **Total:** 36 clean baselines + 3 × 144 attack combos = **468 runs**
- **Array shape:** each array has 36 tasks (`model × seed`). Attack arrays
  loop the 4 rates inside each task so preprocessing and splits are amortized.

---

## Storage layout on Narval

| What | Where |
|---|---|
| Code, notebook, Slurm scripts | `~/projects/def-liyang/$USER/narval_swat_run/` |
| Raw data (`normal.csv`, `attack.csv`) | `$SCRATCH/swat_data/` |
| Experiment outputs (checkpoints, CSVs, figures, logs) | `$SCRATCH/swat_paper_run/` |
| Venv | `~/projects/def-liyang/$USER/narval_swat_run/venv/` |

The three env vars that drive path resolution — `SWAT_PROJECT_DIR`,
`SWAT_DATA_DIR`, `SWAT_OUTPUT_DIR` — all default correctly if you follow this
layout. Override any of them by exporting before invoking sbatch.

---

## Two ways to run

**Option A — Cell-by-cell validation first.** Open
[`notebook/SWaT_Narval_CellByCell.ipynb`](notebook/SWaT_Narval_CellByCell.ipynb)
inside an interactive GPU `salloc` and step through it. Same 49 scientific
cells as the original paper notebook, minus the Colab / Drive bits. Good for
the first end-to-end run because you see every stage (data load, splits,
diagnostics, grid) fire in order. Checkpoints persist, so a dropped session
resumes cleanly. Plan for 3–6 h in the `salloc` if you intend to finish the
full grid this way.

**Option B — Slurm arrays (scripted & parallel).** Use the sbatch scripts
under `slurm/`. Faster wall-clock for the full grid (array tasks run in
parallel), and the pipeline is cheaper to resume because each combo is its
own JSON file. Recommended once you've confirmed Option A works end-to-end.

Note on checkpoint format: A and B use **different** checkpoint layouts —
the notebook writes shared CSVs (`checkpoints/clean_baselines.csv`,
`checkpoints/attack_checkpoint.csv`) while Slurm writes per-combo JSONs
(`checkpoints/clean/*.json`, `checkpoints/attacks/<atk>/*.json`). They
don't cross-resume. A typical workflow is: use A to validate stages 1–5.1b
(data, splits, LSTM-AE diagnostic), then switch to B for the 468-run grid
starting fresh, so you have one clean JSON-per-combo dataset for the paper.

---

## End-to-end launch sequence (Option B — full Slurm)

### 0. Upload the package

From your Mac (one time):

```bash
# Pick your project root on Narval
ssh $USER@narval.alliancecan.ca "mkdir -p ~/projects/def-liyang/\$USER"
rsync -av --exclude venv --exclude logs --exclude __pycache__ --exclude '.ipynb_checkpoints' \
    narval_swat_run/ \
    $USER@narval.alliancecan.ca:~/projects/def-liyang/\$USER/narval_swat_run/
```

Then upload the SWaT CSVs to `$SCRATCH/swat_data/` (also one time):

```bash
ssh $USER@narval.alliancecan.ca "mkdir -p \$SCRATCH/swat_data"
rsync -av normal.csv attack.csv \
    $USER@narval.alliancecan.ca:$SCRATCH/swat_data/
```

### 1. Build the virtualenv

On a **login node** on Narval:

```bash
cd ~/projects/def-liyang/$USER/narval_swat_run
bash env/setup_venv.sh
```

This loads the cluster modules (StdEnv/2023, python/3.11, scipy-stack,
cuda/12.2), creates `venv/`, installs requirements from the Alliance
wheelhouse, and prints a versions summary. Idempotent — safe to re-run.

### 2. Pre-flight check

```bash
bash scripts/check_env.sh
```

Validates: venv, data files present, output dir writable, all imports work,
grid decoder self-consistent. If anything is wrong, it exits non-zero with
a clear message — fix before continuing.

### 3. Smoke test (one task, one A100 MIG slice, ~30 min wall)

```bash
sbatch slurm/00_smoke_test.sh
squeue -u $USER
# once it finishes:
tail logs/smoke_*.out
ls -la $SCRATCH/swat_paper_run/checkpoints/clean/
ls -la $SCRATCH/swat_paper_run/checkpoints/attacks/random_flip/
```

You should see `iforest__42.json` and four `iforest__42__r*.json` files.

### 4. Clean baselines (36 tasks)

```bash
sbatch slurm/10_clean_baselines.sh
```

Each task uses a full A100 for up to 2 h. The `%12` concurrency cap in the
sbatch header limits simultaneous GPUs to 12 — bump that if the queue is
friendly. Per-task JSONs appear in
`$SCRATCH/swat_paper_run/checkpoints/clean/` as they finish.

### 5. Attack grids (3 × 36 tasks, 4 rates each)

Submit all three in parallel:

```bash
sbatch slurm/20_attack_random_flip.sh
sbatch slurm/21_attack_targeted_flip.sh
sbatch slurm/22_attack_feature_noise.sh
```

Each attack task has a 4 h wallclock budget and loops over poison rates
internally, writing one JSON per rate.

### 5b. Phase 1.2 — HPO patch + AE/LSTM-AE rerun (R02 only)

The R02 paper run uses the post-HPO settings for AE (`AE_BATCH_SIZE=2048`,
`AE_ACTIVATION="leaky_relu"`) and LSTM-AE (`W=30`, hidden=256, epochs=30,
lr=1e-3, batch=256, optimizer=adamw). These are already encoded in the
post-patch `src/config.py` and `src/models.py`. The pre-patch versions are
preserved as `src/config.py.pre_hpo` and `src/models.py.pre_hpo` if you ever
need to roll back.

If your `$SWAT_OUTPUT_DIR/checkpoints/` already has AE/LSTM-AE JSONs from a
previous (pre-HPO) run, regenerate just those rows:

```bash
bash scripts/submit_ae_lstm_hpo_rerun.sh
```

This backs up the existing AE/LSTM-AE JSONs (task IDs 30–35) under
`checkpoints/.pre_hpo_backup/`, then submits clean baselines + the three
baseline-attack arrays for those task IDs only, chained via
`--dependency=afterok`. After completion, re-run aggregation (step 6) to
regenerate the HPO result tables and figures.

### 5c. Online / retraining poisoning (archival extension; not reported in the CASCON paper)

The R02 headline experiment is a 990-trajectory grid: **11 detectors ×
2 generators × 3 round-counts T × 3 per-round budgets Δp × 5 seeds**, with
each trajectory writing T+1 per-round JSONs.

```bash
# 1) Build the manifest (idempotent — re-run after editing CONFIG)
source venv/bin/activate
python scripts/make_online_manifest.py
ls -lh $SWAT_OUTPUT_DIR/online_manifest.tsv

# 2) Submit the array
sbatch slurm/40_online_retraining.sh
squeue -u $USER --format='%.18i %.30j %.4t %.10M %.6D %R'
```

Per-task output (T+1 JSONs each):

```
$SWAT_OUTPUT_DIR/checkpoints/online/
    <detector>__<generator>__T<T>__dp<dp>__<seed>__r<round>.json
$SWAT_OUTPUT_DIR/checkpoints/online/_rankings/
    <generator>__<seed>__n<pool_size>.npy
                                # cached attack-pool ranking, reused when the ranked pool matches
```

Resume semantics are stronger than the baseline grid: each round individually
checks for its own JSON and skips, so a mid-trajectory failure costs at most
one round. The `random_injection` generator costs nothing extra (seeded
permutation). The `high_loss` generator trains one clean AE per seed; the
ranking is cached so detectors with the same ranked attack pool reuse it for free.

Configuration knobs (edit `src/config.py`):

| Key | Default | Notes |
|---|---|---|
| `ONLINE_DETECTORS` | 11 (no SOD) | archival extension only; SOD is not part of the reported benchmark |
| `ONLINE_GENERATORS` | `["random_injection", "high_loss"]` | |
| `ONLINE_T_VALUES` | `[3, 5, 10]` | round counts |
| `ONLINE_DELTA_P_VALUES` | `[0.005, 0.01, 0.02]` | per-round budget fraction |
| `ONLINE_SEEDS` | `[42, 123, 456, 789, 1024]` | 5 seeds for trajectory-distribution reporting |

Re-run `python scripts/make_online_manifest.py` whenever any of these change.

### 6. Aggregate results

Once the arrays finish (or anytime mid-run to peek):

```bash
sbatch slurm/30_aggregate.sh        # queued — ~5 min
# or interactively from a login node:
source venv/bin/activate && python -m src.aggregate
```

Produces under `$SWAT_OUTPUT_DIR/`:

```
all_results.csv
table_T4_clean_baselines.csv
table_T5_poisoning_impact.csv
compute_cost.csv
multi_criteria_ranking.csv
checkpoints/clean_baselines.csv
checkpoints/attack_checkpoint.csv
figures/F3_robustness_curves.png
figures/F4_f1_degradation_heatmap.png
figures/F5_fnr_safety.png
figures/F6_seed_variance.png
figures/F7_per_model_comparison.png
figures/F8_precision_recall_tradeoff.png
run_summary.txt       ← which combos are still missing
```

### 7. LSTM-AE diagnostic

Open `notebook/SWaT_Narval_Interactive.ipynb` via VS Code Remote-SSH inside
an interactive GPU allocation and run Section 4:

```bash
salloc --account=def-liyang --gres=gpu:a100:1 --cpus-per-task=4 --mem=32G --time=1:00:00
source venv/bin/activate
# then run the notebook in VS Code
```

This writes `figures/lstm_ae_score_distribution.png` — the evidence that
LSTM-AE separation is real (AUC, AP, min-attack vs max-normal gap).

---

## Resume semantics

Every runner checks for its output JSON before doing any work.

- **Re-submitting an entire array** is safe: completed tasks finish in
  milliseconds (they skip the already-done combos).
- **A task crash mid-rate-loop** only loses that one rate; the other rates'
  JSONs are already on disk.
- **Partial attack combos** (e.g. task 7 finished rates 0.01 and 0.03 but
  not 0.05 and 0.10) resume cleanly on re-submission.

To re-run only the still-missing task IDs (more efficient than resubmitting
the whole array):

```bash
python scripts/missing_combos.py
bash $SWAT_OUTPUT_DIR/resubmit.sh
```

`missing_combos.py` writes a shell script under `$SWAT_OUTPUT_DIR/` that
submits each array with only the missing task IDs via `--array=i,j,k,...`.

---

## Key design choices

- **No Colab / Drive path logic.** Paths come from env vars; errors are
  loud if `SWAT_DATA_DIR` is unset or empty.
- **Checkpoint migration from `v16_*` / `final_paper_run` is DISABLED**
  (`MIGRATE_OLD_CHECKPOINTS = False` in `src/config.py`). The Narval run is
  the single source of truth for the paper.
- **Per-combo JSON checkpoints, not a shared CSV.** Array tasks can write
  concurrently with no races. `aggregate.py` reads them all and builds the
  paper CSVs.
- **Array task = (model, seed), not (model, seed, rate).** Keeps per-task
  overhead (preprocessing + splits) small relative to compute, and keeps
  array sizes sane for the Slurm scheduler.
- **Full A100 per array task.** Autoencoder and LSTM-AE benefit; PyOD tasks
  won't fully use the GPU but the allocation is cheap relative to the
  scientific cost of a failed run. If you want to optimize GPU-hours, split
  the arrays into CPU (PyOD) and GPU (torch) variants later.
- **Atomic writes everywhere** — `*.tmp` + `os.replace` — so a preempted
  job never leaves a truncated CSV or JSON behind.

---

## Troubleshooting

- **`SCRATCH unset`** — you're on a login node with a stripped env. Do
  `module load StdEnv/2023` first or `echo $SCRATCH` to confirm.
- **`ModuleNotFoundError: pyod`** — rerun `env/setup_venv.sh`; the online
  fallback handles pyod.
- **`torch.cuda.is_available()` is False inside a Slurm job** — did the
  sbatch header request `--gres=gpu:...`? Check `scontrol show job $JOBID`.
- **LSTM-AE runs out of memory** — drop `LSTM_AE_HIDDEN` to 64 in
  `src/config.py`, or add `--gres=gpu:a100:1` (full) instead of a MIG slice.
- **Tasks time out** — the attack arrays have 4 h. Heavy models (SOD, ABOD)
  on the largest seed can push this. Bump the `#SBATCH --time=` line.

---




