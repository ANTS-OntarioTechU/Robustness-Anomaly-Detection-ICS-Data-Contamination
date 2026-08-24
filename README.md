# Robustness of Anomaly Detection Models under Training-Time Data Contamination

Reproducibility package for:

> **Robustness of Anomaly Detection Models for Industrial Control Systems
> under Training-Time Data Contamination**

Accepted at **CASCON 2026**. The final author-version paper and arXiv link will
be added after release.

> **Data notice.** The SWaT dataset is access-controlled and is not included.
> See [DATA.md](DATA.md) before running or redistributing this package.

The package was curated from the original experiment working trees. Figures 2
and 3 were regenerated from the released post-HPO results to match the revised
camera-ready manuscript; all other scientific artifacts retain their original
provenance.

---

## Where this came from

Experiments were run on the **Narval** cluster of the Digital Research Alliance
of Canada. Code was developed locally and run through Slurm arrays; results
were consolidated after the post-HPO experiment.

| This folder | Original location |
|---|---|
| `code/` | `swat-results/narval_swat_run/` (canonical tree, Apr 26–28) |
| `tables/final_hpo/` | `swat-results/SWAT-results-R01/` |
| `tables/pre_hpo/` | `swat-results/swat_paper_run/` |
| `figures/` | see the figure table below |

Excluded deliberately: virtual environments, `__pycache__/`, Slurm logs,
internal working notes, and the raw SWaT files. The anonymous manuscript PDF is
also deliberately excluded from the public release; replace it with an approved
final author-version PDF only after the CASCON/arXiv release.

---

## Layout

```
code-adv-paper/
├── code/           experiment engine + Slurm orchestration
├── tables/         all result CSVs
├── figures/        the 4 figures that appear in the paper
├── DATA.md         SWaT data-use and redistribution notice
└── LICENSE         MIT licence for repository-owned materials
```

### `code/`

```
src/         config.py, data.py, models.py, attacks.py, eval_utils.py,
             run_clean.py, run_attack.py, run_online.py, run_sigma_sweep.py,
             aggregate.py, attacks_online.py, hpo/
             (+ config.py.pre_hpo / models.py.pre_hpo — pre-tuning snapshots)
slurm/       23 sbatch scripts — clean baselines, 3 contamination arrays,
             aggregation, online retraining, sigma sweep, 4-stage AE and
             LSTM-AE HPO with promotion gates
scripts/     HPO manifest builders, aggregators, missing-combo resubmitter
notebook/    5 visualization / aggregation notebooks (see below)
env/         setup_venv.sh + Alliance-wheelhouse requirements.txt
bin/
```

### How the code maps to the paper

| Paper | Code |
|---|---|
| Random injection | `random_flip` in `src/attacks.py` |
| Similarity-targeted injection | `targeted_flip` |
| Feature-noise injection | `feature_noise` (σ = 0.15) |
| Budgets 1 / 3 / 5 / 10 % | `POISON_RATES = [0.01, 0.03, 0.05, 0.10]` |
| 3 seeds | `SEEDS = [42, 123, 456]` |
| 11 reported detectors | `MODELS` in `src/config.py`, excluding SOD |
| Tuned PCA / SVM | `TUNED_PARAMS` — `n_components=0.90`, `nu=0.01` |
| Narval, StdEnv/2023, CUDA 12.2 | `slurm/_common.sh` |

The `MODELS` grid in `config.py` lists **12** detectors; the paper reports **11**.
The difference is **SOD**, which was run as a clean baseline and then excluded —
see "Run accounting" below.

---

## Figures

`figures/` contains the four paper figures. Figures 2 and 3 use the revised
camera-ready layout while preserving the reported data, panel titles, model
labels, attack definitions, and contamination rates.

| In paper | File | Reproduction path |
|---|---|---|
| Fig. 1 | `Fig1_methodology.png` | hand-drawn workflow diagram |
| Fig. 2 | `Fig2_f1_vs_contamination.png` | `python code/scripts/plot_paper_figures.py` |
| Fig. 3 | `Fig3_fnr_vs_contamination.png` | `python code/scripts/plot_paper_figures.py` |
| Fig. 4 | `Fig4_f1_change_heatmap.png` | post-HPO visualization notebook |

The figure script reads only the released
`tables/final_hpo/all_results_hpo.csv`; it does not require SWaT data. It
regenerates Figures 2 and 3 byte-for-byte from the released result table.

Note that `src/aggregate.py` emits a different, earlier figure set (`F3`–`F8`);
it did **not** produce the paper figures.

---

## Notebooks

Seven notebooks existed in the original tree; two were redundant older versions
of the results-visualization notebook and have been dropped. The five that
remain each do a different job:

| Notebook | Last saved | Purpose |
|---|---|---|
| `SWaT_Results_Visualization_HPO_Updated.ipynb` | 2026-04-20 | **Paper figures.** Post-HPO V-series incl. V10–V12 (σ-sweep, R01-vs-HPO) |
| `SWaT_R02_Aggregator.ipynb` | 2026-04-28 | Newest file; result aggregation + ΔFNR scatter |
| `SWaT_HPO_Visualization.ipynb` | 2026-04-19 | HPO-only diagnostics (H1–H9: stage funnel, knob sensitivity) |
| `SWaT_Narval_CellByCell.ipynb` | 2026-04-18 | Step-by-step pipeline diagnostics (F-series) |
| `SWaT_Narval_Interactive.ipynb` | 2026-04-17 | Interactive inspection / smoke checks |

Dropped: `SWaT_Results_Visualization.ipynb` and
`SWaT_Results_Visualization_Academic.ipynb` (both 2026-04-18). The HPO-updated
notebook covers 79 % of the Academic notebook's code and supersedes both — they
rendered the pre-HPO figure set, which the paper does not use. Both remain in
`swat-results/` .

---

## Run accounting

`tables/pre_hpo/run_summary.txt` records the grid exactly as the paper describes it:

- Attack grid: 432 combos (12 models × 3 seeds × 3 attacks × 4 rates)
- SOD contributed 36 missing attack combos → **396 retained contamination runs**
- Clean: 36 combos, 2 SOD seeds missing → 34 retained + 1 documented SOD baseline

`tables/final_hpo/all_results_hpo.csv` holds 430 data rows = 396 + 34. This
matches the paper's Section V accounting.

---

## Tables

**`tables/final_hpo/`** — post-HPO, these back the reported results:

| File | Contents |
|---|---|
| `all_results_hpo.csv` | every reported run plus one documented SOD clean baseline; full metrics |
| `table_T4_clean_baselines_hpo.csv` | clean F1 / recall / precision / FNR / time, including the documented SOD baseline |
| `table_T5_poisoning_impact_hpo.csv` | degradation per model × attack × budget |
| `multi_criteria_ranking_hpo.csv` | joint F1 / FNR / cost ranking |
| `compute_cost_hpo.csv` | training cost per detector |

**`tables/pre_hpo/`** — the earlier run, before AE/LSTM-AE tuning. Same schema,
plus `sigma_sweep_*.csv` (feature-noise σ sensitivity) and `run_summary.txt`.

**`tables/checkpoints/`** — raw per-combo checkpoint CSVs the aggregator consumed.

**Standard-deviation note.** The archived clean-baseline table uses the sample
standard deviation across seeds. The archived contamination-impact table uses
the population standard deviation across its three seeds. These conventions are
preserved from the original result artifacts; the means and run counts are the
primary reported quantities and reproduce exactly from `all_results_hpo.csv`.

---

## Reproducing

To regenerate the two revised paper figures from the released result table only:

```bash
python code/scripts/plot_paper_figures.py
```

To rerun the full experiment on an Alliance/Narval-style environment:

```bash
bash code/env/setup_venv.sh
export SWAT_PROJECT_DIR=$PWD/code SWAT_DATA_DIR=... SWAT_OUTPUT_DIR=...
sbatch code/slurm/10_clean_baselines.sh
```

`SWAT_DATA_DIR` must contain the SWaT `normal.csv` and `attack.csv`. The raw
SWaT dataset is **not** included here — it is access-controlled and must be
requested from iTrust, SUTD. See [DATA.md](DATA.md).

---

## Citation

If you use this code or build on these results, please cite the paper:

> M. U. Ozbek, T. Ojo, P. Madani, K. El-Khatib, and L. Yang, "Robustness of
> Anomaly Detection Models for Industrial Control Systems under Training-Time
> Data Contamination," *CASCON 2026*, accepted.

```bibtex
@inproceedings{ozbek2026robustness,
  title     = {Robustness of Anomaly Detection Models for Industrial Control
               Systems under Training-Time Data Contamination},
  author    = {Ozbek, Mustafa Umut and
               Ojo, Taiwo and
               Madani, Pooria and
               El-Khatib, Khalil and
               Yang, Li},
  booktitle = {CASCON 2026},
  year      = {2026},
  note      = {Accepted; proceedings details forthcoming}
}
```

All authors are with the Faculty of Business and IT, Ontario Tech University,
Oshawa, Ontario, Canada.

Replace the citation fields with the final proceedings pages and DOI when they
become available.

**Please also credit the dataset.** The iTrust terms require explicit credit to
"iTrust, Centre for Research in Cyber Security, Singapore University of
Technology and Design" in any published work that uses SWaT, and require that
iTrust be notified once such work is published. See [DATA.md](DATA.md).
