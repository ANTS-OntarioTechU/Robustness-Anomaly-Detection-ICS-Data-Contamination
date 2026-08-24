#!/usr/bin/env python3
"""
Aggregate the 42 σ-HPO per-task JSONs into a single CSV that matches the R01
σ-sweep schema so both datasets can be overlaid in the HPO-updated notebook.

Reads  : $SWAT_OUTPUT_DIR/hpo/results/sigma_hpo/{task_id:04d}.json
Writes : $SWAT_OUTPUT_DIR/hpo/results/sigma_sweep_hpo_all_results.csv
         $SWAT_OUTPUT_DIR/hpo/results/sigma_sweep_hpo_summary.csv

Schema columns match sigma_sweep_all_results.csv so you can `pd.concat` the
two frames. In addition we emit a `source` column = "hpo" here vs "r01" in
the original — handy for filtering in plots.

Usage:
    python scripts/aggregate_sigma_hpo.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev


_COLUMNS = [
    "accuracy", "precision", "recall", "f1",
    "tp", "tn", "fp", "fn", "fnr", "fpr",
    "roc_auc", "pr_auc",
    "model", "seed", "attack", "poison_rate",
    "time", "split_type", "effective_contamination", "n_injected",
    "threshold", "n_features", "noise_sigma",
    "delta_f1", "clean_f1", "source",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path,
                    default=Path(os.environ.get("SWAT_OUTPUT_DIR")
                                 or f"{os.environ['SCRATCH']}/swat_paper_run"))
    args = ap.parse_args()

    sigma_dir = args.output_dir / "hpo" / "results" / "sigma_hpo"
    out_csv   = args.output_dir / "hpo" / "results" / "sigma_sweep_hpo_all_results.csv"
    sum_csv   = args.output_dir / "hpo" / "results" / "sigma_sweep_hpo_summary.csv"

    if not sigma_dir.exists():
        print(f"ERROR: {sigma_dir} not found — did the Slurm array run?",
              file=sys.stderr)
        return 2

    rows = []
    err_count = 0
    for jf in sorted(sigma_dir.glob("[0-9]*.json")):
        if jf.name.endswith(".error.json"):
            err_count += 1
            continue
        rec = json.loads(jf.read_text())
        cfg = rec["config"]
        pois = rec["results"]["poisoned"]
        clean = rec["results"]["clean"]
        # "model" column matches the R01 schema (autoencoder / lstm_ae)
        model_name = "autoencoder" if cfg["detector"] == "ae" else cfg["detector"]
        split_type = "random" if cfg["detector"] == "ae" else "contiguous_normal"
        rows.append({
            "accuracy":               pois.get("accuracy", 0.0),
            "precision":              pois.get("precision", 0.0),
            "recall":                 pois.get("recall", 0.0),
            "f1":                     pois.get("f1", 0.0),
            "tp":                     pois.get("tp", 0),
            "tn":                     pois.get("tn", 0),
            "fp":                     pois.get("fp", 0),
            "fn":                     pois.get("fn", 0),
            "fnr":                    pois.get("fnr", 1.0),
            "fpr":                    pois.get("fpr", 0.0),
            "roc_auc":                pois.get("roc_auc", 0.5),
            "pr_auc":                 pois.get("pr_auc", 0.0),
            "model":                  model_name,
            "seed":                   cfg["seed"],
            "attack":                 "feature_noise",
            "poison_rate":            pois.get("poison_rate", cfg.get("poison_rate")),
            "time":                   pois.get("train_time", 0.0),
            "split_type":             split_type,
            "effective_contamination":pois.get("effective_contamination", 0.0),
            "n_injected":             pois.get("n_injected", 0),
            "threshold":              pois.get("threshold"),
            "n_features":             rec.get("input_dim", 44),
            "noise_sigma":            pois.get("noise_sigma", cfg.get("noise_sigma")),
            "delta_f1":               rec.get("delta_f1", 0.0),
            "clean_f1":               clean.get("f1", 0.0),
            "source":                 "hpo",
        })

    if not rows:
        print(f"ERROR: no valid JSONs in {sigma_dir} (errored: {err_count})",
              file=sys.stderr)
        return 3

    # Write detail CSV
    tmp = out_csv.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out_csv)
    print(f"[aggregate sigma_hpo] {len(rows)} rows, {err_count} errored → {out_csv}")

    # Write summary (per model × σ; matches the R01 `sigma_sweep_summary.csv`)
    summary_rows = []
    by_key: dict[tuple, list[float]] = {}
    for r in rows:
        k = (r["model"], r["noise_sigma"])
        by_key.setdefault(k, []).append(r["f1"])
    for (model_name, sigma), f1s in sorted(by_key.items()):
        summary_rows.append({
            "model":       model_name,
            "noise_sigma": sigma,
            "f1_mean":     mean(f1s),
            "f1_std":      stdev(f1s) if len(f1s) > 1 else 0.0,
            "n_seeds":     len(f1s),
            "source":      "hpo",
        })
    tmp_s = sum_csv.with_suffix(".csv.tmp")
    with open(tmp_s, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    os.replace(tmp_s, sum_csv)
    print(f"[aggregate sigma_hpo] summary → {sum_csv}")

    # Pretty-print the summary table
    print("\n=== HPO-winner σ-sweep summary ===")
    print(f"{'model':<14} {'σ':>6}  {'rate':>8}   F1 mean ± sd   n")
    # Re-group including rate for an at-a-glance view
    by_detail: dict[tuple, list[float]] = {}
    for r in rows:
        by_detail.setdefault(
            (r["model"], r["noise_sigma"], r["poison_rate"]), []).append(r["f1"])
    for (model_name, sigma, rate), f1s in sorted(by_detail.items()):
        m = mean(f1s)
        s = stdev(f1s) if len(f1s) > 1 else 0.0
        print(f"{model_name:<14} {sigma:>6.2f}  {rate:>8.2f}   "
              f"{m:>5.3f} ± {s:>5.3f}   {len(f1s)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
