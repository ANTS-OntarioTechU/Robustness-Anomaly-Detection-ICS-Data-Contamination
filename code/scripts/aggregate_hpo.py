#!/usr/bin/env python3
"""
Aggregate per-task HPO JSONs from one stage into a single CSV.

Reads $SWAT_OUTPUT_DIR/hpo/results/{stage}/{task_id:04d}.json and writes
$SWAT_OUTPUT_DIR/hpo/results/{stage}.csv sorted by composite_f1 descending.

The CSV is consumed by scripts/make_hpo_manifests.py --top-from when
generating the next stage's manifest (requires columns composite_f1 and
config-as-JSON).

Handles two result shapes:
  * Stages 1-3   — results = {"clean": {...}, "poisoned": {...}}
  * Final stage  — results = {"clean": {...}, "random_flip__r0.01": {...},
                              ..., "feature_noise__r0.10": {...}}.
                   For final, "poisoned" columns anchor on targeted_flip__r0.10
                   to match the same composite metric used during staging.

Error-suffixed files ({task_id:04d}.error.json) are skipped and counted.

Usage:
    python scripts/aggregate_hpo.py --stage ae_stage1
    python scripts/aggregate_hpo.py --stage lstm_ae_final
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


STAGES = {
    "ae_stage1", "ae_stage2", "ae_stage3", "ae_final",
    "lstm_ae_stage1", "lstm_ae_stage2", "lstm_ae_stage3", "lstm_ae_final",
}


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, var ** 0.5


def _collapse_by_config(rows: list[dict]) -> list[dict]:
    """
    Group rows whose configs differ only by `seed` into a single summary row
    carrying mean/std of the key metrics. Use for Final-stage ranking so that
    high-variance configs are penalized correctly.
    """
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        cfg = json.loads(r["config"])
        # Drop seed to find sibling rows with the same config
        cfg_noseed = {k: v for k, v in cfg.items() if k != "seed"}
        key = json.dumps(cfg_noseed, sort_keys=True)
        buckets.setdefault(key, []).append(r)

    summaries = []
    for key, rs in buckets.items():
        clean_vals    = [r["clean_f1"]    for r in rs]
        pois_vals     = [r["poisoned_f1"] for r in rs]
        comp_vals     = [r["composite_f1"] for r in rs]
        delta_vals    = [r["delta_f1"]    for r in rs]
        clean_m, clean_s = _mean_std(clean_vals)
        pois_m,  pois_s  = _mean_std(pois_vals)
        comp_m,  comp_s  = _mean_std(comp_vals)
        delta_m, delta_s = _mean_std(delta_vals)
        seeds = sorted(json.loads(r["config"]).get("seed", 0) for r in rs)
        summaries.append({
            "n_seeds":            len(rs),
            "seeds":              ",".join(str(s) for s in seeds),
            "composite_f1_mean":  comp_m,
            "composite_f1_std":   comp_s,
            "clean_f1_mean":      clean_m,
            "clean_f1_std":       clean_s,
            "poisoned_f1_mean":   pois_m,
            "poisoned_f1_std":    pois_s,
            "delta_f1_mean":      delta_m,
            "delta_f1_std":       delta_s,
            "worst_poisoned_f1":  min(pois_vals) if pois_vals else 0.0,
            "task_ids":           ",".join(str(r["task_id"]) for r in rs),
            "config":             key,
        })
    return summaries


def _extract_blocks(record: dict) -> tuple[dict, dict]:
    """Returns (clean_block, poisoned_anchor_block) from a record, handling
    both stage-1/2/3 and final result shapes."""
    results = record.get("results", {}) or {}
    clean = results.get("clean", {}) or {}
    # Stage 1-3 puts a single "poisoned" block
    if "poisoned" in results:
        return clean, results["poisoned"] or {}
    # Final stage: anchor on targeted_flip @ 0.10 to keep composite comparable
    return clean, results.get("targeted_flip__r0.10", {}) or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Override $SWAT_OUTPUT_DIR (mostly for testing).")
    ap.add_argument("--multi-seed", action="store_true",
                    help=("Collapse tasks that share the same config (i.e. differ "
                          "only by `seed`) into a single row per distinct config. "
                          "Reports mean ± sd of clean_f1 / poisoned_f1 and ranks "
                          "by mean composite. USE THIS FOR FINAL STAGES — the "
                          "per-task composite_f1 is a single-seed estimate and "
                          "can mis-rank high-variance configs. Output CSV is "
                          "written to {stage}_by_config.csv."))
    args = ap.parse_args()

    out_root = args.output_dir or Path(os.environ.get(
        "SWAT_OUTPUT_DIR", f"{os.environ.get('SCRATCH','/tmp')}/swat_paper_run"))
    stage_dir = out_root / "hpo" / "results" / args.stage
    out_csv = out_root / "hpo" / "results" / f"{args.stage}.csv"

    if not stage_dir.exists():
        print(f"ERROR: {stage_dir} does not exist — did this stage run?",
              file=sys.stderr)
        return 2

    rows = []
    n_error = 0
    for jf in sorted(stage_dir.iterdir()):
        if jf.suffix != ".json":
            continue
        if jf.name.endswith(".error.json"):
            n_error += 1
            continue
        try:
            with open(jf) as f:
                rec = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  WARNING: {jf.name} is not valid JSON ({e}); skipping",
                  file=sys.stderr)
            continue

        clean_b, pois_b = _extract_blocks(rec)
        cfg = rec.get("config", {}) or {}

        rows.append({
            "task_id":       rec.get("task_id"),
            "composite_f1":  rec.get("composite_f1", 0.0),
            "delta_f1":      rec.get("delta_f1", 0.0),
            "clean_f1":      clean_b.get("f1", 0.0),
            "clean_fnr":     clean_b.get("fnr", 1.0),
            "poisoned_f1":   pois_b.get("f1", 0.0),
            "poisoned_fnr":  pois_b.get("fnr", 1.0),
            "train_time_s":  clean_b.get("train_time", 0.0),
            "n_params":      clean_b.get("n_params", 0),
            "wallclock_s":   rec.get("wallclock_s", 0.0),
            "config":        json.dumps(cfg, sort_keys=True),
        })

    if not rows:
        print(f"ERROR: no successful results in {stage_dir}", file=sys.stderr)
        return 3

    rows.sort(key=lambda r: (r["composite_f1"], r["clean_f1"]), reverse=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out_csv)

    if args.multi_seed:
        out_csv_ms = out_root / "hpo" / "results" / f"{args.stage}_by_config.csv"
        ms_rows = _collapse_by_config(rows)
        # Re-rank by composite mean (NOT per-task composite) — this is the
        # variance-aware ranking that a Final stage actually wants.
        ms_rows.sort(key=lambda r: r["composite_f1_mean"], reverse=True)
        tmp_ms = out_csv_ms.with_suffix(".csv.tmp")
        with open(tmp_ms, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ms_rows[0].keys()))
            w.writeheader()
            w.writerows(ms_rows)
        os.replace(tmp_ms, out_csv_ms)
        print(f"[aggregate {args.stage}] multi-seed: {len(ms_rows)} distinct configs → {out_csv_ms}")
        print(f"[aggregate {args.stage}] multi-seed top 3 by mean composite_f1:")
        for r in ms_rows[:3]:
            cfg = json.loads(r["config"])
            summary_keys = ["hidden_dims", "window", "hidden_dim", "num_layers",
                            "dropout", "activation", "use_batchnorm",
                            "optimizer", "lr", "batch_size", "epochs",
                            "loss_fn", "threshold_strategy", "scaler"]
            summary = {k: cfg[k] for k in summary_keys if k in cfg}
            print(f"  n_seeds={r['n_seeds']}  "
                  f"composite={r['composite_f1_mean']:.4f}±{r['composite_f1_std']:.4f}  "
                  f"clean={r['clean_f1_mean']:.4f}±{r['clean_f1_std']:.4f}  "
                  f"poisoned={r['poisoned_f1_mean']:.4f}±{r['poisoned_f1_std']:.4f}  "
                  f"{summary}")

    print(f"[aggregate {args.stage}] {len(rows)} successful, "
          f"{n_error} errored → {out_csv}")
    print(f"[aggregate {args.stage}] top 3 by composite_f1:")
    for r in rows[:3]:
        cfg = json.loads(r["config"])
        # Summarize the config for the log
        summary_keys = ["hidden_dims", "window", "hidden_dim", "num_layers",
                        "dropout", "activation", "use_batchnorm",
                        "optimizer", "lr", "batch_size", "epochs",
                        "loss_fn", "threshold_strategy", "scaler"]
        summary = {k: cfg[k] for k in summary_keys if k in cfg}
        print(f"  task={r['task_id']:>4}  composite={r['composite_f1']:.4f}  "
              f"clean={r['clean_f1']:.4f}  poisoned={r['poisoned_f1']:.4f}  "
              f"{summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
