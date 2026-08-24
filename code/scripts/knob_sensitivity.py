#!/usr/bin/env python3
"""
Per-knob sensitivity analysis on an HPO Stage-1 CSV.

For each grid dimension, holds every OTHER knob at its baseline (deployed-
config value) and reports ΔF1 averaged over the remaining-free axes, then
applies the same Phase-1 promotion rule used for PCA and SVM:
    promote a tuned value if  ΔF1 ≥ 0.03  OR  ΔFNR ≤ −0.05 vs default.

The output is Markdown-ready — drop it into §4.2 or Table IV of the paper.

Usage:
    python scripts/knob_sensitivity.py --detector ae
    python scripts/knob_sensitivity.py --detector lstm_ae --metric poisoned_f1

Run AFTER stage 1 aggregation has produced the CSVs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev


# ─── Default (deployed-config) baseline for each detector ─────────────────
# These are the values in src/config.py at the time the HPO was designed.
# Any knob that isn't in the grid is omitted; others use the HPO-default row.
AE_BASELINE = {
    "hidden_dims":   "[256, 128, 64]",
    "dropout":       0.1,
    "activation":    "relu",
    "use_batchnorm": True,      # hardcoded True in models.py
    "optimizer":     "adam",
    "lr":            5e-4,
    "batch_size":    1024,
    "epochs":        100,
    "loss_fn":       "mse",
    "threshold_strategy": "f1_optimal",
    "scaler":        "minmax",
}

LSTM_AE_BASELINE = {
    "window":        20,
    "hidden_dim":    128,
    "num_layers":    1,
    "dropout":       0.2,
    "optimizer":     "adam",
    "lr":            5e-4,
    "batch_size":    512,
    "epochs":        50,
    "loss_fn":       "mse",
    "threshold_strategy": "f1_optimal",
    "scaler":        "minmax",
}

# Knobs that Stage 1 actually sweeps — per detector
AE_S1_KNOBS      = ["hidden_dims", "dropout", "activation", "use_batchnorm"]
LSTM_AE_S1_KNOBS = ["window", "hidden_dim", "num_layers", "dropout"]

PROMOTE_DF1  =  0.03   # ΔF1 ≥  0.03  →  promote
PROMOTE_DFNR = -0.05   # ΔFNR ≤ -0.05 →  promote


def _load_stage1(output_dir: Path, detector: str) -> list[dict]:
    csv_path = output_dir / "hpo" / "results" / f"{detector}_stage1.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found — run aggregate_hpo.py --stage {detector}_stage1 first", file=sys.stderr)
        sys.exit(2)
    import csv as _csv
    rows = []
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for r in reader:
            cfg = json.loads(r["config"])
            rows.append({
                "config":       cfg,
                "clean_f1":     float(r["clean_f1"]),
                "clean_fnr":    float(r["clean_fnr"]),
                "poisoned_f1":  float(r["poisoned_f1"]),
                "poisoned_fnr": float(r["poisoned_fnr"]),
                "composite_f1": float(r["composite_f1"]),
                "delta_f1":     float(r["delta_f1"]),
            })
    return rows


def _sweep_knob(rows: list[dict], knob: str, baseline: dict,
                metric: str) -> list[tuple]:
    """
    For each value of `knob` appearing in Stage 1, collect all rows where every
    OTHER grid-knob matches the baseline, then report mean(metric) for that
    value. Returns a list of (value, mean_metric, std_metric, n_rows).
    """
    # Determine which knobs are "other" — those in the baseline dict but not `knob`
    # (and only those that actually appear in the rows' configs).
    # We take the intersection with the first row's keys to be safe.
    sample_cfg = rows[0]["config"]
    other_knobs = [k for k in baseline
                   if k in sample_cfg and k != knob]

    groups: dict = {}
    for r in rows:
        cfg = r["config"]
        # Stage 1 sweeps only a subset of knobs; non-swept knobs share the
        # baseline. Restrict "other knob must match baseline" to keys that
        # could possibly differ in Stage 1.
        if not all(_cfg_eq(cfg.get(ok), baseline.get(ok)) for ok in other_knobs):
            continue
        val = _stringify(cfg.get(knob))
        groups.setdefault(val, []).append(r[metric])

    out = []
    for v, vals in groups.items():
        m = mean(vals)
        s = stdev(vals) if len(vals) > 1 else 0.0
        out.append((v, m, s, len(vals)))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _stringify(v):
    if isinstance(v, list):
        return str(v)
    return v


def _cfg_eq(a, b) -> bool:
    """Loose equality that tolerates type mismatches (list↔string etc.)."""
    if a is None and b is None:
        return True
    if isinstance(a, list) or isinstance(b, list):
        return _stringify(a) == _stringify(b)
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).lower() == str(b).lower()


def _report(detector: str, knobs: list[str], baseline: dict,
            rows: list[dict], metric: str):
    """Print a Markdown sensitivity table."""
    print(f"\n### {detector.upper()} — per-knob sensitivity "
          f"(metric: {metric}, Phase-1 style)\n")
    print(f"Baseline for 'held at default' slice: "
          f"`{{ {', '.join(f'{k}={v!r}' for k,v in baseline.items() if k in knobs)} }}`\n")
    print("| Knob | Value | Mean | SD | n | ΔF1 vs default | Promote? |")
    print("|---|---|---|---|---|---|---|")

    for k in knobs:
        sweep = _sweep_knob(rows, k, baseline, metric)
        if not sweep:
            print(f"| {k} | (no rows match baseline slice) | | | | | |")
            continue
        # Find the default value's metric
        default_val = _stringify(baseline.get(k))
        default_row = next((s for s in sweep if _cfg_eq(s[0], default_val)), None)
        default_metric = default_row[1] if default_row else None

        for i, (v, m, s, n) in enumerate(sweep):
            if default_metric is None:
                df1 = None
            else:
                df1 = m - default_metric
            promote = ""
            if df1 is not None and df1 >= PROMOTE_DF1:
                promote = "**yes**"
            if default_val is not None and _cfg_eq(v, default_val):
                promote = "_(default)_"
            df1_str = f"{df1:+.4f}" if df1 is not None else "—"
            m_str   = f"{m:.4f}"
            s_str   = f"{s:.4f}"
            print(f"| {k if i == 0 else ''} | `{v}` | {m_str} | {s_str} | {n} | {df1_str} | {promote} |")
    print()
    print(f"Promotion rule: promote a tuned value if **ΔF1 ≥ {PROMOTE_DF1}** "
          f"OR **ΔFNR ≤ {PROMOTE_DFNR}** vs. the default. Same threshold as "
          f"the PCA and SVM Phase-1 sweeps.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=["ae", "lstm_ae", "both"],
                    default="both")
    ap.add_argument("--metric",
                    choices=["clean_f1", "poisoned_f1", "composite_f1"],
                    default="composite_f1",
                    help="Which metric to rank against (default: composite_f1, "
                         "i.e. mean of clean + poisoned).")
    ap.add_argument("--output-dir", type=Path,
                    default=Path(os.environ.get("SWAT_OUTPUT_DIR")
                                or f"{os.environ['SCRATCH']}/swat_paper_run"))
    ap.add_argument("--markdown-out", type=Path, default=None,
                    help="Write Markdown tables to this file instead of stdout.")
    args = ap.parse_args()

    output_lines = []
    original_print = print

    def maybe_capture(*a, **kw):
        if args.markdown_out is not None:
            output_lines.append(" ".join(str(x) for x in a))
        else:
            original_print(*a, **kw)

    import builtins
    if args.markdown_out is not None:
        builtins.print = maybe_capture

    detectors = ["ae", "lstm_ae"] if args.detector == "both" else [args.detector]
    for d in detectors:
        rows = _load_stage1(args.output_dir, d)
        knobs = AE_S1_KNOBS if d == "ae" else LSTM_AE_S1_KNOBS
        baseline = AE_BASELINE if d == "ae" else LSTM_AE_BASELINE
        _report(d, knobs, baseline, rows, args.metric)

    if args.markdown_out is not None:
        builtins.print = original_print
        args.markdown_out.write_text("\n".join(output_lines))
        print(f"Wrote Markdown tables to {args.markdown_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
