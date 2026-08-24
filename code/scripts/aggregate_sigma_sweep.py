#!/usr/bin/env python3
"""
Aggregate the feature-noise σ sweep across multiple σ values into a single
CSV and a "F1 vs σ" figure.

Also pulls in the σ = 0.15 result from the main grid's `feature_noise` folder
(at the matching rate, default 0.10) so the sweep figure is rooted at the
original baseline.

Usage:
    python scripts/aggregate_sigma_sweep.py
    python scripts/aggregate_sigma_sweep.py --rate 0.10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CONFIG, get_output_dir  # noqa: E402

EXCLUDE_MODELS = {"sod"}
FAMILY = {
    "iforest":    "Tree",
    "svm":        "Kernel",
    "lof":        "Distance/Density",
    "cluster":    "Distance/Density",
    "knn":        "Distance/Density",
    "abod":       "Distance/Density",
    "histogram":  "Statistical",
    "pca":        "Statistical",
    "mcd":        "Statistical",
    "autoencoder":"Neural",
    "lstm_ae":    "Neural",
}
FAMILY_COLORS = {
    "Tree":             "#4C72B0",
    "Kernel":           "#DD8452",
    "Distance/Density": "#C44E52",
    "Statistical":      "#55A868",
    "Neural":           "#8172B2",
}


def _rate_tag(r: float) -> str:
    return "r" + f"{r:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def load_sweep_rows(out_dir: Path, rate: float) -> pd.DataFrame:
    """Load all σ-sweep JSONs PLUS the matching σ = 0.15 rows from the main grid."""
    rows = []

    # 1. σ = 0.15 from the main grid
    main = out_dir / "checkpoints" / "attacks" / "feature_noise"
    if main.exists():
        tag = _rate_tag(rate)
        for fp in main.glob(f"*__*__{tag}.json"):
            try:
                d = json.loads(fp.read_text())
                if d.get("model") in EXCLUDE_MODELS:
                    continue
                d["noise_sigma"] = float(d.get("noise_sigma") or CONFIG["NOISE_SIGMA"])
                rows.append(d)
            except Exception as e:
                print(f"WARN: could not parse {fp}: {e}")

    # 2. Every σ subdirectory
    sweep = out_dir / "checkpoints" / "sigma_sweep"
    if sweep.exists():
        for sigma_dir in sorted(sweep.iterdir()):
            if not sigma_dir.is_dir():
                continue
            for fp in sigma_dir.glob("*.json"):
                if fp.name.endswith(".error.json"):
                    continue
                try:
                    d = json.loads(fp.read_text())
                    if d.get("model") in EXCLUDE_MODELS:
                        continue
                    # Only include rows at the requested rate
                    file_rate = float(d.get("poison_rate", 0.0))
                    if not np.isclose(file_rate, rate):
                        continue
                    rows.append(d)
                except Exception as e:
                    print(f"WARN: could not parse {fp}: {e}")

    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """F1 mean/std per (model, σ)."""
    g = df.groupby(["model", "noise_sigma"], as_index=False)
    return g["f1"].agg(["mean", "std", "count"]).reset_index().rename(
        columns={"mean": "f1_mean", "std": "f1_std", "count": "n_seeds"}
    )


def plot_sweep(summary: pd.DataFrame, out_png: Path) -> None:
    fig, (ax_f1, ax_df1) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    models = sorted(summary["model"].unique(),
                    key=lambda m: summary[summary["model"] == m]["f1_mean"].mean(),
                    reverse=True)

    # Family-coloured lines + per-model marker
    markers = ["o", "s", "^", "D", "v", "P", "X", ">", "<", "p", "h"]
    for i, m in enumerate(models):
        sub = summary[summary["model"] == m].sort_values("noise_sigma")
        color = FAMILY_COLORS[FAMILY[m]]
        ax_f1.errorbar(sub["noise_sigma"], sub["f1_mean"],
                       yerr=sub["f1_std"].fillna(0), color=color,
                       marker=markers[i % len(markers)], label=m, markersize=5, linewidth=1.3,
                       capsize=3)
        # ΔF1 relative to σ = 0.15 (baseline)
        base = sub[np.isclose(sub["noise_sigma"], 0.15)]
        if len(base):
            base_f1 = base["f1_mean"].iloc[0]
            df1 = sub["f1_mean"] - base_f1
            ax_df1.plot(sub["noise_sigma"], df1, "-", color=color,
                        marker=markers[i % len(markers)], label=m, markersize=5, linewidth=1.3)

    for ax in (ax_f1, ax_df1):
        ax.set_xlabel("Feature-noise σ")
        ax.grid(alpha=0.3)

    ax_f1.set_ylabel("F1 (mean ± σ over seeds)")
    ax_f1.set_title("F1 vs feature-noise σ  (poison rate fixed)")
    ax_f1.set_ylim(0, 1.0)
    ax_df1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_df1.set_ylabel("ΔF1 from σ = 0.15")
    ax_df1.set_title("ΔF1 vs σ  (zero line = no additional damage)")
    ax_df1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)

    plt.suptitle("Feature-noise σ sweep — is 0.15 genuinely null, or only null at that magnitude?",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.10,
                    help="Poisoning rate at which to compare σ values (default 0.10).")
    args = ap.parse_args()

    out_dir = get_output_dir()
    df = load_sweep_rows(out_dir, args.rate)
    print(f"Loaded {len(df)} rows across σ in {sorted(df['noise_sigma'].unique())}")
    if len(df) == 0:
        print("No data found. Did you submit the sigma-sweep arrays?")
        return 1

    df_out = out_dir / "sigma_sweep_all_results.csv"
    df.to_csv(df_out, index=False)
    print(f"wrote {df_out}")

    summary = build_summary(df)
    sum_out = out_dir / "sigma_sweep_summary.csv"
    summary.to_csv(sum_out, index=False)
    print(f"wrote {sum_out}")

    fig_dir = out_dir / "figures_v2"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_sweep(summary, fig_dir / "V10_sigma_sweep.png")

    # Pretty print the summary
    print("\nSummary table (F1 mean over seeds, per model × σ):")
    pivot = summary.pivot(index="model", columns="noise_sigma", values="f1_mean")
    print(pivot.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
