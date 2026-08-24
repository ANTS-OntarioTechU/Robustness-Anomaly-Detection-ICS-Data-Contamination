#!/usr/bin/env python3
"""Regenerate the revised Figures 2 and 3 from the released result table.

The script uses only ``tables/final_hpo/all_results_hpo.csv``. It does not
require, read, or distribute the SWaT dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_ORDER = [
    "pca",
    "svm",
    "histogram",
    "iforest",
    "mcd",
    "autoencoder",
    "cluster",
    "knn",
    "abod",
    "lof",
    "lstm_ae",
]
MODEL_LABELS = {
    "pca": "PCA",
    "svm": "SVM",
    "histogram": "HISTOGRAM",
    "iforest": "IFOREST",
    "mcd": "MCD",
    "autoencoder": "AUTOENCODER",
    "cluster": "CLUSTER",
    "knn": "KNN",
    "abod": "ABOD",
    "lof": "LOF",
    "lstm_ae": "LSTM-AE",
}
ATTACKS = [
    ("feature_noise", "Feature Noise Injection"),
    ("random_flip", "Random Injection"),
    ("targeted_flip", "Similarity-Targeted Injection"),
]
COLORS = list(plt.get_cmap("tab20").colors)
COLOR_MAP = {model: COLORS[index] for index, model in enumerate(MODEL_ORDER)}


def make_plot(
    summary: pd.DataFrame,
    metric: str,
    figure_title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Write one paper figure from seed-averaged metric values."""
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.9), sharex=True)
    handles = []

    for axis, (attack, panel_title) in zip(axes, ATTACKS):
        for model in MODEL_ORDER:
            clean = summary[(summary.model == model) & (summary.attack == "none")]
            attacked = summary[
                (summary.model == model) & (summary.attack == attack)
            ]
            curve = pd.concat([clean, attacked]).sort_values("poison_rate")
            (line,) = axis.plot(
                curve["poison_rate"],
                curve[metric],
                marker="o",
                markersize=3.2,
                linewidth=1.4,
                color=COLOR_MAP[model],
                label=MODEL_LABELS[model],
            )
            if axis is axes[0]:
                handles.append(line)

        axis.set_title(panel_title, fontsize=13, fontweight="semibold", pad=3)
        axis.set_ylim(-0.03, 1.03)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=10, length=3)

    axes[-1].set_xlabel("Contamination Rate", fontsize=11)
    axes[-1].set_xticks([0, 0.01, 0.03, 0.05, 0.10])
    axes[-1].set_xticklabels(["0%", "1%", "3%", "5%", "10%"])
    fig.legend(
        handles,
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        fontsize=9.3,
        columnspacing=1.4,
        handlelength=1.8,
        handletextpad=0.45,
    )
    fig.supylabel(ylabel, x=0.015, fontsize=11)
    fig.suptitle(figure_title, fontsize=16, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.20, hspace=0.27)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=REPOSITORY_ROOT / "tables" / "final_hpo" / "all_results_hpo.csv",
        help="CSV containing the final HPO results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "figures",
        help="Directory for Fig2_f1_vs_contamination.png and Fig3_fnr_vs_contamination.png.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.results)
    required_columns = {"model", "attack", "poison_rate", "f1", "fnr"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Results table is missing: {sorted(missing_columns)}")

    frame = frame[frame["model"].isin(MODEL_ORDER)].copy()
    if set(frame["model"].unique()) != set(MODEL_ORDER):
        raise ValueError("Results table does not contain all 11 reported detectors.")
    summary = frame.groupby(["model", "attack", "poison_rate"], as_index=False)[
        ["f1", "fnr"]
    ].mean()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_plot(
        summary,
        "f1",
        "F1 vs Contamination Rate",
        "F1 Score",
        args.output_dir / "Fig2_f1_vs_contamination.png",
    )
    make_plot(
        summary,
        "fnr",
        "False-Negative Rate vs Contamination Rate",
        "False-Negative Rate",
        args.output_dir / "Fig3_fnr_vs_contamination.png",
    )


if __name__ == "__main__":
    main()
