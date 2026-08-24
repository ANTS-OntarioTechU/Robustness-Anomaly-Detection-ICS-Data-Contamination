#!/usr/bin/env python3
"""
Aggregate Phase 1.3 online-retraining JSONs into the paper's R02 tables and figures.

Reads all per-round JSONs from $SWAT_OUTPUT_DIR/checkpoints/online/ and produces:

    $SWAT_OUTPUT_DIR/online_aggregated/
        online_results.csv               long-format DataFrame (one row per round)
        Table_X_per_seed_minima.csv      55 rows = 11 detectors × 5 seeds, worst trajectory per (det, seed)
        Table_XI_per_config_summary.csv  18 rows = 2 generators × 3 T × 3 Δp
        Table_XII_gradual_vs_oneshot.csv comparison at matched cumulative budgets
        figures/Fig10_per_seed_trajectories.png
        figures/Fig11_dF1_dFNR_scatter.png
        figures/Fig12_gradual_vs_oneshot.png

Also reads $SWAT_OUTPUT_DIR/checkpoints/attacks/random_flip/ for Table XII (one-shot baseline).

Usage:

    cd ~/projects/def-liyang/$USER/narval_swat_run
    source venv/bin/activate
    python scripts/aggregate_online.py

Idempotent — safe to re-run after partial JSON updates.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.config import CONFIG, get_output_dir, online_combos  # noqa: E402

OUT_BASE = get_output_dir()
ONLINE_DIR = OUT_BASE / "checkpoints" / "online"
ATTACK_DIR = OUT_BASE / "checkpoints" / "attacks"
AGG_DIR = OUT_BASE / "online_aggregated"
FIG_DIR = AGG_DIR / "figures"
AGG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────── detector family map ─────────────────────
DETECTOR_FAMILY = {
    "iforest":     "tree",
    "svm":         "kernel",
    "lof":         "local_density",
    "cluster":     "cluster_density",
    "knn":         "distance",
    "histogram":   "statistical",
    "pca":         "subspace",
    "mcd":         "robust_covariance",
    "abod":        "geometric",
    "autoencoder": "neural_pointwise",
    "lstm_ae":     "neural_sequence",
}


# ────────────────────────── load JSONs ──────────────────────────────
def _safe_load(fp: Path) -> dict | None:
    try:
        with open(fp) as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] failed to read {fp.name}: {e}", file=sys.stderr)
        return None


def load_online_long() -> pd.DataFrame:
    """Return one row per (detector, generator, T, dp, seed, round) JSON."""
    print(f"[load] scanning {ONLINE_DIR} ...")
    files = [
        f for f in os.listdir(ONLINE_DIR)
        if f.endswith(".json") and not f.endswith(".error.json")
    ]
    print(f"[load] found {len(files)} round JSONs.")
    rows = []
    for f in files:
        d = _safe_load(ONLINE_DIR / f)
        if d is None:
            continue
        rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No online JSONs loaded — is the path correct?")

    # Normalize column types
    for col in ["round", "T", "seed", "n_in_pool", "n_in_train",
                "n_poison_in_train", "total_injected", "unique_attacks_used",
                "k_per_round", "original_pool_size",
                "tp", "tn", "fp", "fn"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["delta_p", "cumulative_p", "poison_rate", "f1", "fnr",
                "precision", "recall", "threshold", "time", "roc_auc", "pr_auc",
                "effective_in_subsample_p"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["family"] = df["model"].map(DETECTOR_FAMILY)
    df = df.sort_values(
        ["model", "generator", "T", "delta_p", "seed", "round"]
    ).reset_index(drop=True)
    print(f"[load] long-format DataFrame shape = {df.shape}")
    return df


# ────────────────────────── Table X ──────────────────────────────────
def build_table_x(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(detector, seed) trajectory minimum.

    For each (detector, seed), find the trajectory (across all 18 generator/T/Δp
    configs) with the lowest min-F1, and report:
        clean_f1 (round 0 of that worst trajectory),
        min_f1, delta_f1, peak_round, peak_cumulative_p,
        fnr_at_peak, delta_fnr_at_peak,
        worst_generator, worst_T, worst_dp.
    Result: 11 detectors × 5 seeds = 55 rows.
    """
    print("[Table X] computing per-(detector, seed) trajectory minima ...")
    rows = []
    for (model, seed), g in long_df.groupby(["model", "seed"]):
        # group again by trajectory to find each trajectory's min and clean baseline
        traj_summaries = []
        for (gen, T, dp), tg in g.groupby(["generator", "T", "delta_p"]):
            tg = tg.sort_values("round")
            if tg.empty:
                continue
            clean_row = tg[tg["round"] == 0]
            if clean_row.empty:
                continue
            clean_f1 = float(clean_row["f1"].iloc[0])
            clean_fnr = float(clean_row["fnr"].iloc[0])
            non_clean = tg[tg["round"] > 0]
            if non_clean.empty:
                continue
            peak_idx = non_clean["f1"].idxmin()
            peak = non_clean.loc[peak_idx]
            traj_summaries.append({
                "model": model, "seed": int(seed),
                "generator": gen, "T": int(T), "delta_p": float(dp),
                "clean_f1": clean_f1,
                "clean_fnr": clean_fnr,
                "min_f1": float(peak["f1"]),
                "delta_f1": float(peak["f1"]) - clean_f1,
                "peak_round": int(peak["round"]),
                "peak_cumulative_p": float(peak["cumulative_p"]),
                "fnr_at_peak": float(peak["fnr"]),
                "delta_fnr_at_peak": float(peak["fnr"]) - clean_fnr,
            })
        if not traj_summaries:
            continue
        # Pick the most-damaging trajectory (smallest min_f1)
        worst = min(traj_summaries, key=lambda r: r["min_f1"])
        worst_renamed = {
            "model": worst["model"],
            "seed": worst["seed"],
            "family": DETECTOR_FAMILY.get(worst["model"], ""),
            "clean_f1": worst["clean_f1"],
            "min_f1": worst["min_f1"],
            "delta_f1": worst["delta_f1"],
            "peak_round": worst["peak_round"],
            "peak_cumulative_p": worst["peak_cumulative_p"],
            "fnr_at_peak": worst["fnr_at_peak"],
            "delta_fnr_at_peak": worst["delta_fnr_at_peak"],
            "worst_generator": worst["generator"],
            "worst_T": worst["T"],
            "worst_dp": worst["delta_p"],
        }
        rows.append(worst_renamed)
    out = pd.DataFrame(rows).sort_values(["model", "seed"]).reset_index(drop=True)
    print(f"[Table X] {len(out)} rows (expect 55 = 11 detectors × 5 seeds)")
    return out


# ────────────────────────── Table XI ─────────────────────────────────
def build_table_xi(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(generator, T, Δp) summary across all (detector, seed) trajectories.

    For each of the 18 cells, compute median per-seed min_f1, IQR, and
    mean ΔFNR at peak. Aggregates across 11 detectors × 5 seeds = 55 trajectories
    per cell.
    """
    print("[Table XI] computing per-config summary ...")
    rows = []
    for (gen, T, dp), g in long_df.groupby(["generator", "T", "delta_p"]):
        # Compute one peak-min per (detector, seed) within this config
        traj_minima = []
        for (model, seed), tg in g.groupby(["model", "seed"]):
            tg = tg.sort_values("round")
            clean_row = tg[tg["round"] == 0]
            non_clean = tg[tg["round"] > 0]
            if clean_row.empty or non_clean.empty:
                continue
            clean_f1 = float(clean_row["f1"].iloc[0])
            clean_fnr = float(clean_row["fnr"].iloc[0])
            peak_idx = non_clean["f1"].idxmin()
            peak = non_clean.loc[peak_idx]
            traj_minima.append({
                "min_f1": float(peak["f1"]),
                "delta_f1": float(peak["f1"]) - clean_f1,
                "delta_fnr_at_peak": float(peak["fnr"]) - clean_fnr,
            })
        if not traj_minima:
            continue
        df_t = pd.DataFrame(traj_minima)
        rows.append({
            "generator":         gen,
            "T":                 int(T),
            "delta_p":           float(dp),
            "n_trajectories":    int(len(df_t)),
            "median_min_f1":     float(df_t["min_f1"].median()),
            "iqr_min_f1":        float(df_t["min_f1"].quantile(0.75) - df_t["min_f1"].quantile(0.25)),
            "min_min_f1":        float(df_t["min_f1"].min()),
            "max_min_f1":        float(df_t["min_f1"].max()),
            "mean_delta_f1":     float(df_t["delta_f1"].mean()),
            "mean_delta_fnr":    float(df_t["delta_fnr_at_peak"].mean()),
        })
    out = pd.DataFrame(rows).sort_values(["generator", "T", "delta_p"]).reset_index(drop=True)
    print(f"[Table XI] {len(out)} rows (expect 18 = 2 generators × 3 T × 3 Δp)")
    return out


# ────────────────────────── Table XII ────────────────────────────────
def build_table_xii(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gradual-vs-one-shot at matched cumulative budgets {3%, 5%, 10%}.

    Pulls one-shot baseline from $SWAT_OUTPUT_DIR/checkpoints/attacks/random_flip/
    (the existing main grid). Compares median min_f1 across (detector, seed) for
    each matched budget.
    """
    print("[Table XII] comparing gradual-vs-one-shot ...")
    rows = []
    matched_budgets = [0.03, 0.05, 0.10]
    tol = 0.005  # tolerance for matching cumulative budgets

    # ─── one-shot from existing baseline (random_flip is the closest analogue) ───
    rf_dir = ATTACK_DIR / "random_flip"
    if not rf_dir.exists():
        print(f"[Table XII] WARN: {rf_dir} not found — leaving one-shot column blank")
        oneshot_by_rate: dict[float, list[float]] = {b: [] for b in matched_budgets}
    else:
        oneshot_by_rate = {b: [] for b in matched_budgets}
        for f in sorted(os.listdir(rf_dir)):
            if not f.endswith(".json"):
                continue
            d = _safe_load(rf_dir / f)
            if d is None or "f1" not in d:
                continue
            rate = float(d.get("poison_rate", -1))
            for b in matched_budgets:
                if abs(rate - b) < 1e-6:
                    oneshot_by_rate[b].append(float(d["f1"]))
                    break

    # ─── gradual: for each matched budget, find rounds with cumulative_p ≈ b ───
    for b in matched_budgets:
        # Match rows where cumulative_p is close to b
        mask = np.isclose(long_df["cumulative_p"], b, atol=tol)
        gradual_rows = long_df[mask]
        # Take per-(model, seed) minimum F1 within ±tol of this budget
        per_model_seed = (gradual_rows.groupby(["model", "seed"])["f1"]
                                       .min()
                                       .reset_index())
        gradual_f1s = per_model_seed["f1"].values
        oneshot_f1s = oneshot_by_rate.get(b, [])
        rows.append({
            "cumulative_p":          b,
            "n_gradual":             int(len(gradual_f1s)),
            "median_gradual_f1":     float(np.median(gradual_f1s)) if len(gradual_f1s) else float("nan"),
            "iqr_gradual_f1":        float(np.subtract(*np.percentile(gradual_f1s, [75, 25]))) if len(gradual_f1s) else float("nan"),
            "n_oneshot":             int(len(oneshot_f1s)),
            "median_oneshot_f1":     float(np.median(oneshot_f1s)) if len(oneshot_f1s) else float("nan"),
            "iqr_oneshot_f1":        float(np.subtract(*np.percentile(oneshot_f1s, [75, 25]))) if len(oneshot_f1s) else float("nan"),
            "gradual_minus_oneshot": (float(np.median(gradual_f1s)) - float(np.median(oneshot_f1s)))
                                     if (len(gradual_f1s) and len(oneshot_f1s)) else float("nan"),
        })
    out = pd.DataFrame(rows)
    print(f"[Table XII] {len(out)} rows (expect 3 = budgets {matched_budgets})")
    return out


# ────────────────────────── Figures ──────────────────────────────────
def fig10_per_seed_trajectories(long_df: pd.DataFrame, out_path: Path):
    """F1 vs round, one line per seed, for AE and PCA × 2 generators."""
    print("[Fig 10] per-seed trajectories ...")
    detectors = ["autoencoder", "pca"]
    generators = ["random_injection", "high_loss"]
    fig, axes = plt.subplots(len(detectors), len(generators),
                             figsize=(11, 4 * len(detectors)),
                             sharey=True)
    if len(detectors) == 1:
        axes = [axes]

    for i, det in enumerate(detectors):
        for j, gen in enumerate(generators):
            ax = axes[i][j]
            sub = long_df[(long_df["model"] == det) & (long_df["generator"] == gen)]
            # Plot one line per (T, dp, seed) combo, color by seed
            seeds = sorted(sub["seed"].unique())
            cmap = plt.get_cmap("viridis", len(seeds))
            for s_i, s in enumerate(seeds):
                seed_data = sub[sub["seed"] == s]
                # For visual clarity, plot the worst T/dp combo per seed
                # (the trajectory with the lowest min_f1)
                traj_groups = list(seed_data.groupby(["T", "delta_p"]))
                if not traj_groups:
                    continue
                worst_key, worst_df = min(traj_groups, key=lambda kv: kv[1]["f1"].min())
                worst_df = worst_df.sort_values("round")
                ax.plot(worst_df["round"], worst_df["f1"],
                        marker="o", color=cmap(s_i),
                        label=f"seed={s} (T={worst_key[0]}, Δp={worst_key[1]})")
                # Mark the peak damage point
                peak = worst_df.loc[worst_df["f1"].idxmin()]
                ax.scatter([peak["round"]], [peak["f1"]],
                           s=120, marker="x", color=cmap(s_i), zorder=5)
            ax.set_title(f"{det}  /  {gen}")
            ax.set_xlabel("round")
            ax.set_ylabel("F1")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="lower left")
    plt.suptitle("Fig. 10  Per-seed trajectories under online retraining poisoning\n"
                 "(worst T/Δp combo per seed; × marks peak damage)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig11_dF1_dFNR_scatter(table_x: pd.DataFrame, out_path: Path):
    """(ΔF1, ΔFNR) joint scatter at peak, colored by detector family."""
    print("[Fig 11] (ΔF1, ΔFNR) joint scatter ...")
    fig, ax = plt.subplots(figsize=(8, 6))
    families = sorted(table_x["family"].dropna().unique())
    cmap = plt.get_cmap("tab20", len(families))
    family_to_color = {f: cmap(i) for i, f in enumerate(families)}
    for fam in families:
        sub = table_x[table_x["family"] == fam]
        ax.scatter(sub["delta_f1"], sub["delta_fnr_at_peak"],
                   s=80, color=family_to_color[fam], alpha=0.75,
                   label=fam, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("ΔF1 at peak  (negative = damage)")
    ax.set_ylabel("ΔFNR at peak  (positive = recall failure; negative = precision-only)")
    ax.set_title("Fig. 11  Joint (ΔF1, ΔFNR) scatter at each seed's peak\n"
                 "Bottom-right quadrant = combined precision+recall collapse")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig12_gradual_vs_oneshot(table_xii: pd.DataFrame, out_path: Path):
    """Side-by-side bars at matched cumulative budgets {3%, 5%, 10%}."""
    print("[Fig 12] gradual-vs-one-shot bars ...")
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(table_xii))
    w = 0.35
    ax.bar(x - w/2, table_xii["median_oneshot_f1"], width=w,
           label="One-shot (random_flip baseline)", color="#888")
    ax.bar(x + w/2, table_xii["median_gradual_f1"], width=w,
           label="Gradual (online retraining, worst-config aggregate)",
           color="#c44")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0%}" for p in table_xii["cumulative_p"]])
    ax.set_xlabel("cumulative poison budget")
    ax.set_ylabel("median F1 across (detector × seed)")
    ax.set_title("Fig. 12  Gradual vs one-shot at matched cumulative budgets")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    # Annotate the gap on each pair
    for i, row in table_xii.iterrows():
        gap = row["gradual_minus_oneshot"]
        if not np.isnan(gap):
            txt = f"Δ={gap:+.3f}"
            ax.annotate(txt, (x[i], max(row["median_oneshot_f1"], row["median_gradual_f1"]) + 0.02),
                        ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ────────────────────────── main ─────────────────────────────────────
def main():
    print(f"[aggregate_online] reading from {ONLINE_DIR}")
    print(f"[aggregate_online] writing to   {AGG_DIR}")
    long_df = load_online_long()
    long_path = AGG_DIR / "online_results.csv"
    long_df.to_csv(long_path, index=False)
    print(f"[save] online_results.csv -> {long_path}  ({len(long_df)} rows)")

    table_x = build_table_x(long_df)
    table_x.to_csv(AGG_DIR / "Table_X_per_seed_minima.csv", index=False)
    print(f"[save] Table_X_per_seed_minima.csv  ({len(table_x)} rows)")

    table_xi = build_table_xi(long_df)
    table_xi.to_csv(AGG_DIR / "Table_XI_per_config_summary.csv", index=False)
    print(f"[save] Table_XI_per_config_summary.csv  ({len(table_xi)} rows)")

    table_xii = build_table_xii(long_df)
    table_xii.to_csv(AGG_DIR / "Table_XII_gradual_vs_oneshot.csv", index=False)
    print(f"[save] Table_XII_gradual_vs_oneshot.csv  ({len(table_xii)} rows)")

    fig10_per_seed_trajectories(long_df, FIG_DIR / "Fig10_per_seed_trajectories.png")
    print(f"[save] {FIG_DIR / 'Fig10_per_seed_trajectories.png'}")
    fig11_dF1_dFNR_scatter(table_x, FIG_DIR / "Fig11_dF1_dFNR_scatter.png")
    print(f"[save] {FIG_DIR / 'Fig11_dF1_dFNR_scatter.png'}")
    fig12_gradual_vs_oneshot(table_xii, FIG_DIR / "Fig12_gradual_vs_oneshot.png")
    print(f"[save] {FIG_DIR / 'Fig12_gradual_vs_oneshot.png'}")

    # Headline numbers (for the abstract / response-to-reviewer letter)
    print("\n=== HEADLINE NUMBERS ===")
    ae_x = table_x[table_x["model"] == "autoencoder"]
    if len(ae_x):
        print(f"AE per-seed min F1 (worst config):  "
              f"mean={ae_x['min_f1'].mean():.4f}, "
              f"clean={ae_x['clean_f1'].mean():.4f}, "
              f"ΔF1={ae_x['delta_f1'].mean():+.4f}")
    print(f"All-detector mean per-seed min F1:  {table_x['min_f1'].mean():.4f}")
    print(f"All-detector mean ΔF1:              {table_x['delta_f1'].mean():+.4f}")
    print(f"All-detector mean ΔFNR at peak:     {table_x['delta_fnr_at_peak'].mean():+.4f}")

    print("\n[aggregate_online] done. Rsync the folder back to your Mac:")
    print(f"   rsync -av $USER@narval.alliancecan.ca:{AGG_DIR}/ \\")
    print(f"       ./swat_paper_run/online_aggregated/")


if __name__ == "__main__":
    main()
