"""
Aggregator — merges per-combo JSONs into the paper CSVs and figures.

Run after the Slurm arrays finish (or anytime mid-run to peek at progress):

    python -m src.aggregate

Outputs (all under $SWAT_OUTPUT_DIR):
    checkpoints/clean_baselines.csv
    checkpoints/attack_checkpoint.csv
    all_results.csv
    table_T4_clean_baselines.csv
    table_T5_poisoning_impact.csv
    compute_cost.csv
    multi_criteria_ranking.csv
    figures/F3_robustness_curves.png
    figures/F4_f1_degradation_heatmap.png
    figures/F5_fnr_safety.png
    figures/F6_seed_variance.png
    figures/F7_per_model_comparison.png
    figures/F8_precision_recall_tradeoff.png
    run_summary.txt   (which combos are missing, total runtime, etc.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import CONFIG, get_output_dir
    from src.eval_utils import safe_to_csv
else:
    from .config import CONFIG, get_output_dir
    from .eval_utils import safe_to_csv


# ────────────────────── JSON → DataFrame loaders ────────────────────
def _load_jsons(pattern_dir: Path) -> list[dict]:
    rows = []
    if not pattern_dir.exists():
        return rows
    for fp in sorted(pattern_dir.glob("*.json")):
        if fp.name.endswith(".error.json"):
            continue
        try:
            with open(fp) as f:
                rows.append(json.load(f))
        except Exception as e:
            print(f"[aggregate] WARN: could not parse {fp}: {e}", flush=True)
    return rows


def load_clean(out_dir: Path) -> pd.DataFrame:
    rows = _load_jsons(out_dir / "checkpoints" / "clean")
    return pd.DataFrame(rows)


def load_attacks(out_dir: Path) -> pd.DataFrame:
    rows = []
    for atk in CONFIG["ATTACKS"]:
        rows.extend(_load_jsons(out_dir / "checkpoints" / "attacks" / atk))
    return pd.DataFrame(rows)


# ───────────────────── missing-combo reporting ──────────────────────
def report_missing(out_dir: Path) -> dict:
    report = {"clean_missing": [], "attack_missing": []}
    clean_dir = out_dir / "checkpoints" / "clean"
    for m in CONFIG["MODELS"]:
        for s in CONFIG["SEEDS"]:
            if not (clean_dir / f"{m}__{s}.json").exists():
                report["clean_missing"].append((m, s))
    for atk in CONFIG["ATTACKS"]:
        d = out_dir / "checkpoints" / "attacks" / atk
        for m in CONFIG["MODELS"]:
            for s in CONFIG["SEEDS"]:
                for r in CONFIG["POISON_RATES"]:
                    tag = "r" + f"{r:.4f}".rstrip("0").rstrip(".").replace(".", "p")
                    if not (d / f"{m}__{s}__{tag}.json").exists():
                        report["attack_missing"].append((atk, m, s, r))
    return report


# ─────────────────────────── tables ─────────────────────────────────
def build_table_T4(clean_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for m in CONFIG["MODELS"]:
        sub = clean_df[clean_df["model"] == m]
        if "error" in sub.columns:
            sub = sub[sub["error"].isna() | (sub["error"].astype(str) == "nan")]
        if len(sub) == 0:
            continue
        row = {"model": m}
        for k in ("f1", "recall", "precision", "fnr"):
            row[f"{k}_mean"] = sub[k].mean()
            row[f"{k}_std"]  = sub[k].std()
        row["time_mean"] = sub["time"].mean() if "time" in sub else 0.0
        rows.append(row)
    T4 = pd.DataFrame(rows)
    safe_to_csv(T4, out_dir / "table_T4_clean_baselines.csv", index=False)
    return T4


def build_table_T5(clean_df: pd.DataFrame, poisoned_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for m in CONFIG["MODELS"]:
        clean_f1  = clean_df[clean_df["model"] == m]["f1"].mean()  if len(clean_df) else np.nan
        clean_fnr = clean_df[clean_df["model"] == m]["fnr"].mean() if len(clean_df) else np.nan
        row = {"model": m, "clean_f1": clean_f1, "clean_fnr": clean_fnr}
        for a in CONFIG["ATTACKS"]:
            for r in CONFIG["POISON_RATES"]:
                s = poisoned_df[(poisoned_df["model"] == m)
                                & (poisoned_df["attack"] == a)
                                & (np.isclose(poisoned_df["poison_rate"], r))]
                row[f"{a}_{r}_f1"]  = s["f1"].mean()  if len(s) else np.nan
                row[f"{a}_{r}_fnr"] = s["fnr"].mean() if len(s) else np.nan
        rows.append(row)
    T5 = pd.DataFrame(rows)
    safe_to_csv(T5, out_dir / "table_T5_poisoning_impact.csv", index=False)
    return T5


def build_compute_cost(clean_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for m in CONFIG["MODELS"]:
        sub = clean_df[clean_df["model"] == m]
        if len(sub) == 0:
            continue
        f1 = sub["f1"].mean()
        t  = sub["time"].mean()
        verdict = ("lightweight (<10s)" if t < 10 else
                   "moderate (<100s)"   if t < 100 else
                   "heavy (>100s)")
        rows.append({"model": m, "clean_f1": f1, "train_s": t, "verdict": verdict})
    df = pd.DataFrame(rows)
    safe_to_csv(df, out_dir / "compute_cost.csv", index=False)
    return df


def build_multi_criteria_ranking(clean_df: pd.DataFrame, poisoned_df: pd.DataFrame,
                                 out_dir: Path) -> pd.DataFrame:
    rows = []
    for m in CONFIG["MODELS"]:
        clean_sub = clean_df[clean_df["model"] == m]
        pois_sub  = poisoned_df[poisoned_df["model"] == m]
        if len(clean_sub) == 0 or len(pois_sub) == 0:
            continue
        clean_f1 = clean_sub["f1"].mean()
        grouped = pois_sub.groupby(["attack", "poison_rate"])
        worst_drop = clean_f1 - grouped["f1"].mean().min()
        worst_fnr  = grouped["fnr"].mean().max()
        seed_std   = grouped["f1"].std().mean()
        t = clean_sub["time"].mean()
        rows.append({"model": m, "clean_f1": clean_f1,
                     "worst_f1_drop": worst_drop, "worst_fnr": worst_fnr,
                     "mean_seed_std": seed_std, "train_s": t})

    if not rows:
        return pd.DataFrame()

    rank_df = pd.DataFrame(rows)

    def norm(x, higher_better=True):
        lo, hi = x.min(), x.max()
        if hi == lo:
            return np.ones_like(x, dtype=float)
        y = (x - lo) / (hi - lo)
        return y if higher_better else (1 - y)

    rank_df["score_clean"]     = norm(rank_df["clean_f1"],      higher_better=True)
    rank_df["score_robust"]    = norm(rank_df["worst_f1_drop"], higher_better=False)
    rank_df["score_safety"]    = norm(rank_df["worst_fnr"],     higher_better=False)
    rank_df["score_stability"] = norm(rank_df["mean_seed_std"], higher_better=False)
    rank_df["score_speed"]     = norm(rank_df["train_s"],       higher_better=False)
    weights = dict(clean=0.30, robust=0.25, safety=0.25, stability=0.10, speed=0.10)
    rank_df["composite"] = (
        weights["clean"]     * rank_df["score_clean"]
        + weights["robust"]    * rank_df["score_robust"]
        + weights["safety"]    * rank_df["score_safety"]
        + weights["stability"] * rank_df["score_stability"]
        + weights["speed"]     * rank_df["score_speed"]
    )
    rank_df = rank_df.sort_values("composite", ascending=False).reset_index(drop=True)
    safe_to_csv(rank_df, out_dir / "multi_criteria_ranking.csv", index=False)
    return rank_df


# ─────────────────────────── figures ────────────────────────────────
def figure_F3(clean_df, poisoned_df, out_dir):
    n_att = len(CONFIG["ATTACKS"])
    fig, axes = plt.subplots(1, n_att, figsize=(6 * n_att, 5), sharey=True)
    if n_att == 1:
        axes = [axes]
    colors = plt.cm.tab20(np.linspace(0, 1, len(CONFIG["MODELS"])))
    for ax, a in zip(axes, CONFIG["ATTACKS"]):
        for i, m in enumerate(CONFIG["MODELS"]):
            xs, ys = [], []
            clean = clean_df[clean_df["model"] == m]["f1"].mean()
            if pd.notna(clean):
                xs.append(0); ys.append(clean)
            for r in CONFIG["POISON_RATES"]:
                s = poisoned_df[(poisoned_df["model"] == m)
                                & (poisoned_df["attack"] == a)
                                & (np.isclose(poisoned_df["poison_rate"], r))]
                if len(s):
                    xs.append(r); ys.append(s["f1"].mean())
            ax.plot(xs, ys, "-o", color=colors[i], label=m, markersize=5, linewidth=1.5)
        ax.set_title(a.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Poison Rate")
        ax.grid(alpha=0.3)
        ax.set_xticks([0] + CONFIG["POISON_RATES"])
    axes[0].set_ylabel("F1-score")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.suptitle("F3: Robustness Curves (F1 vs Poison Rate)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "F3_robustness_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_F4(clean_df, poisoned_df, out_dir):
    # Δ F1 heatmap: rows = models, cols = (attack, rate)
    cols = [(a, r) for a in CONFIG["ATTACKS"] for r in CONFIG["POISON_RATES"]]
    mat = np.full((len(CONFIG["MODELS"]), len(cols)), np.nan)
    for i, m in enumerate(CONFIG["MODELS"]):
        cf1 = clean_df[clean_df["model"] == m]["f1"].mean()
        for j, (a, r) in enumerate(cols):
            s = poisoned_df[(poisoned_df["model"] == m)
                            & (poisoned_df["attack"] == a)
                            & (np.isclose(poisoned_df["poison_rate"], r))]
            if len(s) and pd.notna(cf1):
                mat[i, j] = s["f1"].mean() - cf1
    fig, ax = plt.subplots(figsize=(1.2 * len(cols), 0.55 * len(CONFIG["MODELS"]) + 2))
    im = ax.imshow(mat, cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_yticks(range(len(CONFIG["MODELS"])), CONFIG["MODELS"])
    ax.set_xticks(range(len(cols)), [f"{a}\n{r}" for a, r in cols], rotation=45, ha="right")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < 0.3 else "white")
    plt.colorbar(im, ax=ax, label="Δ F1 (poisoned − clean)")
    ax.set_title("F4: F1 Degradation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "F4_f1_degradation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_F5(clean_df, poisoned_df, out_dir):
    n_att = len(CONFIG["ATTACKS"])
    fig, axes = plt.subplots(1, n_att, figsize=(6 * n_att, 5), sharey=True)
    if n_att == 1:
        axes = [axes]
    colors = plt.cm.tab20(np.linspace(0, 1, len(CONFIG["MODELS"])))
    for ax, a in zip(axes, CONFIG["ATTACKS"]):
        for i, m in enumerate(CONFIG["MODELS"]):
            xs, ys = [], []
            clean = clean_df[clean_df["model"] == m]["fnr"].mean()
            if pd.notna(clean):
                xs.append(0); ys.append(clean)
            for r in CONFIG["POISON_RATES"]:
                s = poisoned_df[(poisoned_df["model"] == m)
                                & (poisoned_df["attack"] == a)
                                & (np.isclose(poisoned_df["poison_rate"], r))]
                if len(s):
                    xs.append(r); ys.append(s["fnr"].mean())
            ax.plot(xs, ys, "-o", color=colors[i], label=m, markersize=5, linewidth=1.5)
        ax.set_title(a.replace("_", " ").title())
        ax.set_xlabel("Poison Rate"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("False Negative Rate (FNR)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.suptitle("F5: FNR Safety Analysis (lower = safer under poisoning)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "F5_fnr_safety.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_F6(clean_df, poisoned_df, out_dir):
    # F1 with ± 1σ bands over seeds
    n_att = len(CONFIG["ATTACKS"])
    fig, axes = plt.subplots(1, n_att, figsize=(6 * n_att, 5), sharey=True)
    if n_att == 1:
        axes = [axes]
    colors = plt.cm.tab20(np.linspace(0, 1, len(CONFIG["MODELS"])))
    for ax, a in zip(axes, CONFIG["ATTACKS"]):
        for i, m in enumerate(CONFIG["MODELS"]):
            xs, mu, sd = [], [], []
            clean = clean_df[clean_df["model"] == m]["f1"]
            if len(clean):
                xs.append(0); mu.append(clean.mean()); sd.append(clean.std(ddof=0))
            for r in CONFIG["POISON_RATES"]:
                s = poisoned_df[(poisoned_df["model"] == m)
                                & (poisoned_df["attack"] == a)
                                & (np.isclose(poisoned_df["poison_rate"], r))]["f1"]
                if len(s):
                    xs.append(r); mu.append(s.mean()); sd.append(s.std(ddof=0))
            xs = np.array(xs); mu = np.array(mu); sd = np.array(sd)
            ax.plot(xs, mu, "-o", color=colors[i], label=m, linewidth=1.5)
            ax.fill_between(xs, mu - sd, mu + sd, color=colors[i], alpha=0.15)
        ax.set_title(a.replace("_", " ").title())
        ax.set_xlabel("Poison Rate"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("F1 ± σ (3 seeds)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.suptitle("F6: Seed Variance", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "F6_seed_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_F7(clean_df, poisoned_df, out_dir):
    # one subplot per model, lines per attack
    n = len(CONFIG["MODELS"])
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                             sharey=True, sharex=True)
    axes = axes.flatten() if nrow * ncol > 1 else [axes]
    for i, m in enumerate(CONFIG["MODELS"]):
        ax = axes[i]
        clean = clean_df[clean_df["model"] == m]["f1"].mean()
        for a in CONFIG["ATTACKS"]:
            xs, ys = [], []
            if pd.notna(clean):
                xs.append(0); ys.append(clean)
            for r in CONFIG["POISON_RATES"]:
                s = poisoned_df[(poisoned_df["model"] == m)
                                & (poisoned_df["attack"] == a)
                                & (np.isclose(poisoned_df["poison_rate"], r))]
                if len(s):
                    xs.append(r); ys.append(s["f1"].mean())
            ax.plot(xs, ys, "-o", label=a, markersize=4)
        ax.set_title(m, fontsize=10); ax.grid(alpha=0.3)
        ax.set_xticks([0] + CONFIG["POISON_RATES"])
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    axes[0].legend(fontsize=8)
    plt.suptitle("F7: Per-Model Comparison (all 3 attacks)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "F7_per_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_F8(clean_df, poisoned_df, out_dir):
    # Precision vs recall scatter — clean filled circles, poisoned hollow
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, len(CONFIG["MODELS"])))
    for i, m in enumerate(CONFIG["MODELS"]):
        cs = clean_df[clean_df["model"] == m]
        ps = poisoned_df[poisoned_df["model"] == m]
        if len(cs):
            ax.scatter(cs["recall"], cs["precision"], color=colors[i], marker="o",
                       s=70, label=f"{m} (clean)")
        if len(ps):
            ax.scatter(ps["recall"], ps["precision"], color=colors[i], marker="x",
                       s=40, alpha=0.5)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("F8: Precision vs Recall (● clean, × poisoned)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "F8_precision_recall_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────── main ───────────────────────────────────
def main():
    out_dir = get_output_dir()

    clean_df    = load_clean(out_dir)
    poisoned_df = load_attacks(out_dir)

    print(f"[aggregate] clean rows:    {len(clean_df)}")
    print(f"[aggregate] poisoned rows: {len(poisoned_df)}")

    if len(clean_df):
        safe_to_csv(clean_df, out_dir / "checkpoints" / "clean_baselines.csv", index=False)
    if len(poisoned_df):
        safe_to_csv(poisoned_df, out_dir / "checkpoints" / "attack_checkpoint.csv", index=False)

    all_df = pd.concat([clean_df, poisoned_df], ignore_index=True) \
        if len(clean_df) and len(poisoned_df) else \
        (clean_df if len(clean_df) else poisoned_df)
    if len(all_df):
        safe_to_csv(all_df, out_dir / "all_results.csv", index=False)

    if len(clean_df):
        build_table_T4(clean_df, out_dir)
        build_compute_cost(clean_df, out_dir)
    if len(clean_df) and len(poisoned_df):
        build_table_T5(clean_df, poisoned_df, out_dir)
        build_multi_criteria_ranking(clean_df, poisoned_df, out_dir)
        # Figures
        figure_F3(clean_df, poisoned_df, out_dir)
        figure_F4(clean_df, poisoned_df, out_dir)
        figure_F5(clean_df, poisoned_df, out_dir)
        figure_F6(clean_df, poisoned_df, out_dir)
        figure_F7(clean_df, poisoned_df, out_dir)
        figure_F8(clean_df, poisoned_df, out_dir)

    # Missing-combo report
    report = report_missing(out_dir)
    n_clean_missing  = len(report["clean_missing"])
    n_attack_missing = len(report["attack_missing"])
    summary_lines = [
        f"Clean combos missing:  {n_clean_missing} / {len(CONFIG['MODELS']) * len(CONFIG['SEEDS'])}",
        f"Attack combos missing: {n_attack_missing} / "
        f"{len(CONFIG['MODELS']) * len(CONFIG['SEEDS']) * len(CONFIG['ATTACKS']) * len(CONFIG['POISON_RATES'])}",
    ]
    if n_clean_missing:
        summary_lines.append("")
        summary_lines.append("Clean missing (model, seed):")
        summary_lines.extend(f"  {m} {s}" for m, s in report["clean_missing"])
    if n_attack_missing:
        summary_lines.append("")
        summary_lines.append("Attack missing (attack, model, seed, rate):")
        summary_lines.extend(f"  {a} {m} {s} {r}" for a, m, s, r in report["attack_missing"])
    txt = "\n".join(summary_lines)
    (out_dir / "run_summary.txt").write_text(txt + "\n")
    print(txt, flush=True)
    print(f"\n[aggregate] outputs under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
