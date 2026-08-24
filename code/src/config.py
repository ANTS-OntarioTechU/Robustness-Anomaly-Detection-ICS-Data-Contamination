"""
Grid configuration, tuned hyperparameters, and Narval path resolution.

Paths are driven entirely by environment variables so the same code runs in:
  * an interactive VS Code / Remote-SSH terminal on a login node
  * a Slurm job on a compute node
  * a local sanity check on a laptop (by exporting SWAT_DATA_DIR + SWAT_OUTPUT_DIR)

No Colab / Google Drive fallbacks — this is the Narval-specific package.
"""
from __future__ import annotations

import os
from pathlib import Path


# ───────────────────────── experiment grid ──────────────────────────
CONFIG = {
    "POISON_RATES": [0.01, 0.03, 0.05, 0.10],
    "SEEDS":        [42, 123, 456],
    "ATTACKS":      ["random_flip", "targeted_flip", "feature_noise"],
    "MODELS":       [
        "iforest", "svm", "lof", "cluster", "knn", "histogram",
        "pca", "mcd", "abod", "sod", "autoencoder", "lstm_ae",
    ],

    # Split ratios (used only by the pointwise pipeline — LSTM-AE uses its own contiguous split)
    "TEST_SIZE":     0.15,
    "VAL_SIZE":      0.15,
    "CONTAMINATION": 0.05,

    # Scalability — heavy PyOD models get subsampled
    "HEAVY_MODELS":      ["svm", "lof", "knn", "abod", "sod", "mcd"],
    "MAX_TRAIN_SAMPLES": 50000,

    # Autoencoder (V3, post-HPO)
    "AE_HIDDEN_DIMS": [256, 128, 64],
    "AE_EPOCHS":      100,
    "AE_LR":          0.0005,
    "AE_DROPOUT":     0.1,
    "AE_PATIENCE":    15,
    "AE_BATCH_SIZE":  2048,            # HPO winner (was hardcoded 1024)
    "AE_ACTIVATION":  "leaky_relu",    # HPO winner (was hardcoded "relu")

    # LSTM-Autoencoder (post-HPO winners — see paper/SWaT_config_patch.md)
    "LSTM_AE_WINDOW":    30,           # HPO winner (was 20)
    "LSTM_AE_HIDDEN":    256,          # HPO winner (was 128)
    "LSTM_AE_EPOCHS":    30,           # HPO winner (was 50; early-stop converges earlier)
    "LSTM_AE_LR":        0.001,        # HPO winner (was 5e-4)
    "LSTM_AE_DROPOUT":   0.1,          # HPO winner (was 0.2 — no-op at num_layers=1; kept for explicitness)
    "LSTM_AE_PATIENCE":  15,
    "LSTM_AE_BATCH":     256,          # HPO winner (was hardcoded 512)
    "LSTM_AE_OPTIMIZER": "adamw",      # HPO winner (was hardcoded "adam")

    # Feature-noise attack
    "NOISE_SIGMA": 0.15,

    # ─── Online / retraining poisoning (archival extension; not reported in
    #     the CASCON paper) ──────────────────────────────────────────────
    # 11 detectors × 2 generators × 3 T × 3 Δp × 5 seeds = 990 trajectory runs.
    # SOD is excluded because it is not part of the reported benchmark.
    "ONLINE_DETECTORS": [
        "iforest", "svm", "lof", "cluster", "knn", "histogram",
        "pca", "mcd", "abod", "autoencoder", "lstm_ae",
    ],
    "ONLINE_GENERATORS":     ["random_injection", "high_loss"],
    "ONLINE_T_VALUES":       [3, 5, 10],
    "ONLINE_DELTA_P_VALUES": [0.005, 0.01, 0.02],
    "ONLINE_SEEDS":          [42, 123, 456, 789, 1024],

    # high_loss generator: clean-AE proxy (used to score every attack-pool
    # sample by reconstruction error). The same ranking is reused across all
    # 11 detectors at a given seed so cross-detector comparisons are clean.
    # The proxy AE follows the AE_* settings above and is cached on disk.
    "ONLINE_HIGH_LOSS_PROXY": "autoencoder",
}


# ───────────────────── tuned hyperparameters ────────────────────────
# These are the ONLY non-default hyperparameters in the paper run.
# Source: Phase 1 sensitivity study (v16_anomaly_diag/phase1_clean_sensitivity.csv),
# Phase 2 confirmed. Promotion rule: ΔF1 ≥ 0.03 OR ΔFNR ≤ -0.05 vs default.
TUNED_PARAMS = {
    "pca": {"n_components": 0.90},           # PCA-1 winner, ΔF1 = +0.1986
    "svm": {"nu": 0.01, "gamma": "scale"},   # SVM-1 winner, ΔF1 = +0.0585
}


# ─────────────────────────── paths ──────────────────────────────────
def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"Export it before launching, or source env/setup_venv.sh."
        )
    return v


def get_data_paths() -> tuple[Path, Path, Path]:
    """
    Returns (data_dir, normal_csv, attack_csv).

    SWAT_DATA_DIR must point to a directory containing 'normal.csv' and 'attack.csv'.
    Default on Narval: $SCRATCH/swat_data
    """
    data_dir = Path(os.environ.get("SWAT_DATA_DIR") or f"{_require_env('SCRATCH')}/swat_data")
    normal = data_dir / "normal.csv"
    attack = data_dir / "attack.csv"
    if not normal.exists() or not attack.exists():
        raise FileNotFoundError(
            f"Expected normal.csv + attack.csv under {data_dir}. "
            f"Upload the SWaT CSVs there (see README § Data)."
        )
    return data_dir, normal, attack


def get_output_dir() -> Path:
    """
    Returns the paper-run output directory.

    SWAT_OUTPUT_DIR default on Narval: $SCRATCH/swat_paper_run
    """
    out = Path(os.environ.get("SWAT_OUTPUT_DIR") or f"{_require_env('SCRATCH')}/swat_paper_run")
    (out / "checkpoints" / "clean").mkdir(parents=True, exist_ok=True)
    for atk in CONFIG["ATTACKS"]:
        (out / "checkpoints" / "attacks" / atk).mkdir(parents=True, exist_ok=True)
    (out / "checkpoints" / "online").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints" / "online" / "_rankings").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return out


# ───────────────────── grid task-id decoding ────────────────────────
# Clean baseline array: one task per (model, seed) → 12 * 3 = 36 tasks.
# Attack arrays       : one task per (model, seed) → 36 tasks; each task
#                        iterates the 4 poison rates sequentially so splits
#                        and model construction are amortized.
def clean_task_shape() -> int:
    return len(CONFIG["MODELS"]) * len(CONFIG["SEEDS"])


def attack_task_shape() -> int:
    return len(CONFIG["MODELS"]) * len(CONFIG["SEEDS"])


def decode_clean_task(task_id: int) -> tuple[str, int]:
    """(task_id) → (model, seed)."""
    n_seeds = len(CONFIG["SEEDS"])
    model_idx = task_id // n_seeds
    seed_idx  = task_id % n_seeds
    if not (0 <= model_idx < len(CONFIG["MODELS"])):
        raise IndexError(f"clean task_id {task_id} out of range 0..{clean_task_shape() - 1}")
    return CONFIG["MODELS"][model_idx], CONFIG["SEEDS"][seed_idx]


def decode_attack_task(task_id: int) -> tuple[str, int]:
    """(task_id) → (model, seed); rates iterated inside the task."""
    return decode_clean_task(task_id)  # same shape


# ───────────────────── online-grid manifest helpers ─────────────────
def online_combos() -> list[dict]:
    """
    Enumerate every (detector, generator, T, delta_p, seed) combo for the
    online-retraining grid. Order is deterministic so SLURM_ARRAY_TASK_ID
    indexes consistently across submissions.

    Length: len(ONLINE_DETECTORS) × len(ONLINE_GENERATORS)
            × len(ONLINE_T_VALUES) × len(ONLINE_DELTA_P_VALUES)
            × len(ONLINE_SEEDS)
    Default: 11 × 2 × 3 × 3 × 5 = 990
    """
    out = []
    for det in CONFIG["ONLINE_DETECTORS"]:
        for gen in CONFIG["ONLINE_GENERATORS"]:
            for T in CONFIG["ONLINE_T_VALUES"]:
                for dp in CONFIG["ONLINE_DELTA_P_VALUES"]:
                    for seed in CONFIG["ONLINE_SEEDS"]:
                        out.append({
                            "detector":  det,
                            "generator": gen,
                            "T":         int(T),
                            "delta_p":   float(dp),
                            "seed":      int(seed),
                        })
    return out


def online_combo_count() -> int:
    return (
        len(CONFIG["ONLINE_DETECTORS"])
        * len(CONFIG["ONLINE_GENERATORS"])
        * len(CONFIG["ONLINE_T_VALUES"])
        * len(CONFIG["ONLINE_DELTA_P_VALUES"])
        * len(CONFIG["ONLINE_SEEDS"])
    )


def online_dp_tag(dp: float) -> str:
    """0.005 -> 'dp0p005' (filename-safe)."""
    return "dp" + f"{dp:.4f}".rstrip("0").rstrip(".").replace(".", "p")


# ───────────────────── runtime toggles ──────────────────────────────
# Migration from old v16_*/final_paper_run checkpoints is DISABLED by design:
# this Narval run is the single source of truth for the paper.
MIGRATE_OLD_CHECKPOINTS = False
