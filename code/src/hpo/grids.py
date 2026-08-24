"""
Grid definitions for the AE and LSTM-AE hyperparameter search.

Three stages per detector (architecture → training → loss/threshold[/scaler]).
Stage 2 and Stage 3 read the top-3 winners from the preceding stage.

Design notes
------------
* Every candidate is evaluated under TWO conditions in a single Slurm task:
  clean training pool + 10 % targeted-flip poisoning. Per-task JSON records both.
* Stage-1 is single-seed (42). Stages 2-3 single-seed. Final confirmation runs
  top-3 × three seeds × full poisoning grid (3 attacks × 4 rates + clean).
* Grid sizes (single-seed unless noted):
      AE stage 1 = 192, stage 2 = 144, stage 3 = 81, final = 9.
      LSTM-AE stage 1 = 72, stage 2 = 108, stage 3 = 27, final = 9.
* Total Slurm tasks = 426 AE + 216 LSTM-AE = 642.
"""
from __future__ import annotations

from typing import Any


# ════════════════════════════════════════════════════════════════════
# Fixed defaults reused across all stages (match deployed paper config)
# ════════════════════════════════════════════════════════════════════
AE_FIXED_STAGE1: dict[str, Any] = {
    "detector":            "ae",
    "loss_fn":             "mse",
    "optimizer":           "adam",
    "lr":                  5e-4,
    "weight_decay":        1e-5,
    "batch_size":          1024,
    "epochs":              100,
    "patience":            15,
    "threshold_strategy":  "f1_optimal",
    "scaler":              "minmax",
    "seed":                42,
    "poison_attack":       "targeted_flip",
    "poison_rate":         0.10,
}

LSTM_AE_FIXED_STAGE1: dict[str, Any] = {
    "detector":            "lstm_ae",
    "loss_fn":             "mse",
    "optimizer":           "adam",
    "lr":                  5e-4,
    "weight_decay":        1e-5,
    "batch_size":          512,
    "epochs":              50,
    "patience":            15,
    "window":              20,        # overridden in stage 1 grid
    "hidden_dim":          128,       # overridden in stage 1 grid
    "num_layers":          1,         # overridden in stage 1 grid
    "dropout":             0.2,       # overridden in stage 1 grid
    "threshold_strategy":  "f1_optimal",
    "scaler":              "minmax",
    "seed":                42,
    "poison_attack":       "targeted_flip",
    "poison_rate":         0.10,
}


# ════════════════════════════════════════════════════════════════════
# Feed-forward Autoencoder — 3-stage grid
# ════════════════════════════════════════════════════════════════════
#
# Stage 1 reproduces the clean-only HPO notebook's architecture sweep
# (192 configs = 6 × 4 × 4 × 2) but now evaluated under clean + 10 %
# targeted poisoning. That lets us ask the *new* question Phase-1 could
# not: does the clean-F1-optimal architecture also survive poisoning?
# ════════════════════════════════════════════════════════════════════
AE_STAGE1_GRID: dict[str, list] = {
    "hidden_dims": [
        [64, 32, 16],        # baseline
        [128, 64, 32],       # wider
        [128, 64, 32, 16],   # deeper + wider
        [256, 128, 64],      # much wider (current deployed)
        [128, 64],           # shallower
        [64, 32, 16, 8],     # deeper bottleneck
    ],
    "dropout":        [0.0, 0.1, 0.2, 0.3],
    "activation":     ["relu", "leaky_relu", "elu", "selu"],
    "use_batchnorm":  [False, True],
}


AE_STAGE2_GRID: dict[str, list] = {
    # Applied over top-3 architectures from Stage 1.
    # 3 archs × 2 × 4 × 3 × 2 = 144 configs
    "optimizer":   ["adam", "adamw"],
    "lr":          [1e-4, 5e-4, 1e-3, 5e-3],
    "batch_size":  [512, 1024, 2048],
    "epochs":      [50, 100],
}


AE_STAGE3_GRID: dict[str, list] = {
    # Applied over top-3 (arch + training) from Stage 2.
    # 3 × 3 × 3 × 3 = 81 configs
    "loss_fn":             ["mse", "mae", "huber"],
    "threshold_strategy":  ["f1_optimal", "percentile_95", "percentile_99"],
    "scaler":              ["minmax", "standard", "robust"],
}


# ════════════════════════════════════════════════════════════════════
# LSTM-Autoencoder — 3-stage grid (NEW — no prior HPO existed)
# ════════════════════════════════════════════════════════════════════
#
# Stage 1 is smaller than AE's because LSTM-AE is ~2× more expensive per
# config (contiguous training, sequence unrolling). The window dimension
# is the most load-bearing knob (it sets how much temporal context the
# detector sees) so it gets 4 values; the rest follow the AE shape.
# ════════════════════════════════════════════════════════════════════
LSTM_AE_STAGE1_GRID: dict[str, list] = {
    # 4 × 3 × 2 × 3 = 72 configs
    "window":      [10, 20, 30, 50],
    "hidden_dim":  [64, 128, 256],
    "num_layers":  [1, 2],
    "dropout":     [0.1, 0.2, 0.3],
}


LSTM_AE_STAGE2_GRID: dict[str, list] = {
    # 3 × 2 × 3 × 3 × 2 = 108 configs
    "optimizer":   ["adam", "adamw"],
    "lr":          [1e-4, 5e-4, 1e-3],
    "batch_size":  [256, 512, 1024],
    "epochs":      [30, 50],
}


LSTM_AE_STAGE3_GRID: dict[str, list] = {
    # LSTM-AE always uses MinMax in the deployed pipeline (the contiguous
    # split is already scaled before windowing), so scaler is not swept.
    # 3 × 3 × 3 = 27 configs
    "loss_fn":             ["mse", "mae", "huber"],
    "threshold_strategy":  ["f1_optimal", "percentile_95", "percentile_99"],
}


# ════════════════════════════════════════════════════════════════════
# Final confirmation (both detectors): top-3 × 3 seeds × full grid
# ════════════════════════════════════════════════════════════════════
# The final stage re-runs the top-3 winners from stage 3 across three
# seeds and the full poisoning grid (3 attacks × 4 rates + 1 clean = 13
# conditions per run). That matches the main paper's 33-run cell
# for a single detector and gives us the variance + robustness curve.
FINAL_SEEDS = [42, 123, 456]
FINAL_ATTACKS = ["random_flip", "targeted_flip", "feature_noise"]
FINAL_RATES = [0.01, 0.03, 0.05, 0.10]


# ════════════════════════════════════════════════════════════════════
# σ-ablation on HPO winners (mirrors the R01 σ-sweep design)
# ════════════════════════════════════════════════════════════════════
# The R01 σ-ablation explored:
#   (a) σ ∈ {0.30, 0.50, 1.00} at rate 0.10  →  3 × 11 × 3 = 99 runs
#   (b) σ = 1.00 at rates {0.01, 0.03, 0.05} →  3 × 11 × 3 = 99 runs
# plus the main grid's σ=0.15 at rate 0.10 row already on disk.
#
# For HPO winners we run the same design minus the σ=0.15 (which is already
# known to be a null attack at rate 0.10):
#   (a) σ ∈ {0.15, 0.30, 0.50, 1.00} at rate 0.10
#   (b) σ = 1.00 at rates {0.01, 0.03, 0.05}
# = 4 + 3 = 7 unique (σ, rate) pairs × 2 detectors × 3 seeds = 42 tasks.
# Including σ=0.15 lets us plot V11 entirely from HPO data without splicing in
# R01 points.
SIGMA_HPO_SEEDS = [42, 123, 456]
SIGMA_HPO_SIGMAS = [0.15, 0.30, 0.50, 1.00]          # all at rate 0.10
SIGMA_HPO_RATE_SWEEP_SIGMA = 1.00                    # σ fixed
SIGMA_HPO_RATE_SWEEP_RATES = [0.01, 0.03, 0.05, 0.10]  # rate varies


def enumerate_sigma_hpo(ae_winner: dict, lstm_winner: dict) -> list[dict]:
    """
    Build a 42-entry manifest of (detector, HPO-winner config, σ, rate, seed)
    rows for the σ-ablation follow-up on the HPO winners.

    Each manifest entry is a *complete* config — the runner doesn't have to
    look anything up at task time. `poison_attack` is hardcoded to
    "feature_noise" because that's the only attack type σ affects.

    Parameters
    ----------
    ae_winner, lstm_winner : dict
        The HPO-winning configuration dicts (e.g. as read from
        `ae_final_by_config.csv` → `config` column JSON).
    """
    out = []
    for detector, base in [("ae", ae_winner), ("lstm_ae", lstm_winner)]:
        # (a) σ sweep at rate 0.10
        for sigma in SIGMA_HPO_SIGMAS:
            for seed in SIGMA_HPO_SEEDS:
                c = dict(base)
                c["detector"]      = detector
                c["seed"]          = seed
                c["poison_attack"] = "feature_noise"
                c["poison_rate"]   = 0.10
                c["noise_sigma"]   = float(sigma)
                out.append(c)
        # (b) rate sweep at σ = 1.00, excluding the already-covered rate 0.10
        for rate in SIGMA_HPO_RATE_SWEEP_RATES:
            if rate == 0.10:
                continue  # de-duplicate (already in the σ-sweep)
            for seed in SIGMA_HPO_SEEDS:
                c = dict(base)
                c["detector"]      = detector
                c["seed"]          = seed
                c["poison_attack"] = "feature_noise"
                c["poison_rate"]   = float(rate)
                c["noise_sigma"]   = float(SIGMA_HPO_RATE_SWEEP_SIGMA)
                out.append(c)
    return out


# ════════════════════════════════════════════════════════════════════
# Enumeration helpers
# ════════════════════════════════════════════════════════════════════
def enumerate_ae_stage1() -> list[dict]:
    """Cartesian product over AE_STAGE1_GRID, merged with AE_FIXED_STAGE1."""
    out = []
    for hd in AE_STAGE1_GRID["hidden_dims"]:
        for dp in AE_STAGE1_GRID["dropout"]:
            for act in AE_STAGE1_GRID["activation"]:
                for bn in AE_STAGE1_GRID["use_batchnorm"]:
                    c = dict(AE_FIXED_STAGE1)
                    c["hidden_dims"]   = list(hd)
                    c["dropout"]       = dp
                    c["activation"]    = act
                    c["use_batchnorm"] = bn
                    out.append(c)
    return out


def enumerate_lstm_ae_stage1() -> list[dict]:
    out = []
    for w in LSTM_AE_STAGE1_GRID["window"]:
        for h in LSTM_AE_STAGE1_GRID["hidden_dim"]:
            for nl in LSTM_AE_STAGE1_GRID["num_layers"]:
                for dp in LSTM_AE_STAGE1_GRID["dropout"]:
                    c = dict(LSTM_AE_FIXED_STAGE1)
                    c["window"]     = w
                    c["hidden_dim"] = h
                    c["num_layers"] = nl
                    c["dropout"]    = dp
                    out.append(c)
    return out


def enumerate_stage2(base_configs: list[dict],
                     grid: dict[str, list]) -> list[dict]:
    """Cross the top-N configs with stage-2 grid, overwriting overlapping keys."""
    out = []
    for base in base_configs:
        for opt in grid["optimizer"]:
            for lr in grid["lr"]:
                for bs in grid["batch_size"]:
                    for ep in grid["epochs"]:
                        c = dict(base)
                        c["optimizer"]   = opt
                        c["lr"]          = lr
                        c["batch_size"]  = bs
                        c["epochs"]      = ep
                        c["patience"]    = 15
                        out.append(c)
    return out


def enumerate_ae_stage3(base_configs: list[dict]) -> list[dict]:
    out = []
    for base in base_configs:
        for loss in AE_STAGE3_GRID["loss_fn"]:
            for th in AE_STAGE3_GRID["threshold_strategy"]:
                for sc in AE_STAGE3_GRID["scaler"]:
                    c = dict(base)
                    c["loss_fn"]            = loss
                    c["threshold_strategy"] = th
                    c["scaler"]             = sc
                    out.append(c)
    return out


def enumerate_lstm_ae_stage3(base_configs: list[dict]) -> list[dict]:
    out = []
    for base in base_configs:
        for loss in LSTM_AE_STAGE3_GRID["loss_fn"]:
            for th in LSTM_AE_STAGE3_GRID["threshold_strategy"]:
                c = dict(base)
                c["loss_fn"]            = loss
                c["threshold_strategy"] = th
                out.append(c)
    return out


def enumerate_final(base_configs: list[dict]) -> list[dict]:
    """
    Top-3 × 3 seeds = 9 Slurm tasks. Each task loops the full 13-condition
    grid (clean + 3 attacks × 4 rates) internally to amortize data loading.
    """
    out = []
    for base in base_configs:
        for seed in FINAL_SEEDS:
            c = dict(base)
            c["seed"] = seed
            out.append(c)
    return out


# ════════════════════════════════════════════════════════════════════
# Totals (sanity-check)
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"AE      stage 1:  {len(enumerate_ae_stage1()):>4d} configs")
    print(f"AE      stage 2:  {144:>4d} configs (top-3 × 2×4×3×2)")
    print(f"AE      stage 3:  {81:>4d} configs (top-3 × 3×3×3)")
    print(f"AE      final:    {9:>4d} tasks   (top-3 × 3 seeds × 13 cells)")
    print(f"LSTM-AE stage 1:  {len(enumerate_lstm_ae_stage1()):>4d} configs")
    print(f"LSTM-AE stage 2:  {108:>4d} configs (top-3 × 2×3×3×2)")
    print(f"LSTM-AE stage 3:  {27:>4d} configs (top-3 × 3×3)")
    print(f"LSTM-AE final:    {9:>4d} tasks   (top-3 × 3 seeds × 13 cells)")
    # Sanity-check the sigma_hpo enumerator with placeholder winners
    dummy_ae   = {"detector":"ae","hidden_dims":[256,128,64]}
    dummy_lstm = {"detector":"lstm_ae","window":30}
    n_sig = len(enumerate_sigma_hpo(dummy_ae, dummy_lstm))
    print(f"σ-HPO:            {n_sig:>4d} tasks   (2 detectors × 3 seeds × 7 (σ,rate) pairs)")
