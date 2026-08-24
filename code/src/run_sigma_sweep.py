"""
Feature-noise σ sweep — single (model, seed) per invocation.

Runs the feature_noise poisoning attack at a user-specified σ and a fixed
poison rate (default 0.10), for all 11 non-SOD models across all 3 seeds.

Usage (from CLI or Slurm array):

    python -m src.run_sigma_sweep --task-id $SLURM_ARRAY_TASK_ID --sigma 0.30

Task-id decoding is identical to run_attack.py: model_idx = id // 3, seed_idx = id % 3.
Output lands at:

    $SWAT_OUTPUT_DIR/checkpoints/sigma_sweep/s{sigma_tag}/{model}__{seed}.json

which is independent of the main attack grid — so the original feature_noise
results (σ = 0.15) stay untouched.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config as _cfg
    from src.attacks import apply_poison
    from src.config import CONFIG, decode_attack_task, get_data_paths, get_output_dir
    from src.data import (create_lstm_ae_splits, create_splits, load_raw,
                          preprocess_swat, subsample_if_heavy)
    from src.eval_utils import evaluate, save_json_atomic, set_seed
    from src.models import create_model
else:
    from . import config as _cfg
    from .attacks import apply_poison
    from .config import CONFIG, decode_attack_task, get_data_paths, get_output_dir
    from .data import (create_lstm_ae_splits, create_splits, load_raw,
                       preprocess_swat, subsample_if_heavy)
    from .eval_utils import evaluate, save_json_atomic, set_seed
    from .models import create_model

import torch


# Models that are not part of this sweep (SOD is excluded on quality grounds)
EXCLUDE_MODELS = {"sod"}
DEFAULT_RATE = 0.10


def _sigma_tag(sigma: float) -> str:
    # e.g. 0.30 -> "0p30"; 1.0 -> "1p00"
    return f"{sigma:.2f}".replace(".", "p")


def run_one_combo(model_name: str, seed: int, sigma: float, rate: float,
                  df, feature_cols, df_n_clean, df_a_clean,
                  X_tr, X_n, X_v, X_te, y_tr, y_v, y_te,
                  lstm_sp) -> dict:
    input_dim = len(feature_cols)

    # Override CONFIG's default sigma just for this call
    old_sigma = CONFIG["NOISE_SIGMA"]
    CONFIG["NOISE_SIGMA"] = sigma
    try:
        t0 = time.time()
        set_seed(seed)
        det = create_model(model_name, input_dim, seed=seed)

        if model_name == "lstm_ae":
            X_sq_tr, X_sq_vn, X_sq_vm, y_sq_vm, X_sq_te, y_sq_te, _ = lstm_sp
            X_poisoned, info = apply_poison(X_sq_tr, X_tr, y_tr, "feature_noise", rate, seed)
            det.train(X_poisoned, X_sq_vn, X_sq_vm, y_sq_vm)
            y_scores = det.decision_scores(X_sq_te)
            y_eval   = det._create_sequence_labels(y_sq_te)
            n = min(len(y_scores), len(y_eval))
            y_pred = (y_scores[:n] >= det.threshold).astype(int)
            met = evaluate(y_eval[:n], y_pred, y_scores[:n])
            split_type = "contiguous_normal"
        else:
            X_poisoned, info = apply_poison(X_n, X_tr, y_tr, "feature_noise", rate, seed)
            X_train_use = subsample_if_heavy(X_poisoned, model_name, seed)
            det.train(X_train_use, None, X_v, y_v)
            y_pred   = det.predict(X_te)
            y_scores = det.decision_scores(X_te)
            met = evaluate(y_te, y_pred, y_scores)
            split_type = "random"

        met.update({
            "model":        model_name,
            "seed":         int(seed),
            "attack":       "feature_noise",
            "noise_sigma":  float(sigma),
            "poison_rate":  float(rate),
            "time":         time.time() - t0,
            "split_type":   split_type,
            "n_injected":   info.get("n_poisoned"),
            "threshold":    float(det.threshold) if det.threshold is not None else None,
            "n_features":   int(input_dim),
        })
        del det
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return met
    finally:
        CONFIG["NOISE_SIGMA"] = old_sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, required=True,
                    help="SLURM_ARRAY_TASK_ID, 0..32 (11 models × 3 seeds, SOD excluded).")
    ap.add_argument("--sigma", type=float, required=True,
                    help="Feature-noise σ (multiplier on per-feature std).")
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE,
                    help=f"Poisoning rate (default {DEFAULT_RATE}).")
    ap.add_argument("--force", action="store_true", help="Re-run even if JSON already exists.")
    args = ap.parse_args()

    # Decode task-id to (model, seed) with SOD-filtered model list
    non_sod_models = [m for m in CONFIG["MODELS"] if m not in EXCLUDE_MODELS]
    n_seeds = len(CONFIG["SEEDS"])
    if not (0 <= args.task_id < len(non_sod_models) * n_seeds):
        raise SystemExit(
            f"task-id {args.task_id} out of range 0..{len(non_sod_models) * n_seeds - 1}"
        )
    model_name = non_sod_models[args.task_id // n_seeds]
    seed = CONFIG["SEEDS"][args.task_id % n_seeds]

    out_dir = get_output_dir()
    sweep_dir = out_dir / "checkpoints" / "sigma_sweep" / f"s{_sigma_tag(args.sigma)}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rate_tag = "r" + f"{args.rate:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    out_file = sweep_dir / f"{model_name}__{seed}__{rate_tag}.json"

    print(f"[σ-sweep task {args.task_id}] model={model_name} seed={seed} σ={args.sigma} rate={args.rate}")
    print(f"[σ-sweep task {args.task_id}] output: {out_file}")

    if out_file.exists() and not args.force:
        print(f"[σ-sweep task {args.task_id}] already done — skipping.")
        return 0

    # Load + preprocess + split — identical to run_attack.py
    _, normal_csv, attack_csv = get_data_paths()
    df_n, df_a = load_raw(normal_csv, attack_csv)
    df, feature_cols, df_n_clean, df_a_clean = preprocess_swat(df_n, df_a)

    set_seed(seed)
    X_tr, X_n, X_v, X_te, y_tr, y_v, y_te, _ = create_splits(
        df, feature_cols, CONFIG["TEST_SIZE"], CONFIG["VAL_SIZE"], seed)
    lstm_sp = create_lstm_ae_splits(df_n_clean, df_a_clean, feature_cols, seed)

    try:
        result = run_one_combo(
            model_name, seed, args.sigma, args.rate,
            df, feature_cols, df_n_clean, df_a_clean,
            X_tr, X_n, X_v, X_te, y_tr, y_v, y_te, lstm_sp,
        )
        save_json_atomic(result, out_file)
        print(f"[σ-sweep task {args.task_id}] F1={result['f1']:.4f} "
              f"FNR={result['fnr']:.4f} t={result['time']:.1f}s -> saved")
        return 0
    except Exception as e:
        err_file = out_file.with_suffix(".error.json")
        save_json_atomic({
            "model": model_name, "seed": int(seed),
            "attack": "feature_noise", "noise_sigma": float(args.sigma),
            "poison_rate": float(args.rate),
            "error": str(e), "traceback": traceback.format_exc(),
            "f1": 0.0, "fnr": 1.0,
        }, err_file)
        print(f"[σ-sweep task {args.task_id}] FAILED: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
