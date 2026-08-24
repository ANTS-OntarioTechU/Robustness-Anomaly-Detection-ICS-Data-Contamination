"""
Clean-baseline runner — one (model, seed) per invocation.

Invocation on a Slurm compute node:

    python -m src.run_clean --task-id $SLURM_ARRAY_TASK_ID

Output: $SWAT_OUTPUT_DIR/checkpoints/clean/{model}__{seed}.json (atomic write).
Re-runs that find the output file already present skip cleanly, so array
resubmission is safe and idempotent.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

# allow running as `python -m src.run_clean` OR directly
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import CONFIG, decode_clean_task, get_data_paths, get_output_dir
    from src.data import (create_lstm_ae_splits, create_splits, load_raw,
                          preprocess_swat, subsample_if_heavy)
    from src.eval_utils import evaluate, save_json_atomic, set_seed
    from src.models import create_model
else:
    from .config import CONFIG, decode_clean_task, get_data_paths, get_output_dir
    from .data import (create_lstm_ae_splits, create_splits, load_raw,
                       preprocess_swat, subsample_if_heavy)
    from .eval_utils import evaluate, save_json_atomic, set_seed
    from .models import create_model

import torch


def run_one_clean(model_name: str, seed: int) -> dict:
    _, normal_csv, attack_csv = get_data_paths()
    df_n, df_a = load_raw(normal_csv, attack_csv)
    df, feature_cols, df_n_clean, df_a_clean = preprocess_swat(df_n, df_a)

    set_seed(seed)
    X_tr, X_n, X_v, X_te, y_tr, y_v, y_te, _ = create_splits(
        df, feature_cols, CONFIG["TEST_SIZE"], CONFIG["VAL_SIZE"], seed)

    lstm_sp = create_lstm_ae_splits(df_n_clean, df_a_clean, feature_cols, seed)
    X_sq_tr, X_sq_vn, X_sq_vm, y_sq_vm, X_sq_te, y_sq_te, _ = lstm_sp

    input_dim = len(feature_cols)
    t0 = time.time()
    det = create_model(model_name, input_dim, seed=seed)

    if model_name == "lstm_ae":
        det.train(X_sq_tr, X_sq_vn, X_sq_vm, y_sq_vm)
        y_scores = det.decision_scores(X_sq_te)
        y_eval   = det._create_sequence_labels(y_sq_te)
        n = min(len(y_scores), len(y_eval))
        y_pred = (y_scores[:n] >= det.threshold).astype(int)
        met = evaluate(y_eval[:n], y_pred, y_scores[:n])
        train_n = len(X_sq_tr)
        split_type = "contiguous_normal"
    else:
        X_train_use = subsample_if_heavy(X_n, model_name, seed)
        det.train(X_train_use, None, X_v, y_v)
        y_pred   = det.predict(X_te)
        y_scores = det.decision_scores(X_te)
        met = evaluate(y_te, y_pred, y_scores)
        train_n = len(X_train_use)
        split_type = "random"

    met.update({
        "model": model_name,
        "seed": int(seed),
        "attack": "none",
        "poison_rate": 0.0,
        "time": time.time() - t0,
        "train_normal_size": int(train_n),
        "split_type": split_type,
        "threshold": float(det.threshold) if det.threshold is not None else None,
        "n_features": int(input_dim),
    })

    del det
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, required=True,
                    help="SLURM_ARRAY_TASK_ID. 0 <= id < 36.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if an existing result JSON is found.")
    args = ap.parse_args()

    model_name, seed = decode_clean_task(args.task_id)
    out_dir = get_output_dir()
    out_file = out_dir / "checkpoints" / "clean" / f"{model_name}__{seed}.json"

    print(f"[clean task {args.task_id}] model={model_name} seed={seed}", flush=True)
    print(f"[clean task {args.task_id}] output: {out_file}", flush=True)

    if out_file.exists() and not args.force:
        print(f"[clean task {args.task_id}] already done — skipping.", flush=True)
        return 0

    try:
        result = run_one_clean(model_name, seed)
        save_json_atomic(result, out_file)
        print(f"[clean task {args.task_id}] F1={result['f1']:.4f} "
              f"FNR={result['fnr']:.4f} t={result['time']:.1f}s -> saved", flush=True)
        return 0
    except Exception as e:
        err_file = out_file.with_suffix(".error.json")
        save_json_atomic({
            "model": model_name, "seed": int(seed),
            "attack": "none", "poison_rate": 0.0,
            "error": str(e), "traceback": traceback.format_exc(),
            "f1": 0.0, "fnr": 1.0,
        }, err_file)
        print(f"[clean task {args.task_id}] FAILED: {e}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
