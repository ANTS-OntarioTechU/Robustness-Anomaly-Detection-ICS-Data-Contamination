"""
Attack-grid runner — one (model, seed) × all 4 poison rates per invocation.

Invocation on a Slurm compute node:

    python -m src.run_attack --task-id $SLURM_ARRAY_TASK_ID --attack random_flip

Output: $SWAT_OUTPUT_DIR/checkpoints/attacks/{attack}/{model}__{seed}__{rate}.json
Each rate is written atomically as soon as it finishes, so a mid-grid abort
costs at most one rate's worth of compute.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.attacks import apply_poison
    from src.config import CONFIG, decode_attack_task, get_data_paths, get_output_dir
    from src.data import (create_lstm_ae_splits, create_splits, load_raw,
                          preprocess_swat, subsample_if_heavy)
    from src.eval_utils import evaluate, save_json_atomic, set_seed
    from src.models import create_model
else:
    from .attacks import apply_poison
    from .config import CONFIG, decode_attack_task, get_data_paths, get_output_dir
    from .data import (create_lstm_ae_splits, create_splits, load_raw,
                       preprocess_swat, subsample_if_heavy)
    from .eval_utils import evaluate, save_json_atomic, set_seed
    from .models import create_model

import torch


def _rate_tag(rate: float) -> str:
    # "0.01" -> "r0p01" for filename safety
    return "r" + f"{rate:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def run_one_combo(model_name: str, seed: int, attack: str, rate: float,
                  df, feature_cols, df_n_clean, df_a_clean,
                  X_tr, X_n, X_v, X_te, y_tr, y_v, y_te,
                  lstm_sp) -> dict:
    input_dim = len(feature_cols)
    t0 = time.time()
    set_seed(seed)
    det = create_model(model_name, input_dim, seed=seed)

    if model_name == "lstm_ae":
        X_sq_tr, X_sq_vn, X_sq_vm, y_sq_vm, X_sq_te, y_sq_te, _ = lstm_sp
        # Poison the contiguous-normal training block
        X_poisoned, info = apply_poison(X_sq_tr, X_tr, y_tr, attack, rate, seed)
        det.train(X_poisoned, X_sq_vn, X_sq_vm, y_sq_vm)
        y_scores = det.decision_scores(X_sq_te)
        y_eval   = det._create_sequence_labels(y_sq_te)
        n = min(len(y_scores), len(y_eval))
        y_pred = (y_scores[:n] >= det.threshold).astype(int)
        met = evaluate(y_eval[:n], y_pred, y_scores[:n])
        split_type = "contiguous_normal"
    else:
        X_poisoned, info = apply_poison(X_n, X_tr, y_tr, attack, rate, seed)
        X_train_use = subsample_if_heavy(X_poisoned, model_name, seed)
        det.train(X_train_use, None, X_v, y_v)
        y_pred   = det.predict(X_te)
        y_scores = det.decision_scores(X_te)
        met = evaluate(y_te, y_pred, y_scores)
        split_type = "random"

    met.update({
        "model": model_name,
        "seed": int(seed),
        "attack": attack,
        "poison_rate": float(rate),
        "time": time.time() - t0,
        "split_type": split_type,
        "effective_contamination": info.get("effective_contamination"),
        "n_injected": info.get("n_injected") or info.get("n_poisoned"),
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
    ap.add_argument("--attack", type=str, required=True,
                    choices=CONFIG["ATTACKS"],
                    help="Poisoning attack type.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if existing result JSONs are found.")
    args = ap.parse_args()

    model_name, seed = decode_attack_task(args.task_id)
    out_dir = get_output_dir()
    attack_dir = out_dir / "checkpoints" / "attacks" / args.attack
    attack_dir.mkdir(parents=True, exist_ok=True)

    # Figure out which rates still need to run
    remaining = []
    for rate in CONFIG["POISON_RATES"]:
        out_file = attack_dir / f"{model_name}__{seed}__{_rate_tag(rate)}.json"
        if out_file.exists() and not args.force:
            print(f"[{args.attack} task {args.task_id}] rate={rate} already done", flush=True)
            continue
        remaining.append((rate, out_file))

    if not remaining:
        print(f"[{args.attack} task {args.task_id}] all rates done — nothing to do.", flush=True)
        return 0

    print(f"[{args.attack} task {args.task_id}] model={model_name} seed={seed} "
          f"rates_to_run={[r for r, _ in remaining]}", flush=True)

    # Load & preprocess ONCE per (model, seed) — splits are reused across rates
    _, normal_csv, attack_csv = get_data_paths()
    df_n, df_a = load_raw(normal_csv, attack_csv)
    df, feature_cols, df_n_clean, df_a_clean = preprocess_swat(df_n, df_a)

    set_seed(seed)
    X_tr, X_n, X_v, X_te, y_tr, y_v, y_te, _ = create_splits(
        df, feature_cols, CONFIG["TEST_SIZE"], CONFIG["VAL_SIZE"], seed)
    lstm_sp = create_lstm_ae_splits(df_n_clean, df_a_clean, feature_cols, seed)

    overall_rc = 0
    for rate, out_file in remaining:
        try:
            result = run_one_combo(
                model_name, seed, args.attack, rate,
                df, feature_cols, df_n_clean, df_a_clean,
                X_tr, X_n, X_v, X_te, y_tr, y_v, y_te, lstm_sp,
            )
            save_json_atomic(result, out_file)
            print(f"[{args.attack} task {args.task_id}] rate={rate} "
                  f"F1={result['f1']:.4f} FNR={result['fnr']:.4f} "
                  f"t={result['time']:.1f}s -> saved", flush=True)
        except Exception as e:
            err_file = out_file.with_suffix(".error.json")
            save_json_atomic({
                "model": model_name, "seed": int(seed),
                "attack": args.attack, "poison_rate": float(rate),
                "error": str(e), "traceback": traceback.format_exc(),
                "f1": 0.0, "fnr": 1.0,
            }, err_file)
            print(f"[{args.attack} task {args.task_id}] rate={rate} FAILED: {e}", flush=True)
            traceback.print_exc()
            overall_rc = 1

    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
