"""
Online / retraining poisoning runner — Phase 1.3 paper headline.

One Slurm array task = one (detector, generator, T, delta_p, seed) combo.
Each task runs T+1 trainings (round 0 = clean baseline, then T poisoning
rounds) and writes one JSON per round so a mid-trajectory failure costs at
most that round's training.

Invocation on a Narval compute node:

    python -m src.run_online --task-id $SLURM_ARRAY_TASK_ID

Output JSONs:
    $SWAT_OUTPUT_DIR/checkpoints/online/{detector}__{generator}__T{T}__dp{dp}__{seed}__r{round}.json

The manifest mapping task_id → combo is built once by
    python scripts/make_online_manifest.py
and saved to $SWAT_OUTPUT_DIR/online_manifest.tsv. Re-run that script
whenever ONLINE_DETECTORS / ONLINE_GENERATORS / ONLINE_T_VALUES /
ONLINE_DELTA_P_VALUES / ONLINE_SEEDS in src/config.py change.

Design invariants (locked post-pilot, see the online-retraining design note §B/§F/§I):
  * No-attack-reuse: pre-rank the attack pool ONCE per (generator, seed),
    consume next k via cursor each round.
  * Heavy PyOD detectors keep the 50K-subsample cap each round.
    n_in_pool (true cumulative) AND n_in_train (post-subsample) are reported
    plus effective_in_subsample_p.
  * LSTM-AE injects poisons at the END of the contiguous-normal block to
    preserve temporal order.
  * cumulative_p is exact (total_injected / original_pool_size), not
    delta_p * round_index.
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.attacks_online import (
        compute_attack_ranking, expected_round_paths,
        round_inject_indices, trajectory_path,
    )
    from src.config import (
        CONFIG, get_data_paths, get_output_dir, online_dp_tag,
    )
    from src.data import (
        create_lstm_ae_splits, create_splits, load_raw,
        preprocess_swat, subsample_if_heavy,
    )
    from src.eval_utils import evaluate, save_json_atomic, set_seed
    from src.models import TORCH_MODELS, create_model
else:
    from .attacks_online import (
        compute_attack_ranking, expected_round_paths,
        round_inject_indices, trajectory_path,
    )
    from .config import (
        CONFIG, get_data_paths, get_output_dir, online_dp_tag,
    )
    from .data import (
        create_lstm_ae_splits, create_splits, load_raw,
        preprocess_swat, subsample_if_heavy,
    )
    from .eval_utils import evaluate, save_json_atomic, set_seed
    from .models import TORCH_MODELS, create_model

import numpy as np
import torch
from sklearn.model_selection import train_test_split


# ────────────────────────── manifest loading ────────────────────────
def manifest_path() -> Path:
    return get_output_dir() / "online_manifest.tsv"


def load_combo(task_id: int) -> dict:
    mp = manifest_path()
    if not mp.exists():
        raise FileNotFoundError(
            f"Manifest not found at {mp}. "
            f"Run `python scripts/make_online_manifest.py` first."
        )
    with open(mp, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if not (0 <= task_id < len(rows)):
        raise IndexError(
            f"task_id {task_id} out of range 0..{len(rows) - 1} for manifest of {len(rows)} rows."
        )
    row = rows[task_id]
    return {
        "detector":  row["detector"],
        "generator": row["generator"],
        "T":         int(row["T"]),
        "delta_p":   float(row["delta_p"]),
        "seed":      int(row["seed"]),
    }


# ─────────────────────── per-round execution ────────────────────────
def _record_common(detector: str, generator: str, T: int, delta_p: float,
                   seed: int, round_idx: int) -> dict:
    return {
        "model":          detector,
        "seed":           int(seed),
        "attack":         f"online_retraining_{generator}",
        "generator":      generator,
        "T":              int(T),
        "delta_p":        float(delta_p),
        "round":          int(round_idx),
    }


def _run_pointwise_round(
    detector: str, seed: int, X_train_normal_round: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    input_dim: int,
    original_pool_size: int,
) -> tuple[dict, float]:
    """Train one pointwise detector on the round's poisoned pool, evaluate on the held-out test split."""
    set_seed(seed)
    # Heavy PyOD subsampling — same policy as run_attack.py / run_clean.py.
    X_train_use, train_idx = subsample_if_heavy(
        X_train_normal_round, detector, seed, return_indices=True
    )
    n_in_train = int(len(X_train_use))
    n_poison_in_train = int(np.sum(train_idx >= original_pool_size))

    det = create_model(detector, input_dim, seed=seed)
    det.train(X_train_use, None, X_val, y_val)

    y_pred   = det.predict(X_test)
    y_scores = det.decision_scores(X_test)
    met = evaluate(y_test, y_pred, y_scores)

    threshold = float(det.threshold) if det.threshold is not None else None
    del det
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    met["threshold"] = threshold
    met["n_in_train"] = n_in_train
    met["n_poison_in_train"] = n_poison_in_train
    return met, threshold or float("nan")


def _run_lstm_ae_round(
    seed: int, X_sq_tr_round: np.ndarray, X_sq_vn: np.ndarray,
    X_sq_vm: np.ndarray, y_sq_vm: np.ndarray,
    X_sq_te: np.ndarray, y_sq_te: np.ndarray, input_dim: int,
) -> tuple[dict, float]:
    """Train LSTM-AE on the round's poisoned contiguous-normal block, evaluate on sequence-level test."""
    set_seed(seed)
    det = create_model("lstm_ae", input_dim, seed=seed)
    det.train(X_sq_tr_round, X_sq_vn, X_sq_vm, y_sq_vm)
    y_scores = det.decision_scores(X_sq_te)
    y_eval   = det._create_sequence_labels(y_sq_te)
    n = min(len(y_scores), len(y_eval))
    y_pred = (y_scores[:n] >= det.threshold).astype(int)
    met = evaluate(y_eval[:n], y_pred, y_scores[:n])
    threshold = float(det.threshold) if det.threshold is not None else None
    met["threshold"] = threshold
    met["n_in_train"] = int(len(X_sq_tr_round))
    del det
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return met, threshold or float("nan")


# ────────────────────────── trajectory loop ─────────────────────────
def run_trajectory(combo: dict) -> int:
    detector = combo["detector"]
    generator = combo["generator"]
    T = combo["T"]
    delta_p = combo["delta_p"]
    seed = combo["seed"]

    out_paths = expected_round_paths(detector, generator, T, delta_p, seed)
    if all(p.exists() for p in out_paths):
        print(f"[online] all {T+1} rounds already done for "
              f"{detector}/{generator}/T{T}/dp{delta_p}/seed{seed}", flush=True)
        return 0

    # ─── load + preprocess ──────────────────────────────────────────
    _, normal_csv, attack_csv = get_data_paths()
    df_n, df_a = load_raw(normal_csv, attack_csv)
    df, feature_cols, df_n_clean, df_a_clean = preprocess_swat(df_n, df_a)
    input_dim = len(feature_cols)

    # ─── splits ─────────────────────────────────────────────────────
    set_seed(seed)
    X_tr, X_n, X_v, X_te, y_tr, y_v, y_te, pointwise_scaler = create_splits(
        df, feature_cols, CONFIG["TEST_SIZE"], CONFIG["VAL_SIZE"], seed
    )
    # Recreate the training-fold row indices from create_splits() so LSTM-AE
    # can inject the same attack rows, but transformed with the LSTM scaler.
    idx_all = np.arange(len(df))
    y_all = df["label"].values
    idx_tmp, _, y_tmp, _ = train_test_split(
        idx_all, y_all, test_size=CONFIG["TEST_SIZE"],
        stratify=y_all, random_state=seed,
    )
    v_ratio = CONFIG["VAL_SIZE"] / (1 - CONFIG["TEST_SIZE"])
    idx_tr, _, y_idx_tr, _ = train_test_split(
        idx_tmp, y_tmp, test_size=v_ratio,
        stratify=y_tmp, random_state=seed,
    )
    is_sequence = (detector == "lstm_ae")
    if is_sequence:
        X_sq_tr, X_sq_vn, X_sq_vm, y_sq_vm, X_sq_te, y_sq_te, lstm_scaler = \
            create_lstm_ae_splits(df_n_clean, df_a_clean, feature_cols, seed)

    # ─── attack pool + ranking ─────────────────────────────────────
    X_atk_pool = X_tr[y_tr == 1]
    X_inject_pool = X_atk_pool
    if is_sequence:
        # Rank raw attack rows under the clean-AE pointwise proxy, but inject
        # them in the LSTM-AE scaler's space so the appended rows match X_sq_tr.
        raw_attack = df.iloc[idx_tr[y_idx_tr == 1]][feature_cols].values
        X_atk_pool = pointwise_scaler.transform(raw_attack)
        X_inject_pool = lstm_scaler.transform(raw_attack)
    if len(X_atk_pool) == 0:
        raise RuntimeError(f"No attack samples in train pool for seed {seed}.")
    ranked_order = compute_attack_ranking(
        generator=generator,
        X_atk_pool=X_atk_pool,
        X_train_normal=X_n,
        X_val=X_v, y_val=y_v,
        seed=seed,
    )

    # ─── original pool size for cumulative_p ───────────────────────
    if is_sequence:
        original_pool_size = int(len(X_sq_tr))
        cumulative_X = X_sq_tr.copy()
    else:
        original_pool_size = int(len(X_n))
        cumulative_X = X_n.copy()

    k_per_round = max(1, int(round(delta_p * original_pool_size)))

    print(f"[online] {detector}/{generator}/T{T}/dp{delta_p}/seed{seed} "
          f"| input_dim={input_dim} pool0={original_pool_size} "
          f"k_per_round={k_per_round} attack_pool={len(X_atk_pool)} "
          f"is_sequence={is_sequence}", flush=True)

    # ─── round loop ────────────────────────────────────────────────
    cursor = 0
    total_injected = 0
    overall_rc = 0
    for r in range(T + 1):
        out_file = out_paths[r]
        if out_file.exists():
            print(f"[online] round {r} already done -> {out_file.name}", flush=True)
            # We still need to advance the cursor / cumulative_X consistently
            # so subsequent rounds inject the right slice. Re-derive injected
            # count from the saved JSON.
            try:
                import json
                prev = json.loads(out_file.read_text())
                cursor = int(prev.get("unique_attacks_used", cursor))
                total_injected = int(prev.get("total_injected", total_injected))
                if r > 0:
                    # Reapply the appended slice so cumulative_X stays correct.
                    n_already = total_injected
                    if n_already > 0:
                        appended = X_inject_pool[ranked_order[:n_already]]
                        if is_sequence:
                            cumulative_X = np.vstack([X_sq_tr, appended])
                        else:
                            cumulative_X = np.vstack([X_n, appended])
            except Exception as e:
                print(f"[online] WARN: failed to replay round-{r} state ({e}); "
                      f"future rounds may diverge. Delete the file to force fresh.", flush=True)
            continue

        t0 = time.time()
        try:
            # Inject this round's slice (rounds >= 1)
            n_this_round = 0
            if r >= 1:
                idxs, cursor, n_this_round = round_inject_indices(
                    ranked_order, cursor, k_per_round
                )
                if n_this_round > 0:
                    new_poisons = X_inject_pool[idxs]
                    cumulative_X = np.vstack([cumulative_X, new_poisons])
                    total_injected += n_this_round

            cumulative_p = total_injected / original_pool_size

            # Train + evaluate
            if is_sequence:
                met, threshold = _run_lstm_ae_round(
                    seed, cumulative_X, X_sq_vn, X_sq_vm, y_sq_vm,
                    X_sq_te, y_sq_te, input_dim,
                )
                split_type = "contiguous_normal_appended"
            else:
                met, threshold = _run_pointwise_round(
                    detector, seed, cumulative_X, X_v, y_v, X_te, y_te, input_dim,
                    original_pool_size,
                )
                split_type = "random"

            n_in_pool = int(len(cumulative_X))
            n_in_train = int(met.get("n_in_train", n_in_pool))
            n_poison_in_train = int(met.get("n_poison_in_train", total_injected))
            effective_in_subsample_p = (
                n_poison_in_train / n_in_train if n_in_train > 0 else 0.0
            )

            row = _record_common(detector, generator, T, delta_p, seed, r)
            row.update(met)  # F1, FNR, precision, recall, tp/tn/fp/fn, roc_auc, pr_auc, threshold, n_in_train
            row.update({
                "poison_rate":            float(cumulative_p),
                "cumulative_p":           float(cumulative_p),
                "pool_size":              int(n_in_pool),
                "n_in_pool":              int(n_in_pool),
                "n_in_train":             int(n_in_train),
                "n_injected_this_round":  int(n_this_round),
                "n_poison_in_train":      int(n_poison_in_train),
                "total_injected":         int(total_injected),
                "unique_attacks_used":    int(cursor),
                "k_per_round":            int(k_per_round),
                "original_pool_size":     int(original_pool_size),
                "effective_in_subsample_p": float(effective_in_subsample_p),
                "split_type":             split_type,
                "n_features":             int(input_dim),
                "time":                   float(time.time() - t0),
            })

            save_json_atomic(row, out_file)
            print(
                f"[online] r={r} cum_p={cumulative_p:.4f} "
                f"pool={n_in_pool} train={n_in_train} "
                f"F1={row['f1']:.4f} FNR={row['fnr']:.4f} "
                f"t={row['time']:.1f}s -> saved",
                flush=True,
            )

        except Exception as e:
            err_file = out_file.with_suffix(".error.json")
            err = _record_common(detector, generator, T, delta_p, seed, r)
            err.update({
                "error":     str(e),
                "traceback": traceback.format_exc(),
                "f1": 0.0, "fnr": 1.0,
                "cumulative_p": float(total_injected / original_pool_size)
                                if original_pool_size > 0 else 0.0,
                "total_injected": int(total_injected),
                "unique_attacks_used": int(cursor),
                "time": float(time.time() - t0),
            })
            save_json_atomic(err, err_file)
            print(f"[online] r={r} FAILED: {e}", flush=True)
            traceback.print_exc()
            overall_rc = 1
            # Don't bail on the rest of the trajectory — later rounds may still
            # succeed (e.g. transient OOM on one round). cumulative_X is
            # consistent because we always advanced it before the train call.

    return overall_rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, required=True,
                    help="Index into $SWAT_OUTPUT_DIR/online_manifest.tsv "
                         "(usually $SLURM_ARRAY_TASK_ID).")
    args = ap.parse_args()

    combo = load_combo(args.task_id)
    print(f"[online task {args.task_id}] combo = {combo}", flush=True)
    return run_trajectory(combo)


if __name__ == "__main__":
    sys.exit(main())
