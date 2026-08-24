"""
Single-config LSTM-AE runner for the HPO Slurm arrays.

Mirrors runner_ae.py but operates on the contiguous-time-series split from
src.data.create_lstm_ae_splits. Poisoning is applied to the contiguous
training block by drawing attack samples from the *pointwise* split's
training fold (so the same attack pool feeds both detector types and
attack rates are directly comparable).

Output JSON layout: identical to runner_ae.py (clean + poisoned blocks
under `results`, plus composite_f1 and delta_f1).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.attacks import apply_poison
    from src.config import get_data_paths, get_output_dir
    from src.data import create_lstm_ae_splits, create_splits, load_raw, preprocess_swat
    from src.eval_utils import evaluate, save_json_atomic, set_seed
    from src.hpo.configurable_models import ConfigurableLSTMAEDetector
else:
    from ..attacks import apply_poison
    from ..config import get_data_paths, get_output_dir
    from ..data import create_lstm_ae_splits, create_splits, load_raw, preprocess_swat
    from ..eval_utils import evaluate, save_json_atomic, set_seed
    from .configurable_models import ConfigurableLSTMAEDetector

import numpy as np
import torch


def _read_config(manifest: Path, task_id: int) -> dict:
    with open(manifest) as f:
        for i, line in enumerate(f):
            if i == task_id:
                return json.loads(line)
    raise IndexError(f"task_id {task_id} out of range in {manifest}")


def _eval_one(det: ConfigurableLSTMAEDetector, X_te, y_te) -> dict:
    y_pred, scores = det.predict(X_te)
    y_eval = det._seq_labels(y_te)
    n = min(len(y_pred), len(y_eval))
    m = evaluate(y_eval[:n], y_pred[:n], scores[:n])
    m["threshold"]      = float(det.threshold) if det.threshold is not None else None
    m["train_time"]     = float(det.train_time)
    m["stopped_epoch"]  = int(det.stopped_epoch)
    m["best_val_loss"]  = float(det.best_val_loss)
    m["n_params"]       = int(sum(p.numel() for p in det.model.parameters()))
    m["window"]         = int(det.window)
    return m


def _run_full_grid(cfg, X_tr_pw, y_tr_pw,
                   X_sq_tr, X_sq_vn, X_sq_vm, y_sq_vm, X_sq_te, y_sq_te,
                   final_attacks, final_rates) -> dict:
    input_dim = X_sq_tr.shape[1]
    results = {}

    # Clean
    det = ConfigurableLSTMAEDetector(input_dim, cfg)
    det.fit(X_sq_tr, X_sq_vn, X_sq_vm, y_sq_vm)
    results["clean"] = _eval_one(det, X_sq_te, y_sq_te)
    del det
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    for atk in final_attacks:
        for r in final_rates:
            X_pois, info = apply_poison(X_sq_tr, X_tr_pw, y_tr_pw, atk, r, cfg["seed"])
            det = ConfigurableLSTMAEDetector(input_dim, cfg)
            det.fit(X_pois, X_sq_vn, X_sq_vm, y_sq_vm)
            block = _eval_one(det, X_sq_te, y_sq_te)
            block["poison_attack"]            = atk
            block["poison_rate"]              = float(r)
            block["effective_contamination"]  = float(info.get("effective_contamination", 0.0))
            block["n_injected"]               = int(info.get("n_injected") or info.get("n_poisoned", 0))
            results[f"{atk}__r{r:.2f}"] = block
            del det
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--task-id",  required=True, type=int)
    ap.add_argument("--stage",    required=True,
                    choices=["lstm_ae_stage1", "lstm_ae_stage2",
                             "lstm_ae_stage3", "lstm_ae_final"])
    ap.add_argument("--force",    action="store_true")
    args = ap.parse_args()

    cfg = _read_config(args.manifest, args.task_id)

    out_root = get_output_dir() / "hpo" / "results" / args.stage
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{args.task_id:04d}.json"

    if out_path.exists() and not args.force:
        print(f"[{args.stage} t={args.task_id}] already done — skipping.", flush=True)
        return 0

    print(f"[{args.stage} t={args.task_id}] config={json.dumps(cfg)}", flush=True)
    t0 = time.time()

    # Data + splits
    _, normal_csv, attack_csv = get_data_paths()
    df_n_raw, df_a_raw = load_raw(normal_csv, attack_csv)
    df, feats, df_n_clean, df_a_clean = preprocess_swat(df_n_raw, df_a_raw)

    set_seed(int(cfg["seed"]))
    # Pointwise split is needed so we can pull attack samples for poisoning
    # consistent with the main detector grid.
    X_tr_pw, _, _, _, y_tr_pw, _, _, _ = create_splits(df, feats, seed=int(cfg["seed"]))

    # Contiguous LSTM-AE split
    (X_sq_tr, X_sq_vn, X_sq_vm, y_sq_vm,
     X_sq_te, y_sq_te, _) = create_lstm_ae_splits(df_n_clean, df_a_clean, feats,
                                                  seed=int(cfg["seed"]))
    input_dim = X_sq_tr.shape[1]
    print(f"[{args.stage} t={args.task_id}] data ready: "
          f"contig_train={len(X_sq_tr)} val_norm={len(X_sq_vn)} "
          f"val_mixed={len(X_sq_vm)} test={len(X_sq_te)} "
          f"input_dim={input_dim}", flush=True)

    record = {
        "stage":           args.stage,
        "task_id":         args.task_id,
        "config":          cfg,
        "input_dim":       int(input_dim),
        "n_train_normal":  int(len(X_sq_tr)),
    }

    try:
        if args.stage == "lstm_ae_final":
            from .grids import FINAL_ATTACKS, FINAL_RATES
            results = _run_full_grid(cfg, X_tr_pw, y_tr_pw,
                                     X_sq_tr, X_sq_vn, X_sq_vm, y_sq_vm,
                                     X_sq_te, y_sq_te,
                                     FINAL_ATTACKS, FINAL_RATES)
            record["results"] = results
            anchor = results.get("targeted_flip__r0.10", {}).get("f1", 0.0)
            clean_f1 = results.get("clean", {}).get("f1", 0.0)
            record["composite_f1"] = (clean_f1 + anchor) / 2.0
            record["delta_f1"]     = anchor - clean_f1

        else:
            atk  = cfg["poison_attack"]
            rate = float(cfg["poison_rate"])

            # Clean
            det = ConfigurableLSTMAEDetector(input_dim, cfg)
            det.fit(X_sq_tr, X_sq_vn, X_sq_vm, y_sq_vm)
            clean_block = _eval_one(det, X_sq_te, y_sq_te)
            del det
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Poisoned
            X_pois, info = apply_poison(X_sq_tr, X_tr_pw, y_tr_pw,
                                        atk, rate, int(cfg["seed"]))
            det = ConfigurableLSTMAEDetector(input_dim, cfg)
            det.fit(X_pois, X_sq_vn, X_sq_vm, y_sq_vm)
            poisoned_block = _eval_one(det, X_sq_te, y_sq_te)
            poisoned_block["poison_attack"]            = atk
            poisoned_block["poison_rate"]              = rate
            poisoned_block["effective_contamination"]  = float(info.get("effective_contamination", 0.0))
            poisoned_block["n_injected"]               = int(info.get("n_injected") or info.get("n_poisoned", 0))

            record["results"]      = {"clean": clean_block, "poisoned": poisoned_block}
            record["composite_f1"] = (clean_block["f1"] + poisoned_block["f1"]) / 2.0
            record["delta_f1"]     = poisoned_block["f1"] - clean_block["f1"]

        record["wallclock_s"] = time.time() - t0
        save_json_atomic(record, out_path)
        cf1 = record.get("composite_f1", float("nan"))
        df1 = record.get("delta_f1", float("nan"))
        print(f"[{args.stage} t={args.task_id}] done in "
              f"{record['wallclock_s']:.1f}s  composite_f1={cf1:.4f}  ΔF1={df1:+.4f}",
              flush=True)
        return 0

    except Exception as e:
        err = {
            **record,
            "error":      str(e),
            "traceback":  traceback.format_exc(),
            "wallclock_s": time.time() - t0,
        }
        save_json_atomic(err, out_path.with_suffix(".error.json"))
        print(f"[{args.stage} t={args.task_id}] FAILED: {e}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
