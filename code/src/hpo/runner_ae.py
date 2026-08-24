"""
Single-config AE runner for the HPO Slurm arrays.

One Slurm task = one row of the manifest. Each task:
  1. Loads the SWaT data + builds the pointwise split (using the manifest's
     scaler choice).
  2. Trains the AE on the clean normal training pool, evaluates on test.
  3. Re-trains the AE with the same config on a 10 %-targeted-poisoning
     contamination of the normal pool (or whatever attack/rate the manifest
     specifies), evaluates on the same test set.
  4. Writes a single JSON containing both result blocks.

The same input split (X_val, X_test, y_val, y_test) is reused across the two
runs so the only difference is the training pool. That gives us a clean
delta attributable to the poisoning.

Output JSON layout
------------------
    {
      "stage":          "ae_stage1",
      "task_id":        17,
      "config":         {...full config dict...},
      "input_dim":      44,
      "n_train_normal": 246612,
      "results": {
        "clean":    {f1, fnr, precision, recall, roc_auc, pr_auc, accuracy,
                     tp, tn, fp, fn, threshold, train_time, stopped_epoch,
                     best_val_loss, n_params},
        "poisoned": {... + poison_attack, poison_rate,
                          effective_contamination, n_injected}
      },
      "composite_f1":  mean(clean.f1, poisoned.f1),
      "delta_f1":      poisoned.f1 - clean.f1
    }

Idempotent: skips tasks whose output JSON already exists unless --force.

Invocation
----------
    python -m src.hpo.runner_ae \\
        --manifest $SWAT_OUTPUT_DIR/hpo/manifests/ae_stage1.jsonl \\
        --task-id  $SLURM_ARRAY_TASK_ID \\
        --stage    ae_stage1
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
    from src.data import load_raw, preprocess_swat
    from src.eval_utils import evaluate, save_json_atomic, set_seed
    from src.hpo.configurable_models import ConfigurableAEDetector
else:
    from ..attacks import apply_poison
    from ..config import get_data_paths, get_output_dir
    from ..data import load_raw, preprocess_swat
    from ..eval_utils import evaluate, save_json_atomic, set_seed
    from .configurable_models import ConfigurableAEDetector

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


_SCALERS = {
    "minmax":   MinMaxScaler,
    "standard": StandardScaler,
    "robust":   RobustScaler,
}


def _make_split(df, feats, scaler_name: str, seed: int,
                test_size: float = 0.15, val_size: float = 0.15):
    """
    Pointwise stratified split with a configurable scaler. Mirrors
    src.data.create_splits but lets the manifest pick the scaler.
    """
    X = df[feats].values
    y = df["label"].values

    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    v_ratio = val_size / (1 - test_size)
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_tmp, y_tmp, test_size=v_ratio, stratify=y_tmp, random_state=seed)

    sc = _SCALERS[scaler_name]()
    X_tr = sc.fit_transform(X_tr)
    X_v  = sc.transform(X_v)
    X_te = sc.transform(X_te)
    X_tr_normal = X_tr[y_tr == 0]
    return X_tr, X_tr_normal, X_v, X_te, y_tr, y_v, y_te


def _read_config(manifest: Path, task_id: int) -> dict:
    with open(manifest) as f:
        for i, line in enumerate(f):
            if i == task_id:
                return json.loads(line)
    raise IndexError(f"task_id {task_id} out of range in {manifest}")


def _eval_one(detector: ConfigurableAEDetector, X_te, y_te) -> dict:
    y_pred, scores = detector.predict(X_te)
    m = evaluate(y_te, y_pred, scores)
    m["threshold"]      = float(detector.threshold) if detector.threshold is not None else None
    m["train_time"]     = float(detector.train_time)
    m["stopped_epoch"]  = int(detector.stopped_epoch)
    m["best_val_loss"]  = float(detector.best_val_loss)
    m["n_params"]       = int(sum(p.numel() for p in detector.model.parameters()))
    return m


def _run_full_grid(cfg: dict, X_tr, X_n, X_v, X_te, y_tr, y_v, y_te,
                   final_attacks, final_rates) -> dict:
    """
    For final-stage tasks: clean + (3 attacks × 4 rates) = 13 evaluations
    against the same test set. Returns a dict keyed by condition.
    """
    input_dim = X_n.shape[1]
    results = {}

    # Clean
    det = ConfigurableAEDetector(input_dim, cfg)
    det.fit(X_n, X_v, y_v)
    results["clean"] = _eval_one(det, X_te, y_te)
    del det
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Attack × rate grid
    for atk in final_attacks:
        for r in final_rates:
            X_pois, info = apply_poison(X_n, X_tr, y_tr, atk, r, cfg["seed"])
            det = ConfigurableAEDetector(input_dim, cfg)
            det.fit(X_pois, X_v, y_v)
            block = _eval_one(det, X_te, y_te)
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
                    choices=["ae_stage1", "ae_stage2", "ae_stage3", "ae_final"])
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

    # Data + split
    _, normal_csv, attack_csv = get_data_paths()
    df_n_raw, df_a_raw = load_raw(normal_csv, attack_csv)
    df, feats, _, _ = preprocess_swat(df_n_raw, df_a_raw)
    set_seed(int(cfg["seed"]))
    X_tr, X_n, X_v, X_te, y_tr, y_v, y_te = _make_split(
        df, feats, cfg.get("scaler", "minmax"), int(cfg["seed"]))
    input_dim = X_n.shape[1]
    print(f"[{args.stage} t={args.task_id}] data ready: "
          f"normal_pool={len(X_n)} val={len(X_v)} test={len(X_te)} "
          f"input_dim={input_dim}", flush=True)

    record: dict = {
        "stage":           args.stage,
        "task_id":         args.task_id,
        "config":          cfg,
        "input_dim":       int(input_dim),
        "n_train_normal":  int(len(X_n)),
    }

    try:
        if args.stage == "ae_final":
            from .grids import FINAL_ATTACKS, FINAL_RATES  # local import
            results = _run_full_grid(cfg, X_tr, X_n, X_v, X_te, y_tr, y_v, y_te,
                                     FINAL_ATTACKS, FINAL_RATES)
            record["results"] = results
            # Composite uses targeted_flip @ 0.10 as the "poisoned" anchor
            anchor = results.get("targeted_flip__r0.10", {}).get("f1", 0.0)
            clean_f1 = results.get("clean", {}).get("f1", 0.0)
            record["composite_f1"] = (clean_f1 + anchor) / 2.0
            record["delta_f1"]     = anchor - clean_f1

        else:
            # Stages 1-3: clean + single poisoned condition (manifest-specified)
            atk  = cfg["poison_attack"]
            rate = float(cfg["poison_rate"])

            det = ConfigurableAEDetector(input_dim, cfg)
            det.fit(X_n, X_v, y_v)
            clean_block = _eval_one(det, X_te, y_te)
            del det
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            X_pois, info = apply_poison(X_n, X_tr, y_tr, atk, rate, int(cfg["seed"]))
            det = ConfigurableAEDetector(input_dim, cfg)
            det.fit(X_pois, X_v, y_v)
            poisoned_block = _eval_one(det, X_te, y_te)
            poisoned_block["poison_attack"]            = atk
            poisoned_block["poison_rate"]              = rate
            poisoned_block["effective_contamination"]  = float(info.get("effective_contamination", 0.0))
            poisoned_block["n_injected"]               = int(info.get("n_injected") or info.get("n_poisoned", 0))

            record["results"] = {"clean": clean_block, "poisoned": poisoned_block}
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
