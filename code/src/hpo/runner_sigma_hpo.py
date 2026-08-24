"""
σ-ablation runner — re-runs the R01 σ-sweep design but with the HPO-winning
AE / LSTM-AE configurations.

One Slurm task = one (detector, HPO-winner config, σ, rate, seed) tuple from
the manifest. Each task:

  1. Loads the SWaT data (pointwise + contiguous splits).
  2. Trains the detector on the CLEAN training pool (baseline for ΔF1).
  3. Re-trains the detector on a feature-noise-poisoned training pool with
     σ and rate as specified in the manifest.
  4. Evaluates both runs on the same clean held-out test set.
  5. Writes one JSON per task with both blocks.

Key implementation note: `src/attacks.py::feature_noise_injection` reads its
σ from `CONFIG["NOISE_SIGMA"]` (a module-level dict). To override per-task
without editing attacks.py, we monkeypatch CONFIG before calling apply_poison.
This is safe in the single-task Slurm context.

Output JSON layout
------------------
    {
      "stage":          "sigma_hpo",
      "task_id":        17,
      "config":         {...full HPO winner config + σ + rate + seed...},
      "input_dim":      44,
      "n_train_normal": 276708,
      "results": {
        "clean":    {f1, fnr, ..., n_params},
        "poisoned": {... + poison_attack: "feature_noise",
                          poison_rate, noise_sigma,
                          effective_contamination, n_injected}
      },
      "delta_f1":  poisoned.f1 - clean.f1,
      "wallclock_s": ...
    }

Idempotent: skips tasks whose output JSON already exists unless --force.

Invocation
----------
    python -m src.hpo.runner_sigma_hpo \\
        --manifest $SWAT_OUTPUT_DIR/hpo/manifests/sigma_hpo.jsonl \\
        --task-id  $SLURM_ARRAY_TASK_ID
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
    from src.config import CONFIG, get_data_paths, get_output_dir
    from src.data import (create_lstm_ae_splits, create_splits, load_raw,
                          preprocess_swat)
    from src.eval_utils import evaluate, save_json_atomic, set_seed
    from src.hpo.configurable_models import (ConfigurableAEDetector,
                                              ConfigurableLSTMAEDetector)
else:
    from ..attacks import apply_poison
    from ..config import CONFIG, get_data_paths, get_output_dir
    from ..data import (create_lstm_ae_splits, create_splits, load_raw,
                        preprocess_swat)
    from ..eval_utils import evaluate, save_json_atomic, set_seed
    from .configurable_models import (ConfigurableAEDetector,
                                       ConfigurableLSTMAEDetector)

import numpy as np
import torch


def _read_config(manifest: Path, task_id: int) -> dict:
    with open(manifest) as f:
        for i, line in enumerate(f):
            if i == task_id:
                return json.loads(line)
    raise IndexError(f"task_id {task_id} out of range in {manifest}")


def _eval_ae(detector: ConfigurableAEDetector, X_te, y_te) -> dict:
    y_pred, scores = detector.predict(X_te)
    m = evaluate(y_te, y_pred, scores)
    m["threshold"]      = float(detector.threshold) if detector.threshold is not None else None
    m["train_time"]     = float(detector.train_time)
    m["stopped_epoch"]  = int(detector.stopped_epoch)
    m["best_val_loss"]  = float(detector.best_val_loss)
    m["n_params"]       = int(sum(p.numel() for p in detector.model.parameters()))
    return m


def _eval_lstm(detector: ConfigurableLSTMAEDetector, X_te, y_te) -> dict:
    y_pred, scores = detector.predict(X_te)
    y_eval = detector._seq_labels(y_te)
    n = min(len(y_pred), len(y_eval))
    m = evaluate(y_eval[:n], y_pred[:n], scores[:n])
    m["threshold"]      = float(detector.threshold) if detector.threshold is not None else None
    m["train_time"]     = float(detector.train_time)
    m["stopped_epoch"]  = int(detector.stopped_epoch)
    m["best_val_loss"]  = float(detector.best_val_loss)
    m["n_params"]       = int(sum(p.numel() for p in detector.model.parameters()))
    m["window"]         = int(detector.window)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--task-id",  required=True, type=int)
    ap.add_argument("--force",    action="store_true")
    args = ap.parse_args()

    cfg = _read_config(args.manifest, args.task_id)
    detector_kind = cfg["detector"]
    sigma         = float(cfg["noise_sigma"])
    rate          = float(cfg["poison_rate"])
    seed          = int(cfg["seed"])

    out_root = get_output_dir() / "hpo" / "results" / "sigma_hpo"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{args.task_id:04d}.json"

    if out_path.exists() and not args.force:
        print(f"[sigma_hpo t={args.task_id}] already done — skipping.", flush=True)
        return 0

    print(f"[sigma_hpo t={args.task_id}] config={json.dumps(cfg)}", flush=True)
    t0 = time.time()

    # Load data
    _, normal_csv, attack_csv = get_data_paths()
    df_n_raw, df_a_raw = load_raw(normal_csv, attack_csv)
    df, feats, df_n_clean, df_a_clean = preprocess_swat(df_n_raw, df_a_raw)
    set_seed(seed)

    # Pointwise split (used for the AE and as the attack pool for both)
    X_tr_pw, X_n_pw, X_v_pw, X_te_pw, y_tr_pw, y_v_pw, y_te_pw, _ = create_splits(
        df, feats, seed=seed)

    record: dict = {
        "stage":           "sigma_hpo",
        "task_id":         args.task_id,
        "config":          cfg,
        "input_dim":       int(len(feats)),
    }

    try:
        # ── Build clean and poisoned training pools ──
        # Critical: monkey-patch CONFIG["NOISE_SIGMA"] so feature_noise uses
        # this task's σ instead of the global default (0.15).
        original_sigma = CONFIG.get("NOISE_SIGMA")
        CONFIG["NOISE_SIGMA"] = sigma

        try:
            if detector_kind == "ae":
                # Clean training
                det = ConfigurableAEDetector(len(feats), cfg)
                det.fit(X_n_pw, X_v_pw, y_v_pw)
                clean_block = _eval_ae(det, X_te_pw, y_te_pw)
                record["n_train_normal"] = int(len(X_n_pw))
                del det
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Poisoned (feature_noise) training
                X_pois, info = apply_poison(X_n_pw, X_tr_pw, y_tr_pw,
                                            "feature_noise", rate, seed)
                det = ConfigurableAEDetector(len(feats), cfg)
                det.fit(X_pois, X_v_pw, y_v_pw)
                pois_block = _eval_ae(det, X_te_pw, y_te_pw)

            elif detector_kind == "lstm_ae":
                # Contiguous LSTM-AE split
                (X_sq_tr, X_sq_vn, X_sq_vm, y_sq_vm,
                 X_sq_te, y_sq_te, _) = create_lstm_ae_splits(
                    df_n_clean, df_a_clean, feats, seed=seed)

                det = ConfigurableLSTMAEDetector(len(feats), cfg)
                det.fit(X_sq_tr, X_sq_vn, X_sq_vm, y_sq_vm)
                clean_block = _eval_lstm(det, X_sq_te, y_sq_te)
                record["n_train_normal"] = int(len(X_sq_tr))
                del det
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                X_pois, info = apply_poison(X_sq_tr, X_tr_pw, y_tr_pw,
                                            "feature_noise", rate, seed)
                det = ConfigurableLSTMAEDetector(len(feats), cfg)
                det.fit(X_pois, X_sq_vn, X_sq_vm, y_sq_vm)
                pois_block = _eval_lstm(det, X_sq_te, y_sq_te)

            else:
                raise ValueError(f"Unknown detector kind: {detector_kind}")
        finally:
            # Restore the global σ so other code paths keep their default
            CONFIG["NOISE_SIGMA"] = original_sigma

        pois_block["poison_attack"]            = "feature_noise"
        pois_block["poison_rate"]              = rate
        pois_block["noise_sigma"]              = sigma
        pois_block["effective_contamination"]  = float(info.get("effective_contamination", 0.0))
        pois_block["n_injected"]               = int(info.get("n_injected") or info.get("n_poisoned", 0))

        record["results"] = {"clean": clean_block, "poisoned": pois_block}
        record["delta_f1"]    = pois_block["f1"] - clean_block["f1"]
        record["wallclock_s"] = time.time() - t0
        save_json_atomic(record, out_path)
        print(f"[sigma_hpo t={args.task_id}] done in {record['wallclock_s']:.1f}s "
              f"clean_F1={clean_block['f1']:.4f} pois_F1={pois_block['f1']:.4f} "
              f"ΔF1={record['delta_f1']:+.4f}  σ={sigma}  rate={rate}",
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
        print(f"[sigma_hpo t={args.task_id}] FAILED: {e}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
