"""
Seeding, evaluation metrics, F1-optimal threshold search, and an atomic CSV writer.

Logic mirrors cells 1.5 and 3.1 of the notebook exactly.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


# ────────────────────────── reproducibility ─────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ───────────────────────── atomic writers ───────────────────────────
def safe_to_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Atomic CSV write: write to .tmp then os.replace → no partial files."""
    path = str(path)
    tmp = path + ".tmp"
    df.to_csv(tmp, **kwargs)
    os.replace(tmp, path)


def save_json_atomic(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o: Any):
    # numpy scalars → python scalars, numpy arrays → lists
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


# ────────────────────────── metrics ─────────────────────────────────
def evaluate(y_true, y_pred, y_scores=None) -> dict:
    r = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    r["tp"] = int(tp); r["tn"] = int(tn); r["fp"] = int(fp); r["fn"] = int(fn)
    r["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    r["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    if y_scores is not None:
        try:
            r["roc_auc"] = roc_auc_score(y_true, y_scores)
        except Exception:
            r["roc_auc"] = 0.5
        try:
            r["pr_auc"] = average_precision_score(y_true, y_scores)
        except Exception:
            r["pr_auc"] = 0.0
    return r


def find_optimal_threshold(y_val, scores_val) -> float:
    """
    Two-pass F1-optimal threshold search — identical to notebook cell 3.1.
    Coarse percentile scan, then fine search over unique score values.
    """
    if len(np.unique(y_val)) < 2:
        return float(np.median(scores_val))

    # Coarse
    best_f1 = -1.0
    best_t = float(np.median(scores_val))
    for q in np.linspace(1, 99, 99):
        t = float(np.percentile(scores_val, q))
        f1 = f1_score(y_val, (scores_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    # Fine — unique score values within the [1st, 99th] percentile band
    uniq = np.unique(scores_val)
    lo = np.percentile(scores_val, max(0, 1))
    hi = np.percentile(scores_val, min(100, 99))
    fine = uniq[(uniq >= lo) & (uniq <= hi)]
    if len(fine) > 200:
        fine = fine[::len(fine) // 200]
    for t in fine:
        f1 = f1_score(y_val, (scores_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t
