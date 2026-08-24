"""
SWaT data loading, preprocessing, and dual-pipeline split creation.

Pointwise pipeline (10 PyOD + Autoencoder):
    random stratified 70/15/15 split of combined normal.csv + attack.csv,
    MinMax scaler fit on the train fold.

LSTM-AE pipeline (paper v2 full-data contiguous split):
    train      = first 70% of normal.csv           (contiguous reconstruction training)
    val-norm   = next 15% of normal.csv            (contiguous early-stopping loss)
    val-mixed  = first 50% of attack.csv           (threshold calibration)
    test       = last 15% of normal.csv + last 50% of attack.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .config import CONFIG


def load_raw(normal_csv: Path, attack_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_normal = pd.read_csv(normal_csv)
    df_attack = pd.read_csv(attack_csv)
    return df_normal, df_attack


def preprocess_swat(df_normal: pd.DataFrame, df_attack: pd.DataFrame
                    ) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    """
    Identical to notebook cell 2.2 (R02 / v16_2). Returns:
        df_combined    — pooled normal+attack DataFrame with 'label' column
        feature_cols   — list of feature column names
        df_normal_clean, df_attack_clean — per-file cleaned DataFrames for the LSTM-AE contiguous split
    """
    # 1) Identify label column in attack file
    label_col = None
    for col in df_attack.columns:
        if "normal" in col.lower() or "attack" in col.lower() or "label" in col.lower():
            label_col = col
            break

    # 2) Binary labels
    df_n = df_normal.copy()
    df_n["label"] = 0
    if label_col and label_col in df_n.columns:
        df_n = df_n.drop(columns=[label_col])

    df_a = df_attack.copy()
    if label_col:
        df_a["label"] = df_a[label_col].apply(lambda x: 1 if "attack" in str(x).lower() else 0)
        df_a = df_a.drop(columns=[label_col])
    else:
        df_a["label"] = 1

    # 3) Drop timestamp columns
    drop_cols = [c for c in df_n.columns
                 if "timestamp" in c.lower() or c.lower().strip() == "time"]
    df_n = df_n.drop(columns=[c for c in drop_cols if c in df_n.columns], errors="ignore")
    df_a = df_a.drop(columns=[c for c in drop_cols if c in df_a.columns], errors="ignore")

    # 4) Align columns by intersection
    common = sorted(list(set(df_n.columns) & set(df_a.columns)))
    df_n = df_n[common]
    df_a = df_a[common]

    # 5) Combine
    df = pd.concat([df_n, df_a], ignore_index=True)

    # 6) Numeric coercion + NaN drop
    feats = [c for c in df.columns if c != "label"]
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    # 7) Drop constant columns
    const = [c for c in feats if c in df.columns and df[c].nunique() <= 1]
    if const:
        df = df.drop(columns=const)
    feats = [c for c in df.columns if c != "label"]

    # 8) Per-file cleaned frames for LSTM-AE contiguous split
    df_n_final = df_n.copy()
    df_a_final = df_a.copy()
    for c in feats:
        df_n_final[c] = pd.to_numeric(df_n_final[c], errors="coerce")
        df_a_final[c] = pd.to_numeric(df_a_final[c], errors="coerce")
    df_n_final = df_n_final.dropna()
    df_a_final = df_a_final.dropna()
    if const:
        df_n_final = df_n_final.drop(columns=[c for c in const if c in df_n_final.columns],
                                     errors="ignore")
        df_a_final = df_a_final.drop(columns=[c for c in const if c in df_a_final.columns],
                                     errors="ignore")

    return df, feats, df_n_final, df_a_final


def create_splits(df: pd.DataFrame, feats: list[str],
                  test_size: float = CONFIG["TEST_SIZE"],
                  val_size: float = CONFIG["VAL_SIZE"],
                  seed: int = 42):
    """
    Pointwise split — returns (X_tr, X_tr_normal, X_val, X_test, y_tr, y_val, y_test, scaler).
    """
    X = df[feats].values
    y = df["label"].values
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    v_ratio = val_size / (1 - test_size)
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_tmp, y_tmp, test_size=v_ratio, stratify=y_tmp, random_state=seed)
    sc = MinMaxScaler()
    X_tr = sc.fit_transform(X_tr)
    X_v  = sc.transform(X_v)
    X_te = sc.transform(X_te)
    X_tr_normal = X_tr[y_tr == 0]
    return X_tr, X_tr_normal, X_v, X_te, y_tr, y_v, y_te, sc


def create_lstm_ae_splits(df_n: pd.DataFrame, df_a: pd.DataFrame, feats: list[str],
                          seed: int = 42):
    """
    Full-data contiguous split for the LSTM-AE.
    Returns (X_tr, X_val_norm, X_val_mixed, y_val_mixed, X_test, y_test, scaler).
    """
    Xn = df_n[feats].values
    Xa = df_a[feats].values
    ya = df_a["label"].values

    n = len(Xn)
    i_tr = int(n * 0.70)
    i_vn = int(n * 0.85)
    tr     = Xn[:i_tr]
    vn     = Xn[i_tr:i_vn]
    n_tail = Xn[i_vn:]

    mid   = len(Xa) // 2
    vm    = Xa[:mid];     yvm   = ya[:mid]
    a_te  = Xa[mid:];     y_ate = ya[mid:]

    te  = np.vstack([n_tail, a_te])
    yte = np.concatenate([np.zeros(len(n_tail), dtype=int), y_ate])

    sc = MinMaxScaler().fit(tr)
    tr = sc.transform(tr)
    vn = sc.transform(vn)
    vm = sc.transform(vm)
    te = sc.transform(te)
    return tr, vn, vm, yvm, te, yte, sc


def subsample_if_heavy(
    X: np.ndarray,
    model_name: str,
    seed: int = 42,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if model_name in CONFIG["HEAVY_MODELS"] and len(X) > CONFIG["MAX_TRAIN_SAMPLES"]:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), CONFIG["MAX_TRAIN_SAMPLES"], replace=False)
        return (X[idx], idx) if return_indices else X[idx]
    idx = np.arange(len(X))
    return (X, idx) if return_indices else X
