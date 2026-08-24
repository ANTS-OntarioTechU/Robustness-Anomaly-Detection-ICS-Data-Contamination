"""
Poisoning attacks — identical to notebook cell 4.1.

All attacks return (X_poisoned_train_normal, info_dict) where the poisoned set
is what the detector actually sees as its "normal" training data.
"""
from __future__ import annotations

import numpy as np

from .config import CONFIG


def random_injection(X_train_normal, X_train_full, y_train_full, rate, seed=42):
    rng = np.random.RandomState(seed)
    X_atk = X_train_full[y_train_full == 1]
    if len(X_atk) == 0:
        return X_train_normal.copy(), {"n_injected": 0, "effective_contamination": 0.0}
    k = min(int(rate * len(X_train_normal)), len(X_atk))
    idx = rng.choice(len(X_atk), k, replace=False)
    X_new = np.vstack([X_train_normal, X_atk[idx]])
    return X_new, {"n_injected": k, "effective_contamination": k / len(X_new)}


def targeted_injection(X_train_normal, X_train_full, y_train_full, rate, seed=42):
    X_atk = X_train_full[y_train_full == 1]
    if len(X_atk) == 0:
        return X_train_normal.copy(), {"n_injected": 0, "effective_contamination": 0.0}
    mu = X_train_normal.mean(axis=0)
    d = np.linalg.norm(X_atk - mu, axis=1)
    k = min(int(rate * len(X_train_normal)), len(X_atk))
    idx = np.argsort(d)[:k]
    X_new = np.vstack([X_train_normal, X_atk[idx]])
    return X_new, {"n_injected": k, "effective_contamination": k / len(X_new)}


def feature_noise_injection(X_train_normal, X_train_full, y_train_full, rate, seed=42):
    rng = np.random.RandomState(seed)
    X_new = X_train_normal.copy()
    k = int(rate * len(X_new))
    idx = rng.choice(len(X_new), k, replace=False)
    std = X_new.std(axis=0)
    noise = rng.normal(0, CONFIG["NOISE_SIGMA"] * std, size=X_new[idx].shape)
    X_new[idx] = np.clip(X_new[idx] + noise, 0, 1)
    return X_new, {"n_poisoned": k, "effective_contamination": 0.0}


_REGISTRY = {
    "random_flip":   random_injection,
    "targeted_flip": targeted_injection,
    "feature_noise": feature_noise_injection,
}


def apply_poison(X_train_normal, X_train_full, y_train_full,
                 attack_name: str, rate: float, seed: int = 42):
    try:
        fn = _REGISTRY[attack_name]
    except KeyError:
        raise ValueError(f"Unknown attack: {attack_name}")
    return fn(X_train_normal, X_train_full, y_train_full, rate, seed)
