"""
Online / retraining poisoning — Phase 1.3 paper headline attack.

Ported faithfully from `SWaT_PathB_Pilot_Colab.ipynb` cell §4.2 (result6) with
two implementation invariants the pilot established:

  (1) **No attack reuse across rounds.** The attack pool is pre-ranked once
      (per generator, per seed). Round r consumes the next slice of the
      ranking via a cursor. A given attack sample is therefore injected at
      most once across the whole T-round trajectory.

  (2) **Exact cumulative budget.** `cumulative_p = total_injected / original_pool_size`,
      not `delta_p * round_index`. The reported value is exact even when the
      attack pool runs out before round T.

Two generators are supported, chosen so the cross-detector comparison is clean
(see the online-retraining design note §F item 3 and §B):

  * `random_injection` — seeded uniform random permutation of the attack pool.
  * `high_loss`        — attacks ranked by clean-AE reconstruction error
                         (descending). The clean AE is trained ONCE per seed
                         and used to rank attacks for ALL 11 detectors at
                         that seed. This is the "clean-AE proxy" decision
                         locked in 2026-04-26 (post-pilot).

Per-(generator, seed) rankings are cached to
`$SWAT_OUTPUT_DIR/checkpoints/online/_rankings/<generator>__<seed>__n<pool_size>.npy`
so detectors with the same ranked attack pool do not pay the proxy-AE training
cost repeatedly.

For LSTM-AE the round-loop appends poisons at the END of the contiguous-normal
training block to preserve temporal order; see `run_online.py` for the
sequence-aware orchestration.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from .config import CONFIG, get_output_dir


# ────────────────────── attack-pool ranking ─────────────────────────
def _ranking_cache_path(generator: str, seed: int, pool_size: int) -> Path:
    return (
        get_output_dir() / "checkpoints" / "online" / "_rankings" /
        f"{generator}__{seed}__n{pool_size}.npy"
    )


def compute_attack_ranking(
    generator: str,
    X_atk_pool: np.ndarray,
    X_train_normal: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Return an integer array of length len(X_atk_pool) giving the consumption
    order: ranked[0] is the first attack to inject, ranked[k] is the k+1-th, etc.

    If a cached ranking exists at the canonical path, it's loaded; otherwise
    the ranking is computed and (when use_cache) saved.

    Caching matters because the high_loss ranking trains a clean AE; without
    the cache, detectors with identical ranked pools would each pay one extra
    AE training per seed.
    """
    if len(X_atk_pool) == 0:
        raise RuntimeError("Empty attack pool — online retraining not possible.")

    cache = _ranking_cache_path(generator, seed, len(X_atk_pool)) if use_cache else None
    if cache is not None and cache.exists():
        ranked = np.load(cache)
        if len(ranked) == len(X_atk_pool):
            return ranked

    if generator == "random_injection":
        rng = np.random.RandomState(seed)
        ranked = rng.permutation(len(X_atk_pool))

    elif generator == "high_loss":
        # Clean-AE proxy: same ranking used for ALL 11 detectors at this seed.
        # See the online-retraining design note §B and the post-pilot decision (2026-04-26).
        import torch
        from .models import AutoencoderDetector, DEVICE
        from .eval_utils import set_seed

        set_seed(seed)
        proxy = AutoencoderDetector(X_train_normal.shape[1], seed=seed)
        proxy.train(X_train_normal, None, X_val, y_val)
        proxy.model.eval()
        with torch.no_grad():
            atk_t = torch.tensor(X_atk_pool, dtype=torch.float32, device=DEVICE)
            atk_loss = ((proxy.model(atk_t) - atk_t) ** 2).mean(dim=1).cpu().numpy()
        ranked = np.argsort(-atk_loss).astype(np.int64)  # descending

        del proxy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    else:
        raise ValueError(
            f"Unknown online generator: {generator!r}. "
            f"Supported: 'random_injection', 'high_loss'."
        )

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Atomic save: write to .tmp then rename
        tmp = cache.with_suffix(cache.suffix + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, ranked)
        tmp.replace(cache)

    return ranked.astype(np.int64)


# ────────────────────── round-loop helpers ──────────────────────────
def round_inject_indices(
    ranked_order: np.ndarray,
    cursor: int,
    k_per_round: int,
) -> tuple[np.ndarray, int, int]:
    """
    Return (next_indices, new_cursor, n_actually_injected).

    The attack pool may run out before T rounds finish; n_actually_injected
    can be smaller than k_per_round at the end. cumulative_p in the trajectory
    JSON should be derived from total_injected, NOT from delta_p * round_index.
    """
    pool_size = len(ranked_order)
    if cursor >= pool_size:
        return ranked_order[:0], cursor, 0
    end = min(cursor + k_per_round, pool_size)
    return ranked_order[cursor:end], end, end - cursor


def trajectory_filename(detector: str, generator: str, T: int,
                        delta_p: float, seed: int, round_idx: int) -> str:
    """{detector}__{generator}__T{T}__dp<dp>__{seed}__r{round}.json"""
    from .config import online_dp_tag
    return (
        f"{detector}__{generator}"
        f"__T{T}__{online_dp_tag(delta_p)}"
        f"__{seed}__r{round_idx}.json"
    )


def trajectory_path(detector: str, generator: str, T: int,
                    delta_p: float, seed: int, round_idx: int) -> Path:
    """Canonical per-round JSON path under $SWAT_OUTPUT_DIR/checkpoints/online/."""
    return (
        get_output_dir() / "checkpoints" / "online" /
        trajectory_filename(detector, generator, T, delta_p, seed, round_idx)
    )


def expected_round_paths(detector: str, generator: str, T: int,
                         delta_p: float, seed: int) -> list[Path]:
    """Per-round JSON paths for round 0 .. T (inclusive)."""
    return [
        trajectory_path(detector, generator, T, delta_p, seed, r)
        for r in range(T + 1)
    ]
