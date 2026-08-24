#!/usr/bin/env python3
"""
Generate HPO manifests as JSONL files (one config per line).

Stage 1 manifests are derived purely from the grid definitions and can be
generated immediately. Stages 2/3/final require the top-3 winners from the
preceding stage, which are read from the aggregated CSV produced by the
runner outputs (see scripts/aggregate_hpo.py — supplied separately).

Output location: $SWAT_OUTPUT_DIR/hpo/manifests/{stage}.jsonl
                 (default $SCRATCH/swat_paper_run/hpo/manifests/)

Examples
--------
    # Stage 1 (no prerequisites)
    python scripts/make_hpo_manifests.py --stage ae_stage1
    python scripts/make_hpo_manifests.py --stage lstm_ae_stage1

    # Stage 2 (needs --top-from pointing at the stage-1 results CSV)
    python scripts/make_hpo_manifests.py --stage ae_stage2 \\
        --top-from $SWAT_OUTPUT_DIR/hpo/results/ae_stage1.csv

    # Stage 3
    python scripts/make_hpo_manifests.py --stage ae_stage3 \\
        --top-from $SWAT_OUTPUT_DIR/hpo/results/ae_stage2.csv

    # Final (uses stage-3 winners)
    python scripts/make_hpo_manifests.py --stage ae_final \\
        --top-from $SWAT_OUTPUT_DIR/hpo/results/ae_stage3.csv

The aggregate CSV is expected to have at least one row per config with a
column named 'composite_f1' = mean(clean_f1, poisoned_f1) and a JSON-encoded
'config' column. Rank by composite_f1 descending and take the first three
distinct configs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Allow running as a script without `python -m`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hpo.grids import (
    enumerate_ae_stage1,
    enumerate_ae_stage3,
    enumerate_final,
    enumerate_lstm_ae_stage1,
    enumerate_lstm_ae_stage3,
    enumerate_sigma_hpo,
    enumerate_stage2,
    AE_STAGE2_GRID,
    LSTM_AE_STAGE2_GRID,
)


def _output_root() -> Path:
    out = os.environ.get("SWAT_OUTPUT_DIR")
    if not out:
        scratch = os.environ.get("SCRATCH")
        if not scratch:
            raise RuntimeError("Set SWAT_OUTPUT_DIR or SCRATCH before calling this script.")
        out = f"{scratch}/swat_paper_run"
    root = Path(out) / "hpo" / "manifests"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_top3(csv_path: Path) -> list[dict]:
    """
    Read aggregated stage results, sort by composite_f1 desc, return the top 3
    distinct configs as Python dicts. The aggregator writes the config as a
    JSON string in the 'config' column.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                cfg = json.loads(r["config"])
            except (KeyError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Bad row in {csv_path}: {e}")
            cf1 = float(r.get("composite_f1") or r.get("f1") or 0.0)
            rows.append((cf1, cfg))

    rows.sort(key=lambda x: x[0], reverse=True)
    seen, top = set(), []
    for cf1, cfg in rows:
        # De-dup by the variable subset of the config (drop seed)
        key = json.dumps({k: v for k, v in cfg.items() if k != "seed"}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        top.append(cfg)
        if len(top) == 3:
            break
    if len(top) < 3:
        raise RuntimeError(f"Only {len(top)} distinct configs found in {csv_path}; need 3.")
    return top


def _read_top_config(csv_path: Path) -> dict:
    """
    Return the single top config from either:
      (a) a *_by_config.csv (multi-seed, ranked by composite_f1_mean), or
      (b) a plain *_final.csv (per-task, ranked by composite_f1).
    Prefers the by-config form when present. The returned dict is the
    `config` JSON column parsed into a Python dict.
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                cfg = json.loads(r["config"])
            except (KeyError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Bad row in {csv_path}: {e}")
            score_col = "composite_f1_mean" if "composite_f1_mean" in r else "composite_f1"
            score = float(r.get(score_col) or 0.0)
            rows.append((score, cfg))
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[0][1]


def _write_manifest(stage: str, configs: list[dict]) -> Path:
    out_path = _output_root() / f"{stage}.jsonl"
    with open(out_path, "w") as f:
        for cfg in configs:
            f.write(json.dumps(cfg) + "\n")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        choices=[
            "ae_stage1", "ae_stage2", "ae_stage3", "ae_final",
            "lstm_ae_stage1", "lstm_ae_stage2", "lstm_ae_stage3", "lstm_ae_final",
            "sigma_hpo",
        ],
        required=True,
    )
    p.add_argument(
        "--top-from",
        type=Path,
        help="Path to the aggregated stage-(N-1) CSV. Required for stage2/3/final.",
    )
    p.add_argument(
        "--ae-from",
        type=Path,
        help="Path to ae_final_by_config.csv (or ae_final.csv). "
             "Required for sigma_hpo. The top-by-mean-composite row is used.",
    )
    p.add_argument(
        "--lstm-from",
        type=Path,
        help="Path to lstm_ae_final_by_config.csv (or lstm_ae_final.csv). "
             "Required for sigma_hpo.",
    )
    args = p.parse_args()

    if args.stage == "ae_stage1":
        configs = enumerate_ae_stage1()
    elif args.stage == "lstm_ae_stage1":
        configs = enumerate_lstm_ae_stage1()
    elif args.stage == "sigma_hpo":
        if args.ae_from is None or args.lstm_from is None:
            p.error("sigma_hpo requires both --ae-from and --lstm-from.")
        # For sigma_hpo, the winner is the single TOP row of each by-config CSV.
        # Accept either `*_by_config.csv` (multi-seed) or the per-task CSV.
        ae_win   = _read_top_config(args.ae_from)
        lstm_win = _read_top_config(args.lstm_from)
        print(f"AE winner config:      {ae_win}")
        print(f"LSTM-AE winner config: {lstm_win}")
        configs = enumerate_sigma_hpo(ae_win, lstm_win)
    elif args.stage in ("ae_stage2", "lstm_ae_stage2",
                        "ae_stage3", "lstm_ae_stage3",
                        "ae_final", "lstm_ae_final"):
        if args.top_from is None:
            p.error(f"--top-from is required for {args.stage}")
        top = _read_top3(args.top_from)
        if args.stage == "ae_stage2":
            configs = enumerate_stage2(top, AE_STAGE2_GRID)
        elif args.stage == "lstm_ae_stage2":
            configs = enumerate_stage2(top, LSTM_AE_STAGE2_GRID)
        elif args.stage == "ae_stage3":
            configs = enumerate_ae_stage3(top)
        elif args.stage == "lstm_ae_stage3":
            configs = enumerate_lstm_ae_stage3(top)
        else:  # *_final
            configs = enumerate_final(top)

    out_path = _write_manifest(args.stage, configs)
    print(f"Wrote {len(configs)} configs to {out_path}")
    print(f"Submit array with --array=0-{len(configs) - 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
