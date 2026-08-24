#!/usr/bin/env python3
"""
Generate the online-retraining manifest TSV.

Output: $SWAT_OUTPUT_DIR/online_manifest.tsv

The manifest is one row per Slurm array task. Columns:
    task_id  detector  generator  T  delta_p  seed

The order matches `src.config.online_combos()` so SLURM_ARRAY_TASK_ID maps
directly to a manifest row by index.

Total rows (defaults): 11 detectors × 2 generators × 3 T × 3 Δp × 5 seeds = 990.

Idempotent: re-running overwrites the file atomically.

Usage on Narval (login node, after env setup):

    cd ~/projects/def-liyang/$USER/narval_swat_run
    source venv/bin/activate
    python scripts/make_online_manifest.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

# Allow running as a script outside the package
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.config import (  # noqa: E402
    CONFIG, get_output_dir, online_combos, online_combo_count,
)


def main() -> int:
    combos = online_combos()
    expected = online_combo_count()
    if len(combos) != expected:
        print(f"[make_online_manifest] ERROR: combo count mismatch "
              f"(got {len(combos)}, expected {expected}).", file=sys.stderr)
        return 2

    out = get_output_dir() / "online_manifest.tsv"
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["task_id", "detector", "generator", "T", "delta_p", "seed"])
        for i, c in enumerate(combos):
            w.writerow([i, c["detector"], c["generator"], c["T"],
                        f"{c['delta_p']:.4f}", c["seed"]])
    os.replace(tmp, out)

    print(f"[make_online_manifest] wrote {len(combos)} rows -> {out}")
    print(f"[make_online_manifest] grid = {len(CONFIG['ONLINE_DETECTORS'])} detectors "
          f"× {len(CONFIG['ONLINE_GENERATORS'])} generators "
          f"× {len(CONFIG['ONLINE_T_VALUES'])} T "
          f"× {len(CONFIG['ONLINE_DELTA_P_VALUES'])} Δp "
          f"× {len(CONFIG['ONLINE_SEEDS'])} seeds")
    print(f"[make_online_manifest] use:  sbatch --array=0-{len(combos) - 1}%24 "
          f"slurm/40_online_retraining.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
