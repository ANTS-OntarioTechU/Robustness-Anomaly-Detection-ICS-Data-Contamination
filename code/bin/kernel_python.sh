#!/bin/bash
module --force purge 2>/dev/null
module load StdEnv/2023 python/3.11 scipy-stack cuda/12.2 2>/dev/null
exec "${SWAT_VENV:-$HOME/projects/def-liyang/$USER/narval_swat_run/venv}/bin/python" "$@"
