#!/usr/bin/env bash
#
# Shell wrapper for run_experiments.py.
#
# Usage:
#   bash scripts/run_sweep.sh --experiment lb_anchor_sweep --dispatch sbatch
#
# Optional venv: set RUN_SWEEP_VENV=/path/to/venv/bin/activate
VENV_ACTIVATE="${RUN_SWEEP_VENV:-}"
if [ -n "$VENV_ACTIVATE" ] && [ -f "$VENV_ACTIVATE" ]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

set -euo pipefail

if [ -n "${PROJECT_DIR:-}" ]; then
  PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
elif [ -n "${GROWING_PAINS_ROOT:-}" ]; then
  PROJECT_DIR="$(cd "${GROWING_PAINS_ROOT}" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/run_experiments.py" ]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "$PROJECT_DIR"

export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

exec python "${PROJECT_DIR}/scripts/run_experiments.py" "$@"
