#!/usr/bin/env bash
# Optional venv: override with RUN_SWEEP_VENV=/path/to/venv/bin/activate
VENV_ACTIVATE="${RUN_SWEEP_VENV:-/cs/snapless/gabis/eliyahabba/venvs/AdaptEval/bin/activate}"
if [ -f "$VENV_ACTIVATE" ]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

set -euo pipefail

# Never derive repo root from $0 alone under sbatch (script is copied under /var/spool/slurmd/).
if [ -n "${PROJECT_DIR:-}" ]; then
  PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
elif [ -n "${GROWING_PAINS_ROOT:-}" ]; then
  PROJECT_DIR="$(cd "${GROWING_PAINS_ROOT}" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/run_sweep.py" ]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "$PROJECT_DIR"

if [ -n "${HF_HOME_OVERRIDE:-}" ]; then
  export HF_HOME="${HF_HOME_OVERRIDE}"
elif [ -d "/cs/snapless/gabis/gabis/shared/huggingface" ]; then
  export HF_HOME="/cs/snapless/gabis/gabis/shared/huggingface"
fi

export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export UNITXT_ALLOW_UNVERIFIED_CODE="${UNITXT_ALLOW_UNVERIFIED_CODE:-True}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

exec python "${PROJECT_DIR}/scripts/run_sweep.py" "$@"
