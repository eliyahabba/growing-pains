#!/usr/bin/env bash
# One chain_experiment.py run per Slurm job (same idea as AdaptEval's
# sh_run/run_chain_linking_unified.sh). Override resources when submitting, e.g.:
#   sbatch --mem=8g --time=24:0:0 scripts/run_chain_job.sh --output-dir ...
#
#SBATCH --job-name=gp-chain
#SBATCH --mem=24g
#SBATCH --time=12:0:0
#SBATCH --cpus-per-task=2
#SBATCH --gres=gg:g0:2
#SBATCH --killable
#SBATCH --requeue

set -euo pipefail

VENV_ACTIVATE="${RUN_SWEEP_VENV:-/cs/snapless/gabis/eliyahabba/venvs/AdaptEval/bin/activate}"
if [ -f "$VENV_ACTIVATE" ]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

if [ -n "${PROJECT_DIR:-}" ]; then
  PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
elif [ -n "${GROWING_PAINS_ROOT:-}" ]; then
  PROJECT_DIR="$(cd "${GROWING_PAINS_ROOT}" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/run_chain_job.sh" ]; then
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

if command -v module &>/dev/null; then
  module load cuda 2>/dev/null || true
fi

exec python "${PROJECT_DIR}/src/chain_experiment.py" "$@"
