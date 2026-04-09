#!/usr/bin/env bash
# Light Slurm driver: only runs run_sweep.py with --dispatch sbatch (issues many sbatch calls).
# Each real experiment runs under scripts/run_chain_job.sh with ~24g RAM + GPUs.
#
# Usage:
#   sbatch scripts/run_sweep_driver_job.sh --category 1a
#
#SBATCH --job-name=gp-sweep-driver
#SBATCH --mem=2g
#SBATCH --time=4:0:0
#SBATCH --cpus-per-task=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_sweep.sh" --dispatch sbatch "$@"
