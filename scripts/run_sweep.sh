#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
unset PYTHONPATH
export PYTHONPATH="${ROOT}:${ROOT}/src"
exec python "${ROOT}/scripts/run_sweep.py" "$@"
