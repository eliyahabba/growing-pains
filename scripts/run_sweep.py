"""Backward-compatible shim. Use run_experiments.py instead."""
from __future__ import annotations

import sys
from pathlib import Path

# Forward all invocations to run_experiments.py
sys.argv[0] = str(Path(__file__).parent / "run_experiments.py")
exec(compile(open(sys.argv[0]).read(), sys.argv[0], "exec"))
