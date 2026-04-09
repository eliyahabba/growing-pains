"""Crash-recovery and signal handlers for long-running experiments."""
from __future__ import annotations

import atexit
import os
import signal
from pathlib import Path

from src.io import cleanup_training_datasets

_cleanup_paths: dict = {
    'temp_dir': None,
    'chain_cache_dir': None,
    'output_dir': None,
    'cleanup_training_data': True,
    'cleanup_cache': True,
}


def register_cleanup_paths(temp_dir: Path, chain_cache_dir: Path, output_dir: Path,
                           cleanup_training_data: bool, cleanup_cache: bool):
    _cleanup_paths.update(
        temp_dir=temp_dir, chain_cache_dir=chain_cache_dir, output_dir=output_dir,
        cleanup_training_data=cleanup_training_data, cleanup_cache=cleanup_cache,
    )


def emergency_cleanup():
    """Remove temp/cache dirs on crash or interrupt."""
    import shutil

    print("\nEmergency cleanup...")
    for key in ('temp_dir', 'chain_cache_dir'):
        p = _cleanup_paths.get(key)
        if p and p.exists():
            try:
                shutil.rmtree(p)
                print(f"  Removed {p}")
            except Exception as e:
                print(f"  Failed {p}: {e}")
    output_dir = _cleanup_paths.get('output_dir')
    if _cleanup_paths.get('cleanup_training_data') and output_dir and output_dir.exists():
        freed = 0
        for pattern in ["irt_base", "chain_cache/after_*", "dist_*/irt_*"]:
            for d in output_dir.glob(pattern):
                if d.is_dir():
                    freed += cleanup_training_datasets(d)
        if freed:
            print(f"  Cleaned {freed / (1024*1024):.1f}MB training data")


def setup_cleanup_handlers():
    """Register atexit + signal handlers for graceful cleanup."""
    atexit.register(emergency_cleanup)

    def _handler(signum, _frame):
        emergency_cleanup()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handler)
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, _handler)
