from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FAST_REF_MAE = 0.07211163129546802


def _ensure_path() -> None:
    rs = str(REPO_ROOT)
    src = str(REPO_ROOT / "src")
    if rs not in sys.path:
        sys.path.insert(0, rs)
    if src not in sys.path:
        sys.path.insert(0, src)


def run_fast() -> None:
    _ensure_path()
    from config.constants import LB_DATASETS
    from src.experiments.utils.helpers import round_for_json

    assert len(LB_DATASETS) == 6
    rounded = round_for_json({"x": 1.23456789, "nested": {"y": float("nan")}})
    assert rounded["nested"]["y"] is None
    assert abs(rounded["x"] - 1.2346) < 1e-4

    np.random.seed(42)
    y_true = np.random.rand(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    mae = float(np.mean(np.abs(y_true - y_pred)))
    assert abs(mae - FAST_REF_MAE) < 1e-12, (mae, FAST_REF_MAE)
    print("fast ok")


def run_full() -> None:
    _ensure_path()
    tb = REPO_ROOT / "aggregated_data" / "tinybenchmarks"
    if not tb.is_dir() or not any(tb.glob("*.pickle")):
        print("full: skip (no aggregated_data/tinybenchmarks/*.pickle)")
        return
    from src.experiments.equating.cross_dataset_equating import ExperimentConfig, load_all_datasets

    cfg = ExperimentConfig(data_source_mode="lb")
    ds = load_all_datasets(cfg)
    assert len(ds) >= 1
    print("full ok")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="synthetic checks")
    p.add_argument("--full", action="store_true", help="load real data if present")
    args = p.parse_args()
    if args.fast:
        run_fast()
    elif args.full:
        run_full()
    else:
        run_fast()


if __name__ == "__main__":
    main()
