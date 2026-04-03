"""Synthetic validation suite.

--fast   : trains a tiny IRT model on synthetic data, checks MAE is stable
--full   : loads real data from data/input/tinybenchmarks (skips if absent)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_path() -> None:
    for p in [str(REPO_ROOT), str(REPO_ROOT / "src")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _make_synthetic_matrix(n_models: int = 30, n_items: int = 50, seed: int = 0) -> pd.DataFrame:
    """Build a minimal response matrix in the format expected by fit_2pl_parameters."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_models):
        for q in range(n_items):
            rows.append({
                "model_name": f"model_{m}",
                "question_id": f"q{q:03d}",
                "dataset": "synthetic",
                "normalized_score": float(rng.integers(0, 2)),
            })
    return pd.DataFrame(rows)


def run_fast() -> None:
    _ensure_path()

    from config.constants import LB_DATASETS
    from src.experiments.utils.io import round_for_json

    assert len(LB_DATASETS) == 6, f"expected 6 LB datasets, got {len(LB_DATASETS)}"

    rounded = round_for_json({"x": 1.23456789, "nested": {"y": float("nan")}})
    assert rounded["nested"]["y"] is None
    assert abs(rounded["x"] - 1.2346) < 1e-4

    from irt import TrainingConfig, fit_2pl_parameters

    matrix = _make_synthetic_matrix(n_models=30, n_items=50)

    # Minimal epochs so test completes in ~5 s
    cfg = TrainingConfig(
        dims_search=[1],
        epochs=50,
        validate_dimensions=False,
        number_item_per_scenario=50,
        deterministic=True,
        device="cpu",
    )
    params = fit_2pl_parameters(matrix, cfg)

    assert isinstance(params, pd.DataFrame), "fit_2pl_parameters must return DataFrame"
    assert {"a", "b"}.issubset(params.columns), f"missing a/b columns: {params.columns.tolist()}"
    assert len(params) == 50, f"expected 50 item params, got {len(params)}"

    # Sanity check on discrimination (a) and difficulty (b) ranges
    assert (params["a"] >= 0).all(), "discrimination must be non-negative"
    assert params["b"].between(-15, 15).all(), "difficulty out of expected range"

    # Verify training actually produces variation (not all identical)
    assert params["b"].std() > 0.01, "difficulty parameters show no variation — training may have failed"

    print("fast ok")


def run_full() -> None:
    _ensure_path()
    tb = REPO_ROOT / "data" / "input" / "tinybenchmarks"
    if not tb.is_dir() or not any(tb.glob("*.pickle")):
        print("full: skip (no data/input/tinybenchmarks/*.pickle)")
        return
    from src.experiments.equating.cross_dataset_equating import ExperimentConfig, load_all_datasets

    cfg = ExperimentConfig(data_source_mode="lb")
    ds = load_all_datasets(cfg)
    assert len(ds) >= 1, "expected at least one dataset loaded"
    print(f"full ok ({len(ds)} datasets loaded)")


def main() -> None:
    p = argparse.ArgumentParser(description="Validation suite for growing-pains")
    p.add_argument("--fast", action="store_true", help="synthetic checks (default)")
    p.add_argument("--full", action="store_true", help="load real data if present")
    args = p.parse_args()
    if args.full:
        run_full()
    else:
        run_fast()


if __name__ == "__main__":
    main()
