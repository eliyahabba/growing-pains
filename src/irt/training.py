from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .fit import TrainingConfig, fit_2pl_parameters


def train_item_parameters(
        train_matrix_df: pd.DataFrame,
        test_matrix_df: pd.DataFrame | None = None,
        config: TrainingConfig | None = None,
        output_dir: str | None = None,
        anchor_items: list[dict] | None = None,
) -> pd.DataFrame:
    """Train or estimate 2PL item parameters (a,b) per question_id using tinyBenchmarks utilities.
    
    Args:
        train_matrix_df: Training data matrix for IRT parameter estimation
        test_matrix_df: Optional test data matrix (currently unused, reserved for future validation)
        config: Training configuration
        output_dir: Optional directory to save IRT dataset files (if None, uses temporary directory)
    
    Returns:
        DataFrame indexed by question_id with columns ["a", "b"] and attached metadata.
    """
    # Train IRT model on training data only
    # The notebook's internal validation logic will use cross-validation within the training set
    return fit_2pl_parameters(train_matrix_df, config, output_dir, anchor_items=anchor_items)


def save_item_parameters(df: pd.DataFrame, out_path: str) -> None:
    """Save item parameters to parquet, plus MIRT matrices to JSON if available."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    
    # Also save MIRT matrices and metadata to JSON if available in attrs
    if hasattr(df, 'attrs') and df.attrs:
        metadata_path = p.with_suffix('.meta.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(df.attrs, f)
        except (TypeError, ValueError) as e:
            print(f"Warning: Could not save metadata: {e}")
