

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EvaluationMetrics:
    """Results of comparing full vs selected question performance."""
    rmse: float
    mae: float
    correlation: float
    bias: float
    relative_error: float
    n_samples: int


def estimate_error_to_full_eval(partial_scores: pd.Series, full_scores: pd.Series) -> float:
    """Return RMSE between partial and full normalized scores for overlap of indices."""
    idx = partial_scores.index.intersection(full_scores.index)
    if len(idx) == 0:
        return float("nan")
    diff = partial_scores.loc[idx] - full_scores.loc[idx]
    return float((diff.pow(2).mean()) ** 0.5)


def compute_evaluation_metrics(partial_scores: pd.Series, full_scores: pd.Series) -> EvaluationMetrics:
    """Compute comprehensive metrics comparing partial vs full evaluation scores."""
    idx = partial_scores.index.intersection(full_scores.index)
    
    if len(idx) == 0:
        return EvaluationMetrics(
            rmse=float("nan"), mae=float("nan"), correlation=float("nan"),
            bias=float("nan"), relative_error=float("nan"), n_samples=0
        )
    
    partial = partial_scores.loc[idx]
    full = full_scores.loc[idx]
    
    # Basic error metrics
    diff = partial - full
    rmse = float(np.sqrt(diff.pow(2).mean()))
    mae = float(diff.abs().mean())
    
    # Bias (systematic over/under estimation)
    bias = float(diff.mean())
    
    # Correlation
    correlation = float(partial.corr(full)) if len(idx) > 1 else float("nan")
    
    # Relative error (as percentage of full score range)
    full_range = full.max() - full.min()
    relative_error = float(rmse / full_range * 100) if full_range > 0 else float("nan")
    
    return EvaluationMetrics(
        rmse=rmse, mae=mae, correlation=correlation,
        bias=bias, relative_error=relative_error, n_samples=len(idx)
    )


def compute_model_performance(matrix_df: pd.DataFrame, model_name: str, 
                             dataset_name: str = None, question_ids: list[str] = None) -> pd.Series:
    """Compute aggregated performance for a model on a dataset/question subset."""
    # Filter by model
    model_data = matrix_df[matrix_df["model_name"] == model_name].copy()
    
    if len(model_data) == 0:
        return pd.Series(dtype=float)
    
    # Filter by dataset if specified
    if dataset_name is not None:
        model_data = model_data[model_data["dataset"] == dataset_name]
    
    # Filter by specific questions if provided
    if question_ids is not None:
        model_data = model_data[model_data["question_id"].isin(question_ids)]
    
    if len(model_data) == 0:
        return pd.Series(dtype=float)
    
    # Group by dataset and compute mean normalized score
    if dataset_name is None:
        # Aggregate across all datasets
        return model_data.groupby("dataset")["normalized_score"].mean()
    else:
        # Return single score for the specified dataset
        return pd.Series({dataset_name: model_data["normalized_score"].mean()})


