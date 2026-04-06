"""
Anchor item selection for TinyBenchmarks.

Methods (via `AnchorConfig.method`):
- irt_clustering (default, alias: anchor-irt): KMeans on IRT parameters (a, b)
- correctness_clustering (alias: anchor): KMeans on model response patterns
- top_k_discrimination: Top-K items by discrimination parameter
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

from config.constants import (
    ANCHOR_IRT_CLUSTERING,
    ANCHOR_TOP_K,
    ANCHOR_CORRECTNESS,
    ANCHOR_METHOD_ALIASES,
)


@dataclass
class AnchorConfig:
    number_items: int = 100
    method: Literal["irt_clustering", "correctness_clustering", "top_k_discrimination"] = ANCHOR_IRT_CLUSTERING
    random_state: int = 42
    n_trials: int = 1
    balance_weights: np.ndarray | None = None


def find_anchor_items_clustering(
    item_params: pd.DataFrame, 
    matrix_df: pd.DataFrame | None = None,
    config: AnchorConfig | None = None,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray]:
    """Find anchor items using KMeans clustering.
    
    Methods:
    - irt_clustering/anchor-irt: KMeans on IRT parameters (A, B)
    - correctness_clustering/anchor: KMeans on response patterns (matrix_df)
    
    Args:
        item_params: DataFrame with IRT parameters (a,b) indexed by question_id
        matrix_df: Matrix DataFrame for correctness-based clustering (required for anchor method)
        config: Configuration for anchor selection
        A_matrix: Full discrimination matrix shape (1, D, n_items)
        B_matrix: Full difficulty matrix shape (1, D, n_items)
        
    Returns:
        Tuple of (anchor_question_ids, anchor_weights)
    """
    cfg = config or AnchorConfig()
    
    if item_params.empty:
        return [], np.array([])
    
    if not {"a", "b"}.issubset(item_params.columns):
        raise ValueError("item_params must have columns 'a' and 'b'")

    # top_k_discrimination: bypass clustering entirely
    if cfg.method == ANCHOR_TOP_K:
        anchor_ids = find_anchor_items_top_k_discrimination(item_params, cfg)
        weights = np.ones(len(anchor_ids)) / max(len(anchor_ids), 1)
        return anchor_ids, weights

    # Resolve legacy aliases (e.g. efficbench naming)
    method = ANCHOR_METHOD_ALIASES.get(cfg.method, cfg.method)

    # Prepare clustering features (X) based on method
    if method == ANCHOR_IRT_CLUSTERING:
        question_ids = item_params.index.tolist()
        
        if A_matrix is not None and B_matrix is not None:
            # Use full MIRT parameters: X = vstack(A, B).T -> (n_items, 2*D)
            A_squeezed = A_matrix.squeeze()  # (D, n_items)
            B_squeezed = B_matrix.squeeze()  # (D, n_items)
            if A_squeezed.ndim == 1:
                A_squeezed = A_squeezed.reshape(1, -1)
                B_squeezed = B_squeezed.reshape(1, -1)
            X = np.vstack((A_squeezed, B_squeezed)).T  # (n_items, 2*D)
        else:
            # Fallback to scalar (a,b)
            X = np.column_stack([item_params["a"].values, item_params["b"].values])
            
    elif method == ANCHOR_CORRECTNESS:
        # Use correctness patterns: X = scores_train.T (questions × models)
        # Supports both:
        #   - long format: columns [question_id, model_name, normalized_score]
        #   - wide pivoted format: index=question_id, columns=model_name
        if matrix_df is None:
            raise ValueError("matrix_df required for correctness_clustering/anchor method")

        question_ids = item_params.index.tolist()

        has_long_cols = {"question_id", "model_name", "normalized_score"}.issubset(matrix_df.columns)
        if has_long_cols:
            models = sorted(matrix_df["model_name"].astype(str).unique())
            X = np.full((len(question_ids), len(models)), np.nan)
            question_to_idx = {q: i for i, q in enumerate(question_ids)}
            model_to_idx = {m: i for i, m in enumerate(models)}

            for _, row in matrix_df.iterrows():
                qid = row["question_id"]
                mname = str(row["model_name"])
                if qid in question_to_idx and mname in model_to_idx:
                    q_idx = question_to_idx[qid]
                    m_idx = model_to_idx[mname]
                    X[q_idx, m_idx] = row["normalized_score"]
        else:
            # Assume matrix_df is already pivoted: index=question_id, columns=model_name
            wide = matrix_df.copy()
            wide.index = wide.index.astype(str)
            wide.columns = [str(c) for c in wide.columns]
            # Keep only relevant questions, in deterministic item_params order
            question_ids = [q for q in question_ids if q in set(wide.index)]
            if not question_ids:
                return [], np.array([])
            wide = wide.loc[question_ids]
            X = wide.to_numpy(dtype=float)

        # Remove questions with no data
        valid_mask = ~np.isnan(X).all(axis=1)
        X = X[valid_mask]
        question_ids = [q for i, q in enumerate(question_ids) if valid_mask[i]]

        if len(question_ids) == 0:
            return [], np.array([])

        # KMeans does not support NaN; impute remaining missing entries with global mean.
        if np.isnan(X).any():
            global_mean = np.nanmean(X)
            if np.isnan(global_mean):
                global_mean = 0.5
            X = np.where(np.isnan(X), global_mean, X)
    else:
        raise ValueError(f"Unknown clustering method: {cfg.method}")
    
    # Prepare balance weights
    if cfg.balance_weights is not None:
        # Convert to numpy array if needed (for JSON deserialization compatibility)
        balance_weights_array = np.array(cfg.balance_weights) if isinstance(cfg.balance_weights, list) else cfg.balance_weights
        
        # Map balance weights to current question set
        question_indices = [i for i, q in enumerate(item_params.index) if q in question_ids]
        norm_balance_weights = balance_weights_array[question_indices]
    else:
        # Uniform weights
        norm_balance_weights = np.ones(len(question_ids))
    
    # Normalize weights to sum to 1
    norm_balance_weights = norm_balance_weights / norm_balance_weights.sum()
    
    n_clusters = min(cfg.number_items, len(question_ids))
    
    # Fit KMeans (n_trials=1 for backward compat, >1 for multiple trials picking best by inertia)
    n_trials = cfg.n_trials
    if n_trials == 1:
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=cfg.random_state)
        kmeans.fit(X, sample_weight=norm_balance_weights)
    else:
        kmeans_models = []
        for t in range(n_trials):
            km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=1000 * t + cfg.random_state)
            km.fit(X, sample_weight=norm_balance_weights)
            kmeans_models.append(km)
        kmeans = kmeans_models[np.argmin([m.inertia_ for m in kmeans_models])]
    
    # Find anchor points: closest real point to each cluster center
    distances = pairwise_distances(kmeans.cluster_centers_, X, metric='euclidean')
    anchor_indices = distances.argmin(axis=1)
    anchor_question_ids = [str(question_ids[i]) for i in anchor_indices]
    
    # Calculate anchor weights: sum of balance weights per cluster
    anchor_weights = np.array([
        np.sum(norm_balance_weights[kmeans.labels_ == c]) 
        for c in range(n_clusters)
    ])
    
    return anchor_question_ids, anchor_weights


def find_anchor_items_top_k_discrimination(
    item_params: pd.DataFrame,
    config: AnchorConfig | None = None,
) -> list[str]:
    """Select the top-K items with the highest discrimination parameter (a).

    This is the naive baseline for the reviewer's ablation:
    "What if you just pick the most discriminative items?"
    Unlike IRT-clustering which spreads coverage across the (a,b) space,
    this concentrates anchors on the items with highest a — which all tend
    to cluster near medium difficulty (b ≈ 0), leaving the difficulty axis
    uncovered and leading to biased estimates for extreme-ability models.
    """
    cfg = config or AnchorConfig()
    if item_params.empty:
        return []
    if "a" not in item_params.columns:
        raise ValueError("item_params must have column 'a'")
    n = min(cfg.number_items, len(item_params))
    return item_params.nlargest(n, "a").index.tolist()


