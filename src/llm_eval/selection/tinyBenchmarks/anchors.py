from __future__ import annotations
"""
Anchor Point Selection for TinyBenchmarks.

## Methods:

1. **irt_clustering** (alias: anchor-irt):
   - KMeans clustering on IRT parameters: X = vstack(A, B).T
   - Default method, used in all existing experiments

2. **correctness_clustering** (alias: anchor):
   - KMeans clustering on correctness patterns (scores_train.T)
   - Groups questions by model response patterns

3. **difficulty_binning**:
   - Bins by difficulty + Fisher information selection

Methods can be chosen via the `method` parameter in `AnchorConfig`.
Set `n_trials > 1` for efficbench-style multiple trials (default=1 for backward compat).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances


@dataclass
class AnchorConfig:
    number_items: int = 100  # total number of anchor points per dataset (from notebook)
    method: Literal["irt_clustering", "correctness_clustering", "anchor", "anchor-irt", "difficulty_binning", "top_k_discrimination"] = "irt_clustering"  # selection method
    random_state: int = 42  # for reproducible clustering (base seed)
    n_trials: int = 1  # number of KMeans trials (1=backward compatible, efficbench uses 5)
    balance_weights: np.ndarray | None = None  # balance weights for multi-subscenario datasets


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
    if cfg.method == "top_k_discrimination":
        anchor_ids = find_anchor_items_top_k_discrimination(item_params, cfg)
        weights = np.ones(len(anchor_ids)) / max(len(anchor_ids), 1)
        return anchor_ids, weights

    # Normalize method names (support both efficbench and our naming)
    method = cfg.method
    if method == "anchor-irt":
        method = "irt_clustering"
    elif method == "anchor":
        method = "correctness_clustering"
    
    # Prepare clustering features (X) based on method
    if method == "irt_clustering":
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
            
    elif method == "correctness_clustering":
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


def find_anchor_items_difficulty_binning(item_params: pd.DataFrame, config: AnchorConfig | None = None) -> list[str]:
    """Select anchor items distributed across difficulty levels.

    - Bin by difficulty (b parameter) into 10 levels
    - Within each bin, rank by Fisher information anchor score and take items
    - Total items taken = config.number_items, distributed across bins
    """
    cfg = config or AnchorConfig()
    if item_params.empty:
        return []
    
    if not {"a", "b"}.issubset(item_params.columns):
        raise ValueError("item_params must have columns 'a' and 'b'")
    
    # Compute Fisher information anchor scores
    anchorscore = compute_anchor_scores(item_params)
    df = item_params.copy()
    df["anchor_score"] = anchorscore
    
    # Default to 10 levels for binning
    levels = 10
    per_level = max(1, cfg.number_items // levels)  # Distribute items across levels
    
    # Bin by difficulty b into levels quantiles
    try:
        df["b_bin"] = pd.qcut(df["b"], q=levels, duplicates="drop")
    except Exception:
        df["b_bin"] = pd.cut(df["b"], bins=levels)
    
    picked: list[str] = []
    remaining_items = cfg.number_items
    
    for _, group in df.groupby("b_bin", observed=True):
        if len(group) == 0 or remaining_items <= 0:
            continue
        
        # Take up to per_level items from this bin, but don't exceed remaining_items
        items_to_take = min(per_level, len(group), remaining_items)
        top = group.sort_values("anchor_score", ascending=False).head(items_to_take)
        picked.extend([str(i) for i in top.index.tolist()])
        remaining_items -= items_to_take
    
    return picked


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


def find_anchor_items(item_params: pd.DataFrame, config: AnchorConfig | None = None) -> list[str]:
    """Find anchor items using the specified method.
    
    Supports methods (with efficbench aliases):
    - anchor-irt / irt_clustering: KMeans clustering in IRT parameter space
    - anchor / correctness_clustering: KMeans clustering on response patterns
    - difficulty_binning: Binning by difficulty + Fisher information (original approach)
    """
    cfg = config or AnchorConfig()
    
    if cfg.method == "difficulty_binning":
        return find_anchor_items_difficulty_binning(item_params, config)
    elif cfg.method == "top_k_discrimination":
        return find_anchor_items_top_k_discrimination(item_params, config)
    elif cfg.method in ["irt_clustering", "correctness_clustering", "anchor", "anchor-irt"]:
        anchor_ids, _ = find_anchor_items_clustering(item_params, config=config)
        return anchor_ids
    else:
        raise ValueError(f"Unknown anchor selection method: {cfg.method}")


def find_anchor_items_by_dataset(
    item_params: pd.DataFrame, 
    dataset_column: str | None, 
    number_items: int = 100,
    method: str = "irt_clustering",
    matrix_df: pd.DataFrame | None = None
) -> dict[str, list[str]]:
    """Find anchor items per dataset using specified method.
    
    Args:
        item_params: DataFrame with IRT parameters
        dataset_column: Column name for dataset grouping
        number_items: Fixed number of anchor items per dataset (from notebook, default=100)
        method: Selection method - "irt_clustering"/"anchor-irt", "correctness_clustering"/"anchor", or "difficulty_binning"
        matrix_df: Optional matrix for correctness-based clustering
    """
    if dataset_column is None or dataset_column not in item_params.columns:
        config = AnchorConfig(method=method, number_items=number_items)
        return {"__all__": find_anchor_items(item_params, config)}
    
    out: dict[str, list[str]] = {}
    for ds, grp in item_params.groupby(dataset_column):
        # Ensure we don't request more anchors than available questions
        actual_number_items = min(number_items, len(grp))
        config = AnchorConfig(method=method, number_items=actual_number_items)
        
        # For correctness clustering, we need to filter matrix_df to this dataset
        if method in ("correctness_clustering", "anchor") and matrix_df is not None:
            # Filter matrix to this dataset and questions in this group
            dataset_matrix = matrix_df[
                (matrix_df["dataset"] == ds) & 
                (matrix_df["question_id"].isin(grp.index))
            ]
            anchor_ids, _ = find_anchor_items_clustering(grp, dataset_matrix, config)
            out[str(ds)] = anchor_ids
        else:
            out[str(ds)] = find_anchor_items(grp, config)
    return out


# Legacy functions for backward compatibility
def compute_anchor_scores(item_params: pd.DataFrame) -> pd.Series:
    """Legacy function: compute Fisher information scores.
    
    Kept for backward compatibility, but the clustering approach
    doesn't use these scores.
    """
    if not {"a", "b"}.issubset(item_params.columns):
        raise ValueError("item_params must have columns 'a' and 'b'")
    thetas = np.linspace(-2.0, 2.0, 9)
    infos = []
    for theta in thetas:
        p = 1.0 / (1.0 + np.exp(-item_params["a"] * (theta - item_params["b"])) )
        info = (item_params["a"] ** 2) * p * (1 - p)
        infos.append(info)
    mean_info = pd.concat(infos, axis=1).mean(axis=1)
    return mean_info


