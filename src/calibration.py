"""Base scoring frame calibration, fixed-anchor calibration, and anchor selection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from irt import (
    TrainingConfig,
    fit_2pl_parameters,
    find_anchor_items_clustering,
    AnchorConfig,
    train_item_parameters,
    save_item_parameters,
    estimate_theta_from_anchors,
)
from src.data_loading import ExperimentConfig, PROJECT_ROOT

# =============================================================================

# =============================================================================
# IRT Training & Validation
# =============================================================================

def load_irt_params_from_cache(output_dir: Path) -> tuple[pd.DataFrame | None, np.ndarray | None, np.ndarray | None]:
    """Try to load IRT parameters from cached files.
    
    Returns: (item_params, A_matrix, B_matrix) or (None, None, None) if not cached
    """
    # Check for item_params.parquet first (saved by save_item_parameters)
    parquet_path = output_dir / "item_params.parquet"
    meta_path = output_dir / "item_params.meta.json"
    
    # Also check for irt_dataset_final.jsonlines (created during training)
    jsonlines_path = output_dir / "irt_dataset_final.jsonlines"
    
    if parquet_path.exists():
        try:
            item_params = pd.read_parquet(parquet_path)
            
            # Load metadata if exists
            A_matrix = None
            B_matrix = None
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                item_params.attrs = meta
                
                if 'A_matrix' in meta and 'B_matrix' in meta:
                    A_matrix = np.array(meta['A_matrix'])
                    B_matrix = np.array(meta['B_matrix'])
            
            return item_params, A_matrix, B_matrix
        except Exception as e:
            print(f"      Warning: Failed to load cached params: {e}")
    
    return None, None, None


def train_irt_on_base(
    train_base_df: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path,
    force_retrain: bool = False,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None]:
    """Train IRT model on base datasets (with caching).
    
    Returns: (item_params, A_matrix, B_matrix)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load from cache first
    if not force_retrain:
        item_params, A_matrix, B_matrix = load_irt_params_from_cache(output_dir)
        if item_params is not None:
            print(f"      ✓ Loaded cached IRT params from {output_dir.name}")
            return item_params, A_matrix, B_matrix
    
    # Train new IRT model
    irt_config = TrainingConfig(
        dims_search=config.dims_search,
        epochs=config.epochs,
        lr=config.lr,
        number_item_per_scenario=config.n_anchors_per_dataset,
        deterministic=True,
        filter_zero_variance=getattr(config, 'filter_zero_variance', True),
        validate_dimensions=getattr(config, 'validate_dimensions', True),
    )
    
    item_params = fit_2pl_parameters(
        train_base_df,
        config=irt_config,
        output_dir=str(output_dir),
    )
    
    # Extract matrices
    A_matrix = None
    B_matrix = None
    if hasattr(item_params, 'attrs') and item_params.attrs:
        A_list = item_params.attrs.get('A_matrix')
        B_list = item_params.attrs.get('B_matrix')
        if A_list is not None and B_list is not None:
            A_matrix = np.array(A_list)
            B_matrix = np.array(B_list)
    
    # Save for future caching
    save_item_parameters(item_params, str(output_dir / "item_params.parquet"))
    
    return item_params, A_matrix, B_matrix


def build_anchor_items_for_fixed_calibration(
    baseline_params: pd.DataFrame,
    available_questions: set[str],
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    selected_anchor_ids: list[str] | None = None,
) -> list[dict]:
    """Build anchor items from baseline parameters for Fixed-Anchor Calibration.
    
    These anchors are passed to train_item_parameters to FREEZE the Base item
    parameters while training Link items on the same scale.
    
    Args:
        baseline_params: DataFrame with IRT parameters indexed by question_id
        available_questions: Set of question IDs available in the combined dataset
        A_matrix: Full discrimination matrix, shape (1, D, n_items) or (D, n_items)
        B_matrix: Full difficulty matrix, shape (1, D, n_items) or (D, n_items)
        selected_anchor_ids: If provided, only freeze these specific items (faster training).
                            If None, freeze all items in baseline_params (original behavior).
    
    Returns:
        List of anchor item dicts with either vector or scalar parameters
    """
    baseline_params = baseline_params.copy()
    baseline_params.index = baseline_params.index.astype(str)
    
    # Filter to selected anchors if provided
    if selected_anchor_ids is not None:
        selected_set = set(str(a) for a in selected_anchor_ids)
        filter_to = available_questions & selected_set
    else:
        filter_to = available_questions
    
    subset = baseline_params.loc[baseline_params.index.intersection(filter_to)]
    
    if subset.empty:
        raise ValueError("No overlap between baseline item params and current matrix for anchoring")
    
    baseline_qids = list(baseline_params.index)
    anchors = []
    
    for item_id in subset.index:
        anchor = {"item_id": item_id}
        
        # Try to use vector parameters if available (MIRT)
        if A_matrix is not None and B_matrix is not None:
            try:
                base_idx = baseline_qids.index(item_id)
                # Handle both (1, D, n_items) and (D, n_items) shapes
                if A_matrix.ndim == 3:
                    anchor["discrimination_vector"] = A_matrix[0, :, base_idx].tolist()
                    anchor["difficulty_vector"] = B_matrix[0, :, base_idx].tolist()
                else:
                    anchor["discrimination_vector"] = A_matrix[:, base_idx].tolist()
                    anchor["difficulty_vector"] = B_matrix[:, base_idx].tolist()
            except (ValueError, IndexError):
                # Fall back to scalar if vector extraction fails
                anchor["difficulty"] = float(subset.loc[item_id, "b"])
                anchor["discrimination"] = float(subset.loc[item_id, "a"])
        else:
            # Use scalar parameters
            anchor["difficulty"] = float(subset.loc[item_id, "b"])
            anchor["discrimination"] = float(subset.loc[item_id, "a"])
        
        anchors.append(anchor)
    
    return anchors


def select_anchors_for_dataset(
    item_params: pd.DataFrame,
    n_anchors: int,
    dataset_name: str,
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    method: str = "irt_clustering",
) -> tuple[list[str], list[float]]:
    """Select anchor items from a SPECIFIC dataset.
    
    Args:
        item_params: DataFrame with IRT parameters indexed by question_id
        n_anchors: Number of anchors to select
        dataset_name: Name of the dataset to select anchors from
        train_df: Training data to identify which questions belong to which dataset
        A_matrix, B_matrix: MIRT matrices for clustering
        method: Anchor selection method (irt_clustering | top_k_discrimination | correctness_clustering)
    
    Returns:
        (anchor_ids, anchor_weights) for the specified dataset
    """
    # Get dataset for each question
    question_to_dataset = train_df.groupby('question_id')['dataset'].first().to_dict()
    
    # Filter item_params to only the specified dataset
    item_params_with_dataset = item_params.copy()
    item_params_with_dataset['dataset'] = item_params_with_dataset.index.map(
        lambda q: question_to_dataset.get(q, 'unknown')
    )
    
    ds_mask = item_params_with_dataset['dataset'] == dataset_name
    ds_items = item_params_with_dataset[ds_mask].drop(columns=['dataset'])
    
    if len(ds_items) == 0:
        print(f"      Warning: No items found for dataset {dataset_name}")
        return [], []
    
    n_anchors = min(n_anchors, len(ds_items))
    
    if n_anchors < 5:
        print(f"      Warning: {dataset_name} has only {len(ds_items)} items, need at least 5")
        return [], []
    
    # For top_k_discrimination, skip the expensive O(n²) MIRT index lookup entirely —
    # the method only needs item_params["a"] and doesn't use A/B matrices at all.
    if method == "top_k_discrimination":
        anchor_config = AnchorConfig(number_items=n_anchors, method=method)
        try:
            from irt.anchors import find_anchor_items_top_k_discrimination
            anchor_ids = [str(q) for q in find_anchor_items_top_k_discrimination(ds_items, anchor_config)]
            uniform_w = [1.0 / max(len(anchor_ids), 1)] * len(anchor_ids)
            print(f"      ✓ {dataset_name}: {len(anchor_ids)} anchors selected (method={method})")
            return anchor_ids, uniform_w
        except Exception as e:
            print(f"      Warning: Failed top_k_discrimination for {dataset_name}: {e}")
            return [], []
    
    # For clustering methods: build MIRT sub-matrices (O(n²) index lookup but needed for K-means)
    all_question_ids = list(item_params.index)
    ds_indices = [all_question_ids.index(q) for q in ds_items.index if q in all_question_ids]
    
    # Extract sub-matrices for this dataset
    ds_A = A_matrix[:, :, ds_indices] if A_matrix is not None else None
    ds_B = B_matrix[:, :, ds_indices] if B_matrix is not None else None
    
    # Copy attrs to subset
    ds_items_for_clustering = ds_items.copy()
    if hasattr(item_params, 'attrs'):
        ds_items_for_clustering.attrs = item_params.attrs.copy()
        if 'balance_weights' in item_params.attrs:
            orig_weights = np.array(item_params.attrs['balance_weights'])
            ds_weights = orig_weights[ds_indices]
            ds_items_for_clustering.attrs['balance_weights'] = ds_weights.tolist()
    
    balance_weights = None
    if hasattr(ds_items_for_clustering, 'attrs'):
        bw = ds_items_for_clustering.attrs.get('balance_weights')
        if bw is not None:
            balance_weights = np.array(bw)
    
    anchor_config = AnchorConfig(
        number_items=n_anchors,
        method=method,
        balance_weights=balance_weights,
    )
    
    try:
        anchor_ids, anchor_weights = find_anchor_items_clustering(
            ds_items_for_clustering,
            config=anchor_config,
            A_matrix=ds_A,
            B_matrix=ds_B,
        )
        
        weights_list = anchor_weights.tolist() if hasattr(anchor_weights, 'tolist') else list(anchor_weights)
        print(f"      ✓ {dataset_name}: {len(anchor_ids)} anchors selected (method={method})")
        return anchor_ids, weights_list
        
    except Exception as e:
        print(f"      Warning: Failed to select anchors from {dataset_name}: {e}")
        return [], []


def select_anchors_pooled(
    item_params: pd.DataFrame,
    n_anchors: int,
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
) -> tuple[list[str], list[float]]:
    """Select anchor items from ALL datasets combined (pooled).
    
    Instead of selecting N anchors per dataset, this selects N anchors
    from the entire pool using IRT clustering on all questions together.
    
    Args:
        item_params: DataFrame with IRT parameters indexed by question_id
        n_anchors: Total number of anchors to select from the combined pool
        train_df: Training data (used for balance weights if available)
        A_matrix, B_matrix: MIRT matrices for clustering
    
    Returns:
        (anchor_ids, anchor_weights) from the combined pool
    """
    if len(item_params) == 0:
        print("      Warning: No items in item_params for pooled selection")
        return [], []
    
    n_anchors = min(n_anchors, len(item_params))
    
    if n_anchors < 5:
        print(f"      Warning: Only {len(item_params)} items available, need at least 5")
        return [], []
    
    # Get balance weights if available
    balance_weights = None
    if hasattr(item_params, 'attrs'):
        bw = item_params.attrs.get('balance_weights')
        if bw is not None:
            balance_weights = np.array(bw)
    
    anchor_config = AnchorConfig(
        number_items=n_anchors,
        method="irt_clustering",
        balance_weights=balance_weights,
    )
    
    try:
        anchor_ids, anchor_weights = find_anchor_items_clustering(
            item_params,
            config=anchor_config,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
        )
        
        weights_list = anchor_weights.tolist() if hasattr(anchor_weights, 'tolist') else list(anchor_weights)
        print(f"      ✓ Pooled selection: {len(anchor_ids)} anchors selected from {len(item_params)} total items")
        return anchor_ids, weights_list
        
    except Exception as e:
        print(f"      Warning: Failed to select pooled anchors: {e}")
        return [], []


def precompute_thetas_from_all_anchors(
    test_df: pd.DataFrame,
    item_params: pd.DataFrame,
    anchor_ids: list[str],
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    """Precompute theta for each test model using ALL anchors across all datasets.
    
    This allows cross-dataset theta estimation: we use anchors from Base datasets
    to estimate theta, then use that theta to predict on Link datasets.
    
    Args:
        test_df: Test data containing responses for all models on all datasets
        item_params: DataFrame with IRT parameters indexed by question_id
        anchor_ids: List of anchor question IDs (can be from multiple datasets)
        A_matrix, B_matrix: MIRT matrices
    
    Returns:
        Dict mapping model_name -> estimated theta
    """
    models = test_df['model_name'].unique()
    question_ids_order = list(item_params.index) if hasattr(item_params, 'index') else None
    
    precomputed_thetas = {}
    n_success = 0
    n_failed = 0
    
    for model_name in models:
        # Get all responses for this model
        model_responses = test_df[test_df['model_name'] == model_name].set_index('question_id')['normalized_score']
        model_responses = model_responses[~model_responses.index.duplicated(keep='first')]
        
        # Find anchors that have responses
        available_anchors = [a for a in anchor_ids if a in model_responses.index and a in item_params.index]
        
        if len(available_anchors) < 3:
            n_failed += 1
            continue
        
        anchor_responses = model_responses.loc[available_anchors]
        
        try:
            theta = estimate_theta_from_anchors(
                item_params,
                anchor_responses,
                A_matrix=A_matrix,
                B_matrix=B_matrix,
                question_ids_order=question_ids_order,
            )
            precomputed_thetas[model_name] = theta
            n_success += 1
        except Exception as e:
            n_failed += 1
    
    print(f"      Precomputed thetas: {n_success} success, {n_failed} failed")
    return precomputed_thetas


def select_anchors(
    item_params: pd.DataFrame,
    n_anchors_per_dataset: int,
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    clustering_method: str = "irt_clustering",
) -> tuple[list[str], list[float]]:
    """Select anchor items using clustering - n_anchors PER dataset.

    Args:
        item_params: DataFrame with IRT parameters indexed by question_id
        n_anchors_per_dataset: Number of anchors to select from EACH dataset
        train_df: Training data to identify which questions belong to which dataset
        A_matrix, B_matrix: MIRT matrices for clustering
        clustering_method: "irt_clustering" or "correctness_clustering"

    Returns:
        Combined anchor_ids and weights from all datasets
    """
    # Get dataset for each question
    question_to_dataset = train_df.groupby('question_id')['dataset'].first().to_dict()

    # Group item_params by dataset
    item_params_with_dataset = item_params.copy()
    item_params_with_dataset['dataset'] = item_params_with_dataset.index.map(
        lambda q: question_to_dataset.get(q, 'unknown')
    )

    # IMPORTANT: Sort for deterministic order across runs
    datasets = sorted(item_params_with_dataset['dataset'].unique())

    all_anchor_ids = []
    all_anchor_weights = []

    for dataset in datasets:
        if dataset == 'unknown':
            continue

        # Get items for this dataset
        ds_mask = item_params_with_dataset['dataset'] == dataset
        ds_items = item_params_with_dataset[ds_mask].drop(columns=['dataset'])

        if len(ds_items) == 0:
            continue

        # How many anchors to select from this dataset
        n_anchors = min(n_anchors_per_dataset, len(ds_items))

        if n_anchors < 5:
            print(f"      Warning: {dataset} has only {len(ds_items)} items, skipping")
            continue

        # Fast path for top_k_discrimination: skip O(n²) MIRT index lookup
        if clustering_method == "top_k_discrimination":
            from irt.anchors import find_anchor_items_top_k_discrimination
            _cfg = AnchorConfig(number_items=n_anchors, method="top_k_discrimination")
            try:
                anchor_ids = [str(q) for q in find_anchor_items_top_k_discrimination(ds_items, _cfg)]
                anchor_weights = [1.0 / max(len(anchor_ids), 1)] * len(anchor_ids)
                all_anchor_ids.extend(anchor_ids)
                all_anchor_weights.extend(anchor_weights)
                print(f"      ✓ {dataset}: {len(anchor_ids)} anchors selected (method={clustering_method})")
            except Exception as e:
                print(f"      Warning: top_k_discrimination failed for {dataset}: {e}")
            continue

        # Get indices for MIRT matrices (needed for clustering methods)
        all_question_ids = list(item_params.index)
        ds_indices = [all_question_ids.index(q) for q in ds_items.index if q in all_question_ids]

        # Extract sub-matrices for this dataset
        ds_A = A_matrix[:, :, ds_indices] if A_matrix is not None else None
        ds_B = B_matrix[:, :, ds_indices] if B_matrix is not None else None

        # Copy attrs to subset
        ds_items_for_clustering = ds_items.copy()
        if hasattr(item_params, 'attrs'):
            ds_items_for_clustering.attrs = item_params.attrs.copy()
            # Update balance weights for this subset
            if 'balance_weights' in item_params.attrs:
                orig_weights = np.array(item_params.attrs['balance_weights'])
                ds_weights = orig_weights[ds_indices]
                ds_items_for_clustering.attrs['balance_weights'] = ds_weights.tolist()

        balance_weights = None
        if hasattr(ds_items_for_clustering, 'attrs'):
            bw = ds_items_for_clustering.attrs.get('balance_weights')
            if bw is not None:
                balance_weights = np.array(bw)

        # For correctness_clustering, prepare matrix_df
        matrix_df = None
        if clustering_method == 'correctness_clustering':
            # Filter to this dataset in long format expected by anchor selector
            ds_train_df = train_df[train_df['dataset'] == dataset].copy()
            matrix_df = ds_train_df[['question_id', 'model_name', 'normalized_score']].copy()

        anchor_config = AnchorConfig(
            number_items=n_anchors,
            method=clustering_method,
            balance_weights=balance_weights,
        )

        try:
            anchor_ids, anchor_weights = find_anchor_items_clustering(
                ds_items_for_clustering,
                matrix_df=matrix_df,
                config=anchor_config,
                A_matrix=ds_A,
                B_matrix=ds_B,
            )

            all_anchor_ids.extend(anchor_ids)
            weights_list = anchor_weights.tolist() if hasattr(anchor_weights, 'tolist') else list(anchor_weights)
            all_anchor_weights.extend(weights_list)

            print(f"      ✓ {dataset}: {len(anchor_ids)} anchors selected")

        except Exception as e:
            print(f"      Warning: Failed to select anchors from {dataset}: {e}")

    return all_anchor_ids, all_anchor_weights

