"""Base scoring frame calibration, fixed-anchor calibration, and anchor selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from irt import (
    TrainingConfig,
    fit_2pl_parameters,
    find_anchor_items_clustering,
    AnchorConfig,
    save_item_parameters,
    estimate_theta_from_anchors,
)
from irt.anchors import find_anchor_items_top_k_discrimination
from src.data_loading import ExperimentConfig


def load_irt_params_from_cache(output_dir: Path) -> tuple[pd.DataFrame | None, np.ndarray | None, np.ndarray | None]:
    parquet_path = output_dir / "item_params.parquet"
    meta_path = output_dir / "item_params.meta.json"
    if parquet_path.exists():
        try:
            item_params = pd.read_parquet(parquet_path)
            A_matrix = B_matrix = None
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                item_params.attrs = meta
                if 'A_matrix' in meta and 'B_matrix' in meta:
                    A_matrix = np.array(meta['A_matrix'])
                    B_matrix = np.array(meta['B_matrix'])
            return item_params, A_matrix, B_matrix
        except (OSError, ValueError, KeyError) as e:
            print(f"      Warning: Failed to load cached params: {e}")
    return None, None, None


def train_irt_on_base(
    train_base_df: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path,
    force_retrain: bool = False,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None]:
    """Train IRT model on base datasets (with caching). Returns (item_params, A_matrix, B_matrix)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force_retrain:
        item_params, A_matrix, B_matrix = load_irt_params_from_cache(output_dir)
        if item_params is not None:
            print(f"      ✓ Loaded cached IRT params from {output_dir.name}")
            return item_params, A_matrix, B_matrix

    irt_config = TrainingConfig(
        dims_search=config.dims_search, epochs=config.epochs, lr=config.lr,
        number_item_per_scenario=config.n_anchors_per_dataset, deterministic=True,
        filter_zero_variance=getattr(config, 'filter_zero_variance', True),
        validate_dimensions=getattr(config, 'validate_dimensions', True),
    )
    item_params = fit_2pl_parameters(train_base_df, config=irt_config, output_dir=str(output_dir))
    A_matrix = B_matrix = None
    if hasattr(item_params, 'attrs') and item_params.attrs:
        A_list = item_params.attrs.get('A_matrix')
        B_list = item_params.attrs.get('B_matrix')
        if A_list is not None and B_list is not None:
            A_matrix, B_matrix = np.array(A_list), np.array(B_list)
    save_item_parameters(item_params, str(output_dir / "item_params.parquet"))
    return item_params, A_matrix, B_matrix


def build_anchor_items_for_fixed_calibration(
    baseline_params: pd.DataFrame,
    available_questions: set[str],
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    selected_anchor_ids: list[str] | None = None,
) -> list[dict]:
    """Build anchor items from baseline parameters for Fixed-Anchor Calibration."""
    baseline_params = baseline_params.copy()
    baseline_params.index = baseline_params.index.astype(str)
    filter_to = available_questions & (set(str(a) for a in selected_anchor_ids) if selected_anchor_ids else available_questions)
    subset = baseline_params.loc[baseline_params.index.intersection(filter_to)]
    if subset.empty:
        raise ValueError("No overlap between baseline item params and current matrix for anchoring")

    baseline_qids = list(baseline_params.index)
    anchors = []
    for item_id in subset.index:
        anchor = {"item_id": item_id}
        if A_matrix is not None and B_matrix is not None:
            try:
                idx = baseline_qids.index(item_id)
                if A_matrix.ndim == 3:
                    anchor["discrimination_vector"] = A_matrix[0, :, idx].tolist()
                    anchor["difficulty_vector"] = B_matrix[0, :, idx].tolist()
                else:
                    anchor["discrimination_vector"] = A_matrix[:, idx].tolist()
                    anchor["difficulty_vector"] = B_matrix[:, idx].tolist()
            except (ValueError, IndexError):
                anchor["difficulty"] = float(subset.loc[item_id, "b"])
                anchor["discrimination"] = float(subset.loc[item_id, "a"])
        else:
            anchor["difficulty"] = float(subset.loc[item_id, "b"])
            anchor["discrimination"] = float(subset.loc[item_id, "a"])
        anchors.append(anchor)
    return anchors


def _attach_dataset_column(item_params: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Return item_params copy with a 'dataset' column mapped from train_df."""
    q2ds = train_df.groupby('question_id')['dataset'].first().to_dict()
    out = item_params.copy()
    out['dataset'] = out.index.map(lambda q: q2ds.get(q, 'unknown'))
    return out


def _select_anchors_for_ds_items(
    ds_items: pd.DataFrame,
    item_params: pd.DataFrame,
    ds_indices: list[int],
    n_anchors: int,
    method: str,
    A_matrix: np.ndarray | None,
    B_matrix: np.ndarray | None,
    train_df: pd.DataFrame | None,
    dataset_name: str,
) -> tuple[list[str], list[float]]:
    """Shared anchor selection logic for a pre-filtered set of dataset items."""
    if method == "top_k_discrimination":
        cfg = AnchorConfig(number_items=n_anchors, method=method)
        anchor_ids = [str(q) for q in find_anchor_items_top_k_discrimination(ds_items, cfg)]
        return anchor_ids, [1.0 / max(len(anchor_ids), 1)] * len(anchor_ids)

    ds_A = A_matrix[:, :, ds_indices] if A_matrix is not None else None
    ds_B = B_matrix[:, :, ds_indices] if B_matrix is not None else None

    ds_items_for_clustering = ds_items.copy()
    if hasattr(item_params, 'attrs'):
        ds_items_for_clustering.attrs = item_params.attrs.copy()
        if 'balance_weights' in item_params.attrs:
            orig_weights = np.array(item_params.attrs['balance_weights'])
            ds_items_for_clustering.attrs['balance_weights'] = orig_weights[ds_indices].tolist()

    bw = getattr(ds_items_for_clustering, 'attrs', {}).get('balance_weights')
    balance_weights = np.array(bw) if bw is not None else None

    matrix_df = None
    if method == 'correctness_clustering' and train_df is not None:
        ds_train = train_df[train_df['dataset'] == dataset_name]
        matrix_df = ds_train[['question_id', 'model_name', 'normalized_score']].copy()

    anchor_config = AnchorConfig(number_items=n_anchors, method=method, balance_weights=balance_weights)
    anchor_ids, anchor_weights = find_anchor_items_clustering(
        ds_items_for_clustering, matrix_df=matrix_df, config=anchor_config, A_matrix=ds_A, B_matrix=ds_B,
    )
    return anchor_ids, (anchor_weights.tolist() if hasattr(anchor_weights, 'tolist') else list(anchor_weights))


def select_anchors_for_dataset(
    item_params: pd.DataFrame,
    n_anchors: int,
    dataset_name: str,
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    method: str = "irt_clustering",
) -> tuple[list[str], list[float]]:
    """Select anchor items from a specific dataset."""
    tagged = _attach_dataset_column(item_params, train_df)
    ds_items = tagged[tagged['dataset'] == dataset_name].drop(columns=['dataset'])
    if len(ds_items) == 0:
        print(f"      Warning: No items for dataset {dataset_name}")
        return [], []
    n_anchors = min(n_anchors, len(ds_items))
    if n_anchors < 5:
        print(f"      Warning: {dataset_name} has only {len(ds_items)} items, need at least 5")
        return [], []
    all_qids = list(item_params.index)
    ds_indices = [all_qids.index(q) for q in ds_items.index if q in all_qids]
    try:
        anchor_ids, weights = _select_anchors_for_ds_items(
            ds_items, item_params, ds_indices, n_anchors, method, A_matrix, B_matrix, train_df, dataset_name)
        if method != "top_k_discrimination":
            print(f"      ✓ {dataset_name}: {len(anchor_ids)} anchors (method={method})")
        return anchor_ids, weights
    except (ValueError, RuntimeError) as e:
        print(f"      Warning: Failed to select anchors from {dataset_name}: {e}")
        return [], []


def select_anchors(
    item_params: pd.DataFrame,
    n_anchors_per_dataset: int,
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    clustering_method: str = "irt_clustering",
) -> tuple[list[str], list[float]]:
    """Select n_anchors per dataset from all datasets in item_params."""
    tagged = _attach_dataset_column(item_params, train_df)
    all_qids = list(item_params.index)
    all_anchor_ids: list = []
    all_anchor_weights: list = []

    for dataset in sorted(tagged['dataset'].unique()):
        if dataset == 'unknown':
            continue
        ds_items = tagged[tagged['dataset'] == dataset].drop(columns=['dataset'])
        if len(ds_items) == 0:
            continue
        n = min(n_anchors_per_dataset, len(ds_items))
        if n < 5:
            print(f"      Warning: {dataset} has only {len(ds_items)} items, skipping")
            continue
        ds_indices = [all_qids.index(q) for q in ds_items.index if q in all_qids]
        try:
            anchor_ids, weights = _select_anchors_for_ds_items(
                ds_items, item_params, ds_indices, n, clustering_method, A_matrix, B_matrix, train_df, dataset)
            all_anchor_ids.extend(anchor_ids)
            all_anchor_weights.extend(weights)
            if clustering_method != "top_k_discrimination":
                print(f"      ✓ {dataset}: {len(anchor_ids)} anchors")
        except (ValueError, RuntimeError) as e:
            print(f"      Warning: Failed to select anchors from {dataset}: {e}")

    return all_anchor_ids, all_anchor_weights


def precompute_thetas_from_all_anchors(
    test_df: pd.DataFrame,
    item_params: pd.DataFrame,
    anchor_ids: list[str],
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    """Precompute theta for each test model using all anchors across all datasets."""
    question_ids_order = list(item_params.index) if hasattr(item_params, 'index') else None
    precomputed_thetas: dict = {}
    n_success = n_failed = 0
    for model_name in test_df['model_name'].unique():
        responses = test_df[test_df['model_name'] == model_name].set_index('question_id')['normalized_score']
        responses = responses[~responses.index.duplicated(keep='first')]
        available = [a for a in anchor_ids if a in responses.index and a in item_params.index]
        if len(available) < 3:
            n_failed += 1
            continue
        try:
            precomputed_thetas[model_name] = estimate_theta_from_anchors(
                item_params, responses.loc[available],
                A_matrix=A_matrix, B_matrix=B_matrix, question_ids_order=question_ids_order,
            )
            n_success += 1
        except Exception:
            n_failed += 1
    print(f"      Precomputed thetas: {n_success} success, {n_failed} failed")
    return precomputed_thetas
