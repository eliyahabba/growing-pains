"""Dataset loading and grouping for chain calibration experiments."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config.data_sources import get_data_source_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ExperimentConfig:
    """Configuration for cross-dataset equating experiments."""
    input_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "input")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cross_dataset_equating")
    data_source_mode: str = "lb"
    dims_search: list = field(default_factory=lambda: [2, 5])
    epochs: int = 2000
    lr: float = 0.1
    n_anchors_per_dataset: int = 100
    test_ratio: float = 0.25
    seed: int = 42
    force_retrain: bool = False
    filter_zero_variance: bool = False
    validate_dimensions: bool = True


def load_pickle_data(pickle_path: str) -> dict:
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def extract_from_pickle(pickle_data: dict, dataset_name: str, keys=None, key_pattern=None) -> pd.DataFrame:
    models = np.array(pickle_data.get('models', []))
    all_data = pickle_data.get('data', {})
    if keys:
        keys_to_use = [k for k in keys if k in all_data]
    elif key_pattern:
        keys_to_use = [k for k in all_data if key_pattern in k]
    else:
        keys_to_use = list(all_data.keys())
    if not keys_to_use:
        return pd.DataFrame()

    dfs = []
    for key in keys_to_use:
        item = all_data[key]
        scores = item.get('correctness', item.get('scores')) if isinstance(item, dict) else item
        if scores is None:
            continue
        scores = np.array(scores)
        if scores.ndim != 2:
            continue
        if scores.shape[0] == len(models):
            scores = scores.T
        elif scores.shape[1] != len(models):
            continue
        n_q = scores.shape[0]
        q_grid, m_grid = np.meshgrid(np.arange(n_q), np.arange(len(models)), indexing='ij')
        flat = scores.flatten()
        mask = ~np.isnan(flat)
        dfs.append(pd.DataFrame({
            'model_name': models[m_grid.flatten()[mask]],
            'question_id': [f"{dataset_name}:{key}:{q}" for q in q_grid.flatten()[mask]],
            'dataset': dataset_name,
            'sub_dataset': key,
            'normalized_score': flat[mask],
        }))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['model_name', 'question_id'])


def extract_from_parquet(parquet_df: pd.DataFrame, dataset_name: str, filter_pattern: str) -> pd.DataFrame:
    df = parquet_df[parquet_df['dataset_name'].str.contains(filter_pattern, case=False, na=False)].copy()
    if df.empty:
        return pd.DataFrame()
    df['question_id'] = dataset_name + ":" + df['dataset_name'].astype(str) + ":" + df['hf_split'].astype(str) + ":" + df['hf_index'].astype(str)
    return pd.DataFrame({
        'model_name': df['model_name'],
        'question_id': df['question_id'],
        'dataset': dataset_name,
        'sub_dataset': df['dataset_name'],
        'normalized_score': df['evaluation_score'],
    }).drop_duplicates(subset=['model_name', 'question_id'])


def load_all_datasets(config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    source_config = get_data_source_config(config.data_source_mode)
    datasets_config = source_config.get('datasets', {})
    paths_config = source_config.get('paths', {})
    print(f"   Data source mode: {config.data_source_mode}, {len(datasets_config)} datasets")

    input_dir = Path(paths_config.get('input_dir', config.input_dir))

    loaded_pickles: dict = {}
    datasets: dict = {}

    for dataset_name in sorted(datasets_config):
        ds_config = datasets_config[dataset_name]
        source_type = ds_config.get('source_type')
        source_file = ds_config.get('source_file')
        try:
            if source_type == 'pickle':
                pickle_path = input_dir / source_file
                if source_file not in loaded_pickles:
                    if not pickle_path.exists():
                        print(f"  Warning: {pickle_path} not found, skipping {dataset_name}")
                        continue
                    loaded_pickles[source_file] = load_pickle_data(str(pickle_path))
                df = extract_from_pickle(loaded_pickles[source_file], dataset_name,
                                         ds_config.get('pickle_keys'), ds_config.get('pickle_key_pattern'))

            else:
                print(f"  Warning: Unknown source type {source_type!r} for {dataset_name}")
                continue

            if not df.empty:
                datasets[dataset_name] = df
                print(f"  ✓ {dataset_name}: {df['question_id'].nunique()} questions, {df['model_name'].nunique()} models")
            else:
                print(f"  Warning: No data for {dataset_name}")
        except Exception as e:
            import traceback
            print(f"  Error loading {dataset_name}: {e}")
            traceback.print_exc()

    return datasets


def group_all_datasets_together(datasets: dict[str, pd.DataFrame], min_common_models: int = 4) -> dict[str, list[str]]:
    """Find the largest subset of datasets sharing at least min_common_models."""
    all_names = sorted(datasets.keys())
    if len(all_names) < 2:
        return {}
    model_sets = {ds: set(datasets[ds]['model_name'].unique()) for ds in all_names}
    print(f"\n   Finding common models across {len(all_names)} datasets...")

    all_common = set.intersection(*model_sets.values()) if model_sets else set()
    if len(all_common) >= min_common_models:
        print(f"   ✓ All {len(all_names)} datasets share {len(all_common)} common models")
        return {"All_Datasets": all_names}

    best_subset: list = []
    best_count = 0
    for i, ds1 in enumerate(all_names):
        for ds2 in all_names[i + 1:]:
            common = model_sets[ds1] & model_sets[ds2]
            if len(common) < min_common_models:
                continue
            subset, shared = [ds1, ds2], common
            for ds3 in all_names:
                if ds3 not in subset:
                    new_common = shared & model_sets[ds3]
                    if len(new_common) >= min_common_models:
                        subset.append(ds3)
                        shared = new_common
            if len(subset) > len(best_subset) or (len(subset) == len(best_subset) and len(shared) > best_count):
                best_subset, best_count = subset, len(shared)

    if len(best_subset) >= 2:
        best_subset = sorted(best_subset)
        print(f"   ✓ Found {len(best_subset)} datasets with {best_count} common models")
        return {"All_Datasets": best_subset}
    print(f"   ✗ No valid subset with >= {min_common_models} common models")
    return {}
