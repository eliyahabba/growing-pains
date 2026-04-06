"""Chain calibration experiment with parallel task execution."""
from __future__ import annotations

import json
import multiprocessing
import os
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config.constants import (
    ANCHOR_IRT_CLUSTERING, ANCHOR_TOP_K, ANCHOR_CORRECTNESS,
    CHAIN_DIRECT, ERROR_METRICS, EXCLUDED_DATASETS,
    MAX_RETRIES, METHOD_CONCURRENT, METHOD_FIXED, MIN_ANCHORS_PER_DATASET,
)
from irt import TrainingConfig, train_item_parameters
from src.calibration import (
    build_anchor_items_for_fixed_calibration,
    precompute_thetas_from_all_anchors,
    select_anchors,
    select_anchors_for_dataset,
    train_irt_on_base,
)
from src.cleanup import register_cleanup_paths, setup_cleanup_handlers
from src.data_loading import PROJECT_ROOT, ExperimentConfig, group_all_datasets_together, load_all_datasets
from src.evaluation import run_discriminative_baseline_validation, run_random_baseline_validation, run_random_simple_baseline, run_validation
from src.io import cleanup_training_datasets, detect_gpus, round_df_for_save, round_for_json, save_df

multiprocessing.set_start_method('spawn', force=True)


# Sensible per-mode defaults that differ from the general defaults
_MODE_DEFAULTS: dict[str, dict] = {
    "tinybenchmarks": {"n_base_datasets": 1, "max_chain_length": 5},
    "lb_only":        {"n_base_datasets": 1, "max_chain_length": 5},
    "lb":             {"n_base_datasets": 1, "max_chain_length": 5},
    "mmlu_split":     {"n_base_datasets": 1},
    "mmlu_fields":    {"n_base_datasets": 1},
}


@dataclass
class ParallelChainConfig(ExperimentConfig):
    """Configuration for parallel chain linking experiments."""
    n_base_datasets: int = 6
    max_chain_length: int = 10
    shuffle_seed: int = 42
    random_seed: int = 1000
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "chain_parallel_b")
    data_source_mode: str = "helm_lite"
    filter_zero_variance: bool = False
    validate_dimensions: bool = True
    # epochs: concurrent full-retrain; epochs_fixed: fixed-anchor (fewer needed — anchors constrain scale)
    epochs: int = 2000
    epochs_fixed: int = 1000
    n_anchors_per_dataset: int = 100
    num_workers: int = 4
    target_dataset: str | None = None
    n_models_per_chain: int | None = None
    cleanup_cache: bool = True
    cleanup_training_data: bool = True
    force_resume: bool = False
    # All anchor methods are always run per task; this constant is here for chain-cache anchor selection
    chain_anchor_method: str = ANCHOR_IRT_CLUSTERING

    def __post_init__(self):
        for key, val in _MODE_DEFAULTS.get(self.data_source_mode, {}).items():
            if getattr(self, key) == self.__dataclass_fields__[key].default:
                setattr(self, key, val)


ALL_ANCHOR_METHODS = [ANCHOR_IRT_CLUSTERING, ANCHOR_TOP_K, ANCHOR_CORRECTNESS]


@dataclass
class ScenarioTask:
    """A single scenario task to be executed by a worker."""
    task_id: int
    distance: int
    method: str
    chain_list: list
    chain_str: str
    scenario_dir: str
    final_df_path: str
    target_train_df_path: str
    prev_irt_path: str | None
    prev_A_path: str | None
    prev_B_path: str | None
    prev_anchors: list | None
    prev_weights: list | None
    dims: list[int]
    epochs: int
    n_anchors_per_dataset: int
    filter_zero_variance: bool
    validate_dimensions: bool
    lr: float
    target_name: str
    train_models: list
    random_seed: int
    seed: int
    cleanup_training_data: bool
    cumulative_chain_time: float


def _extract_irt_matrices(irt_params: pd.DataFrame) -> tuple:
    """Extract best_dimension and A/B matrices from irt_params.attrs."""
    best_dimension = A_matrix = B_matrix = None
    if hasattr(irt_params, 'attrs') and irt_params.attrs:
        best_dimension = irt_params.attrs.get('best_dimension')
        A_list = irt_params.attrs.get('A_matrix')
        B_list = irt_params.attrs.get('B_matrix')
        if A_list is not None and B_list is not None:
            A_matrix, B_matrix = np.array(A_list), np.array(B_list)
    return best_dimension, A_matrix, B_matrix


def _load_task_data(task: ScenarioTask) -> tuple:
    """Load pickle inputs for a scenario task."""
    final_df = pd.read_pickle(task.final_df_path)
    target_train_df = pd.read_pickle(task.target_train_df_path) if task.distance >= 1 else pd.DataFrame()

    anchor_items = None
    if task.method == METHOD_FIXED and task.prev_irt_path:
        prev_irt = pd.read_pickle(task.prev_irt_path)
        prev_A = np.load(task.prev_A_path) if task.prev_A_path else None
        prev_B = np.load(task.prev_B_path) if task.prev_B_path else None
        anchor_items = build_anchor_items_for_fixed_calibration(
            prev_irt, set(final_df['question_id'].astype(str).unique()),
            prev_A, prev_B, task.prev_anchors,
        )
    return final_df, target_train_df, anchor_items


def _train_irt_with_retry(
    final_df: pd.DataFrame, irt_config: TrainingConfig, output_dir: Path,
    anchor_items, task_id: int, distance: int, method: str, chain_list: list, chain_str: str,
) -> tuple[pd.DataFrame | None, dict | None]:
    for attempt in range(MAX_RETRIES):
        try:
            return train_item_parameters(final_df, config=irt_config, output_dir=str(output_dir), anchor_items=anchor_items), None
        except Exception as e:
            print(f"      Task {task_id} attempt {attempt+1}/{MAX_RETRIES} failed: {str(e)[:100]}")
    print(f"      Task {task_id}: all retries failed")
    return None, {'task_id': task_id, 'distance': distance, 'method': method,
                  'chain': chain_list, 'chain_str': chain_str, 'failed': True}


def _add_metrics(result: dict, df, prefix: str) -> None:
    if df is None or len(df) == 0:
        return
    result[f'{prefix}_n_models'] = len(df)
    for metric in ERROR_METRICS:
        if metric in df.columns:
            vals = df[metric].dropna()
            if len(vals) > 0:
                result[f'{prefix}_{metric}_mean'] = float(vals.mean())
                result[f'{prefix}_{metric}_std'] = float(vals.std())
    if 'true_performance' in df.columns:
        result[f'{prefix}_true_perf_mean'] = float(df['true_performance'].mean())
        result[f'{prefix}_true_perf_std'] = float(df['true_performance'].std())


def run_scenario_task(task: ScenarioTask, gpu_id: int | None = None) -> dict:
    """Execute a single scenario task (runs in a worker process)."""
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    final_df, target_train_df, anchor_items = _load_task_data(task)

    output_dir = Path(task.scenario_dir) / f"irt_{task.method}"
    output_dir.mkdir(parents=True, exist_ok=True)

    irt_config = TrainingConfig(
        dims_search=task.dims, epochs=task.epochs, lr=task.lr,
        number_item_per_scenario=task.n_anchors_per_dataset,
        deterministic=True, filter_zero_variance=task.filter_zero_variance,
        validate_dimensions=task.validate_dimensions,
    )
    start_time = time.time()
    irt_params, failure = _train_irt_with_retry(
        final_df, irt_config, output_dir, anchor_items,
        task.task_id, task.distance, task.method, task.chain_list, task.chain_str,
    )
    if failure is not None:
        return failure
    training_time = time.time() - start_time

    if task.cleanup_training_data and output_dir.exists():
        cleanup_training_datasets(output_dir)

    n_items = len(irt_params)
    best_dimension, A_matrix, B_matrix = _extract_irt_matrices(irt_params)

    # Run all anchor selection methods; results keyed by method name
    anchors_by_method: dict[str, tuple[list, list]] = {}
    for anchor_method in ALL_ANCHOR_METHODS:
        if task.method == METHOD_CONCURRENT:
            a, w = select_anchors(
                irt_params, task.n_anchors_per_dataset, final_df, A_matrix, B_matrix,
                clustering_method=anchor_method,
            )
        else:
            ta, tw = select_anchors_for_dataset(
                irt_params, task.n_anchors_per_dataset, task.target_name, final_df, A_matrix, B_matrix,
                method=anchor_method,
            )
            if task.prev_anchors is not None:
                a = list(task.prev_anchors) + ta
                w = list(task.prev_weights) + tw
            else:
                a, w = ta, tw
        anchors_by_method[anchor_method] = (a, w)

    # Use irt_clustering anchors for coverage check and validation (primary method)
    all_anchors, all_weights = anchors_by_method[ANCHOR_IRT_CLUSTERING]

    anchor_counts = {
        ds: sum(1 for a in all_anchors if str(a).startswith(f"{ds}:"))
        for ds in sorted(target_train_df['dataset'].unique())
    }
    low = {ds: c for ds, c in anchor_counts.items() if c < MIN_ANCHORS_PER_DATASET}
    if low:
        raise ValueError(f"Too few anchors for datasets: {low} (need >= {MIN_ANCHORS_PER_DATASET})")

    result: dict = {
        'task_id': task.task_id, 'distance': task.distance,
        'method': task.method, 'chain': task.chain_list, 'chain_str': task.chain_str,
        'n_items': n_items, 'best_dimension': best_dimension,
        'training_time_sec': round(training_time, 2),
        'cumulative_chain_time': task.cumulative_chain_time, 'gpu_id': gpu_id,
        'min_anchors_per_eval_dataset': int(min(anchor_counts.values())) if anchor_counts else 0,
    }

    per_model_dfs: dict = {}

    if not target_train_df.empty:
        base_seed = task.random_seed + task.distance * 100

        # Validation and baselines for each anchor method
        for anchor_method, (anchors, weights) in anchors_by_method.items():
            result[f'n_anchors_{anchor_method}'] = len(anchors)
            thetas = precompute_thetas_from_all_anchors(
                test_df=target_train_df, item_params=irt_params,
                anchor_ids=anchors, A_matrix=A_matrix, B_matrix=B_matrix,
            )
            res = run_validation(target_train_df, irt_params, anchors, weights,
                                 final_df, A_matrix, B_matrix, thetas)
            val_df = pd.DataFrame(res) if res else None
            _add_metrics(result, val_df, anchor_method)
            per_model_dfs[f'validation_{anchor_method}'] = val_df

        # Baselines (independent of anchor selection method)
        disc, disc_pm = run_discriminative_baseline_validation(
            test_df=target_train_df, item_params=irt_params, n_anchors=task.n_anchors_per_dataset,
            target_name=task.target_name, train_df=final_df, A_matrix=A_matrix, B_matrix=B_matrix,
            precomputed_thetas=None, return_per_model=True,
        )
        # rand_irt: random anchor selection but still uses the IRT model for theta estimation
        # rand_simple: no IRT — just averages raw scores on random questions
        rand_irt, rand_irt_pm = run_random_baseline_validation(
            test_df=target_train_df, item_params=irt_params, n_random_questions=task.n_anchors_per_dataset,
            target_name=task.target_name, train_df=final_df, A_matrix=A_matrix, B_matrix=B_matrix,
            precomputed_thetas=None, n_seeds=1, base_seed=base_seed, return_per_model=True,
        )
        rand_simple, rand_simple_pm = run_random_simple_baseline(
            test_df=target_train_df, target_name=task.target_name,
            n_random_questions=task.n_anchors_per_dataset, n_seeds=1, base_seed=base_seed,
            return_per_model=True,
        )
        result.update({f'disc_{k}': v for k, v in disc.items()})
        result.update({f'rand_irt_{k}': v for k, v in rand_irt.items()})
        result.update({f'rand_simple_{k}': v for k, v in rand_simple.items()})
        per_model_dfs.update({'discriminative_irt': disc_pm, 'random_irt': rand_irt_pm, 'random_simple': rand_simple_pm})

    for name, df in per_model_dfs.items():
        save_df(df, name, output_dir, task.method)
    return result


def _build_scenario_tasks(
    config: ParallelChainConfig,
    datasets: dict,
    target_name: str,
    chain_pool: list,
    chain_cache: dict,
    chain_cache_times: dict,
    max_chain: int,
    base_df: pd.DataFrame,
    base_irt: pd.DataFrame,
    A_base, B_base,
    base_anchors: list,
    base_weights: list,
    train_models: set,
    temp_dir: Path,
    output_dir: Path,
) -> tuple[list, list]:
    """Build ScenarioTask list for parallel execution. Returns (tasks, already_done)."""
    target_df = datasets[target_name]
    target_train_df = target_df[target_df['model_name'].isin(train_models)].copy()
    target_train_path = temp_dir / "target_train.pkl"
    target_train_df.to_pickle(target_train_path)

    base_irt_pkl = temp_dir / "base_irt.pkl"
    base_A_path = temp_dir / "base_A.npy"
    base_B_path = temp_dir / "base_B.npy"
    base_irt.to_pickle(base_irt_pkl)
    if A_base is not None:
        np.save(base_A_path, A_base)
    if B_base is not None:
        np.save(base_B_path, B_base)

    for dist, (irt, A, B, anchors, weights, df) in chain_cache.items():
        irt.to_pickle(temp_dir / f"chain_{dist}_irt.pkl")
        if A is not None:
            np.save(temp_dir / f"chain_{dist}_A.npy", A)
        if B is not None:
            np.save(temp_dir / f"chain_{dist}_B.npy", B)

    tasks: list = []
    already_done: list = []
    task_id = 0

    # Build a flat list of (distance, chain_str, chain_list, prev_irt_path, prev_A_path, prev_B_path,
    #                        prev_anchors, prev_weights, prev_df, A_for_dims, cumulative_time)
    def _dist_entries():
        # distance=1: direct (base IRT → target)
        yield (1, CHAIN_DIRECT, [], str(base_irt_pkl),
               str(base_A_path) if A_base is not None else None,
               str(base_B_path) if B_base is not None else None,
               list(base_anchors), list(base_weights), base_df, A_base, 0)
        # distance>=2: chain steps
        for dist in range(2, max_chain + 2):
            cache_idx = dist - 1
            if cache_idx not in chain_cache:
                print(f"   Distance {dist}: skipped (chain not built)")
                continue
            cl = chain_pool[:cache_idx]
            irt, A, B, pa, pw, pf = chain_cache[cache_idx]
            yield (dist, "_".join([d.replace(' ', '_')[:10] for d in cl]), cl,
                   str(temp_dir / f"chain_{cache_idx}_irt.pkl"),
                   str(temp_dir / f"chain_{cache_idx}_A.npy") if A is not None else None,
                   str(temp_dir / f"chain_{cache_idx}_B.npy") if B is not None else None,
                   pa, pw, pf, A,
                   sum(chain_cache_times.get(j, 0) for j in range(1, cache_idx + 1)))

    for (distance, chain_str, chain_list, prev_irt_path, prev_A_path, prev_B_path,
         prev_anchors, prev_weights, prev_df, A_for_dims, cumulative_chain_time) in _dist_entries():

        scenario_dir = output_dir / f"dist_{distance}_{chain_str}"
        results_file = scenario_dir / "results.json"
        if config.force_resume and results_file.exists():
            with open(results_file) as f:
                already_done.append(json.load(f))
            continue

        scenario_dir.mkdir(exist_ok=True)
        final_df = pd.concat([prev_df, target_train_df], ignore_index=True)
        final_df_path = temp_dir / f"final_df_dist_{distance}.pkl"
        final_df.to_pickle(final_df_path)

        dims = [A_for_dims.shape[1] if A_for_dims.ndim == 3 else A_for_dims.shape[0]] if A_for_dims is not None else config.dims_search

        for method in [METHOD_FIXED, METHOD_CONCURRENT]:
            task_epochs = config.epochs_fixed if method == METHOD_FIXED else config.epochs
            tasks.append(ScenarioTask(
                task_id=task_id, distance=distance, method=method,
                chain_list=chain_list, chain_str=chain_str,
                scenario_dir=str(scenario_dir),
                final_df_path=str(final_df_path),
                target_train_df_path=str(target_train_path),
                prev_irt_path=prev_irt_path if method == METHOD_FIXED else None,
                prev_A_path=prev_A_path if method == METHOD_FIXED else None,
                prev_B_path=prev_B_path if method == METHOD_FIXED else None,
                prev_anchors=prev_anchors, prev_weights=prev_weights,
                dims=dims, epochs=task_epochs,
                n_anchors_per_dataset=config.n_anchors_per_dataset,
                filter_zero_variance=config.filter_zero_variance,
                validate_dimensions=config.validate_dimensions,
                lr=config.lr, target_name=target_name,
                train_models=list(train_models),
                seed=config.seed, random_seed=config.random_seed,
                cleanup_training_data=config.cleanup_training_data,
                cumulative_chain_time=cumulative_chain_time,
            ))
            task_id += 1

    return tasks, already_done


def _build_chain_cache(
    config: ParallelChainConfig,
    chain_pool: list[str],
    datasets: dict,
    base_irt: pd.DataFrame,
    A_base, B_base,
    base_anchors: list,
    base_weights: list,
    base_df: pd.DataFrame,
    train_models: set,
    chain_cache_dir: Path,
) -> tuple[dict, dict, list, float]:
    """Build the sequential chain cache. Returns (chain_cache, chain_cache_times, successful_chain, total_time)."""
    max_chain = min(config.max_chain_length, len(chain_pool))
    chain_cache: dict = {}
    chain_cache_times: dict = {}
    current_irt = base_irt
    current_A, current_B = A_base, B_base
    current_anchors = list(base_anchors)
    current_weights = list(base_weights)
    current_df = base_df.copy()
    checkpoint_file = chain_cache_dir / "checkpoint.pkl"
    total_chain_time = 0.0
    successful_chain: list = []

    if checkpoint_file.exists() and config.force_resume:
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        successful_chain = checkpoint['successful_chain']
        chain_cache.update(checkpoint['chain_cache'])
        chain_cache_times.update(checkpoint.get('chain_cache_times', {}))
        if successful_chain:
            current_irt, current_A, current_B, current_anchors, current_weights, current_df = chain_cache[len(successful_chain)]
            total_chain_time = sum(chain_cache_times.get(j, 0) for j in range(1, len(successful_chain) + 1))
            print(f"   Resumed from step {len(successful_chain)}")

    for i in range(max_chain):
        chain_ds = chain_pool[i]
        if chain_ds in successful_chain:
            continue

        prefix = "_".join([d.replace(' ', '_')[:10] for d in successful_chain + [chain_ds]])
        cache_dir = chain_cache_dir / f"after_{prefix}"
        print(f"   Chain step {i+1}: {chain_ds}")

        chain_df = datasets[chain_ds][datasets[chain_ds]['model_name'].isin(train_models)].copy()
        combined_df = pd.concat([current_df, chain_df], ignore_index=True)
        available_questions = set(combined_df['question_id'].astype(str).unique())
        anchor_items = build_anchor_items_for_fixed_calibration(
            current_irt, available_questions, current_A, current_B, current_anchors)

        dims = [current_A.shape[1] if current_A.ndim == 3 else current_A.shape[0]] if current_A is not None else config.dims_search
        irt_config = TrainingConfig(
            dims_search=dims, epochs=config.epochs_fixed, lr=config.lr,
            number_item_per_scenario=config.n_anchors_per_dataset,
            deterministic=True, filter_zero_variance=config.filter_zero_variance,
            validate_dimensions=config.validate_dimensions,
        )

        chain_start = time.time()
        new_irt = None
        for attempt in range(MAX_RETRIES):
            try:
                new_irt = train_item_parameters(combined_df, config=irt_config, output_dir=str(cache_dir), anchor_items=anchor_items)
                break
            except Exception as e:
                print(f"      Attempt {attempt+1}/{MAX_RETRIES} failed: {str(e)[:80]}")
        if new_irt is None:
            print(f"      Chain step {i+1} failed, skipping")
            continue

        chain_time = time.time() - chain_start
        total_chain_time += chain_time
        if config.cleanup_training_data and cache_dir.exists():
            cleanup_training_datasets(cache_dir)

        new_A, new_B = _extract_irt_matrices(new_irt)[1:]
        chain_anchors, chain_weights = select_anchors_for_dataset(
            new_irt, config.n_anchors_per_dataset, chain_ds, combined_df, new_A, new_B,
            method=config.chain_anchor_method)
        new_anchors = current_anchors + chain_anchors
        new_weights = current_weights + chain_weights

        successful_chain.append(chain_ds)
        distance = len(successful_chain)
        chain_cache[distance] = (new_irt, new_A, new_B, new_anchors, new_weights, combined_df)
        chain_cache_times[distance] = chain_time
        current_irt, current_A, current_B = new_irt, new_A, new_B
        current_anchors, current_weights, current_df = new_anchors, new_weights, combined_df

        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'successful_chain': successful_chain, 'chain_cache': chain_cache,
                        'chain_cache_times': chain_cache_times}, f)
        print(f"      {len(new_irt)} items, {len(new_anchors)} anchors, {chain_time:.1f}s")

    return chain_cache, chain_cache_times, successful_chain, total_chain_time


def _setup_experiment(config: ParallelChainConfig) -> tuple:
    """Load datasets, assign target/base/chain, create directories."""
    print("1. Loading datasets...")
    datasets = load_all_datasets(config)
    for ds in EXCLUDED_DATASETS:
        datasets.pop(ds, None)
    print(f"   {len(datasets)} datasets loaded")

    skill_to_datasets = group_all_datasets_together(datasets, min_common_models=4)
    if not skill_to_datasets:
        raise ValueError("No valid dataset groups found")
    all_dataset_names = list(skill_to_datasets.values())[0]

    np.random.seed(config.shuffle_seed)
    shuffled = list(all_dataset_names)
    np.random.shuffle(shuffled)
    if config.target_dataset:
        target_name = config.target_dataset
        shuffled = [d for d in shuffled if d != target_name]
    else:
        target_name = shuffled.pop(config.n_base_datasets)
    base_names = shuffled[:config.n_base_datasets]
    chain_pool = shuffled[config.n_base_datasets:]

    target_n_questions = int(datasets[target_name]['question_id'].nunique())
    output_dir = Path(config.output_dir)
    if f"target_{target_name}" not in output_dir.name:
        output_dir = output_dir.parent / f"{output_dir.name}_target_{target_name}"
        config.output_dir = output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / ".temp"
    temp_dir.mkdir(exist_ok=True)
    chain_cache_dir = output_dir / "chain_cache"
    chain_cache_dir.mkdir(exist_ok=True)
    register_cleanup_paths(temp_dir, chain_cache_dir, output_dir,
                           config.cleanup_training_data, config.cleanup_cache)

    print(f"   Target: {target_name}, Base: {base_names}, Chain pool: {len(chain_pool)}")
    return datasets, target_name, base_names, chain_pool, target_n_questions, output_dir, temp_dir, chain_cache_dir


def _train_test_split(config: ParallelChainConfig, datasets: dict, base_names: list,
                      target_name: str, target_n_questions: int, chain_pool: list, output_dir: Path) -> tuple[set, set, dict]:
    """Compute train/test model sets and save config.json. Returns (train_models, test_models, config_dict)."""
    print("3. Train/test split...")
    base_models = sorted({m for ds in base_names for m in datasets[ds]['model_name'].unique()})
    np.random.seed(config.seed)
    n_test = max(1, int(len(base_models) * config.test_ratio))
    test_models = set(np.random.choice(base_models, size=n_test, replace=False))
    train_models = set(base_models) - test_models
    print(f"   Train: {len(train_models)}, Test (held-out): {len(test_models)}")

    config_dict = {
        'n_base_datasets': config.n_base_datasets, 'max_chain_length': config.max_chain_length,
        'seed': config.seed, 'shuffle_seed': config.shuffle_seed,
        'num_workers': config.num_workers, 'epochs': config.epochs,
        'dims_search': config.dims_search, 'n_anchors_per_dataset': config.n_anchors_per_dataset,
        'n_models_per_chain': config.n_models_per_chain, 'base_datasets': base_names,
        'target_dataset': target_name, 'chain_pool': chain_pool,
        'target_n_questions': target_n_questions,
        'n_train_models': len(train_models), 'n_test_models': len(test_models),
        'train_models': sorted(train_models), 'test_models': sorted(test_models),
    }
    (output_dir / "config.json").write_text(json.dumps(round_for_json(config_dict), indent=2))
    return train_models, test_models, config_dict


def _train_base_irt(config: ParallelChainConfig, datasets: dict, base_names: list,
                    train_models: set, output_dir: Path) -> tuple:
    """Train base IRT, select anchors. Returns (base_df, base_irt, A_base, B_base, base_anchors, base_weights)."""
    print("4. Training Base IRT...")
    base_df = pd.concat(
        [datasets[ds][datasets[ds]['model_name'].isin(train_models)].copy() for ds in base_names],
        ignore_index=True,
    )
    base_irt_dir = output_dir / "irt_base"
    base_irt, A_base, B_base = train_irt_on_base(base_df, config, base_irt_dir)
    if config.cleanup_training_data and base_irt_dir.exists():
        cleanup_training_datasets(base_irt_dir)
    base_anchors, base_weights = select_anchors(
        base_irt, config.n_anchors_per_dataset, base_df, A_base, B_base,
        clustering_method=config.chain_anchor_method)
    print(f"   {len(base_irt)} items, {len(base_anchors)} anchors")
    return base_df, base_irt, A_base, B_base, base_anchors, base_weights


def _aggregate_distance_results(
    all_results: list, already_done: list, chain_cache: dict,
    max_chain: int, target_name: str, target_n_questions: int,
    config: ParallelChainConfig, output_dir: Path,
) -> list:
    """Merge new + resumed results, compute deltas. Returns sorted final_results."""
    all_results.sort(key=lambda x: (x['distance'], x['method']))
    final_results = list(already_done)
    processed = {r['distance'] for r in already_done}

    for distance in range(1, max_chain + 2):
        if distance in processed:
            continue
        cache_idx = distance - 1
        if cache_idx not in chain_cache and distance != 1:
            continue

        dist_results = [r for r in all_results if r['distance'] == distance]
        fixed_result = next((r for r in dist_results if r['method'] == METHOD_FIXED), None)
        concurrent_result = next((r for r in dist_results if r['method'] == METHOD_CONCURRENT), None)
        if not fixed_result and not concurrent_result:
            continue

        fixed_result = fixed_result or {}
        concurrent_result = concurrent_result or {}
        chain_list = fixed_result.get('chain', concurrent_result.get('chain', []))
        chain_str = fixed_result.get('chain_str', concurrent_result.get('chain_str', CHAIN_DIRECT))
        n_datasets = config.n_base_datasets + distance

        result: dict = {
            'target_dataset': target_name, 'distance': distance, 'chain': chain_list,
            'n_datasets_in_training': n_datasets, 'target_n_questions': target_n_questions,
            'n_anchors_per_dataset': config.n_anchors_per_dataset,
            'n_base_datasets': config.n_base_datasets,
            'cost_full_eval_target': target_n_questions,
            'cost_fixed_target_anchors': config.n_anchors_per_dataset,
            'cost_concurrent_all_anchors': config.n_anchors_per_dataset * n_datasets,
        }
        skip_keys = {'task_id', 'distance', 'method', 'chain', 'chain_str', 'gpu_id', 'failed'}
        for key, val in fixed_result.items():
            if key not in skip_keys:
                result[f'fixed_{key}'] = val
        for key, val in concurrent_result.items():
            if key not in skip_keys:
                result[f'concurrent_{key}'] = val

        for metric in ERROR_METRICS:
            fv = fixed_result.get(f'{metric}_mean')
            cv = concurrent_result.get(f'{metric}_mean')
            if fv is not None and cv is not None:
                result[f'delta_{metric}'] = fv - cv

        scenario_dir = output_dir / f"dist_{distance}_{chain_str}"
        scenario_dir.mkdir(exist_ok=True)
        with open(scenario_dir / "results.json", 'w') as f:
            json.dump(round_for_json({**result, 'chain': list(result['chain'])}), f, indent=2)
        final_results.append(result)

    final_results.sort(key=lambda x: x['distance'])
    return final_results


def _run_parallel_tasks(tasks: list, config: ParallelChainConfig) -> tuple[list, float]:
    if not tasks:
        return [], 0.0
    print(f"7. Running {len(tasks)} tasks ({config.num_workers} workers)...")
    gpu_ids = detect_gpus()
    task_args = [(task, gpu_ids[i % len(gpu_ids)] if gpu_ids else None) for i, task in enumerate(tasks)]
    all_results: list = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        futures = {executor.submit(worker_wrapper, args): args[0].task_id for args in task_args}
        for future in as_completed(futures):
            try:
                result = future.result()
                if not result.get('failed'):
                    all_results.append(result)
            except Exception as e:
                print(f"   Task {futures[future]} failed: {e}")
    parallel_time = time.time() - start
    print(f"   {len(all_results)}/{len(tasks)} completed in {parallel_time:.1f}s")
    return all_results, parallel_time


def _save_and_cleanup(final_results: list, output_dir: Path, temp_dir: Path,
                      chain_cache_dir: Path, config: ParallelChainConfig) -> pd.DataFrame:
    results_for_df = [
        {**r, 'chain': "_".join(r['chain']) if r.get('chain') else CHAIN_DIRECT}
        for r in final_results
    ]

    results_df = round_df_for_save(pd.DataFrame(results_for_df))
    results_df.to_parquet(output_dir / "all_results.parquet", compression='snappy', index=False)
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    (output_dir / "all_results.json").write_text(
        json.dumps(round_for_json([{**r, 'chain': list(r['chain'])} for r in final_results]), indent=2))

    print("9. Cleanup...")
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    if config.cleanup_cache and chain_cache_dir.exists():
        shutil.rmtree(chain_cache_dir, ignore_errors=True)
    return results_df


def _print_summary(final_results: list, target_name: str, config: ParallelChainConfig,
                   total_time: float, parallel_time: float) -> None:
    print("\n" + "=" * 70)
    print(f"SUMMARY — target: {target_name}  workers: {config.num_workers}")
    print(f"Total time: {total_time:.1f}s (parallel phase: {parallel_time:.1f}s)")
    print(f"\n{'Dist':<6} {'Chain':<20} {'Fixed':<10} {'Concurrent':<12} {'Delta':<10}")
    print("-" * 60)
    for r in final_results:
        dist = r['distance']
        chain = "_".join([c[:6] for c in r['chain']]) if r['chain'] else CHAIN_DIRECT
        if len(chain) > 18:
            chain = chain[:15] + "..."
        fixed_err = r.get('fixed_gp_irt_error_mean', float('nan'))
        concurrent_err = r.get('concurrent_gp_irt_error_mean', float('nan'))
        delta = r.get('delta_gp_irt_error', float('nan'))
        print(f"{dist:<6} {chain:<20} {fixed_err:<10.4f} {concurrent_err:<12.4f} {delta:+10.4f}")

    # Print mean gp_irt_error for each anchor method and each baseline
    print("\n" + "-" * 60)
    print("MEAN GP-IRT ERROR (fixed method, averaged over distances):")
    comparisons = {
        f"IRT-anchors/{am}": f'fixed_{am}_gp_irt_error_mean'
        for am in ALL_ANCHOR_METHODS
    }
    comparisons["Baseline/discriminative"] = 'fixed_disc_discriminative_gp_irt_error_mean'
    comparisons["Baseline/rand-irt"] = 'fixed_rand_irt_random_gp_irt_error_mean'
    comparisons["Baseline/rand-simple"] = 'fixed_rand_simple_simple_random_error_mean'
    for label, key in comparisons.items():
        vals = [r.get(key) for r in final_results if r.get(key) is not None]
        if vals:
            print(f"    {label:<35} {np.mean(vals):.4f}")


def worker_wrapper(args: tuple) -> dict:
    task, gpu_id = args
    return run_scenario_task(task, gpu_id)


def run_chain_linking_parallel(config: ParallelChainConfig):
    """Run the parallel chain linking experiment."""
    setup_cleanup_handlers()
    experiment_start = time.time()
    print(f"Chain calibration experiment (workers={config.num_workers})")

    datasets, target_name, base_names, chain_pool, target_n_questions, output_dir, temp_dir, chain_cache_dir = (
        _setup_experiment(config)
    )
    train_models, test_models, _ = _train_test_split(
        config, datasets, base_names, target_name, target_n_questions, chain_pool, output_dir)
    base_df, base_irt, A_base, B_base, base_anchors, base_weights = _train_base_irt(
        config, datasets, base_names, train_models, output_dir)

    print(f"5. Building chain cache ({min(config.max_chain_length, len(chain_pool))} steps)...")
    chain_cache, chain_cache_times, successful_chain, _ = _build_chain_cache(
        config=config, chain_pool=chain_pool, datasets=datasets,
        base_irt=base_irt, A_base=A_base, B_base=B_base,
        base_anchors=base_anchors, base_weights=base_weights, base_df=base_df,
        train_models=train_models, chain_cache_dir=chain_cache_dir,
    )
    chain_pool, max_chain = successful_chain, len(successful_chain)

    print("6. Preparing tasks...")
    tasks, already_done = _build_scenario_tasks(
        config=config, datasets=datasets, target_name=target_name,
        chain_pool=chain_pool, chain_cache=chain_cache, chain_cache_times=chain_cache_times,
        max_chain=max_chain, base_df=base_df, base_irt=base_irt,
        A_base=A_base, B_base=B_base, base_anchors=base_anchors, base_weights=base_weights,
        train_models=train_models, temp_dir=temp_dir, output_dir=output_dir,
    )
    print(f"   {len(tasks)} tasks ({len(already_done)} resumed)")
    del datasets

    all_results, parallel_time = _run_parallel_tasks(tasks, config)

    print("8. Aggregating...")
    final_results = _aggregate_distance_results(
        all_results=all_results, already_done=already_done,
        chain_cache=chain_cache, max_chain=max_chain,
        target_name=target_name, target_n_questions=target_n_questions,
        config=config, output_dir=output_dir,
    )

    results_df = _save_and_cleanup(final_results, output_dir, temp_dir, chain_cache_dir, config)
    total_time = time.time() - experiment_start
    _print_summary(final_results, target_name, config, total_time, parallel_time)
    print(f"\nResults saved to: {output_dir}")
    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chain Linking — old model + new data evaluation")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-base", type=int, default=6)
    parser.add_argument("--max-chain", type=int, default=10)
    parser.add_argument("--n-anchors-per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--dims", type=int, nargs="+", default=[5])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--epochs-fixed", type=int, default=1000)
    parser.add_argument("--data-source-mode", type=str, default="helm_lite",
                        choices=["helm_lite", "helm_classic", "lb_only", "lb", "reeval", "mmlu_split", "mmlu_fields"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-dataset", type=str, default=None)
    parser.add_argument("--n-models-per-chain", type=int, default=None)
    parser.add_argument("--keep-cache", dest="cleanup_cache", action="store_false", default=True,
                        help="Keep chain_cache dir after completion (default: delete)")
    parser.add_argument("--keep-training-data", dest="cleanup_training_data", action="store_false", default=True,
                        help="Keep *.jsonlines training files (default: delete)")
    parser.add_argument("--force-resume", action="store_true", default=False,
                        help="Skip already-completed distances")

    args = parser.parse_args()

    config = ParallelChainConfig(
        n_base_datasets=args.n_base,
        max_chain_length=args.max_chain,
        n_anchors_per_dataset=args.n_anchors_per_dataset,
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        dims_search=args.dims,
        epochs=args.epochs,
        epochs_fixed=args.epochs_fixed,
        data_source_mode=args.data_source_mode,
        num_workers=args.num_workers,
        target_dataset=args.target_dataset,
        n_models_per_chain=args.n_models_per_chain,
        cleanup_cache=args.cleanup_cache,
        cleanup_training_data=args.cleanup_training_data,
        force_resume=args.force_resume,
    )

    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    else:
        dims_str = "-".join(map(str, args.dims))
        config.output_dir = PROJECT_ROOT / "data" / (
            f"chain_{args.data_source_mode}_seed_{args.shuffle_seed}"
            f"_anchors_{args.n_anchors_per_dataset}_dims_{dims_str}"
        )

    run_chain_linking_parallel(config)
