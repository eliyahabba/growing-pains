"""
Chain Linking Experiment - Parallel Version B (Full Parallelization)

This version parallelizes ALL scenario tasks after the chain cache is built.
Each (distance, method) combination runs as an independent worker.

Key differences from V2:
- Uses multiprocessing to run scenarios in parallel
- Each worker is assigned to a specific GPU (if multiple available)
- Requires multiple GPUs for maximum speedup

Distance scheme (distance = number of datasets added beyond Base):
     0: Base only (no additions) - for Validation 3 baseline
     1: Base + Target (1 dataset added: Target)
     2: Base + Chain[0] + Target (1 chain + Target)
     3: Base + Chain[0] + Chain[1] + Target (2 chains + Target)
    ...

Parallelization structure:
    Sequential:
        1. Load datasets
        2. Train Base IRT
        3. Build Chain Cache (sequential - each step depends on previous)
    
    Parallel (all at once):
        - dist_0/concurrent (Base only - Validation 3 baseline)
        - dist_1/fixed, dist_1/concurrent (Base+Target)
        - dist_2/fixed, dist_2/concurrent (Base+Chain[0]+Target)
        - dist_3/fixed, dist_3/concurrent (Base+Chain[0]+Chain[1]+Target)
        - ...

Expected speedup: up to 2 × (max_chain + 2) with enough GPUs/workers

Usage:
    python chain_linking_parallel_b.py --num-workers 4 --output-dir /path/to/output
    
    # Or via shell script:
    sbatch run_chain_linking_parallel_b.sh /path/to/output 42 helm_classic "5" 4
"""

from __future__ import annotations

import atexit
import json

# Set multiprocessing start method before any other imports that might use it
import multiprocessing
import os
import pickle
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

multiprocessing.set_start_method('spawn', force=True)

from config.constants import (
    ERROR_METRICS,
    EXCLUDED_DATASETS,
    MAX_RETRIES,
    MIN_ANCHORS_PER_DATASET,
)
from irt import TrainingConfig, train_item_parameters
from src.experiments.equating.cross_dataset_equating import (
    PROJECT_ROOT,
    ExperimentConfig,
    build_anchor_items_for_fixed_calibration,
    group_all_datasets_together,
    load_all_datasets,
    precompute_thetas_from_all_anchors,
    run_discriminative_baseline_validation,
    run_random_baseline_validation,
    run_random_simple_baseline,
    run_validation,
    select_anchors,
    select_anchors_for_dataset,
    select_anchors_pooled,
    train_irt_on_base,
)
from src.experiments.utils.io import round_df_for_save, round_for_json

# =============================================================================
# Configuration
# =============================================================================

def cleanup_training_datasets(output_dir: Path) -> int:
    """Remove training dataset files (*.jsonlines) from an IRT output directory.
    
    These files are only needed during training and can be safely deleted after
    training completes to save ~88% storage in IRT directories.
    
    Args:
        output_dir: Path to IRT output directory
        
    Returns:
        Number of bytes freed
    """
    total_freed = 0
    jsonlines_files = list(output_dir.glob("*.jsonlines"))
    
    for jsonlines_file in jsonlines_files:
        try:
            size = jsonlines_file.stat().st_size
            jsonlines_file.unlink()
            total_freed += size
        except Exception:
            pass  # Silently ignore errors
    
    return total_freed


# Global cleanup state for crash recovery
_cleanup_paths = {
    'temp_dir': None,
    'chain_cache_dir': None,
    'output_dir': None,
    'cleanup_training_data': True,
    'cleanup_cache': True,
}


def register_cleanup_paths(temp_dir: Path, chain_cache_dir: Path, output_dir: Path,
                          cleanup_training_data: bool, cleanup_cache: bool):
    """Register paths for cleanup in case of crash."""
    _cleanup_paths['temp_dir'] = temp_dir
    _cleanup_paths['chain_cache_dir'] = chain_cache_dir
    _cleanup_paths['output_dir'] = output_dir
    _cleanup_paths['cleanup_training_data'] = cleanup_training_data
    _cleanup_paths['cleanup_cache'] = cleanup_cache


def emergency_cleanup():
    """Emergency cleanup function - called on crash/interrupt."""
    import shutil
    
    temp_dir = _cleanup_paths.get('temp_dir')
    chain_cache_dir = _cleanup_paths.get('chain_cache_dir')
    output_dir = _cleanup_paths.get('output_dir')
    cleanup_training_data = _cleanup_paths.get('cleanup_training_data', True)
    cleanup_cache = _cleanup_paths.get('cleanup_cache', True)
    
    print("\n🚨 Emergency cleanup triggered...")
    
    # Always clean .temp
    if temp_dir and temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            print(f"   ✅ Removed temp directory: {temp_dir}")
        except Exception as e:
            print(f"   ⚠️ Failed to remove temp: {e}")
    
    # Clean training data if enabled
    if cleanup_training_data and output_dir and output_dir.exists():
        try:
            freed_total = 0
            # Clean from irt_base
            irt_base = output_dir / "irt_base"
            if irt_base.exists():
                freed = cleanup_training_datasets(irt_base)
                freed_total += freed
            
            # Clean from chain_cache
            if chain_cache_dir and chain_cache_dir.exists():
                for cache_subdir in chain_cache_dir.glob("after_*"):
                    if cache_subdir.is_dir():
                        freed = cleanup_training_datasets(cache_subdir)
                        freed_total += freed
            
            # Clean from dist_* directories
            for dist_dir in output_dir.glob("dist_*"):
                if dist_dir.is_dir():
                    for irt_dir in dist_dir.glob("irt_*"):
                        if irt_dir.is_dir():
                            freed = cleanup_training_datasets(irt_dir)
                            freed_total += freed
            
            if freed_total > 0:
                freed_mb = freed_total / (1024 * 1024)
                print(f"   🧹 Cleaned training data: {freed_mb:.1f}MB freed")
        except Exception as e:
            print(f"   ⚠️ Failed to clean training data: {e}")
    
    # Clean cache if enabled
    if cleanup_cache and chain_cache_dir and chain_cache_dir.exists():
        try:
            shutil.rmtree(chain_cache_dir)
            print(f"   ✅ Removed chain cache: {chain_cache_dir}")
        except Exception as e:
            print(f"   ⚠️ Failed to remove cache: {e}")
    
    print("   Emergency cleanup complete")


def setup_cleanup_handlers():
    """Setup handlers to cleanup on crash/interrupt."""
    # Register atexit handler (called on normal exit and some crashes)
    atexit.register(lambda: None)  # Dummy to ensure atexit is initialized
    
    # Register signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, cleaning up...")
        emergency_cleanup()
        # Re-raise to allow normal signal handling
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    
    # Handle common termination signals
    signal.signal(signal.SIGTERM, signal_handler)  # SLURM job cancellation
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)  # Terminal closed


@dataclass
class ParallelChainConfig(ExperimentConfig):
    """Configuration for parallel chain linking experiments."""
    n_base_datasets: int = 6
    max_chain_length: int = 10
    shuffle_seed: int = 42
    random_seed: int = 1000  # Seed for random baseline scenarios
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "chain_parallel_b")
    data_source_mode: str = "helm_lite"
    filter_zero_variance: bool = False
    validate_dimensions: bool = True
    anchor_method: str = "irt_clustering"  # irt_clustering | top_k_discrimination | correctness_clustering
    epochs: int = 2000
    epochs_fixed: int = 1000
    n_anchors_per_dataset: int = 100
    num_workers: int = 4  # Number of parallel workers
    target_dataset: str | None = None  # Specific target dataset (if None, use shuffled[n_base])
    n_models_per_chain: int | None = None  # Number of models to use for chain steps (None = all train models)
    cleanup_cache: bool = True  # Remove chain_cache after successful completion
    cleanup_models: bool = False  # Remove IRT model files from dist_* directories (saves space, keeps only metrics)
    cleanup_training_data: bool = True  # Remove training datasets (*.jsonlines) immediately after training (saves ~88% per IRT dir)
    force_resume: bool = False  # Force resume existing experiment (skip auto-increment check)
    
    def __post_init__(self):
        # Auto-adjust for tinybenchmarks/lb (only 6 datasets available)
        if self.data_source_mode in ["tinybenchmarks", "lb_only", "lb"]:
            if self.n_base_datasets == 6:
                self.n_base_datasets = 1
            if self.max_chain_length == 10:
                self.max_chain_length = 5
        
        # Auto-adjust for mmlu_fields (57 datasets, recommend smaller base)
        if self.data_source_mode in ["mmlu_split", "mmlu_fields"]:
            self.n_base_datasets = 1
            # Recommend lower anchor count per dataset for mmlu_fields (many small datasets)
            if self.n_anchors_per_dataset == 100:
                print("   ⚠️ Note: For mmlu_fields mode with 57 datasets, consider using --n-anchors-per-dataset 10")
                print("      (100 anchors/dataset × 57 = 5700 total anchors may be excessive)")


# =============================================================================
# Worker Task Definition
# =============================================================================

@dataclass
class ScenarioTask:
    """A single scenario task to be executed by a worker."""
    task_id: int
    distance: int
    method: str  # 'fixed' or 'concurrent'
    chain_list: list
    chain_str: str
    scenario_dir: str

    # Data (will be serialized/deserialized)
    final_df_path: str  # Path to pickled DataFrame
    target_test_df_path: str
    base_chain_test_df_path: str | None  # Path to test models' responses on Base+Chain (for cross-dataset theta)
    target_train_df_path: str | None  # Path to train models' responses on Target (for old model + new data)
    target_all_train_df_path: str | None  # Path to ALL train_models' responses on Target (for linking generalization test)

    # IRT parameters (for Fixed-Anchor)
    prev_irt_path: str | None  # Path to pickled IRT params
    prev_A_path: str | None
    prev_B_path: str | None
    prev_anchors: list | None
    prev_weights: list | None

    # Config values
    dims: list[int]
    epochs: int
    n_anchors_per_dataset: int
    filter_zero_variance: bool
    validate_dimensions: bool
    lr: float
    target_name: str
    test_models: list  # Serialized as list
    random_seed: int  # Seed for random baseline scenarios
    train_models: list  # Serialized as list (for old model validation - chain_train_models)
    all_train_models: list  # ALL train_models for linking generalization test
    seed: int  # Base seed for random sampling
    cleanup_training_data: bool  # Whether to delete training jsonlines after training

    # Timing info
    cumulative_chain_time: float

    anchor_method: str = "irt_clustering"  # irt_clustering | top_k_discrimination | correctness_clustering


def run_scenario_task(task: ScenarioTask, gpu_id: int | None = None) -> dict:
    """Execute a single scenario task. This runs in a worker process.
    
    Special case for distance=0 (Base only):
        - Only runs Validation 3 (new model + old data)
        - Validations 1 & 2 are skipped (no target dataset involved)
        - Used as baseline for Validation 3 to measure improvement from adding Target
    
    Args:
        task: The scenario task definition
        gpu_id: Which GPU to use (if None, uses default)
    
    Returns:
        Result dictionary with all metrics
    """
    # Set GPU for this worker
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Import inside worker to ensure fresh CUDA context
    
    # Load data from disk
    final_df = pd.read_pickle(task.final_df_path)
    
    # For distance=0 (Base only), we don't need target test/train data
    if task.distance >= 1:
        target_test_df = pd.read_pickle(task.target_test_df_path)
    else:
        target_test_df = pd.DataFrame()  # Empty - not used for Base only
    
    test_models = set(task.test_models)
    train_models = set(task.train_models) if task.train_models else set()
    all_train_models = set(task.all_train_models) if task.all_train_models else set()
    
    # Load additional data for extra validations
    target_train_df = None
    if task.distance >= 1 and task.target_train_df_path:
        target_train_df = pd.read_pickle(task.target_train_df_path)
    
    # Load ALL train_models on target for linking generalization test
    target_all_train_df = None
    if task.distance >= 1 and task.target_all_train_df_path:
        target_all_train_df = pd.read_pickle(task.target_all_train_df_path)
    
    base_chain_test_df = None
    if task.base_chain_test_df_path:
        base_chain_test_df = pd.read_pickle(task.base_chain_test_df_path)
    
    # Load IRT params if Fixed-Anchor
    anchor_items = None
    prev_anchors = None
    prev_weights = None
    
    if task.method == 'fixed' and task.prev_irt_path:
        prev_irt = pd.read_pickle(task.prev_irt_path)
        prev_A = np.load(task.prev_A_path) if task.prev_A_path else None
        prev_B = np.load(task.prev_B_path) if task.prev_B_path else None
        
        available_questions = set(final_df['question_id'].astype(str).unique())
        anchor_items = build_anchor_items_for_fixed_calibration(
            prev_irt, available_questions, prev_A, prev_B, task.prev_anchors
        )
        prev_anchors = task.prev_anchors
        prev_weights = task.prev_weights
    else:
        # For concurrent: still need prev_anchors for validation (not training)
        prev_anchors = task.prev_anchors
        prev_weights = task.prev_weights
    
    # Create output directory
    output_dir = Path(task.scenario_dir) / f"irt_{task.method}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train IRT with retry
    irt_config = TrainingConfig(
        dims_search=task.dims,
        epochs=task.epochs,
        lr=task.lr,
        number_item_per_scenario=task.n_anchors_per_dataset,
        deterministic=True,
        filter_zero_variance=task.filter_zero_variance,
        validate_dimensions=task.validate_dimensions,
    )
    
    start_time = time.time()
    irt_params = None
    for attempt in range(MAX_RETRIES):
        try:
            irt_params = train_item_parameters(
                final_df,
                config=irt_config,
                output_dir=str(output_dir),
                anchor_items=anchor_items,
            )
            break
        except Exception as e:
            print(f"      ⚠️ Task {task.task_id} attempt {attempt+1}/{MAX_RETRIES} failed: {str(e)[:100]}")
            if attempt == MAX_RETRIES - 1:
                print(f"      ❌ Task {task.task_id} all retries failed")
                return {'task_id': task.task_id, 'distance': task.distance, 'method': task.method, 
                        'chain': task.chain_list, 'chain_str': task.chain_str, 'failed': True}
    training_time = time.time() - start_time
    
    # Clean up training datasets immediately after training (saves ~88% space)
    # Only keep item_params.parquet and metadata - training data no longer needed
    if task.cleanup_training_data and output_dir.exists():
        freed_bytes = cleanup_training_datasets(output_dir)
        if freed_bytes > 0:
            freed_mb = freed_bytes / (1024 * 1024)
            print(f"      🧹 Cleaned training data: {freed_mb:.1f}MB freed")
    
    # Extract info
    n_items = len(irt_params)
    best_dimension = None
    A_matrix, B_matrix = None, None
    
    if hasattr(irt_params, 'attrs') and irt_params.attrs:
        best_dimension = irt_params.attrs.get('best_dimension')
        A_list = irt_params.attrs.get('A_matrix')
        B_list = irt_params.attrs.get('B_matrix')
        if A_list is not None and B_list is not None:
            A_matrix = np.array(A_list)
            B_matrix = np.array(B_list)
    
    # Select anchors
    # Note: We need BOTH:
    #   - all_anchors (Base+Chain+Target) for Validation 1 & 2
    #   - prev_anchors (Base+Chain only) for Validation 3 (to avoid "cheating" with Target)

    anchor_method = getattr(task, 'anchor_method', 'irt_clustering')
    if task.method == 'concurrent':
        all_anchors, all_weights = select_anchors(
            irt_params, task.n_anchors_per_dataset, final_df, A_matrix, B_matrix,
            clustering_method=anchor_method,
        )
    else:
        target_anchors, target_weights = select_anchors_for_dataset(
            irt_params, task.n_anchors_per_dataset, task.target_name, final_df, A_matrix, B_matrix,
            method=anchor_method,
        )
        if prev_anchors is not None:
            all_anchors = list(prev_anchors) + target_anchors
            all_weights = list(prev_weights) + target_weights
        else:
            all_anchors = target_anchors
            all_weights = target_weights

    # For Validation 3: we need Base+Chain anchors only (without Target)
    # Filter out Target anchors from all_anchors
    target_prefix = f"{task.target_name}:"
    prev_anchors = [a for a in all_anchors if not str(a).startswith(target_prefix)]
    prev_weights = [w for a, w in zip(all_anchors, all_weights) if not str(a).startswith(target_prefix)]

    # Paper-grade sanity: ensure every dataset we evaluate has enough LOCAL anchors (prefix-based).
    datasets_to_check = set()
    
    # For distance=0 (Base only): only check base datasets
    # For distance>=1: check target + base/chain datasets
    if task.distance >= 1:
        datasets_to_check.update(target_test_df['dataset'].unique())
    
    if base_chain_test_df is not None and len(base_chain_test_df) > 0:
        datasets_to_check.update(base_chain_test_df['dataset'].unique())
    if task.distance >= 1 and target_train_df is not None and len(target_train_df) > 0:
        datasets_to_check.update(target_train_df['dataset'].unique())

    anchor_counts_by_dataset = {
        ds: sum(1 for a in all_anchors if str(a).startswith(f"{ds}:"))
        for ds in sorted(datasets_to_check)
    }
    low_anchor_datasets = {ds: c for ds, c in anchor_counts_by_dataset.items() if c < MIN_ANCHORS_PER_DATASET}
    if low_anchor_datasets:
        raise ValueError(
            "Anchor coverage check failed (too few local anchors for some evaluated datasets). "
            f"Need >= {MIN_ANCHORS_PER_DATASET} anchors per dataset. "
            f"Low: {low_anchor_datasets}"
        )
    
    # Prepare test data for theta precomputation
    # CRITICAL: Include test_models' responses on Base+Chain datasets (not just target)
    # This enables cross-dataset theta estimation using historical anchor responses
    if task.distance >= 1:
        # For distance>=1: include both target and base/chain datasets
        if base_chain_test_df is not None and len(base_chain_test_df) > 0:
            test_df = pd.concat([base_chain_test_df, target_test_df], ignore_index=True)
        else:
            test_df = target_test_df.copy()
    else:
        # For distance=0 (Base only): only base datasets
        test_df = base_chain_test_df.copy() if base_chain_test_df is not None else pd.DataFrame()
    
    # Precompute thetas for Validation 1 & 2 (uses ALL anchors including Target)
    precomputed_thetas = precompute_thetas_from_all_anchors(
        test_df=test_df,
        item_params=irt_params,
        anchor_ids=all_anchors,
        A_matrix=A_matrix,
        B_matrix=B_matrix,
    )
    
    # Precompute thetas for Validation 3: New Model + Old Data
    # IMPORTANT: Use only Base+Chain anchors (without Target) to avoid "cheating"
    precomputed_thetas_base_chain = None
    if base_chain_test_df is not None and len(base_chain_test_df) > 0 and prev_anchors is not None:
        precomputed_thetas_base_chain = precompute_thetas_from_all_anchors(
            test_df=base_chain_test_df,
            item_params=irt_params,
            anchor_ids=prev_anchors,  # Only Base+Chain anchors (no Target)
            A_matrix=A_matrix,
            B_matrix=B_matrix,
        )
    
    # ==========================================================================
    # Validation 1: New Models + New Dataset (test_models on target)
    # SKIP for distance=0 (Base only - no target dataset)
    # ==========================================================================
    validation_results = []
    validation_df = None

    if task.distance >= 1:  # Only run if target is involved
        validation_results = run_validation(
            test_df=target_test_df,
            item_params=irt_params,
            anchor_ids=all_anchors,
            anchor_weights=all_weights,
            train_df=final_df,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            precomputed_thetas=precomputed_thetas,
        )
        validation_df = pd.DataFrame(validation_results) if validation_results else None

    # ==========================================================================
    # Validation 2: Old Models + New Dataset (train_models on target)
    # SKIP for distance=0 (Base only - no target dataset)
    # ==========================================================================
    val_train_on_target_df = None
    precomputed_thetas_train = None  # Initialize before conditional block
    if task.distance >= 1 and target_train_df is not None and len(target_train_df) > 0:
        precomputed_thetas_train = precompute_thetas_from_all_anchors(
            test_df=final_df,  # final_df contains train_models on all datasets
            item_params=irt_params,
            anchor_ids=all_anchors,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
        )
        train_on_target_results = run_validation(
            test_df=target_train_df,
            item_params=irt_params,
            anchor_ids=all_anchors,
            anchor_weights=all_weights,
            train_df=final_df,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            precomputed_thetas=precomputed_thetas_train,
        )
        val_train_on_target_df = pd.DataFrame(train_on_target_results) if train_on_target_results else None
    
    # ==========================================================================
    # Validation 2b: ALL Train Models on New Dataset (linking generalization test)
    # Tests: "If we link with N models, can we predict ALL train models on Target?"
    # Uses Base+Chain anchors only for theta estimation (no Target "cheating")
    # This is the KEY metric for the model sweep experiment!
    # ==========================================================================
    val_all_train_on_target_df = None
    precomputed_thetas_all_train = None  # Initialize before conditional block
    if task.distance >= 1 and target_all_train_df is not None and len(target_all_train_df) > 0 and prev_anchors is not None:
        print(f"      Task {task.task_id}: Running Validation 2b (All Train Models on Target)...")
        
        # Precompute theta for ALL train_models using Base+Chain anchors only
        # This ensures fair comparison: theta is estimated without seeing Target data
        # For chain_train_models: they have responses in final_df (Base+Chain+Target)
        # For other train_models: they only have responses in Base (not in Chain or Target)
        
        # First, get Base data for ALL train_models (they all have Base responses)
        base_df_for_theta = final_df[final_df['model_name'].isin(all_train_models)].copy()
        
        precomputed_thetas_all_train = precompute_thetas_from_all_anchors(
            test_df=base_df_for_theta,  # Use Base data (includes all train_models)
            item_params=irt_params,
            anchor_ids=prev_anchors,  # Only Base+Chain anchors (no Target)
            A_matrix=A_matrix,
            B_matrix=B_matrix,
        )
        
        all_train_on_target_results = run_validation(
            test_df=target_all_train_df,
            item_params=irt_params,
            anchor_ids=prev_anchors,  # Use Base+Chain anchors
            anchor_weights=prev_weights,
            train_df=final_df,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            precomputed_thetas=precomputed_thetas_all_train,
        )
        val_all_train_on_target_df = pd.DataFrame(all_train_on_target_results) if all_train_on_target_results else None
        
        if val_all_train_on_target_df is not None:
            n_all = len(val_all_train_on_target_df)
            n_chain = len([m for m in val_all_train_on_target_df['model_name'].unique() if m in train_models])
            print(f"      Task {task.task_id}: Validated {n_all} models ({n_chain} from chain, {n_all - n_chain} generalized)")
    
    # ==========================================================================
    # Validation 3: New Models + Old Datasets (test_models on Base+Chain)
    # Uses only Base+Chain anchors for theta estimation (no Target "cheating")
    # ==========================================================================
    val_test_on_base_df = None

    if base_chain_test_df is not None and len(base_chain_test_df) > 0 and prev_anchors is not None:
        test_on_base_results = run_validation(
            test_df=base_chain_test_df,
            item_params=irt_params,
            anchor_ids=prev_anchors,  # Only Base+Chain anchors
            anchor_weights=prev_weights,  # Corresponding weights
            train_df=final_df,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            precomputed_thetas=precomputed_thetas_base_chain,  # Theta without Target
        )
        val_test_on_base_df = pd.DataFrame(test_on_base_results) if test_on_base_results else None

    # ==========================================================================
    # Validation 3 POOLED: IRT with N anchors from combined Base+Chain pool
    # Also runs for distance=0 (Base only)
    # ==========================================================================
    val_test_on_base_pooled_df = None
    pooled_irt_new_model_old_data = {}
    if base_chain_test_df is not None and len(base_chain_test_df) > 0:
        print(f"      Task {task.task_id}: Running Validation 3 POOLED IRT...")
        
        # Filter to Base+Chain questions
        base_chain_questions = base_chain_test_df['question_id'].unique()
        pooled_irt_params = irt_params[irt_params.index.isin(base_chain_questions)].copy()
        all_question_ids = list(irt_params.index)
        pooled_indices = [all_question_ids.index(q) for q in pooled_irt_params.index if q in all_question_ids]
        pooled_A = A_matrix[:, :, pooled_indices] if A_matrix is not None else None
        pooled_B = B_matrix[:, :, pooled_indices] if B_matrix is not None else None
        
        # Select N anchors using IRT clustering on combined pool
        pooled_anchors, pooled_weights = select_anchors_pooled(
            pooled_irt_params, task.n_anchors_per_dataset, final_df, pooled_A, pooled_B
        )
        
        if pooled_anchors:
            precomputed_thetas_pooled = precompute_thetas_from_all_anchors(
                base_chain_test_df, irt_params, pooled_anchors, A_matrix, B_matrix
            )
            test_on_base_pooled_results = run_validation(
                base_chain_test_df, irt_params, pooled_anchors, pooled_weights,
                final_df, A_matrix, B_matrix, precomputed_thetas_pooled
            )
            val_test_on_base_pooled_df = pd.DataFrame(test_on_base_pooled_results) if test_on_base_pooled_results else None
            
            # Find dataset column (may be 'dataset', 'dataset_name', or 'scenario_name')
            dataset_col = None
            if val_test_on_base_pooled_df is not None:
                for col in ['dataset', 'dataset_name', 'scenario_name']:
                    if col in val_test_on_base_pooled_df.columns:
                        dataset_col = col
                        break

            if val_test_on_base_pooled_df is not None and dataset_col is not None:
                pooled_irt_per_dataset_errors = {}
                for metric in ERROR_METRICS:
                    if metric in val_test_on_base_pooled_df.columns:
                        means = val_test_on_base_pooled_df.groupby(dataset_col)[metric].mean()
                        pooled_irt_per_dataset_errors = means.to_dict()
                        pooled_irt_new_model_old_data[f'pooled_irt_{metric}_mean'] = means.mean()
                        pooled_irt_new_model_old_data[f'pooled_irt_{metric}_std'] = means.std()
                pooled_irt_new_model_old_data['n_pooled_anchors'] = len(pooled_anchors)
                pooled_irt_new_model_old_data['per_dataset_errors'] = pooled_irt_per_dataset_errors
    
    # ==========================================================================
    # Discriminative Baselines for Validation 1: New Model + New Data
    # SKIP for distance=0 (Base only - no target dataset)
    # ==========================================================================
    discriminative_baseline_results = {}
    discriminative_baseline_per_model_df = pd.DataFrame()
    if task.distance >= 1:  # Only run if target is involved
        print(f"      Task {task.task_id}: Running discriminative baselines for Validation 1...")
        discriminative_baseline_results, discriminative_baseline_per_model_df = run_discriminative_baseline_validation(
            test_df=target_test_df,
            item_params=irt_params,
            n_anchors=task.n_anchors_per_dataset,
            target_name=task.target_name,
            train_df=final_df,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            precomputed_thetas=None,  # Keep baseline truly discriminative (theta from top-K anchors)
            return_per_model=True,
        )

    # ==========================================================================
    # Random Baselines for Validation 1: New Model + New Data
    # SKIP for distance=0 (Base only - no target dataset)
    # ==========================================================================
    random_baseline_results = {}
    random_simple_results = {}
    random_baseline_per_model_df = pd.DataFrame()
    random_simple_per_model_df = pd.DataFrame()
    if task.distance >= 1:  # Only run if target is involved
        print(f"      Task {task.task_id}: Running random baselines for Validation 1...")
        random_baseline_results, random_baseline_per_model_df = run_random_baseline_validation(
            test_df=target_test_df,
            item_params=irt_params,
            n_random_questions=task.n_anchors_per_dataset,
            target_name=task.target_name,
            train_df=final_df,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            precomputed_thetas=None,  # Keep baseline truly random (theta from random anchors)
            n_seeds=1,
            base_seed=task.random_seed + task.distance * 100,  # Random seed for baseline scenarios
            return_per_model=True,
        )
        random_simple_results, random_simple_per_model_df = run_random_simple_baseline(
            test_df=target_test_df,
            target_name=task.target_name,
            n_random_questions=task.n_anchors_per_dataset,
            n_seeds=1,
            base_seed=task.random_seed + task.distance * 100,  # Random seed for baseline scenarios
            return_per_model=True,
        )
    
    # ==========================================================================
    # Random Baselines for Validation 2: Old Model + New Data
    # SKIP for distance=0 (Base only - no target dataset)
    # ==========================================================================
    random_baseline_old_model = {}
    random_simple_old_model = {}
    random_baseline_old_model_per_model_df = pd.DataFrame()
    random_simple_old_model_per_model_df = pd.DataFrame()
    if task.distance >= 1 and target_train_df is not None and len(target_train_df) > 0:
        print(f"      Task {task.task_id}: Running random baselines for Validation 2...")
        random_baseline_old_model, random_baseline_old_model_per_model_df = run_random_baseline_validation(
            test_df=target_train_df,
            item_params=irt_params,
            n_random_questions=task.n_anchors_per_dataset,
            target_name=task.target_name,
            train_df=final_df,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            precomputed_thetas=None,  # Keep baseline truly random (theta from random anchors)
            n_seeds=1,
            base_seed=task.random_seed + task.distance * 100,  # Random seed for baseline scenarios
            return_per_model=True,
        )
        random_simple_old_model, random_simple_old_model_per_model_df = run_random_simple_baseline(
            test_df=target_train_df,
            target_name=task.target_name,
            n_random_questions=task.n_anchors_per_dataset,
            n_seeds=1,
            base_seed=task.random_seed + task.distance * 100,  # Random seed for baseline scenarios
            return_per_model=True,
        )
    
    # ==========================================================================
    # Discriminative Baselines for Validation 2b: All Train Models on New Data
    # KEY for model sweep!
    # ==========================================================================
    discriminative_baseline_all_train = {}
    discriminative_baseline_all_train_per_model_df = pd.DataFrame()
    if target_all_train_df is not None and len(target_all_train_df) > 0:
        print(f"      Task {task.task_id}: Running discriminative baselines for Validation 2b (All Train on Target)...")
        discriminative_baseline_all_train, discriminative_baseline_all_train_per_model_df = run_discriminative_baseline_validation(
            test_df=target_all_train_df,
            item_params=irt_params,
            n_anchors=task.n_anchors_per_dataset,
            target_name=task.target_name,
            train_df=final_df,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            precomputed_thetas=None,  # Keep baseline truly discriminative (theta from top-K anchors)
            return_per_model=True,
        )

    # ==========================================================================
    # Random Baselines for Validation 2b: All Train Models on New Data
    # This is the KEY random baseline for the model sweep experiment!
    # ==========================================================================
    random_baseline_all_train = {}
    random_simple_all_train = {}
    random_baseline_all_train_per_model_df = pd.DataFrame()
    random_simple_all_train_per_model_df = pd.DataFrame()
    if task.distance >= 1 and target_all_train_df is not None and len(target_all_train_df) > 0:
        print(f"      Task {task.task_id}: Running random baselines for Validation 2b (All Train on Target)...")
        random_baseline_all_train, random_baseline_all_train_per_model_df = run_random_baseline_validation(
            test_df=target_all_train_df,
            item_params=irt_params,
            n_random_questions=task.n_anchors_per_dataset,
            target_name=task.target_name,
            train_df=final_df,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            precomputed_thetas=None,  # Keep baseline truly random (theta from random anchors)
            n_seeds=1,
            base_seed=task.random_seed + task.distance * 100 + 500,  # Different seed offset
            return_per_model=True,
        )
        random_simple_all_train, random_simple_all_train_per_model_df = run_random_simple_baseline(
            test_df=target_all_train_df,
            target_name=task.target_name,
            n_random_questions=task.n_anchors_per_dataset,
            n_seeds=1,
            base_seed=task.random_seed + task.distance * 100 + 500,  # Different seed offset
            return_per_model=True,
        )
    
    # ==========================================================================
    # Discriminative Baselines for Validation 3: New Model + Old Data
    # ==========================================================================
    discriminative_baseline_new_model_old_data = {}
    discriminative_baseline_new_model_old_data_per_model_dfs = []
    if base_chain_test_df is not None and len(base_chain_test_df) > 0 and precomputed_thetas_base_chain is not None:
        print(f"      Task {task.task_id}: Running discriminative baselines for Validation 3...")
        base_chain_datasets = base_chain_test_df['dataset'].unique()
        all_discriminative_errors = []
        
        for ds_name in base_chain_datasets:
            ds_test_df = base_chain_test_df[base_chain_test_df['dataset'] == ds_name].copy()
            if len(ds_test_df) == 0:
                continue
            
            ds_disc, ds_disc_per_model = run_discriminative_baseline_validation(
                test_df=ds_test_df,
                item_params=irt_params,
                n_anchors=min(task.n_anchors_per_dataset, ds_test_df['question_id'].nunique()),
                target_name=ds_name,
                train_df=final_df,
                A_matrix=A_matrix,
                B_matrix=B_matrix,
                precomputed_thetas=None,
                return_per_model=True,
            )
            
            if 'discriminative_gp_irt_error_mean' in ds_disc:
                all_discriminative_errors.append(ds_disc['discriminative_gp_irt_error_mean'])
            
            if len(ds_disc_per_model) > 0:
                ds_disc_per_model['dataset'] = ds_name
                discriminative_baseline_new_model_old_data_per_model_dfs.append(ds_disc_per_model)
        
        if all_discriminative_errors:
            discriminative_baseline_new_model_old_data = {
                'discriminative_gp_irt_error_mean': np.mean(all_discriminative_errors),
                'discriminative_gp_irt_error_std': np.std(all_discriminative_errors),
                'n_datasets': len(all_discriminative_errors),
            }
    
    discriminative_baseline_new_model_old_data_per_model_df = pd.concat(discriminative_baseline_new_model_old_data_per_model_dfs) if discriminative_baseline_new_model_old_data_per_model_dfs else pd.DataFrame()

    # ==========================================================================
    # Random Baselines for Validation 3: New Model + Old Data
    # ==========================================================================
    random_baseline_new_model_old_data = {}
    random_simple_new_model_old_data = {}
    random_baseline_new_model_old_data_per_model_dfs = []
    random_simple_new_model_old_data_per_model_dfs = []
    if base_chain_test_df is not None and len(base_chain_test_df) > 0 and precomputed_thetas_base_chain is not None:
        print(f"      Task {task.task_id}: Running random baselines for Validation 3...")
        base_chain_datasets = base_chain_test_df['dataset'].unique()
        all_random_irt_errors = []
        all_random_simple_errors = []
        
        for ds_name in base_chain_datasets:
            ds_test_df = base_chain_test_df[base_chain_test_df['dataset'] == ds_name].copy()
            if len(ds_test_df) == 0:
                continue
            
            ds_random_irt, ds_random_irt_per_model = run_random_baseline_validation(
                test_df=ds_test_df,
                item_params=irt_params,
                n_random_questions=min(task.n_anchors_per_dataset, ds_test_df['question_id'].nunique()),
                target_name=ds_name,
                train_df=final_df,
                A_matrix=A_matrix,
                B_matrix=B_matrix,
                precomputed_thetas=None,  # Keep baseline truly random (theta from random anchors)
                n_seeds=1,
                base_seed=task.random_seed + task.distance * 100,  # Random seed for baseline scenarios
                return_per_model=True,
            )
            ds_random_simple, ds_random_simple_per_model = run_random_simple_baseline(
                test_df=ds_test_df,
                target_name=ds_name,
                n_random_questions=min(task.n_anchors_per_dataset, ds_test_df['question_id'].nunique()),
                n_seeds=1,
                base_seed=task.random_seed + task.distance * 100,  # Random seed for baseline scenarios
                return_per_model=True,
            )
            
            if 'random_gp_irt_error_mean' in ds_random_irt:
                all_random_irt_errors.append(ds_random_irt['random_gp_irt_error_mean'])
            if 'simple_random_error_mean' in ds_random_simple:
                all_random_simple_errors.append(ds_random_simple['simple_random_error_mean'])
            
            # Collect per-model results per dataset
            if len(ds_random_irt_per_model) > 0:
                ds_random_irt_per_model['dataset'] = ds_name
                random_baseline_new_model_old_data_per_model_dfs.append(ds_random_irt_per_model)
            if len(ds_random_simple_per_model) > 0:
                ds_random_simple_per_model['dataset'] = ds_name
                random_simple_new_model_old_data_per_model_dfs.append(ds_random_simple_per_model)
        
        # Aggregate across datasets
        if all_random_irt_errors:
            random_baseline_new_model_old_data = {
                'random_gp_irt_error_mean': np.mean(all_random_irt_errors),
                'random_gp_irt_error_std': np.std(all_random_irt_errors),
                'n_datasets': len(all_random_irt_errors),
            }
        if all_random_simple_errors:
            random_simple_new_model_old_data = {
                'simple_random_error_mean': np.mean(all_random_simple_errors),
                'simple_random_error_std': np.std(all_random_simple_errors),
            }
    
    # Combine Validation 3 per-model DataFrames
    random_baseline_new_model_old_data_per_model_df = pd.concat(random_baseline_new_model_old_data_per_model_dfs) if random_baseline_new_model_old_data_per_model_dfs else pd.DataFrame()
    random_simple_new_model_old_data_per_model_df = pd.concat(random_simple_new_model_old_data_per_model_dfs) if random_simple_new_model_old_data_per_model_dfs else pd.DataFrame()
    
    # ==========================================================================
    # Validation 3 POOLED Random: N random questions from combined Base+Chain pool
    # Computes BOTH simple error and GP-IRT error for fair comparison
    # ==========================================================================
    random_simple_pooled_new_model_old_data = {}
    random_irt_pooled_new_model_old_data = {}
    if base_chain_test_df is not None and len(base_chain_test_df) > 0:
        print(f"      Task {task.task_id}: Running Validation 3 POOLED Random...")
        
        all_pooled_questions = list(base_chain_test_df['question_id'].unique())
        n_pooled_anchors = min(task.n_anchors_per_dataset, len(all_pooled_questions))
        
        np.random.seed(task.seed + task.distance * 100 + 1000)
        pooled_random_questions = list(np.random.choice(all_pooled_questions, size=n_pooled_anchors, replace=False))
        
        score_col = 'normalized_score' if 'normalized_score' in base_chain_test_df.columns else 'score'
        pooled_per_dataset_errors = {}  # Save per-dataset errors (simple)
        
        for ds_name in base_chain_test_df['dataset'].unique():
            ds_df = base_chain_test_df[base_chain_test_df['dataset'] == ds_name]
            ds_pooled = ds_df[ds_df['question_id'].isin(pooled_random_questions)]
            if len(ds_pooled) == 0:
                continue
            
            true_perf = ds_df.groupby('model_name')[score_col].mean()
            pred_perf = ds_pooled.groupby('model_name')[score_col].mean()
            common_models = set(true_perf.index) & set(pred_perf.index) & set(test_models)
            if common_models:
                errors = [abs(pred_perf[m] - true_perf[m]) for m in common_models]
                pooled_per_dataset_errors[ds_name] = np.mean(errors)
        
        if pooled_per_dataset_errors:
            random_simple_pooled_new_model_old_data = {
                'pooled_simple_random_error_mean': np.mean(list(pooled_per_dataset_errors.values())),
                'pooled_simple_random_error_std': np.std(list(pooled_per_dataset_errors.values())),
                'n_pooled_anchors': n_pooled_anchors,
                'per_dataset_errors': pooled_per_dataset_errors,
            }
    
        # --- POOLED RANDOM with GP-IRT (for fair comparison with Pooled IRT) ---
        # Use random questions as anchors and compute GP-IRT error
        pooled_random_anchors = pooled_random_questions
        pooled_random_weights = [1.0] * len(pooled_random_anchors)  # Equal weights for random

        if pooled_random_anchors:
            precomputed_thetas_pooled_random = precompute_thetas_from_all_anchors(
                base_chain_test_df, irt_params, pooled_random_anchors, A_matrix, B_matrix
            )
            pooled_random_irt_results = run_validation(
                base_chain_test_df, irt_params, pooled_random_anchors, pooled_random_weights,
                final_df, A_matrix, B_matrix, precomputed_thetas_pooled_random
            )
            if pooled_random_irt_results:
                pooled_random_df = pd.DataFrame(pooled_random_irt_results)
                # Find dataset column
                pr_dataset_col = None
                for col in ['dataset', 'dataset_name', 'scenario_name']:
                    if col in pooled_random_df.columns:
                        pr_dataset_col = col
                        break

                if pr_dataset_col is not None:
                    for metric in ERROR_METRICS:
                        if metric in pooled_random_df.columns:
                            means = pooled_random_df.groupby(pr_dataset_col)[metric].mean()
                            random_irt_pooled_new_model_old_data[f'pooled_random_irt_{metric}_mean'] = means.mean()
                            random_irt_pooled_new_model_old_data[f'pooled_random_irt_{metric}_std'] = means.std()
                    random_irt_pooled_new_model_old_data['n_pooled_random_anchors'] = len(pooled_random_anchors)

    # ==========================================================================
    # Validation 3 PROPORTIONAL: N total anchors distributed by dataset size
    # ==========================================================================
    proportional_irt_new_model_old_data = {}
    proportional_random_new_model_old_data = {}
    proportional_random_irt_new_model_old_data = {}
    if base_chain_test_df is not None and len(base_chain_test_df) > 0:
        print(f"      Task {task.task_id}: Running Validation 3 PROPORTIONAL...")
        
        # Calculate proportional allocation per dataset
        datasets = base_chain_test_df['dataset'].unique()
        dataset_sizes = {ds: base_chain_test_df[base_chain_test_df['dataset'] == ds]['question_id'].nunique() for ds in datasets}
        total_questions = sum(dataset_sizes.values())
        n_total = task.n_anchors_per_dataset
        
        # Allocate proportionally with rounding (ensure exact total)
        raw_alloc = {ds: n_total * size / total_questions for ds, size in dataset_sizes.items()}
        alloc = {ds: int(np.floor(v)) for ds, v in raw_alloc.items()}
        remainder = n_total - sum(alloc.values())
        # Give remainder to datasets with largest fractional parts
        fractional = {ds: raw_alloc[ds] - alloc[ds] for ds in datasets}
        for ds in sorted(fractional, key=fractional.get, reverse=True)[:remainder]:
            alloc[ds] += 1
        
        score_col = 'normalized_score' if 'normalized_score' in base_chain_test_df.columns else 'score'
        np.random.seed(task.seed + task.distance * 100 + 2000)
        
        # --- PROPORTIONAL IRT ---
        prop_irt_per_dataset_errors = {}
        prop_anchors_all = []
        prop_weights_all = []
        for ds_name, n_ds in alloc.items():
            if n_ds < 1:
                continue
            ds_anchors, ds_weights = select_anchors_for_dataset(
                irt_params, n_ds, ds_name, final_df, A_matrix, B_matrix
            )
            prop_anchors_all.extend(ds_anchors)
            prop_weights_all.extend(ds_weights)
        
        if prop_anchors_all:
            prop_thetas = precompute_thetas_from_all_anchors(
                base_chain_test_df, irt_params, prop_anchors_all, A_matrix, B_matrix
            )
            prop_results = run_validation(
                base_chain_test_df, irt_params, prop_anchors_all, prop_weights_all,
                final_df, A_matrix, B_matrix, prop_thetas
            )
            if prop_results:
                prop_df = pd.DataFrame(prop_results)
                # Find dataset column (may be 'dataset', 'dataset_name', or 'scenario_name')
                prop_dataset_col = None
                for col in ['dataset', 'dataset_name', 'scenario_name']:
                    if col in prop_df.columns:
                        prop_dataset_col = col
                        break

                if prop_dataset_col is not None:
                    for metric in ERROR_METRICS:
                        if metric in prop_df.columns:
                            means = prop_df.groupby(prop_dataset_col)[metric].mean()
                            prop_irt_per_dataset_errors = means.to_dict()
                            proportional_irt_new_model_old_data[f'proportional_irt_{metric}_mean'] = means.mean()
                            proportional_irt_new_model_old_data[f'proportional_irt_{metric}_std'] = means.std()
                    proportional_irt_new_model_old_data['n_proportional_anchors'] = len(prop_anchors_all)
                    proportional_irt_new_model_old_data['allocation'] = alloc
                    proportional_irt_new_model_old_data['per_dataset_errors'] = prop_irt_per_dataset_errors
        
        # --- PROPORTIONAL RANDOM (Simple Error) ---
        prop_random_per_dataset_errors = {}
        prop_random_anchors_all = []
        for ds_name, n_ds in alloc.items():
            if n_ds < 1:
                continue
            ds_df = base_chain_test_df[base_chain_test_df['dataset'] == ds_name]
            ds_questions = list(ds_df['question_id'].unique())
            n_sample = min(n_ds, len(ds_questions))
            random_qs = list(np.random.choice(ds_questions, size=n_sample, replace=False))
            prop_random_anchors_all.extend(random_qs)
            
            ds_sampled = ds_df[ds_df['question_id'].isin(random_qs)]
            true_perf = ds_df.groupby('model_name')[score_col].mean()
            pred_perf = ds_sampled.groupby('model_name')[score_col].mean()
            common_models = set(true_perf.index) & set(pred_perf.index) & set(test_models)
            if common_models:
                errors = [abs(pred_perf[m] - true_perf[m]) for m in common_models]
                prop_random_per_dataset_errors[ds_name] = np.mean(errors)
        
        if prop_random_per_dataset_errors:
            proportional_random_new_model_old_data = {
                'proportional_random_error_mean': np.mean(list(prop_random_per_dataset_errors.values())),
                'proportional_random_error_std': np.std(list(prop_random_per_dataset_errors.values())),
                'n_proportional_anchors': sum(alloc.values()),
                'allocation': alloc,
                'per_dataset_errors': prop_random_per_dataset_errors,
            }
    
        # --- PROPORTIONAL RANDOM with GP-IRT (for fair comparison with Proportional IRT) ---
        if prop_random_anchors_all:
            prop_random_weights = [1.0] * len(prop_random_anchors_all)  # Equal weights for random
            precomputed_thetas_prop_random = precompute_thetas_from_all_anchors(
                base_chain_test_df, irt_params, prop_random_anchors_all, A_matrix, B_matrix
            )
            prop_random_irt_results = run_validation(
                base_chain_test_df, irt_params, prop_random_anchors_all, prop_random_weights,
                final_df, A_matrix, B_matrix, precomputed_thetas_prop_random
            )
            if prop_random_irt_results:
                prop_random_df = pd.DataFrame(prop_random_irt_results)
                # Find dataset column
                pr_ds_col = None
                for col in ['dataset', 'dataset_name', 'scenario_name']:
                    if col in prop_random_df.columns:
                        pr_ds_col = col
                        break

                if pr_ds_col is not None:
                    for metric in ERROR_METRICS:
                        if metric in prop_random_df.columns:
                            means = prop_random_df.groupby(pr_ds_col)[metric].mean()
                            proportional_random_irt_new_model_old_data[f'proportional_random_irt_{metric}_mean'] = means.mean()
                            proportional_random_irt_new_model_old_data[f'proportional_random_irt_{metric}_std'] = means.std()
                    proportional_random_irt_new_model_old_data['n_proportional_random_anchors'] = len(prop_random_anchors_all)

    # ==========================================================================
    # Build result
    # ==========================================================================
    result = {
        'task_id': task.task_id,
        'distance': task.distance,
        'method': task.method,
        'chain': task.chain_list,
        'chain_str': task.chain_str,
        'n_items': n_items,
        'n_anchors': len(all_anchors),
        'best_dimension': best_dimension,
        'training_time_sec': round(training_time, 2),
        'cumulative_chain_time': task.cumulative_chain_time,
        'gpu_id': gpu_id,
        'min_anchors_per_eval_dataset': int(min(anchor_counts_by_dataset.values())) if anchor_counts_by_dataset else 0,
    }
    
    # Helper to add metrics from a validation DataFrame (flat mean across all rows)
    def add_metrics(df, prefix):
        if df is not None and len(df) > 0:
            result[f'{prefix}_n_models'] = len(df)
            for metric in ERROR_METRICS:
                if metric in df.columns:
                    vals = df[metric].dropna()
                    if len(vals) > 0:
                        result[f'{prefix}_{metric}_mean'] = vals.mean()
                        result[f'{prefix}_{metric}_std'] = vals.std()
            if 'true_performance' in df.columns:
                result[f'{prefix}_true_perf_mean'] = df['true_performance'].mean()
                result[f'{prefix}_true_perf_std'] = df['true_performance'].std()
    
    # Helper to add metrics using mean-of-means (first average per dataset, then across datasets)
    # This ensures each dataset has equal weight regardless of size
    def add_metrics_mean_of_means(df, prefix):
        if df is None or len(df) == 0:
            return
        
        # Find the dataset column
        dataset_col = None
        for col in ['scenario_name', 'dataset', 'dataset_name']:
            if col in df.columns:
                dataset_col = col
                break
        
        if dataset_col is None:
            # Fallback to flat mean if no dataset column
            add_metrics(df, prefix)
            return
        
        datasets = df[dataset_col].unique()
        result[f'{prefix}_n_models'] = df['model_name'].nunique() if 'model_name' in df.columns else len(df)
        result[f'{prefix}_n_datasets'] = len(datasets)
        
        for metric in ERROR_METRICS:
            if metric not in df.columns:
                continue
            # Compute mean per dataset, then mean across datasets
            per_dataset_means = []
            for ds in datasets:
                ds_vals = df[df[dataset_col] == ds][metric].dropna()
                if len(ds_vals) > 0:
                    per_dataset_means.append(ds_vals.mean())
            
            if per_dataset_means:
                result[f'{prefix}_{metric}_mean'] = np.mean(per_dataset_means)
                result[f'{prefix}_{metric}_std'] = np.std(per_dataset_means)
        
        if 'true_performance' in df.columns:
            per_dataset_perf = []
            for ds in datasets:
                ds_vals = df[df[dataset_col] == ds]['true_performance'].dropna()
                if len(ds_vals) > 0:
                    per_dataset_perf.append(ds_vals.mean())
            if per_dataset_perf:
                result[f'{prefix}_true_perf_mean'] = np.mean(per_dataset_perf)
                result[f'{prefix}_true_perf_std'] = np.std(per_dataset_perf)
    
    # Add metrics for all three validation types
    # Validation 1 & 2: single dataset (target) - use flat mean
    add_metrics(validation_df, 'new_model_new_data')
    add_metrics(val_train_on_target_df, 'old_model_new_data')
    # Validation 2b: ALL train_models on target - KEY metric for model sweep!
    add_metrics(val_all_train_on_target_df, 'all_train_new_data')
    # Validation 3: multiple datasets (Base+Chain) - use mean-of-means for consistency with random baselines
    add_metrics_mean_of_means(val_test_on_base_df, 'new_model_old_data')
    
    # Add Discriminative baselines
    for key, val in discriminative_baseline_results.items():
        result[f'discriminative_{key}'] = val
    for key, val in discriminative_baseline_all_train.items():
        result[f'discriminative_all_train_{key}'] = val
    for key, val in discriminative_baseline_new_model_old_data.items():
        result[f'discriminative_new_model_old_data_{key}'] = val

    # Add Random baselines for Validation 1 (New Model + New Data) - backward compatible
    for key, val in random_baseline_results.items():
        result[key] = val
    for key, val in random_simple_results.items():
        result[key] = val
    
    # Add Random baselines for Validation 2b (All Train Models on New Data) - KEY for model sweep!
    for key, val in random_baseline_all_train.items():
        result[f'all_train_new_data_{key}'] = val
    for key, val in random_simple_all_train.items():
        result[f'all_train_new_data_{key}'] = val
    
    # Add Random baselines for Validation 2 (Old Model + New Data)
    for key, val in random_baseline_old_model.items():
        result[f'old_model_new_data_{key}'] = val
    for key, val in random_simple_old_model.items():
        result[f'old_model_new_data_{key}'] = val
    
    # Add Random baselines for Validation 3 (New Model + Old Data)
    for key, val in random_baseline_new_model_old_data.items():
        result[f'new_model_old_data_{key}'] = val
    for key, val in random_simple_new_model_old_data.items():
        result[f'new_model_old_data_{key}'] = val
    
    # Add POOLED Random baselines for Validation 3 (New Model + Old Data)
    for key, val in random_simple_pooled_new_model_old_data.items():
        result[f'new_model_old_data_{key}'] = val
    
    # Add POOLED Random with GP-IRT for Validation 3 (fair comparison with Pooled IRT)
    for key, val in random_irt_pooled_new_model_old_data.items():
        result[f'new_model_old_data_{key}'] = val

    # Add POOLED IRT results for Validation 3 (New Model + Old Data)
    for key, val in pooled_irt_new_model_old_data.items():
        result[f'new_model_old_data_{key}'] = val
    
    # Add PROPORTIONAL results for Validation 3 (New Model + Old Data)
    for key, val in proportional_irt_new_model_old_data.items():
        result[f'new_model_old_data_{key}'] = val
    for key, val in proportional_random_new_model_old_data.items():
        result[f'new_model_old_data_{key}'] = val
    
    # Add PROPORTIONAL Random with GP-IRT for Validation 3 (fair comparison with Proportional IRT)
    for key, val in proportional_random_irt_new_model_old_data.items():
        result[f'new_model_old_data_{key}'] = val

    # Backward compatibility - also add without prefix for main metric
    if validation_df is not None and len(validation_df) > 0:
        result['n_test_models'] = len(validation_df)
        for metric in ERROR_METRICS:
            if metric in validation_df.columns:
                vals = validation_df[metric].dropna()
                if len(vals) > 0:
                    result[f'{metric}_mean'] = vals.mean()
                    result[f'{metric}_std'] = vals.std()
        if 'true_performance' in validation_df.columns:
            result['true_performance_mean'] = validation_df['true_performance'].mean()
            result['true_performance_std'] = validation_df['true_performance'].std()
    
    # Helper to save DataFrame as Parquet (or CSV/JSON for backward compatibility)
    def save_per_model_df(df, name):
        """Save DataFrame as Parquet (and optionally CSV for backward compatibility)."""
        if df is None or len(df) == 0:
            return
        
        # Round numeric columns to 4 decimal places for cleaner output
        df_rounded = round_df_for_save(df)
        
        # Save as Parquet (primary format - smaller and faster)
        parquet_path = output_dir.parent / f"{name}_{task.method}.parquet"
        df_rounded.to_parquet(parquet_path, compression='snappy', index=False)
        
        # Also save CSV for backward compatibility (can be disabled to save space)
        csv_path = output_dir.parent / f"{name}_{task.method}.csv"
        df_rounded.to_csv(csv_path, index=False)
        
        # JSON dict is now redundant with Parquet, but keep for backward compatibility
        if 'model_name' in df_rounded.columns:
            import json
            json_path = output_dir.parent / f"{name}_{task.method}.json"
            # Find dataset column (may be 'dataset', 'dataset_name', or 'scenario_name')
            dataset_col = None
            for col in ['dataset', 'dataset_name', 'scenario_name']:
                if col in df_rounded.columns:
                    dataset_col = col
                    break
            
            has_duplicates = df_rounded['model_name'].duplicated().any()
            
            if dataset_col is not None:
                # Nested structure for per-model-per-dataset results:
                # {model_name: {dataset: {metric: value, ...}}}
                nested_dict = {}
                for _, row in df_rounded.iterrows():
                    model = row['model_name']
                    dataset = row[dataset_col]
                    if model not in nested_dict:
                        nested_dict[model] = {}
                    metrics = {k: (v if not pd.isna(v) else None) 
                               for k, v in row.items() if k not in ['model_name', dataset_col]}
                    nested_dict[model][dataset] = metrics
                json_dict = nested_dict
            elif has_duplicates:
                # Fallback: duplicates but no dataset column - save as list of records
                print(f"      ⚠️ WARNING {name}_{task.method}: duplicate model_names without dataset column!")
                print(f"         Columns: {list(df_rounded.columns)}")
                print("         Saving as list of records instead of nested dict")
                json_dict = df_rounded.to_dict(orient='records')
            else:
                # Simple structure: {model_name: {metric: value, ...}}
                json_dict = df_rounded.set_index('model_name').to_dict(orient='index')
            with open(json_path, 'w') as f:
                json.dump(round_for_json(json_dict), f, indent=2)
    
    # Save all per-model DataFrames
    save_per_model_df(validation_df, 'validation')
    save_per_model_df(val_train_on_target_df, 'validation_old_model_new_data')
    save_per_model_df(val_all_train_on_target_df, 'validation_all_train_new_data')  # KEY metric!
    save_per_model_df(val_test_on_base_df, 'validation_new_model_old_data')
    save_per_model_df(val_test_on_base_pooled_df, 'validation_new_model_old_data_pooled')
    save_per_model_df(discriminative_baseline_per_model_df, 'discriminative_irt')
    save_per_model_df(discriminative_baseline_all_train_per_model_df, 'discriminative_irt_all_train')
    save_per_model_df(discriminative_baseline_new_model_old_data_per_model_df, 'discriminative_irt_new_model_old_data')

    save_per_model_df(random_baseline_per_model_df, 'random_irt')
    save_per_model_df(random_simple_per_model_df, 'random_simple')
    save_per_model_df(random_baseline_old_model_per_model_df, 'random_irt_old_model')
    save_per_model_df(random_simple_old_model_per_model_df, 'random_simple_old_model')
    save_per_model_df(random_baseline_all_train_per_model_df, 'random_irt_all_train')  # KEY baseline!
    save_per_model_df(random_simple_all_train_per_model_df, 'random_simple_all_train')  # KEY baseline!
    save_per_model_df(random_baseline_new_model_old_data_per_model_df, 'random_irt_new_model_old_data')
    save_per_model_df(random_simple_new_model_old_data_per_model_df, 'random_simple_new_model_old_data')
    
    # Save additional validation CSVs (already rounded via save_per_model_df)
    if val_train_on_target_df is not None:
        round_df_for_save(val_train_on_target_df).to_csv(
            output_dir.parent / f"validation_{task.method}_old_model_new_data.csv", index=False)
    if val_all_train_on_target_df is not None:
        round_df_for_save(val_all_train_on_target_df).to_csv(
            output_dir.parent / f"validation_{task.method}_all_train_new_data.csv", index=False)
    if val_test_on_base_df is not None:
        round_df_for_save(val_test_on_base_df).to_csv(
            output_dir.parent / f"validation_{task.method}_new_model_old_data.csv", index=False)
    
    return result


def worker_wrapper(args: tuple) -> dict:
    """Wrapper to unpack arguments for ProcessPoolExecutor."""
    task, gpu_id = args
    return run_scenario_task(task, gpu_id)


# =============================================================================
# Main Experiment
# =============================================================================

def run_chain_linking_parallel(config: ParallelChainConfig):
    """Run the parallel chain linking experiment."""
    
    # Setup cleanup handlers for crash recovery
    setup_cleanup_handlers()
    
    # output_dir creation deferred until target_name is known

    
    experiment_start = time.time()
    
    print("=" * 70)
    print("Chain Linking PARALLEL B - Full Scenario Parallelization")
    print("=" * 70)
    print(f"Workers: {config.num_workers}")

    # -------------------------------------------------------------------------
    # Step 1: Load datasets (sequential)
    # -------------------------------------------------------------------------
    print("\n1. Loading datasets...")
    datasets = load_all_datasets(config)
    print(f"   Loaded {len(datasets)} datasets")
    
    # Filter out degenerate datasets (near-zero mean, trivial for random baseline)
    excluded_found = [ds for ds in EXCLUDED_DATASETS if ds in datasets]
    if excluded_found:
        for ds in excluded_found:
            del datasets[ds]
        print(f"   ⚠️  Excluded {len(excluded_found)} degenerate datasets: {excluded_found}")
        print(f"   Remaining: {len(datasets)} datasets")
    
    skill_to_datasets = group_all_datasets_together(datasets, min_common_models=4)
    if not skill_to_datasets:
        raise ValueError("No valid dataset groups found!")
    all_dataset_names = list(skill_to_datasets.values())[0]
    
    np.random.seed(config.shuffle_seed)
    shuffled = list(all_dataset_names)
    np.random.shuffle(shuffled)
    
    # -------------------------------------------------------------------------
    # Deterministic dataset assignment (NO Python-side auto-increment)
    #
    # Rationale: resuming a crashed run must be 100% reproducible. The bash layer
    # may choose to auto-increment seeds when launching *new* experiments, but
    # this Python script should never change the requested seed just because an
    # output directory exists.
    # -------------------------------------------------------------------------
    if config.target_dataset:
        # User-specified target: keep seed fixed, just pick base/chain deterministically.
        target_name = config.target_dataset
        if target_name not in all_dataset_names:
            available = ", ".join(all_dataset_names[:10]) + "..."
            raise ValueError(f"Target dataset '{target_name}' not found. Available: {available}")

        np.random.seed(config.shuffle_seed)
        shuffled = list(all_dataset_names)
        np.random.shuffle(shuffled)
        shuffled = [d for d in shuffled if d != target_name]
        base_names = shuffled[:config.n_base_datasets]
        chain_pool = shuffled[config.n_base_datasets:]
        print(f"   Using user-specified target: {target_name}")
    else:
        # Auto-select target: deterministic given shuffle_seed.
        np.random.seed(config.shuffle_seed)
        shuffled = list(all_dataset_names)
        np.random.shuffle(shuffled)
        base_names = shuffled[:config.n_base_datasets]
        target_name = shuffled[config.n_base_datasets]
        chain_pool = shuffled[config.n_base_datasets + 1:]
        print(f"   Auto-selected target (deterministic): {target_name}")
    
    target_n_questions = int(datasets[target_name]['question_id'].nunique())

    # Update output directory with target name (avoid duplication)
    initial_output_dir = Path(config.output_dir)
    # Check if target name is already in the path to avoid duplication if run multiple times or manually named
    if f"target_{target_name}" not in initial_output_dir.name:
        new_name = f"{initial_output_dir.name}_target_{target_name}"
        output_dir = initial_output_dir.parent / new_name
        config.output_dir = output_dir
        print(f"   Updated output directory: {output_dir}")
    else:
        output_dir = initial_output_dir
        print(f"   Using existing output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_dir / ".temp"
    temp_dir.mkdir(exist_ok=True)
    
    chain_cache_dir = output_dir / "chain_cache"
    chain_cache_dir.mkdir(exist_ok=True)
    
    # Register cleanup paths for crash recovery
    register_cleanup_paths(temp_dir, chain_cache_dir, output_dir, 
                          config.cleanup_training_data, config.cleanup_cache)
    
    print("\n2. Dataset assignment:")
    print(f"   Base ({len(base_names)}): {base_names}")
    print(f"   Target: {target_name}")
    print(f"   Chain pool: {chain_pool[:5]}...")
    
    # Save config (initial - will be updated after train/test split)
    config_dict = {
        'n_base_datasets': config.n_base_datasets,
        'max_chain_length': config.max_chain_length,
        'seed': config.seed,
        'shuffle_seed': config.shuffle_seed,  # Actual seed used for dataset selection
        'num_workers': config.num_workers,
        'epochs': config.epochs,
        'dims_search': config.dims_search,
        'n_anchors_per_dataset': config.n_anchors_per_dataset,
        'n_models_per_chain': config.n_models_per_chain,
        'base_datasets': base_names,
        'target_dataset': target_name,
        'chain_pool': chain_pool,
        'target_n_questions': target_n_questions,
    }
    
    # -------------------------------------------------------------------------
    # Step 2: Train/test split
    # -------------------------------------------------------------------------
    print("\n3. Defining train/test split...")
    
    base_models = set()
    for ds_name in base_names:
        base_models.update(datasets[ds_name]['model_name'].unique())
    
    base_models_list = sorted(list(base_models))

    # Use seed for model splitting (shuffle_seed is for dataset ordering/target selection).
    np.random.seed(config.seed)
    n_test = max(1, int(len(base_models_list) * config.test_ratio))
    test_models = set(np.random.choice(base_models_list, size=n_test, replace=False))
    train_models = base_models - test_models
    
    # Subset of train_models for chain steps (if specified)
    if config.n_models_per_chain is not None:
        n_chain_models = min(config.n_models_per_chain, len(train_models))
        train_models_list = sorted(list(train_models))
        np.random.seed(config.seed + 1000)  # Different seed offset for chain model selection
        chain_train_models = set(np.random.choice(train_models_list, size=n_chain_models, replace=False))
        print(f"   Train: {len(train_models)}, Test: {len(test_models)}, Chain models: {len(chain_train_models)}")
    else:
        chain_train_models = train_models
    print(f"   Train: {len(train_models)}, Test: {len(test_models)}")
    
    # Save config with model counts and explicit deterministic split artifacts.
    # We keep the exact sorted lists (no hashing) so sweeps can compare equality
    # directly across runs with different n_models_per_chain.
    candidate_models_sorted = base_models_list
    test_models_sorted = sorted(list(test_models))
    train_models_sorted = sorted(list(train_models))
    chain_train_models_sorted = sorted(list(chain_train_models))

    config_dict['n_train_models'] = len(train_models)
    config_dict['n_chain_train_models'] = len(chain_train_models)
    config_dict['n_test_models'] = len(test_models_sorted)
    config_dict['test_model_split_seed'] = config.seed
    config_dict['candidate_models'] = candidate_models_sorted
    config_dict['train_models'] = train_models_sorted
    config_dict['chain_train_models'] = chain_train_models_sorted
    config_dict['test_models'] = test_models_sorted
    with open(output_dir / "config.json", 'w') as f:
        json.dump(round_for_json(config_dict), f, indent=2)
    print(f"   Test-model set saved ({len(test_models_sorted)} models)")

    # -------------------------------------------------------------------------
    # Step 3: Train Base IRT (sequential)
    # -------------------------------------------------------------------------
    print("\n4. Training Base IRT...")
    
    base_dfs = []
    for ds_name in base_names:
        df = datasets[ds_name]
        # Base uses ALL train_models for proper baseline
        df = df[df['model_name'].isin(train_models)].copy()
        base_dfs.append(df)
    base_df = pd.concat(base_dfs, ignore_index=True)
    
    base_irt_dir = output_dir / "irt_base"
    base_irt, A_base, B_base = train_irt_on_base(base_df, config, base_irt_dir)
    
    # Clean up training datasets from base IRT
    if config.cleanup_training_data and base_irt_dir.exists():
        freed_bytes = cleanup_training_datasets(base_irt_dir)
        if freed_bytes > 0:
            freed_mb = freed_bytes / (1024 * 1024)
            print(f"   🧹 Cleaned base training data: {freed_mb:.1f}MB freed")
    
    base_anchors, base_weights = select_anchors(
        base_irt, config.n_anchors_per_dataset, base_df, A_base, B_base,
        clustering_method=config.anchor_method,
    )
    # Ensure each Base dataset has enough LOCAL anchors (prefix-based) for stable evaluation.
    base_anchor_counts = {ds: sum(1 for a in base_anchors if str(a).startswith(f"{ds}:")) for ds in base_names}
    low_base = {ds: c for ds, c in base_anchor_counts.items() if c < MIN_ANCHORS_PER_DATASET}
    if low_base:
        raise ValueError(
            "Base anchor selection produced too few anchors for some Base datasets. "
            f"Need >= {MIN_ANCHORS_PER_DATASET} per dataset. Low: {low_base}"
        )
    print(f"   Base IRT: {len(base_irt)} items, {len(base_anchors)} anchors")
    
    # -------------------------------------------------------------------------
    # Step 4: Build chain cache (sequential)
    # -------------------------------------------------------------------------
    max_chain = min(config.max_chain_length, len(chain_pool))
    print(f"\n5. Building chain cache (up to {max_chain} steps)...")
    
    # Cache stores: (irt_params, A, B, anchors, weights, df, time)
    chain_cache = {}
    chain_cache_times = {}
    
    current_irt = base_irt
    current_A = A_base
    current_B = B_base
    current_anchors = list(base_anchors)
    current_weights = list(base_weights)
    current_df = base_df.copy()
    
    checkpoint_file = chain_cache_dir / "checkpoint.pkl"
    
    total_chain_time = 0
    successful_chain = []
    
    # Resume behavior is explicit: only load prior checkpoint when --force-resume is set.
    if checkpoint_file.exists() and config.force_resume:
        print("   📂 Found checkpoint, loading (--force-resume enabled)...")
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        successful_chain = checkpoint['successful_chain']
        
        # Update (not replace) to avoid scope issues
        chain_cache.update(checkpoint['chain_cache'])
        chain_cache_times.update(checkpoint.get('chain_cache_times', {}))
        
        # Only restore state if we have a successful chain
        if len(successful_chain) > 0:
            current_irt, current_A, current_B, current_anchors, current_weights, current_df = chain_cache[len(successful_chain)]
            total_chain_time = sum(chain_cache_times.get(j, 0) for j in range(1, len(successful_chain) + 1))
            print(f"   ✅ Resumed from step {len(successful_chain)}: {successful_chain}")
        else:
            print("   📂 Checkpoint loaded but chain is empty, starting from base")
    elif checkpoint_file.exists():
        print("   ⚠️  Checkpoint exists but --force-resume is not set; starting fresh for this run")
    
    for i in range(max_chain):
        chain_ds = chain_pool[i]
        
        # Skip if already in checkpoint
        if chain_ds in successful_chain:
            print(f"   Chain step {i+1}: {chain_ds} ✅ (from checkpoint)")
            continue
        
        prefix = "_".join([d.replace(' ', '_')[:10] for d in successful_chain + [chain_ds]])
        cache_dir = chain_cache_dir / f"after_{prefix}"
        
        print(f"   Chain step {i+1}: adding {chain_ds}...")
        
        chain_df = datasets[chain_ds]
        chain_df = chain_df[chain_df['model_name'].isin(chain_train_models)].copy()
        combined_df = pd.concat([current_df, chain_df], ignore_index=True)
        
        available_questions = set(combined_df['question_id'].astype(str).unique())
        anchor_items = build_anchor_items_for_fixed_calibration(
            current_irt, available_questions, current_A, current_B, current_anchors
        )
        
        if current_A is not None:
            dim = current_A.shape[1] if current_A.ndim == 3 else current_A.shape[0]
            dims = [dim]
        else:
            dims = config.dims_search
        
        irt_config = TrainingConfig(
            dims_search=dims,
            epochs=config.epochs_fixed,
            lr=config.lr,
            number_item_per_scenario=config.n_anchors_per_dataset,
            deterministic=True,
            filter_zero_variance=config.filter_zero_variance,
            validate_dimensions=config.validate_dimensions,
        )
        
        chain_start = time.time()
        new_irt = None
        for attempt in range(MAX_RETRIES):
            try:
                new_irt = train_item_parameters(
                    combined_df,
                    config=irt_config,
                    output_dir=str(cache_dir),
                    anchor_items=anchor_items,
                )
                break
            except Exception as e:
                print(f"      ⚠️ Attempt {attempt+1}/{MAX_RETRIES} failed: {str(e)[:80]}")
        
        if new_irt is None:
            print(f"      ❌ Chain step {i+1} failed, skipping {chain_ds}...")
            continue  # Skip this dataset, try the next one
            
        chain_time = time.time() - chain_start
        total_chain_time += chain_time
        
        # Clean up training datasets from chain cache
        if config.cleanup_training_data and cache_dir.exists():
            freed_bytes = cleanup_training_datasets(cache_dir)
            if freed_bytes > 0:
                freed_mb = freed_bytes / (1024 * 1024)
                print(f"      🧹 Cleaned chain training data: {freed_mb:.1f}MB freed")
        
        new_A, new_B = None, None
        if hasattr(new_irt, 'attrs') and new_irt.attrs:
            A_list = new_irt.attrs.get('A_matrix')
            B_list = new_irt.attrs.get('B_matrix')
            if A_list is not None and B_list is not None:
                new_A = np.array(A_list)
                new_B = np.array(B_list)
        
        chain_anchors, chain_weights = select_anchors_for_dataset(
            new_irt, config.n_anchors_per_dataset, chain_ds, combined_df, new_A, new_B,
            method=config.anchor_method,
        )
        
        new_anchors = current_anchors + chain_anchors
        new_weights = current_weights + chain_weights
        
        successful_chain.append(chain_ds)
        distance = len(successful_chain)
        chain_cache[distance] = (new_irt, new_A, new_B, new_anchors, new_weights, combined_df)
        chain_cache_times[distance] = chain_time
        
        current_irt = new_irt
        current_A = new_A
        current_B = new_B
        current_anchors = new_anchors
        current_weights = new_weights
        current_df = combined_df
        
        # Save checkpoint after each successful step
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'successful_chain': successful_chain, 'chain_cache': chain_cache, 
                        'chain_cache_times': chain_cache_times}, f)
        
        print(f"      ✅ {len(new_irt)} items, {len(new_anchors)} anchors, {chain_time:.1f}s (checkpoint saved)")
    
    # Update chain_pool to reflect actual successful chain
    chain_pool = successful_chain
    max_chain = len(successful_chain)
    
    # -------------------------------------------------------------------------
    # Step 5: Prepare scenario tasks
    # -------------------------------------------------------------------------
    print("\n6. Preparing parallel tasks...")
    
    # Get target data
    target_df = datasets[target_name]
    # Use chain_train_models for target (same N models as chain steps)
    # This tests: "Can we link a new dataset when it only has N models?"
    target_train_df = target_df[target_df['model_name'].isin(chain_train_models)].copy()
    target_test_df = target_df[target_df['model_name'].isin(test_models)].copy()
    # ALL train_models on target - for testing linking generalization
    # This tests: "If we link with N models, can we predict ALL train models?"
    target_all_train_df = target_df[target_df['model_name'].isin(train_models)].copy()
    
    # Save target test df for workers
    target_test_path = temp_dir / "target_test.pkl"
    target_test_df.to_pickle(target_test_path)
    
    # Save target train df for workers (old model + new data validation)
    target_train_path = temp_dir / "target_train.pkl"
    target_train_df.to_pickle(target_train_path)
    
    # Save ALL train_models on target for linking generalization test
    target_all_train_path = temp_dir / "target_all_train.pkl"
    target_all_train_df.to_pickle(target_all_train_path)
    
    # Save base IRT params
    base_irt_pkl = temp_dir / "base_irt.pkl"
    base_irt.to_pickle(base_irt_pkl)
    base_A_path = temp_dir / "base_A.npy"
    base_B_path = temp_dir / "base_B.npy"
    if A_base is not None:
        np.save(base_A_path, A_base)
    if B_base is not None:
        np.save(base_B_path, B_base)
    
    # Save chain cache to disk for workers
    for dist, (irt, A, B, anchors, weights, df) in chain_cache.items():
        irt.to_pickle(temp_dir / f"chain_{dist}_irt.pkl")
        if A is not None:
            np.save(temp_dir / f"chain_{dist}_A.npy", A)
        if B is not None:
            np.save(temp_dir / f"chain_{dist}_B.npy", B)
    
    # Note: We keep chain_cache in memory because we need it for the loop below
    # Memory will be freed when the function exits
    
    tasks = []
    already_done = []
    task_id = 0
    
    # Distance scheme (distance = number of datasets added beyond Base):
    #  0 = Base only (no additions) - for Validation 3 baseline
    #  1 = Base + Target (1 dataset added)
    #  2 = Base + Chain[0] + Target (1 chain + Target)
    #  3 = Base + Chain[0] + Chain[1] + Target (2 chains + Target)
    # ...
    for distance in range(0, max_chain + 2):  # 0 to max_chain+1 (inclusive)
        # Get chain info
        if distance == 0:
            # Base only (for Validation 3 baseline)
            chain_str = "base_only"
            chain_list = []
            prev_df = base_df  # Only base data
            prev_irt_path = str(base_irt_pkl)
            prev_A_path = str(base_A_path) if A_base is not None else None
            prev_B_path = str(base_B_path) if B_base is not None else None
            prev_anchors = list(base_anchors)
            prev_weights = list(base_weights)
            cumulative_chain_time = 0
        elif distance == 1:
            # Base + Target (direct linking)
            chain_str = "direct"
            chain_list = []
            prev_df = base_df
            prev_irt_path = str(base_irt_pkl)
            prev_A_path = str(base_A_path) if A_base is not None else None
            prev_B_path = str(base_B_path) if B_base is not None else None
            prev_anchors = list(base_anchors)
            prev_weights = list(base_weights)
            cumulative_chain_time = 0
        else:
            # Base + Chain datasets + Target
            # Chain cache uses 1-based indexing (chain_cache[1] = first chain dataset)
            cache_idx = distance - 1
            if cache_idx not in chain_cache:
                print(f"   Distance {distance}: ⏭️ Skipped (chain not built)")
                continue
            chain_list = chain_pool[:cache_idx]
            chain_str = "_".join([d.replace(' ', '_')[:10] for d in chain_list])
            irt, A, B, anchors, weights, prev_df = chain_cache[cache_idx]
            prev_irt_path = str(temp_dir / f"chain_{cache_idx}_irt.pkl")
            prev_A_path = str(temp_dir / f"chain_{cache_idx}_A.npy") if A is not None else None
            prev_B_path = str(temp_dir / f"chain_{cache_idx}_B.npy") if B is not None else None
            prev_anchors = anchors
            prev_weights = weights
            cumulative_chain_time = sum(chain_cache_times.get(j, 0) for j in range(1, cache_idx + 1))
        
        scenario_dir = output_dir / f"dist_{distance}_{chain_str}"
        
        # Resume: skip if results already exist (only when --force-resume is set)
        results_file = scenario_dir / "results.json"
        if config.force_resume and results_file.exists():
            print(f"   Distance {distance} ({chain_str}): ✅ Already done, loading...")
            with open(results_file) as f:
                result = json.load(f)
            already_done.append(result)
            continue
        
        scenario_dir.mkdir(exist_ok=True)
        
        # Combine with target
        if distance == 0:
            # Base only - no target (Validation 3 baseline)
            final_df = prev_df.copy()
        else:
            # Combine previous datasets with target
            final_df = pd.concat([prev_df, target_train_df], ignore_index=True)
        final_df_path = temp_dir / f"final_df_dist_{distance}.pkl"
        final_df.to_pickle(final_df_path)
        
        # Build test data for Base+Chain datasets (for cross-dataset theta estimation)
        # This allows computing theta from historical anchor responses, not just target
        if distance == 0:
            # Base only - test on base datasets
            base_chain_datasets = base_names
        elif distance == 1:
            # Base + Target
            base_chain_datasets = base_names
        else:
            # Base + Chain + Target
            base_chain_datasets = base_names + chain_list
        
        base_chain_test_dfs = []
        for ds_name in base_chain_datasets:
            ds_df = datasets[ds_name]
            ds_test_df = ds_df[ds_df['model_name'].isin(test_models)].copy()
            base_chain_test_dfs.append(ds_test_df)
        
        if base_chain_test_dfs:
            base_chain_test_df = pd.concat(base_chain_test_dfs, ignore_index=True)
            base_chain_test_df_path = temp_dir / f"base_chain_test_dist_{distance}.pkl"
            base_chain_test_df.to_pickle(base_chain_test_df_path)
            base_chain_test_df_path_str = str(base_chain_test_df_path)
        else:
            base_chain_test_df_path_str = None
        
        # Determine dimension
        if distance == 0:
            # Base only - use base dimension (already trained)
            dims = [A_base.shape[1] if A_base.ndim == 3 else A_base.shape[0]] if A_base is not None else config.dims_search
        elif distance == 1:
            # Base + Target
            dims = [A_base.shape[1] if A_base.ndim == 3 else A_base.shape[0]] if A_base is not None else config.dims_search
        else:
            # Base + Chain + Target
            cache_idx = distance - 1
            A = chain_cache[cache_idx][1]
            dims = [A.shape[1] if A.ndim == 3 else A.shape[0]] if A is not None else config.dims_search
        
        # Use seed for task-level random sampling (shuffle_seed is dataset ordering only).
        task_seed = config.seed

        # Create tasks for both methods
        # For distance=0 (Base only), we only evaluate on Base datasets (Validation 3)
        # No need for Fixed-Anchor vs Concurrent since we're not adding new data
        if distance == 0:
            methods = ['concurrent']  # Base-only: just one run (no fixed vs concurrent distinction)
        else:
            methods = ['fixed', 'concurrent']
        
        for method in methods:
            # For distance=0 (Base only), use fixed epochs (base already trained, just evaluating)
            if distance == 0:
                task_epochs = config.epochs_fixed  # Just re-run validation on base
            else:
                task_epochs = config.epochs_fixed if method == 'fixed' else config.epochs
            task = ScenarioTask(
                task_id=task_id,
                distance=distance,
                method=method,
                chain_list=chain_list,
                chain_str=chain_str,
                scenario_dir=str(scenario_dir),
                final_df_path=str(final_df_path),
                target_test_df_path=str(target_test_path),
                base_chain_test_df_path=base_chain_test_df_path_str,
                target_train_df_path=str(target_train_path),
                target_all_train_df_path=str(target_all_train_path),
                prev_irt_path=prev_irt_path if method == 'fixed' else None,
                prev_A_path=prev_A_path if method == 'fixed' else None,
                prev_B_path=prev_B_path if method == 'fixed' else None,
                prev_anchors=prev_anchors,  # Always pass for validation (training uses anchor_items)
                prev_weights=prev_weights,
                dims=dims,
                epochs=task_epochs,
                n_anchors_per_dataset=config.n_anchors_per_dataset,
                filter_zero_variance=config.filter_zero_variance,
                validate_dimensions=config.validate_dimensions,
                lr=config.lr,
                target_name=target_name,
                test_models=list(test_models),
                train_models=list(chain_train_models),  # Same N models as chain for consistent model sweep
                all_train_models=list(train_models),  # ALL train models for linking generalization test
                seed=task_seed,
                random_seed=config.random_seed,
                cleanup_training_data=config.cleanup_training_data,
                cumulative_chain_time=cumulative_chain_time,
                anchor_method=config.anchor_method,
            )
            tasks.append(task)
            task_id += 1
    
    print(f"   Created {len(tasks)} new tasks ({len(already_done)} already completed)")
    
    # -------------------------------------------------------------------------
    # Step 6: Run tasks in parallel
    # -------------------------------------------------------------------------
    all_results = []
    parallel_time = 0
    
    # Free memory before spawning worker processes
    # Workers will load data from disk (.temp/*.pkl files), they don't need the in-memory datasets
    print("\n   🧹 Freeing main process memory before spawning workers...")
    del datasets  # Large dict of all datasets - no longer needed
    del target_df, target_train_df, target_test_df, target_all_train_df  # Already saved to disk
    if 'base_chain_test_dfs' in locals():
        del base_chain_test_dfs
    if 'base_chain_test_df' in locals():
        del base_chain_test_df
    import gc
    gc.collect()
    print("   ✅ Memory freed")
    
    if tasks:
        print(f"\n7. Running {len(tasks)} tasks with {config.num_workers} workers...")
        
        # Determine available GPUs
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            gpu_ids = [int(x) for x in cuda_visible.split(',') if x.strip()]
        else:
            # Try to detect GPUs
            try:
                import torch
                gpu_ids = list(range(torch.cuda.device_count()))
            except:
                gpu_ids = [0]
        
        print(f"   Available GPUs: {gpu_ids}")
        
        # Assign GPUs round-robin to tasks
        task_args = []
        for i, task in enumerate(tasks):
            gpu_id = gpu_ids[i % len(gpu_ids)] if gpu_ids else None
            task_args.append((task, gpu_id))
        
        completed = 0
        
        parallel_start = time.time()
        
        with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
            futures = {executor.submit(worker_wrapper, args): args[0].task_id for args in task_args}
            
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    # Check if task failed
                    if result.get('failed'):
                        print(f"   ❌ Task {task_id} (dist_{result['distance']}/{result['method']}) failed after retries")
                        continue
                    all_results.append(result)
                    completed += 1
                    
                    # Print progress
                    dist = result['distance']
                    method = result['method']
                    err = result.get('gp_irt_error_mean', float('nan'))
                    t = result.get('training_time_sec', 0)
                    print(f"   [{completed}/{len(tasks)}] dist_{dist}/{method}: error={err:.4f}, time={t:.1f}s")
                    
                except Exception as e:
                    print(f"   ❌ Task {task_id} exception: {e}")
        
        parallel_time = time.time() - parallel_start
        print(f"\n   Parallel execution: {parallel_time:.1f}s")
    else:
        print("\n7. No new tasks to run (all scenarios already completed)")
    
    # -------------------------------------------------------------------------
    # Step 7: Aggregate results
    # -------------------------------------------------------------------------
    print("\n8. Aggregating results...")
    
    # Sort by distance and method
    all_results.sort(key=lambda x: (x['distance'], x['method']))
    
    # Combine Fixed and Concurrent for each distance
    final_results = []
    
    # First add already completed results
    for result in already_done:
        final_results.append(result)
        print(f"   Distance {result['distance']}: loaded from previous run")
    
    # Then process new results
    processed_distances = {r['distance'] for r in already_done}
    
    for distance in range(0, max_chain + 2):  # 0 to max_chain+1 (inclusive)
        if distance in processed_distances:
            continue
        cache_idx = distance - 1
        if cache_idx not in chain_cache and distance not in [0, 1]:
            continue
            
        dist_results = [r for r in all_results if r['distance'] == distance]
        fixed_result = next((r for r in dist_results if r['method'] == 'fixed'), None)
        concurrent_result = next((r for r in dist_results if r['method'] == 'concurrent'), None)
        
        # For distance=0 (Base-only), we only have concurrent results
        if distance == 0 and not concurrent_result:
            print(f"   Distance {distance}: ⏭️ No results (concurrent failed)")
            continue
        elif distance >= 1 and not fixed_result and not concurrent_result:
            print(f"   Distance {distance}: ⏭️ No results (both methods failed)")
            continue
        
        # Allow partial results
        fixed_result = fixed_result or {}
        concurrent_result = concurrent_result or {}
        
        chain_list = fixed_result.get('chain', concurrent_result.get('chain', []))
        chain_str = fixed_result.get('chain_str', concurrent_result.get('chain_str', 'base_only' if distance == 0 else 'direct'))
        
        # Calculate n_datasets_in_training based on distance (distance = number of datasets added beyond Base)
        if distance == 0:
            n_datasets_in_training = config.n_base_datasets  # Base only
        elif distance == 1:
            n_datasets_in_training = config.n_base_datasets + 1  # Base + Target
        else:
            n_datasets_in_training = config.n_base_datasets + distance  # Base + (distance-1) chains + Target
        
        result = {
            'target_dataset': target_name,
            'distance': distance,
            'chain': chain_list,
            'n_datasets_in_training': n_datasets_in_training,
            # For correct cost reporting: Full evaluation is only the Target dataset size
            'target_n_questions': target_n_questions,
            'n_anchors_per_dataset': config.n_anchors_per_dataset,
            'n_base_datasets': config.n_base_datasets,
            # Cost model (per target addition) in #questions / API calls
            'cost_full_eval_target': target_n_questions,
        }
        
        # Cost calculations depend on distance (distance = number of datasets added beyond Base)
        if distance == 0:
            # Base only - for Validation 3 baseline (evaluate on base datasets)
            result['cost_fixed_target_anchors'] = 0  # No target involved
            result['cost_concurrent_all_anchors'] = config.n_anchors_per_dataset * config.n_base_datasets
        elif distance == 1:
            # Base + Target
            result['cost_fixed_target_anchors'] = config.n_anchors_per_dataset
            result['cost_concurrent_all_anchors'] = config.n_anchors_per_dataset * (config.n_base_datasets + 1)
        else:
            # Base + Chain + Target (distance-1 chains + Target)
            result['cost_fixed_target_anchors'] = config.n_anchors_per_dataset
            result['cost_concurrent_all_anchors'] = config.n_anchors_per_dataset * (config.n_base_datasets + distance)
        
        # Add Fixed results (only for distance >= 1)
        if distance >= 1:
            for key, val in fixed_result.items():
                if key not in ['task_id', 'distance', 'method', 'chain', 'chain_str', 'gpu_id', 'failed']:
                    result[f'fixed_{key}'] = val
        
        # Add Concurrent results
        for key, val in concurrent_result.items():
            if key not in ['task_id', 'distance', 'method', 'chain', 'chain_str', 'gpu_id', 'failed']:
                result[f'concurrent_{key}'] = val
        
        # Compute deltas (only for distance >= 1 where we have both methods)
        if distance >= 1:
            for metric in ERROR_METRICS:
                fixed_val = fixed_result.get(f'{metric}_mean')
                concurrent_val = concurrent_result.get(f'{metric}_mean')
                if fixed_val is not None and concurrent_val is not None:
                    result[f'delta_{metric}'] = fixed_val - concurrent_val
        
        # Save per-scenario result
        scenario_dir = output_dir / f"dist_{distance}_{chain_str}"
        scenario_dir.mkdir(exist_ok=True)
        with open(scenario_dir / "results.json", 'w') as f:
            result_save = {**result, 'chain': list(result['chain'])}
            json.dump(round_for_json(result_save), f, indent=2)
        
        final_results.append(result)
    
    # Sort final results by distance
    final_results.sort(key=lambda x: x['distance'])
    
    # Save summary
    results_for_df = []
    for r in final_results:
        r_copy = r.copy()
        if r_copy['distance'] == 0:
            r_copy['chain'] = "base_only"
        elif r_copy['distance'] == 1:
            r_copy['chain'] = "direct"
        else:
            r_copy['chain'] = "_".join(r_copy['chain']) if r_copy['chain'] else "direct"
        results_for_df.append(r_copy)
    
    results_df = pd.DataFrame(results_for_df)
    
    # Save as Parquet (primary format - faster and smaller)
    round_df_for_save(results_df).to_parquet(output_dir / "all_results.parquet", compression='snappy', index=False)
    
    # Also save as CSV for backward compatibility
    round_df_for_save(results_df).to_csv(output_dir / "all_results.csv", index=False)
    
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(round_for_json([{**r, 'chain': list(r['chain'])} for r in final_results]), f, indent=2)
    
    # Clean up temp files
    import shutil
    print("\n9. Cleaning up temporary files...")
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"   ✅ Removed temp directory: {temp_dir}")
    except Exception as e:
        print(f"   ⚠️ Failed to remove temp directory: {e}")
    
    # Optionally clean up chain cache (only needed for resuming failed runs)
    if config.cleanup_cache:
        try:
            if chain_cache_dir.exists():
                shutil.rmtree(chain_cache_dir)
                print(f"   ✅ Removed chain cache: {chain_cache_dir}")
        except Exception as e:
            print(f"   ⚠️ Failed to remove chain cache: {e}")
    
    # Optionally clean up IRT model files from dist_* directories
    if config.cleanup_models:
        print("   Cleaning up IRT model files from scenario directories...")
        for scenario_dir in output_dir.glob("dist_*"):
            if scenario_dir.is_dir():
                for irt_dir in scenario_dir.glob("irt_*"):
                    if irt_dir.is_dir():
                        try:
                            shutil.rmtree(irt_dir)
                            print(f"     ✅ Removed {irt_dir.relative_to(output_dir)}")
                        except Exception as e:
                            print(f"     ⚠️ Failed to remove {irt_dir}: {e}")
    
    total_time = time.time() - experiment_start
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - Parallel Execution")
    print("=" * 70)
    print(f"Target: {target_name}")
    print(f"Workers: {config.num_workers}")
    print(f"Train models: {len(train_models)}" +
          (f" (chain subset: {len(chain_train_models)})" if config.n_models_per_chain else ""))
    print(f"Total time: {total_time:.1f}s (parallel phase: {parallel_time:.1f}s)")
    print(f"\n{'Dist':<6} {'Chain':<20} {'Fixed':<10} {'Concurrent':<12} {'Delta':<10}")
    print("-" * 60)
    
    for r in final_results:
        dist = r['distance']
        if dist == 0:
            chain = "base_only"
        elif dist == 1:
            chain = "direct"
        else:
            chain = "_".join([c[:6] for c in r['chain']]) if r['chain'] else "direct"
        if len(chain) > 18:
            chain = chain[:15] + "..."
        fixed_err = r.get('fixed_gp_irt_error_mean', float('nan'))
        concurrent_err = r.get('concurrent_gp_irt_error_mean', float('nan'))
        delta = r.get('delta_gp_irt_error', float('nan'))
        
        # For distance=0, show only concurrent (Base only - no fixed vs concurrent)
        if dist == 0:
            print(f"{dist:<6} {chain:<20} {'N/A':<10} {concurrent_err:<12.4f} {'N/A':<10}")
        else:
            print(f"{dist:<6} {chain:<20} {fixed_err:<10.4f} {concurrent_err:<12.4f} {delta:+10.4f}")
    
    # Random baseline comparison
    print("\n" + "-" * 60)
    print("RANDOM BASELINE COMPARISON:")
    print("  Method comparison (lower error = better):")
    
    # Collect averages for comparison
    fixed_irt_errors = [r.get('fixed_gp_irt_error_mean') for r in final_results if r.get('fixed_gp_irt_error_mean') is not None]
    fixed_random_irt_errors = [r.get('fixed_random_gp_irt_error_mean') for r in final_results if r.get('fixed_random_gp_irt_error_mean') is not None]
    fixed_simple_errors = [r.get('fixed_simple_random_error_mean') for r in final_results if r.get('fixed_simple_random_error_mean') is not None]
    
    if fixed_irt_errors:
        print(f"    IRT Anchors (smart selection):   {np.mean(fixed_irt_errors):.4f}")
    if fixed_random_irt_errors:
        print(f"    Random-IRT (random as anchors):  {np.mean(fixed_random_irt_errors):.4f}")
    if fixed_simple_errors:
        print(f"    Random-Simple (just average):    {np.mean(fixed_simple_errors):.4f}")
    
    # Determine winner
    if fixed_irt_errors and fixed_random_irt_errors and fixed_simple_errors:
        methods = {
            'IRT Anchors': np.mean(fixed_irt_errors),
            'Random-IRT': np.mean(fixed_random_irt_errors),
            'Random-Simple': np.mean(fixed_simple_errors),
        }
        winner = min(methods, key=methods.get)
        print(f"\n  Winner: {winner} with error = {methods[winner]:.4f}")

    print(f"\nResults saved to: {output_dir}")
    
    return results_df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chain Linking Parallel B - Full Parallelization")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--n-base", type=int, default=6, help="Number of base datasets")
    parser.add_argument("--max-chain", type=int, default=10, help="Maximum chain length")
    parser.add_argument("--n-anchors-per-dataset", type=int, default=100, help="Anchors per dataset")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (anchor selection, model splits)")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Dataset shuffle seed (controls dataset order)")
    parser.add_argument("--random-seed", type=int, default=1000, help="Seed for random baseline scenarios")
    parser.add_argument("--dims", type=int, nargs="+", default=[5], help="IRT dimensions")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs (concurrent/base)")
    parser.add_argument("--epochs-fixed", type=int, default=1000, help="Training epochs (fixed-anchor)")
    parser.add_argument("--data-source-mode", type=str, default="helm_lite",
                        choices=["mixed", "helm_lite", "helm_classic", "lb_only", "lb", "reeval", "mmlu_split", "mmlu_fields", "tinybenchmarks"])
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--target-dataset", type=str, default=None, 
                        help="Specific target dataset name (if not specified, uses shuffled[n_base])")
    parser.add_argument("--n-models-per-chain", type=int, default=None,
                        help="Number of models to use for training (None = all train models)")
    parser.add_argument("--cleanup-cache", action="store_true", default=True,
                        help="Remove chain_cache after successful completion (default: True)")
    parser.add_argument("--no-cleanup-cache", dest="cleanup_cache", action="store_false",
                        help="Keep chain_cache (useful for debugging or resuming)")
    parser.add_argument("--cleanup-models", action="store_true", default=False,
                        help="Remove IRT model files from dist_* directories to save space")
    parser.add_argument("--cleanup-training-data", action="store_true", default=True,
                        help="Remove training datasets (*.jsonlines) immediately after training (default: True, saves ~88%% per IRT dir)")
    parser.add_argument("--no-cleanup-training-data", dest="cleanup_training_data", action="store_false",
                        help="Keep training datasets (useful for debugging)")
    parser.add_argument("--force-resume", action="store_true", default=False,
                        help="Force resume existing experiment (skip auto-increment seed check)")
    parser.add_argument("--anchor-method", type=str, default="irt_clustering",
                        choices=["irt_clustering", "top_k_discrimination", "correctness_clustering"],
                        help="Anchor selection method (default: irt_clustering)")

    args = parser.parse_args()
    
    config = ParallelChainConfig(
        n_base_datasets=args.n_base,
        max_chain_length=args.max_chain,
        n_anchors_per_dataset=args.n_anchors_per_dataset,
        test_ratio=args.test_ratio,
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        random_seed=args.random_seed,
        dims_search=args.dims,
        epochs=args.epochs,
        epochs_fixed=args.epochs_fixed,
        data_source_mode=args.data_source_mode,
        num_workers=args.num_workers,
        target_dataset=args.target_dataset,
        n_models_per_chain=args.n_models_per_chain,
        cleanup_cache=args.cleanup_cache,
        cleanup_models=args.cleanup_models,
        cleanup_training_data=args.cleanup_training_data,
        force_resume=args.force_resume,
        anchor_method=args.anchor_method,
    )
    
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    else:
        dims_str = "-".join(map(str, args.dims))
        base_name = f"chain_parallel_b_{args.data_source_mode}_seed_{args.shuffle_seed}_anchors_{args.n_anchors_per_dataset}_dims_{dims_str}"
        if args.n_models_per_chain is not None:
            base_name += f"_models_{args.n_models_per_chain}"
        config.output_dir = PROJECT_ROOT / "data" / base_name
    
    # Setup crash handlers first
    setup_cleanup_handlers()
    
    # Run with exception handling for cleanup
    try:
        run_chain_linking_parallel(config)
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        print("   Running emergency cleanup...")
        emergency_cleanup()
        raise  # Re-raise to preserve stack trace

