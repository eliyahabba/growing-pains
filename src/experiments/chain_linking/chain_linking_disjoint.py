"""
Chain Linking Experiment - Disjoint Models Version

This experiment tests IRT's ability to rank models that have ZERO dataset overlap.
It uses "bridge models" to connect isolated model groups, each trained on different
chain steps.

Key concept:
- Bridge models: A small set of models used in ALL chain steps (the "glue")
- Isolated groups: New models added at each chain step (no overlap between groups)
- Validation: Can we correctly rank models from different isolated groups?

Experiment structure:
    Base (DS A):    Models [bridge] + [base_only]
    Chain 1 (DS B): Models [bridge] + [isolated_B]
    Chain 2 (DS C): Models [bridge] + [isolated_C]
    ...
    Target (DS T):  Validate ranking of isolated groups

Primary Metric: Spearman Rank Correlation between IRT-based ranking and true ranking.

Usage:
    python chain_linking_disjoint.py --disjoint-mode --n-bridge-models 10 --n-isolated-per-chain 10
"""

from __future__ import annotations

import json
import multiprocessing
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

multiprocessing.set_start_method('spawn', force=True)

from config.constants import (
    EXCLUDED_DATASETS,
    MAX_RETRIES,
)
from llm_eval.selection.tinyBenchmarks.training import TrainingConfig
from llm_eval.training import train_item_parameters
from src.experiments.equating.cross_dataset_equating import (
    PROJECT_ROOT,
    ExperimentConfig,
    build_anchor_items_for_fixed_calibration,
    group_all_datasets_together,
    load_all_datasets,
    precompute_thetas_from_all_anchors,
    select_anchors,
    select_anchors_for_dataset,
    train_irt_on_base,
)
from src.experiments.utils.helpers import round_for_json

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DisjointChainConfig(ExperimentConfig):
    """Configuration for disjoint chain linking experiments."""
    n_base_datasets: int = 1
    max_chain_length: int = 5
    shuffle_seed: int = 42
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "chain_disjoint")
    data_source_mode: str = "lb"  # Default to LB for disjoint experiment
    filter_zero_variance: bool = False
    validate_dimensions: bool = True
    epochs: int = 2000
    epochs_fixed: int = 1000
    n_anchors_per_dataset: int = 100
    num_workers: int = 4
    target_dataset: str | None = None
    
    # Disjoint mode parameters
    disjoint_mode: bool = True  # Enable disjoint models experiment
    n_bridge_models: int = 20   # Number of bridge models (used in ALL chain steps)
    n_isolated_per_chain: int = 50  # Number of new isolated models per chain step (larger chains)
    unseen_test_ratio: float = 0.15  # Ratio of models never seen in ANY training
    fixed_bridge: bool = True   # If True: same bridge models in ALL steps; If False: random sample per step
    
    def __post_init__(self):
        # Auto-adjust for tinybenchmarks/lb (395 models, 6 datasets)
        # With 395 models:
        # - unseen_test: 15% = ~60 models (never in training)
        # - Remaining 335 for training:
        #   - bridge: 20 (in all steps)
        #   - isolated: 4 × 50 = 200 (each group in ONE step, can be tested on target)
        #   - base_only: ~115 (only in base)
        if self.data_source_mode in ["tinybenchmarks", "lb_only", "lb"]:
            if self.n_base_datasets == 1:
                pass  # Keep as 1
            if self.max_chain_length == 5:
                self.max_chain_length = 4  # 6 datasets: 1 base + 4 chain + 1 target
            # Print allocation summary
            total_models = 395
            n_unseen = int(total_models * self.unseen_test_ratio)
            n_remaining = total_models - n_unseen
            n_isolated_total = self.max_chain_length * self.n_isolated_per_chain
            n_base_only = n_remaining - self.n_bridge_models - n_isolated_total
            print(f"   ℹ️ LB mode allocation (~{total_models} models):")
            print(f"      Unseen test: {n_unseen} ({self.unseen_test_ratio:.0%})")
            print(f"      Bridge: {self.n_bridge_models}")
            print(f"      Isolated: {self.max_chain_length} × {self.n_isolated_per_chain} = {n_isolated_total}")
            print(f"      Base-only: ~{max(0, n_base_only)}")
        
        # Auto-adjust for mmlu_fields
        if self.data_source_mode in ["mmlu_split", "mmlu_fields"]:
            self.n_base_datasets = 1
            if self.n_anchors_per_dataset == 100:
                print("   ⚠️ Note: For mmlu_fields mode, consider using --n-anchors-per-dataset 10")


# =============================================================================
# Helper Functions
# =============================================================================

def flatten_thetas(thetas: dict[str, any]) -> dict[str, float]:
    """Convert multi-dimensional thetas to scalars (mean across dimensions).
    
    For MIRT models, theta can be a multi-dimensional array (e.g., shape (1, D, 1)).
    This function converts each theta to a scalar by taking the mean across dimensions.
    
    Args:
        thetas: Dict mapping model_name -> theta (can be scalar or ndarray)
    
    Returns:
        Dict mapping model_name -> scalar theta
    """
    flattened = {}
    for model_name, theta in thetas.items():
        if isinstance(theta, np.ndarray):
            flattened[model_name] = float(np.mean(theta))
        else:
            flattened[model_name] = float(theta)
    return flattened


# =============================================================================
# Disjoint Ranking Validation
# =============================================================================

def compute_disjoint_ranking_metrics(
    isolated_groups: dict[int, set[str]],
    thetas: dict[str, float],
    datasets: dict[str, pd.DataFrame],
    target_name: str,
    all_dataset_names: list[str],
) -> dict:
    """
    Compare theta-based ranking to ground truth ranking for isolated model groups.
    
    Args:
        isolated_groups: Dict mapping chain step index -> set of isolated model names
        thetas: Dict mapping model_name -> theta value
        datasets: All loaded datasets
        target_name: Name of target dataset for ground truth
        all_dataset_names: List of all dataset names for computing global mean
    
    Returns:
        Dict with ranking metrics
    """
    # Collect all isolated models
    all_isolated = set()
    for group in isolated_groups.values():
        all_isolated.update(group)
    
    # Filter to models that have theta estimates
    models_with_theta = [m for m in all_isolated if m in thetas]
    
    if len(models_with_theta) < 4:
        return {
            'disjoint_spearman_rho': None,
            'disjoint_kendall_tau': None,
            'n_isolated_models': len(all_isolated),
            'n_models_with_theta': len(models_with_theta),
            'error': 'Not enough models with theta estimates'
        }
    
    # IRT ranking (by theta)
    theta_values = [thetas[m] for m in models_with_theta]
    
    # Ground truth: mean score on Target dataset
    target_df = datasets[target_name]
    target_scores = target_df.groupby('model_name')['normalized_score'].mean()
    
    # Also compute global mean across all datasets
    all_scores = []
    for ds_name in all_dataset_names:
        if ds_name in datasets:
            ds_df = datasets[ds_name]
            ds_scores = ds_df.groupby('model_name')['normalized_score'].mean()
            all_scores.append(ds_scores)
    
    if all_scores:
        global_scores = pd.concat(all_scores, axis=1).mean(axis=1)
    else:
        global_scores = target_scores
    
    # Get true scores for isolated models
    true_target_scores = [target_scores.get(m, np.nan) for m in models_with_theta]
    true_global_scores = [global_scores.get(m, np.nan) for m in models_with_theta]
    
    # Filter out NaN
    valid_mask_target = [not np.isnan(s) for s in true_target_scores]
    valid_mask_global = [not np.isnan(s) for s in true_global_scores]
    
    results = {
        'n_isolated_models': len(all_isolated),
        'n_models_with_theta': len(models_with_theta),
        'n_isolated_groups': len(isolated_groups),
        'isolated_group_sizes': {k: len(v) for k, v in isolated_groups.items()},
    }
    
    # Compute correlations for Target
    valid_thetas_target = [theta_values[i] for i, v in enumerate(valid_mask_target) if v]
    valid_true_target = [true_target_scores[i] for i, v in enumerate(valid_mask_target) if v]
    
    if len(valid_thetas_target) >= 4:
        spearman_target = spearmanr(valid_thetas_target, valid_true_target)
        kendall_target = kendalltau(valid_thetas_target, valid_true_target)
        results['disjoint_spearman_rho_target'] = spearman_target.correlation
        results['disjoint_kendall_tau_target'] = kendall_target.correlation
        results['disjoint_spearman_pvalue_target'] = spearman_target.pvalue
        results['n_valid_target'] = len(valid_thetas_target)
    
    # Compute correlations for Global mean
    valid_thetas_global = [theta_values[i] for i, v in enumerate(valid_mask_global) if v]
    valid_true_global = [true_global_scores[i] for i, v in enumerate(valid_mask_global) if v]
    
    if len(valid_thetas_global) >= 4:
        spearman_global = spearmanr(valid_thetas_global, valid_true_global)
        kendall_global = kendalltau(valid_thetas_global, valid_true_global)
        results['disjoint_spearman_rho_global'] = spearman_global.correlation
        results['disjoint_kendall_tau_global'] = kendall_global.correlation
        results['disjoint_spearman_pvalue_global'] = spearman_global.pvalue
        results['n_valid_global'] = len(valid_thetas_global)
    
    # Primary metric (use target)
    results['disjoint_spearman_rho'] = results.get('disjoint_spearman_rho_target')
    results['disjoint_kendall_tau'] = results.get('disjoint_kendall_tau_target')
    
    return results


def compute_pairwise_ranking_accuracy(
    isolated_groups: dict[int, set[str]],
    thetas: dict[str, float],
    datasets: dict[str, pd.DataFrame],
    target_name: str,
) -> dict:
    """
    Compute pairwise ranking accuracy between isolated groups.
    For each pair of models from different groups, check if IRT ranking matches true ranking.
    
    Args:
        isolated_groups: Dict mapping chain step index -> set of isolated model names
        thetas: Dict mapping model_name -> theta value
        datasets: All loaded datasets
        target_name: Name of target dataset for ground truth
    
    Returns:
        Dict with pairwise accuracy metrics
    """
    target_df = datasets[target_name]
    target_scores = target_df.groupby('model_name')['normalized_score'].mean().to_dict()
    
    # Collect all pairs from different groups
    group_keys = sorted(isolated_groups.keys())
    total_pairs = 0
    correct_pairs = 0
    
    for i, key_i in enumerate(group_keys):
        for key_j in group_keys[i+1:]:
            group_i = isolated_groups[key_i]
            group_j = isolated_groups[key_j]
            
            for model_a in group_i:
                for model_b in group_j:
                    if model_a not in thetas or model_b not in thetas:
                        continue
                    if model_a not in target_scores or model_b not in target_scores:
                        continue
                    
                    theta_a, theta_b = thetas[model_a], thetas[model_b]
                    true_a, true_b = target_scores[model_a], target_scores[model_b]
                    
                    # Check if ranking direction matches
                    irt_ranking = (theta_a > theta_b)
                    true_ranking = (true_a > true_b)
                    
                    total_pairs += 1
                    if irt_ranking == true_ranking:
                        correct_pairs += 1
    
    if total_pairs == 0:
        return {'pairwise_accuracy': None, 'total_pairs': 0}
    
    return {
        'pairwise_accuracy': correct_pairs / total_pairs,
        'correct_pairs': correct_pairs,
        'total_pairs': total_pairs,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_disjoint_chain_experiment(config: DisjointChainConfig):
    """Run the disjoint chain linking experiment."""
    
    experiment_start = time.time()
    
    print("=" * 70)
    print("Chain Linking DISJOINT - Zero Dataset Overlap Between Model Groups")
    print("=" * 70)
    print(f"Disjoint Mode: {config.disjoint_mode}")
    print(f"Bridge Models: {config.n_bridge_models}")
    print(f"Bridge Mode: {'FIXED (same in all steps)' if config.fixed_bridge else 'RANDOM (sampled per step)'}")
    print(f"Isolated per Chain: {config.n_isolated_per_chain}")
    print(f"Unseen Test Ratio: {config.unseen_test_ratio:.0%}")

    # -------------------------------------------------------------------------
    # Step 1: Load datasets
    # -------------------------------------------------------------------------
    print("\n1. Loading datasets...")
    datasets = load_all_datasets(config)
    print(f"   Loaded {len(datasets)} datasets")
    
    # Filter excluded datasets
    excluded_found = [ds for ds in EXCLUDED_DATASETS if ds in datasets]
    if excluded_found:
        for ds in excluded_found:
            del datasets[ds]
        print(f"   ⚠️ Excluded {len(excluded_found)} degenerate datasets")
        print(f"   Remaining: {len(datasets)} datasets")
    
    skill_to_datasets = group_all_datasets_together(datasets, min_common_models=4)
    if not skill_to_datasets:
        raise ValueError("No valid dataset groups found!")
    all_dataset_names = list(skill_to_datasets.values())[0]
    
    np.random.seed(config.shuffle_seed)
    shuffled = list(all_dataset_names)
    np.random.shuffle(shuffled)
    
    # Assign datasets
    base_names = shuffled[:config.n_base_datasets]
    if config.target_dataset and config.target_dataset in all_dataset_names:
        target_name = config.target_dataset
        shuffled = [d for d in shuffled if d != target_name]
        chain_pool = shuffled[config.n_base_datasets:]
    else:
        target_name = shuffled[config.n_base_datasets]
        chain_pool = shuffled[config.n_base_datasets + 1:]
    
    target_n_questions = int(datasets[target_name]['question_id'].nunique())
    
    # Setup output directory
    output_dir = Path(config.output_dir)
    if f"target_{target_name}" not in output_dir.name:
        output_dir = output_dir.parent / f"{output_dir.name}_target_{target_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n2. Dataset assignment:")
    print(f"   Base ({len(base_names)}): {base_names}")
    print(f"   Target: {target_name}")
    print(f"   Chain pool: {chain_pool[:5]}...")
    
    # -------------------------------------------------------------------------
    # Step 2: Disjoint Model Allocation
    # -------------------------------------------------------------------------
    print("\n3. Allocating models (DISJOINT mode)...")
    
    # Get all models that appear in base datasets
    base_models = set()
    for ds_name in base_names:
        base_models.update(datasets[ds_name]['model_name'].unique())
    
    all_models = sorted(list(base_models))
    np.random.seed(config.shuffle_seed)
    np.random.shuffle(all_models)
    
    # Split into TWO test sets + training groups:
    # 1. unseen_test: Models NEVER in any training (true held-out)
    # 2. isolated_test: Models in isolated groups (trained on ONE dataset, tested on target)
    
    # First, allocate unseen test models
    n_unseen_test = max(1, int(len(all_models) * config.unseen_test_ratio))
    unseen_test_models = set(all_models[:n_unseen_test])
    remaining = all_models[n_unseen_test:]
    
    max_chain = min(config.max_chain_length, len(chain_pool))
    
    # Bridge models allocation depends on mode
    if config.fixed_bridge:
        # FIXED: Same bridge models in ALL chain steps
        n_bridge = min(config.n_bridge_models, len(remaining))
        bridge_models = set(remaining[:n_bridge])
        bridge_pool = None  # Not used in fixed mode
        remaining = remaining[n_bridge:]
    else:
        # RANDOM: Sample different bridge models per step from a pool
        # Pool size = n_bridge_models * max_chain (so each step can have unique bridge models)
        bridge_pool_size = min(config.n_bridge_models * max_chain, len(remaining))
        bridge_pool = list(remaining[:bridge_pool_size])
        bridge_models = set()  # Will be populated per step
        remaining = remaining[bridge_pool_size:]
    
    # Isolated groups for chain steps
    # These models are trained on ONE chain dataset only
    # They can be evaluated on the TARGET dataset to measure chain linking effectiveness
    isolated_groups: dict[int, set[str]] = {}
    
    # Calculate how many models we need for isolated groups
    total_isolated_needed = max_chain * config.n_isolated_per_chain
    
    # Allocate isolated groups from remaining models
    isolated_models_pool = remaining[:total_isolated_needed]
    base_only_pool = remaining[total_isolated_needed:]
    
    # Split isolated pool into groups
    idx = 0
    for i in range(max_chain):
        n_isolated = min(config.n_isolated_per_chain, len(isolated_models_pool) - idx)
        if n_isolated > 0:
            isolated_groups[i] = set(isolated_models_pool[idx:idx + n_isolated])
            idx += n_isolated
        else:
            isolated_groups[i] = set()
    
    # Base-only models (only in Base, not in chain steps)
    base_only_models = set(base_only_pool)
    
    # For random bridge mode, pre-compute which bridge models go to which step
    bridge_per_step: dict[int, set[str]] = {}
    if not config.fixed_bridge and bridge_pool:
        np.random.shuffle(bridge_pool)
        bp_idx = 0
        for i in range(max_chain):
            n_step_bridge = min(config.n_bridge_models, len(bridge_pool) - bp_idx)
            bridge_per_step[i] = set(bridge_pool[bp_idx:bp_idx + n_step_bridge])
            bp_idx += n_step_bridge
    
    # Collect all isolated models for reporting
    all_isolated_for_test = set()
    for group in isolated_groups.values():
        all_isolated_for_test.update(group)
    
    # Collect all bridge models (for reporting)
    all_bridge_models = bridge_models.copy() if config.fixed_bridge else set()
    if not config.fixed_bridge:
        for step_bridges in bridge_per_step.values():
            all_bridge_models.update(step_bridges)
    
    print("   📊 Model Allocation Summary:")
    print(f"   ├── Unseen Test (never trained): {len(unseen_test_models)}")
    print(f"   ├── Isolated Test (trained on 1 DS): {len(all_isolated_for_test)}")
    print("   │   (can also be evaluated on target)")
    if config.fixed_bridge:
        print(f"   ├── Bridge (FIXED, in all steps): {len(bridge_models)}")
    else:
        print(f"   ├── Bridge (RANDOM per step): {len(all_bridge_models)} total, {config.n_bridge_models} per step")
        for i, step_bridges in bridge_per_step.items():
            print(f"   │   Step {i+1}: {len(step_bridges)} bridge models")
    print(f"   └── Base-only: {len(base_only_models)}")
    print("   ")
    print("   Isolated groups breakdown:")
    for i, group in isolated_groups.items():
        print(f"      Chain {i+1}: {len(group)} models")
    
    # For backward compatibility, keep test_models as unseen_test
    test_models = unseen_test_models
    
    # Models for Base training: bridge + base_only
    base_train_models = bridge_models | base_only_models
    
    # Save config
    config_dict = {
        'disjoint_mode': config.disjoint_mode,
        'fixed_bridge': config.fixed_bridge,
        'n_bridge_models': config.n_bridge_models,
        'n_bridge_models_actual': len(all_bridge_models),
        'n_isolated_per_chain': config.n_isolated_per_chain,
        'n_base_datasets': config.n_base_datasets,
        'max_chain_length': max_chain,
        'shuffle_seed': config.shuffle_seed,
        'base_datasets': base_names,
        'target_dataset': target_name,
        'chain_pool': chain_pool[:max_chain],
        'target_n_questions': target_n_questions,
        # Test set info (dual test sets)
        'n_unseen_test_models': len(unseen_test_models),
        'n_isolated_test_models': len(all_isolated_for_test),
        'n_test_models': len(test_models),  # backward compat (= unseen)
        'n_base_train_models': len(base_train_models),
        'isolated_group_sizes': {k: len(v) for k, v in isolated_groups.items()},
        'bridge_models': list(all_bridge_models),
        'bridge_per_step': {k: list(v) for k, v in bridge_per_step.items()} if not config.fixed_bridge else None,
        'unseen_test_models': list(unseen_test_models),
        'isolated_test_models': list(all_isolated_for_test),
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(round_for_json(config_dict), f, indent=2)
    
    # -------------------------------------------------------------------------
    # Step 3: Train Base IRT
    # -------------------------------------------------------------------------
    print("\n4. Training Base IRT...")
    
    base_dfs = []
    for ds_name in base_names:
        df = datasets[ds_name]
        df = df[df['model_name'].isin(base_train_models)].copy()
        base_dfs.append(df)
    base_df = pd.concat(base_dfs, ignore_index=True)
    
    base_irt_dir = output_dir / "irt_base"
    base_irt, A_base, B_base = train_irt_on_base(base_df, config, base_irt_dir)
    
    base_anchors, base_weights = select_anchors(
        base_irt, config.n_anchors_per_dataset, base_df, A_base, B_base
    )
    print(f"   Base IRT: {len(base_irt)} items, {len(base_anchors)} anchors")
    
    # -------------------------------------------------------------------------
    # Step 4: Build Chain with Disjoint Models
    # -------------------------------------------------------------------------
    print(f"\n5. Building chain with DISJOINT models (up to {max_chain} steps)...")
    
    chain_cache = {}
    chain_cache_dir = output_dir / "chain_cache"
    chain_cache_dir.mkdir(exist_ok=True)
    
    current_irt = base_irt
    current_A = A_base
    current_B = B_base
    current_anchors = list(base_anchors)
    current_weights = list(base_weights)
    current_df = base_df.copy()
    
    successful_chain = []
    
    for i in range(max_chain):
        chain_ds = chain_pool[i]
        isolated_group = isolated_groups.get(i, set())
        
        # Get bridge models for this step (depends on mode)
        if config.fixed_bridge:
            step_bridge = bridge_models
        else:
            step_bridge = bridge_per_step.get(i, set())
        
        # Models for this chain step: bridge + this step's isolated group
        step_models = step_bridge | isolated_group
        
        print(f"   Chain step {i+1}: {chain_ds}")
        bridge_mode_str = "fixed" if config.fixed_bridge else "random"
        print(f"      Models: {len(step_bridge)} bridge ({bridge_mode_str}) + {len(isolated_group)} isolated = {len(step_models)}")
        
        chain_df = datasets[chain_ds]
        chain_df = chain_df[chain_df['model_name'].isin(step_models)].copy()
        
        if len(chain_df) == 0:
            print(f"      ⚠️ No data for step {i+1}, skipping...")
            continue
        
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
        
        new_irt = None
        for attempt in range(MAX_RETRIES):
            try:
                cache_dir = chain_cache_dir / f"step_{i+1}_{chain_ds.replace(' ', '_')[:15]}"
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
            print(f"      ❌ Chain step {i+1} failed, stopping chain...")
            break
        
        new_A, new_B = None, None
        if hasattr(new_irt, 'attrs') and new_irt.attrs:
            A_list = new_irt.attrs.get('A_matrix')
            B_list = new_irt.attrs.get('B_matrix')
            if A_list is not None and B_list is not None:
                new_A = np.array(A_list)
                new_B = np.array(B_list)
        
        chain_anchors, chain_weights = select_anchors_for_dataset(
            new_irt, config.n_anchors_per_dataset, chain_ds, combined_df, new_A, new_B
        )
        
        new_anchors = current_anchors + chain_anchors
        new_weights = current_weights + chain_weights
        
        successful_chain.append(chain_ds)
        distance = len(successful_chain)
        chain_cache[distance] = (new_irt, new_A, new_B, new_anchors, new_weights, combined_df, isolated_group)
        
        current_irt = new_irt
        current_A = new_A
        current_B = new_B
        current_anchors = new_anchors
        current_weights = new_weights
        current_df = combined_df
        
        print(f"      ✅ {len(new_irt)} items, {len(new_anchors)} anchors")
    
    if len(successful_chain) == 0:
        raise ValueError("No chain steps completed successfully!")
    
    print(f"\n   Chain completed: {len(successful_chain)} steps")
    
    # -------------------------------------------------------------------------
    # Step 5: Compute Thetas for Isolated Groups
    # -------------------------------------------------------------------------
    print("\n6. Computing thetas for isolated models...")
    
    # Get the final IRT params
    final_distance = len(successful_chain)
    final_irt, final_A, final_B, final_anchors, final_weights, final_df, _ = chain_cache[final_distance]
    
    # Collect all isolated models
    all_isolated_models = set()
    for i, group in isolated_groups.items():
        if i < len(successful_chain):  # Only include groups from successful chain steps
            all_isolated_models.update(group)
    
    print(f"   Total isolated models to evaluate: {len(all_isolated_models)}")
    
    # For each isolated group, compute theta using anchors from their respective chain step
    all_thetas = {}
    
    for i in range(len(successful_chain)):
        group = isolated_groups.get(i, set())
        if not group:
            continue
        
        chain_ds = successful_chain[i]
        
        # Get anchors available for this chain step
        # Always include base anchors (all isolated models participated in base training)
        # Plus anchors from their specific chain dataset
        step_dataset_anchors = [a for a in final_anchors if a.startswith(f"{chain_ds}:")]
        
        # Combine: base anchors + chain dataset anchors
        available_anchors = list(set(base_anchors) | set(step_dataset_anchors))
        
        # Get responses for this group's models on BOTH base AND their chain dataset
        # Isolated models participated in base training, so they have responses on base too!
        chain_responses = datasets[chain_ds][datasets[chain_ds]['model_name'].isin(group)].copy()
        
        # Also get base dataset responses
        base_responses_list = []
        for base_ds in base_names:
            if base_ds in datasets:
                base_df = datasets[base_ds][datasets[base_ds]['model_name'].isin(group)].copy()
                base_responses_list.append(base_df)
        
        # Combine base + chain responses
        if base_responses_list:
            group_df = pd.concat([chain_responses] + base_responses_list, ignore_index=True)
        else:
            group_df = chain_responses
        
        if len(group_df) == 0:
            print(f"   ⚠️ No data for isolated group {i+1}")
            continue
        
        # Compute thetas
        group_thetas = precompute_thetas_from_all_anchors(
            test_df=group_df,
            item_params=final_irt,
            anchor_ids=available_anchors,
            A_matrix=final_A,
            B_matrix=final_B,
        )
        # Flatten multi-dimensional thetas to scalars for ranking
        group_thetas = flatten_thetas(group_thetas)
        
        all_thetas.update(group_thetas)
        print(f"   Chain {i+1} ({chain_ds}): {len(group_thetas)}/{len(group)} models got theta")
    
    # -------------------------------------------------------------------------
    # Step 6: Compute Disjoint Ranking Metrics
    # -------------------------------------------------------------------------
    print("\n7. Computing ranking metrics for BOTH test sets...")
    
    # Filter isolated_groups to only successful chain steps
    active_isolated_groups = {i: isolated_groups[i] for i in range(len(successful_chain)) if isolated_groups.get(i)}
    
    # ===== PART A: Isolated Test Set (trained on ONE dataset) =====
    print("\n   A. ISOLATED TEST SET (models trained on ONE chain dataset):")
    
    ranking_metrics = compute_disjoint_ranking_metrics(
        isolated_groups=active_isolated_groups,
        thetas=all_thetas,
        datasets=datasets,
        target_name=target_name,
        all_dataset_names=all_dataset_names,
    )
    
    pairwise_metrics = compute_pairwise_ranking_accuracy(
        isolated_groups=active_isolated_groups,
        thetas=all_thetas,
        datasets=datasets,
        target_name=target_name,
    )
    
    print(f"      Isolated models: {ranking_metrics.get('n_isolated_models', 0)}")
    print(f"      Models with theta: {ranking_metrics.get('n_models_with_theta', 0)}")
    print(f"      Spearman ρ (Target): {ranking_metrics.get('disjoint_spearman_rho_target', 'N/A')}")
    print(f"      Pairwise Accuracy: {pairwise_metrics.get('pairwise_accuracy', 'N/A')}")
    
    # ===== PART B: Unseen Test Set (NEVER in training) =====
    print("\n   B. UNSEEN TEST SET (models NEVER in any training):")
    
    # Estimate theta for unseen test models using the FINAL IRT model
    unseen_thetas = {}
    target_df = datasets[target_name]
    unseen_target_df = target_df[target_df['model_name'].isin(unseen_test_models)].copy()
    
    if len(unseen_target_df) > 0 and final_irt is not None:
        # For unseen models, we estimate theta using items from the CHAIN (not target)
        # because the IRT model only has parameters for chain datasets
        # Then we compare their theta ranking to their actual performance on target
        
        # Get all available items from chain datasets for unseen models
        chain_datasets = successful_chain
        unseen_chain_dfs = []
        for chain_ds in chain_datasets:
            chain_df = datasets[chain_ds]
            unseen_chain_df = chain_df[chain_df['model_name'].isin(unseen_test_models)].copy()
            unseen_chain_dfs.append(unseen_chain_df)
        
        if unseen_chain_dfs:
            unseen_all_df = pd.concat(unseen_chain_dfs, ignore_index=True)
            
            # Use all IRT items as potential anchors
            irt_items = list(final_irt.index.astype(str)) if hasattr(final_irt, 'index') else []
            print(f"      Using {len(irt_items)} chain items for theta estimation")
            
            # Estimate theta for unseen models using chain dataset responses
            unseen_thetas = precompute_thetas_from_all_anchors(
                test_df=unseen_all_df,
                item_params=final_irt,
                anchor_ids=irt_items,
                A_matrix=final_A,
                B_matrix=final_B,
            )
            # Flatten multi-dimensional thetas to scalars for ranking
            unseen_thetas = flatten_thetas(unseen_thetas)
        else:
            unseen_thetas = {}
        print(f"      Unseen models: {len(unseen_test_models)}")
        print(f"      Models with theta: {len(unseen_thetas)}")
        
        # Compute correlation with ground truth on target
        target_scores = target_df.groupby('model_name')['normalized_score'].mean()
        
        unseen_with_theta = [m for m in unseen_thetas.keys() if m in target_scores.index]
        if len(unseen_with_theta) >= 4:
            theta_vals = [unseen_thetas[m] for m in unseen_with_theta]
            true_vals = [target_scores[m] for m in unseen_with_theta]
            
            spearman_unseen = spearmanr(theta_vals, true_vals)
            kendall_unseen = kendalltau(theta_vals, true_vals)
            
            print(f"      Spearman ρ (Target): {spearman_unseen.statistic:.4f}")
            print(f"      Kendall τ (Target): {kendall_unseen.statistic:.4f}")
            
            ranking_metrics['unseen_spearman_rho'] = spearman_unseen.statistic
            ranking_metrics['unseen_kendall_tau'] = kendall_unseen.statistic
            ranking_metrics['unseen_n_models'] = len(unseen_with_theta)
        else:
            print("      ⚠️ Not enough unseen models with theta for correlation")
    else:
        print("      ⚠️ No unseen test data or IRT model available")
    
    print("\n   === SUMMARY ===")
    print(f"   Isolated Test (chain linking effect): Spearman ρ = {ranking_metrics.get('disjoint_spearman_rho_target', 'N/A')}")
    print(f"   Unseen Test (true generalization): Spearman ρ = {ranking_metrics.get('unseen_spearman_rho', 'N/A')}")
    
    # -------------------------------------------------------------------------
    # Step 7: Save Results
    # -------------------------------------------------------------------------
    print("\n8. Saving results...")
    
    results = {
        'experiment': 'disjoint_chain_linking',
        'target_dataset': target_name,
        'n_chain_steps': len(successful_chain),
        'chain_datasets': successful_chain,
        'fixed_bridge': config.fixed_bridge,
        'n_bridge_models': len(all_bridge_models),
        'n_isolated_total': len(all_isolated_models),
        'n_unseen_test': len(unseen_test_models),
        **ranking_metrics,
        **pairwise_metrics,
    }
    
    # Save main results
    with open(output_dir / "disjoint_results.json", 'w') as f:
        json.dump(round_for_json(results), f, indent=2)
    
    # Save isolated thetas
    theta_df = pd.DataFrame([
        {'model_name': m, 'theta': t, 'group': next((k for k, v in active_isolated_groups.items() if m in v), None), 'test_type': 'isolated'}
        for m, t in all_thetas.items()
    ])
    theta_df.to_csv(output_dir / "isolated_thetas.csv", index=False)
    
    # Save unseen thetas
    if unseen_thetas:
        unseen_theta_df = pd.DataFrame([
            {'model_name': m, 'theta': t, 'group': None, 'test_type': 'unseen'}
            for m, t in unseen_thetas.items()
        ])
        unseen_theta_df.to_csv(output_dir / "unseen_thetas.csv", index=False)
        
        # Combine all thetas for unified comparison
        all_combined_thetas = {**all_thetas, **unseen_thetas}
    else:
        all_combined_thetas = all_thetas
    
    # Save per-model ground truth (both test sets)
    target_scores = target_df.groupby('model_name')['normalized_score'].mean()
    
    gt_rows = []
    for m, t in all_combined_thetas.items():
        test_type = 'unseen' if m in unseen_test_models else 'isolated'
        group = next((k for k, v in active_isolated_groups.items() if m in v), None)
        gt_rows.append({
            'model_name': m,
            'theta': t,
            'true_target_score': target_scores.get(m, np.nan),
            'test_type': test_type,
            'group': group,
        })
    gt_df = pd.DataFrame(gt_rows)
    gt_df.to_csv(output_dir / "disjoint_comparison.csv", index=False)
    
    total_time = time.time() - experiment_start
    
    print("\n" + "=" * 70)
    print("DISJOINT EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Target: {target_name}")
    print(f"Chain length: {len(successful_chain)}")
    print("")
    print("📊 MODEL ALLOCATION:")
    bridge_mode_str = "FIXED" if config.fixed_bridge else "RANDOM"
    print(f"   Bridge models: {len(all_bridge_models)} ({bridge_mode_str})")
    print(f"   Isolated groups: {len(active_isolated_groups)} (total: {len(all_isolated_models)} models)")
    print(f"   Unseen test: {len(unseen_test_models)} models")
    print("")
    print("📈 RESULTS:")
    print(f"   Isolated Test (chain linking): Spearman ρ = {ranking_metrics.get('disjoint_spearman_rho_target', 'N/A')}")
    print(f"   Unseen Test (generalization):  Spearman ρ = {ranking_metrics.get('unseen_spearman_rho', 'N/A')}")
    print(f"   Pairwise Accuracy: {pairwise_metrics.get('pairwise_accuracy', 'N/A')}")
    print("")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results saved to: {output_dir}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chain Linking Disjoint - Zero Overlap Models")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--n-base", type=int, default=1, help="Number of base datasets")
    parser.add_argument("--max-chain", type=int, default=5, help="Maximum chain length")
    parser.add_argument("--n-anchors-per-dataset", type=int, default=100, help="Anchors per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Dataset shuffle seed")
    parser.add_argument("--dims", type=int, nargs="+", default=[5], help="IRT dimensions")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--epochs-fixed", type=int, default=1000, help="Training epochs (fixed-anchor)")
    parser.add_argument("--data-source-mode", type=str, default="lb",
                        choices=["mixed", "helm_lite", "helm_classic", "lb_only", "lb", 
                                "reeval", "mmlu_split", "mmlu_fields", "tinybenchmarks"])
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--target-dataset", type=str, default=None, help="Specific target dataset")
    
    # Disjoint mode parameters
    parser.add_argument("--disjoint-mode", action="store_true", default=True,
                        help="Enable disjoint models experiment (default: True)")
    parser.add_argument("--n-bridge-models", type=int, default=50,
                        help="Number of bridge models used in all chain steps")
    parser.add_argument("--n-isolated-per-chain", type=int, default=50,
                        help="Number of new isolated models per chain step")
    parser.add_argument("--unseen-test-ratio", type=float, default=0.15,
                        help="Ratio of models that are NEVER in any training (true held-out)")
    parser.add_argument("--fixed-bridge", action="store_true", default=True,
                        help="Use FIXED bridge models (same in all steps)")
    parser.add_argument("--random-bridge", action="store_true", default=False,
                        help="Use RANDOM bridge models (different per step)")
    
    args = parser.parse_args()
    
    # Handle bridge mode flags (--random-bridge overrides --fixed-bridge)
    fixed_bridge = not args.random_bridge
    
    config = DisjointChainConfig(
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
        disjoint_mode=args.disjoint_mode,
        n_bridge_models=args.n_bridge_models,
        n_isolated_per_chain=args.n_isolated_per_chain,
        unseen_test_ratio=args.unseen_test_ratio,
        fixed_bridge=fixed_bridge,
    )
    
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    else:
        bridge_mode = "fixed" if fixed_bridge else "random"
        config.output_dir = (
            PROJECT_ROOT
            / "data"
            / f"chain_disjoint_{args.data_source_mode}_seed_{args.shuffle_seed}_bridge_{args.n_bridge_models}_{bridge_mode}_isolated_{args.n_isolated_per_chain}"
        )
    
    run_disjoint_chain_experiment(config)

