"""Validation and baseline evaluation routines."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from irt import (
    TrainingConfig,
    compute_lambda_values,
    run_estimation_validation,
    estimate_theta_from_anchors,
)
from src.data_loading import ExperimentConfig
from src.calibration import precompute_thetas_from_all_anchors, train_irt_on_base, select_anchors


def run_validation(
    test_df: pd.DataFrame,
    item_params: pd.DataFrame,
    anchor_ids: list[str],
    anchor_weights: list[float],
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    precomputed_thetas: dict[str, float] | None = None,
) -> list[dict]:
    """Run estimation validation.
    
    Args:
        test_df: Test data to evaluate on
        item_params: IRT item parameters
        anchor_ids: List of anchor question IDs
        anchor_weights: Weights for anchor questions
        train_df: Training data (used for computing lambdas)
        A_matrix, B_matrix: MIRT matrices
        precomputed_thetas: Optional dict mapping model_name -> theta.
            If provided, these thetas are used instead of estimating from local anchors.
            This enables cross-dataset theta estimation.
    """
    # Compute lambda values
    attrs = getattr(item_params, 'attrs', {})
    validation_errors = attrs.get('validation_errors', {})
    best_dim = attrs.get('best_dimension', 5)
    dims_search = attrs.get('config_dims_search', [5, 10])
    best_dim_idx = dims_search.index(best_dim) if best_dim in dims_search else 0
    
    # Get unique datasets in test_df
    datasets_in_test = test_df['dataset'].unique()
    
    # Build anchors and lambdas per dataset
    anchors_by_dataset = {}
    anchor_weights_by_dataset = {}
    
    for ds in datasets_in_test:
        # Filter anchors to those in this dataset
        ds_anchors = [a for a in anchor_ids if a.startswith(f"{ds}:")]
        if ds_anchors:
            anchors_by_dataset[ds] = ds_anchors
            # Get corresponding weights
            indices = [anchor_ids.index(a) for a in ds_anchors]
            anchor_weights_by_dataset[ds] = [anchor_weights[i] for i in indices]
        else:
            # Use all anchors (cross-dataset prediction)
            anchors_by_dataset[ds] = anchor_ids
            anchor_weights_by_dataset[ds] = anchor_weights
    
    lambdas_by_dataset = compute_lambda_values(
        original_matrix_df=train_df,
        validation_errors=validation_errors,
        best_dim_idx=best_dim_idx,
        number_item=len(anchor_ids),
    )
    
    question_ids_order = list(item_params.index) if hasattr(item_params, 'index') else None
    
    results = run_estimation_validation(
        test_matrix=test_df,
        item_params=item_params,
        anchors_by_dataset=anchors_by_dataset,
        lambdas_by_dataset=lambdas_by_dataset,
        anchor_weights_by_dataset=anchor_weights_by_dataset,
        precomputed_thetas=precomputed_thetas,
        A_matrix=A_matrix,
        B_matrix=B_matrix,
        question_ids_order=question_ids_order,
    )
    
    return results


def run_random_baseline_validation(
    test_df: pd.DataFrame,
    item_params: pd.DataFrame,
    n_random_questions: int,
    target_name: str,
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    precomputed_thetas: dict[str, float] | None = None,
    n_seeds: int = 1,
    base_seed: int = 42,
    return_per_model: bool = False,
) -> dict | tuple[dict, pd.DataFrame]:
    """Run validation using randomly selected questions instead of IRT-selected anchors.
    
    This provides a baseline to compare against the IRT anchor selection method.
    By running multiple seeds, we get variance estimates for the random baseline.
    
    Args:
        test_df: Test data to evaluate on
        item_params: IRT item parameters
        n_random_questions: Number of random questions to select (same as n_anchors)
        target_name: Name of the target dataset to select random questions from
        train_df: Training data
        A_matrix, B_matrix: MIRT matrices
        precomputed_thetas: Optional dict mapping model_name -> theta
        n_seeds: Number of random seeds to run (default 10)
        base_seed: Base seed for reproducibility
        return_per_model: If True, also return per-model results DataFrame
    
    Returns:
        If return_per_model=False: Dict with aggregated statistics
        If return_per_model=True: Tuple of (aggregated_dict, per_model_df)
        
        aggregated_dict contains:
        {
            'random_anchor_error_mean': float,
            'random_anchor_error_std': float,
            'random_gp_irt_error_mean': float,
            'random_gp_irt_error_std': float,
            'n_seeds': int,
            'n_random_questions': int,
        }
        
        per_model_df contains per-model results averaged across seeds:
        - model_name
        - random_anchor_error, random_gp_irt_error, etc.
    """
    # Get all questions from target dataset that have IRT parameters
    target_questions = [q for q in item_params.index if q.startswith(f"{target_name}:")]
    
    if len(target_questions) < n_random_questions:
        print(f"      Warning: Only {len(target_questions)} questions available, using all")
        n_random_questions = len(target_questions)
    
    if len(target_questions) < 10:
        print(f"      Warning: Too few questions ({len(target_questions)}) for random baseline, skipping")
        return {}
    
    # Collect results from each seed
    all_seed_results = {
        'anchor_error': [],
        'irt_error': [],
        'gp_irt_error': [],
        'pirt_error': [],
    }
    
    # Collect per-model results across all seeds (for averaging)
    per_model_results = {}  # model_name -> {metric -> [values across seeds]}
    
    # Get lambda values (needed for validation)
    attrs = getattr(item_params, 'attrs', {})
    validation_errors = attrs.get('validation_errors', {})
    best_dim = attrs.get('best_dimension', 5)
    dims_search = attrs.get('config_dims_search', [5, 10])
    best_dim_idx = dims_search.index(best_dim) if best_dim in dims_search else 0
    
    question_ids_order = list(item_params.index) if hasattr(item_params, 'index') else None
    
    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        np.random.seed(seed)
        
        # Randomly select questions
        random_anchors = list(np.random.choice(target_questions, size=n_random_questions, replace=False))
        
        # Assign uniform weights
        random_weights = [1.0 / n_random_questions] * n_random_questions
        
        # Build anchors dict
        anchors_by_dataset = {target_name: random_anchors}
        anchor_weights_by_dataset = {target_name: random_weights}
        
        # Compute lambdas for this anchor count
        lambdas_by_dataset = compute_lambda_values(
            original_matrix_df=train_df,
            validation_errors=validation_errors,
            best_dim_idx=best_dim_idx,
            number_item=n_random_questions,
        )
        
        # Run validation (silently)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Suppress print statements during random validation
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                results = run_estimation_validation(
                    test_matrix=test_df,
                    item_params=item_params,
                    anchors_by_dataset=anchors_by_dataset,
                    lambdas_by_dataset=lambdas_by_dataset,
                    anchor_weights_by_dataset=anchor_weights_by_dataset,
                    precomputed_thetas=precomputed_thetas,
                    A_matrix=A_matrix,
                    B_matrix=B_matrix,
                    question_ids_order=question_ids_order,
                )
            finally:
                sys.stdout = old_stdout
        
        # Aggregate results from this seed
        if results:
            for metric in ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']:
                vals = [r[metric] for r in results if not np.isnan(r.get(metric, np.nan))]
                if vals:
                    all_seed_results[metric].append(np.mean(vals))
            
            # Collect per-model results
            if return_per_model:
                for r in results:
                    model_name = r['model_name']
                    if model_name not in per_model_results:
                        per_model_results[model_name] = {
                            'anchor_error': [], 'irt_error': [], 
                            'gp_irt_error': [], 'pirt_error': [],
                            'true_performance': [], 'anchor_prediction': [],
                            'gp_irt_prediction': [],
                        }
                    for metric in per_model_results[model_name].keys():
                        val = r.get(metric, np.nan)
                        if not np.isnan(val):
                            per_model_results[model_name][metric].append(val)
    
    # Compute statistics across seeds
    output = {
        'n_seeds': n_seeds,
        'n_random_questions': n_random_questions,
    }
    
    for metric in ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']:
        vals = all_seed_results[metric]
        if vals:
            output[f'random_{metric}_mean'] = float(np.mean(vals))
            output[f'random_{metric}_std'] = float(np.std(vals))
    
    n_successful_seeds = len(all_seed_results['anchor_error'])
    print(f"      Random baseline: {n_successful_seeds}/{n_seeds} seeds successful")
    if n_successful_seeds > 0:
        print(f"         anchor_error: {output.get('random_anchor_error_mean', 'N/A'):.4f} ± {output.get('random_anchor_error_std', 'N/A'):.4f}")
        print(f"         gp_irt_error: {output.get('random_gp_irt_error_mean', 'N/A'):.4f} ± {output.get('random_gp_irt_error_std', 'N/A'):.4f}")
    
    if return_per_model:
        # Build per-model DataFrame with averaged results across seeds
        per_model_rows = []
        for model_name, metrics in per_model_results.items():
            row = {'model_name': model_name, 'n_seeds': n_seeds}
            for metric, values in metrics.items():
                if values:
                    row[f'random_{metric}_mean'] = float(np.mean(values))
                    row[f'random_{metric}_std'] = float(np.std(values))
            per_model_rows.append(row)
        per_model_df = pd.DataFrame(per_model_rows) if per_model_rows else pd.DataFrame()
        return output, per_model_df
    
    return output


def run_discriminative_baseline_validation(
    test_df: pd.DataFrame,
    item_params: pd.DataFrame,
    n_anchors: int,
    target_name: str,
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    precomputed_thetas: dict[str, float] | None = None,
    return_per_model: bool = False,
) -> dict | tuple[dict, pd.DataFrame]:
    """Run validation using top-K most discriminative items as anchors.
    
    This provides a baseline to compare against IRT clustering.
    Unlike random baseline, this is deterministic for a given set of item parameters.
    
    Args:
        test_df: Test data to evaluate on
        item_params: IRT item parameters
        n_anchors: Number of discriminative items to select
        target_name: Name of the target dataset
        train_df: Training data
        A_matrix, B_matrix: MIRT matrices
        precomputed_thetas: Optional dict mapping model_name -> theta
        return_per_model: If True, also return per-model results DataFrame
    """
    # Get all questions from target dataset that have IRT parameters
    target_questions = [q for q in item_params.index if q.startswith(f"{target_name}:")]
    
    if len(target_questions) < n_anchors:
        print(f"      Warning: Only {len(target_questions)} questions available, using all")
        n_anchors = len(target_questions)
    
    if len(target_questions) < 5:
        print(f"      Warning: Too few questions ({len(target_questions)}) for discriminative baseline, skipping")
        return {}
    
    # Select top-K items by discrimination parameter
    # For MIRT, we use the norm of the discrimination vector (a_i)
    target_params = item_params.loc[target_questions].copy()
    if 'discrimination' in target_params.columns:
        # 1PL/2PL/3PL
        target_params['disc_norm'] = target_params['discrimination']
    elif any(col.startswith('a_') for col in target_params.columns):
        # MIRT
        a_cols = [col for col in target_params.columns if col.startswith('a_')]
        target_params['disc_norm'] = np.linalg.norm(target_params[a_cols].values, axis=1)
    else:
        # Fallback to first column if no standard names found
        target_params['disc_norm'] = target_params.iloc[:, 0]
        
    top_k_questions = target_params.sort_values('disc_norm', ascending=False).head(n_anchors).index.tolist()
    
    # Assign uniform weights
    weights = [1.0 / n_anchors] * n_anchors
    
    # Build anchors dict
    anchors_by_dataset = {target_name: top_k_questions}
    anchor_weights_by_dataset = {target_name: weights}
    
    # Get lambda values (needed for validation)
    attrs = getattr(item_params, 'attrs', {})
    validation_errors = attrs.get('validation_errors', {})
    best_dim = attrs.get('best_dimension', 5)
    dims_search = attrs.get('config_dims_search', [5, 10])
    best_dim_idx = dims_search.index(best_dim) if best_dim in dims_search else 0
    
    question_ids_order = list(item_params.index) if hasattr(item_params, 'index') else None
    
    # Compute lambdas for this anchor count
    lambdas_by_dataset = compute_lambda_values(
        original_matrix_df=train_df,
        validation_errors=validation_errors,
        best_dim_idx=best_dim_idx,
        number_item=n_anchors,
    )
    
    # Run validation
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            results = run_estimation_validation(
                test_matrix=test_df,
                item_params=item_params,
                anchors_by_dataset=anchors_by_dataset,
                lambdas_by_dataset=lambdas_by_dataset,
                anchor_weights_by_dataset=anchor_weights_by_dataset,
                precomputed_thetas=precomputed_thetas,
                A_matrix=A_matrix,
                B_matrix=B_matrix,
                question_ids_order=question_ids_order,
            )
        finally:
            sys.stdout = old_stdout
            
    if not results:
        return {}
        
    # Aggregate results
    output = {
        'n_anchors': n_anchors,
    }
    
    for metric in ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']:
        vals = [r[metric] for r in results if not np.isnan(r.get(metric, np.nan))]
        if vals:
            output[f'discriminative_{metric}_mean'] = float(np.mean(vals))
            output[f'discriminative_{metric}_std'] = float(np.std(vals))
            
    print(f"      Discriminative baseline ({n_anchors} anchors):")
    print(f"         anchor_error: {output.get('discriminative_anchor_error_mean', 'N/A'):.4f}")
    print(f"         gp_irt_error: {output.get('discriminative_gp_irt_error_mean', 'N/A'):.4f}")
    
    if return_per_model:
        per_model_rows = []
        for r in results:
            model_name = r['model_name']
            row = {'model_name': model_name}
            for metric in ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error', 'true_performance', 'anchor_prediction', 'gp_irt_prediction']:
                val = r.get(metric, np.nan)
                if not np.isnan(val):
                    row[f'discriminative_{metric}_mean'] = float(val)
            per_model_rows.append(row)
        per_model_df = pd.DataFrame(per_model_rows) if per_model_rows else pd.DataFrame()
        return output, per_model_df
        
    return output


def run_random_simple_baseline(
    test_df: pd.DataFrame,
    target_name: str,
    n_random_questions: int,
    n_seeds: int = 1,
    base_seed: int = 42,
    return_per_model: bool = False,
) -> dict | tuple[dict, pd.DataFrame]:
    """Run a simple random baseline: predict performance using just the average of random questions.
    
    This is the simplest possible baseline - no IRT model at all:
    - Select n_random_questions randomly from the target dataset
    - prediction = mean(model's responses to those questions)
    - true_performance = mean(model's responses to ALL questions)
    - error = |prediction - true_performance|
    
    This shows what you get from random sampling without any sophisticated modeling.
    
    Args:
        test_df: Test data with columns ['model_name', 'question_id', 'normalized_score'/'correct', 'dataset']
        target_name: Name of the target dataset
        n_random_questions: Number of random questions to sample
        n_seeds: Number of random seeds to run
        base_seed: Base seed for reproducibility
        return_per_model: If True, also return per-model results DataFrame
    
    Returns:
        If return_per_model=False: Dict with aggregated statistics
        If return_per_model=True: Tuple of (aggregated_dict, per_model_df)
        
        aggregated_dict contains:
        {
            'simple_random_error_mean': float,  # Average |prediction - true| across models and seeds
            'simple_random_error_std': float,   # Std of errors across seeds
            'simple_random_prediction_mean': float,  # Average prediction
            'simple_random_true_perf_mean': float,   # Average true performance
            'n_seeds': int,
            'n_random_questions': int,
        }
        
        per_model_df contains per-model results averaged across seeds
    """
    # Filter to target dataset only
    target_df = test_df[test_df['dataset'] == target_name].copy()
    
    if len(target_df) == 0:
        print(f"      Warning: No data for target '{target_name}' in test_df, skipping simple random baseline")
        return {}
    
    # Determine score column (could be 'correct' or 'normalized_score')
    score_col = 'correct' if 'correct' in target_df.columns else 'normalized_score'
    
    # Get all unique questions
    all_questions = target_df['question_id'].unique()
    
    if len(all_questions) < n_random_questions:
        print(f"      Warning: Only {len(all_questions)} questions available, using all")
        n_random_questions = len(all_questions)
    
    if len(all_questions) < 10:
        print(f"      Warning: Too few questions ({len(all_questions)}) for simple random baseline, skipping")
        return {}
    
    # Get all test models
    test_models = target_df['model_name'].unique()
    
    # Compute true performance for each model (average over ALL questions)
    true_perf_by_model = target_df.groupby('model_name')[score_col].mean().to_dict()
    
    # Collect errors from each seed
    all_seed_errors = []
    all_seed_predictions = []
    
    # Collect per-model results across all seeds (for averaging)
    per_model_results = {}  # model_name -> {metric -> [values across seeds]}
    
    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        np.random.seed(seed)
        
        # Randomly select questions
        random_questions = np.random.choice(all_questions, size=n_random_questions, replace=False)
        
        # Filter to only selected questions
        random_df = target_df[target_df['question_id'].isin(random_questions)]
        
        # Compute prediction for each model (average over random questions)
        pred_by_model = random_df.groupby('model_name')[score_col].mean().to_dict()
        
        # Compute errors
        seed_errors = []
        seed_predictions = []
        for model in test_models:
            if model in pred_by_model and model in true_perf_by_model:
                pred = pred_by_model[model]
                true_perf = true_perf_by_model[model]
                error = abs(pred - true_perf)
                seed_errors.append(error)
                seed_predictions.append(pred)
                
                # Collect per-model results
                if return_per_model:
                    if model not in per_model_results:
                        per_model_results[model] = {
                            'error': [], 'prediction': [], 'true_performance': []
                        }
                    per_model_results[model]['error'].append(error)
                    per_model_results[model]['prediction'].append(pred)
                    per_model_results[model]['true_performance'].append(true_perf)
        
        if seed_errors:
            all_seed_errors.append(np.mean(seed_errors))
            all_seed_predictions.append(np.mean(seed_predictions))
    
    # Compute statistics across seeds
    output = {
        'n_seeds': n_seeds,
        'n_random_questions': n_random_questions,
    }
    
    if all_seed_errors:
        output['simple_random_error_mean'] = float(np.mean(all_seed_errors))
        output['simple_random_error_std'] = float(np.std(all_seed_errors))
        output['simple_random_prediction_mean'] = float(np.mean(all_seed_predictions))
        output['simple_random_true_perf_mean'] = float(np.mean(list(true_perf_by_model.values())))
        
        n_successful = len(all_seed_errors)
        print(f"      Simple random baseline: {n_successful}/{n_seeds} seeds successful")
        print(f"         simple_random_error: {output['simple_random_error_mean']:.4f} ± {output['simple_random_error_std']:.4f}")
    else:
        print(f"      Simple random baseline: 0/{n_seeds} seeds successful")
    
    if return_per_model:
        # Build per-model DataFrame with averaged results across seeds
        per_model_rows = []
        for model_name, metrics in per_model_results.items():
            row = {'model_name': model_name, 'n_seeds': n_seeds}
            for metric, values in metrics.items():
                if values:
                    row[f'simple_random_{metric}_mean'] = float(np.mean(values))
                    row[f'simple_random_{metric}_std'] = float(np.std(values))
            per_model_rows.append(row)
        per_model_df = pd.DataFrame(per_model_rows) if per_model_rows else pd.DataFrame()
        return output, per_model_df
    
    return output


# =============================================================================
