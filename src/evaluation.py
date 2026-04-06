"""Validation and baseline evaluation routines."""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd

from irt import compute_lambda_values, run_estimation_validation


def _get_irt_attrs(item_params: pd.DataFrame) -> tuple[dict, int]:
    attrs = getattr(item_params, 'attrs', {})
    validation_errors = attrs.get('validation_errors', {})
    best_dim = attrs.get('best_dimension', 5)
    dims_search = attrs.get('config_dims_search', [5, 10])
    best_dim_idx = dims_search.index(best_dim) if best_dim in dims_search else 0
    return validation_errors, best_dim_idx


def _run_estimation_with_anchors(
    item_params: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    anchors_by_dataset: dict[str, list[str]],
    anchor_weights_by_dataset: dict[str, list[float]],
    n_anchors: int,
    A_matrix: np.ndarray | None,
    B_matrix: np.ndarray | None,
    precomputed_thetas: dict[str, float] | None,
) -> list[dict]:
    """Run estimation validation for a given anchor set; suppress stdout/warnings."""
    validation_errors, best_dim_idx = _get_irt_attrs(item_params)
    lambdas_by_dataset = compute_lambda_values(
        original_matrix_df=train_df, validation_errors=validation_errors,
        best_dim_idx=best_dim_idx, number_item=n_anchors,
    )
    question_ids_order = list(item_params.index) if hasattr(item_params, 'index') else None
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return run_estimation_validation(
            test_matrix=test_df, item_params=item_params,
            anchors_by_dataset=anchors_by_dataset, lambdas_by_dataset=lambdas_by_dataset,
            anchor_weights_by_dataset=anchor_weights_by_dataset,
            precomputed_thetas=precomputed_thetas,
            A_matrix=A_matrix, B_matrix=B_matrix,
            question_ids_order=question_ids_order,
        )


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
    """Run estimation validation using anchors split by dataset prefix."""
    validation_errors, best_dim_idx = _get_irt_attrs(item_params)
    datasets_in_test = test_df['dataset'].unique()
    anchors_by_dataset: dict = {}
    anchor_weights_by_dataset: dict = {}
    for ds in datasets_in_test:
        ds_anchors = [a for a in anchor_ids if a.startswith(f"{ds}:")]
        if ds_anchors:
            indices = [anchor_ids.index(a) for a in ds_anchors]
            anchors_by_dataset[ds] = ds_anchors
            anchor_weights_by_dataset[ds] = [anchor_weights[i] for i in indices]
        else:
            anchors_by_dataset[ds] = anchor_ids
            anchor_weights_by_dataset[ds] = anchor_weights

    lambdas_by_dataset = compute_lambda_values(
        original_matrix_df=train_df, validation_errors=validation_errors,
        best_dim_idx=best_dim_idx, number_item=len(anchor_ids),
    )
    question_ids_order = list(item_params.index) if hasattr(item_params, 'index') else None
    return run_estimation_validation(
        test_matrix=test_df, item_params=item_params,
        anchors_by_dataset=anchors_by_dataset, lambdas_by_dataset=lambdas_by_dataset,
        anchor_weights_by_dataset=anchor_weights_by_dataset,
        precomputed_thetas=precomputed_thetas,
        A_matrix=A_matrix, B_matrix=B_matrix,
        question_ids_order=question_ids_order,
    )


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
    """Run validation using randomly selected questions as anchors (multi-seed)."""
    target_questions = [q for q in item_params.index if q.startswith(f"{target_name}:")]
    if len(target_questions) < 10:
        print(f"      Warning: Too few questions ({len(target_questions)}) for random baseline, skipping")
        return ({}, pd.DataFrame()) if return_per_model else {}
    if len(target_questions) < n_random_questions:
        print(f"      Warning: Only {len(target_questions)} questions available, using all")
        n_random_questions = len(target_questions)

    seed_errors: dict = {m: [] for m in ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']}
    per_model_results: dict = {}

    for seed_offset in range(n_seeds):
        np.random.seed(base_seed + seed_offset)
        random_anchors = list(np.random.choice(target_questions, size=n_random_questions, replace=False))
        random_weights = [1.0 / n_random_questions] * n_random_questions
        results = _run_estimation_with_anchors(
            item_params=item_params, test_df=test_df, train_df=train_df,
            anchors_by_dataset={target_name: random_anchors},
            anchor_weights_by_dataset={target_name: random_weights},
            n_anchors=n_random_questions, A_matrix=A_matrix, B_matrix=B_matrix,
            precomputed_thetas=precomputed_thetas,
        )
        if not results:
            continue
        for metric in seed_errors:
            vals = [r[metric] for r in results if not np.isnan(r.get(metric, np.nan))]
            if vals:
                seed_errors[metric].append(np.mean(vals))
        if return_per_model:
            for r in results:
                m = r['model_name']
                if m not in per_model_results:
                    per_model_results[m] = {k: [] for k in ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error', 'true_performance', 'anchor_prediction', 'gp_irt_prediction']}
                for metric in per_model_results[m]:
                    val = r.get(metric, np.nan)
                    if not np.isnan(val):
                        per_model_results[m][metric].append(val)

    output: dict = {'n_seeds': n_seeds, 'n_random_questions': n_random_questions}
    for metric, vals in seed_errors.items():
        if vals:
            output[f'random_{metric}_mean'] = float(np.mean(vals))
            output[f'random_{metric}_std'] = float(np.std(vals))

    n_ok = len(seed_errors['anchor_error'])
    print(f"      Random baseline: {n_ok}/{n_seeds} seeds successful")
    if n_ok:
        print(f"         anchor_error: {output.get('random_anchor_error_mean', 'N/A'):.4f} ± {output.get('random_anchor_error_std', 'N/A'):.4f}")
        print(f"         gp_irt_error: {output.get('random_gp_irt_error_mean', 'N/A'):.4f} ± {output.get('random_gp_irt_error_std', 'N/A'):.4f}")

    if not return_per_model:
        return output
    rows = []
    for model_name, metrics in per_model_results.items():
        row = {'model_name': model_name, 'n_seeds': n_seeds}
        for metric, values in metrics.items():
            if values:
                row[f'random_{metric}_mean'] = float(np.mean(values))
                row[f'random_{metric}_std'] = float(np.std(values))
        rows.append(row)
    return output, pd.DataFrame(rows) if rows else pd.DataFrame()


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
    """Run validation using top-K most discriminative items as anchors."""
    target_questions = [q for q in item_params.index if q.startswith(f"{target_name}:")]
    if len(target_questions) < 5:
        print(f"      Warning: Too few questions ({len(target_questions)}) for discriminative baseline, skipping")
        return ({}, pd.DataFrame()) if return_per_model else {}
    if len(target_questions) < n_anchors:
        print(f"      Warning: Only {len(target_questions)} questions available, using all")
        n_anchors = len(target_questions)

    target_params = item_params.loc[target_questions].copy()
    if 'discrimination' in target_params.columns:
        target_params['disc_norm'] = target_params['discrimination']
    elif any(c.startswith('a_') for c in target_params.columns):
        a_cols = [c for c in target_params.columns if c.startswith('a_')]
        target_params['disc_norm'] = np.linalg.norm(target_params[a_cols].values, axis=1)
    else:
        target_params['disc_norm'] = target_params.iloc[:, 0]

    top_k = target_params.sort_values('disc_norm', ascending=False).head(n_anchors).index.tolist()
    weights = [1.0 / n_anchors] * n_anchors

    results = _run_estimation_with_anchors(
        item_params=item_params, test_df=test_df, train_df=train_df,
        anchors_by_dataset={target_name: top_k},
        anchor_weights_by_dataset={target_name: weights},
        n_anchors=n_anchors, A_matrix=A_matrix, B_matrix=B_matrix,
        precomputed_thetas=precomputed_thetas,
    )
    if not results:
        return ({}, pd.DataFrame()) if return_per_model else {}

    output: dict = {'n_anchors': n_anchors}
    for metric in ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']:
        vals = [r[metric] for r in results if not np.isnan(r.get(metric, np.nan))]
        if vals:
            output[f'discriminative_{metric}_mean'] = float(np.mean(vals))
            output[f'discriminative_{metric}_std'] = float(np.std(vals))
    print(f"      Discriminative baseline ({n_anchors} anchors):")
    print(f"         anchor_error: {output.get('discriminative_anchor_error_mean', 'N/A'):.4f}")
    print(f"         gp_irt_error: {output.get('discriminative_gp_irt_error_mean', 'N/A'):.4f}")

    if not return_per_model:
        return output
    rows = []
    for r in results:
        row = {'model_name': r['model_name']}
        for metric in ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error', 'true_performance', 'anchor_prediction', 'gp_irt_prediction']:
            val = r.get(metric, np.nan)
            if not np.isnan(val):
                row[f'discriminative_{metric}_mean'] = float(val)
        rows.append(row)
    return output, pd.DataFrame(rows) if rows else pd.DataFrame()


def run_random_simple_baseline(
    test_df: pd.DataFrame,
    target_name: str,
    n_random_questions: int,
    n_seeds: int = 1,
    base_seed: int = 42,
    return_per_model: bool = False,
) -> dict | tuple[dict, pd.DataFrame]:
    """Predict performance as mean of random questions (no IRT)."""
    target_df = test_df[test_df['dataset'] == target_name].copy()
    if len(target_df) == 0:
        return ({}, pd.DataFrame()) if return_per_model else {}
    score_col = 'correct' if 'correct' in target_df.columns else 'normalized_score'
    all_questions = target_df['question_id'].unique()
    if len(all_questions) < 10:
        print(f"      Warning: Too few questions ({len(all_questions)}) for simple random baseline, skipping")
        return ({}, pd.DataFrame()) if return_per_model else {}
    if len(all_questions) < n_random_questions:
        n_random_questions = len(all_questions)

    true_perf = target_df.groupby('model_name')[score_col].mean().to_dict()
    seed_errors: list = []
    seed_preds: list = []
    per_model_results: dict = {}

    for seed_offset in range(n_seeds):
        np.random.seed(base_seed + seed_offset)
        chosen = np.random.choice(all_questions, size=n_random_questions, replace=False)
        pred = target_df[target_df['question_id'].isin(chosen)].groupby('model_name')[score_col].mean().to_dict()
        errs, preds = [], []
        for model in target_df['model_name'].unique():
            if model in pred and model in true_perf:
                e = abs(pred[model] - true_perf[model])
                errs.append(e)
                preds.append(pred[model])
                if return_per_model:
                    if model not in per_model_results:
                        per_model_results[model] = {'error': [], 'prediction': [], 'true_performance': []}
                    per_model_results[model]['error'].append(e)
                    per_model_results[model]['prediction'].append(pred[model])
                    per_model_results[model]['true_performance'].append(true_perf[model])
        if errs:
            seed_errors.append(np.mean(errs))
            seed_preds.append(np.mean(preds))

    output: dict = {'n_seeds': n_seeds, 'n_random_questions': n_random_questions}
    if seed_errors:
        output['simple_random_error_mean'] = float(np.mean(seed_errors))
        output['simple_random_error_std'] = float(np.std(seed_errors))
        output['simple_random_prediction_mean'] = float(np.mean(seed_preds))
        output['simple_random_true_perf_mean'] = float(np.mean(list(true_perf.values())))
        print(f"      Simple random baseline: {len(seed_errors)}/{n_seeds} seeds successful")
        print(f"         simple_random_error: {output['simple_random_error_mean']:.4f} ± {output['simple_random_error_std']:.4f}")

    if not return_per_model:
        return output
    rows = []
    for model_name, metrics in per_model_results.items():
        row = {'model_name': model_name, 'n_seeds': n_seeds}
        for metric, values in metrics.items():
            if values:
                row[f'simple_random_{metric}_mean'] = float(np.mean(values))
                row[f'simple_random_{metric}_std'] = float(np.std(values))
        rows.append(row)
    return output, pd.DataFrame(rows) if rows else pd.DataFrame()
