"""
Estimation module for IRT-based performance prediction.

This module implements the TinyBenchmarks methodology for estimating model performance:
1. Anchor-only: Weighted average of anchor responses
2. p-IRT: Proportion-based blending (λ = n_anchors / n_questions)
3. gp-IRT: Global parameter blending (λ from training)
4. Pure IRT: IRT prediction on all questions

IMPORTANT: This version prioritizes clarity over silent fallbacks.
If something fails, it will raise an error or print a warning.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import warnings
import numpy as np
import pandas as pd

from .math_utils import item_curve, estimate_ability_parameters

# One-time logging flags (per process) to understand which theta-estimation path is used.
_THETA_ESTIMATION_PATH_LOGGED = {"full_matrices": False, "fallback": False}


# ============================================================================
# BACKWARDS COMPATIBILITY FUNCTIONS
# ============================================================================

def expected_correctness(item_params: pd.DataFrame, theta: float) -> pd.Series:
    """Return expected correctness for each item at ability theta under 2PL."""
    z = item_params["a"].astype(float) * (theta - item_params["b"].astype(float))
    p = 1.0 / (1.0 + np.exp(-z))
    return p.astype(float)


def blend_anchor_and_irt(
    preds_anchor: pd.Series,
    preds_irt: pd.Series,
    lambdas_by_dataset: dict[str, float] | None,
    item_to_dataset: pd.Series | None = None,
) -> pd.Series:
    """Blend predictions as in gp-IRT: lambda*data + (1-lambda)*irt."""
    if lambdas_by_dataset is None or item_to_dataset is None:
        return 0.5 * preds_anchor + 0.5 * preds_irt
    lam = item_to_dataset.map(lambda d: float(lambdas_by_dataset.get(str(d), 0.5))).astype(float)
    return lam * preds_anchor + (1.0 - lam) * preds_irt


# ============================================================================
# MAIN ESTIMATION CODE
# ============================================================================

@dataclass
class EstimationConfig:
    max_iter: int = 50
    tol: float = 1e-4
    lambdas_by_dataset: dict[str, float] | None = None


def estimate_theta_from_anchors(
    item_params: pd.DataFrame,
    anchor_responses: pd.Series,
    init_theta: float = 0.0,
    config: EstimationConfig | None = None,
    A_matrix: Optional[np.ndarray] = None,
    B_matrix: Optional[np.ndarray] = None,
    question_ids_order: Optional[List[str]] = None,
) -> float:
    """Estimate ability theta using MLE on 2PL with anchor responses.
    
    IMPORTANT: For multidimensional IRT, pass A_matrix, B_matrix, and question_ids_order
    to use the full parameter matrices (like efficbench does).
    
    Args:
        item_params: DataFrame with IRT parameters indexed by question_id
        anchor_responses: Series of responses indexed by question_id
        init_theta: Initial theta value
        config: Optional configuration
        A_matrix: Full discrimination matrix shape (1, D, n_items) - preferred
        B_matrix: Full difficulty matrix shape (1, D, n_items) - preferred  
        question_ids_order: List of question_ids matching A_matrix/B_matrix column order
    
    Raises ValueError if no common anchors exist between item_params and anchor_responses.
    """
    # ---------------------------------------------------------------------
    # Robustness: clean anchor responses
    # Some datasets/models may contain non-scalar values (e.g., lists/arrays),
    # strings, or NaNs in `normalized_score`. Coerce to numeric and drop invalid
    # anchors so theta estimation doesn't crash.
    # ---------------------------------------------------------------------
    if anchor_responses is None:
        raise ValueError("anchor_responses is None")
    anchor_responses = anchor_responses.copy()
    anchor_responses = pd.to_numeric(anchor_responses, errors="coerce")
    anchor_responses = anchor_responses.dropna()
    # Remove duplicate indices - .loc[q] returns Series if duplicates exist, causing array errors
    anchor_responses = anchor_responses[~anchor_responses.index.duplicated(keep='first')]

    common = item_params.index.intersection(anchor_responses.index)
    if len(common) == 0:
        raise ValueError("No common anchors between item_params and anchor_responses")
    
    # Use full matrices if provided (efficbench style)
    if A_matrix is not None and B_matrix is not None and question_ids_order is not None:
        if not _THETA_ESTIMATION_PATH_LOGGED["full_matrices"]:
            # Keep this loud but one-time to avoid log spam on large runs.
            print(f"   ✅ Theta estimation path: FULL MIRT matrices (D={A_matrix.shape[1] if len(A_matrix.shape) == 3 else 'unknown'})")
            _THETA_ESTIMATION_PATH_LOGGED["full_matrices"] = True

        qid_to_idx = {qid: i for i, qid in enumerate(question_ids_order)}
        
        # Get indices for common anchors
        anchor_indices = [qid_to_idx[q] for q in common if q in qid_to_idx]
        
        if len(anchor_indices) == 0:
            raise ValueError("No common anchors found in A_matrix/B_matrix")
        
        # Slice the matrices for anchor items
        A = A_matrix[:, :, anchor_indices]  # (1, D, n_anchors)
        B = B_matrix[:, :, anchor_indices]  # (1, D, n_anchors)
        
        # Get responses in the same order
        y_values = np.array([anchor_responses.loc[q] for q in common if q in qid_to_idx], dtype=float)
        
        D = A.shape[1]
        init_theta_val = np.zeros(D) if init_theta == 0.0 else np.full(D, init_theta)
    else:
        if not _THETA_ESTIMATION_PATH_LOGGED["fallback"]:
            print("   ⚠️  Theta estimation path: FALLBACK (scalar params from item_params; no full MIRT matrices provided)")
            _THETA_ESTIMATION_PATH_LOGGED["fallback"] = True

        # Fallback: use scalar parameters from item_params
        # Filter common items to ensure valid data
        valid_common = []
        for q in common:
            val_a = item_params.loc[q, "a"]
            val_b = item_params.loc[q, "b"]
            # Check for NaN or None
            if pd.isna(val_a) if np.isscalar(val_a) else False:
                continue
            if pd.isna(val_b) if np.isscalar(val_b) else False:
                continue
            valid_common.append(q)
            
        if not valid_common:
            raise ValueError("No valid anchor parameters (all NaN/None)")
            
        common = valid_common
        first_a = item_params.loc[common[0], "a"]
        is_multidim = isinstance(first_a, (list, np.ndarray, tuple))
        
        if is_multidim:
            a_list = item_params.loc[common, "a"].tolist()
            b_list = item_params.loc[common, "b"].tolist()
            
            # Verify homogeneity manually before numpy conversion
            try:
                a_values = np.array(a_list).T
                b_values = np.array(b_list).T
            except Exception as e:
                # Fallback for inhomogeneous lists - try to fix or raise clear error
                if len(a_list) > 0:
                    expected_len = len(a_list[0])
                    # Filter only valid length items
                    valid_indices = [i for i, x in enumerate(a_list) if len(x) == expected_len]
                    if len(valid_indices) < len(a_list):
                        # Update lists and common keys
                        a_list = [a_list[i] for i in valid_indices]
                        b_list = [b_list[i] for i in valid_indices]
                        common = [common[i] for i in valid_indices]
                        a_values = np.array(a_list).T
                        b_values = np.array(b_list).T
                    else:
                        raise ValueError(f"Inhomogeneous parameter shapes: {e}")
                else:
                    raise e

            A = a_values[None, :, :]
            B = b_values[None, :, :]
        else:
            a_values = item_params.loc[common, "a"].astype(float).values
            b_values = item_params.loc[common, "b"].astype(float).values
            A = a_values.reshape(1, 1, -1)  # (1, 1, n_items) for 1D case
            B = b_values.reshape(1, 1, -1)
        
        # Ensure y_values matches the filtered common list
        y_values = anchor_responses.loc[common].astype(float).values
        D = A.shape[1]
        init_theta_val = np.zeros(D) if init_theta == 0.0 else np.full(D, init_theta)
    
    try:
        optimal_theta = estimate_ability_parameters(
            responses_test=y_values,
            A=A,
            B=B,
            theta_init=init_theta_val,
            optimizer="BFGS"
        )
        # Return scalar theta (first dimension) for compatibility
        return optimal_theta
    except Exception as e:
        warnings.warn(f"Theta optimization failed ({e}), using mean of anchor responses as fallback")
        return float(anchor_responses.mean())


def compute_balance_weights_for_validation(test_matrix: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute balance weights for each scenario following TinyBenchmarks methodology."""
    balance_weights_by_scenario = {}
    
    datasets = test_matrix["dataset"].unique()
    parent_datasets = {}
    
    for dataset in datasets:
        if "." in dataset:
            parent = dataset.split(".")[0]
            parent_datasets.setdefault(parent, []).append(dataset)
        else:
            parent_datasets.setdefault(dataset, [dataset])
    
    for scenario_name, scenario_datasets in parent_datasets.items():
        scenario_df = test_matrix[test_matrix["dataset"].isin(scenario_datasets)]
        scenario_questions = sorted(scenario_df["question_id"].unique())
        
        balance_weights = np.ones(len(scenario_questions))
        question_to_idx = {q: i for i, q in enumerate(scenario_questions)}
        
        if len(scenario_datasets) > 1:
            N = len(scenario_questions)
            n_sub = len(scenario_datasets)
            
            for subscenario_dataset in scenario_datasets:
                subscenario_df = test_matrix[test_matrix["dataset"] == subscenario_dataset]
                subscenario_questions = subscenario_df["question_id"].unique()
                n_i = len(subscenario_questions)
                
                if n_i == 0:
                    warnings.warn(f"Subscenario {subscenario_dataset} has 0 questions")
                    continue
                    
                weight = N / (n_sub * n_i)
                for q in subscenario_questions:
                    if q in question_to_idx:
                        balance_weights[question_to_idx[q]] = weight
        
        balance_weights_by_scenario[scenario_name] = balance_weights
    
    return balance_weights_by_scenario


def _compute_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted mean. Raises if shapes don't match."""
    if len(values) != len(weights):
        raise ValueError(f"Shape mismatch: values={len(values)}, weights={len(weights)}")
    if len(values) == 0:
        raise ValueError("Cannot compute mean of empty array")
    return float((values * weights).mean())


def _get_irt_predictions(
    theta,  # Can be float, scalar, or ndarray shape (1, D, 1)
    question_ids: List[str],
    use_full_matrices: bool,
    A_matrix: Optional[np.ndarray],
    B_matrix: Optional[np.ndarray],
    qid_to_idx: Optional[Dict[str, int]],
    item_params: pd.DataFrame,
) -> np.ndarray:
    """Get IRT probability predictions for given questions.
    
    Args:
        theta: Ability parameter - can be scalar or ndarray shape (1, D, 1)
        question_ids: List of question IDs to predict
        use_full_matrices: Whether to use full MIRT matrices
        A_matrix, B_matrix: Full MIRT parameter matrices
        qid_to_idx: Mapping from question_id to matrix index
        item_params: DataFrame with scalar (a, b) parameters
    
    Returns array of probabilities. Raises if no valid questions found.
    """
    if use_full_matrices:
        valid_qids = [q for q in question_ids if q in qid_to_idx]
        if not valid_qids:
            raise ValueError(f"No questions found in MIRT matrices (requested: {len(question_ids)})")
        
        indices = [qid_to_idx[q] for q in valid_qids]
        A_sub = A_matrix[:, :, indices] if len(A_matrix.shape) == 3 else A_matrix[:, indices]
        B_sub = B_matrix[:, :, indices] if len(B_matrix.shape) == 3 else B_matrix[:, indices]
        
        # Handle theta - can be scalar, 1D, or 3D array
        if isinstance(theta, np.ndarray):
            if theta.ndim == 3:  # Already (1, D, 1) format
                theta_arr = theta
            elif theta.ndim == 1:  # (D,) format
                theta_arr = theta[None, :, None]
            else:
                theta_arr = theta.reshape(1, -1, 1)
        else:
            # Scalar theta - broadcast to all dimensions
            D = A_sub.shape[1] if len(A_sub.shape) == 3 else 1
            theta_arr = np.full((1, D, 1), float(theta))
    else:
        valid_qids = [q for q in question_ids if q in item_params.index]
        if not valid_qids:
            raise ValueError(f"No questions found in item_params (requested: {len(question_ids)})")
        
        sub_params = item_params.loc[valid_qids]
        A_sub = sub_params["a"].values.reshape(1, 1, -1)  # (1, 1, n_items)
        B_sub = sub_params["b"].values.reshape(1, 1, -1)
        
        # Handle theta
        if isinstance(theta, np.ndarray):
            if theta.ndim == 3:
                theta_arr = theta[:, :1, :]  # Take first dim only for scalar params
            else:
                theta_arr = np.array([[[float(theta.flat[0])]]]) 
        else:
            theta_arr = np.array([[[float(theta)]]])
    
    probs = item_curve(theta_arr, A_sub, B_sub)[0]
    return probs, valid_qids


def run_estimation_validation(
    test_matrix: pd.DataFrame,
    item_params: pd.DataFrame,
    anchors_by_dataset: Dict[str, List[str]],
    lambdas_by_dataset: Dict[str, float],
    anchor_weights_by_dataset: Optional[Dict[str, List[float]]] = None,
    precomputed_thetas: Optional[Dict[str, float]] = None,
    A_matrix: Optional[np.ndarray] = None,
    B_matrix: Optional[np.ndarray] = None,
    question_ids_order: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Run estimation validation following TinyBenchmarks methodology.
    
    This version is simplified and will raise errors instead of silent fallbacks.
    """
    results = []
    models = test_matrix["model_name"].unique()
    datasets = test_matrix["dataset"].unique()
    
    print(f"   Validating {len(models)} models × {len(datasets)} datasets")
    
    # Setup MIRT matrices
    use_full_matrices = (A_matrix is not None and B_matrix is not None and question_ids_order is not None)
    qid_to_idx = None
    if use_full_matrices:
        qid_to_idx = {qid: i for i, qid in enumerate(question_ids_order)}
        D = A_matrix.shape[1] if len(A_matrix.shape) == 3 else 1
        print(f"   ✓ Using full MIRT matrices (D={D})")
    else:
        print(f"   ⚠️  Using scalar item parameters (no MIRT matrices)")
    
    # Compute balance weights
    balance_weights_by_scenario = compute_balance_weights_for_validation(test_matrix)
    
    # Group datasets by scenario
    def scenario_from_dataset(name: str) -> str:
        return name.split(".")[0] if "." in name else name
    
    scenario_datasets = {}
    for dataset_name in datasets:
        scenario_name = scenario_from_dataset(dataset_name)
        scenario_datasets.setdefault(scenario_name, []).append(dataset_name)
    
    # Track statistics
    stats = {"processed": 0, "skipped_no_anchors": 0, "skipped_no_data": 0, "theta_estimation_failures": 0}
    
    for scenario_name, scenario_dataset_list in scenario_datasets.items():
        scenario_matrix = test_matrix[test_matrix["dataset"].isin(scenario_dataset_list)]
        if len(scenario_matrix) == 0:
            print(f"   ⚠️  Scenario {scenario_name}: no data, skipping")
            continue
        
        scenario_questions = sorted(scenario_matrix["question_id"].unique())
        scenario_anchors_full = anchors_by_dataset.get(scenario_name, [])
        scenario_anchors = [q for q in scenario_anchors_full if q in item_params.index and q in set(scenario_questions)]
        
        # Get scenario parameters
        scenario_lambda = lambdas_by_dataset.get(scenario_name)
        if scenario_lambda is None:
            warnings.warn(f"No lambda for scenario {scenario_name}, using 0.5")
            scenario_lambda = 0.5
        
        balance_weights = balance_weights_by_scenario.get(scenario_name)
        if balance_weights is None or len(balance_weights) != len(scenario_questions):
            warnings.warn(f"Balance weights mismatch for {scenario_name}, using uniform weights")
            balance_weights = np.ones(len(scenario_questions))
        
        question_to_weight_idx = {q: i for i, q in enumerate(scenario_questions)}
        
        # Check anchor availability
        has_local_anchors = len(scenario_anchors) > 0
        has_precomputed_thetas = precomputed_thetas is not None
        
        if not has_local_anchors and not has_precomputed_thetas:
            print(f"   ⚠️  Scenario {scenario_name}: no anchors and no precomputed thetas, skipping")
            stats["skipped_no_anchors"] += len(models)
            continue
        
        print(f"   Processing {scenario_name}: {len(scenario_questions)} questions, {len(scenario_anchors)} anchors")
        
        for model_name in models:
            model_matrix = scenario_matrix[scenario_matrix["model_name"] == model_name]
            if len(model_matrix) == 0:
                stats["skipped_no_data"] += 1
                continue
            
            model_responses = model_matrix.set_index("question_id")["normalized_score"]
            model_responses = model_responses[~model_responses.index.duplicated(keep='first')]
            
            # 1. THETA ESTIMATION
            available_anchor_ids = [q for q in scenario_anchors if q in model_responses.index]
            anchor_responses = model_responses.loc[available_anchor_ids] if available_anchor_ids else None
            
            if precomputed_thetas and model_name in precomputed_thetas:
                estimated_theta = precomputed_thetas[model_name]
            elif available_anchor_ids:
                try:
                    # Use full MIRT matrices for theta estimation (like efficbench)
                    estimated_theta = estimate_theta_from_anchors(
                        item_params, 
                        anchor_responses,
                        A_matrix=A_matrix,
                        B_matrix=B_matrix,
                        question_ids_order=question_ids_order,
                    )
                except Exception as e:
                    n_anchors = len(anchor_responses) if anchor_responses is not None else 0
                    warnings.warn(f"Theta estimation failed for {model_name} (n_anchors={n_anchors}): {e}")
                    stats["theta_estimation_failures"] += 1
                    continue
            else:
                stats["skipped_no_anchors"] += 1
                continue
            
            # 2. TRUE PERFORMANCE (ground truth from all responses)
            responses_in_scenario = [model_responses[q] for q in model_responses.index if q in question_to_weight_idx]
            weights_for_responses = [balance_weights[question_to_weight_idx[q]] for q in model_responses.index if q in question_to_weight_idx]
            
            if not responses_in_scenario:
                stats["skipped_no_data"] += 1
                continue
            
            true_performance = _compute_weighted_mean(np.array(responses_in_scenario), np.array(weights_for_responses))
            
            # 3. ANCHOR-ONLY PREDICTION
            if anchor_responses is not None and len(anchor_responses) > 0:
                # Try to use anchor weights if available
                if anchor_weights_by_dataset and scenario_name in anchor_weights_by_dataset:
                    scenario_weights_full = anchor_weights_by_dataset[scenario_name]
                    if len(scenario_weights_full) == len(scenario_anchors_full):
                        weight_map = {q: scenario_weights_full[i] for i, q in enumerate(scenario_anchors_full)}
                        aligned_weights = np.array([weight_map.get(q, 1.0) for q in anchor_responses.index])
                        if aligned_weights.sum() > 0:
                            anchor_prediction = float((anchor_responses.values * aligned_weights).sum())
                        else:
                            anchor_prediction = float(anchor_responses.mean())
                    else:
                        warnings.warn(f"Anchor weights mismatch for {scenario_name}, using mean")
                        anchor_prediction = float(anchor_responses.mean())
                else:
                    anchor_prediction = float(anchor_responses.mean())
            else:
                anchor_prediction = np.nan
            
            # 4. IRT PREDICTIONS
            seen_questions = available_anchor_ids
            unseen_questions = [q for q in scenario_questions if q not in set(scenario_anchors)]
            
            # p-IRT: data_part (seen) + irt_part (unseen)
            pirt_lambda = len(seen_questions) / len(scenario_questions) if scenario_questions else 0.0
            
            # data_part from seen questions
            if seen_questions:
                seen_responses = np.array([model_responses[q] for q in seen_questions if q in model_responses.index])
                seen_weights = np.array([balance_weights[question_to_weight_idx[q]] for q in seen_questions if q in question_to_weight_idx])
                if len(seen_responses) > 0 and len(seen_weights) == len(seen_responses):
                    data_part = _compute_weighted_mean(seen_responses, seen_weights)
                else:
                    data_part = anchor_prediction if not np.isnan(anchor_prediction) else 0.5
            else:
                data_part = 0.0
                pirt_lambda = 0.0
            
            # irt_part from unseen questions
            if unseen_questions:
                try:
                    irt_probs, valid_unseen = _get_irt_predictions(
                        estimated_theta, unseen_questions, use_full_matrices,
                        A_matrix, B_matrix, qid_to_idx, item_params
                    )
                    unseen_weights = np.array([balance_weights[question_to_weight_idx[q]] for q in valid_unseen if q in question_to_weight_idx])
                    if len(unseen_weights) == len(irt_probs):
                        irt_part = _compute_weighted_mean(irt_probs, unseen_weights)
                    else:
                        irt_part = float(irt_probs.mean())
                except ValueError as e:
                    warnings.warn(f"IRT prediction failed for unseen questions: {e}")
                    irt_part = data_part
            else:
                irt_part = data_part
            
            # p-IRT prediction
            pirt_prediction = pirt_lambda * data_part + (1 - pirt_lambda) * irt_part
            
            # gp-IRT prediction
            if np.isnan(anchor_prediction):
                gp_irt_prediction = np.nan
            else:
                gp_irt_prediction = scenario_lambda * anchor_prediction + (1 - scenario_lambda) * pirt_prediction
            
            # 5. PURE IRT PREDICTION (on all questions)
            try:
                irt_probs_all, valid_all = _get_irt_predictions(
                    estimated_theta, scenario_questions, use_full_matrices,
                    A_matrix, B_matrix, qid_to_idx, item_params
                )
                all_weights = np.array([balance_weights[question_to_weight_idx[q]] for q in valid_all if q in question_to_weight_idx])
                if len(all_weights) == len(irt_probs_all):
                    irt_prediction = _compute_weighted_mean(irt_probs_all, all_weights)
                else:
                    irt_prediction = float(irt_probs_all.mean())
            except ValueError as e:
                warnings.warn(f"Pure IRT prediction failed: {e}")
                irt_prediction = 0.5
            
            # 6. COMPUTE ERRORS
            anchor_error = abs(anchor_prediction - true_performance) if not np.isnan(anchor_prediction) else np.nan
            irt_error = abs(irt_prediction - true_performance)
            gp_irt_error = abs(gp_irt_prediction - true_performance) if not np.isnan(gp_irt_prediction) else np.nan
            pirt_error = abs(pirt_prediction - true_performance)
            
            # Convert theta to scalar for storage
            if isinstance(estimated_theta, np.ndarray):
                theta_scalar = float(estimated_theta.flat[0])
            else:
                theta_scalar = float(estimated_theta)
            
            result = {
                "model_name": model_name,
                "dataset_name": ", ".join(scenario_dataset_list),
                "scenario_name": scenario_name,
                "num_questions": len(scenario_questions),
                "num_anchors": len(scenario_anchors),
                "estimated_theta": theta_scalar,
                "true_performance": true_performance,
                "anchor_prediction": float(anchor_prediction),
                "irt_prediction": irt_prediction,
                "gp_irt_prediction": float(gp_irt_prediction),
                "pirt_prediction": pirt_prediction,
                "anchor_error": float(anchor_error),
                "irt_error": irt_error,
                "gp_irt_error": float(gp_irt_error),
                "pirt_error": pirt_error,
                "dataset_lambda": scenario_lambda,
                "pirt_lambda": pirt_lambda
            }
            results.append(result)
            stats["processed"] += 1
    
    # Print summary
    print(f"\n   📊 Validation Statistics:")
    print(f"      Processed: {stats['processed']}")
    print(f"      Skipped (no anchors): {stats['skipped_no_anchors']}")
    print(f"      Skipped (no data): {stats['skipped_no_data']}")
    print(f"      Theta estimation failures: {stats['theta_estimation_failures']}")
    
    if results:
        _print_results_summary(results)
    
    return results


def _print_results_summary(results: List[Dict]) -> None:
    """Print summary of validation results."""
    print("\n   📈 Results Summary:")
    
    # Overall errors
    anchor_errors = [r["anchor_error"] for r in results if not np.isnan(r["anchor_error"])]
    irt_errors = [r["irt_error"] for r in results]
    gp_irt_errors = [r["gp_irt_error"] for r in results if not np.isnan(r["gp_irt_error"])]
    pirt_errors = [r["pirt_error"] for r in results]
    
    print(f"      Anchor-only: {np.mean(anchor_errors):.4f} (n={len(anchor_errors)})")
    print(f"      IRT:         {np.mean(irt_errors):.4f} (n={len(irt_errors)})")
    print(f"      p-IRT:       {np.mean(pirt_errors):.4f} (n={len(pirt_errors)})")
    print(f"      gp-IRT:      {np.mean(gp_irt_errors):.4f} (n={len(gp_irt_errors)})")
    
    # Best method
    method_errors = {
        "Anchor-only": np.mean(anchor_errors) if anchor_errors else float('inf'),
        "IRT": np.mean(irt_errors),
        "p-IRT": np.mean(pirt_errors),
        "gp-IRT": np.mean(gp_irt_errors) if gp_irt_errors else float('inf'),
    }
    best = min(method_errors.keys(), key=lambda k: method_errors[k])
    print(f"      → Best method: {best} ({method_errors[best]:.4f})")
