
"""
IRT Model Training for Question Selection.

This module replicates EXACTLY the TinyBenchmarks notebook workflow for IRT training.
It follows the notebook cells step-by-step to ensure identical results.

Notebook workflow (training_irt.ipynb):
1. Load data and prepare scenarios structure (Cells 1-5)  
2. Create response matrix Y (Cell 5)
3. Compute balance weights for MMLU subscenarios (Cell 7)
4. Perform binarization with threshold optimization (Cell 10)
5. Dimension validation with cross-validation (Cell 11)
6. Train final IRT model (Cell 14)
7. Compute lambda values for each scenario (Cells 17-18)

This module imports and uses the exact functions from irt.py and utils.py
to maintain complete compatibility with the notebook workflow.
"""

from dataclasses import dataclass, field
from typing import Any
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import json
import torch

# Import the exact functions from notebook files
from irt.core import create_irt_dataset, train_irt_model, train_irt_model_python_api, load_irt_parameters, load_irt_parameters_from_trainer, estimate_ability_parameters
from irt.math_utils import sigmoid, item_curve


def get_best_device() -> str:
    """Auto-detect the best available device for py-irt training.
    
    Returns 'cuda' if available, otherwise 'cpu'.
    
    Note: MPS (Apple Silicon GPU) is not supported by py-irt because:
    1. py-irt validates device to only accept 'cpu' or 'cuda'
    2. Even with a monkey-patch, PyTorch's MPS backend is missing operators
       needed for Pyro's probabilistic sampling (e.g., _standard_gamma)
    3. MPS with fallback is ~20x slower than pure CPU for IRT training
    
    Therefore, CPU is the best choice for Mac users until py-irt adds MPS support.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class TrainingConfig:
    """Configuration matching the notebook parameters exactly."""
    # Core parameters from notebook
    dims_search: list[int] = field(default_factory=lambda: [2, 5])  # Match efficbench [2, 5]
    device: str = field(default_factory=get_best_device)  # auto-detect best device
    epochs: int = 2000  # Reduced for testing
    lr: float = .1  # Reduced learning rate for stability
    random_state: int = 42  # notebook default
    lr_decay = 0.9999
    # Validation parameters (from notebook Cell 11)
    val_stride: int = 5  # val_ind = list(range(0,Y_bin_train.shape[0],5))

    # Dimension validation is always enabled to ensure proper lambda computation.
    # Without validation_errors, GP-IRT falls back to default lambda=0.5 which affects results.
    validate_dimensions: bool = True
    
    # Lambda calculation parameters (from notebook Cells 17-18)
    number_item_per_scenario: int = 100  # number_item = 100 from notebook
    
    # Additional py-irt parameters (if needed)
    model_type: str = "multidim_2pl"
    # priors: str = "hierarchical"
    deterministic: bool = True
    log_every: int = 200
    
    # Zero-variance filtering
    filter_zero_variance: bool = False  # If True, remove zero-variance questions (uninformative for IRT)


def compute_balance_weights(matrix_df: pd.DataFrame) -> np.ndarray:
    """Compute balance weights for datasets with multiple subscenarios.
    
    This follows the TinyBenchmarks methodology exactly:
    For datasets with subscenarios (like MMLU with 57 subjects), apply the formula:
    weight = N / (n_sub * n_i) where:
    - N = total questions in the scenario
    - n_sub = number of subscenarios 
    - n_i = number of questions in subscenario i
    
    This gives higher weight to items from smaller subscenarios.
    
    The function checks for:
    1. "original_dataset" column (preferred - used by load_pickle_mmlu)
    2. "subscenario" column
    3. Dot notation in "dataset" column (e.g., "legalbench.abercrombie")
    """
    # Get all unique questions and initialize weights
    all_questions = sorted(matrix_df["question_id"].unique())
    balance_weights = np.ones(len(all_questions))
    question_to_idx = {q: i for i, q in enumerate(all_questions)}
    
    # Determine subscenario column
    subscenario_col = None
    if "original_dataset" in matrix_df.columns:
        # Check if original_dataset has different values than dataset
        if matrix_df["original_dataset"].nunique() > matrix_df["dataset"].nunique():
            subscenario_col = "original_dataset"
    if subscenario_col is None and "subscenario" in matrix_df.columns:
        subscenario_col = "subscenario"
    
    if subscenario_col is not None:
        # Use explicit subscenario column
        print(f"   ⚖️  Using '{subscenario_col}' for subscenario balance weights")
        
        # Group by parent scenario (dataset column)
        for scenario_name in matrix_df["dataset"].unique():
            scenario_df = matrix_df[matrix_df["dataset"] == scenario_name]
            subscenarios = scenario_df[subscenario_col].unique()
            
            if len(subscenarios) > 1:
                # Multi-subscenario scenario
                N = scenario_df["question_id"].nunique()  # Total questions
                n_sub = len(subscenarios)  # Number of subscenarios
                
                print(f"      {scenario_name}: {N} questions, {n_sub} subscenarios")
                
                for subscenario in subscenarios:
                    sub_df = scenario_df[scenario_df[subscenario_col] == subscenario]
                    sub_questions = sub_df["question_id"].unique()
                    n_i = len(sub_questions)  # Questions in this subscenario
                    
                    if n_i > 0:
                        # Apply formula: N/(n_sub*n_i)
                        weight = N / (n_sub * n_i)
                        
                        for q in sub_questions:
                            if q in question_to_idx:
                                balance_weights[question_to_idx[q]] = weight
    else:
        # Fallback: Look for dot notation in dataset names
        datasets = matrix_df["dataset"].unique()
        parent_datasets = {}
        
        for dataset in datasets:
            if "." in dataset:
                parent = dataset.split(".")[0]
                parent_datasets.setdefault(parent, []).append(dataset)
            else:
                parent_datasets.setdefault(dataset, [dataset])
        
        for parent_name, child_datasets in parent_datasets.items():
            if len(child_datasets) > 1:
                print(f"   ⚖️  Applying balance weights for {parent_name}: {len(child_datasets)} subscenarios")
                
                parent_df = matrix_df[matrix_df["dataset"].isin(child_datasets)]
                N = parent_df["question_id"].nunique()
                n_sub = len(child_datasets)
                
                for child_dataset in child_datasets:
                    child_df = matrix_df[matrix_df["dataset"] == child_dataset]
                    child_questions = child_df["question_id"].unique()
                    n_i = len(child_questions)
                    
                    if n_i > 0:
                        weight = N / (n_sub * n_i)
                        for q in child_questions:
                            if q in question_to_idx:
                                balance_weights[question_to_idx[q]] = weight
    
    # Print summary
    unique_weights = len(set(balance_weights))
    if unique_weights > 1:
        print(f"   ⚖️  Balance weights: min={balance_weights.min():.4f}, max={balance_weights.max():.4f}, unique={unique_weights}")
    
    return balance_weights


def binarize_responses(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Binarize responses using optimal thresholds per dataset.
    
    For AdaptEval data that's already in [0,1] range, we need to determine if it's
    already binary or needs thresholding. If scores are only 0.0 and 1.0, we keep as-is.
    Otherwise, we find optimal thresholds per dataset.
    """
    # Check if data is already binary
    unique_scores = sorted(matrix_df["normalized_score"].unique())
    is_already_binary = len(unique_scores) == 2 and set(unique_scores) == {0.0, 1.0}
    
    if is_already_binary:
        print("   ✓ Data is already binary (0.0, 1.0), no binarization needed")
        return matrix_df.copy()
    
    print(f"   🔄 Data has {len(unique_scores)} unique scores, applying thresholding...")

    # NOTE (memory):
    # The original notebook approach built a dense (models x questions) matrix and then
    # tried many thresholds by creating temporary boolean matrices repeatedly.
    # On large scenarios this can OOM (especially under multiprocessing).
    #
    # This implementation avoids dense matrices and avoids duplicating the full DataFrame
    # per dataset. It finds the threshold by working in the long-form table:
    # for each model, sort its scores once; binary mean for threshold c is computed via
    # searchsorted without allocating a giant (models x questions) boolean matrix.

    # Keep the original candidate grid to preserve behavior as closely as possible.
    # (We change only the implementation to avoid dense matrices / large temporary allocations.)
    n_thresholds = 1000

    # Build dataset iterator without materializing copies of each slice up-front
    if "dataset" not in matrix_df.columns:
        dataset_iter = [("all", matrix_df)]
        datasets = ["all"]
    else:
        datasets = sorted(matrix_df["dataset"].unique())
        dataset_iter = ((d, matrix_df[matrix_df["dataset"] == d]) for d in datasets)

    result_frames: list[pd.DataFrame] = []

    for dataset, dataset_df in tqdm(dataset_iter, total=len(datasets), desc="Finding optimal thresholds per dataset"):
        if dataset_df.empty:
            continue

        # Check if THIS dataset is already binary (skip threshold search if so)
        unique_scores_in_dataset = dataset_df["normalized_score"].unique()
        is_dataset_binary = (
            len(unique_scores_in_dataset) == 2 
            and set(unique_scores_in_dataset) == {0.0, 1.0}
        )
        
        if is_dataset_binary:
            # Already binary - no need to compute threshold
            result_frames.append(dataset_df.copy())
            print(f"     ✓ {dataset}: already binary (0.0, 1.0), skipped")
            continue

        # Stabilize duplicates: original code implicitly overwrote duplicates when filling the dense matrix.
        # Using mean is deterministic and usually the intended behavior (also matches pivot_table defaults).
        dataset_df = (
            dataset_df.groupby(["model_name", "question_id"], as_index=False, sort=False)["normalized_score"]
            .mean()
        )

        # Denominator matches the original dense-matrix width: total unique questions in this dataset.
        # Missing entries are treated as 0 in the original implementation because (np.nan > c) is False.
        n_questions_total = int(dataset_df["question_id"].nunique())
        if n_questions_total <= 0:
            result_frames.append(dataset_df.copy())
            continue

        # Collect per-model score arrays (drop NaNs) and precompute continuous means (nanmean on row)
        grouped = dataset_df.groupby("model_name", sort=False)["normalized_score"]
        sorted_scores_per_model: list[np.ndarray] = []
        continuous_avg: list[float] = []

        for _, s in grouped:
            arr = s.to_numpy(dtype=np.float32, copy=False)
            if np.isnan(arr).any():
                arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue
            arr.sort()  # in-place sort
            sorted_scores_per_model.append(arr)
            continuous_avg.append(float(arr.mean()))

        if not sorted_scores_per_model:
            # Nothing to threshold; keep as-is
            result_frames.append(dataset_df.copy())
            continue

        cont = np.asarray(continuous_avg, dtype=np.float32)

        # Original candidate thresholds
        cs = np.linspace(0.01, 0.99, n_thresholds, dtype=np.float32)

        best_error = float("inf")
        best_threshold = float(cs[0])

        # Evaluate thresholds without allocating (models x questions) matrices
        for c in cs:
            bin_means = np.empty(len(sorted_scores_per_model), dtype=np.float32)
            for i, arr in enumerate(sorted_scores_per_model):
                # fraction > c, with missing treated as 0 => divide by total #questions in dataset
                idx = int(np.searchsorted(arr, c, side="right"))
                bin_means[i] = (arr.size - idx) / n_questions_total
            error = float(np.mean(np.abs(bin_means - cont)))
            if error < best_error:
                best_error = error
                best_threshold = float(c)

        print(f"     📊 {dataset}: threshold={best_threshold:.3f}, error={best_error:.4f} (candidates={cs.size})")

        # Apply threshold to the ORIGINAL slice (not the de-duplicated one), so downstream keeps full rows/metadata.
        # (This preserves earlier behavior which operated on the original long-form table.)
        dataset_orig = matrix_df if "dataset" not in matrix_df.columns else matrix_df[matrix_df["dataset"] == dataset]
        dataset_binary = dataset_orig.copy()
        dataset_binary["normalized_score"] = (dataset_binary["normalized_score"] > best_threshold).astype(float)
        result_frames.append(dataset_binary)

        # Help Python free temp arrays earlier (useful under SLURM memory limits)
        del dataset_binary, dataset_df, dataset_orig, sorted_scores_per_model, continuous_avg, cont, cs

    return pd.concat(result_frames, ignore_index=True) if result_frames else matrix_df.copy()


def get_lambda(b: float, v: float) -> float:
    """Compute lambda exactly as in notebook Cell 15.
    
    From notebook: lambda = (b^2)/(v+(b^2))
    """
    return (b**2) / (v + (b**2))


# Removed old functions - now using exact notebook implementations


def validate_irt_dimensions(
    binary_matrix_df: pd.DataFrame,
    original_matrix_df: pd.DataFrame,
    balance_weights: np.ndarray,
    config: TrainingConfig,
    output_dir: str | None = None
) -> tuple[int, dict[str, list[float]]]:
    """Validate IRT dimensions using cross-validation.
    
    Split models into train/validation, use half the questions as 'seen' for estimation,
    and evaluate on the other half ('unseen') to choose the best dimension.
    
    This follows the notebook validation logic but works with any dataset structure.
    
    Args:
        binary_matrix_df: Binary matrix DataFrame for training
        original_matrix_df: Original matrix DataFrame for validation  
        balance_weights: Balance weights for multi-subscenario datasets
        config: Training configuration
        output_dir: Optional directory to save IRT dataset files (if None, uses temporary directory)
    """
    Ds = config.dims_search
    
    # Split models for validation
    # Preserve insertion order like the notebook (no sorting)
    models = list(pd.unique(binary_matrix_df["model_name"]))
    val_models = models[::config.val_stride]  # Every 5th model starting at 0
    train_models = [m for m in models if m not in set(val_models)]
    
    train_df = binary_matrix_df[binary_matrix_df["model_name"].isin(train_models)]
    val_df = binary_matrix_df[binary_matrix_df["model_name"].isin(val_models)]
    original_val_df = original_matrix_df[original_matrix_df["model_name"].isin(val_models)]
    
    # Get all questions and split into seen/unseen
    # Preserve insertion order of questions as appeared in the data
    all_questions = list(pd.unique(binary_matrix_df["question_id"]))
    seen_questions = all_questions[::2]  # Every other question
    unseen_questions = all_questions[1::2]
    
    errors_by_dimension = []
    errors_by_dataset = {}

    # Helper to extract scenario root from dataset name
    def scenario_from_dataset(name: str) -> str:
        return name.split(".")[0] if isinstance(name, str) and "." in name else name
    
    failed_dimensions = {}  # Track which dimensions failed and why
    
    for D in tqdm(Ds, desc="Validating dimensions"):
        # Train IRT model on training data
        if output_dir:
            # Save in permanent directory if provided
            os.makedirs(output_dir, exist_ok=True)
            dataset_path = os.path.join(output_dir, f'irt_val_dataset_dim{D}.jsonlines')
            temp_dir = None
        else:
            # Use temporary directory as fallback
            temp_dir = tempfile.TemporaryDirectory()
            dataset_path = os.path.join(temp_dir.name, 'irt_val_dataset.jsonlines')
        
        try:
            
            # Convert training data to IRT format
            train_responses = _df_to_irt_matrix(train_df)
            _ = create_irt_dataset(train_responses, dataset_path)
            if output_dir:
                print(f"   📁 Saved validation dataset: {dataset_path}")
            
            # Train model using Python API - catch NaN/numerical errors
            try:
                trainer = train_irt_model_python_api(dataset_path, D, config.lr, config.epochs, config.device, deterministic=config.deterministic)
                A, B, Theta = load_irt_parameters_from_trainer(trainer)
            except (ValueError, RuntimeError) as e:
                # Handle NaN errors in Gamma distribution or other numerical issues
                error_msg = str(e)
                if "nan" in error_msg.lower() or "invalid values" in error_msg.lower():
                    print(f"\n   ⚠️ Dimension D={D} failed with numerical error: {error_msg[:100]}...")
                    failed_dimensions[D] = error_msg
                    errors_by_dimension.append(float('inf'))  # Mark as worst possible
                    continue
                else:
                    # Re-raise non-numerical errors
                    raise
            
            # Validate on each dataset separately
            dataset_errors = []
            
            if "dataset" in val_df.columns:
                # Group by scenario root (prefix before '.')
                scenarios = sorted({scenario_from_dataset(d) for d in val_df["dataset"].unique()})
                for scenario in tqdm(scenarios):
                    dataset_val_df = val_df[val_df["dataset"].map(lambda d: scenario_from_dataset(d) == scenario)]
                    dataset_orig_df = original_val_df[original_val_df["dataset"].map(lambda d: scenario_from_dataset(d) == scenario)]

                    if dataset_val_df.empty:
                        continue

                    model_errors = []
                    for model_name in dataset_val_df["model_name"].unique():
                        model_val_df = dataset_val_df[dataset_val_df["model_name"] == model_name]
                        model_orig_df = dataset_orig_df[dataset_orig_df["model_name"] == model_name]

                        # Get seen responses for theta estimation
                        seen_responses = _get_model_responses(model_val_df, seen_questions)
                        if len(seen_responses) == 0:
                            continue

                        # Estimate theta using seen questions with multi-D IRT estimator
                        present_questions = [q for q in seen_questions if q in seen_responses]
                        if not present_questions:
                            continue
                        seen_indices = [all_questions.index(q) for q in present_questions]
                        A_seen = A[:, :, seen_indices]
                        B_seen = B[:, :, seen_indices]
                        seen_vec = np.array([seen_responses[q] for q in present_questions], dtype=float)
                        theta = estimate_ability_parameters(seen_vec, A_seen, B_seen)

                        # Predict on unseen questions and compare to actual
                        unseen_actual = _get_model_responses(model_orig_df, unseen_questions)
                        if len(unseen_actual) == 0:
                            continue

                        unseen_pred = _predict_responses(theta, A, B, unseen_questions, all_questions)

                        # Apply balance weights if available
                        if balance_weights is not None and len(balance_weights) == len(all_questions):
                            weighted_pred = np.mean([
                                balance_weights[all_questions.index(q)] * unseen_pred.get(q, 0)
                                for q in unseen_actual.keys()
                            ])
                            weighted_actual = np.mean([
                                balance_weights[all_questions.index(q)] * unseen_actual[q]
                                for q in unseen_actual.keys()
                            ])
                        else:
                            weighted_pred = np.mean(list(unseen_pred.values()))
                            weighted_actual = np.mean(list(unseen_actual.values()))

                        model_errors.append(abs(weighted_pred - weighted_actual))

                    if model_errors:
                        dataset_error = np.mean(model_errors)
                        dataset_errors.append(dataset_error)

                        if scenario not in errors_by_dataset:
                            errors_by_dataset[scenario] = []
                        errors_by_dataset[scenario].append(dataset_error)
            
            else:
                # No dataset separation, treat as one dataset
                model_errors = []
                for model_name in val_df["model_name"].unique():
                    model_val_df = val_df[val_df["model_name"] == model_name]
                    model_orig_df = original_val_df[original_val_df["model_name"] == model_name]
                    
                    seen_responses = _get_model_responses(model_val_df, seen_questions)
                    if len(seen_responses) == 0:
                        continue
                    
                    present_questions = [q for q in seen_questions if q in seen_responses]
                    if not present_questions:
                        continue
                    seen_indices = [all_questions.index(q) for q in present_questions]
                    A_seen = A[:, :, seen_indices]
                    B_seen = B[:, :, seen_indices]
                    seen_vec = np.array([seen_responses[q] for q in present_questions], dtype=float)
                    theta = estimate_ability_parameters(seen_vec, A_seen, B_seen)
                    
                    unseen_actual = _get_model_responses(model_orig_df, unseen_questions)
                    if len(unseen_actual) == 0:
                        continue
                    
                    unseen_pred = _predict_responses(theta, A, B, unseen_questions, all_questions)
                    
                    pred_avg = np.mean(list(unseen_pred.values()))
                    actual_avg = np.mean(list(unseen_actual.values()))
                    model_errors.append(abs(pred_avg - actual_avg))
                
                if model_errors:
                    dataset_errors.append(np.mean(model_errors))
            
            # Overall error for this dimension
            if dataset_errors:
                errors_by_dimension.append(np.mean(dataset_errors))
            else:
                errors_by_dimension.append(float('inf'))
        
        finally:
            # Clean up temporary directory if used
            if temp_dir:
                temp_dir.cleanup()
    
    # Choose best dimension
    if not errors_by_dimension or all(e == float('inf') for e in errors_by_dimension):
        # All dimensions failed - raise error with details
        failed_dims_str = ", ".join([f"D={d}: {err[:50]}..." for d, err in failed_dimensions.items()])
        raise ValueError(f"All IRT dimensions failed validation. Failures: {failed_dims_str}")
    
    best_idx = np.argmin(errors_by_dimension)
    best_dimension = Ds[best_idx]
    
    # Warn if some dimensions failed
    if failed_dimensions:
        print(f"   ⚠️ Note: {len(failed_dimensions)} dimension(s) failed: {list(failed_dimensions.keys())}")
    
    return best_dimension, errors_by_dataset


def _df_to_irt_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to matrix format for IRT training."""
    models = sorted(df["model_name"].unique())
    questions = sorted(df["question_id"].unique())
    
    matrix = np.zeros((len(models), len(questions)))
    model_to_idx = {m: i for i, m in enumerate(models)}
    question_to_idx = {q: i for i, q in enumerate(questions)}
    
    # Vectorized filling - much faster than iterrows
    model_indices = df["model_name"].map(model_to_idx).values
    question_indices = df["question_id"].map(question_to_idx).values
    scores = df["normalized_score"].values
    matrix[model_indices, question_indices] = scores
    
    return matrix


def _get_model_responses(model_df: pd.DataFrame, question_subset: list) -> dict:
    """Get responses for a specific model and question subset."""
    # Vectorized filtering and conversion to dict - much faster
    subset_df = model_df[model_df["question_id"].isin(question_subset)]
    return dict(zip(subset_df["question_id"], subset_df["normalized_score"]))


def _estimate_theta_mle(responses: dict, A: np.ndarray, B: np.ndarray, all_questions: list) -> float:
    """Estimate theta using MLE from observed responses."""
    # Simple MLE estimation - this is a simplified version
    # In practice, you might want to use the exact estimate_ability_parameters function
    if not responses:
        return 0.0
    
    # Convert responses to arrays aligned with A and B
    response_array = []
    a_array = []
    b_array = []
    
    for q in responses.keys():
        if q in all_questions:
            q_idx = all_questions.index(q)
            if q_idx < A.shape[2]:  # Make sure we have parameters for this question
                response_array.append(responses[q])
                a_array.append(np.linalg.norm(A[0, :, q_idx]))  # Collapse multi-dim to scalar
                b_array.append(np.mean(B[0, :, q_idx]))  # Collapse multi-dim to scalar
    
    if not response_array:
        return 0.0
    
    # Simple Newton-Raphson for theta estimation
    theta = 0.0
    for _ in range(50):
        z = np.array(a_array) * theta - np.array(b_array)
        p = sigmoid(z)
        
        grad = np.sum(np.array(a_array) * (np.array(response_array) - p))
        hess = -np.sum((np.array(a_array) ** 2) * p * (1 - p)) - 1e-6
        
        if abs(hess) < 1e-10:
            break
            
        step = grad / hess
        theta_new = theta - step
        
        if abs(theta_new - theta) < 1e-4:
            break
        theta = theta_new
    
    return theta


def _predict_responses(theta: np.ndarray | float, A: np.ndarray, B: np.ndarray, questions: list, all_questions: list) -> dict:
    """Predict responses for given questions using estimated theta (supports multi-D)."""
    predictions = {}

    # Normalize theta shape to (1, D, 1)
    if isinstance(theta, np.ndarray):
        if theta.ndim == 3:
            theta_arr = theta
        elif theta.ndim == 1:
            theta_arr = theta[None, :, None]
        elif theta.ndim == 2 and theta.shape[0] == 1:
            theta_arr = theta[:, :, None]
        else:
            # Fallback to scalar interpretation
            theta_arr = np.array(theta, dtype=float).reshape(1, 1, 1)
    else:
        theta_arr = np.array([[theta]], dtype=float)[:, :, None]

    for q in questions:
        if q in all_questions:
            q_idx = all_questions.index(q)
            if q_idx < A.shape[2]:
                # Use item_curve to compute probability with multi-D parameters
                P = item_curve(theta_arr, A[:, :, q_idx:q_idx+1], B[:, :, q_idx:q_idx+1])
                predictions[q] = float(np.squeeze(P))

    return predictions


def compute_lambda_values(
    original_matrix_df: pd.DataFrame,
    validation_errors: dict[str, list[float]],
    best_dim_idx: int,
    number_item: int = 100
) -> dict[str, float]:
    """Compute lambda values for blending anchor and IRT predictions.
    
    Lambda = (b²)/(v + b²) where:
    - b = validation error for the dataset
    - v = variance of scores per dataset, scaled by number_item
    
    This follows the notebook logic but works with any dataset structure.
    """
    lambdas = {}

    # If validation errors are missing (e.g., validation was skipped), return empty dict.
    # Downstream blending will fall back to a default (typically 0.5) when lambda is missing.
    if not validation_errors:
        print("   ⚠️  No validation_errors found; skipping lambda computation (will use defaults downstream).")
        return lambdas
    
    if "dataset" not in original_matrix_df.columns:
        # No dataset separation, compute single lambda exactly like notebook:
        # v = mean over models of var over items (questions)
        variance = _compute_dataset_variance(original_matrix_df)
        if "all" not in validation_errors or len(validation_errors["all"]) <= best_dim_idx:
            raise ValueError("Missing validation error for 'all' dataset; cannot compute lambda without fallback.")
        error = validation_errors["all"][best_dim_idx]

        v_scaled = variance / (4 * number_item)
        lambda_val = get_lambda(error, v_scaled)
        lambdas["all"] = lambda_val

        return lambdas
    
    # Helper to scenario root
    def scenario_from_dataset(name: str) -> str:
        return name.split(".")[0] if isinstance(name, str) and "." in name else name

    # Compute lambda for each scenario (group of subdatasets)
    # Only process scenarios that have validation errors
    available_scenarios = set(validation_errors.keys())
    all_scenarios = sorted({scenario_from_dataset(d) for d in original_matrix_df["dataset"].unique()})
    
    # Filter to scenarios that have validation errors
    scenarios_to_process = [s for s in all_scenarios if s in available_scenarios]
    
    if not scenarios_to_process:
        raise ValueError(f"No scenarios with validation errors found. Available: {available_scenarios}, Required: {all_scenarios}")

    for scenario in scenarios_to_process:
        scenario_df = original_matrix_df[original_matrix_df["dataset"].map(lambda d: scenario_from_dataset(d) == scenario)]

        # Compute variance for this scenario: mean over models of var across items
        variance = _compute_dataset_variance(scenario_df)

        # Get validation error for this scenario (no fallback to match notebook)
        if scenario not in validation_errors or len(validation_errors[scenario]) <= best_dim_idx:
            raise ValueError(f"Missing validation error for scenario '{scenario}'; cannot compute lambda without fallback.")
        error = validation_errors[scenario][best_dim_idx]

        # Apply notebook scaling and compute lambda
        v_scaled = variance / (4 * number_item)
        lambda_val = get_lambda(error, v_scaled)
        lambdas[scenario] = lambda_val
    
    return lambdas


def _compute_dataset_variance(dataset_df: pd.DataFrame) -> float:
    """Compute variance exactly like the notebook: per-model variance across items, then mean."""
    # Preserve insertion order for models and questions
    models = list(pd.unique(dataset_df["model_name"]))
    questions = list(pd.unique(dataset_df["question_id"]))

    if len(models) == 0 or len(questions) == 0:
        return 0.0

    matrix = np.full((len(models), len(questions)), np.nan)
    model_to_idx = {m: i for i, m in enumerate(models)}
    question_to_idx = {q: i for i, q in enumerate(questions)}

    # Vectorized filling
    model_indices = dataset_df["model_name"].map(model_to_idx).values
    question_indices = dataset_df["question_id"].map(question_to_idx).values
    scores = dataset_df["normalized_score"].values
    matrix[model_indices, question_indices] = scores

    # Per-model variance across items, then mean across models
    model_variances = []
    for m_idx in range(len(models)):
        model_scores = matrix[m_idx, :]
        valid_scores = model_scores[~np.isnan(model_scores)]
        if len(valid_scores) > 1:
            model_variances.append(np.var(valid_scores, ddof=0))

    return float(np.mean(model_variances)) if model_variances else 0.0


# Validation functions removed - data is already processed by normalization pipeline


def fit_2pl_parameters(
    matrix_df: pd.DataFrame,
    config: TrainingConfig | None = None,
    output_dir: str | None = None,
    anchor_items: list[dict] | None = None,
) -> pd.DataFrame:
    """Fit 2PL parameters following the TinyBenchmarks methodology.

    This is a generalized version that works with any dataset structure while
    following the key algorithmic steps from the notebook:
    
    1. Compute balance weights for multi-subscenario datasets
    2. Binarize responses with optimal thresholds per dataset
    3. Validate dimensions using cross-validation 
    4. Train final IRT model with best dimension
    5. Compute lambda values for anchor-IRT blending

    Args:
        matrix_df: DataFrame with columns [model_name, question_id, normalized_score, dataset?, subscenario?]
        config: Training configuration
        output_dir: Optional directory to save IRT dataset files (if None, uses temporary directory)

    Returns:
        DataFrame indexed by question_id with columns ["a", "b"] and attached metadata.
    """
    cfg = config or TrainingConfig()
    
    # Check for cached results first (if output_dir provided and no anchor_items)
    # Skip cache if anchor_items provided since that changes the training
    if output_dir and not anchor_items:
        import json
        params_path = os.path.join(output_dir, "item_params.parquet")
        meta_path = os.path.join(output_dir, "item_params.meta.json")
        
        if os.path.exists(params_path):
            try:
                params = pd.read_parquet(params_path)
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        params.attrs = json.load(f)
                print(f"   ✓ Loaded cached IRT params from {params_path}")
                return params
            except Exception as e:
                print(f"   ⚠️ Failed to load cache ({e}), retraining...")
    
    print("Starting IRT training following TinyBenchmarks methodology...")
    
    # Step 1: Compute balance weights for multi-subscenario datasets
    print("Step 1: Computing balance weights...")
    balance_weights = compute_balance_weights(matrix_df)
    print(f"Balance weights computed for {len(balance_weights)} questions")
    
    # Step 2: Binarize responses with optimal thresholds per dataset
    print("Step 2: Binarizing responses...")
    binary_matrix_df = binarize_responses(matrix_df)
    print("Responses binarized with optimal thresholds per dataset")
    
    # Step 2b: Filter out zero-variance questions (uninformative for IRT)
    # These questions have identical responses from all models and provide no discriminative information
    if cfg.filter_zero_variance:
        print("Step 2b: Filtering zero-variance questions...")
        variance_per_question = binary_matrix_df.groupby("question_id")["normalized_score"].var()
        zero_var_questions = set(variance_per_question[variance_per_question == 0].index)
        if zero_var_questions:
            original_count = binary_matrix_df["question_id"].nunique()
            binary_matrix_df = binary_matrix_df[~binary_matrix_df["question_id"].isin(zero_var_questions)].copy()
            # Also filter matrix_df for consistency in validation steps (use copy to avoid modifying input)
            matrix_df = matrix_df[~matrix_df["question_id"].isin(zero_var_questions)].copy()
            filtered_count = binary_matrix_df["question_id"].nunique()
            print(f"   ⚠️  Removed {len(zero_var_questions)} zero-variance questions ({original_count} → {filtered_count})")
            # Recompute balance weights for filtered data
            balance_weights = compute_balance_weights(matrix_df)
            print(f"   ✓ Recomputed balance weights for {len(balance_weights)} questions")
        else:
            print("   ✓ No zero-variance questions found")
    else:
        print("Step 2b: Skipping zero-variance filtering (disabled)")
    
    # Step 3: Validate dimensions using cross-validation (optional)
    if cfg.validate_dimensions:
        print("Step 3: Validating dimensions...")
        best_dimension, validation_errors = validate_irt_dimensions(
            binary_matrix_df, matrix_df, balance_weights, cfg, output_dir
        )
        best_dim_idx = cfg.dims_search.index(best_dimension) if best_dimension in cfg.dims_search else 0
        print(f"Best dimension: {best_dimension}")
    else:
        best_dimension = cfg.dims_search[0] if cfg.dims_search else 2
        validation_errors = {}
        best_dim_idx = 0
        print(f"Step 3: Skipping dimension validation; using fixed dimension D={best_dimension}")
    
    # Step 4: Train final IRT model
    print("Step 4: Training final IRT model...")
    if output_dir:
        # Save in permanent directory if provided
        os.makedirs(output_dir, exist_ok=True)
        dataset_path = os.path.join(output_dir, 'irt_dataset_final.jsonlines')
        temp_dir = None
    else:
        # Use temporary directory as fallback
        temp_dir = tempfile.TemporaryDirectory()
        dataset_path = os.path.join(temp_dir.name, 'irt_dataset.jsonlines')
    
    # Dimensions to try (best first, then fallback to lower)
    dims_to_try = [best_dimension]
    # Add fallback dimensions in order of preference
    for fallback_dim in sorted(cfg.dims_search):
        if fallback_dim not in dims_to_try and fallback_dim < best_dimension:
            dims_to_try.append(fallback_dim)
    
    trainer = None
    final_dimension = best_dimension
    last_error = None
    
    try:
        
        # Convert to IRT format and train
        train_matrix = _df_to_irt_matrix(binary_matrix_df)
        question_ids = sorted(binary_matrix_df["question_id"].unique())
        question_id_mapping = create_irt_dataset(train_matrix, dataset_path, question_ids=question_ids)
        if output_dir:
            print(f"   📁 Saved final training dataset: {dataset_path}")
        
        # Try dimensions with fallback on NaN errors
        for attempt_dim in dims_to_try:
            try:
                if attempt_dim != best_dimension:
                    print(f"   ⚠️ Falling back to dimension D={attempt_dim}...")
                
                trainer = train_irt_model_python_api(
                    dataset_path,
                    attempt_dim,
                    cfg.lr,
                    cfg.epochs,
                    cfg.device,
                    anchor_items=anchor_items,
                    question_id_mapping=question_id_mapping,
                    lr_decay=cfg.lr_decay,
                    deterministic=cfg.deterministic,
                )
                final_dimension = attempt_dim
                break  # Success, exit loop
                
            except (ValueError, RuntimeError) as e:
                error_msg = str(e)
                if "nan" in error_msg.lower() or "invalid values" in error_msg.lower():
                    print(f"   ⚠️ Dimension D={attempt_dim} failed: {error_msg[:80]}...")
                    last_error = e
                    continue  # Try next dimension
                else:
                    raise  # Re-raise non-numerical errors
        
        if trainer is None:
            # All dimensions failed
            raise ValueError(f"Final IRT training failed for all dimensions. Last error: {last_error}")
        
        # Load trained parameters directly from trainer
        A, B, Theta = load_irt_parameters_from_trainer(trainer)
    
    finally:
        # Clean up temporary directory if used
        if temp_dir:
            temp_dir.cleanup()
    
    # Report if we used a different dimension than originally selected
    if final_dimension != best_dimension:
        print(f"   ℹ️ Used fallback dimension D={final_dimension} (original: D={best_dimension})")
    print("IRT model training completed")
    
    # Convert parameters to DataFrame format
    question_ids = sorted(matrix_df["question_id"].unique())
    
    # Store full matrices for proper MIRT computation (like TinyBenchmarks original)
    # A shape: (1, D, num_items), B shape: (1, D, num_items)
    A_full = A
    B_full = B
    
    # Compute scalar summaries for backward compatibility and anchor selection
    # IMPORTANT: Use norm for 'a' and mean for 'b' to match py-irt anchor scaling
    # py-irt divides anchor discrimination by sqrt(D) so norm recovers original value
    if len(A.shape) == 3:  # (1, D, num_items)
        a_values = np.linalg.norm(A[0], axis=0)  # L2 norm of discrimination vector
        b_values = np.mean(B[0], axis=0)  # Mean of difficulty across dims
    else:  # Already scalar
        a_values = A.flatten()
        b_values = B.flatten()
    
    # Ensure we have parameters for all questions
    min_len = min(len(question_ids), len(a_values), len(b_values))
    params = pd.DataFrame({
        "a": a_values[:min_len], 
        "b": b_values[:min_len]
    }, index=question_ids[:min_len])
    params.index.name = "question_id"
    
    # Add dataset information from matrix (needed for per-dataset anchor selection)
    if "dataset" in matrix_df.columns:
        question_to_dataset = matrix_df.groupby("question_id")["dataset"].first()
        params["dataset"] = params.index.map(question_to_dataset)
    
    # Step 5: Compute lambda values for anchor-IRT blending
    print("Step 5: Computing lambda values...")
    lambdas = compute_lambda_values(
        matrix_df, validation_errors, best_dim_idx, cfg.number_item_per_scenario
    )
    print(f"Lambda values computed for {len(lambdas)} datasets: {lambdas}")
    
    # Attach metadata for downstream use (ensure JSON serializable)
    def make_json_serializable(obj):
        """Convert numpy arrays and other non-serializable objects to JSON-safe formats."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj
    
    params.attrs = {
        "lambdas_by_dataset": make_json_serializable(lambdas),
        "balance_weights": make_json_serializable(balance_weights),
        "best_dimension": int(final_dimension),  # Actual dimension used (may differ from selected if fallback)
        "selected_dimension": int(best_dimension),  # Originally selected dimension
        "validation_errors": make_json_serializable(validation_errors),
        "config_epochs": cfg.epochs,
        "config_lr": cfg.lr,
        "config_device": cfg.device,
        "config_dims_search": cfg.dims_search,
        # Store full MIRT matrices (like TinyBenchmarks original)
        "A_matrix": make_json_serializable(A_full),  # (1, D, num_items)
        "B_matrix": make_json_serializable(B_full),  # (1, D, num_items)
    }
    
    # Auto-save item params for caching (if output_dir was provided)
    if output_dir:
        import json
        params_path = os.path.join(output_dir, "item_params.parquet")
        meta_path = os.path.join(output_dir, "item_params.meta.json")
        
        params.to_parquet(params_path)
        with open(meta_path, 'w') as f:
            json.dump(params.attrs, f, indent=2)
        print(f"   📁 Saved item params: {params_path}")
    
    print(f"IRT training completed successfully. Parameters for {len(params)} questions.")
    return params


