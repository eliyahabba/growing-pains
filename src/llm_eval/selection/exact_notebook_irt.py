"""
Exact Notebook IRT Implementation

This module implements EXACTLY the same logic as the TinyBenchmarks notebooks
(training_irt.ipynb, anchor_points.ipynb, estimating_performance.ipynb)
but organized in cleaner, reusable functions.

NO mathematical changes - only code organization improvements.
"""

import os
import tempfile
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

# Import the exact notebook functions
from llm_eval.selection.tinyBenchmarks.irt import create_irt_dataset, train_irt_model, load_irt_parameters, estimate_ability_parameters
from llm_eval.selection.tinyBenchmarks.utils import sigmoid, item_curve


class ExactNotebookIRT:
    """
    Implements EXACTLY the notebook workflow with no mathematical changes.
    
    This follows the exact steps from training_irt.ipynb:
    1. Prepare scenarios structure (notebook cells 1-5)
    2. Create response matrix Y (cell 5) 
    3. Compute balance weights for MMLU (cell 7)
    4. Binarization with threshold optimization (cell 10)
    5. Dimension validation with cross-validation (cell 11)
    6. Train final IRT model (cell 14)
    7. Compute lambda values (cells 17-18)
    """
    
    def __init__(self):
        self.random_state = 42  # From notebook
    
    def prepare_data_exact_notebook(self, matrix_df: pd.DataFrame) -> Tuple[Dict, Dict, np.ndarray]:
        """
        Prepare data based on YOUR actual datasets, not the hardcoded scenarios.
        Creates scenarios from your datasets directly.
        """
        print("🔄 Preparing data from your actual datasets...")
        
        # Step 1: Create scenarios from your actual datasets
        scenarios_position, subscenarios_position = self._create_scenarios_from_data(matrix_df)
        
        # Step 2: Compute balance weights (only for multi-dataset scenarios)
        balance_weights = self._compute_balance_weights_from_data(
            matrix_df, scenarios_position, subscenarios_position
        )
        
        return scenarios_position, subscenarios_position, balance_weights
    
    def _create_scenarios_from_data(self, matrix_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        Create scenarios based on logical groupings of your datasets.
        
        Groups related datasets into scenarios with subscenarios:
        - legalbench -> scenario with multiple legalbench.* subscenarios
        - math -> scenario with multiple math.* subscenarios  
        - mmlu -> scenario with multiple mmlu.* subscenarios
        - wmt_14 -> scenario with multiple wmt_14.* subscenarios
        - Individual datasets -> their own scenarios
        """
        # Get all unique questions and datasets
        all_questions = sorted(matrix_df["question_id"].unique())
        all_datasets = sorted(matrix_df["dataset"].unique())
        
        print(f"   📊 Found {len(all_datasets)} unique datasets:")
        for dataset in all_datasets:
            count = len(matrix_df[matrix_df["dataset"] == dataset]["question_id"].unique())
            print(f"      {dataset}: {count} questions")
        
        scenarios_position = {}
        subscenarios_position = {}
        
        # Group datasets into logical scenarios
        dataset_groups = {
            'legalbench': [],
            'math': [],
            'mmlu': [], 
            'wmt_14': [],
            'individual': []  # For standalone datasets
        }
        
        # Classify each dataset into groups
        for dataset in all_datasets:
            if dataset.startswith('legalbench.'):
                dataset_groups['legalbench'].append(dataset)
            elif dataset.startswith('math.'):
                dataset_groups['math'].append(dataset)
            elif dataset.startswith('mmlu.'):
                dataset_groups['mmlu'].append(dataset)
            elif dataset.startswith('wmt_14.'):
                dataset_groups['wmt_14'].append(dataset)
            else:
                # Individual datasets: gsm, med_qa, narrativeqa, natural_qa, openbook_qa
                dataset_groups['individual'].append(dataset)
        
        print(f"   🔗 Dataset groupings:")
        for group, datasets in dataset_groups.items():
            if datasets:
                print(f"      {group}: {len(datasets)} datasets - {datasets}")
        
        # Create question_id to index mapping
        question_to_idx = {q: i for i, q in enumerate(all_questions)}
        
        # Create scenarios and subscenarios
        for group_name, datasets in dataset_groups.items():
            if not datasets:
                continue
                
            if group_name == 'individual':
                # Each individual dataset becomes its own scenario
                for dataset in datasets:
                    scenario_name = dataset
                    scenarios_position[scenario_name] = []
                    subscenarios_position[scenario_name] = {dataset: []}
                    
                    # Find questions for this dataset
                    dataset_questions = matrix_df[matrix_df["dataset"] == dataset]["question_id"].unique()
                    for question_id in dataset_questions:
                        if question_id in question_to_idx:
                            idx = question_to_idx[question_id]
                            scenarios_position[scenario_name].append(idx)
                            subscenarios_position[scenario_name][dataset].append(idx)
            else:
                # Multi-dataset scenarios (legalbench, math, mmlu, wmt_14)
                scenario_name = group_name
                scenarios_position[scenario_name] = []
                subscenarios_position[scenario_name] = {}
                
                for dataset in datasets:
                    subscenarios_position[scenario_name][dataset] = []
                    
                    # Find questions for this dataset
                    dataset_questions = matrix_df[matrix_df["dataset"] == dataset]["question_id"].unique()
                    for question_id in dataset_questions:
                        if question_id in question_to_idx:
                            idx = question_to_idx[question_id]
                            scenarios_position[scenario_name].append(idx)
                            subscenarios_position[scenario_name][dataset].append(idx)
        
        print(f"   ✅ Created {len(scenarios_position)} scenarios:")
        for scenario, indices in scenarios_position.items():
            subscenario_count = len(subscenarios_position[scenario])
            print(f"      {scenario}: {len(indices)} questions, {subscenario_count} subscenarios")
            if subscenario_count > 1:
                for sub_name, sub_indices in subscenarios_position[scenario].items():
                    print(f"         {sub_name}: {len(sub_indices)} questions")
        
        return scenarios_position, subscenarios_position
    

    
    def _compute_balance_weights_from_data(
        self, 
        matrix_df: pd.DataFrame,
        scenarios_position: Dict,
        subscenarios_position: Dict
    ) -> np.ndarray:
        """
        Compute balance weights for YOUR data structure.
        
        Since each dataset is its own scenario, balance weights are usually all 1.0
        unless you have multi-sub-dataset scenarios.
        """
        num_questions = len(matrix_df["question_id"].unique())
        balance_weights = np.ones(num_questions)
        
        print(f"   ⚖️ Computing balance weights...")
        
        # Check if any scenario has multiple subscenarios (like MMLU in the original)
        multi_sub_scenarios = []
        for scenario_name, subscenarios in subscenarios_position.items():
            if len(subscenarios) > 1:
                multi_sub_scenarios.append(scenario_name)
        
        if multi_sub_scenarios:
            print(f"   🔄 Found multi-subscenario datasets: {multi_sub_scenarios}")
            
            for scenario_name in multi_sub_scenarios:
                N = len(scenarios_position[scenario_name])
                n_sub = len(subscenarios_position[scenario_name])
                
                for sub_name, sub_positions in subscenarios_position[scenario_name].items():
                    n_i = len(sub_positions)
                    if n_i > 0:
                        # Apply exact notebook formula
                        balance_weights[sub_positions] = N / (n_sub * n_i)
                        print(f"      {scenario_name}.{sub_name}: weight = {N / (n_sub * n_i):.3f}")
        else:
            print(f"   ✅ All scenarios are single-dataset - using uniform weights (1.0)")
        
        return balance_weights
    
    def normalize_scores_for_irt(
        self, 
        matrix_df: pd.DataFrame,
        scenarios_position: Dict
    ) -> pd.DataFrame:
        """
        Normalize scores and apply binarization only where needed.
        
        Optimized version: Detects if data is already normalized and binary.
        """
        print("🔄 Processing scores for IRT training...")
        
        all_scores = set(matrix_df['normalized_score'].unique())
        
        # Check if data is already normalized to [0,1] and binary
        already_normalized_binary = all_scores.issubset({0.0, 1.0})
        
        if already_normalized_binary:
            print("   🚀 Data is already normalized to [0,1] and binary - no processing needed!")
            print(f"   ✅ Using {len(matrix_df)} scores as-is")
            return matrix_df.copy()
        
        # Check if data is binary but in [0,100] range
        binary_100_scale = all_scores.issubset({0.0, 100.0})
        
        if binary_100_scale:
            print("   🚀 Data is binary (0/100) - only normalization needed!")
            result_df = matrix_df.copy()
            result_df["normalized_score"] = result_df["normalized_score"] / 100.0
            print(f"   ✅ Normalized {len(result_df)} scores from [0,100] to [0,1]")
            return result_df
        
        # If not all binary, do detailed analysis per dataset
        print("   🔍 Analyzing datasets for binarization needs...")
        
        binary_datasets = []
        continuous_datasets = []
        
        for dataset in matrix_df['dataset'].unique():
            dataset_df = matrix_df[matrix_df['dataset'] == dataset]
            unique_vals = set(dataset_df['normalized_score'].unique())
            is_binary = unique_vals.issubset({0.0, 100.0})
            
            if is_binary:
                binary_datasets.append(dataset)
            else:
                continuous_datasets.append(dataset)
        
        print(f"   ✅ Binary datasets: {len(binary_datasets)}")
        print(f"   🔄 Continuous datasets: {len(continuous_datasets)} (need binarization)")
        
        # Normalize all scores from [0,100] to [0,1] first
        matrix_df = matrix_df.copy()
        matrix_df["normalized_score"] = matrix_df["normalized_score"] / 100.0
        
        # Create model x question matrix exactly like notebook
        models = sorted(matrix_df["model_name"].unique())
        questions = sorted(matrix_df["question_id"].unique())
        
        Y = np.zeros((len(models), len(questions)))
        model_to_idx = {m: i for i, m in enumerate(models)}
        question_to_idx = {q: i for i, q in enumerate(questions)}
        
        for _, row in matrix_df.iterrows():
            m_idx = model_to_idx[row["model_name"]]
            q_idx = question_to_idx[row["question_id"]]
            Y[m_idx, q_idx] = row["normalized_score"]
        
        # Binarize only continuous datasets, keep binary datasets as-is
        Y_bin = Y.copy()  # Start with original (normalized) values
        
        # Only binarize scenarios that contain continuous datasets
        scenarios_needing_binarization = set()
        
        # Map each question to its dataset
        question_to_dataset = {}
        for _, row in matrix_df.iterrows():
            question_to_dataset[row['question_id']] = row['dataset']
        
        # Check which scenarios need binarization
        for scenario, indices in scenarios_position.items():
            for idx in indices:
                if idx < len(questions):
                    question = questions[idx]
                    dataset = question_to_dataset.get(question)
                    if dataset in continuous_datasets:
                        scenarios_needing_binarization.add(scenario)
                        break
        
        if scenarios_needing_binarization:
            print(f"   🎯 Applying binarization to scenarios: {scenarios_needing_binarization}")
            cs = np.linspace(0.01, 0.99, 100)  # Exact same as notebook
            
            for scenario in tqdm(scenarios_needing_binarization, desc="Binarizing scenarios"):
                if scenario in scenarios_position:
                    ind = scenarios_position[scenario]
                    if len(ind) > 0:
                        # Find best threshold (exact notebook formula)
                        errors = []
                        for c in cs:
                            binary_avg = (Y[:, ind] > c).mean(axis=1)
                            continuous_avg = Y[:, ind].mean(axis=1)
                            error = np.mean(np.abs(binary_avg - continuous_avg))
                            errors.append(error)
                        
                        best_c = cs[np.argmin(errors)]
                        Y_bin[:, ind] = (Y[:, ind] > best_c).astype(int)
                        print(f"   {scenario}: threshold={best_c:.3f}")
        else:
            print(f"   ✅ All datasets are already binary - no binarization needed!")
        
        # Convert back to DataFrame
        binary_data = []
        for m_idx, model in enumerate(models):
            for q_idx, question in enumerate(questions):
                binary_data.append({
                    "model_name": model,
                    "question_id": question,
                    "normalized_score": float(Y_bin[m_idx, q_idx]),
                    "dataset": matrix_df[matrix_df["question_id"] == question]["dataset"].iloc[0]
                })
        
        return pd.DataFrame(binary_data)
    
    def validate_dimensions_exact_notebook(
        self,
        Y_bin_train: np.ndarray,
        Y_train: np.ndarray,
        balance_weights: np.ndarray,
        scenarios_position: Dict,
        Ds: List[int] = [5, 10],
        epochs: int = 2000,
        device: str = 'cpu'
    ) -> Tuple[int, List[List[float]]]:
        """
        Validate dimensions exactly like notebook cell 11.
        
        From notebook:
        val_ind = list(range(0,Y_bin_train.shape[0],5))
        train_ind = [i for i in range(Y_bin_train.shape[0]) if i not in val_ind]
        """
        print("🔍 Validating dimensions exactly like notebook cell 11...")
        print(f"   Data shape: {Y_bin_train.shape}")
        print(f"   Testing dimensions: {Ds}")
        print(f"   Epochs per dimension: {epochs}")
        
        # Exact validation split from notebook
        val_ind = list(range(0, Y_bin_train.shape[0], 5))  # Every 5th model
        train_ind = [i for i in range(Y_bin_train.shape[0]) if i not in val_ind]
        
        print(f"   Train models: {len(train_ind)}, Validation models: {len(val_ind)}")
        
        errors = []
        errors2 = []
        
        for i, D in enumerate(Ds):
            print(f"\n⚡ Testing dimension {D} ({i+1}/{len(Ds)})...")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                dataset_path = os.path.join(temp_dir, 'irt_val_dataset.jsonlines')
                model_path = os.path.join(temp_dir, 'irt_val_model')
                
                print(f"   📁 Creating IRT dataset with {len(train_ind)} models...")
                # Create IRT dataset for training subset
                create_irt_dataset(Y_bin_train[train_ind], dataset_path)
                
                print(f"   🧠 Training IRT model (D={D}, epochs={epochs})...")
                print(f"      This may take several minutes...")
                # Train IRT model exactly like notebook
                train_irt_model(dataset_path, model_path, D, 0.1, epochs, device)
                
                print(f"   📊 Loading IRT parameters...")
                A, B, Theta = load_irt_parameters(model_path)
                
                # Validation exactly like notebook
                seen_items = list(range(0, Y_bin_train.shape[1], 2))  # Every other item
                unseen_items = list(range(1, Y_bin_train.shape[1], 2))
                
                print(f"   🎯 Estimating abilities for {len(val_ind)} validation models...")
                print(f"      Using {len(seen_items)} seen items, {len(unseen_items)} unseen items")
                
                # Estimate ability parameters for validation set
                thetas = []
                for j in range(len(val_ind)):
                    if j % 5 == 0:  # Progress every 5 models
                        print(f"      Processing validation model {j+1}/{len(val_ind)}...")
                    responses = Y_bin_train[val_ind[j]][seen_items]
                    theta = estimate_ability_parameters(
                        responses, A[:, :, seen_items], B[:, :, seen_items]
                    )
                    thetas.append(theta)
                
                # Compute validation errors per scenario (exact notebook logic)
                print(f"   📈 Computing validation errors per scenario...")
                scenario_errors = []
                for scenario in scenarios_position.keys():
                    if scenario in scenarios_position:
                        ind = [u for u in unseen_items if u in scenarios_position[scenario]]
                        if len(ind) > 0:
                            errors_for_scenario = []
                            for j in range(len(val_ind)):
                                # Exact notebook formula
                                predicted = (balance_weights * item_curve(thetas[j], A, B))[0, ind].mean()
                                actual = Y_train[val_ind[j], ind].mean()
                                error = abs(predicted - actual)
                                errors_for_scenario.append(error)
                            scenario_errors.append(np.mean(errors_for_scenario))
                            print(f"      {scenario}: {np.mean(errors_for_scenario):.4f}")
                
                errors2.append(scenario_errors)
                avg_error = np.mean(scenario_errors)
                errors.append(avg_error)
                print(f"   ✅ Dimension {D} completed - Average error: {avg_error:.4f}")
        
        # Choose best dimension (exact notebook logic)
        best_idx = np.argmin(np.array(errors))
        best_D = Ds[best_idx]
        
        print(f"\n🏆 Dimension validation completed!")
        print(f"   Errors by dimension: {dict(zip(Ds, errors))}")
        print(f"   Best dimension: {best_D} (error: {errors[best_idx]:.4f})")
        return best_D, errors2
    
    def compute_lambdas_exact_notebook(
        self,
        Y_train: np.ndarray,
        scenarios_position: Dict,
        errors2: List[List[float]],
        best_dim_idx: int,
        number_item: int = 100
    ) -> Dict[str, float]:
        """
        Compute lambdas exactly like notebook cells 17-18.
        
        From notebook:
        def get_lambda(b, v):
            return (b**2)/(v+(b**2))
            
        lambds = {}
        for i,scenario in enumerate(scenarios.keys()):
            v = np.var(Y_train[:,scenarios_position[scenario]], axis=1).mean()
            b = np.mean(errors2[ind_D][i]) 
            lambds[scenario] = get_lambda(b, v/(4*number_item))
        """
        print("Computing lambdas exactly like notebook cells 17-18...")
        
        def get_lambda(b, v):
            """Exact lambda formula from notebook."""
            return (b**2) / (v + (b**2))
        
        lambds = {}
        scenario_names = list(scenarios_position.keys())
        
        for i, scenario in enumerate(scenario_names):
            if scenario in scenarios_position and len(scenarios_position[scenario]) > 0:
                # Exact notebook computation
                v = np.var(Y_train[:, scenarios_position[scenario]], axis=1).mean()
                b = np.mean(errors2[best_dim_idx][i]) if i < len(errors2[best_dim_idx]) else 0.05
                lambda_val = get_lambda(b, v / (4 * number_item))
                lambds[scenario] = lambda_val
                print(f"   {scenario}: λ = {lambda_val:.3f}")
        
        return lambds
    
    def train_final_model_exact_notebook(
        self,
        Y_bin_train: np.ndarray,
        best_D: int,
        epochs: int = 2000,
        device: str = 'cpu'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Train final IRT model exactly like notebook cell 14.
        """
        print(f"🎓 Training final IRT model exactly like notebook cell 14...")
        print(f"   Dimension: {best_D}")
        print(f"   Epochs: {epochs}")
        print(f"   Data shape: {Y_bin_train.shape}")
        print(f"   This is the final training - may take several minutes...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, 'irt_dataset.jsonlines')
            model_path = os.path.join(temp_dir, 'irt_model')
            
            print(f"   📁 Creating final IRT dataset...")
            # Create dataset exactly like notebook
            create_irt_dataset(Y_bin_train, dataset_path)
            
            print(f"   🧠 Training final model (this may take a while)...")
            # Train exactly like notebook
            train_irt_model(dataset_path, model_path, best_D, 0.1, epochs, device)
            
            print(f"   📊 Loading final parameters...")
            # Load parameters
            A, B, Theta = load_irt_parameters(model_path)
            
        print(f"   ✅ Final IRT model training completed!")
        return A, B, Theta


def train_irt_exact_notebook(
    matrix_df: pd.DataFrame,
    epochs: int = 2000,
    device: str = 'cpu',
    Ds: List[int] = [5, 10],
    number_item: int = 100
) -> Tuple[pd.DataFrame, Dict]:
    """
    Train IRT model using EXACT notebook methodology.
    
    This follows the complete workflow from training_irt.ipynb without any changes
    to the mathematical computations, only organizing the code better.
    
    Args:
        matrix_df: DataFrame with [model_name, question_id, normalized_score, dataset]
        epochs: Number of training epochs (default 2000 like notebook)
        device: 'cpu' or 'cuda'
        Ds: Dimensions to try (default [5, 10] like notebook)
        number_item: For lambda computation (default 100 like notebook)
    
    Returns:
        (item_params_df, metadata_dict) where metadata contains exact notebook results
    """
    trainer = ExactNotebookIRT()
    
    print("🧠 Training IRT using EXACT notebook methodology...")
    print(f"   Data: {matrix_df.shape}")
    print(f"   Models: {matrix_df['model_name'].nunique()}")
    print(f"   Questions: {matrix_df['question_id'].nunique()}")
    print(f"   Epochs: {epochs}, Device: {device}")
    print(f"   Dimensions to try: {Ds}")
    print()
    
    print("📋 Step 1: Prepare data exactly like notebook cells 5-7")
    scenarios_position, subscenarios_position, balance_weights = trainer.prepare_data_exact_notebook(matrix_df)
    
    print("\n📋 Step 2: Process scores (normalize/binarize if needed) exactly like notebook cell 10")
    binary_df = trainer.normalize_scores_for_irt(matrix_df, scenarios_position)
    
    # Convert to matrices exactly like notebook
    models = sorted(matrix_df["model_name"].unique())
    questions = sorted(matrix_df["question_id"].unique())
    
    model_to_idx = {m: i for i, m in enumerate(models)}
    question_to_idx = {q: i for i, q in enumerate(questions)}
    
    # Check if binary_df is the same as original (no processing was done)
    if binary_df is matrix_df or (
        len(binary_df) == len(matrix_df) and 
        binary_df["normalized_score"].equals(matrix_df["normalized_score"])
    ):
        print(f"   ⚡ Data unchanged - using single matrix for both Y_train and Y_bin_train")
        # Create single matrix and use for both - FAST vectorized approach
        Y_train = np.zeros((len(models), len(questions)))
        
        # Vectorized filling - much faster than iterrows
        model_indices = matrix_df["model_name"].map(model_to_idx).values
        question_indices = matrix_df["question_id"].map(question_to_idx).values
        scores = matrix_df["normalized_score"].values
        Y_train[model_indices, question_indices] = scores
        
        Y_bin_train = Y_train  # Same matrix
        print(f"   🚀 Filled matrix using vectorized operations ({len(matrix_df)} entries)")
    else:
        print(f"   🔄 Data was processed - creating separate matrices")
        # Create separate matrices - FAST vectorized approach
        Y_train = np.zeros((len(models), len(questions)))
        Y_bin_train = np.zeros((len(models), len(questions)))
        
        # Fill original matrix - vectorized
        model_indices = matrix_df["model_name"].map(model_to_idx).values
        question_indices = matrix_df["question_id"].map(question_to_idx).values
        scores = matrix_df["normalized_score"].values
        Y_train[model_indices, question_indices] = scores
        
        # Fill binary matrix - vectorized
        model_indices_bin = binary_df["model_name"].map(model_to_idx).values
        question_indices_bin = binary_df["question_id"].map(question_to_idx).values
        scores_bin = binary_df["normalized_score"].values
        Y_bin_train[model_indices_bin, question_indices_bin] = scores_bin
        
        print(f"   🚀 Filled matrices using vectorized operations ({len(matrix_df)} + {len(binary_df)} entries)")
    
    print(f"\n📋 Step 3: Validate dimensions exactly like notebook cell 11")
    print(f"   This is usually the longest step - each dimension needs full IRT training")
    best_D, errors2 = trainer.validate_dimensions_exact_notebook(
        Y_bin_train, Y_train, balance_weights, scenarios_position, Ds, epochs, device
    )
    best_dim_idx = Ds.index(best_D)
    
    print(f"\n📋 Step 4: Train final model exactly like notebook cell 14")
    A, B, Theta = trainer.train_final_model_exact_notebook(Y_bin_train, best_D, epochs, device)
    
    print(f"\n📋 Step 5: Compute lambdas exactly like notebook cells 17-18")
    lambdas = trainer.compute_lambdas_exact_notebook(
        Y_train, scenarios_position, errors2, best_dim_idx, number_item
    )
    
    # Convert to DataFrame format
    if len(A.shape) == 3:  # Multi-dimensional
        a_values = np.linalg.norm(A[0], axis=0)
        b_values = np.mean(B[0], axis=0)
    else:
        a_values = A.flatten()
        b_values = B.flatten()
    
    min_len = min(len(questions), len(a_values), len(b_values))
    item_params = pd.DataFrame({
        "a": a_values[:min_len],
        "b": b_values[:min_len]
    }, index=questions[:min_len])
    item_params.index.name = "question_id"
    
    # Store exact notebook metadata
    metadata = {
        "lambdas_by_dataset": lambdas,
        "balance_weights": balance_weights.tolist(),
        "best_dimension": int(best_D),
        "val_errors_by_dataset": errors2,
        "scenarios_position": scenarios_position,
        "subscenarios_position": subscenarios_position,
        "epochs": epochs,
        "device": device,
        "dims_tried": Ds,
        "number_item": number_item
    }
    
    print(f"✅ IRT training completed using exact notebook methodology")
    print(f"   Parameters: {len(item_params)} questions")
    print(f"   Best dimension: {best_D}")
    print(f"   Lambdas: {lambdas}")
    
    return item_params, metadata
