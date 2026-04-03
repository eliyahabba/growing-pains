"""
Exact Notebook Anchor Selection Implementation

This implements EXACTLY the anchor selection logic from anchor_points.ipynb
with no mathematical changes, only code organization improvements.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from typing import Dict, Tuple, List

from llm_eval.selection.tinyBenchmarks.irt import load_irt_parameters


class ExactNotebookAnchors:
    """
    Implements EXACTLY the anchor selection from anchor_points.ipynb.
    
    From notebook cell 13:
    clustering = 'irt' # 'correct.' or 'irt'
    number_item = 100
    
    The notebook supports two clustering methods:
    1. 'correct.' - uses correctness patterns: X = Y_train[:,scenarios_position[scenario]].T
    2. 'irt' - uses IRT parameters: X = np.vstack((A.squeeze(), B.squeeze().reshape((1,-1)))).T
    """
    
    def __init__(self):
        self.random_state = 42  # From notebook
    
    def select_anchors_exact_notebook(
        self,
        scenarios_position: Dict,
        subscenarios_position: Dict,
        balance_weights: np.ndarray,
        clustering: str = 'irt',
        number_item: int = 100,
        Y_train: np.ndarray = None,
        irt_model_path: str = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Select anchors exactly like notebook cell 13.
        
        From notebook:
        anchor_points = {}
        anchor_weights = {}
        
        for scenario in scenarios.keys():
            if clustering=='correct.':
                X = Y_train[:,scenarios_position[scenario]].T
            elif clustering=='irt':
                A, B, _ = load_irt_parameters('data/irt_model/')
                X = np.vstack((A.squeeze(), B.squeeze().reshape((1,-1)))).T
                X = X[scenarios_position[scenario]]
                
            # Normalizing balance_weights
            norm_balance_weights = balance_weights[scenarios_position[scenario]]
            norm_balance_weights /= norm_balance_weights.sum()
            
            # Fitting the KMeans model
            kmeans = KMeans(n_clusters=number_item, n_init="auto", random_state=random_state)
            kmeans.fit(X, sample_weight=norm_balance_weights)
            
            # Calculating anchor points
            anchor_points[scenario] = pairwise_distances(kmeans.cluster_centers_, X, metric='euclidean').argmin(axis=1)
            
            # Calculating anchor weights
            anchor_weights[scenario] = np.array([np.sum(norm_balance_weights[kmeans.labels_==c]) for c in range(number_item)])
        """
        print(f"Selecting anchors exactly like notebook with clustering='{clustering}'...")
        
        anchor_points = {}
        anchor_weights = {}
        
        # Load IRT parameters if using IRT clustering
        if clustering == 'irt':
            if irt_model_path is None:
                raise ValueError("irt_model_path required for IRT clustering")
            A, B, _ = load_irt_parameters(irt_model_path)
        
        for scenario in scenarios_position.keys():
            if scenario not in scenarios_position or len(scenarios_position[scenario]) == 0:
                continue
                
            print(f"   Processing scenario: {scenario}")
            
            # Prepare features exactly like notebook
            if clustering == 'correct.':
                if Y_train is None:
                    raise ValueError("Y_train required for correctness clustering")
                X = Y_train[:, scenarios_position[scenario]].T
            elif clustering == 'irt':
                # Exact notebook formula
                X = np.vstack((A.squeeze(), B.squeeze().reshape((1, -1)))).T
                X = X[scenarios_position[scenario]]
            else:
                raise ValueError(f"Unknown clustering method: {clustering}")
            
            # Normalizing balance_weights exactly like notebook
            norm_balance_weights = balance_weights[scenarios_position[scenario]]
            norm_balance_weights /= norm_balance_weights.sum()
            
            # Fitting KMeans exactly like notebook
            kmeans = KMeans(
                n_clusters=number_item, 
                n_init="auto", 
                random_state=self.random_state
            )
            kmeans.fit(X, sample_weight=norm_balance_weights)
            
            # Calculating anchor points exactly like notebook
            distances = pairwise_distances(kmeans.cluster_centers_, X, metric='euclidean')
            anchor_points[scenario] = distances.argmin(axis=1)
            
            # Calculating anchor weights exactly like notebook
            anchor_weights[scenario] = np.array([
                np.sum(norm_balance_weights[kmeans.labels_ == c]) 
                for c in range(number_item)
            ])
            
            print(f"     Selected {len(anchor_points[scenario])} anchors")
        
        return anchor_points, anchor_weights
    
    def convert_anchor_indices_to_question_ids(
        self,
        anchor_points: Dict[str, np.ndarray],
        scenarios_position: Dict,
        question_ids: List[str]
    ) -> Dict[str, List[str]]:
        """
        Convert anchor indices to actual question IDs.
        """
        anchor_question_ids = {}
        
        for scenario in anchor_points.keys():
            if scenario in scenarios_position:
                scenario_questions = [question_ids[i] for i in scenarios_position[scenario]]
                anchor_indices = anchor_points[scenario]
                anchor_question_ids[scenario] = [
                    scenario_questions[idx] for idx in anchor_indices 
                    if idx < len(scenario_questions)
                ]
        
        return anchor_question_ids


def select_anchors_exact_notebook(
    matrix_df: pd.DataFrame,
    item_params: pd.DataFrame,
    metadata: Dict,
    clustering: str = 'irt',
    number_item: int = 100,
    irt_model_path: str = None
) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
    """
    Select anchors using EXACT notebook methodology.
    
    Args:
        matrix_df: Original matrix DataFrame
        item_params: Trained IRT parameters
        metadata: Metadata from IRT training (contains scenarios_position, balance_weights)
        clustering: 'irt' or 'correct.' (from notebook)
        number_item: Number of anchors per scenario (default 100 from notebook)
        irt_model_path: Path to IRT model (required for IRT clustering)
    
    Returns:
        (anchor_question_ids_by_scenario, anchor_weights_by_scenario)
    """
    selector = ExactNotebookAnchors()
    
    # Extract data from metadata
    scenarios_position = metadata['scenarios_position']
    subscenarios_position = metadata['subscenarios_position']
    balance_weights = np.array(metadata['balance_weights'])
    
    # Prepare Y_train matrix if needed for correctness clustering
    Y_train = None
    if clustering == 'correct.':
        models = sorted(matrix_df["model_name"].unique())
        questions = sorted(matrix_df["question_id"].unique())
        
        Y_train = np.zeros((len(models), len(questions)))
        model_to_idx = {m: i for i, m in enumerate(models)}
        question_to_idx = {q: i for i, q in enumerate(questions)}
        
        for _, row in matrix_df.iterrows():
            m_idx = model_to_idx[row["model_name"]]
            q_idx = question_to_idx[row["question_id"]]
            Y_train[m_idx, q_idx] = row["normalized_score"]
    
    # Select anchors using exact notebook logic
    anchor_points, anchor_weights = selector.select_anchors_exact_notebook(
        scenarios_position=scenarios_position,
        subscenarios_position=subscenarios_position,
        balance_weights=balance_weights,
        clustering=clustering,
        number_item=number_item,
        Y_train=Y_train,
        irt_model_path=irt_model_path
    )
    
    # Convert indices to question IDs
    questions = sorted(matrix_df["question_id"].unique())
    anchor_question_ids = selector.convert_anchor_indices_to_question_ids(
        anchor_points, scenarios_position, questions
    )
    
    print(f"✅ Anchor selection completed using exact notebook methodology")
    total_anchors = sum(len(anchors) for anchors in anchor_question_ids.values())
    print(f"   Total anchors selected: {total_anchors}")
    
    return anchor_question_ids, anchor_weights
