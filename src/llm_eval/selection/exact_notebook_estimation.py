"""
Exact Notebook Estimation Implementation

This implements EXACTLY the estimation logic from estimating_performance.ipynb
with no mathematical changes, only code organization improvements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from llm_eval.selection.tinyBenchmarks.irt import load_irt_parameters, estimate_ability_parameters
from llm_eval.selection.tinyBenchmarks.utils import item_curve


class ExactNotebookEstimation:
    """
    Implements EXACTLY the estimation methodology from estimating_performance.ipynb.
    
    The notebook shows three prediction methods:
    1. Anchor-only: Y_hat = (Y_anchor * anchor_weights[scenario]).sum(axis=1)
    2. p-IRT: pirt_lambd * data_part + (1-pirt_lambd) * irt_part
    3. gp-IRT: lambds[scenario] * preds[scenario] + (1-lambds[scenario]) * pirt_preds[scenario]
    """
    
    def __init__(self):
        pass  # No hardcoded scenarios
    
    def predict_anchor_only_exact_notebook(
        self,
        Y_test: np.ndarray,
        scenarios_position: Dict,
        balance_weights: np.ndarray,
        anchor_points: Dict[str, np.ndarray],
        anchor_weights: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Predict using anchor-only method exactly like notebook.
        
        From notebook cell 11:
        preds = {}
        for scenario in scenarios.keys():
            Y_anchor = Y_test[:,scenarios_position[scenario]][:,anchor_points[scenario]]
            preds[scenario] = (Y_anchor*anchor_weights[scenario]).sum(axis=1)
        """
        preds = {}
        
        for scenario in scenarios_position.keys():
            if scenario in scenarios_position and scenario in anchor_points:
                # Get anchor responses exactly like notebook
                scenario_indices = scenarios_position[scenario]
                anchor_indices = anchor_points[scenario]
                
                Y_anchor = Y_test[:, scenario_indices][:, anchor_indices]
                
                # Predict exactly like notebook
                preds[scenario] = (Y_anchor * anchor_weights[scenario]).sum(axis=1)
        
        return preds
    
    def predict_pirt_exact_notebook(
        self,
        Y_test: np.ndarray,
        scenarios_position: Dict,
        balance_weights: np.ndarray,
        anchor_points: Dict[str, np.ndarray],
        thetas: List[np.ndarray],
        A: np.ndarray,
        B: np.ndarray,
        seen_items: List[int],
        unseen_items: List[int]
    ) -> Dict[str, np.ndarray]:
        """
        Predict using p-IRT method exactly like notebook.
        
        From notebook cell 14:
        pirt_preds = {}
        for scenario in scenarios.keys():
            ind_seen = [u for u in seen_items if u in scenarios_position[scenario]]
            ind_unseen = [u for u in unseen_items if u in scenarios_position[scenario]]
            pirt_lambd = Y_anchor.shape[1]/len(scenarios_position[scenario])
            
            pirt_pred = []
            for j in range(Y_test.shape[0]):
                data_part = (balance_weights*Y_test)[j,ind_seen].mean()
                irt_part = (balance_weights*item_curve(thetas[j], A, B))[0,ind_unseen].mean()
                pirt_pred.append(pirt_lambd*data_part + (1-pirt_lambd)*irt_part)
        """
        pirt_preds = {}
        
        for scenario in scenarios_position.keys():
            if scenario in scenarios_position and scenario in anchor_points:
                scenario_indices = scenarios_position[scenario]
                
                # Find seen and unseen items for this scenario exactly like notebook
                ind_seen = [u for u in seen_items if u in scenario_indices]
                ind_unseen = [u for u in unseen_items if u in scenario_indices]
                
                if len(ind_seen) == 0 or len(ind_unseen) == 0:
                    continue
                
                # Compute pirt_lambda exactly like notebook
                Y_anchor = Y_test[:, scenario_indices][:, anchor_points[scenario]]
                pirt_lambd = Y_anchor.shape[1] / len(scenario_indices)
                
                # Predict for each test model exactly like notebook
                pirt_pred = []
                for j in range(Y_test.shape[0]):
                    # Data part (from seen items)
                    data_part = (balance_weights * Y_test)[j, ind_seen].mean()
                    
                    # IRT part (from unseen items)
                    irt_part = (balance_weights * item_curve(thetas[j], A, B))[0, ind_unseen].mean()
                    
                    # Combine exactly like notebook
                    prediction = pirt_lambd * data_part + (1 - pirt_lambd) * irt_part
                    pirt_pred.append(prediction)
                
                pirt_preds[scenario] = np.array(pirt_pred)
        
        return pirt_preds
    
    def predict_gpirt_exact_notebook(
        self,
        preds: Dict[str, np.ndarray],
        pirt_preds: Dict[str, np.ndarray],
        lambds: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """
        Predict using gp-IRT method exactly like notebook.
        
        From notebook cell 16:
        gpirt_preds = {}
        for scenario in scenarios.keys():
            gpirt_preds[scenario] = lambds[scenario]*preds[scenario] + (1-lambds[scenario])*pirt_preds[scenario]
        """
        gpirt_preds = {}
        
        for scenario in preds.keys():
            if scenario in preds and scenario in pirt_preds and scenario in lambds:
                # Exact notebook formula
                gpirt_preds[scenario] = (
                    lambds[scenario] * preds[scenario] + 
                    (1 - lambds[scenario]) * pirt_preds[scenario]
                )
        
        return gpirt_preds
    
    def estimate_thetas_exact_notebook(
        self,
        Y_test: np.ndarray,
        seen_items: List[int],
        A: np.ndarray,
        B: np.ndarray
    ) -> List[np.ndarray]:
        """
        Estimate thetas exactly like notebook.
        
        From notebook cell 13:
        thetas = [estimate_ability_parameters(Y_test[j][seen_items], A[:, :, seen_items], B[:, :, seen_items]) for j in tqdm(range(Y_test.shape[0]))]
        """
        thetas = []
        
        for j in range(Y_test.shape[0]):
            responses = Y_test[j][seen_items]
            theta = estimate_ability_parameters(
                responses, 
                A[:, :, seen_items], 
                B[:, :, seen_items]
            )
            thetas.append(theta)
        
        return thetas
    
    def compute_validation_errors_exact_notebook(
        self,
        Y_test: np.ndarray,
        scenarios_position: Dict,
        balance_weights: np.ndarray,
        preds: Dict[str, np.ndarray],
        pirt_preds: Dict[str, np.ndarray],
        gpirt_preds: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute validation errors exactly like notebook.
        
        From notebook:
        true = (balance_weights*Y_test)[:,scenarios_position[scenario]].mean(axis=1)
        print(f"scenario: {scenario}, avg. error: {np.abs(Y_hat-true).mean():.3f}")
        """
        errors = {}
        
        for scenario in scenarios_position.keys():
            if scenario in scenarios_position:
                # True performance exactly like notebook
                true = (balance_weights * Y_test)[:, scenarios_position[scenario]].mean(axis=1)
                
                scenario_errors = {}
                
                # Anchor-only error
                if scenario in preds:
                    anchor_error = np.abs(preds[scenario] - true).mean()
                    scenario_errors['anchor'] = float(anchor_error)
                
                # p-IRT error
                if scenario in pirt_preds:
                    pirt_error = np.abs(pirt_preds[scenario] - true).mean()
                    scenario_errors['pirt'] = float(pirt_error)
                
                # gp-IRT error
                if scenario in gpirt_preds:
                    gpirt_error = np.abs(gpirt_preds[scenario] - true).mean()
                    scenario_errors['gpirt'] = float(gpirt_error)
                
                errors[scenario] = scenario_errors
        
        return errors


def run_estimation_exact_notebook(
    Y_test: np.ndarray,
    scenarios_position: Dict,
    subscenarios_position: Dict,
    balance_weights: np.ndarray,
    anchor_points: Dict[str, np.ndarray],
    anchor_weights: Dict[str, np.ndarray],
    lambds: Dict[str, float],
    irt_model_path: str
) -> Dict:
    """
    Run complete estimation exactly like estimating_performance.ipynb.
    
    This follows the exact workflow from the notebook:
    1. Load IRT parameters
    2. Define seen/unseen items
    3. Estimate thetas for test models
    4. Compute anchor-only predictions
    5. Compute p-IRT predictions  
    6. Compute gp-IRT predictions
    7. Evaluate all methods
    """
    estimator = ExactNotebookEstimation()
    
    print("🔮 Running estimation exactly like estimating_performance.ipynb...")
    
    # Step 1: Load IRT parameters exactly like notebook
    A, B, _ = load_irt_parameters(irt_model_path)
    
    # Step 2: Define seen/unseen items exactly like notebook
    seen_items = np.hstack([
        np.array(scenarios_position[scenario])[anchor_points[scenario]] 
        for scenario in scenarios_position.keys() 
        if scenario in scenarios_position and scenario in anchor_points
    ]).tolist()
    unseen_items = [i for i in range(Y_test.shape[1]) if i not in seen_items]
    
    print(f"   Seen items: {len(seen_items)}, Unseen items: {len(unseen_items)}")
    
    # Step 3: Estimate thetas exactly like notebook cell 13
    print("   Estimating ability parameters...")
    thetas = estimator.estimate_thetas_exact_notebook(Y_test, seen_items, A, B)
    
    # Step 4: Anchor-only predictions exactly like notebook cell 11
    print("   Computing anchor-only predictions...")
    preds = estimator.predict_anchor_only_exact_notebook(
        Y_test, scenarios_position, balance_weights, anchor_points, anchor_weights
    )
    
    # Step 5: p-IRT predictions exactly like notebook cell 14
    print("   Computing p-IRT predictions...")
    pirt_preds = estimator.predict_pirt_exact_notebook(
        Y_test, scenarios_position, balance_weights, anchor_points, 
        thetas, A, B, seen_items, unseen_items
    )
    
    # Step 6: gp-IRT predictions exactly like notebook cell 16
    print("   Computing gp-IRT predictions...")
    gpirt_preds = estimator.predict_gpirt_exact_notebook(preds, pirt_preds, lambds)
    
    # Step 7: Compute errors exactly like notebook
    print("   Computing validation errors...")
    errors = estimator.compute_validation_errors_exact_notebook(
        Y_test, scenarios_position, balance_weights, preds, pirt_preds, gpirt_preds
    )
    
    # Print results exactly like notebook
    print("\n📊 Results by scenario:")
    for scenario in scenarios_position.keys():
        if scenario in errors:
            scenario_errors = errors[scenario]
            if 'anchor' in scenario_errors:
                print(f"   {scenario} - Anchor: {scenario_errors['anchor']:.3f}")
            if 'pirt' in scenario_errors:
                print(f"   {scenario} - p-IRT: {scenario_errors['pirt']:.3f}")
            if 'gpirt' in scenario_errors:
                print(f"   {scenario} - gp-IRT: {scenario_errors['gpirt']:.3f}")
    
    results = {
        'anchor_predictions': preds,
        'pirt_predictions': pirt_preds,
        'gpirt_predictions': gpirt_preds,
        'errors_by_scenario': errors,
        'thetas': thetas,
        'seen_items': seen_items,
        'unseen_items': unseen_items
    }
    
    print("✅ Estimation completed using exact notebook methodology")
    
    return results
