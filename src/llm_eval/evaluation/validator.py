
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from llm_eval.selection import QuestionSelector, ModelProfile
from .metrics import compute_evaluation_metrics, compute_model_performance, EvaluationMetrics


@dataclass
class SelectionValidationResult:
    """Results of validating a question selection method."""
    model_name: str
    dataset_name: str
    selector_name: str
    k: int
    selected_questions: List[str]
    metrics: EvaluationMetrics
    full_score: float
    selected_score: float


@dataclass 
class ValidationSummary:
    """Summary of validation results across models and datasets."""
    results: List[SelectionValidationResult]
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Compute average metrics across all validations."""
        if not self.results:
            return {}
        
        metrics_list = [r.metrics for r in self.results if not pd.isna(r.metrics.rmse)]
        if not metrics_list:
            return {}
        
        # Helper function to safely compute averages
        def safe_avg(values):
            valid_values = [v for v in values if not pd.isna(v)]
            return sum(valid_values) / len(valid_values) if valid_values else float('nan')
        
        return {
            "avg_rmse": sum(m.rmse for m in metrics_list) / len(metrics_list),
            "avg_mae": sum(m.mae for m in metrics_list) / len(metrics_list), 
            "avg_correlation": safe_avg([m.correlation for m in metrics_list]),
            "avg_bias": sum(m.bias for m in metrics_list) / len(metrics_list),
            "avg_relative_error": safe_avg([m.relative_error for m in metrics_list]),
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        data = []
        for result in self.results:
            data.append({
                "model_name": result.model_name,
                "dataset_name": result.dataset_name,
                "selector_name": result.selector_name,
                "k": result.k,
                "n_selected": len(result.selected_questions),
                "full_score": result.full_score,
                "selected_score": result.selected_score,
                "rmse": result.metrics.rmse,
                "mae": result.metrics.mae,
                "correlation": result.metrics.correlation,
                "bias": result.metrics.bias,
                "relative_error": result.metrics.relative_error,
                "n_samples": result.metrics.n_samples,
            })
        return pd.DataFrame(data)


class SelectionValidator:
    """Validates the quality of question selection methods."""
    
    def __init__(self, matrix_df: pd.DataFrame):
        self.matrix_df = matrix_df
        
    def validate_selection(
        self,
        selector: QuestionSelector,
        selector_name: str,
        model_name: str,
        k: int,
        dataset_name: Optional[str] = None
    ) -> SelectionValidationResult:
        """Validate a single selection for a model on a dataset."""
        
        # Create model profile
        profile = ModelProfile(model_name=model_name)
        
        # Filter matrix for specific dataset if provided
        matrix_subset = self.matrix_df.copy()
        if dataset_name:
            matrix_subset = matrix_subset[matrix_subset["dataset"] == dataset_name]
        
        if len(matrix_subset) == 0:
            # Return empty result if no data
            return SelectionValidationResult(
                model_name=model_name,
                dataset_name=dataset_name or "all",
                selector_name=selector_name,
                k=k,
                selected_questions=[],
                metrics=EvaluationMetrics(0, 0, 0, 0, 0, 0),
                full_score=0.0,
                selected_score=0.0
            )
        
        # Get selected questions
        selected_questions = selector.select(profile, k, matrix_subset)
        
        # Compute full performance (all questions in dataset)
        full_performance = compute_model_performance(
            matrix_subset, model_name, dataset_name
        )
        
        # Compute selected performance (only selected questions)
        selected_performance = compute_model_performance(
            matrix_subset, model_name, dataset_name, selected_questions
        )
        
        # Compare performances
        metrics = compute_evaluation_metrics(selected_performance, full_performance)
        
        # Get scalar scores for reporting
        full_score = full_performance.mean() if len(full_performance) > 0 else 0.0
        selected_score = selected_performance.mean() if len(selected_performance) > 0 else 0.0
        
        return SelectionValidationResult(
            model_name=model_name,
            dataset_name=dataset_name or "all",
            selector_name=selector_name,
            k=k,
            selected_questions=selected_questions,
            metrics=metrics,
            full_score=full_score,
            selected_score=selected_score
        )
    
    def validate_all_models(
        self,
        selectors: Dict[str, QuestionSelector],
        k: int,
        dataset_names: Optional[List[str]] = None
    ) -> ValidationSummary:
        """Validate all selectors on all models and datasets."""
        
        # Get unique models and datasets
        models = self.matrix_df["model_name"].unique()
        
        if dataset_names is None:
            datasets = self.matrix_df["dataset"].unique()
        else:
            datasets = dataset_names
        
        results = []
        
        for model_name in models:
            for dataset_name in datasets:
                for selector_name, selector in selectors.items():
                    try:
                        result = self.validate_selection(
                            selector, selector_name, model_name, k, dataset_name
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"Error validating {selector_name} for {model_name} on {dataset_name}: {e}")
                        continue
        
        return ValidationSummary(results=results)
    
    def compare_selectors(
        self,
        selectors: Dict[str, QuestionSelector],
        k_values: List[int],
        model_name: str,
        dataset_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Compare different selectors for a specific model across different k values."""
        
        results = []
        
        for k in k_values:
            for selector_name, selector in selectors.items():
                result = self.validate_selection(
                    selector, selector_name, model_name, k, dataset_name
                )
                results.append(result)
        
        summary = ValidationSummary(results=results)
        return summary.to_dataframe()

