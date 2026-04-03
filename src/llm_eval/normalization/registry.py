

from dataclasses import dataclass
from typing import Mapping, Any, Callable, Dict, List
import pandas as pd
from tqdm import tqdm

from llm_eval.config import AppConfig
from llm_eval.normalization.base import AbstractNormalizer, NormalizationResult
from llm_eval.normalization.rules import minmax_normalize, zscore_cdf_normalize, binary_normalize, find_optimal_threshold, irt_binary_normalize


@dataclass
class MetricRegistry:
    config: AppConfig

    def normalize(self, metric_name: str, raw_score: float, metadata: Mapping[str, Any]) -> NormalizationResult:
        metric_cfg = self.config.metrics.get(metric_name)
        if metric_cfg is None:
            raise KeyError(f"Unknown metric '{metric_name}'. Configure it in metrics.yaml")

        method = metric_cfg.method

        if method == "binary":
            value = binary_normalize(raw_score, metric_cfg.higher_is_better)
            return NormalizationResult(
                normalized=value,
                method="binary",
                params={"higher_is_better": metric_cfg.higher_is_better},
            )

        if method == "minmax":
            if metric_cfg.min is None or metric_cfg.max is None:
                raise ValueError(f"Metric '{metric_name}' requires min and max for minmax normalization")
            value = minmax_normalize(raw_score, float(metric_cfg.min), float(metric_cfg.max), metric_cfg.higher_is_better)  # noqa: E501
            return NormalizationResult(
                normalized=value,
                method="minmax",
                params={"min": metric_cfg.min, "max": metric_cfg.max, "higher_is_better": metric_cfg.higher_is_better},  # noqa: E501
            )

        if method == "zscore":
            mean = float(metadata.get("mean"))
            std = float(metadata.get("std"))
            higher = bool(metric_cfg.higher_is_better)
            value = zscore_cdf_normalize(raw_score, mean, std, higher)
            return NormalizationResult(
                normalized=value,
                method="zscore_cdf",
                params={"mean": mean, "std": std, "higher_is_better": higher},
            )

        raise ValueError(f"Unsupported normalization method '{method}' for metric '{metric_name}'")

    def compute_irt_thresholds(self, matrix_df: pd.DataFrame, method: str = 'direct') -> Dict[str, float]:
        """
        Compute optimal IRT thresholds for each dataset/scenario.
        
        Args:
            matrix_df: DataFrame with columns [model_name, question_id, dataset, raw_score]
            method: 'direct' or 'normalized' binarization method
            
        Returns:
            Dictionary mapping dataset names to optimal thresholds
        """
        thresholds = {}
        
        # Group by dataset to handle each scenario separately
        for dataset in tqdm(
                matrix_df['dataset'].unique()):
            dataset_data = matrix_df[matrix_df['dataset'] == dataset]
            
            # Create pivot table: models x questions
            try:
                pivot_df = dataset_data.pivot_table(
                    index='model_name',
                    columns='question_id',
                    values='raw_score',
                    aggfunc='mean'  # Handle duplicates
                )
                
                if not pivot_df.empty:
                    # Convert pandas DataFrame to list of lists
                    scores_matrix = pivot_df.values.tolist()
                    threshold = find_optimal_threshold(scores_matrix, method=method)
                    thresholds[dataset] = threshold
                    
            except Exception as e:
                # Fallback to simple threshold if pivot fails
                median_score = dataset_data['raw_score'].median()
                thresholds[dataset] = float(median_score)
        
        return thresholds

    def normalize_with_irt_thresholds(self, matrix_df: pd.DataFrame, method: str = 'direct') -> pd.DataFrame:
        """
        Apply IRT-style binarization to the entire matrix using optimal thresholds per dataset.
        
        Args:
            matrix_df: DataFrame with columns [model_name, question_id, dataset, raw_score, ...]
            method: 'direct' or 'normalized' binarization method
            
        Returns:
            DataFrame with added 'normalized_score' column using IRT binarization
        """
        result_df = matrix_df.copy()
        
        # Compute optimal thresholds for each dataset
        thresholds = self.compute_irt_thresholds(matrix_df, method=method)
        
        # Apply thresholds
        def apply_irt_threshold(row):
            dataset = row['dataset']
            raw_score = row['raw_score']
            threshold = thresholds.get(dataset, raw_score)  # Fallback to raw score if no threshold
            
            # For IRT, we assume higher is always better (correctness)
            return irt_binary_normalize(raw_score, threshold, higher_is_better=True)
        
        result_df['normalized_score'] = result_df.apply(apply_irt_threshold, axis=1)
        
        # Store threshold info for debugging/analysis
        result_df.attrs['irt_thresholds'] = thresholds
        result_df.attrs['irt_method'] = method
        
        return result_df


