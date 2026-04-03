from dataclasses import dataclass
from typing import Mapping, Any

import pandas as pd
from tqdm import tqdm

from llm_eval.normalization import MetricRegistry
from llm_eval.utils import get_logger

logger = get_logger(__name__)


@dataclass
class MatrixBuilder:
    registry: MetricRegistry
    use_irt_normalization: bool = True  # Use IRT-style normalization if True, else standard normalization
    irt_method: str = 'direct'  # 'direct' or 'normalized'

    def build(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize, returning standard long-format DataFrame."""

        if self.use_irt_normalization:
            return self._build_with_irt_normalization(raw_df)
        else:
            return self._build_with_standard_normalization(raw_df)

    def _prepare_base_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare base DataFrame with all required columns using vectorized operations."""
        # Use vectorized operations instead of iterrows for much better performance
        df = raw_df.copy()

        # Convert required columns with proper defaults
        df["metric_name"] = df["metric_name"].astype(str)
        df["raw_score"] = df["raw_score"].astype(float)

        return df

    def _get_higher_is_better_mapping(self, unique_metrics: pd.Series) -> dict[str, bool]:
        """Create a mapping of metric -> is_higher_better for unique metrics only."""
        mapping = {}
        for metric in unique_metrics:
            is_higher = self.registry.config.metrics.get(metric, None)
            mapping[metric] = is_higher.higher_is_better if is_higher else True
        return mapping

    def _normalize_metric_group(self, group_data):
        """Normalize all scores for a single metric group."""
        metric = group_data.name  # group name is the metric
        metadata: Mapping[str, Any] = {}

        normalized_scores = []
        for raw_score in group_data["raw_score"]:
            norm = self.registry.normalize(metric, raw_score, metadata)
            normalized_scores.append(float(norm.normalized))

        # Return Series with same index as input group
        return pd.Series(normalized_scores, index=group_data.index)

    def _build_with_standard_normalization(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Original normalization method - per-score normalization using vectorized operations."""
        logger.info("Building matrix with standard normalization using vectorized operations...")

        # Prepare base DataFrame
        df = self._prepare_base_dataframe(raw_df)

        # Use groupby for more efficient batch processing
        logger.info("Normalizing scores by metric groups...")
        tqdm.pandas(desc="Normalizing metrics")

        # Group by metric and apply normalization to each group
        normalized_series_list = []
        for metric_name, group in tqdm(df.groupby("metric_name"), desc="Processing metric groups"):
            normalized_group = self._normalize_metric_group(group)
            normalized_series_list.append(normalized_group)

        # Concatenate all normalized scores and sort by index
        all_normalized = pd.concat(normalized_series_list).sort_index()
        df["normalized_score"] = all_normalized

        # Get is_higher_better values efficiently using mapping
        unique_metrics = df["metric_name"].unique()
        is_higher_mapping = self._get_higher_is_better_mapping(unique_metrics)
        df["is_higher_better"] = df["metric_name"].map(is_higher_mapping)

        # Select and rename columns to match ObservationRow schema
        result_df = df[[
            "dataset", "split", "question_id", "model_name",
            "model_family", "metric_name", "raw_score",
            "normalized_score", "is_higher_better"
        ]].copy()

        logger.info("Built matrix with %d rows using standard normalization", len(result_df))
        return result_df

    def _build_with_irt_normalization(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """IRT-style normalization - per-scenario optimal thresholds using vectorized operations."""
        logger.info("Building matrix with IRT normalization using vectorized operations...")

        # Prepare base DataFrame
        df = self._prepare_base_dataframe(raw_df)

        # Set initial normalized_score to 0.0 (will be computed by IRT)
        df["normalized_score"] = 0.0

        # Get is_higher_better values efficiently using mapping
        unique_metrics = df["metric_name"].unique()
        is_higher_mapping = self._get_higher_is_better_mapping(unique_metrics)
        df["is_higher_better"] = df["metric_name"].map(is_higher_mapping)

        # Select and rename columns to match ObservationRow schema
        result_df = df[[
            "dataset", "hf_split", "question_id", "model_name",
            "model_family", "metric_name", "raw_score",
            "normalized_score", "is_higher_better"
        ]].copy()

        # Apply IRT normalization to the entire dataset
        result_df = self.registry.normalize_with_irt_thresholds(result_df, method=self.irt_method)

        logger.info("Built matrix with %d rows using IRT normalization (%s method)", len(result_df), self.irt_method)
        if hasattr(result_df, 'attrs') and 'irt_thresholds' in result_df.attrs:
            logger.info("IRT thresholds computed: %s", result_df.attrs['irt_thresholds'])

        return result_df
