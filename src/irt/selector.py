

from dataclasses import dataclass
from typing import Sequence, Optional
from pathlib import Path
import json

import numpy as np
import pandas as pd

from irt.interfaces import QuestionSelector, ModelProfile
from irt.cold_start import simple_cold_start_theta
from irt.anchors import find_anchor_items, AnchorConfig
from irt.estimation import estimate_theta_from_anchors, EstimationConfig


@dataclass
class TinyBenchmarksSelector(QuestionSelector):
    """Selector implementing the TinyBenchmarks tutorial pipeline.

    Steps:
      1) Fit/estimate 2PL item parameters from existing matrix.
      2) Choose anchor items distributed across difficulty.
      3) Estimate target model ability from (optional) anchor responses; if not provided,
         fall back to cold-start prior.
      4) Rank remaining questions by Fisher information at estimated theta.
    """

    # Anchor selection parameters
    number_items: int = 100  # fixed number of anchor items per dataset (from notebook)

    # No training parameters - selector only uses pre-trained artifacts

    # Optional cached artifacts to avoid retraining across calls
    _cached_item_params: Optional[pd.DataFrame] = None
    _cached_anchor_ids: Optional[list[str]] = None

    # Optional persistence paths (if provided, selector will load instead of training)
    item_params_path: Optional[str] = None  # Parquet path as produced by training CLI
    anchors_path: Optional[str] = None      # JSON path with {"anchors": [question_id,...]}

    def _fisher_information(self, theta: float, a: float, b: float) -> float:
        p = 1.0 / (1.0 + np.exp(-a * (theta - b)))
        return float((a ** 2) * p * (1 - p))

    def _load_item_params_from_disk(self) -> Optional[pd.DataFrame]:
        if not self.item_params_path:
            return None
        p = Path(str(self.item_params_path))
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if not {"a", "b"}.issubset(df.columns):
            return None
        return df

    def _get_item_params(self, matrix_df: pd.DataFrame) -> pd.DataFrame:
        if self._cached_item_params is not None and not self._cached_item_params.empty:
            return self._cached_item_params
        # Try to load from disk if path provided
        loaded = self._load_item_params_from_disk()
        if loaded is not None:
            self._cached_item_params = loaded
            return loaded
        
        # If no pre-trained parameters available, raise error
        # Training should be done externally via the pipeline
        raise ValueError(
            "No pre-trained IRT item parameters available. "
            "Either provide item_params_path to load existing parameters, "
            "or enable train_selection_artifacts=True in run_evaluation_pipeline.py "
            "to train parameters on the training split."
        )

    def _load_anchors_from_disk(self) -> Optional[list[str]]:
        if not self.anchors_path:
            return None
        p = Path(str(self.anchors_path))
        if not p.exists():
            return None
        try:
            with open(p, "r") as f:
                data = json.load(f)
            anchors = [str(x) for x in data.get("anchors", [])]
            return anchors if anchors else None
        except Exception:
            return None

    def _get_anchor_ids(self, item_params: pd.DataFrame) -> list[str]:
        if self._cached_anchor_ids is not None:
            return self._cached_anchor_ids
        loaded = self._load_anchors_from_disk()
        if loaded is not None:
            self._cached_anchor_ids = loaded
            return loaded
        acfg = AnchorConfig(number_items=self.number_items)
        anchor_ids = find_anchor_items(item_params, acfg)
        self._cached_anchor_ids = anchor_ids
        return anchor_ids

    def select(self, model: ModelProfile, k: int, matrix_df: pd.DataFrame) -> list[str]:
        # 1) Fit or load 2PL item params (reused across calls when possible)
        params = self._get_item_params(matrix_df)

        # 2) Choose anchors (reused across calls when possible)
        anchor_ids = self._get_anchor_ids(params)
        anchor_set = set(str(x) for x in anchor_ids)

        # 3) Estimate theta from anchor responses if we can derive them; otherwise cold start
        theta: float
        try:
            sub = matrix_df[
                (matrix_df["model_name"].astype(str) == str(model.model_name))
                & (matrix_df["question_id"].astype(str).isin(anchor_ids))
            ]
            if not sub.empty:
                # Use normalized_score directly (already binary from normalization pipeline)
                responses = pd.Series(sub["normalized_score"].values, index=sub["question_id"].astype(str).values)
                theta = estimate_theta_from_anchors(params, responses, config=EstimationConfig())
            else:
                theta = simple_cold_start_theta(model)
        except Exception:
            theta = simple_cold_start_theta(model)

        # 4) Score items by Fisher information at theta; exclude anchors from selection set
        infos: list[tuple[str, float]] = []
        for qid, row in params.iterrows():
            if str(qid) in anchor_set:
                continue
            a = float(row["a"])
            b = float(row["b"])
            info = self._fisher_information(theta, a, b)
            infos.append((str(qid), info))
        infos.sort(key=lambda x: x[1], reverse=True)
        return [qid for qid, _ in infos[:k]]

    def get_training_metadata(self) -> dict:
        """Get metadata from the IRT training process.
        
        Returns metadata such as validation errors, lambda values, balance weights,
        and best dimension from the training process. Only available after calling
        select() or explicitly training the model.
        """
        if self._cached_item_params is None:
            return {}
        
        metadata = {}
        for attr_name in ["val_errors_by_dataset", "lambdas_by_dataset", 
                         "balance_weights", "best_dimension"]:
            if hasattr(self._cached_item_params, "attrs") and attr_name in self._cached_item_params.attrs:
                metadata[attr_name] = self._cached_item_params.attrs[attr_name]
        
        return metadata


