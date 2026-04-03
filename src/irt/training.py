from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from irt.anchors import AnchorConfig, find_anchor_items, find_anchor_items_clustering
from irt.fit import TrainingConfig, fit_2pl_parameters


def train_item_parameters(
        train_matrix_df: pd.DataFrame,
        test_matrix_df: pd.DataFrame | None = None,
        config: TrainingConfig | None = None,
        output_dir: str | None = None,
        anchor_items: list[dict] | None = None,
) -> pd.DataFrame:
    """Train or estimate 2PL item parameters (a,b) per question_id using tinyBenchmarks utilities.
    
    Args:
        train_matrix_df: Training data matrix for IRT parameter estimation
        test_matrix_df: Optional test data matrix (currently unused, reserved for future validation)
        config: Training configuration
        output_dir: Optional directory to save IRT dataset files (if None, uses temporary directory)
    
    Returns:
        DataFrame indexed by question_id with columns ["a", "b"] and attached metadata.
    """
    # Train IRT model on training data only
    # The notebook's internal validation logic will use cross-validation within the training set
    return fit_2pl_parameters(train_matrix_df, config, output_dir, anchor_items=anchor_items)


def save_item_parameters(df: pd.DataFrame, out_path: str) -> None:
    """Save item parameters to parquet, plus MIRT matrices to JSON if available."""
    import json
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    
    # Also save MIRT matrices and metadata to JSON if available in attrs
    if hasattr(df, 'attrs') and df.attrs:
        metadata_path = p.with_suffix('.meta.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(df.attrs, f)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")


# New structured API to support per-dataset anchors with weights (scenario-based)
def select_anchors_structured_with_matrix(
        item_params: pd.DataFrame,
        matrix_df: pd.DataFrame | None = None,
        number_items: int = 100,
        method: str = "irt_clustering",
        dataset_column: str = "dataset"
) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
    """Select anchors per dataset and return questions and weights by dataset.

    - For "irt_clustering" and "correctness_clustering": compute KMeans-based anchors and weights.
    - For "difficulty_binning": return anchors with uniform weights.
    """
    anchors_by_dataset: dict[str, list[str]] = {}
    weights_by_dataset: dict[str, list[float]] = {}

    # If dataset info is not available in item_params, but matrix_df is provided,
    # build a mapping from question_id to scenario (dataset prefix before '.') using the matrix.
    if matrix_df is not None and dataset_column in matrix_df.columns and "question_id" in matrix_df.columns:
        q_to_ds = (
            matrix_df[["question_id", dataset_column]]
            .dropna()
            .drop_duplicates()
            .set_index("question_id")[dataset_column]
            .to_dict()
        )

        # Helper to extract scenario root from dataset name
        def scenario_from_dataset(name: str) -> str:
            return name.split(".")[0] if "." in name else name

        # Group question_ids by scenario (dataset root, not subdataset)
        ds_to_questions: dict[str, list[str]] = {}
        for q in item_params.index.tolist():
            ds = q_to_ds.get(q)
            if ds is None:
                continue
            scenario = scenario_from_dataset(str(ds))
            ds_to_questions.setdefault(scenario, []).append(q)

        for ds, qids in ds_to_questions.items():
            if len(qids) == 0:
                continue
            grp = item_params.loc[qids]
            actual_number_items = min(number_items, len(grp))
            cfg = AnchorConfig(method=method, number_items=actual_number_items)

            if method == "correctness_clustering":
                if matrix_df is None:
                    raise ValueError("matrix_df required for correctness-based clustering")
                # For scenario grouping: restrict only by questions in this scenario group
                dataset_matrix = matrix_df[matrix_df["question_id"].isin(grp.index)]
                anchor_ids, anchor_weights = find_anchor_items_clustering(grp, dataset_matrix, cfg)
                anchors_by_dataset[str(ds)] = anchor_ids
                weights_by_dataset[str(ds)] = anchor_weights.tolist()
            elif method == "irt_clustering":
                anchor_ids, anchor_weights = find_anchor_items_clustering(grp, None, cfg)
                anchors_by_dataset[str(ds)] = anchor_ids
                weights_by_dataset[str(ds)] = anchor_weights.tolist()
            elif method == "difficulty_binning":
                anchor_ids = find_anchor_items(grp, cfg)
                anchors_by_dataset[str(ds)] = anchor_ids
                if len(anchor_ids) > 0:
                    uniform = [1.0 / float(len(anchor_ids))] * len(anchor_ids)
                else:
                    uniform = []
                weights_by_dataset[str(ds)] = uniform
            else:
                raise ValueError(f"Unknown anchor selection method: {method}")

        if anchors_by_dataset:
            return anchors_by_dataset, weights_by_dataset

    # Fallback: no dataset column; select from all data and provide uniform weights
    cfg = AnchorConfig(method=method, number_items=number_items)
    if method == "correctness_clustering":
        if matrix_df is None:
            raise ValueError("matrix_df required for correctness-based clustering")
        anchor_ids, anchor_weights = find_anchor_items_clustering(item_params, matrix_df, cfg)
        anchors_by_dataset["__all__"] = anchor_ids
        weights_by_dataset["__all__"] = anchor_weights.tolist()
    elif method == "irt_clustering":
        anchor_ids, anchor_weights = find_anchor_items_clustering(item_params, None, cfg)
        anchors_by_dataset["__all__"] = anchor_ids
        weights_by_dataset["__all__"] = anchor_weights.tolist()
    elif method == "difficulty_binning":
        anchor_ids = find_anchor_items(item_params, cfg)
        anchors_by_dataset["__all__"] = anchor_ids
        weights_by_dataset["__all__"] = [1.0 / float(len(anchor_ids))] * len(anchor_ids) if len(anchor_ids) > 0 else []
    else:
        raise ValueError(f"Unknown anchor selection method: {method}")

    return anchors_by_dataset, weights_by_dataset


def save_anchors_structured(
        anchors_by_dataset: dict[str, list[str]],
        anchor_weights_by_dataset: dict[str, list[float]],
        out_path: str,
) -> None:
    """Save structured anchors with weights by dataset to JSON.

    Schema:
    {
      "anchors_by_dataset": { dataset: [question_id, ...], ... },
      "anchor_weights_by_dataset": { dataset: [w1, w2, ...], ... }
    }
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchors_by_dataset": anchors_by_dataset,
        "anchor_weights_by_dataset": anchor_weights_by_dataset,
    }
    with open(p, "w") as f:
        json.dump(payload, f)
