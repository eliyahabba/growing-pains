

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from llm_eval.utils import read_parquet_safely, write_parquet_safely, utc_now_iso


@dataclass
class SnapshotManager:
    path: str
    decay: float = 0.9

    def create_snapshot(self, matrix_df: pd.DataFrame) -> str:
        snapshot_id = utc_now_iso()
        df = matrix_df.copy()
        df["snapshot_id"] = snapshot_id
        out = Path(self.path)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_parquet_safely(df, out)
        return snapshot_id

    def aggregate_scores(self, matrix_df: pd.DataFrame) -> pd.DataFrame:
        """Compute decayed cumulative scores per model.

        Strategy: per model, average normalized scores by snapshot with decay weights.
        """
        df = matrix_df.copy()
        if "snapshot_id" not in df.columns:
            df["snapshot_id"] = "current"
        # order snapshots by time-like id if possible
        snapshots = list(dict.fromkeys(df["snapshot_id"].tolist()))
        weights = {sid: self.decay ** i for i, sid in enumerate(reversed(snapshots))}
        df["weight"] = df["snapshot_id"].map(weights)
        grouped = df.groupby(["model_name", "snapshot_id"]).agg(avg_score=("normalized_score", "mean"), weight=("weight", "first"))  # noqa: E501
        grouped = grouped.reset_index()
        agg = grouped.groupby("model_name").apply(lambda g: (g["avg_score"] * g["weight"]).sum() / g["weight"].sum()).reset_index(name="cumulative_score")  # noqa: E501
        return agg


