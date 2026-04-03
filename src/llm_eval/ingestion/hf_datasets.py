

from dataclasses import dataclass
from typing import Mapping, Any
import pandas as pd

from llm_eval.ingestion.base import IngestionSource


def _require_datasets():  # pragma: no cover - optional
    try:
        import datasets  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"datasets not available. Install with extras: pip install .[hf]. Details: {e}")


@dataclass
class HuggingFaceDatasetSource(IngestionSource):
    """Ingest a HF dataset and map to the long-format schema fields minimally.

    Parameters
    ----------
    dataset_name: str
        e.g., "gsm8k" or "openai_humaneval" (depends on mapping below)
    split: str
        dataset split to load
    field_map: Mapping[str, str]
        Optional mapping from HF columns to our long-format columns. If absent,
        uses a small set of known defaults for demo datasets.
    """

    dataset_name: str
    split: str = "test"
    field_map: Mapping[str, str] | None = None

    def load(self) -> pd.DataFrame:
        _require_datasets()
        from datasets import load_dataset

        ds = load_dataset(self.dataset_name, split=self.split)
        df = ds.to_pandas()

        mapping = self._resolve_mapping(df.columns)
        out = pd.DataFrame()
        # Minimal set; downstream builder can fill defaults
        out["dataset"] = self.dataset_name
        out["task_type"] = mapping.get("task_type", "unknown")
        out["question_id"] = df[mapping["question_id"]].astype(str)
        out["model_name"] = df.get(mapping.get("model_name", "model_name"), "unknown")
        out["metric_name"] = df.get(mapping.get("metric_name", "metric_name"), "binary_acc")
        out["raw_score"] = df[mapping["raw_score"]].astype(float)
        out["timestamp"] = pd.Timestamp.utcnow().isoformat()
        return out

    def _resolve_mapping(self, cols: Any) -> Mapping[str, str]:
        # Example defaults for a demo binary task; adapt per dataset
        if self.field_map is not None:
            return self.field_map
        # Heuristic defaults
        colset = set(map(str, cols))
        mapping: dict[str, str] = {
            "question_id": next((c for c in colset if c in {"id", "question_id"}), "id"),
            "raw_score": next((c for c in colset if c in {"label", "answer", "correct"}), "label"),
        }
        # Optional hints
        if "model_name" in colset:
            mapping["model_name"] = "model_name"
        if "metric_name" in colset:
            mapping["metric_name"] = "metric_name"
        mapping["task_type"] = "binary_qa"
        return mapping


