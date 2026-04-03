

from abc import ABC, abstractmethod
from typing import Protocol
import pandas as pd


class IngestionSource(ABC):
    """Abstract base class for ingestion sources.

    Responsibility
    --------------
    - Connect to one concrete data origin (CSV/JSONL/Parquet/HF dataset/API).
    - Download or read raw records as-is.
    - Optionally map fields into the unified long-record schema columns if feasible.

    Separation of concerns
    ----------------------
    - Adapters under `ingestion/` focus on fetching and minimal mapping.
    - Any union/merging across multiple sources happens in a higher-level
      orchestration step (CLI or a small coordinator), before building the matrix.
    """

    @abstractmethod
    def load(self) -> pd.DataFrame:  # pragma: no cover - interface
        """Load raw records to a DataFrame (long format or close)."""
        raise NotImplementedError


