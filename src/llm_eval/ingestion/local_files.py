

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from llm_eval.ingestion.base import IngestionSource
from llm_eval.utils import read_csv_safely, get_logger


logger = get_logger(__name__)


@dataclass
class LocalCSVSource(IngestionSource):
    """Load long-format observations from a local CSV file.

    The CSV is expected to contain at least the raw fields from the schema. Normalization is
    applied later in the pipeline.
    """

    path: str

    def load(self) -> pd.DataFrame:
        p = Path(self.path)
        if not p.is_absolute() and not p.exists():
            # Resolve relative to an ancestor directory (project root) when invoked from a different CWD (e.g., tests)
            for depth in range(1, 7):
                try:
                    ancestor = Path(__file__).parents[depth]
                except IndexError:
                    break
                candidate = ancestor / p
                if candidate.exists():
                    p = candidate
                    break
        logger.info("Loading CSV from %s", p)
        df = read_csv_safely(p)
        logger.info("Loaded %d rows", len(df))
        return df


