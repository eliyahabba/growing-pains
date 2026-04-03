

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from llm_eval.utils import write_parquet_safely, read_parquet_safely


@dataclass
class MatrixStorage:
    path: str

    def save(self, df: pd.DataFrame) -> None:
        write_parquet_safely(df, self.path)

    def load(self) -> pd.DataFrame:
        return read_parquet_safely(self.path)


