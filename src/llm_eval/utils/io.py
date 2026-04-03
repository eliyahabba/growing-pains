

from pathlib import Path
from typing import Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p)


def write_parquet_safely(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, p)


def read_parquet_safely(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {p}")
    table = pq.read_table(p)
    return table.to_pandas()


