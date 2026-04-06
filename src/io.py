"""I/O helpers: rounding, saving DataFrames, metrics aggregation, GPU detection."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Rounding
# =============================================================================


def round_for_json(obj, decimals: int = 4):
    if isinstance(obj, dict):
        return {k: round_for_json(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_for_json(item, decimals) for item in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(obj, decimals)
    if isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return round(val, decimals)
    return obj


def round_df_for_save(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    df_rounded = df.copy()
    for col in df_rounded.select_dtypes(include=[np.number]).columns:
        df_rounded[col] = df_rounded[col].round(decimals)
    return df_rounded


# =============================================================================
# DataFrame persistence
# =============================================================================

_DS_COLS = ('dataset', 'dataset_name', 'scenario_name')


def save_df(df, name: str, output_dir: Path, method: str) -> None:
    """Save a per-model DataFrame as parquet + csv + json."""
    if df is None or len(df) == 0:
        return
    df_r = round_df_for_save(df)
    base = output_dir.parent / f"{name}_{method}"
    df_r.to_parquet(base.with_suffix('.parquet'), compression='snappy', index=False)
    df_r.to_csv(base.with_suffix('.csv'), index=False)
    if 'model_name' not in df_r.columns:
        return
    ds_col = next((c for c in _DS_COLS if c in df_r.columns), None)
    if ds_col:
        jdict: dict = {}
        for _, row in df_r.iterrows():
            jdict.setdefault(row['model_name'], {})[row[ds_col]] = {
                k: (None if pd.isna(v) else v)
                for k, v in row.items() if k not in ('model_name', ds_col)}
    elif df_r['model_name'].duplicated().any():
        jdict = df_r.to_dict(orient='records')
    else:
        jdict = df_r.set_index('model_name').to_dict(orient='index')
    base.with_suffix('.json').write_text(json.dumps(round_for_json(jdict), indent=2))


# =============================================================================
# Filesystem helpers
# =============================================================================


def cleanup_training_datasets(output_dir: Path) -> int:
    """Delete *.jsonlines training files from output_dir. Returns bytes freed."""
    total = 0
    for f in output_dir.glob("*.jsonlines"):
        try:
            total += f.stat().st_size
            f.unlink()
        except OSError:
            pass
    return total


def detect_gpus() -> list[int]:
    """Return available GPU ids from CUDA_VISIBLE_DEVICES or torch."""
    env = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if env:
        return [int(x) for x in env.split(',') if x.strip()]
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except Exception:
        return [0]
