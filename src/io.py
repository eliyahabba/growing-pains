from __future__ import annotations

import numpy as np
import pandas as pd


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
