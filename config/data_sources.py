"""Data source configuration builders for the two benchmark suites used in the paper."""
from __future__ import annotations

import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"


def build_lb_only_config() -> dict:
    """Open LLM Leaderboard: 6 datasets, 395 models (Table 1)."""
    datasets_config = {
        "ARC Challenge": {"source_type": "pickle", "source_file": "lb.pickle", "pickle_keys": ["harness_arc_challenge_25"]},
        "GSM8K": {"source_type": "pickle", "source_file": "lb.pickle", "pickle_keys": ["harness_gsm8k_5"]},
        "HellaSwag": {"source_type": "pickle", "source_file": "lb.pickle", "pickle_keys": ["harness_hellaswag_10"]},
        "MMLU": {"source_type": "pickle", "source_file": "mmlu_fields.pickle", "pickle_key_pattern": "hendrycksTest"},
        "TruthfulQA": {"source_type": "pickle", "source_file": "lb.pickle", "pickle_keys": ["harness_truthfulqa_mc_0"]},
        "Winogrande": {"source_type": "pickle", "source_file": "lb.pickle", "pickle_keys": ["harness_winogrande_5"]},
    }
    return {"datasets": datasets_config, "paths": {
        "input_dir": str(INPUT_DIR),
    }}


def build_mmlu_split_config() -> dict:
    """MMLU: 57 subject subdomains, 428 models (Table 1)."""
    pickle_path = INPUT_DIR / "mmlu_fields.pickle"
    datasets_config = {}
    if pickle_path.exists():
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        for key in data.get('data', {}):
            if "hendrycksTest" in key:
                datasets_config[key.replace("hendrycksTest-", "MMLU-")] = {
                    "source_type": "pickle", "source_file": "mmlu_fields.pickle", "pickle_keys": [key],
                }
    return {"datasets": datasets_config, "paths": {
        "input_dir": str(INPUT_DIR),
    }}


def get_data_source_config(mode: str) -> dict:
    if mode in ("lb", "lb_only"):
        return build_lb_only_config()
    elif mode in ("mmlu_split", "mmlu_fields"):
        return build_mmlu_split_config()
    else:
        raise ValueError(f"Unknown data source mode: {mode!r}. Options: lb, mmlu_fields")
