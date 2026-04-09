"""Data source configuration builders for each benchmark source."""
from __future__ import annotations

import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_helm_classic_config() -> dict:
    classic_datasets = {
        "BabiQA": "babiqascenario", "BBQ": "bbqscenario", "BLiMP": "blimpscenario",
        "BoolQ": "boolqscenario", "CivilComments": "civilcommentsscenario",
        "Code": "codescenario", "CommonSense": "commonsensescenario",
        "DyckLanguage": "dycklanguagescenario", "EntityDataImputation": "entitydataimputationscenario",
        "EntityMatching": "entitymatchingscenario", "GSM8K-Classic": "gsm8kscenario",
        "IMDB": "imdbscenario", "LegalSupport": "legalsupportscenario",
        "LSAT": "lsatscenario", "MATH-Classic": "mathscenario",
        "MMLU-Classic": "mmluscenario", "MS MARCO": "msmarcoscenario",
        "NarrativeQA-Classic": "narrativeqascenario", "NaturalQA-Classic": "naturalqascenario",
        "QuAC": "quacscenario", "RAFT": "raftscenario", "SRN": "srnscenario",
        "TruthfulQA-Classic": "truthfulqascenario", "WikiFact": "wikifactscenario",
    }
    datasets_config = {
        name: {"source_type": "aggregated", "source_file": "helm_classic_aggregated.parquet", "parquet_filter": pattern}
        for name, pattern in classic_datasets.items()
    }
    return {"datasets": datasets_config, "paths": {
        "tinybenchmarks_dir": str(PROJECT_ROOT / "aggregated_data/tinybenchmarks"),
        "aggregated_dir": str(PROJECT_ROOT / "aggregated_data/aggregated"),
    }}


def build_helm_lite_config() -> dict:
    lite_datasets = {
        "GSM8K-Lite": "gsm8kscenario", "LegalBench": "legalbench",
        "MATH Competition": "mathscenario", "MedQA": "medqascenario",
        "MMLU-Lite": "mmluscenario", "NarrativeQA": "narrativeqascenario",
        "NaturalQA": "naturalqascenario", "OpenBookQA": "openbookqa",
        "WMT-14 Translation": "wmt14scenario",
    }
    datasets_config = {
        name: {"source_type": "aggregated", "source_file": "helm_lite_aggregated.parquet", "parquet_filter": pattern}
        for name, pattern in lite_datasets.items()
    }
    return {"datasets": datasets_config, "paths": {
        "tinybenchmarks_dir": str(PROJECT_ROOT / "aggregated_data/tinybenchmarks"),
        "aggregated_dir": str(PROJECT_ROOT / "aggregated_data/aggregated"),
    }}


def build_lb_only_config() -> dict:
    datasets_config = {
        "ARC Challenge": {"source_type": "tinybenchmarks", "source_file": "lb.pickle", "pickle_keys": ["harness_arc_challenge_25"]},
        "GSM8K": {"source_type": "tinybenchmarks", "source_file": "lb.pickle", "pickle_keys": ["harness_gsm8k_5"]},
        "HellaSwag": {"source_type": "tinybenchmarks", "source_file": "lb.pickle", "pickle_keys": ["harness_hellaswag_10"]},
        "MMLU": {"source_type": "tinybenchmarks", "source_file": "mmlu_fields.pickle", "pickle_key_pattern": "hendrycksTest"},
        "TruthfulQA": {"source_type": "tinybenchmarks", "source_file": "lb.pickle", "pickle_keys": ["harness_truthfulqa_mc_0"]},
        "Winogrande": {"source_type": "tinybenchmarks", "source_file": "lb.pickle", "pickle_keys": ["harness_winogrande_5"]},
    }
    return {"datasets": datasets_config, "paths": {
        "tinybenchmarks_dir": str(PROJECT_ROOT / "aggregated_data/tinybenchmarks"),
        "aggregated_dir": str(PROJECT_ROOT / "aggregated_data/aggregated"),
    }}


def build_reeval_config() -> dict:
    import json
    import pandas as pd
    reeval_dir = PROJECT_ROOT / "aggregated_data" / "reeval"
    part1 = reeval_dir / "reeval_formatted_part1.parquet"
    part2 = reeval_dir / "reeval_formatted_part2.parquet"
    single = reeval_dir / "reeval_formatted.parquet"
    meta = reeval_dir / "reeval_metadata.json"

    scenario_names = []
    if meta.exists():
        info = json.loads(meta.read_text()).get('datasets', {})
        scenario_names = list(info.get('n_questions', {}).keys())
    if not scenario_names:
        if part1.exists() and part2.exists():
            scenario_names = sorted(set(pd.read_parquet(part1)['dataset'].unique()) | set(pd.read_parquet(part2)['dataset'].unique()))
        elif single.exists():
            scenario_names = sorted(pd.read_parquet(single)['dataset'].unique())
        else:
            raise FileNotFoundError("reeval dataset not found. Run: python src/experiments/prepare_reeval_dataset.py")

    datasets_config = {s: {"source_type": "reeval", "source_file": "reeval_formatted", "scenario_name": s} for s in scenario_names}
    return {"paths": {"reeval_dir": str(reeval_dir)}, "datasets": datasets_config}


def build_mmlu_split_config() -> dict:
    pickle_path = PROJECT_ROOT / "aggregated_data/tinybenchmarks/mmlu_fields.pickle"
    datasets_config = {}
    if pickle_path.exists():
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        for key in data.get('data', {}):
            if "hendrycksTest" in key:
                datasets_config[key.replace("hendrycksTest-", "MMLU-")] = {
                    "source_type": "tinybenchmarks", "source_file": "mmlu_fields.pickle", "pickle_keys": [key],
                }
    return {"datasets": datasets_config, "paths": {
        "tinybenchmarks_dir": str(PROJECT_ROOT / "aggregated_data/tinybenchmarks"),
        "aggregated_dir": str(PROJECT_ROOT / "aggregated_data/aggregated"),
    }}


def get_data_source_config(mode: str) -> dict:
    if mode == "helm_lite":
        return build_helm_lite_config()
    elif mode == "helm_classic":
        return build_helm_classic_config()
    elif mode in ("lb", "lb_only", "tinybenchmarks"):
        return build_lb_only_config()
    elif mode == "reeval":
        return build_reeval_config()
    elif mode in ("mmlu_split", "mmlu_fields"):
        return build_mmlu_split_config()
    else:
        raise ValueError(f"Unknown data source mode: {mode!r}. Options: lb, helm_lite, helm_classic, reeval, mmlu_split, mmlu_fields")
