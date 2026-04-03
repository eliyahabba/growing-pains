"""
Cross-Dataset Equating: data loading, IRT helpers, and validation functions.

Provides shared utilities for the chain-linking sweep:
- Dataset loading and grouping by skill/source
- IRT training on the base set
- Anchor selection and fixed-parameter calibration
- Theta precomputation
- Validation and baseline routines (random, discriminative)
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from irt import (
    TrainingConfig,
    fit_2pl_parameters,
    compute_lambda_values,
    run_estimation_validation,
    estimate_theta_from_anchors,
    find_anchor_items_clustering,
    AnchorConfig,
    train_item_parameters,
    save_item_parameters,
)

from src.experiments.utils.io import round_df_for_save, round_for_json


# =============================================================================
# Configuration
# =============================================================================

# Project root (src/experiments/equating/cross_dataset_equating.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class ExperimentConfig:
    """Configuration for cross-dataset equating experiments."""
    tinybenchmarks_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "input" / "tinybenchmarks"
    )
    skill_labels_csv: Path = field(
        default_factory=lambda: PROJECT_ROOT / "src" / "dataset_skill_labels.csv"
    )
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cross_dataset_equating")
    
    # Data source mode - determines which datasets to use
    # Options: "mixed" (current default), "helm_lite", "helm_classic", "lb_only", "reeval"
    data_source_mode: str = "mixed"
    
    # IRT training
    dims_search: list = field(default_factory=lambda: [2, 5])
    epochs: int = 2000
    lr: float = 0.1
    n_anchors_per_dataset: int = 100  # Anchors to select from EACH dataset
    
    # Split
    test_ratio: float = 0.25
    seed: int = 42
    
    # Caching
    force_retrain: bool = False
    
    # Experiment mode
    all_datasets_mode: bool = False  # If True, combine all datasets instead of grouping by skill
    
    # Fixed-anchor calibration mode
    anchor_only_fixed: bool = True  # If True, only freeze selected anchors (faster). If False, freeze all Base items.
    
    # Zero-variance filtering for IRT training
    filter_zero_variance: bool = False  # If True, remove zero-variance questions (uninformative for IRT)

    # Dimension selection
    # Always enabled to ensure proper lambda computation for GP-IRT.
    validate_dimensions: bool = True


# =============================================================================
# Data Loading
# =============================================================================

def load_skill_labels(csv_path: str | Path) -> pd.DataFrame:
    """Load and parse skill labels CSV with multi-label support."""
    df = pd.read_csv(csv_path)
    
    # Parse skills into lists
    def parse_skills(row):
        skills = set()
        if pd.notna(row.get('Primary skills')):
            skills.add(row['Primary skills'].strip())
        if pd.notna(row.get('Secondary skills')):
            for s in str(row['Secondary skills']).split(';'):
                s = s.strip()
                if s and s != 'nan':
                    skills.add(s)
        return list(skills)
    
    df['all_skills'] = df.apply(parse_skills, axis=1)
    return df


def load_data_source_config(config_path: str | None = None) -> dict:
    """Load the data source configuration file and resolve relative paths."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "data_source_config.json"

    with open(config_path) as f:
        config = json.load(f)
    
    # Resolve relative paths to absolute (relative to project root)
    project_root = PROJECT_ROOT  # resolve once to avoid off-by-one mistakes
    if "paths" in config:
        for key, value in config["paths"].items():
            if key.startswith("_"):  # skip notes
                continue
            path = Path(value)
            if not path.is_absolute():
                config["paths"][key] = str(project_root / path)
    
    return config


def build_helm_classic_config() -> dict:
    """Build data source config for HELM Classic datasets only (70 models, 30 datasets)."""
    project_root = PROJECT_ROOT
    
    # Dataset patterns in helm_classic_aggregated.parquet
    classic_datasets = {
        "BabiQA": "babiqascenario",
        "BBQ": "bbqscenario",
        "BLiMP": "blimpscenario",
        "BOLD": "boldscenario",
        "BoolQ": "boolqscenario",
        "CivilComments": "civilcommentsscenario",
        "Code": "codescenario",
        "CommonSense": "commonsensescenario",
        "Copyright": "copyrightscenario",
        "Disinformation": "disinformationscenario",
        "DyckLanguage": "dycklanguagescenario",
        "EntityDataImputation": "entitydataimputationscenario",
        "EntityMatching": "entitymatchingscenario",
        "GSM8K-Classic": "gsm8kscenario",
        "IMDB": "imdbscenario",
        "LegalSupport": "legalsupportscenario",
        "LSAT": "lsatscenario",
        "MATH-Classic": "mathscenario",
        "MMLU-Classic": "mmluscenario",
        "MS MARCO": "msmarcoscenario",
        "NarrativeQA-Classic": "narrativeqascenario",
        "NaturalQA-Classic": "naturalqascenario",
        "QuAC": "quacscenario",
        "RAFT": "raftscenario",
        "RealToxicityPrompts": "realtoxicityprompts",
        "SRN": "srnscenario",
        "Summarization": "summarizationscenario",
        "SyntheticReasoning": "syntheticreasoningscenario",
        "TruthfulQA-Classic": "truthfulqascenario",
        "WikiFact": "wikifactscenario",
    }
    
    datasets_config = {}
    for name, pattern in classic_datasets.items():
        datasets_config[name] = {
            "source_type": "aggregated",
            "source_file": "helm_classic_aggregated.parquet",
            "parquet_filter": pattern,
            "models": 70,
        }
    
    return {
        "datasets": datasets_config,
        "paths": {
            "tinybenchmarks_dir": str(project_root / "aggregated_data/tinybenchmarks"),
            "aggregated_dir": str(project_root / "aggregated_data/aggregated"),
        }
    }


def build_helm_lite_config() -> dict:
    """Build data source config for HELM Lite datasets only (91 models, 9 datasets)."""
    project_root = PROJECT_ROOT
    
    lite_datasets = {
        "GSM8K-Lite": "gsm8kscenario",
        "LegalBench": "legalbench",
        "MATH Competition": "mathscenario",
        "MedQA": "medqascenario",
        "MMLU-Lite": "mmluscenario",
        "NarrativeQA": "narrativeqascenario",
        "NaturalQA": "naturalqascenario",
        "OpenBookQA": "openbookqa",
        "WMT-14 Translation": "wmt14scenario",
    }
    
    datasets_config = {}
    for name, pattern in lite_datasets.items():
        datasets_config[name] = {
            "source_type": "aggregated",
            "source_file": "helm_lite_aggregated.parquet",
            "parquet_filter": pattern,
            "models": 91,
        }
    
    return {
        "datasets": datasets_config,
        "paths": {
            "tinybenchmarks_dir": str(project_root / "aggregated_data/tinybenchmarks"),
            "aggregated_dir": str(project_root / "aggregated_data/aggregated"),
        }
    }


def build_lb_only_config() -> dict:
    """Build data source config for Open LLM Leaderboard datasets only (395 models, 6 datasets)."""
    project_root = PROJECT_ROOT
    
    datasets_config = {
        "ARC Challenge": {
            "source_type": "tinybenchmarks",
            "source_file": "lb.pickle",
            "pickle_keys": ["harness_arc_challenge_25"],
            "models": 395,
        },
        "GSM8K": {
            "source_type": "tinybenchmarks",
            "source_file": "lb.pickle",
            "pickle_keys": ["harness_gsm8k_5"],
            "models": 395,
        },
        "HellaSwag": {
            "source_type": "tinybenchmarks",
            "source_file": "lb.pickle",
            "pickle_keys": ["harness_hellaswag_10"],
            "models": 395,
        },
        "MMLU": {
            "source_type": "tinybenchmarks",
            "source_file": "mmlu_fields.pickle",
            "pickle_keys": None,
            "pickle_key_pattern": "hendrycksTest",
            "models": 428,
        },
        "TruthfulQA": {
            "source_type": "tinybenchmarks",
            "source_file": "lb.pickle",
            "pickle_keys": ["harness_truthfulqa_mc_0"],
            "models": 395,
        },
        "Winogrande": {
            "source_type": "tinybenchmarks",
            "source_file": "lb.pickle",
            "pickle_keys": ["harness_winogrande_5"],
            "models": 395,
        },
    }
    
    return {
        "datasets": datasets_config,
        "paths": {
            "tinybenchmarks_dir": str(project_root / "aggregated_data/tinybenchmarks"),
            "aggregated_dir": str(project_root / "aggregated_data/aggregated"),
        }
    }


def build_reeval_config() -> dict:
    """
    Build configuration for reeval dataset (stair-lab/reeval).
    
    This dataset contains:
    - 183 models
    - 22 scenarios (datasets)
    - ~5.7M rows total
    - Pre-converted to our format and saved as parquet (split into 2 parts)
    
    Returns:
        Configuration dict with all reeval scenarios
    """
    project_root = PROJECT_ROOT
    reeval_dir = project_root / "aggregated_data" / "reeval"
    
    # Check for split files (preferred)
    part1_file = reeval_dir / "reeval_formatted_part1.parquet"
    part2_file = reeval_dir / "reeval_formatted_part2.parquet"
    
    # Also check for single file (backward compatibility)
    single_file = reeval_dir / "reeval_formatted.parquet"
    
    # Load metadata to get list of scenarios
    metadata_file = reeval_dir / "reeval_metadata.json"
    scenario_names = []
    
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        # The metadata has datasets -> n_questions -> {scenario: count}
        # We need to extract the scenario names
        datasets_info = metadata.get('datasets', {})
        if 'n_questions' in datasets_info:
            scenario_names = list(datasets_info['n_questions'].keys())
    
    # Fallback to loading the data to get scenarios
    if not scenario_names:
        if part1_file.exists() and part2_file.exists():
            df1 = pd.read_parquet(part1_file)
            df2 = pd.read_parquet(part2_file)
            scenario_names = sorted(set(df1['dataset'].unique()) | set(df2['dataset'].unique()))
        elif single_file.exists():
            df = pd.read_parquet(single_file)
            scenario_names = sorted(df['dataset'].unique())
        else:
            raise FileNotFoundError(
                f"reeval dataset not found. "
                f"Please run: python src/experiments/prepare_reeval_dataset.py"
            )
    
    # Build config with all scenarios
    datasets_config = {}
    for scenario in scenario_names:
        datasets_config[scenario] = {
            "source_type": "reeval",
            "source_file": "reeval_formatted",  # Will load both parts
            "scenario_name": scenario,
        }
    
    return {
        "paths": {
            "reeval_dir": str(reeval_dir),
        },
        "datasets": datasets_config,
    }


def build_mmlu_split_config() -> dict:
    """Build config that treats every MMLU subtask as a separate dataset."""
    project_root = PROJECT_ROOT
    tinybenchmarks_dir = project_root / "aggregated_data/tinybenchmarks"
    pickle_path = tinybenchmarks_dir / "mmlu_fields.pickle"
    
    datasets_config = {}
    
    # We need to peek at the pickle to get the subtask names
    if pickle_path.exists():
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            all_data = data.get('data', {})
            # Find all MMLU keys (hendrycksTest)
            mmlu_keys = [k for k in all_data.keys() if "hendrycksTest" in k]
            
            for key in mmlu_keys:
                # Clean name: "hendrycksTest-abstract_algebra" -> "MMLU-abstract_algebra"
                clean_name = key.replace("hendrycksTest-", "MMLU-")
                datasets_config[clean_name] = {
                    "source_type": "tinybenchmarks",
                    "source_file": "mmlu_fields.pickle",
                    "pickle_keys": [key],  # Load only this specific subtask
                    "models": 428,
                }
    else:
        print(f"Warning: {pickle_path} not found, cannot build MMLU split config")

    return {
        "datasets": datasets_config,
        "paths": {
            "tinybenchmarks_dir": str(tinybenchmarks_dir),
            "aggregated_dir": str(project_root / "aggregated_data/aggregated"),
        }
    }


def get_data_source_config(mode: str) -> dict:
    """Get data source configuration based on mode.
    
    Args:
        mode: One of "mixed", "helm_lite", "helm_classic", "lb_only", "lb", "reeval", "mmlu_split", "mmlu_fields", "tinybenchmarks"
    
    Returns:
        Data source configuration dict
    """
    if mode == "mixed":
        return load_data_source_config()
    elif mode == "helm_lite":
        return build_helm_lite_config()
    elif mode == "helm_classic":
        return build_helm_classic_config()
    elif mode in ["lb_only", "lb", "tinybenchmarks"]:
        return build_lb_only_config()
    elif mode == "reeval":
        return build_reeval_config()
    elif mode in ["mmlu_split", "mmlu_fields"]:
        return build_mmlu_split_config()
    else:
        raise ValueError(f"Unknown data source mode: {mode}. "
                        f"Options: mixed, helm_lite, helm_classic, lb_only, lb, reeval, mmlu_split, mmlu_fields, tinybenchmarks")


def load_pickle_data(pickle_path: str) -> dict:
    """Load data from a TinyBenchmarks pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def extract_from_pickle(
    pickle_data: dict,
    dataset_name: str,
    keys: list[str] | None = None,
    key_pattern: str | None = None,
) -> pd.DataFrame:
    """Extract a dataset from pickle data.
    
    Args:
        pickle_data: Loaded pickle data dict
        dataset_name: Clean dataset name
        keys: Specific keys to extract, or None
        key_pattern: Pattern to match keys (used if keys is None)
    
    Returns:
        DataFrame with columns: model_name, question_id, dataset, normalized_score
    """
    models = np.array(pickle_data.get('models', []))
    all_data = pickle_data.get('data', {})
    
    # Determine which keys to use
    if keys:
        keys_to_use = [k for k in keys if k in all_data]
    elif key_pattern:
        keys_to_use = [k for k in all_data.keys() if key_pattern in k]
    else:
        keys_to_use = list(all_data.keys())
    
    if not keys_to_use:
        return pd.DataFrame()
    
    dfs = []
    for key in keys_to_use:
        data_item = all_data[key]
        
        # Extract scores matrix
        if isinstance(data_item, dict):
            scores = data_item.get('correctness', data_item.get('scores'))
        else:
            scores = data_item
        
        if scores is None:
            continue
            
        scores = np.array(scores)
        if len(scores.shape) != 2:
            continue
        
        # Ensure shape is (n_questions, n_models)
        if scores.shape[0] == len(models):
            scores = scores.T
        elif scores.shape[1] != len(models):
            continue
        
        n_questions = scores.shape[0]
        
        # Build DataFrame efficiently
        q_indices = np.arange(n_questions)
        m_indices = np.arange(len(models))
        q_grid, m_grid = np.meshgrid(q_indices, m_indices, indexing='ij')
        
        scores_flat = scores.flatten()
        valid_mask = ~np.isnan(scores_flat)
        
        df = pd.DataFrame({
            'model_name': models[m_grid.flatten()[valid_mask]],
            'question_id': [f"{dataset_name}:{key}:{q}" for q in q_grid.flatten()[valid_mask]],
            'dataset': dataset_name,
            'sub_dataset': key,
            'normalized_score': scores_flat[valid_mask],
        })
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['model_name', 'question_id'])


def extract_from_parquet(
    parquet_df: pd.DataFrame,
    dataset_name: str,
    filter_pattern: str,
) -> pd.DataFrame:
    """Extract a dataset from aggregated parquet DataFrame.
    
    Args:
        parquet_df: Loaded parquet DataFrame
        dataset_name: Clean dataset name
        filter_pattern: Pattern to filter dataset_name column
    
    Returns:
        DataFrame with columns: model_name, question_id, dataset, normalized_score
    """
    # Filter by dataset name pattern
    mask = parquet_df['dataset_name'].str.contains(filter_pattern, case=False, na=False)
    df = parquet_df[mask].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Build question_id from dataset_name + hf_split + hf_index
    df['question_id'] = (
        dataset_name + ":" + 
        df['dataset_name'].astype(str) + ":" + 
        df['hf_split'].astype(str) + ":" + 
        df['hf_index'].astype(str)
    )
    
    # Rename columns to match expected format
    result = pd.DataFrame({
        'model_name': df['model_name'],
        'question_id': df['question_id'],
        'dataset': dataset_name,
        'sub_dataset': df['dataset_name'],
        'normalized_score': df['evaluation_score'],
    })
    
    return result.drop_duplicates(subset=['model_name', 'question_id'])


def extract_from_reeval(
    reeval_path: str,
    scenario_name: str,
) -> pd.DataFrame:
    """Extract a scenario from reeval formatted parquet file.
    
    The reeval data is already in our format, just filter by scenario.
    
    Args:
        reeval_path: Path to reeval_formatted.parquet
        scenario_name: Scenario name to filter
    
    Returns:
        DataFrame with columns: model_name, question_id, dataset, normalized_score
    """
    df = pd.read_parquet(reeval_path)
    
    # Filter by scenario name
    df = df[df['dataset'] == scenario_name].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Select required columns (already in correct format)
    result = df[['model_name', 'question_id', 'dataset', 'normalized_score']].copy()
    
    return result.drop_duplicates(subset=['model_name', 'question_id'])


def load_all_datasets(config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """Load all datasets using the data source configuration.
    
    Uses data_source_config.json or mode-specific config to determine the best source.
    
    Args:
        config: ExperimentConfig with data_source_mode field
    
    Returns: dict mapping dataset_name -> DataFrame
    """
    # Load config based on mode
    source_config = get_data_source_config(config.data_source_mode)
    datasets_config = source_config.get('datasets', {})
    paths_config = source_config.get('paths', {})
    
    print(f"   Data source mode: {config.data_source_mode}")
    print(f"   Available datasets in config: {len(datasets_config)}")
    
    tinybenchmarks_dir = Path(paths_config.get('tinybenchmarks_dir', config.tinybenchmarks_dir))
    aggregated_dir = Path(paths_config.get('aggregated_dir', 
                          str(Path(config.tinybenchmarks_dir).parent / 'aggregated')))
    reeval_dir = Path(paths_config.get('reeval_dir', 
                      str(Path(config.tinybenchmarks_dir).parent / 'reeval')))
    
    # Cache loaded pickle files, parquet files, and reeval data
    loaded_pickles = {}
    loaded_parquets = {}
    reeval_data = None
    
    # Determine which datasets to load
    if config.data_source_mode == "mixed":
        # For mixed mode, filter by skill_labels.csv
        skill_labels = load_skill_labels(config.skill_labels_csv)
        needed_datasets = sorted(set(skill_labels['Dataset'].unique()))
    else:
        # For specific modes, load ALL datasets from the config
        needed_datasets = sorted(datasets_config.keys())
    
    datasets = {}
    
    for dataset_name in needed_datasets:
        if dataset_name not in datasets_config:
            print(f"  Warning: No source config for {dataset_name}, skipping")
            continue
        
        ds_config = datasets_config[dataset_name]
        source_type = ds_config.get('source_type')
        source_file = ds_config.get('source_file')
        
        try:
            if source_type == 'tinybenchmarks':
                # Load from pickle
                pickle_path = tinybenchmarks_dir / source_file
                
                if source_file not in loaded_pickles:
                    if pickle_path.exists():
                        loaded_pickles[source_file] = load_pickle_data(str(pickle_path))
                        print(f"  Loaded pickle: {source_file} "
                              f"({len(loaded_pickles[source_file].get('models', []))} models)")
                    else:
                        print(f"  Warning: {pickle_path} not found, skipping {dataset_name}")
                        continue
                
                pickle_data = loaded_pickles[source_file]
                keys = ds_config.get('pickle_keys')
                key_pattern = ds_config.get('pickle_key_pattern')
                
                df = extract_from_pickle(pickle_data, dataset_name, keys, key_pattern)
                
            elif source_type == 'aggregated':
                # Load from parquet (with caching)
                parquet_path = aggregated_dir / source_file
                
                if source_file not in loaded_parquets:
                    if not parquet_path.exists():
                        print(f"  Warning: {parquet_path} not found, skipping {dataset_name}")
                        continue
                    loaded_parquets[source_file] = pd.read_parquet(parquet_path)
                    print(f"  Loaded parquet: {source_file}")
                
                parquet_df = loaded_parquets[source_file]
                filter_pattern = ds_config.get('parquet_filter', dataset_name)
                df = extract_from_parquet(parquet_df, dataset_name, filter_pattern)
            
            elif source_type == 'reeval':
                # Load from reeval parquet (load once and cache, supports split files)
                reeval_dir_path = Path(reeval_dir)
                part1_path = reeval_dir_path / "reeval_formatted_part1.parquet"
                part2_path = reeval_dir_path / "reeval_formatted_part2.parquet"
                single_path = reeval_dir_path / "reeval_formatted.parquet"
                
                # Check which files exist
                has_split = part1_path.exists() and part2_path.exists()
                has_single = single_path.exists()
                
                if not has_split and not has_single:
                    print(f"  Warning: reeval data not found, skipping {dataset_name}")
                    print(f"  Please run: python src/experiments/prepare_reeval_dataset.py")
                    continue
                
                scenario_name = ds_config.get('scenario_name', dataset_name)
                
                # Load reeval data once and cache
                if reeval_data is None:
                    if has_split:
                        print(f"  Loading reeval data from split files...")
                        df1 = pd.read_parquet(part1_path)
                        df2 = pd.read_parquet(part2_path)
                        reeval_data = pd.concat([df1, df2], ignore_index=True)
                        print(f"  Loaded reeval: {reeval_data['model_name'].nunique()} models, "
                              f"{reeval_data['dataset'].nunique()} scenarios")
                    else:
                        print(f"  Loading reeval data from single file...")
                        reeval_data = pd.read_parquet(single_path)
                        print(f"  Loaded reeval: {reeval_data['model_name'].nunique()} models, "
                              f"{reeval_data['dataset'].nunique()} scenarios")
                
                # Extract this scenario (already in correct format)
                df = reeval_data[reeval_data['dataset'] == scenario_name].copy()
                if not df.empty:
                    df = df[['model_name', 'question_id', 'dataset', 'normalized_score']].copy()
                    df = df.drop_duplicates(subset=['model_name', 'question_id'])
            
            else:
                print(f"  Warning: Unknown source type '{source_type}' for {dataset_name}")
                continue
            
            if not df.empty:
                datasets[dataset_name] = df
                print(f"  ✓ {dataset_name}: {df['question_id'].nunique()} questions, "
                      f"{df['model_name'].nunique()} models from {source_type}/{source_file}")
            else:
                print(f"  Warning: No data extracted for {dataset_name}")
                
        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return datasets


# =============================================================================
# Skill Grouping
# =============================================================================

def get_dataset_source(dataset_name: str, source_config: dict) -> str:
    """Get the source type for a dataset."""
    ds_config = source_config.get('datasets', {}).get(dataset_name, {})
    source_type = ds_config.get('source_type', 'unknown')
    source_file = ds_config.get('source_file', 'unknown')
    return f"{source_type}:{source_file}"


def group_datasets_by_skill_and_source(
    skill_labels: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
) -> dict[str, list[str]]:
    """Group dataset names by skill, ensuring they share the same source.
    
    Since different data sources (TinyBenchmarks vs Aggregated) have completely
    different model sets with no overlap, we can only run experiments within
    datasets from the same source.
    
    Returns: dict mapping "skill|source" -> list of dataset names
    """
    source_config = load_data_source_config()
    
    # First, group by skill
    skill_to_datasets = defaultdict(list)
    for _, row in skill_labels.iterrows():
        dataset_name = row['Dataset']
        if dataset_name not in datasets:
            continue
        for skill in row['all_skills']:
            skill_to_datasets[skill].append(dataset_name)
    
    # Now, sub-group by source within each skill
    valid_groups = {}
    
    for skill, ds_list in skill_to_datasets.items():
        if len(ds_list) < 2:
            continue
        
        # Group by source
        source_to_datasets = defaultdict(list)
        for ds_name in ds_list:
            source = get_dataset_source(ds_name, source_config)
            source_to_datasets[source].append(ds_name)
        
        # Create experiment groups for sources with 2+ datasets
        for source, source_ds_list in source_to_datasets.items():
            if len(source_ds_list) >= 2:
                group_key = f"{skill}|{source}"
                valid_groups[group_key] = source_ds_list
    
    return valid_groups


def group_datasets_by_skill(
    skill_labels: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
) -> dict[str, list[str]]:
    """Group dataset names by skill, only including datasets with common models.
    
    This is a smarter version that finds actual model overlap.
    """
    skill_to_datasets = defaultdict(list)
    
    for _, row in skill_labels.iterrows():
        dataset_name = row['Dataset']
        if dataset_name not in datasets:
            continue
        for skill in row['all_skills']:
            skill_to_datasets[skill].append(dataset_name)
    
    # For each skill, find datasets that actually share models
    valid_skills = {}
    
    for skill, ds_list in skill_to_datasets.items():
        if len(ds_list) < 2:
            continue
        
        # Get model sets for each dataset
        model_sets = {
            ds: set(datasets[ds]['model_name'].unique())
            for ds in ds_list
        }
        
        # Find the largest subset of datasets that share common models
        # Start by trying all datasets, then remove one at a time
        best_subset = []
        
        # Try all pairs first
        for i, ds1 in enumerate(ds_list):
            for ds2 in ds_list[i+1:]:
                common = model_sets[ds1] & model_sets[ds2]
                if len(common) >= 4:
                    # Found a valid pair, try to extend it
                    subset = [ds1, ds2]
                    shared_models = common
                    
                    for ds3 in ds_list:
                        if ds3 not in subset:
                            new_common = shared_models & model_sets[ds3]
                            if len(new_common) >= 4:
                                subset.append(ds3)
                                shared_models = new_common
                    
                    if len(subset) > len(best_subset):
                        best_subset = subset
        
        if len(best_subset) >= 2:
            valid_skills[skill] = best_subset
    
    return valid_skills


def group_all_datasets_together(
    datasets: dict[str, pd.DataFrame],
    min_common_models: int = 4,
) -> dict[str, list[str]]:
    """Group ALL datasets together (ignoring skills) if they share common models.
    
    This finds the largest subset of all datasets that share at least min_common_models.
    
    Returns: dict with single key "All_Datasets" -> list of dataset names
    """
    # IMPORTANT: Sort for deterministic order across runs
    all_ds_names = sorted(datasets.keys())
    
    if len(all_ds_names) < 2:
        return {}
    
    # Get model sets for each dataset
    model_sets = {
        ds: set(datasets[ds]['model_name'].unique())
        for ds in all_ds_names
    }
    
    print(f"\n   Finding common models across {len(all_ds_names)} datasets...")
    
    # Start with all datasets and iteratively find the best subset
    # First, try all datasets together
    all_common = set.intersection(*model_sets.values()) if model_sets else set()
    
    if len(all_common) >= min_common_models:
        print(f"   ✓ All {len(all_ds_names)} datasets share {len(all_common)} common models")
        return {"All_Datasets": all_ds_names}
    
    # Find the largest subset that shares enough models
    # Greedy approach: start with the pair with most common models
    best_subset = []
    best_common_count = 0
    
    # Try all pairs as starting points
    for i, ds1 in enumerate(all_ds_names):
        for ds2 in all_ds_names[i+1:]:
            common = model_sets[ds1] & model_sets[ds2]
            if len(common) < min_common_models:
                continue
            
            # Try to extend this pair
            subset = [ds1, ds2]
            shared_models = common
            
            # Add datasets that maintain enough overlap
            for ds3 in all_ds_names:
                if ds3 not in subset:
                    new_common = shared_models & model_sets[ds3]
                    if len(new_common) >= min_common_models:
                        subset.append(ds3)
                        shared_models = new_common
            
            # Check if this is the best subset so far
            if len(subset) > len(best_subset) or \
               (len(subset) == len(best_subset) and len(shared_models) > best_common_count):
                best_subset = subset
                best_common_count = len(shared_models)
    
    if len(best_subset) >= 2:
        # IMPORTANT: Sort for deterministic order across runs
        best_subset = sorted(best_subset)
        print(f"   ✓ Found subset of {len(best_subset)} datasets with {best_common_count} common models")
        print(f"   Datasets: {best_subset}")
        return {"All_Datasets": best_subset}
    
    print(f"   ✗ No valid subset found with at least {min_common_models} common models")
    return {}


# =============================================================================

# =============================================================================
# IRT Training & Validation
# =============================================================================

def load_irt_params_from_cache(output_dir: Path) -> tuple[pd.DataFrame | None, np.ndarray | None, np.ndarray | None]:
    """Try to load IRT parameters from cached files.
    
    Returns: (item_params, A_matrix, B_matrix) or (None, None, None) if not cached
    """
    # Check for item_params.parquet first (saved by save_item_parameters)
    parquet_path = output_dir / "item_params.parquet"
    meta_path = output_dir / "item_params.meta.json"
    
    # Also check for irt_dataset_final.jsonlines (created during training)
    jsonlines_path = output_dir / "irt_dataset_final.jsonlines"
    
    if parquet_path.exists():
        try:
            item_params = pd.read_parquet(parquet_path)
            
            # Load metadata if exists
            A_matrix = None
            B_matrix = None
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                item_params.attrs = meta
                
                if 'A_matrix' in meta and 'B_matrix' in meta:
                    A_matrix = np.array(meta['A_matrix'])
                    B_matrix = np.array(meta['B_matrix'])
            
            return item_params, A_matrix, B_matrix
        except Exception as e:
            print(f"      Warning: Failed to load cached params: {e}")
    
    return None, None, None


def train_irt_on_base(
    train_base_df: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path,
    force_retrain: bool = False,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None]:
    """Train IRT model on base datasets (with caching).
    
    Returns: (item_params, A_matrix, B_matrix)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load from cache first
    if not force_retrain:
        item_params, A_matrix, B_matrix = load_irt_params_from_cache(output_dir)
        if item_params is not None:
            print(f"      ✓ Loaded cached IRT params from {output_dir.name}")
            return item_params, A_matrix, B_matrix
    
    # Train new IRT model
    irt_config = TrainingConfig(
        dims_search=config.dims_search,
        epochs=config.epochs,
        lr=config.lr,
        number_item_per_scenario=config.n_anchors_per_dataset,
        deterministic=True,
        filter_zero_variance=getattr(config, 'filter_zero_variance', True),
        validate_dimensions=getattr(config, 'validate_dimensions', True),
    )
    
    item_params = fit_2pl_parameters(
        train_base_df,
        config=irt_config,
        output_dir=str(output_dir),
    )
    
    # Extract matrices
    A_matrix = None
    B_matrix = None
    if hasattr(item_params, 'attrs') and item_params.attrs:
        A_list = item_params.attrs.get('A_matrix')
        B_list = item_params.attrs.get('B_matrix')
        if A_list is not None and B_list is not None:
            A_matrix = np.array(A_list)
            B_matrix = np.array(B_list)
    
    # Save for future caching
    save_item_parameters(item_params, str(output_dir / "item_params.parquet"))
    
    return item_params, A_matrix, B_matrix


def build_anchor_items_for_fixed_calibration(
    baseline_params: pd.DataFrame,
    available_questions: set[str],
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    selected_anchor_ids: list[str] | None = None,
) -> list[dict]:
    """Build anchor items from baseline parameters for Fixed-Anchor Calibration.
    
    These anchors are passed to train_item_parameters to FREEZE the Base item
    parameters while training Link items on the same scale.
    
    Args:
        baseline_params: DataFrame with IRT parameters indexed by question_id
        available_questions: Set of question IDs available in the combined dataset
        A_matrix: Full discrimination matrix, shape (1, D, n_items) or (D, n_items)
        B_matrix: Full difficulty matrix, shape (1, D, n_items) or (D, n_items)
        selected_anchor_ids: If provided, only freeze these specific items (faster training).
                            If None, freeze all items in baseline_params (original behavior).
    
    Returns:
        List of anchor item dicts with either vector or scalar parameters
    """
    baseline_params = baseline_params.copy()
    baseline_params.index = baseline_params.index.astype(str)
    
    # Filter to selected anchors if provided
    if selected_anchor_ids is not None:
        selected_set = set(str(a) for a in selected_anchor_ids)
        filter_to = available_questions & selected_set
    else:
        filter_to = available_questions
    
    subset = baseline_params.loc[baseline_params.index.intersection(filter_to)]
    
    if subset.empty:
        raise ValueError("No overlap between baseline item params and current matrix for anchoring")
    
    baseline_qids = list(baseline_params.index)
    anchors = []
    
    for item_id in subset.index:
        anchor = {"item_id": item_id}
        
        # Try to use vector parameters if available (MIRT)
        if A_matrix is not None and B_matrix is not None:
            try:
                base_idx = baseline_qids.index(item_id)
                # Handle both (1, D, n_items) and (D, n_items) shapes
                if A_matrix.ndim == 3:
                    anchor["discrimination_vector"] = A_matrix[0, :, base_idx].tolist()
                    anchor["difficulty_vector"] = B_matrix[0, :, base_idx].tolist()
                else:
                    anchor["discrimination_vector"] = A_matrix[:, base_idx].tolist()
                    anchor["difficulty_vector"] = B_matrix[:, base_idx].tolist()
            except (ValueError, IndexError):
                # Fall back to scalar if vector extraction fails
                anchor["difficulty"] = float(subset.loc[item_id, "b"])
                anchor["discrimination"] = float(subset.loc[item_id, "a"])
        else:
            # Use scalar parameters
            anchor["difficulty"] = float(subset.loc[item_id, "b"])
            anchor["discrimination"] = float(subset.loc[item_id, "a"])
        
        anchors.append(anchor)
    
    return anchors


def select_anchors_for_dataset(
    item_params: pd.DataFrame,
    n_anchors: int,
    dataset_name: str,
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    method: str = "irt_clustering",
) -> tuple[list[str], list[float]]:
    """Select anchor items from a SPECIFIC dataset.
    
    Args:
        item_params: DataFrame with IRT parameters indexed by question_id
        n_anchors: Number of anchors to select
        dataset_name: Name of the dataset to select anchors from
        train_df: Training data to identify which questions belong to which dataset
        A_matrix, B_matrix: MIRT matrices for clustering
        method: Anchor selection method (irt_clustering | top_k_discrimination | correctness_clustering)
    
    Returns:
        (anchor_ids, anchor_weights) for the specified dataset
    """
    # Get dataset for each question
    question_to_dataset = train_df.groupby('question_id')['dataset'].first().to_dict()
    
    # Filter item_params to only the specified dataset
    item_params_with_dataset = item_params.copy()
    item_params_with_dataset['dataset'] = item_params_with_dataset.index.map(
        lambda q: question_to_dataset.get(q, 'unknown')
    )
    
    ds_mask = item_params_with_dataset['dataset'] == dataset_name
    ds_items = item_params_with_dataset[ds_mask].drop(columns=['dataset'])
    
    if len(ds_items) == 0:
        print(f"      Warning: No items found for dataset {dataset_name}")
        return [], []
    
    n_anchors = min(n_anchors, len(ds_items))
    
    if n_anchors < 5:
        print(f"      Warning: {dataset_name} has only {len(ds_items)} items, need at least 5")
        return [], []
    
    # For top_k_discrimination, skip the expensive O(n²) MIRT index lookup entirely —
    # the method only needs item_params["a"] and doesn't use A/B matrices at all.
    if method == "top_k_discrimination":
        anchor_config = AnchorConfig(number_items=n_anchors, method=method)
        try:
            from irt.anchors import find_anchor_items_top_k_discrimination
            anchor_ids = [str(q) for q in find_anchor_items_top_k_discrimination(ds_items, anchor_config)]
            uniform_w = [1.0 / max(len(anchor_ids), 1)] * len(anchor_ids)
            print(f"      ✓ {dataset_name}: {len(anchor_ids)} anchors selected (method={method})")
            return anchor_ids, uniform_w
        except Exception as e:
            print(f"      Warning: Failed top_k_discrimination for {dataset_name}: {e}")
            return [], []
    
    # For clustering methods: build MIRT sub-matrices (O(n²) index lookup but needed for K-means)
    all_question_ids = list(item_params.index)
    ds_indices = [all_question_ids.index(q) for q in ds_items.index if q in all_question_ids]
    
    # Extract sub-matrices for this dataset
    ds_A = A_matrix[:, :, ds_indices] if A_matrix is not None else None
    ds_B = B_matrix[:, :, ds_indices] if B_matrix is not None else None
    
    # Copy attrs to subset
    ds_items_for_clustering = ds_items.copy()
    if hasattr(item_params, 'attrs'):
        ds_items_for_clustering.attrs = item_params.attrs.copy()
        if 'balance_weights' in item_params.attrs:
            orig_weights = np.array(item_params.attrs['balance_weights'])
            ds_weights = orig_weights[ds_indices]
            ds_items_for_clustering.attrs['balance_weights'] = ds_weights.tolist()
    
    balance_weights = None
    if hasattr(ds_items_for_clustering, 'attrs'):
        bw = ds_items_for_clustering.attrs.get('balance_weights')
        if bw is not None:
            balance_weights = np.array(bw)
    
    anchor_config = AnchorConfig(
        number_items=n_anchors,
        method=method,
        balance_weights=balance_weights,
    )
    
    try:
        anchor_ids, anchor_weights = find_anchor_items_clustering(
            ds_items_for_clustering,
            config=anchor_config,
            A_matrix=ds_A,
            B_matrix=ds_B,
        )
        
        weights_list = anchor_weights.tolist() if hasattr(anchor_weights, 'tolist') else list(anchor_weights)
        print(f"      ✓ {dataset_name}: {len(anchor_ids)} anchors selected (method={method})")
        return anchor_ids, weights_list
        
    except Exception as e:
        print(f"      Warning: Failed to select anchors from {dataset_name}: {e}")
        return [], []


def select_anchors_pooled(
    item_params: pd.DataFrame,
    n_anchors: int,
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
) -> tuple[list[str], list[float]]:
    """Select anchor items from ALL datasets combined (pooled).
    
    Instead of selecting N anchors per dataset, this selects N anchors
    from the entire pool using IRT clustering on all questions together.
    
    Args:
        item_params: DataFrame with IRT parameters indexed by question_id
        n_anchors: Total number of anchors to select from the combined pool
        train_df: Training data (used for balance weights if available)
        A_matrix, B_matrix: MIRT matrices for clustering
    
    Returns:
        (anchor_ids, anchor_weights) from the combined pool
    """
    if len(item_params) == 0:
        print("      Warning: No items in item_params for pooled selection")
        return [], []
    
    n_anchors = min(n_anchors, len(item_params))
    
    if n_anchors < 5:
        print(f"      Warning: Only {len(item_params)} items available, need at least 5")
        return [], []
    
    # Get balance weights if available
    balance_weights = None
    if hasattr(item_params, 'attrs'):
        bw = item_params.attrs.get('balance_weights')
        if bw is not None:
            balance_weights = np.array(bw)
    
    anchor_config = AnchorConfig(
        number_items=n_anchors,
        method="irt_clustering",
        balance_weights=balance_weights,
    )
    
    try:
        anchor_ids, anchor_weights = find_anchor_items_clustering(
            item_params,
            config=anchor_config,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
        )
        
        weights_list = anchor_weights.tolist() if hasattr(anchor_weights, 'tolist') else list(anchor_weights)
        print(f"      ✓ Pooled selection: {len(anchor_ids)} anchors selected from {len(item_params)} total items")
        return anchor_ids, weights_list
        
    except Exception as e:
        print(f"      Warning: Failed to select pooled anchors: {e}")
        return [], []


def precompute_thetas_from_all_anchors(
    test_df: pd.DataFrame,
    item_params: pd.DataFrame,
    anchor_ids: list[str],
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    """Precompute theta for each test model using ALL anchors across all datasets.
    
    This allows cross-dataset theta estimation: we use anchors from Base datasets
    to estimate theta, then use that theta to predict on Link datasets.
    
    Args:
        test_df: Test data containing responses for all models on all datasets
        item_params: DataFrame with IRT parameters indexed by question_id
        anchor_ids: List of anchor question IDs (can be from multiple datasets)
        A_matrix, B_matrix: MIRT matrices
    
    Returns:
        Dict mapping model_name -> estimated theta
    """
    models = test_df['model_name'].unique()
    question_ids_order = list(item_params.index) if hasattr(item_params, 'index') else None
    
    precomputed_thetas = {}
    n_success = 0
    n_failed = 0
    
    for model_name in models:
        # Get all responses for this model
        model_responses = test_df[test_df['model_name'] == model_name].set_index('question_id')['normalized_score']
        model_responses = model_responses[~model_responses.index.duplicated(keep='first')]
        
        # Find anchors that have responses
        available_anchors = [a for a in anchor_ids if a in model_responses.index and a in item_params.index]
        
        if len(available_anchors) < 3:
            n_failed += 1
            continue
        
        anchor_responses = model_responses.loc[available_anchors]
        
        try:
            theta = estimate_theta_from_anchors(
                item_params,
                anchor_responses,
                A_matrix=A_matrix,
                B_matrix=B_matrix,
                question_ids_order=question_ids_order,
            )
            precomputed_thetas[model_name] = theta
            n_success += 1
        except Exception as e:
            n_failed += 1
    
    print(f"      Precomputed thetas: {n_success} success, {n_failed} failed")
    return precomputed_thetas


def select_anchors(
    item_params: pd.DataFrame,
    n_anchors_per_dataset: int,
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    clustering_method: str = "irt_clustering",
) -> tuple[list[str], list[float]]:
    """Select anchor items using clustering - n_anchors PER dataset.

    Args:
        item_params: DataFrame with IRT parameters indexed by question_id
        n_anchors_per_dataset: Number of anchors to select from EACH dataset
        train_df: Training data to identify which questions belong to which dataset
        A_matrix, B_matrix: MIRT matrices for clustering
        clustering_method: "irt_clustering" or "correctness_clustering"

    Returns:
        Combined anchor_ids and weights from all datasets
    """
    # Get dataset for each question
    question_to_dataset = train_df.groupby('question_id')['dataset'].first().to_dict()

    # Group item_params by dataset
    item_params_with_dataset = item_params.copy()
    item_params_with_dataset['dataset'] = item_params_with_dataset.index.map(
        lambda q: question_to_dataset.get(q, 'unknown')
    )

    # IMPORTANT: Sort for deterministic order across runs
    datasets = sorted(item_params_with_dataset['dataset'].unique())

    all_anchor_ids = []
    all_anchor_weights = []

    for dataset in datasets:
        if dataset == 'unknown':
            continue

        # Get items for this dataset
        ds_mask = item_params_with_dataset['dataset'] == dataset
        ds_items = item_params_with_dataset[ds_mask].drop(columns=['dataset'])

        if len(ds_items) == 0:
            continue

        # How many anchors to select from this dataset
        n_anchors = min(n_anchors_per_dataset, len(ds_items))

        if n_anchors < 5:
            print(f"      Warning: {dataset} has only {len(ds_items)} items, skipping")
            continue

        # Fast path for top_k_discrimination: skip O(n²) MIRT index lookup
        if clustering_method == "top_k_discrimination":
            from irt.anchors import find_anchor_items_top_k_discrimination
            _cfg = AnchorConfig(number_items=n_anchors, method="top_k_discrimination")
            try:
                anchor_ids = [str(q) for q in find_anchor_items_top_k_discrimination(ds_items, _cfg)]
                anchor_weights = [1.0 / max(len(anchor_ids), 1)] * len(anchor_ids)
                all_anchor_ids.extend(anchor_ids)
                all_anchor_weights.extend(anchor_weights)
                print(f"      ✓ {dataset}: {len(anchor_ids)} anchors selected (method={clustering_method})")
            except Exception as e:
                print(f"      Warning: top_k_discrimination failed for {dataset}: {e}")
            continue

        # Get indices for MIRT matrices (needed for clustering methods)
        all_question_ids = list(item_params.index)
        ds_indices = [all_question_ids.index(q) for q in ds_items.index if q in all_question_ids]

        # Extract sub-matrices for this dataset
        ds_A = A_matrix[:, :, ds_indices] if A_matrix is not None else None
        ds_B = B_matrix[:, :, ds_indices] if B_matrix is not None else None

        # Copy attrs to subset
        ds_items_for_clustering = ds_items.copy()
        if hasattr(item_params, 'attrs'):
            ds_items_for_clustering.attrs = item_params.attrs.copy()
            # Update balance weights for this subset
            if 'balance_weights' in item_params.attrs:
                orig_weights = np.array(item_params.attrs['balance_weights'])
                ds_weights = orig_weights[ds_indices]
                ds_items_for_clustering.attrs['balance_weights'] = ds_weights.tolist()

        balance_weights = None
        if hasattr(ds_items_for_clustering, 'attrs'):
            bw = ds_items_for_clustering.attrs.get('balance_weights')
            if bw is not None:
                balance_weights = np.array(bw)

        # For correctness_clustering, prepare matrix_df
        matrix_df = None
        if clustering_method == 'correctness_clustering':
            # Filter to this dataset in long format expected by anchor selector
            ds_train_df = train_df[train_df['dataset'] == dataset].copy()
            matrix_df = ds_train_df[['question_id', 'model_name', 'normalized_score']].copy()

        anchor_config = AnchorConfig(
            number_items=n_anchors,
            method=clustering_method,
            balance_weights=balance_weights,
        )

        try:
            anchor_ids, anchor_weights = find_anchor_items_clustering(
                ds_items_for_clustering,
                matrix_df=matrix_df,
                config=anchor_config,
                A_matrix=ds_A,
                B_matrix=ds_B,
            )

            all_anchor_ids.extend(anchor_ids)
            weights_list = anchor_weights.tolist() if hasattr(anchor_weights, 'tolist') else list(anchor_weights)
            all_anchor_weights.extend(weights_list)

            print(f"      ✓ {dataset}: {len(anchor_ids)} anchors selected")

        except Exception as e:
            print(f"      Warning: Failed to select anchors from {dataset}: {e}")

    return all_anchor_ids, all_anchor_weights


def run_validation(
    test_df: pd.DataFrame,
    item_params: pd.DataFrame,
    anchor_ids: list[str],
    anchor_weights: list[float],
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    precomputed_thetas: dict[str, float] | None = None,
) -> list[dict]:
    """Run estimation validation.
    
    Args:
        test_df: Test data to evaluate on
        item_params: IRT item parameters
        anchor_ids: List of anchor question IDs
        anchor_weights: Weights for anchor questions
        train_df: Training data (used for computing lambdas)
        A_matrix, B_matrix: MIRT matrices
        precomputed_thetas: Optional dict mapping model_name -> theta.
            If provided, these thetas are used instead of estimating from local anchors.
            This enables cross-dataset theta estimation.
    """
    # Compute lambda values
    attrs = getattr(item_params, 'attrs', {})
    validation_errors = attrs.get('validation_errors', {})
    best_dim = attrs.get('best_dimension', 5)
    dims_search = attrs.get('config_dims_search', [5, 10])
    best_dim_idx = dims_search.index(best_dim) if best_dim in dims_search else 0
    
    # Get unique datasets in test_df
    datasets_in_test = test_df['dataset'].unique()
    
    # Build anchors and lambdas per dataset
    anchors_by_dataset = {}
    anchor_weights_by_dataset = {}
    
    for ds in datasets_in_test:
        # Filter anchors to those in this dataset
        ds_anchors = [a for a in anchor_ids if a.startswith(f"{ds}:")]
        if ds_anchors:
            anchors_by_dataset[ds] = ds_anchors
            # Get corresponding weights
            indices = [anchor_ids.index(a) for a in ds_anchors]
            anchor_weights_by_dataset[ds] = [anchor_weights[i] for i in indices]
        else:
            # Use all anchors (cross-dataset prediction)
            anchors_by_dataset[ds] = anchor_ids
            anchor_weights_by_dataset[ds] = anchor_weights
    
    lambdas_by_dataset = compute_lambda_values(
        original_matrix_df=train_df,
        validation_errors=validation_errors,
        best_dim_idx=best_dim_idx,
        number_item=len(anchor_ids),
    )
    
    question_ids_order = list(item_params.index) if hasattr(item_params, 'index') else None
    
    results = run_estimation_validation(
        test_matrix=test_df,
        item_params=item_params,
        anchors_by_dataset=anchors_by_dataset,
        lambdas_by_dataset=lambdas_by_dataset,
        anchor_weights_by_dataset=anchor_weights_by_dataset,
        precomputed_thetas=precomputed_thetas,
        A_matrix=A_matrix,
        B_matrix=B_matrix,
        question_ids_order=question_ids_order,
    )
    
    return results


def run_random_baseline_validation(
    test_df: pd.DataFrame,
    item_params: pd.DataFrame,
    n_random_questions: int,
    target_name: str,
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    precomputed_thetas: dict[str, float] | None = None,
    n_seeds: int = 1,
    base_seed: int = 42,
    return_per_model: bool = False,
) -> dict | tuple[dict, pd.DataFrame]:
    """Run validation using randomly selected questions instead of IRT-selected anchors.
    
    This provides a baseline to compare against the IRT anchor selection method.
    By running multiple seeds, we get variance estimates for the random baseline.
    
    Args:
        test_df: Test data to evaluate on
        item_params: IRT item parameters
        n_random_questions: Number of random questions to select (same as n_anchors)
        target_name: Name of the target dataset to select random questions from
        train_df: Training data
        A_matrix, B_matrix: MIRT matrices
        precomputed_thetas: Optional dict mapping model_name -> theta
        n_seeds: Number of random seeds to run (default 10)
        base_seed: Base seed for reproducibility
        return_per_model: If True, also return per-model results DataFrame
    
    Returns:
        If return_per_model=False: Dict with aggregated statistics
        If return_per_model=True: Tuple of (aggregated_dict, per_model_df)
        
        aggregated_dict contains:
        {
            'random_anchor_error_mean': float,
            'random_anchor_error_std': float,
            'random_gp_irt_error_mean': float,
            'random_gp_irt_error_std': float,
            'n_seeds': int,
            'n_random_questions': int,
        }
        
        per_model_df contains per-model results averaged across seeds:
        - model_name
        - random_anchor_error, random_gp_irt_error, etc.
    """
    # Get all questions from target dataset that have IRT parameters
    target_questions = [q for q in item_params.index if q.startswith(f"{target_name}:")]
    
    if len(target_questions) < n_random_questions:
        print(f"      Warning: Only {len(target_questions)} questions available, using all")
        n_random_questions = len(target_questions)
    
    if len(target_questions) < 10:
        print(f"      Warning: Too few questions ({len(target_questions)}) for random baseline, skipping")
        return {}
    
    # Collect results from each seed
    all_seed_results = {
        'anchor_error': [],
        'irt_error': [],
        'gp_irt_error': [],
        'pirt_error': [],
    }
    
    # Collect per-model results across all seeds (for averaging)
    per_model_results = {}  # model_name -> {metric -> [values across seeds]}
    
    # Get lambda values (needed for validation)
    attrs = getattr(item_params, 'attrs', {})
    validation_errors = attrs.get('validation_errors', {})
    best_dim = attrs.get('best_dimension', 5)
    dims_search = attrs.get('config_dims_search', [5, 10])
    best_dim_idx = dims_search.index(best_dim) if best_dim in dims_search else 0
    
    question_ids_order = list(item_params.index) if hasattr(item_params, 'index') else None
    
    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        np.random.seed(seed)
        
        # Randomly select questions
        random_anchors = list(np.random.choice(target_questions, size=n_random_questions, replace=False))
        
        # Assign uniform weights
        random_weights = [1.0 / n_random_questions] * n_random_questions
        
        # Build anchors dict
        anchors_by_dataset = {target_name: random_anchors}
        anchor_weights_by_dataset = {target_name: random_weights}
        
        # Compute lambdas for this anchor count
        lambdas_by_dataset = compute_lambda_values(
            original_matrix_df=train_df,
            validation_errors=validation_errors,
            best_dim_idx=best_dim_idx,
            number_item=n_random_questions,
        )
        
        # Run validation (silently)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Suppress print statements during random validation
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                results = run_estimation_validation(
                    test_matrix=test_df,
                    item_params=item_params,
                    anchors_by_dataset=anchors_by_dataset,
                    lambdas_by_dataset=lambdas_by_dataset,
                    anchor_weights_by_dataset=anchor_weights_by_dataset,
                    precomputed_thetas=precomputed_thetas,
                    A_matrix=A_matrix,
                    B_matrix=B_matrix,
                    question_ids_order=question_ids_order,
                )
            finally:
                sys.stdout = old_stdout
        
        # Aggregate results from this seed
        if results:
            for metric in ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']:
                vals = [r[metric] for r in results if not np.isnan(r.get(metric, np.nan))]
                if vals:
                    all_seed_results[metric].append(np.mean(vals))
            
            # Collect per-model results
            if return_per_model:
                for r in results:
                    model_name = r['model_name']
                    if model_name not in per_model_results:
                        per_model_results[model_name] = {
                            'anchor_error': [], 'irt_error': [], 
                            'gp_irt_error': [], 'pirt_error': [],
                            'true_performance': [], 'anchor_prediction': [],
                            'gp_irt_prediction': [],
                        }
                    for metric in per_model_results[model_name].keys():
                        val = r.get(metric, np.nan)
                        if not np.isnan(val):
                            per_model_results[model_name][metric].append(val)
    
    # Compute statistics across seeds
    output = {
        'n_seeds': n_seeds,
        'n_random_questions': n_random_questions,
    }
    
    for metric in ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']:
        vals = all_seed_results[metric]
        if vals:
            output[f'random_{metric}_mean'] = float(np.mean(vals))
            output[f'random_{metric}_std'] = float(np.std(vals))
    
    n_successful_seeds = len(all_seed_results['anchor_error'])
    print(f"      Random baseline: {n_successful_seeds}/{n_seeds} seeds successful")
    if n_successful_seeds > 0:
        print(f"         anchor_error: {output.get('random_anchor_error_mean', 'N/A'):.4f} ± {output.get('random_anchor_error_std', 'N/A'):.4f}")
        print(f"         gp_irt_error: {output.get('random_gp_irt_error_mean', 'N/A'):.4f} ± {output.get('random_gp_irt_error_std', 'N/A'):.4f}")
    
    if return_per_model:
        # Build per-model DataFrame with averaged results across seeds
        per_model_rows = []
        for model_name, metrics in per_model_results.items():
            row = {'model_name': model_name, 'n_seeds': n_seeds}
            for metric, values in metrics.items():
                if values:
                    row[f'random_{metric}_mean'] = float(np.mean(values))
                    row[f'random_{metric}_std'] = float(np.std(values))
            per_model_rows.append(row)
        per_model_df = pd.DataFrame(per_model_rows) if per_model_rows else pd.DataFrame()
        return output, per_model_df
    
    return output


def run_discriminative_baseline_validation(
    test_df: pd.DataFrame,
    item_params: pd.DataFrame,
    n_anchors: int,
    target_name: str,
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    precomputed_thetas: dict[str, float] | None = None,
    return_per_model: bool = False,
) -> dict | tuple[dict, pd.DataFrame]:
    """Run validation using top-K most discriminative items as anchors.
    
    This provides a baseline to compare against IRT clustering.
    Unlike random baseline, this is deterministic for a given set of item parameters.
    
    Args:
        test_df: Test data to evaluate on
        item_params: IRT item parameters
        n_anchors: Number of discriminative items to select
        target_name: Name of the target dataset
        train_df: Training data
        A_matrix, B_matrix: MIRT matrices
        precomputed_thetas: Optional dict mapping model_name -> theta
        return_per_model: If True, also return per-model results DataFrame
    """
    # Get all questions from target dataset that have IRT parameters
    target_questions = [q for q in item_params.index if q.startswith(f"{target_name}:")]
    
    if len(target_questions) < n_anchors:
        print(f"      Warning: Only {len(target_questions)} questions available, using all")
        n_anchors = len(target_questions)
    
    if len(target_questions) < 5:
        print(f"      Warning: Too few questions ({len(target_questions)}) for discriminative baseline, skipping")
        return {}
    
    # Select top-K items by discrimination parameter
    # For MIRT, we use the norm of the discrimination vector (a_i)
    target_params = item_params.loc[target_questions].copy()
    if 'discrimination' in target_params.columns:
        # 1PL/2PL/3PL
        target_params['disc_norm'] = target_params['discrimination']
    elif any(col.startswith('a_') for col in target_params.columns):
        # MIRT
        a_cols = [col for col in target_params.columns if col.startswith('a_')]
        target_params['disc_norm'] = np.linalg.norm(target_params[a_cols].values, axis=1)
    else:
        # Fallback to first column if no standard names found
        target_params['disc_norm'] = target_params.iloc[:, 0]
        
    top_k_questions = target_params.sort_values('disc_norm', ascending=False).head(n_anchors).index.tolist()
    
    # Assign uniform weights
    weights = [1.0 / n_anchors] * n_anchors
    
    # Build anchors dict
    anchors_by_dataset = {target_name: top_k_questions}
    anchor_weights_by_dataset = {target_name: weights}
    
    # Get lambda values (needed for validation)
    attrs = getattr(item_params, 'attrs', {})
    validation_errors = attrs.get('validation_errors', {})
    best_dim = attrs.get('best_dimension', 5)
    dims_search = attrs.get('config_dims_search', [5, 10])
    best_dim_idx = dims_search.index(best_dim) if best_dim in dims_search else 0
    
    question_ids_order = list(item_params.index) if hasattr(item_params, 'index') else None
    
    # Compute lambdas for this anchor count
    lambdas_by_dataset = compute_lambda_values(
        original_matrix_df=train_df,
        validation_errors=validation_errors,
        best_dim_idx=best_dim_idx,
        number_item=n_anchors,
    )
    
    # Run validation
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            results = run_estimation_validation(
                test_matrix=test_df,
                item_params=item_params,
                anchors_by_dataset=anchors_by_dataset,
                lambdas_by_dataset=lambdas_by_dataset,
                anchor_weights_by_dataset=anchor_weights_by_dataset,
                precomputed_thetas=precomputed_thetas,
                A_matrix=A_matrix,
                B_matrix=B_matrix,
                question_ids_order=question_ids_order,
            )
        finally:
            sys.stdout = old_stdout
            
    if not results:
        return {}
        
    # Aggregate results
    output = {
        'n_anchors': n_anchors,
    }
    
    for metric in ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']:
        vals = [r[metric] for r in results if not np.isnan(r.get(metric, np.nan))]
        if vals:
            output[f'discriminative_{metric}_mean'] = float(np.mean(vals))
            output[f'discriminative_{metric}_std'] = float(np.std(vals))
            
    print(f"      Discriminative baseline ({n_anchors} anchors):")
    print(f"         anchor_error: {output.get('discriminative_anchor_error_mean', 'N/A'):.4f}")
    print(f"         gp_irt_error: {output.get('discriminative_gp_irt_error_mean', 'N/A'):.4f}")
    
    if return_per_model:
        per_model_rows = []
        for r in results:
            model_name = r['model_name']
            row = {'model_name': model_name}
            for metric in ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error', 'true_performance', 'anchor_prediction', 'gp_irt_prediction']:
                val = r.get(metric, np.nan)
                if not np.isnan(val):
                    row[f'discriminative_{metric}_mean'] = float(val)
            per_model_rows.append(row)
        per_model_df = pd.DataFrame(per_model_rows) if per_model_rows else pd.DataFrame()
        return output, per_model_df
        
    return output


def run_random_simple_baseline(
    test_df: pd.DataFrame,
    target_name: str,
    n_random_questions: int,
    n_seeds: int = 1,
    base_seed: int = 42,
    return_per_model: bool = False,
) -> dict | tuple[dict, pd.DataFrame]:
    """Run a simple random baseline: predict performance using just the average of random questions.
    
    This is the simplest possible baseline - no IRT model at all:
    - Select n_random_questions randomly from the target dataset
    - prediction = mean(model's responses to those questions)
    - true_performance = mean(model's responses to ALL questions)
    - error = |prediction - true_performance|
    
    This shows what you get from random sampling without any sophisticated modeling.
    
    Args:
        test_df: Test data with columns ['model_name', 'question_id', 'normalized_score'/'correct', 'dataset']
        target_name: Name of the target dataset
        n_random_questions: Number of random questions to sample
        n_seeds: Number of random seeds to run
        base_seed: Base seed for reproducibility
        return_per_model: If True, also return per-model results DataFrame
    
    Returns:
        If return_per_model=False: Dict with aggregated statistics
        If return_per_model=True: Tuple of (aggregated_dict, per_model_df)
        
        aggregated_dict contains:
        {
            'simple_random_error_mean': float,  # Average |prediction - true| across models and seeds
            'simple_random_error_std': float,   # Std of errors across seeds
            'simple_random_prediction_mean': float,  # Average prediction
            'simple_random_true_perf_mean': float,   # Average true performance
            'n_seeds': int,
            'n_random_questions': int,
        }
        
        per_model_df contains per-model results averaged across seeds
    """
    # Filter to target dataset only
    target_df = test_df[test_df['dataset'] == target_name].copy()
    
    if len(target_df) == 0:
        print(f"      Warning: No data for target '{target_name}' in test_df, skipping simple random baseline")
        return {}
    
    # Determine score column (could be 'correct' or 'normalized_score')
    score_col = 'correct' if 'correct' in target_df.columns else 'normalized_score'
    
    # Get all unique questions
    all_questions = target_df['question_id'].unique()
    
    if len(all_questions) < n_random_questions:
        print(f"      Warning: Only {len(all_questions)} questions available, using all")
        n_random_questions = len(all_questions)
    
    if len(all_questions) < 10:
        print(f"      Warning: Too few questions ({len(all_questions)}) for simple random baseline, skipping")
        return {}
    
    # Get all test models
    test_models = target_df['model_name'].unique()
    
    # Compute true performance for each model (average over ALL questions)
    true_perf_by_model = target_df.groupby('model_name')[score_col].mean().to_dict()
    
    # Collect errors from each seed
    all_seed_errors = []
    all_seed_predictions = []
    
    # Collect per-model results across all seeds (for averaging)
    per_model_results = {}  # model_name -> {metric -> [values across seeds]}
    
    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        np.random.seed(seed)
        
        # Randomly select questions
        random_questions = np.random.choice(all_questions, size=n_random_questions, replace=False)
        
        # Filter to only selected questions
        random_df = target_df[target_df['question_id'].isin(random_questions)]
        
        # Compute prediction for each model (average over random questions)
        pred_by_model = random_df.groupby('model_name')[score_col].mean().to_dict()
        
        # Compute errors
        seed_errors = []
        seed_predictions = []
        for model in test_models:
            if model in pred_by_model and model in true_perf_by_model:
                pred = pred_by_model[model]
                true_perf = true_perf_by_model[model]
                error = abs(pred - true_perf)
                seed_errors.append(error)
                seed_predictions.append(pred)
                
                # Collect per-model results
                if return_per_model:
                    if model not in per_model_results:
                        per_model_results[model] = {
                            'error': [], 'prediction': [], 'true_performance': []
                        }
                    per_model_results[model]['error'].append(error)
                    per_model_results[model]['prediction'].append(pred)
                    per_model_results[model]['true_performance'].append(true_perf)
        
        if seed_errors:
            all_seed_errors.append(np.mean(seed_errors))
            all_seed_predictions.append(np.mean(seed_predictions))
    
    # Compute statistics across seeds
    output = {
        'n_seeds': n_seeds,
        'n_random_questions': n_random_questions,
    }
    
    if all_seed_errors:
        output['simple_random_error_mean'] = float(np.mean(all_seed_errors))
        output['simple_random_error_std'] = float(np.std(all_seed_errors))
        output['simple_random_prediction_mean'] = float(np.mean(all_seed_predictions))
        output['simple_random_true_perf_mean'] = float(np.mean(list(true_perf_by_model.values())))
        
        n_successful = len(all_seed_errors)
        print(f"      Simple random baseline: {n_successful}/{n_seeds} seeds successful")
        print(f"         simple_random_error: {output['simple_random_error_mean']:.4f} ± {output['simple_random_error_std']:.4f}")
    else:
        print(f"      Simple random baseline: 0/{n_seeds} seeds successful")
    
    if return_per_model:
        # Build per-model DataFrame with averaged results across seeds
        per_model_rows = []
        for model_name, metrics in per_model_results.items():
            row = {'model_name': model_name, 'n_seeds': n_seeds}
            for metric, values in metrics.items():
                if values:
                    row[f'simple_random_{metric}_mean'] = float(np.mean(values))
                    row[f'simple_random_{metric}_std'] = float(np.std(values))
            per_model_rows.append(row)
        per_model_df = pd.DataFrame(per_model_rows) if per_model_rows else pd.DataFrame()
        return output, per_model_df
    
    return output


# =============================================================================
