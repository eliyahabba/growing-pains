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
# Data Splitting
# =============================================================================

def split_models(
    df: pd.DataFrame,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Split models into train and test sets."""
    np.random.seed(seed)
    # IMPORTANT: Sort models to ensure deterministic order across runs
    # Without sorting, different DataFrame row orders lead to different splits
    models = sorted(df['model_name'].unique())
    n_test = max(1, int(len(models) * test_ratio))
    test_models = set(np.random.choice(models, size=n_test, replace=False))
    train_models = set(models) - test_models
    return train_models, test_models


def create_leave_one_out_splits(
    skill: str,
    datasets: dict[str, pd.DataFrame],
    dataset_names: list[str],
    test_ratio: float = 0.25,
    seed: int = 42,
) -> list[dict]:
    """Create leave-one-out splits for a skill.
    
    For each dataset in the skill:
    - That dataset becomes "Link"
    - All other datasets become "Base"
    - Models are split into train/test
    
    Returns list of split configs, each containing:
    - link_dataset: name of the held-out dataset
    - base_datasets: list of other dataset names
    - train_base_df, test_base_df: DataFrames for base datasets
    - train_link_df, test_link_df: DataFrames for link dataset
    """
    splits = []
    
    # Combine all datasets to find common models
    all_dfs = [datasets[name] for name in dataset_names]
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Get models that appear in ALL datasets
    models_per_dataset = {
        name: set(datasets[name]['model_name'].unique())
        for name in dataset_names
    }
    common_models = set.intersection(*models_per_dataset.values())
    
    if len(common_models) < 4:
        print(f"  Warning: Only {len(common_models)} common models for skill '{skill}', need at least 4")
        return []
    
    # Split common models
    train_models, test_models = split_models(
        combined[combined['model_name'].isin(common_models)],
        test_ratio=test_ratio,
        seed=seed,
    )
    
    # Create leave-one-out splits
    for link_dataset in dataset_names:
        base_datasets = [d for d in dataset_names if d != link_dataset]
        
        # Filter to common models only
        link_df = datasets[link_dataset][
            datasets[link_dataset]['model_name'].isin(common_models)
        ].copy()
        
        base_dfs = [
            datasets[d][datasets[d]['model_name'].isin(common_models)].copy()
            for d in base_datasets
        ]
        base_df = pd.concat(base_dfs, ignore_index=True)
        
        # Split by train/test models
        train_base_df = base_df[base_df['model_name'].isin(train_models)].copy()
        test_base_df = base_df[base_df['model_name'].isin(test_models)].copy()
        train_link_df = link_df[link_df['model_name'].isin(train_models)].copy()
        test_link_df = link_df[link_df['model_name'].isin(test_models)].copy()
        
        splits.append({
            'skill': skill,
            'link_dataset': link_dataset,
            'base_datasets': base_datasets,
            'train_base_df': train_base_df,
            'test_base_df': test_base_df,
            'train_link_df': train_link_df,
            'test_link_df': test_link_df,
            'n_train_models': len(train_models),
            'n_test_models': len(test_models),
            'n_common_models': len(common_models),
        })
    
    return splits


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


def select_anchors_comparison(
    item_params: pd.DataFrame,
    n_anchors_per_dataset: int,
    train_df: pd.DataFrame,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
    """Select anchor items using BOTH clustering methods for comparison - n_anchors PER dataset.

    Args:
        item_params: DataFrame with IRT parameters indexed by question_id
        n_anchors_per_dataset: Number of anchors to select from EACH dataset
        train_df: Training data to identify which questions belong to which dataset
        A_matrix, B_matrix: MIRT matrices for clustering

    Returns:
        Dict with method names as keys, containing (anchor_ids, anchor_weights) for each method
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

    # Results for both methods
    results = {
        'irt_clustering': {'anchor_ids': [], 'anchor_weights': []},
        'correctness_clustering': {'anchor_ids': [], 'anchor_weights': []}
    }

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

        # Get indices for MIRT matrices
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

        # Test both clustering methods
        for method in ['irt_clustering', 'correctness_clustering']:
            try:
                anchor_config = AnchorConfig(
                    number_items=n_anchors,
                    method=method,
                    balance_weights=balance_weights,
                )

                # For correctness_clustering, we need the actual response matrix
                matrix_df = None
                if method == 'correctness_clustering':
                    # Filter to this dataset in long format expected by anchor selector
                    ds_train_df = train_df[train_df['dataset'] == dataset].copy()
                    matrix_df = ds_train_df[['question_id', 'model_name', 'normalized_score']].copy()

                anchor_ids, anchor_weights = find_anchor_items_clustering(
                    ds_items_for_clustering,
                    matrix_df=matrix_df,  # Only used for correctness_clustering
                    config=anchor_config,
                    A_matrix=ds_A,
                    B_matrix=ds_B,
                )

                results[method]['anchor_ids'].extend(anchor_ids)
                weights_list = anchor_weights.tolist() if hasattr(anchor_weights, 'tolist') else list(anchor_weights)
                results[method]['anchor_weights'].extend(weights_list)

                print(f"      ✓ {dataset} ({method}): {len(anchor_ids)} anchors selected")

            except Exception as e:
                print(f"      Warning: Failed to select anchors from {dataset} ({method}): {e}")

    # Return as (anchor_ids_dict, anchor_weights_dict)
    return (
        {method: data['anchor_ids'] for method, data in results.items()},
        {method: data['anchor_weights'] for method, data in results.items()}
    )


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
# Main Experiment
# =============================================================================

def run_single_split_experiment(
    split: dict,
    config: ExperimentConfig,
    output_dir: Path,
) -> dict:
    """Run experiment on a single leave-one-out split.
    
    Returns dict with:
    - Base→Base validation results
    - Base→Link validation results (the key test!)
    - Metadata
    """
    skill = split['skill']
    link_dataset = split['link_dataset']
    base_datasets = split['base_datasets']
    
    split_dir = output_dir / skill / f"link_{link_dataset.replace(' ', '_')}"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Check cache - but only skip if results are COMPLETE
    results_file = split_dir / "results.json"
    if results_file.exists() and not config.force_retrain:
        with open(results_file) as f:
            cached_results = json.load(f)
        
        # Check if results have both concurrent AND fixed results
        has_concurrent = 'link_concurrent_gp_irt_error_mean' in cached_results
        has_fixed = 'link_fixed_gp_irt_error_mean' in cached_results
        
        if has_concurrent and has_fixed:
            print(f"    Loading complete cached results for {link_dataset}")
            return cached_results
        else:
            print(f"    Incomplete cached results for {link_dataset} - continuing...")
            print(f"      (has_concurrent={has_concurrent}, has_fixed={has_fixed})")
    
    print(f"    Training IRT on Base datasets: {base_datasets}")
    
    # 1. Train IRT on Base
    item_params, A_matrix, B_matrix = train_irt_on_base(
        split['train_base_df'],
        config,
        split_dir / "irt",
        force_retrain=config.force_retrain,
    )
    
    print(f"      Trained {len(item_params)} items, best_dim={item_params.attrs.get('best_dimension', '?')}")
    
    # 2. Select anchors from Base items (n_anchors PER dataset)
    print(f"    Selecting {config.n_anchors_per_dataset} anchors per dataset...")
    anchor_ids, anchor_weights = select_anchors(
        item_params, 
        config.n_anchors_per_dataset, 
        split['train_base_df'],
        A_matrix, 
        B_matrix
    )
    print(f"      Total anchors selected: {len(anchor_ids)}")
    
    # 3. Validate on Base (internal consistency)
    print(f"    Validating Base→Base...")
    base_results = run_validation(
        test_df=split['test_base_df'],
        item_params=item_params,
        anchor_ids=anchor_ids,
        anchor_weights=anchor_weights,
        train_df=split['train_base_df'],
        A_matrix=A_matrix,
        B_matrix=B_matrix,
    )
    
    # 4. Validate on Link (cross-dataset prediction - THE KEY TEST)
    print(f"    Calibrating Link dataset: {link_dataset}")
    
    # Combine Base + Link for calibration training data
    train_combined = pd.concat([split['train_base_df'], split['train_link_df']], ignore_index=True)
    available_questions = set(train_combined['question_id'].astype(str).unique())
    
    # =========================================================================
    # 4a. CONCURRENT CALIBRATION - Retrain everything from scratch
    # =========================================================================
    print(f"      Running Concurrent Calibration...")
    item_params_concurrent, A_concurrent, B_concurrent = train_irt_on_base(
        train_combined,
        config,
        split_dir / "irt_concurrent",
        force_retrain=config.force_retrain,
    )
    
    # Select anchors from Link dataset (now calibrated)
    print(f"      Selecting anchors from Link dataset ({link_dataset})...")
    link_anchor_ids_conc, link_anchor_weights_conc = select_anchors_for_dataset(
        item_params_concurrent,
        config.n_anchors_per_dataset,
        link_dataset,
        train_combined,
        A_concurrent,
        B_concurrent,
    )
    
    # Combine Base anchors + Link anchors for theta estimation
    combined_anchor_ids_conc = anchor_ids + link_anchor_ids_conc
    combined_anchor_weights_conc = anchor_weights + link_anchor_weights_conc
    print(f"      Combined anchors: {len(anchor_ids)} (Base) + {len(link_anchor_ids_conc)} (Link) = {len(combined_anchor_ids_conc)}")
    
    # Combine test data from Base + Link for theta estimation
    test_combined = pd.concat([split['test_base_df'], split['test_link_df']], ignore_index=True)
    
    # Precompute thetas using ALL anchors (Base + Link) from ALL test data
    print(f"      Precomputing thetas from all anchors (Concurrent)...")
    precomputed_thetas_conc = precompute_thetas_from_all_anchors(
        test_df=test_combined,
        item_params=item_params_concurrent,
        anchor_ids=combined_anchor_ids_conc,
        A_matrix=A_concurrent,
        B_matrix=B_concurrent,
    )
    
    link_results_concurrent = run_validation(
        test_df=split['test_link_df'],
        item_params=item_params_concurrent,
        anchor_ids=combined_anchor_ids_conc,  # Use Base + Link anchors
        anchor_weights=combined_anchor_weights_conc,
        train_df=train_combined,
        A_matrix=A_concurrent,
        B_matrix=B_concurrent,
        precomputed_thetas=precomputed_thetas_conc,
    )
    
    # =========================================================================
    # 4b. FIXED-ANCHOR CALIBRATION - Keep Base items fixed, train only Link
    # =========================================================================
    print(f"      Running Fixed-Anchor Calibration...")
    
    # Build anchor items from Base parameters to FREEZE them
    # If anchor_only_fixed=True, only freeze selected anchors (faster training)
    # If anchor_only_fixed=False, freeze all Base items (original behavior)
    anchor_items = build_anchor_items_for_fixed_calibration(
        item_params,
        available_questions,
        A_matrix,
        B_matrix,
        selected_anchor_ids=anchor_ids if config.anchor_only_fixed else None,
    )
    mode_str = "selected anchors only" if config.anchor_only_fixed else "all Base items"
    print(f"        Using {len(anchor_items)} anchor items from Base (frozen, {mode_str})")
    
    # Determine dimension - MUST match anchor vectors from Base training
    if A_matrix is not None:
        base_dim = A_matrix.shape[1] if A_matrix.ndim == 3 else A_matrix.shape[0]
        fixed_dims_search = [base_dim]
        print(f"        Using dimension {base_dim} from Base A_matrix (no search)")
    else:
        fixed_dims_search = config.dims_search
        print(f"        ⚠ No A_matrix, using dims_search={fixed_dims_search}")
    
    # Train with fixed anchors
    irt_config_fixed = TrainingConfig(
        dims_search=fixed_dims_search,
        epochs=config.epochs,
        lr=config.lr,
        number_item_per_scenario=config.n_anchors_per_dataset,
        deterministic=True,
    )
    
    item_params_fixed = train_item_parameters(
        train_combined,
        test_matrix_df=split['test_link_df'],
        config=irt_config_fixed,
        output_dir=str(split_dir / "irt_fixed_anchor"),
        anchor_items=anchor_items,
    )
    
    # Extract matrices from fixed-anchor results
    A_fixed = None
    B_fixed = None
    if hasattr(item_params_fixed, 'attrs') and item_params_fixed.attrs:
        A_list = item_params_fixed.attrs.get('A_matrix')
        B_list = item_params_fixed.attrs.get('B_matrix')
        if A_list is not None and B_list is not None:
            A_fixed = np.array(A_list)
            B_fixed = np.array(B_list)
    
    # Select anchors from Link dataset (now calibrated via fixed-anchor)
    print(f"      Selecting anchors from Link dataset ({link_dataset}) for Fixed-Anchor...")
    link_anchor_ids_fixed, link_anchor_weights_fixed = select_anchors_for_dataset(
        item_params_fixed,
        config.n_anchors_per_dataset,
        link_dataset,
        train_combined,
        A_fixed,
        B_fixed,
    )
    
    # Combine Base anchors + Link anchors for theta estimation
    combined_anchor_ids_fixed = anchor_ids + link_anchor_ids_fixed
    combined_anchor_weights_fixed = anchor_weights + link_anchor_weights_fixed
    print(f"      Combined anchors: {len(anchor_ids)} (Base) + {len(link_anchor_ids_fixed)} (Link) = {len(combined_anchor_ids_fixed)}")
    
    # Precompute thetas using ALL anchors (Base + Link) from ALL test data
    print(f"      Precomputing thetas from all anchors (Fixed-Anchor)...")
    precomputed_thetas_fixed = precompute_thetas_from_all_anchors(
        test_df=test_combined,  # Reuse the combined test data
        item_params=item_params_fixed,
        anchor_ids=combined_anchor_ids_fixed,
        A_matrix=A_fixed,
        B_matrix=B_fixed,
    )
    
    link_results_fixed = run_validation(
        test_df=split['test_link_df'],
        item_params=item_params_fixed,
        anchor_ids=combined_anchor_ids_fixed,  # Use Base + Link anchors
        anchor_weights=combined_anchor_weights_fixed,
        train_df=train_combined,
        A_matrix=A_fixed,
        B_matrix=B_fixed,
        precomputed_thetas=precomputed_thetas_fixed,
    )
    
    # Note: item_params are saved automatically by fit_2pl_parameters() when output_dir is provided
    
    # =========================================================================
    # Compile results
    # =========================================================================
    ERROR_METRICS = ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']
    
    def summarize_results(results: list[dict], prefix: str) -> dict:
        if not results:
            return {}
        df = pd.DataFrame(results)
        summary = {f'{prefix}_n_validations': len(df)}
        for metric in ERROR_METRICS:
            if metric in df.columns:
                vals = df[metric].dropna()
                if len(vals) > 0:
                    summary[f'{prefix}_{metric}_mean'] = float(vals.mean())
                    summary[f'{prefix}_{metric}_std'] = float(vals.std())
        return summary
    
    def summarize_per_dataset(results: list[dict]) -> dict:
        """Compute per-dataset statistics for all error metrics."""
        if not results:
            return {}
        df = pd.DataFrame(results)
        if 'scenario_name' not in df.columns:
            return {}
        
        per_dataset = {}
        for scenario, group in df.groupby('scenario_name'):
            stats = {'n_models': len(group)}
            for metric in ERROR_METRICS:
                if metric in group.columns:
                    vals = group[metric].dropna()
                    if len(vals) > 0:
                        stats[f'{metric}_mean'] = float(vals.mean())
                        stats[f'{metric}_std'] = float(vals.std())
            per_dataset[scenario] = stats
        return per_dataset
    
    result = {
        'skill': skill,
        'link_dataset': link_dataset,
        'base_datasets': base_datasets,
        'n_train_models': split['n_train_models'],
        'n_test_models': split['n_test_models'],
        'n_common_models': split['n_common_models'],
        'n_base_items': len(item_params),
        'n_combined_items_concurrent': len(item_params_concurrent),
        'n_combined_items_fixed': len(item_params_fixed),
        'n_base_anchors': len(anchor_ids),
        'n_link_anchors_concurrent': len(link_anchor_ids_conc),
        'n_link_anchors_fixed': len(link_anchor_ids_fixed),
        'n_total_anchors_concurrent': len(combined_anchor_ids_conc),
        'n_total_anchors_fixed': len(combined_anchor_ids_fixed),
        'n_fixed_anchor_items': len(anchor_items),
        **summarize_results(base_results, 'base'),
        **summarize_results(link_results_concurrent, 'link_concurrent'),
        **summarize_results(link_results_fixed, 'link_fixed'),
        # Per-dataset breakdown
        'base_per_dataset': summarize_per_dataset(base_results),
        'link_concurrent_per_dataset': summarize_per_dataset(link_results_concurrent),
        'link_fixed_per_dataset': summarize_per_dataset(link_results_fixed),
    }
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(round_for_json(result), f, indent=2)
    
    # Save detailed results
    if base_results:
        round_df_for_save(pd.DataFrame(base_results)).to_csv(split_dir / "base_validation.csv", index=False)
    if link_results_concurrent:
        round_df_for_save(pd.DataFrame(link_results_concurrent)).to_csv(split_dir / "link_concurrent_validation.csv", index=False)
    if link_results_fixed:
        round_df_for_save(pd.DataFrame(link_results_fixed)).to_csv(split_dir / "link_fixed_validation.csv", index=False)
    
    return result


def run_cross_dataset_equating(config: Optional[ExperimentConfig] = None):
    """Run the full cross-dataset equating experiment."""
    if config is None:
        config = ExperimentConfig()
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Cross-Dataset Equating Experiment")
    print("=" * 70)
    print(f"Data Source Mode: {config.data_source_mode}")
    
    # Print mode-specific info
    mode_info = {
        "mixed": "Using data_source_config.json (various sources)",
        "helm_lite": "HELM Lite only (91 models, 9 datasets)",
        "helm_classic": "HELM Classic only (70 models, 30 datasets)",
        "lb_only": "Open LLM Leaderboard only (395 models, 6 datasets)",
        "lb": "Open LLM Leaderboard only (395 models, 6 datasets)",
        "reeval": "reeval dataset (183 models, 22 scenarios)",
    }
    print(f"   {mode_info.get(config.data_source_mode, 'Unknown mode')}")
    
    # 1. Load skill labels (only needed for mixed mode)
    if config.data_source_mode == "mixed":
        print("\n1. Loading skill labels...")
        skill_labels = load_skill_labels(config.skill_labels_csv)
        print(f"   Loaded {len(skill_labels)} datasets from skill labels")
    else:
        print("\n1. Skipping skill labels (using predefined datasets for mode)")
        skill_labels = None
    
    # 2. Load all datasets
    print(f"\n2. Loading datasets (mode: {config.data_source_mode})...")
    datasets = load_all_datasets(config)
    print(f"   Loaded {len(datasets)} datasets")
    
    # 3. Group datasets based on mode
    if config.all_datasets_mode or config.data_source_mode != "mixed":
        # For non-mixed modes or all_datasets_mode, combine all datasets together
        mode_name = config.data_source_mode if config.data_source_mode != "mixed" else "ALL_DATASETS"
        print(f"\n3. {mode_name.upper()} MODE - Combining all datasets together...")
        skill_to_datasets = group_all_datasets_together(datasets, min_common_models=4)
        
        if skill_to_datasets:
            for group_key, ds_list in skill_to_datasets.items():
                model_sets = [set(datasets[ds]['model_name'].unique()) for ds in ds_list]
                common = set.intersection(*model_sets) if model_sets else set()
                # Rename group to include mode name
                print(f"   ✓ {mode_name}: {len(ds_list)} datasets, {len(common)} common models")
                for ds in ds_list:
                    print(f"       - {ds}")
            # Rename the group key to the mode name
            skill_to_datasets = {mode_name: list(skill_to_datasets.values())[0]}
    else:
        print("\n3. Analyzing model overlap between datasets (by skill)...")
        
        # First show raw skill groupings
        raw_skill_groups = defaultdict(list)
        for _, row in skill_labels.iterrows():
            dataset_name = row['Dataset']
            if dataset_name not in datasets:
                continue
            for skill in row['all_skills']:
                raw_skill_groups[skill].append(dataset_name)
        
        print(f"   Raw skill groupings (before model overlap check):")
        for skill, ds_list in sorted(raw_skill_groups.items()):
            if len(ds_list) >= 2:
                print(f"     • {skill}: {ds_list}")
        
        # Analyze model overlap
        print(f"\n   Analyzing model overlap...")
        for skill, ds_list in sorted(raw_skill_groups.items()):
            if len(ds_list) < 2:
                continue
            print(f"\n   Skill: {skill}")
            for ds in ds_list:
                n_models = datasets[ds]['model_name'].nunique()
                sample_models = list(datasets[ds]['model_name'].unique()[:2])
                print(f"     - {ds}: {n_models} models (e.g., {sample_models[0][:40]}...)")
            
            # Check pairwise overlap
            for i, ds1 in enumerate(ds_list):
                for ds2 in ds_list[i+1:]:
                    m1 = set(datasets[ds1]['model_name'].unique())
                    m2 = set(datasets[ds2]['model_name'].unique())
                    common = m1 & m2
                    print(f"     {ds1} ∩ {ds2}: {len(common)} common models")
        
        # Now do actual grouping
        print(f"\n   Finding valid experiment groups...")
        skill_to_datasets = group_datasets_by_skill(skill_labels, datasets)
        
        if skill_to_datasets:
            print(f"   Found {len(skill_to_datasets)} valid skills with overlapping models:")
            for skill, ds_list in sorted(skill_to_datasets.items()):
                # Get common model count
                model_sets = [set(datasets[ds]['model_name'].unique()) for ds in ds_list]
                common = set.intersection(*model_sets)
                print(f"     • {skill}: {ds_list} ({len(common)} common models)")
        else:
            print("   ⚠️  No skills found with overlapping models across datasets!")
            print("   This usually happens when datasets come from different sources.")
            print("\n   Trying source-aware grouping...")
            
            # Try source-aware grouping
            skill_to_datasets = group_datasets_by_skill_and_source(skill_labels, datasets)
            
            if skill_to_datasets:
                print(f"   Found {len(skill_to_datasets)} valid skill+source groups:")
                for group_key, ds_list in sorted(skill_to_datasets.items()):
                    model_sets = [set(datasets[ds]['model_name'].unique()) for ds in ds_list]
                    common = set.intersection(*model_sets) if model_sets else set()
                    print(f"     • {group_key}: {ds_list} ({len(common)} common models)")
    
    # 4. Run experiments
    print("\n4. Running leave-one-out experiments...")
    all_results = []
    
    if not skill_to_datasets:
        print("   No valid experiment groups found!")
        return pd.DataFrame()
    
    for group_key, dataset_names in skill_to_datasets.items():
        # Handle both "skill" and "skill|source" formats
        skill = group_key.split('|')[0] if '|' in group_key else group_key
        print(f"\n  Group: {group_key} ({len(dataset_names)} datasets)")
        
        # Create splits
        splits = create_leave_one_out_splits(
            skill, datasets, dataset_names,
            test_ratio=config.test_ratio,
            seed=config.seed,
        )
        
        if not splits:
            print(f"    Skipping - insufficient common models")
            continue
        
        print(f"    Created {len(splits)} leave-one-out splits")
        print(f"    Common models: {splits[0]['n_common_models']} "
              f"(train: {splits[0]['n_train_models']}, test: {splits[0]['n_test_models']})")
        
        for split in splits:
            try:
                result = run_single_split_experiment(split, config, output_dir)
                all_results.append(result)
                
                # Print summary
                base_err = result.get('base_gp_irt_error_mean', float('nan'))
                link_concurrent = result.get('link_concurrent_gp_irt_error_mean', float('nan'))
                link_fixed = result.get('link_fixed_gp_irt_error_mean', float('nan'))
                print(f"      Link={split['link_dataset']}: "
                      f"Base={base_err:.4f}, Concurrent={link_concurrent:.4f}, Fixed={link_fixed:.4f}")
                
            except Exception as e:
                print(f"      Error with Link={split['link_dataset']}: {e}")
                import traceback
                traceback.print_exc()
    
    # 5. Save summary
    print("\n5. Saving summary...")
    results_df = pd.DataFrame(all_results)
    round_df_for_save(results_df).to_csv(output_dir / "all_results.csv", index=False)
    
    # 6. Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if not results_df.empty:
        print(f"\nTotal experiments: {len(results_df)}")
        print(f"Skills tested: {results_df['skill'].nunique()}")
        
        # Per-skill summary - comparing both methods
        print(f"\n{'Skill':<20} {'Base':<10} {'Concurrent':<12} {'Fixed':<10} {'Δ Conc.':<10} {'Δ Fixed':<10}")
        print("-" * 80)
        
        for skill in results_df['skill'].unique():
            skill_df = results_df[results_df['skill'] == skill]
            base_mean = skill_df['base_gp_irt_error_mean'].mean()
            concurrent_mean = skill_df['link_concurrent_gp_irt_error_mean'].mean()
            fixed_mean = skill_df['link_fixed_gp_irt_error_mean'].mean()
            delta_concurrent = concurrent_mean - base_mean
            delta_fixed = fixed_mean - base_mean
            
            skill_short = skill[:19] if len(skill) > 19 else skill
            print(f"{skill_short:<20} {base_mean:<10.4f} {concurrent_mean:<12.4f} {fixed_mean:<10.4f} "
                  f"{delta_concurrent:+.4f}     {delta_fixed:+.4f}")
        
        # Overall
        print("-" * 80)
        overall_base = results_df['base_gp_irt_error_mean'].mean()
        overall_concurrent = results_df['link_concurrent_gp_irt_error_mean'].mean()
        overall_fixed = results_df['link_fixed_gp_irt_error_mean'].mean()
        delta_concurrent = overall_concurrent - overall_base
        delta_fixed = overall_fixed - overall_base
        print(f"{'OVERALL':<20} {overall_base:<10.4f} {overall_concurrent:<12.4f} {overall_fixed:<10.4f} "
              f"{delta_concurrent:+.4f}     {delta_fixed:+.4f}")
        
        # Method comparison
        print(f"\n{'='*80}")
        print("METHOD COMPARISON: Fixed-Anchor vs Concurrent")
        print(f"{'='*80}")
        
        fixed_better = (results_df['link_fixed_gp_irt_error_mean'] < 
                       results_df['link_concurrent_gp_irt_error_mean']).sum()
        total = len(results_df)
        avg_diff = (results_df['link_fixed_gp_irt_error_mean'] - 
                   results_df['link_concurrent_gp_irt_error_mean']).mean()
        
        print(f"  Fixed-Anchor better in: {fixed_better}/{total} experiments ({100*fixed_better/total:.1f}%)")
        print(f"  Average difference (Fixed - Concurrent): {avg_diff:+.4f}")
        print(f"  Winner: {'Fixed-Anchor' if avg_diff < 0 else 'Concurrent'}")
    
    print(f"\nResults saved to: {output_dir}")
    return results_df


def analyze_dataset_role_impact(output_dir: str | Path) -> pd.DataFrame:
    """Analyze the impact of dataset role (Base vs Link) on prediction error.
    
    For each dataset:
    - Calculate average error when it was part of Base (across all experiments)
    - Calculate error when it was the Link dataset
    - Compare the two
    
    This reveals whether being in the initial IRT training (Base) vs being
    calibrated later (Link) affects prediction accuracy.
    
    Returns DataFrame with columns:
    - dataset: dataset name
    - skill: skill group
    - as_base_error_mean: average error when dataset was in Base
    - as_base_error_std: std of error when dataset was in Base
    - as_base_n_experiments: number of times it was in Base
    - as_link_concurrent_error: error when dataset was Link (concurrent calibration)
    - as_link_fixed_error: error when dataset was Link (fixed-anchor calibration)
    - delta_concurrent: difference (link - base) for concurrent method
    - delta_fixed: difference (link - base) for fixed method
    """
    output_dir = Path(output_dir)
    
    # Structure to collect results: dataset -> skill -> data
    dataset_results = defaultdict(lambda: defaultdict(lambda: {
        'as_base_errors': [],
        'as_link_concurrent': None,
        'as_link_fixed': None,
    }))
    
    # Scan all experiment results
    for skill_dir in output_dir.iterdir():
        if not skill_dir.is_dir() or skill_dir.name == '__pycache__':
            continue
        
        skill_name = skill_dir.name
        
        for link_dir in skill_dir.iterdir():
            if not link_dir.is_dir():
                continue
            
            # Load experiment metadata
            results_file = link_dir / "results.json"
            if not results_file.exists():
                continue
            
            with open(results_file) as f:
                metadata = json.load(f)
            
            link_dataset = metadata['link_dataset']
            base_datasets = metadata['base_datasets']
            
            # Load detailed validation results
            base_val_file = link_dir / "base_validation.csv"
            link_concurrent_file = link_dir / "link_concurrent_validation.csv"
            link_fixed_file = link_dir / "link_fixed_validation.csv"
            
            # Process base validation - calculate error per dataset
            if base_val_file.exists():
                base_val = pd.read_csv(base_val_file)
                
                # Group by scenario_name (which is the dataset name)
                for scenario_name, group in base_val.groupby('scenario_name'):
                    # Match scenario to base_datasets
                    matching_base = [d for d in base_datasets if d.startswith(scenario_name) or scenario_name.startswith(d)]
                    if matching_base:
                        ds_name = matching_base[0]
                    else:
                        ds_name = scenario_name
                    
                    # Calculate mean error for this dataset in this experiment
                    ds_error = group['gp_irt_error'].mean()
                    dataset_results[ds_name][skill_name]['as_base_errors'].append(ds_error)
            
            # Process Link validation results
            # Try CSV files first (newer format)
            if link_concurrent_file.exists():
                link_concurrent_val = pd.read_csv(link_concurrent_file)
                concurrent_error = link_concurrent_val['gp_irt_error'].mean()
                dataset_results[link_dataset][skill_name]['as_link_concurrent'] = concurrent_error
            elif 'link_concurrent_gp_irt_error_mean' in metadata:
                # Fallback to JSON for newer experiments
                dataset_results[link_dataset][skill_name]['as_link_concurrent'] = metadata['link_concurrent_gp_irt_error_mean']
            elif 'link_gp_irt_error_mean' in metadata:
                # Fallback to JSON for older experiments (single link method)
                dataset_results[link_dataset][skill_name]['as_link_concurrent'] = metadata['link_gp_irt_error_mean']
            
            if link_fixed_file.exists():
                link_fixed_val = pd.read_csv(link_fixed_file)
                fixed_error = link_fixed_val['gp_irt_error'].mean()
                dataset_results[link_dataset][skill_name]['as_link_fixed'] = fixed_error
            elif 'link_fixed_gp_irt_error_mean' in metadata:
                # Fallback to JSON for newer experiments
                dataset_results[link_dataset][skill_name]['as_link_fixed'] = metadata['link_fixed_gp_irt_error_mean']
    
    # Compile into DataFrame
    rows = []
    for dataset, skill_data in dataset_results.items():
        for skill, data in skill_data.items():
            base_errors = data['as_base_errors']
            
            if not base_errors and data['as_link_concurrent'] is None:
                continue
            
            row = {
                'dataset': dataset,
                'skill': skill,
                'as_base_error_mean': np.mean(base_errors) if base_errors else None,
                'as_base_error_std': np.std(base_errors) if len(base_errors) > 1 else None,
                'as_base_n_experiments': len(base_errors),
                'as_link_concurrent_error': data['as_link_concurrent'],
                'as_link_fixed_error': data['as_link_fixed'],
            }
            
            # Calculate deltas (positive = Link is worse than Base)
            if base_errors and data['as_link_concurrent'] is not None:
                row['delta_concurrent'] = data['as_link_concurrent'] - row['as_base_error_mean']
            else:
                row['delta_concurrent'] = None
            
            if base_errors and data['as_link_fixed'] is not None:
                row['delta_fixed'] = data['as_link_fixed'] - row['as_base_error_mean']
            else:
                row['delta_fixed'] = None
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def plot_role_impact_analysis(output_dir: str | Path, df: pd.DataFrame | None = None) -> list[Path]:
    """Create visualizations for role impact analysis.
    
    Generates:
    1. Bar chart comparing Base vs Link errors per dataset
    2. Delta (Link - Base) showing calibration penalty
    3. Per-skill summary boxplot
    
    Returns list of generated figure paths.
    """
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if df is None:
        df = analyze_dataset_role_impact(output_dir)
    
    if df.empty:
        print("No data for plotting")
        return []
    
    generated_figures = []
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')
    
    # =========================================================================
    # Figure 1: Bar chart comparing Base vs Link errors per dataset
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Prepare data
    plot_df = df.dropna(subset=['as_base_error_mean']).copy()
    has_link_data = False
    
    if 'as_link_concurrent_error' in plot_df.columns and plot_df['as_link_concurrent_error'].notna().any():
        has_link_data = True
    
    if not plot_df.empty:
        x = np.arange(len(plot_df))
        width = 0.25
        
        bars1 = ax.bar(x - width, plot_df['as_base_error_mean'], width, 
                       label='As Base (in initial IRT)', color='#4f6ad7', alpha=0.8)
        
        if has_link_data:
            concurrent_vals = plot_df['as_link_concurrent_error'].fillna(0)
            bars2 = ax.bar(x, concurrent_vals, width,
                           label='As Link (Concurrent)', color='#f4a259', alpha=0.8)
        
        if 'as_link_fixed_error' in plot_df.columns and plot_df['as_link_fixed_error'].notna().any():
            fixed_vals = plot_df['as_link_fixed_error'].fillna(0)
            bars3 = ax.bar(x + width, fixed_vals, width,
                           label='As Link (Fixed-Anchor)', color='#5aa469', alpha=0.8)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Mean gp-IRT Error')
        ax.set_title('Prediction Error by Dataset Role\n(Base = initial IRT training, Link = calibrated later)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['dataset'][:15]}\n({row['skill'][:10]})" 
                           for _, row in plot_df.iterrows()], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        fig.tight_layout()
        path1 = figures_dir / "role_impact_comparison.png"
        fig.savefig(path1, dpi=150)
        plt.close(fig)
        generated_figures.append(path1)
        print(f"  ✓ Generated: {path1}")
    
    # =========================================================================
    # Figure 2: Delta (Link - Base) showing calibration penalty
    # =========================================================================
    delta_df = df.dropna(subset=['delta_concurrent']).copy()
    
    if not delta_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        x = np.arange(len(delta_df))
        width = 0.35
        
        colors_c = ['#e74c3c' if d > 0 else '#27ae60' for d in delta_df['delta_concurrent']]
        bars1 = ax.bar(x - width/2, delta_df['delta_concurrent'], width,
                       label='Concurrent Calibration', color=colors_c, alpha=0.7)
        
        if 'delta_fixed' in delta_df.columns and delta_df['delta_fixed'].notna().any():
            delta_f_vals = delta_df['delta_fixed'].fillna(0)
            colors_f = ['#c0392b' if d > 0 else '#229954' for d in delta_f_vals]
            bars2 = ax.bar(x + width/2, delta_f_vals, width,
                           label='Fixed-Anchor Calibration', color=colors_f, alpha=0.7, hatch='//')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Δ Error (Link - Base)')
        ax.set_title('Calibration Penalty: Error Increase When Dataset is Link vs Base\n'
                     '(Positive = Link worse, Negative = Link better)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['dataset'][:12]}" for _, row in delta_df.iterrows()], 
                          rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        fig.tight_layout()
        path2 = figures_dir / "calibration_penalty.png"
        fig.savefig(path2, dpi=150)
        plt.close(fig)
        generated_figures.append(path2)
        print(f"  ✓ Generated: {path2}")
    
    # =========================================================================
    # Figure 3: Per-skill summary boxplot
    # =========================================================================
    skills = df['skill'].unique()
    if len(skills) > 1 and df['delta_concurrent'].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Concurrent deltas by skill
        skill_data_c = [df[df['skill'] == skill]['delta_concurrent'].dropna() 
                       for skill in skills]
        skill_labels = [s[:15] for s in skills]
        
        valid_skill_data_c = [(d, l) for d, l in zip(skill_data_c, skill_labels) if len(d) > 0]
        
        if valid_skill_data_c:
            bp1 = axes[0].boxplot([d.values for d, _ in valid_skill_data_c],
                                  labels=[l for _, l in valid_skill_data_c],
                                  patch_artist=True)
            for patch in bp1['boxes']:
                patch.set_facecolor('#f4a259')
                patch.set_alpha(0.6)
            axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0].set_ylabel('Δ Error (Link - Base)')
            axes[0].set_title('Concurrent Calibration\nby Skill')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Fixed deltas by skill
        if 'delta_fixed' in df.columns and df['delta_fixed'].notna().any():
            skill_data_f = [df[df['skill'] == skill]['delta_fixed'].dropna() 
                           for skill in skills]
            valid_skill_data_f = [(d, l) for d, l in zip(skill_data_f, skill_labels) if len(d) > 0]
            
            if valid_skill_data_f:
                bp2 = axes[1].boxplot([d.values for d, _ in valid_skill_data_f],
                                      labels=[l for _, l in valid_skill_data_f],
                                      patch_artist=True)
                for patch in bp2['boxes']:
                    patch.set_facecolor('#5aa469')
                    patch.set_alpha(0.6)
                axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                axes[1].set_ylabel('Δ Error (Link - Base)')
                axes[1].set_title('Fixed-Anchor Calibration\nby Skill')
                axes[1].tick_params(axis='x', rotation=45)
        
        fig.tight_layout()
        path3 = figures_dir / "skill_comparison_boxplot.png"
        fig.savefig(path3, dpi=150)
        plt.close(fig)
        generated_figures.append(path3)
        print(f"  ✓ Generated: {path3}")
    
    return generated_figures


def print_role_impact_analysis(output_dir: str | Path = None):
    """Print analysis of dataset role impact on prediction error."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data/cross_dataset_equating"
    
    df = analyze_dataset_role_impact(output_dir)
    
    if df.empty:
        print("No data found for role impact analysis")
        return df
    
    print("=" * 90)
    print("DATASET ROLE IMPACT ANALYSIS")
    print("=" * 90)
    print("\nComparing prediction error when dataset is in Base (initial IRT) vs Link (calibrated later)")
    print("Positive delta = Link has higher error (worse) than Base")
    print()
    
    # Overall summary
    valid_concurrent = df['delta_concurrent'].dropna()
    valid_fixed = df['delta_fixed'].dropna()
    
    print(f"{'Dataset':<20} {'Skill':<20} {'As Base':<10} {'As Link(C)':<12} {'As Link(F)':<12} {'Δ Conc.':<10} {'Δ Fixed':<10}")
    print("-" * 94)
    
    for _, row in df.sort_values(['skill', 'dataset']).iterrows():
        dataset_short = row['dataset'][:19] if len(row['dataset']) > 19 else row['dataset']
        skill_short = row['skill'][:19] if len(row['skill']) > 19 else row['skill']
        
        base_str = f"{row['as_base_error_mean']:.4f}" if row['as_base_error_mean'] is not None else "N/A"
        conc_str = f"{row['as_link_concurrent_error']:.4f}" if row['as_link_concurrent_error'] is not None else "N/A"
        fixed_str = f"{row['as_link_fixed_error']:.4f}" if row['as_link_fixed_error'] is not None else "N/A"
        delta_c_str = f"{row['delta_concurrent']:+.4f}" if row['delta_concurrent'] is not None else "N/A"
        delta_f_str = f"{row['delta_fixed']:+.4f}" if row['delta_fixed'] is not None else "N/A"
        
        print(f"{dataset_short:<20} {skill_short:<20} {base_str:<10} {conc_str:<12} {fixed_str:<12} {delta_c_str:<10} {delta_f_str:<10}")
    
    print("-" * 94)
    
    # Summary statistics
    print("\n📊 SUMMARY STATISTICS:")
    
    if len(valid_concurrent) > 0:
        print(f"\n  Concurrent Calibration:")
        print(f"    Mean Δ (Link - Base): {valid_concurrent.mean():+.4f}")
        print(f"    Median Δ:             {valid_concurrent.median():+.4f}")
        print(f"    Std Δ:                {valid_concurrent.std():.4f}")
        print(f"    Link worse in:        {(valid_concurrent > 0).sum()}/{len(valid_concurrent)} cases ({100*(valid_concurrent > 0).mean():.1f}%)")
    
    if len(valid_fixed) > 0:
        print(f"\n  Fixed-Anchor Calibration:")
        print(f"    Mean Δ (Link - Base): {valid_fixed.mean():+.4f}")
        print(f"    Median Δ:             {valid_fixed.median():+.4f}")
        print(f"    Std Δ:                {valid_fixed.std():.4f}")
        print(f"    Link worse in:        {(valid_fixed > 0).sum()}/{len(valid_fixed)} cases ({100*(valid_fixed > 0).mean():.1f}%)")
    
    # Per-skill summary
    print("\n📈 PER-SKILL SUMMARY:")
    for skill in df['skill'].unique():
        skill_df = df[df['skill'] == skill]
        skill_delta_c = skill_df['delta_concurrent'].dropna()
        skill_delta_f = skill_df['delta_fixed'].dropna()
        
        if len(skill_delta_c) > 0 or len(skill_delta_f) > 0:
            print(f"\n  {skill}:")
            if len(skill_delta_c) > 0:
                print(f"    Concurrent: mean Δ={skill_delta_c.mean():+.4f}, Link worse in {(skill_delta_c > 0).sum()}/{len(skill_delta_c)}")
            if len(skill_delta_f) > 0:
                print(f"    Fixed:      mean Δ={skill_delta_f.mean():+.4f}, Link worse in {(skill_delta_f > 0).sum()}/{len(skill_delta_f)}")
    
    # Save results
    output_path = Path(output_dir) / "role_impact_analysis.csv"
    round_df_for_save(df).to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Generate plots
    print("\n📊 Generating visualizations...")
    try:
        figures = plot_role_impact_analysis(output_dir, df)
        if figures:
            print(f"\n✅ Generated {len(figures)} figures in: {Path(output_dir) / 'figures'}")
    except Exception as e:
        print(f"\n⚠️  Could not generate plots: {e}")
    
    return df


def rebuild_all_results_csv(output_dir: str | Path) -> pd.DataFrame:
    """Rebuild all_results.csv from individual results.json files.
    
    This is useful if:
    - The all_results.csv is empty or corrupted
    - You want to aggregate results from experiments run at different times
    - The experiment was interrupted
    
    Returns the aggregated DataFrame.
    """
    output_dir = Path(output_dir)
    
    print(f"Scanning {output_dir} for results.json files...")
    
    results = []
    for skill_dir in output_dir.iterdir():
        if not skill_dir.is_dir() or skill_dir.name in ('__pycache__', 'figures'):
            continue
        for link_dir in skill_dir.iterdir():
            if not link_dir.is_dir():
                continue
            results_file = link_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        result = json.load(f)
                    results.append(result)
                    print(f"  ✓ {skill_dir.name}/{link_dir.name}")
                except Exception as e:
                    print(f"  ✗ {skill_dir.name}/{link_dir.name}: {e}")
    
    if not results:
        print("No results found!")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = output_dir / "all_results.csv"
    round_df_for_save(results_df).to_csv(output_file, index=False)
    print(f"\n✅ Saved {len(results)} results to: {output_file}")
    
    # Print column summary
    print(f"\nColumns in all_results.csv:")
    for col in results_df.columns:
        print(f"  - {col}")
    
    return results_df


def print_existing_results(output_dir: str | Path = None, force_rebuild: bool = False):
    """Print summary of existing results without running experiments.
    
    Args:
        output_dir: Directory containing experiment results
        force_rebuild: If True, rebuild all_results.csv from individual files
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data/cross_dataset_equating"
    else:
        output_dir = Path(output_dir)
    
    all_results_file = output_dir / "all_results.csv"
    
    if force_rebuild or not all_results_file.exists():
        if force_rebuild:
            print("Force rebuilding all_results.csv...")
        else:
            print(f"No results file found at {all_results_file}")
        
        results_df = rebuild_all_results_csv(output_dir)
        if results_df.empty:
            return None
    else:
        results_df = pd.read_csv(all_results_file)
    
    # Print summary
    print("=" * 80)
    print("EXISTING RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal experiments: {len(results_df)}")
    print(f"Skills: {results_df['skill'].unique().tolist()}")
    
    # Check what columns exist
    has_concurrent = 'link_concurrent_gp_irt_error_mean' in results_df.columns
    has_fixed = 'link_fixed_gp_irt_error_mean' in results_df.columns
    has_link = 'link_gp_irt_error_mean' in results_df.columns
    
    print(f"\nResults completeness:")
    print(f"  - Base results: ✅")
    print(f"  - Concurrent results: {'✅' if has_concurrent else '❌'}")
    print(f"  - Fixed-Anchor results: {'✅' if has_fixed else '❌'}")
    
    # Show what's available
    if has_concurrent and has_fixed:
        print(f"\n{'Skill':<20} {'Base':<10} {'Concurrent':<12} {'Fixed':<10}")
        print("-" * 55)
        
        for skill in results_df['skill'].unique():
            skill_df = results_df[results_df['skill'] == skill]
            base = skill_df['base_gp_irt_error_mean'].mean()
            conc = skill_df['link_concurrent_gp_irt_error_mean'].mean()
            fixed = skill_df['link_fixed_gp_irt_error_mean'].mean()
            skill_short = skill[:19] if len(skill) > 19 else skill
            print(f"{skill_short:<20} {base:<10.4f} {conc:<12.4f} {fixed:<10.4f}")
    elif has_link:
        print(f"\n{'Skill':<20} {'Base':<10} {'Link':<10}")
        print("-" * 42)
        
        for skill in results_df['skill'].unique():
            skill_df = results_df[results_df['skill'] == skill]
            base = skill_df['base_gp_irt_error_mean'].mean()
            link = skill_df['link_gp_irt_error_mean'].mean()
            skill_short = skill[:19] if len(skill) > 19 else skill
            print(f"{skill_short:<20} {base:<10.4f} {link:<10.4f}")
    else:
        print(f"\n{'Skill':<20} {'Base':<10}")
        print("-" * 32)
        
        for skill in results_df['skill'].unique():
            skill_df = results_df[results_df['skill'] == skill]
            base = skill_df['base_gp_irt_error_mean'].mean()
            skill_short = skill[:19] if len(skill) > 19 else skill
            print(f"{skill_short:<20} {base:<10.4f}")
    
    # List incomplete experiments
    incomplete = []
    for skill_dir in output_dir.iterdir():
        if not skill_dir.is_dir() or skill_dir.name == '__pycache__':
            continue
        for link_dir in skill_dir.iterdir():
            if not link_dir.is_dir():
                continue
            results_file = link_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    result = json.load(f)
                if 'link_fixed_gp_irt_error_mean' not in result:
                    incomplete.append(f"{skill_dir.name}/{link_dir.name}")
    
    if incomplete:
        print(f"\n⚠️  Incomplete experiments ({len(incomplete)}):")
        for exp in incomplete[:10]:
            print(f"   - {exp}")
        if len(incomplete) > 10:
            print(f"   ... and {len(incomplete) - 10} more")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Dataset Equating Experiment")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--n-anchors-per-dataset", type=int, default=100, 
                        help="Number of anchors to select from EACH dataset")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Force retrain all")
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 5], help="Dimensions to search")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--print-only", action="store_true", help="Only print existing results, don't run")
    parser.add_argument("--rebuild-csv", action="store_true", 
                        help="Rebuild all_results.csv from individual results.json files")
    parser.add_argument("--analyze-role", action="store_true", 
                        help="Analyze impact of dataset role (Base vs Link) on prediction error")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Combine ALL datasets together instead of grouping by skill")
    parser.add_argument("--anchor-only-fixed", action="store_true", default=True,
                        help="Only freeze selected anchors in fixed-anchor calibration (faster, default)")
    parser.add_argument("--freeze-all-base", action="store_true",
                        help="Freeze ALL Base items in fixed-anchor calibration (slower, original behavior)")
    parser.add_argument("--data-source-mode", type=str, default="mixed",
                        choices=["mixed", "helm_lite", "helm_classic", "lb_only", "lb", "reeval", "mmlu_split", "tinybenchmarks"],
                        help="Data source mode: 'mixed' (default, uses data_source_config.json), "
                             "'helm_lite' (91 models, 9 datasets), "
                             "'helm_classic' (70 models, 30 datasets), "
                             "'lb_only' (395 models, 6 datasets from Open LLM Leaderboard)")
    
    args = parser.parse_args()
    
    if args.rebuild_csv:
        rebuild_all_results_csv(args.output_dir or PROJECT_ROOT / "data/cross_dataset_equating")
    elif args.analyze_role:
        print_role_impact_analysis(args.output_dir)
    elif args.print_only:
        print_existing_results(args.output_dir, force_rebuild=False)
    else:
        # --freeze-all-base overrides --anchor-only-fixed
        anchor_only = not args.freeze_all_base
        
        config = ExperimentConfig(
            n_anchors_per_dataset=args.n_anchors_per_dataset,
            test_ratio=args.test_ratio,
            seed=args.seed,
            force_retrain=args.force,
            dims_search=args.dims,
            epochs=args.epochs,
            all_datasets_mode=args.all_datasets,
            anchor_only_fixed=anchor_only,
            data_source_mode=args.data_source_mode,
        )
        
        if args.output_dir:
            config.output_dir = Path(args.output_dir)
        
        run_cross_dataset_equating(config)

