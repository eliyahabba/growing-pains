"""Dataset loading, configuration, and grouping for chain calibration experiments."""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

# Project root (src/experiments/equating/cross_dataset_equating.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


