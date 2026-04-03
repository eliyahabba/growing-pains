"""
Matrix data cleaning utilities - MANDATORY cleaning for evaluation quality.

This module provides functionality to clean evaluation matrices by removing
models and questions with insufficient coverage. This cleaning is MANDATORY
and runs automatically to ensure high-quality data.

The cleaning process includes:
1. Model filtering - keeps models in top 80% of coverage per dataset
2. Question filtering - keeps questions answered by top 80% of models  
3. Dataset filtering - removes datasets with insufficient size
4. Matrix completion - ensures all remaining models answer the same questions (complete matrix)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import json


@dataclass
class CleaningConfig:
    """Configuration for matrix cleaning - optimized defaults."""
    # Model filtering - keep models in top 80% of coverage per dataset
    min_model_coverage_percentile: float = 80.0
    
    # Question filtering - keep questions answered by top 80% of models  
    min_question_coverage_percentile: float = 80.0
    
    # Dataset filtering - minimum viable dataset size
    min_models_per_dataset: int = 5
    min_questions_per_dataset: int = 10
    
    # Strategy
    iterative_cleaning: bool = True
    max_iterations: int = 5


@dataclass 
class CleaningStats:
    """Statistics about the cleaning process."""
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    removed_models: List[str]
    removed_questions: List[str] 
    removed_datasets: List[str]
    coverage_stats_before: Dict[str, Dict]
    coverage_stats_after: Dict[str, Dict]
    iterations: int
    dataset_cleaning_details: Dict[str, Dict]


class MatrixCleaner:
    """MANDATORY matrix cleaner - ensures evaluation data quality."""
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        
    def clean(self, matrix_df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningStats]:
        """Clean the matrix - MANDATORY step for data quality."""
        if matrix_df.empty:
            raise ValueError("Cannot clean empty matrix")
            
        print("   🧹 Cleaning data (mandatory quality control)...")
        
        original_shape = matrix_df.shape
        
        # Calculate initial coverage stats (silent)
        coverage_stats_before = self._calculate_coverage_stats_silent(matrix_df)
        
        # Track what we remove
        removed_models = set()
        removed_questions = set()
        removed_datasets = set()
        dataset_details = {}
        
        df = matrix_df.copy()
        
        # Iterative cleaning (silent)
        df, cleaning_info = self._iterative_clean_silent(df)
        removed_models.update(cleaning_info['removed_models'])
        removed_questions.update(cleaning_info['removed_questions']) 
        removed_datasets.update(cleaning_info['removed_datasets'])
        dataset_details = cleaning_info['dataset_details']
        iterations = cleaning_info['iterations']
        
        # Calculate final coverage stats (silent)
        coverage_stats_after = self._calculate_coverage_stats_silent(df)
        
        cleaning_stats = CleaningStats(
            original_shape=original_shape,
            final_shape=df.shape,
            removed_models=list(removed_models),
            removed_questions=list(removed_questions),
            removed_datasets=list(removed_datasets),
            coverage_stats_before=coverage_stats_before,
            coverage_stats_after=coverage_stats_after,
            iterations=iterations,
            dataset_cleaning_details=dataset_details
        )
        
        self._print_compact_summary(cleaning_stats)
        
        return df, cleaning_stats
    
    def _iterative_clean_silent(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Clean iteratively without detailed logging."""
        removed_models = set()
        removed_questions = set()
        removed_datasets = set()
        dataset_details = {}
        
        for iteration in range(self.config.max_iterations):
            prev_shape = df.shape
            
            # Clean models, questions, datasets (silent)
            df, iter_removed_models, model_details = self._clean_models_silent(df)
            df, iter_removed_questions, question_details = self._clean_questions_silent(df)
            df, iter_removed_datasets, dataset_iter_details = self._clean_datasets_silent(df)
            
            # NEW: Ensure complete matrix (remove models with insufficient question coverage)
            df, iter_removed_matrix_models, matrix_details = self._ensure_complete_matrix_silent(df)
            
            removed_models.update(iter_removed_models)
            removed_models.update(iter_removed_matrix_models)  # Add matrix completion removals
            removed_questions.update(iter_removed_questions)
            removed_datasets.update(iter_removed_datasets)
            
            dataset_details[f"iteration_{iteration + 1}"] = {
                "model_details": model_details,
                "question_details": question_details,
                "dataset_details": dataset_iter_details,
                "matrix_completion_details": matrix_details
            }
            
            # Check for convergence
            if df.shape == prev_shape:
                break
        
        return df, {
            'removed_models': removed_models,
            'removed_questions': removed_questions,
            'removed_datasets': removed_datasets,
            'dataset_details': dataset_details,
            'iterations': iteration + 1
        }
    
    def _iterative_clean_with_detailed_logging(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Clean iteratively with detailed logging."""
        removed_models = set()
        removed_questions = set()
        removed_datasets = set()
        dataset_details = {}
        
        print(f"\n🔄 ITERATIVE CLEANING (max {self.config.max_iterations} iterations):")
        print("-" * 50)
        
        for iteration in range(self.config.max_iterations):
            print(f"\n📍 ITERATION {iteration + 1}:")
            print(f"   Starting with: {df.shape[0]:,} rows, {df['model_name'].nunique()} models, {df['question_id'].nunique()} questions")
            
            prev_shape = df.shape
            
            # Clean models with detailed logging
            df, iter_removed_models, model_details = self._clean_models_with_logging(df)
            removed_models.update(iter_removed_models)
            
            # Clean questions with detailed logging
            df, iter_removed_questions, question_details = self._clean_questions_with_logging(df)
            removed_questions.update(iter_removed_questions)
            
            # Clean datasets with detailed logging
            df, iter_removed_datasets, dataset_iter_details = self._clean_datasets_with_logging(df)
            removed_datasets.update(iter_removed_datasets)
            
            # NEW: Ensure complete matrix with detailed logging
            df, iter_removed_matrix_models, matrix_details = self._ensure_complete_matrix_with_logging(df)
            removed_models.update(iter_removed_matrix_models)
            
            # Store details for this iteration
            dataset_details[f"iteration_{iteration + 1}"] = {
                "model_details": model_details,
                "question_details": question_details,
                "dataset_details": dataset_iter_details,
                "matrix_completion_details": matrix_details
            }
            
            print(f"   Ending with: {df.shape[0]:,} rows, {df['model_name'].nunique()} models, {df['question_id'].nunique()} questions")
            
            # Check for convergence
            if df.shape == prev_shape:
                print(f"   ✅ CONVERGED - No more changes needed")
                break
            else:
                removed_rows = prev_shape[0] - df.shape[0]
                print(f"   🔄 Removed {removed_rows:,} rows this iteration, continuing...")
        else:
            print(f"   ⚠️  Reached maximum iterations ({self.config.max_iterations})")
        
        return df, {
            'removed_models': removed_models,
            'removed_questions': removed_questions,
            'removed_datasets': removed_datasets,
            'dataset_details': dataset_details,
            'iterations': iteration + 1
        }
    
    def _clean_models_with_logging(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
        """Remove models with insufficient coverage - with detailed logging."""
        if df.empty:
            return df, [], {}
            
        print(f"   🤖 CLEANING MODELS (keeping top {self.config.min_model_coverage_percentile}% coverage):")
        
        # Calculate questions per model per dataset
        model_coverage = df.groupby(['dataset', 'model_name'])['question_id'].nunique().reset_index()
        model_coverage.columns = ['dataset', 'model_name', 'questions_answered']
        
        models_to_keep = set()
        dataset_details = {}
        
        for dataset in sorted(df['dataset'].unique()):
            dataset_models = model_coverage[model_coverage['dataset'] == dataset]
            
            if len(dataset_models) == 0:
                continue
            
            # Get statistics
            total_models = len(dataset_models)
            coverage_values = dataset_models['questions_answered'].values
            
            # Calculate threshold
            threshold_questions = np.percentile(coverage_values, self.config.min_model_coverage_percentile)
            
            # Find models above threshold
            good_models = dataset_models[
                dataset_models['questions_answered'] >= threshold_questions
            ]
            
            models_to_keep.update(good_models['model_name'].tolist())
            
            # Detailed logging per dataset
            kept_count = len(good_models)
            removed_count = total_models - kept_count
            
            print(f"     📊 {dataset}:")
            print(f"        Total models: {total_models}")
            print(f"        Coverage range: {coverage_values.min()}-{coverage_values.max()} questions")
            print(f"        Threshold (p{self.config.min_model_coverage_percentile}): {threshold_questions:.1f} questions")
            print(f"        ✅ Kept: {kept_count} models")
            print(f"        🗑️  Removed: {removed_count} models")
            
            dataset_details[dataset] = {
                "total_models": total_models,
                "kept_models": kept_count,
                "removed_models": removed_count,
                "coverage_min": int(coverage_values.min()),
                "coverage_max": int(coverage_values.max()),
                "coverage_mean": float(coverage_values.mean()),
                "threshold": float(threshold_questions)
            }
        
        # Filter the dataframe
        original_models = set(df['model_name'].unique())
        df_filtered = df[df['model_name'].isin(models_to_keep)]
        removed_models = list(original_models - models_to_keep)
        
        if removed_models:
            print(f"   🗑️  TOTAL MODELS REMOVED: {len(removed_models)}")
            if len(removed_models) <= 10:
                print(f"      Removed models: {removed_models}")
            else:
                print(f"      First 10 removed: {removed_models[:10]}...")
        else:
            print(f"   ✅ NO MODELS REMOVED")
            
        return df_filtered, removed_models, dataset_details
    
    def _clean_models_silent(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
        """Remove models with insufficient coverage - silent version."""
        if df.empty:
            return df, [], {}
            
        # Calculate questions per model per dataset
        model_coverage = df.groupby(['dataset', 'model_name'])['question_id'].nunique().reset_index()
        model_coverage.columns = ['dataset', 'model_name', 'questions_answered']
        
        models_to_keep = set()
        dataset_details = {}
        
        for dataset in sorted(df['dataset'].unique()):
            dataset_models = model_coverage[model_coverage['dataset'] == dataset]
            
            if len(dataset_models) == 0:
                continue
            
            # Get statistics
            total_models = len(dataset_models)
            coverage_values = dataset_models['questions_answered'].values
            
            # Calculate threshold
            threshold_questions = np.percentile(coverage_values, self.config.min_model_coverage_percentile)
            
            # Find models above threshold
            good_models = dataset_models[
                dataset_models['questions_answered'] >= threshold_questions
            ]
            
            models_to_keep.update(good_models['model_name'].tolist())
            
            # Store details silently
            kept_count = len(good_models)
            removed_count = total_models - kept_count
            
            dataset_details[dataset] = {
                "total_models": total_models,
                "kept_models": kept_count,
                "removed_models": removed_count,
                "coverage_min": int(coverage_values.min()),
                "coverage_max": int(coverage_values.max()),
                "coverage_mean": float(coverage_values.mean()),
                "threshold": float(threshold_questions)
            }
        
        # Filter the dataframe
        original_models = set(df['model_name'].unique())
        df_filtered = df[df['model_name'].isin(models_to_keep)]
        removed_models = list(original_models - models_to_keep)
            
        return df_filtered, removed_models, dataset_details
    
    def _clean_questions_with_logging(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
        """Remove questions with insufficient model coverage - with detailed logging."""
        if df.empty:
            return df, [], {}
            
        print(f"   ❓ CLEANING QUESTIONS (keeping top {self.config.min_question_coverage_percentile}% coverage):")
        
        # Calculate models per question per dataset
        question_coverage = df.groupby(['dataset', 'question_id'])['model_name'].nunique().reset_index()
        question_coverage.columns = ['dataset', 'question_id', 'models_answered']
        
        questions_to_keep = set()
        dataset_details = {}
        
        for dataset in sorted(df['dataset'].unique()):
            dataset_questions = question_coverage[question_coverage['dataset'] == dataset]
            
            if len(dataset_questions) == 0:
                continue
            
            # Get statistics
            total_questions = len(dataset_questions)
            coverage_values = dataset_questions['models_answered'].values
            
            # Calculate threshold
            threshold_models = np.percentile(coverage_values, self.config.min_question_coverage_percentile)
            
            # Find questions above threshold
            good_questions = dataset_questions[
                dataset_questions['models_answered'] >= threshold_models
            ]
            
            questions_to_keep.update(good_questions['question_id'].tolist())
            
            # Detailed logging per dataset
            kept_count = len(good_questions)
            removed_count = total_questions - kept_count
            
            print(f"     📊 {dataset}:")
            print(f"        Total questions: {total_questions}")
            print(f"        Coverage range: {coverage_values.min()}-{coverage_values.max()} models")
            print(f"        Threshold (p{self.config.min_question_coverage_percentile}): {threshold_models:.1f} models")
            print(f"        ✅ Kept: {kept_count} questions")
            print(f"        🗑️  Removed: {removed_count} questions")
            
            dataset_details[dataset] = {
                "total_questions": total_questions,
                "kept_questions": kept_count,
                "removed_questions": removed_count,
                "coverage_min": int(coverage_values.min()),
                "coverage_max": int(coverage_values.max()),
                "coverage_mean": float(coverage_values.mean()),
                "threshold": float(threshold_models)
            }
        
        # Filter the dataframe
        original_questions = set(df['question_id'].unique())
        df_filtered = df[df['question_id'].isin(questions_to_keep)]
        removed_questions = list(original_questions - questions_to_keep)
        
        if removed_questions:
            print(f"   🗑️  TOTAL QUESTIONS REMOVED: {len(removed_questions)}")
        else:
            print(f"   ✅ NO QUESTIONS REMOVED")
            
        return df_filtered, removed_questions, dataset_details
    
    def _clean_questions_silent(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
        """Remove questions with insufficient model coverage - silent version."""
        if df.empty:
            return df, [], {}
            
        # Calculate models per question per dataset
        question_coverage = df.groupby(['dataset', 'question_id'])['model_name'].nunique().reset_index()
        question_coverage.columns = ['dataset', 'question_id', 'models_answered']
        
        questions_to_keep = set()
        dataset_details = {}
        
        for dataset in sorted(df['dataset'].unique()):
            dataset_questions = question_coverage[question_coverage['dataset'] == dataset]
            
            if len(dataset_questions) == 0:
                continue
            
            # Get statistics
            total_questions = len(dataset_questions)
            coverage_values = dataset_questions['models_answered'].values
            
            # Calculate threshold
            threshold_models = np.percentile(coverage_values, self.config.min_question_coverage_percentile)
            
            # Find questions above threshold
            good_questions = dataset_questions[
                dataset_questions['models_answered'] >= threshold_models
            ]
            
            questions_to_keep.update(good_questions['question_id'].tolist())
            
            # Store details silently
            kept_count = len(good_questions)
            removed_count = total_questions - kept_count
            
            dataset_details[dataset] = {
                "total_questions": total_questions,
                "kept_questions": kept_count,
                "removed_questions": removed_count,
                "coverage_min": int(coverage_values.min()),
                "coverage_max": int(coverage_values.max()),
                "coverage_mean": float(coverage_values.mean()),
                "threshold": float(threshold_models)
            }
        
        # Filter the dataframe
        original_questions = set(df['question_id'].unique())
        df_filtered = df[df['question_id'].isin(questions_to_keep)]
        removed_questions = list(original_questions - questions_to_keep)
            
        return df_filtered, removed_questions, dataset_details
    
    def _clean_datasets_with_logging(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
        """Remove datasets with insufficient data - with detailed logging."""
        if df.empty:
            return df, [], {}
            
        print(f"   📚 CLEANING DATASETS (min {self.config.min_models_per_dataset} models, {self.config.min_questions_per_dataset} questions):")
        
        dataset_stats = df.groupby('dataset').agg({
            'model_name': 'nunique',
            'question_id': 'nunique'
        }).reset_index()
        dataset_stats.columns = ['dataset', 'num_models', 'num_questions']
        
        dataset_details = {}
        
        for _, row in dataset_stats.iterrows():
            dataset = row['dataset']
            num_models = row['num_models']
            num_questions = row['num_questions']
            
            meets_model_req = num_models >= self.config.min_models_per_dataset
            meets_question_req = num_questions >= self.config.min_questions_per_dataset
            keep_dataset = meets_model_req and meets_question_req
            
            status = "✅ KEEP" if keep_dataset else "🗑️  REMOVE"
            
            print(f"     📊 {dataset}: {status}")
            print(f"        Models: {num_models} (min: {self.config.min_models_per_dataset}) {'✅' if meets_model_req else '❌'}")
            print(f"        Questions: {num_questions} (min: {self.config.min_questions_per_dataset}) {'✅' if meets_question_req else '❌'}")
            
            dataset_details[dataset] = {
                "num_models": num_models,
                "num_questions": num_questions,
                "meets_model_requirement": meets_model_req,
                "meets_question_requirement": meets_question_req,
                "kept": keep_dataset
            }
        
        # Keep datasets that meet both criteria
        good_datasets = dataset_stats[
            (dataset_stats['num_models'] >= self.config.min_models_per_dataset) &
            (dataset_stats['num_questions'] >= self.config.min_questions_per_dataset)
        ]['dataset'].tolist()
        
        # Filter the dataframe
        original_datasets = set(df['dataset'].unique())
        df_filtered = df[df['dataset'].isin(good_datasets)]
        removed_datasets = list(original_datasets - set(good_datasets))
        
        if removed_datasets:
            print(f"   🗑️  TOTAL DATASETS REMOVED: {len(removed_datasets)}")
            print(f"      Removed datasets: {removed_datasets}")
        else:
            print(f"   ✅ NO DATASETS REMOVED")
            
        return df_filtered, removed_datasets, dataset_details
    
    def _clean_datasets_silent(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
        """Remove datasets with insufficient data - silent version."""
        if df.empty:
            return df, [], {}
            
        dataset_stats = df.groupby('dataset').agg({
            'model_name': 'nunique',
            'question_id': 'nunique'
        }).reset_index()
        dataset_stats.columns = ['dataset', 'num_models', 'num_questions']
        
        dataset_details = {}
        
        for _, row in dataset_stats.iterrows():
            dataset = row['dataset']
            num_models = row['num_models']
            num_questions = row['num_questions']
            
            meets_model_req = num_models >= self.config.min_models_per_dataset
            meets_question_req = num_questions >= self.config.min_questions_per_dataset
            keep_dataset = meets_model_req and meets_question_req
            
            dataset_details[dataset] = {
                "num_models": num_models,
                "num_questions": num_questions,
                "meets_model_requirement": meets_model_req,
                "meets_question_requirement": meets_question_req,
                "kept": keep_dataset
            }
        
        # Keep datasets that meet both criteria
        good_datasets = dataset_stats[
            (dataset_stats['num_models'] >= self.config.min_models_per_dataset) &
            (dataset_stats['num_questions'] >= self.config.min_questions_per_dataset)
        ]['dataset'].tolist()
        
        # Filter the dataframe
        original_datasets = set(df['dataset'].unique())
        df_filtered = df[df['dataset'].isin(good_datasets)]
        removed_datasets = list(original_datasets - set(good_datasets))
            
        return df_filtered, removed_datasets, dataset_details
    
    def _ensure_complete_matrix_with_logging(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
        """Ensure complete matrix using simple 80/20 rule - with detailed logging."""
        if df.empty:
            return df, [], {}
            
        print(f"   🎯 ENSURING COMPLETE MATRIX (80/20 rule per dataset + global cleanup):")
        
        dataset_details = {}
        
        # Step 1: Clean each dataset separately using 80/20 rule
        cleaned_df = df.copy()
        for dataset in sorted(df['dataset'].unique()):
            dataset_df = cleaned_df[cleaned_df['dataset'] == dataset]
            if len(dataset_df) == 0:
                continue
                
            print(f"     📊 {dataset}:")
            
            # Count questions per model in this dataset
            model_question_counts = dataset_df.groupby('model_name')['question_id'].nunique().sort_values(ascending=False)
            total_models = len(model_question_counts)
            
            # Find 80th percentile threshold (top 80% of models)
            threshold_idx = int(total_models * 0.2)  # Bottom 20%
            if threshold_idx < len(model_question_counts):
                min_questions = model_question_counts.iloc[threshold_idx]
            else:
                min_questions = model_question_counts.min()
            
            # Keep models that answer at least this many questions
            good_models = model_question_counts[model_question_counts >= min_questions].index.tolist()
            
            print(f"        Total models: {total_models}")
            print(f"        Question threshold (80th percentile): {min_questions}")
            print(f"        Kept models: {len(good_models)}")
            print(f"        Removed models: {total_models - len(good_models)}")
            
            # Remove bad models from this dataset
            cleaned_df = cleaned_df[
                (cleaned_df['dataset'] != dataset) | 
                (cleaned_df['model_name'].isin(good_models))
            ]
            
            dataset_details[dataset] = {
                "total_models": total_models,
                "kept_models": len(good_models),
                "threshold": min_questions,
                "kept_model_list": sorted(good_models)
            }
        
        # Step 2: Global cleanup - remove models with too few total questions
        print(f"     🌍 GLOBAL CLEANUP:")
        
        # Count total questions per model across all datasets
        global_model_counts = cleaned_df.groupby('model_name')['question_id'].nunique().sort_values(ascending=False)
        total_models = len(global_model_counts)
        
        # Find top 80% threshold globally
        threshold_idx = int(total_models * 0.2)  # Bottom 20%
        if threshold_idx < len(global_model_counts):
            global_min_questions = global_model_counts.iloc[threshold_idx]
        else:
            global_min_questions = global_model_counts.min()
        
        # Keep only models above global threshold
        final_good_models = global_model_counts[global_model_counts >= global_min_questions].index.tolist()
        removed_models = list(set(global_model_counts.index) - set(final_good_models))
        
        print(f"        Total models: {total_models}")
        print(f"        Global question threshold (80th percentile): {global_min_questions}")
        print(f"        Final kept models: {len(final_good_models)}")
        print(f"        Final removed models: {len(removed_models)}")
        
        # Apply global filter
        final_df = cleaned_df[cleaned_df['model_name'].isin(final_good_models)]
        
        dataset_details['_global'] = {
            "total_models": total_models,
            "kept_models": len(final_good_models),
            "removed_models": len(removed_models),
            "threshold": global_min_questions,
            "removed_model_list": sorted(removed_models)
        }
        
        if removed_models:
            print(f"   🗑️  TOTAL REMOVED MODELS: {len(removed_models)}")
            if len(removed_models) <= 10:
                print(f"      Removed models: {removed_models}")
        else:
            print(f"   ✅ NO MODELS REMOVED")
            
        return final_df, removed_models, dataset_details
    
    def _ensure_complete_matrix_silent(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
        """Ensure complete matrix using simple 80/20 rule - silent version."""
        if df.empty:
            return df, [], {}
            
        dataset_details = {}
        
        # Step 1: Clean each dataset separately using 80/20 rule
        cleaned_df = df.copy()
        for dataset in sorted(df['dataset'].unique()):
            dataset_df = cleaned_df[cleaned_df['dataset'] == dataset]
            if len(dataset_df) == 0:
                continue
                
            # Count questions per model in this dataset
            model_question_counts = dataset_df.groupby('model_name')['question_id'].nunique().sort_values(ascending=False)
            total_models = len(model_question_counts)
            
            # Find 80th percentile threshold (top 80% of models)
            threshold_idx = int(total_models * 0.2)  # Bottom 20%
            if threshold_idx < len(model_question_counts):
                min_questions = model_question_counts.iloc[threshold_idx]
            else:
                min_questions = model_question_counts.min()
            
            # Keep models that answer at least this many questions
            good_models = model_question_counts[model_question_counts >= min_questions].index.tolist()
            
            # Remove bad models from this dataset
            cleaned_df = cleaned_df[
                (cleaned_df['dataset'] != dataset) | 
                (cleaned_df['model_name'].isin(good_models))
            ]
            
            dataset_details[dataset] = {
                "total_models": total_models,
                "kept_models": len(good_models),
                "threshold": min_questions,
                "kept_model_list": sorted(good_models)
            }
        
        # Step 2: Global cleanup - remove models with too few total questions
        global_model_counts = cleaned_df.groupby('model_name')['question_id'].nunique().sort_values(ascending=False)
        total_models = len(global_model_counts)
        
        # Find top 80% threshold globally
        threshold_idx = int(total_models * 0.2)  # Bottom 20%
        if threshold_idx < len(global_model_counts):
            global_min_questions = global_model_counts.iloc[threshold_idx]
        else:
            global_min_questions = global_model_counts.min()
        
        # Keep only models above global threshold
        final_good_models = global_model_counts[global_model_counts >= global_min_questions].index.tolist()
        removed_models = list(set(global_model_counts.index) - set(final_good_models))
        
        # Apply global filter
        final_df = cleaned_df[cleaned_df['model_name'].isin(final_good_models)]
        
        dataset_details['_global'] = {
            "total_models": total_models,
            "kept_models": len(final_good_models),
            "removed_models": len(removed_models),
            "threshold": global_min_questions,
            "removed_model_list": sorted(removed_models)
        }
            
        return final_df, removed_models, dataset_details
    
    def _calculate_coverage_stats_silent(self, df: pd.DataFrame) -> Dict:
        """Calculate coverage statistics silently."""
        if df.empty:
            return {}
            
        stats = {}
        
        for dataset in sorted(df['dataset'].unique()):
            dataset_df = df[df['dataset'] == dataset]
            
            # Model coverage (questions per model)
            model_coverage = dataset_df.groupby('model_name')['question_id'].nunique()
            
            # Question coverage (models per question)  
            question_coverage = dataset_df.groupby('question_id')['model_name'].nunique()
            
            stats[dataset] = {
                'num_models': len(model_coverage),
                'num_questions': len(question_coverage),
                'total_observations': len(dataset_df),
                'model_coverage_stats': {
                    'min': int(model_coverage.min()),
                    'max': int(model_coverage.max()),
                    'mean': float(model_coverage.mean()),
                    'std': float(model_coverage.std())
                },
                'question_coverage_stats': {
                    'min': int(question_coverage.min()),
                    'max': int(question_coverage.max()),
                    'mean': float(question_coverage.mean()),
                    'std': float(question_coverage.std())
                }
            }
        
        return stats
    
    def _print_compact_summary(self, stats: CleaningStats):
        """Print a compact summary of the cleaning process."""
        original_rows = stats.original_shape[0]
        final_rows = stats.final_shape[0]
        removed_rows = original_rows - final_rows
        removal_percentage = 100 * removed_rows / original_rows if original_rows > 0 else 0
        
        print(f"   ✓ Cleaned: {original_rows:,} → {final_rows:,} rows ({removal_percentage:.1f}% removed)")
        print(f"   ✓ Removed: {len(stats.removed_models)} models, {len(stats.removed_questions)} questions, {len(stats.removed_datasets)} datasets")
        print(f"   ✓ Matrix completion: ensured all remaining models answer the same questions")
        
        remaining_datasets = list(stats.coverage_stats_after.keys())
        print(f"   ✓ Final: {len(remaining_datasets)} datasets: {remaining_datasets}")
    
    def _calculate_and_print_coverage_stats(self, df: pd.DataFrame, stage: str) -> Dict:
        """Calculate and print detailed coverage statistics."""
        if df.empty:
            return {}
            
        print(f"   📈 {stage} STATISTICS:")
        print(f"      Total rows: {len(df):,}")
        print(f"      Total models: {df['model_name'].nunique()}")
        print(f"      Total questions: {df['question_id'].nunique()}")
        print(f"      Total datasets: {df['dataset'].nunique()}")
        
        stats = {}
        
        for dataset in sorted(df['dataset'].unique()):
            dataset_df = df[df['dataset'] == dataset]
            
            # Model coverage (questions per model)
            model_coverage = dataset_df.groupby('model_name')['question_id'].nunique()
            
            # Question coverage (models per question)  
            question_coverage = dataset_df.groupby('question_id')['model_name'].nunique()
            
            print(f"      📊 {dataset}:")
            print(f"         Models: {len(model_coverage)}, Questions: {len(question_coverage)}, Observations: {len(dataset_df)}")
            print(f"         Model coverage (questions/model): {model_coverage.min()}-{model_coverage.max()} (avg: {model_coverage.mean():.1f})")
            print(f"         Question coverage (models/question): {question_coverage.min()}-{question_coverage.max()} (avg: {question_coverage.mean():.1f})")
            
            stats[dataset] = {
                'num_models': len(model_coverage),
                'num_questions': len(question_coverage),
                'total_observations': len(dataset_df),
                'model_coverage_stats': {
                    'min': int(model_coverage.min()),
                    'max': int(model_coverage.max()),
                    'mean': float(model_coverage.mean()),
                    'std': float(model_coverage.std())
                },
                'question_coverage_stats': {
                    'min': int(question_coverage.min()),
                    'max': int(question_coverage.max()),
                    'mean': float(question_coverage.mean()),
                    'std': float(question_coverage.std())
                }
            }
        
        return stats
    
    def _print_final_summary(self, stats: CleaningStats):
        """Print final comprehensive summary."""
        print("\n" + "=" * 60)
        print("🎯 FINAL CLEANING SUMMARY")
        print("=" * 60)
        
        # Overall numbers
        original_rows = stats.original_shape[0]
        final_rows = stats.final_shape[0]
        removed_rows = original_rows - final_rows
        removal_percentage = 100 * removed_rows / original_rows if original_rows > 0 else 0
        
        print(f"📊 OVERALL IMPACT:")
        print(f"   Original rows: {original_rows:,}")
        print(f"   Final rows: {final_rows:,}")
        print(f"   Removed rows: {removed_rows:,} ({removal_percentage:.1f}%)")
        print(f"   Cleaning iterations: {stats.iterations}")
        
        print(f"\n🗑️  REMOVED ENTITIES:")
        print(f"   Models: {len(stats.removed_models)}")
        print(f"   Questions: {len(stats.removed_questions)}")
        print(f"   Datasets: {len(stats.removed_datasets)}")
        
        if stats.removed_datasets:
            print(f"   Removed datasets: {stats.removed_datasets}")
        
        print(f"\n✅ FINAL DATA QUALITY:")
        remaining_datasets = list(stats.coverage_stats_after.keys())
        print(f"   Remaining datasets ({len(remaining_datasets)}): {remaining_datasets}")
        
        for dataset, stats_dict in stats.coverage_stats_after.items():
            model_stats = stats_dict['model_coverage_stats']
            question_stats = stats_dict['question_coverage_stats']
            print(f"   📊 {dataset}:")
            print(f"      {stats_dict['num_models']} models × {stats_dict['num_questions']} questions = {stats_dict['total_observations']} observations")
            print(f"      Model coverage: {model_stats['min']}-{model_stats['max']} questions (avg: {model_stats['mean']:.1f})")
            print(f"      Question coverage: {question_stats['min']}-{question_stats['max']} models (avg: {question_stats['mean']:.1f})")
        
        print("=" * 60)
        print("✅ DATA CLEANING COMPLETE - Ready for evaluation!")
        print("=" * 60)


def create_cleaner() -> MatrixCleaner:
    """Create a MatrixCleaner with optimized defaults."""
    return MatrixCleaner()
