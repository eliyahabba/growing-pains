"""
Matrix splitting utilities for train/test separation.

This module provides functionality to split evaluation matrices into training and testing sets
using various strategies, supporting proper evaluation of selection methods.
"""

from typing import Tuple, Optional, List
from enum import Enum
import pandas as pd
import numpy as np
from dataclasses import dataclass


class SplitStrategy(Enum):
    """Available splitting strategies."""
    TEMPORAL = "temporal"
    RANDOM = "random" 
    MODEL_BASED = "model_based"
    QUESTION_BASED = "question_based"


@dataclass
class SplitConfig:
    """Configuration for matrix splitting."""
    strategy: SplitStrategy = SplitStrategy.MODEL_BASED
    test_ratio: float = 0.2
    random_seed: Optional[int] = 42
    temporal_split_column: str = "model_name"  # Column to use for temporal ordering
    stratify_by: Optional[str] = None  # Column to stratify by (e.g., "dataset")


class MatrixSplitter:
    """Splits evaluation matrices into train/test sets using various strategies."""
    
    def __init__(self, config: Optional[SplitConfig] = None):
        """Initialize splitter with configuration.
        
        Args:
            config: Split configuration. If None, uses defaults.
        """
        self.config = config or SplitConfig()
        
    def split(self, matrix_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split matrix into train/test sets.
        
        Args:
            matrix_df: Input matrix DataFrame
            
        Returns:
            Tuple of (train_df, test_df)
            
        Raises:
            ValueError: If matrix is empty or invalid strategy
        """
        if matrix_df.empty:
            raise ValueError("Cannot split empty matrix")
            
        if self.config.strategy == SplitStrategy.TEMPORAL:
            return self._temporal_split(matrix_df)
        elif self.config.strategy == SplitStrategy.RANDOM:
            return self._random_split(matrix_df)
        elif self.config.strategy == SplitStrategy.MODEL_BASED:
            return self._model_based_split(matrix_df)
        elif self.config.strategy == SplitStrategy.QUESTION_BASED:
            return self._question_based_split(matrix_df)
        else:
            raise ValueError(f"Unknown split strategy: {self.config.strategy}")
    
    def _temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split based on temporal ordering of models (newer models to test).
        
        Assumes models can be ordered temporally. Uses model names as proxy for time.
        """
        if self.config.temporal_split_column not in df.columns:
            raise ValueError(f"Column '{self.config.temporal_split_column}' not found in matrix")
        
        # Get unique models and sort them (assuming lexicographic order correlates with time)
        unique_models = sorted(df[self.config.temporal_split_column].unique())
        n_models = len(unique_models)
        n_test_models = max(1, int(n_models * self.config.test_ratio))
        
        # Take the "newest" models for test (last in sorted order)
        test_models = set(unique_models[-n_test_models:])
        train_models = set(unique_models[:-n_test_models])
        
        train_df = df[df[self.config.temporal_split_column].isin(train_models)].copy()
        test_df = df[df[self.config.temporal_split_column].isin(test_models)].copy()
        
        return train_df, test_df
    
    def _random_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Random split of the entire matrix."""
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        n_rows = len(df)
        n_test = int(n_rows * self.config.test_ratio)
        
        # Random indices for test set
        test_indices = np.random.choice(n_rows, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(n_rows), test_indices)
        
        train_df = df.iloc[train_indices].copy()
        test_df = df.iloc[test_indices].copy()
        
        return train_df, test_df
    
    def _model_based_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split by models - some models go to train, others to test.
        
        This is similar to temporal but uses random selection of models.
        """
        if "model_name" not in df.columns:
            raise ValueError("model_name column required for model-based split")
        
        unique_models = df["model_name"].unique()
        n_models = len(unique_models)
        n_test_models = max(1, int(n_models * self.config.test_ratio))
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        # Randomly select models for test
        test_models = set(np.random.choice(unique_models, size=n_test_models, replace=False))
        train_models = set(unique_models) - test_models
        
        train_df = df[df["model_name"].isin(train_models)].copy()
        test_df = df[df["model_name"].isin(test_models)].copy()
        
        return train_df, test_df
    
    def _question_based_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split by questions - some questions go to train, others to test.
        
        This creates a held-out set of questions for evaluation.
        """
        if "question_id" not in df.columns:
            raise ValueError("question_id column required for question-based split")
        
        unique_questions = df["question_id"].unique()
        n_questions = len(unique_questions)
        n_test_questions = max(1, int(n_questions * self.config.test_ratio))
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        # Randomly select questions for test
        test_questions = set(np.random.choice(unique_questions, size=n_test_questions, replace=False))
        train_questions = set(unique_questions) - test_questions
        
        train_df = df[df["question_id"].isin(train_questions)].copy()
        test_df = df[df["question_id"].isin(test_questions)].copy()
        
        return train_df, test_df
    
    def get_split_info(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """Get information about the split.
        
        Args:
            train_df: Training set DataFrame
            test_df: Test set DataFrame
            
        Returns:
            Dictionary with split statistics
        """
        info = {
            "strategy": self.config.strategy.value,
            "test_ratio_target": self.config.test_ratio,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "total_size": len(train_df) + len(test_df),
            "actual_test_ratio": len(test_df) / (len(train_df) + len(test_df)) if (len(train_df) + len(test_df)) > 0 else 0,
        }
        
        # Add strategy-specific info
        if self.config.strategy in [SplitStrategy.TEMPORAL, SplitStrategy.MODEL_BASED]:
            info.update({
                "train_models": train_df["model_name"].nunique() if "model_name" in train_df.columns else 0,
                "test_models": test_df["model_name"].nunique() if "model_name" in test_df.columns else 0,
                "train_model_names": sorted(train_df["model_name"].unique().tolist()) if "model_name" in train_df.columns else [],
                "test_model_names": sorted(test_df["model_name"].unique().tolist()) if "model_name" in test_df.columns else [],
            })
        
        if self.config.strategy == SplitStrategy.QUESTION_BASED:
            info.update({
                "train_questions": train_df["question_id"].nunique() if "question_id" in train_df.columns else 0,
                "test_questions": test_df["question_id"].nunique() if "question_id" in test_df.columns else 0,
            })
        
        # Dataset distribution
        if "dataset" in train_df.columns and "dataset" in test_df.columns:
            info.update({
                "train_datasets": sorted(train_df["dataset"].unique().tolist()),
                "test_datasets": sorted(test_df["dataset"].unique().tolist()),
                "common_datasets": sorted(set(train_df["dataset"].unique()) & set(test_df["dataset"].unique())),
            })
        
        return info


def create_splitter(strategy: str = "temporal", test_ratio: float = 0.2, 
                   random_seed: Optional[int] = 42, **kwargs) -> MatrixSplitter:
    """Convenience function to create a MatrixSplitter.
    
    Args:
        strategy: Split strategy name
        test_ratio: Fraction of data for test set
        random_seed: Random seed for reproducibility
        **kwargs: Additional configuration options
        
    Returns:
        Configured MatrixSplitter instance
    """
    try:
        strategy_enum = SplitStrategy(strategy)
    except ValueError:
        available = [s.value for s in SplitStrategy]
        raise ValueError(f"Invalid strategy '{strategy}'. Available: {available}")
    
    config = SplitConfig(
        strategy=strategy_enum,
        test_ratio=test_ratio,
        random_seed=random_seed,
        **kwargs
    )
    
    return MatrixSplitter(config)
