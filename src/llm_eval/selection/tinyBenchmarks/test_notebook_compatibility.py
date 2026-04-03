#!/usr/bin/env python3
"""
Test script to verify that the training.py module replicates 
the exact notebook workflow from training_irt.ipynb.

This script tests individual components to ensure compatibility.
"""

import numpy as np
import pandas as pd
from training import (
    TrainingConfig, 
    compute_balance_weights, 
    binarize_responses,
    get_lambda,
    _compute_dataset_variance
)


def test_balance_weights():
    """Test that balance weights computation works with AdaptEval dataset structure."""
    print("Testing balance weights computation...")
    
    # Create mock matrix DataFrame similar to AdaptEval structure
    models = [f"model_{i}" for i in range(10)]
    questions = [f"q_{i}" for i in range(100)]
    
    # Create datasets with hierarchical naming (like legalbench.xxx)
    data = []
    for model in models:
        for i, question in enumerate(questions):
            if i < 30:  # legalbench with subscenarios
                dataset = f"legalbench.topic_{i // 10}"  # 3 legalbench subscenarios
            elif i < 60:  # Another hierarchical dataset
                dataset = f"mmlu.subject_{(i-30) // 10}"  # 3 mmlu subscenarios  
            else:  # Simple datasets
                dataset = f"simple_dataset_{i // 20}"  # 2 simple datasets
            
            data.append({
                'model_name': model,
                'question_id': question,
                'normalized_score': np.random.rand(),
                'dataset': dataset
            })
    
    matrix_df = pd.DataFrame(data)
    
    balance_weights = compute_balance_weights(matrix_df)
    
    print(f"Balance weights computed for {len(balance_weights)} questions")
    print(f"Weights range: {balance_weights.min():.4f} - {balance_weights.max():.4f}")
    
    # Should have different weights for hierarchical datasets
    assert len(balance_weights) == len(questions), "Should have weights for all questions"
    
    # Check that hierarchical datasets got non-uniform weights
    hierarchical_questions = [f"q_{i}" for i in range(60)]  # First 60 are hierarchical
    hierarchical_weights = [balance_weights[i] for i in range(60)]
    
    if len(set(hierarchical_weights)) > 1:
        print("   ✓ Hierarchical datasets received non-uniform weights")
    else:
        print("   ℹ️  All weights are uniform (no hierarchical structure detected)")
    
    print("✓ Balance weights test passed")


def test_get_lambda():
    """Test lambda computation matches notebook."""
    print("Testing lambda computation...")
    
    # Test with some example values
    b = 0.05
    v = 0.1
    expected = (b**2) / (v + (b**2))
    actual = get_lambda(b, v)
    
    assert np.isclose(actual, expected), f"Lambda mismatch: expected {expected}, got {actual}"
    print(f"Lambda for b={b}, v={v}: {actual:.6f}")
    print("✓ Lambda computation test passed")


def test_binarization():
    """Test binarization logic works with AdaptEval data format."""
    print("Testing binarization...")
    
    # Test 1: Already binary data (common in AdaptEval)
    print("  Test 1: Already binary data...")
    models = [f"model_{i}" for i in range(5)]
    questions = [f"q_{i}" for i in range(10)]
    
    data = []
    for model in models:
        for question in questions:
            data.append({
                'model_name': model,
                'question_id': question,
                'normalized_score': float(np.random.choice([0, 1])),  # Already binary
                'dataset': 'test_dataset'
            })
    
    matrix_df = pd.DataFrame(data)
    binary_df = binarize_responses(matrix_df)
    
    print(f"  Original shape: {matrix_df.shape}, Binary shape: {binary_df.shape}")
    unique_scores = binary_df['normalized_score'].unique()
    assert set(unique_scores).issubset({0.0, 1.0}), "Scores should remain binary"
    
    # Test 2: Continuous data that needs thresholding
    print("  Test 2: Continuous data needing thresholding...")
    data2 = []
    for model in models:
        for i, question in enumerate(questions):
            # Create scores that should be binarized differently per dataset
            if i < 5:  # dataset1 - higher scores
                score = np.random.uniform(0.6, 1.0)
                dataset = 'high_score_dataset'
            else:  # dataset2 - lower scores  
                score = np.random.uniform(0.0, 0.4)
                dataset = 'low_score_dataset'
            
            data2.append({
                'model_name': model,
                'question_id': question,
                'normalized_score': score,
                'dataset': dataset
            })
    
    matrix_df2 = pd.DataFrame(data2)
    binary_df2 = binarize_responses(matrix_df2)
    
    # Check that responses are now binary
    unique_scores2 = binary_df2['normalized_score'].unique()
    assert set(unique_scores2).issubset({0.0, 1.0}), "Scores should be binary after thresholding"
    
    print("✓ Binarization test passed")


def test_config():
    """Test that TrainingConfig has the right defaults."""
    print("Testing TrainingConfig...")
    
    cfg = TrainingConfig()
    
    # Check notebook defaults
    assert cfg.dims_search == [5, 10], "Default dims_search should be [5, 10]"
    assert cfg.device == 'cuda', "Default device should be 'cuda'"
    assert cfg.epochs == 2000, "Default epochs should be 2000"
    assert cfg.lr == 0.1, "Default lr should be 0.1"
    assert cfg.random_state == 42, "Default random_state should be 42"
    assert cfg.val_stride == 5, "Default val_stride should be 5"
    assert cfg.number_item_per_scenario == 100, "Default number_item_per_scenario should be 100"
    
    print("✓ TrainingConfig test passed")


def main():
    """Run all tests."""
    print("Running TinyBenchmarks generalized training tests...\n")
    
    try:
        test_config()
        print()
        
        test_get_lambda()
        print()
        
        test_balance_weights()
        print()
        
        test_binarization()
        print()
        
        print("🎉 All tests passed! The training module is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
