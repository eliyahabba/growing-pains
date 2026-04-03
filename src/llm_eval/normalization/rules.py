

from math import erf, sqrt
from typing import Dict, List, Optional


def minmax_normalize(x: float, min_value: float, max_value: float, higher_is_better: bool) -> float:
    if max_value == min_value:
        return 50.0
    ratio = (x - min_value) / (max_value - min_value)
    if not higher_is_better:
        ratio = 1.0 - ratio
    return float(max(0.0, min(100.0, ratio * 100.0)))


def binary_normalize(x: float, higher_is_better: bool = True, threshold: float = 0.5) -> float:
    """Normalize a binary score to [0,100].

    Assumes inputs are already 0 or 1. If not, applies a simple threshold.
    When ``higher_is_better`` is False, the mapping is inverted.
    """
    value = 1.0 if float(x) >= threshold else 0.0
    if not higher_is_better:
        value = 1.0 - value
    return 100.0 if value >= 0.5 else 0.0


def _phi(z: float) -> float:
    # standard normal CDF
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def zscore_cdf_normalize(x: float, mean: float, std: float, higher_is_better: bool) -> float:
    if std <= 0:
        return 50.0
    z = (x - mean) / std
    p = _phi(z)
    if not higher_is_better:
        p = 1.0 - p
    return float(max(0.0, min(100.0, p * 100.0)))


def find_optimal_threshold(scores_matrix: List[List[float]], method: str = 'direct', n_thresholds: int = 100) -> float:
    """
    Find optimal threshold for binarization that preserves mean scores across models.
    
    Args:
        scores_matrix: Matrix of scores (list of lists: models x questions) 
        method: 'direct' (original scale) or 'normalized' (0-1 scale)
        n_thresholds: Number of candidate thresholds to try
        
    Returns:
        Optimal threshold value
    """
    if not scores_matrix or not scores_matrix[0]:
        return 0.5
    
    # Flatten matrix to find min/max
    all_scores = [score for row in scores_matrix for score in row]
    score_min = min(all_scores)
    score_max = max(all_scores)
    
    if score_max <= score_min:
        return score_min if method == 'direct' else 0.5
    
    # Prepare data for threshold search
    if method == 'normalized':
        # Normalize to [0,1] first
        work_matrix = [[(score - score_min) / (score_max - score_min) for score in row] 
                      for row in scores_matrix]
        # Create candidate thresholds
        candidate_thresholds = [0.01 + i * (0.98) / (n_thresholds - 1) for i in range(n_thresholds)]
    else:  # direct
        work_matrix = [row[:] for row in scores_matrix]  # Copy
        # Use small margin to avoid edge effects
        margin = (score_max - score_min) * 0.01
        threshold_min = score_min + margin
        threshold_max = score_max - margin
        if threshold_max <= threshold_min:
            return score_min
        candidate_thresholds = [threshold_min + i * (threshold_max - threshold_min) / (n_thresholds - 1) 
                               for i in range(n_thresholds)]
    
    # Find threshold that minimizes mean preservation error
    best_threshold = candidate_thresholds[0]
    best_error = float('inf')
    
    for threshold in candidate_thresholds:
        # Calculate original means per model (across questions)
        original_means = [sum(row) / len(row) for row in work_matrix]
        
        # Calculate binarized means per model
        binarized_means = [sum(1 for score in row if score > threshold) / len(row) for row in work_matrix]
        
        # Calculate average absolute error across models
        errors = [abs(orig - bin_mean) for orig, bin_mean in zip(original_means, binarized_means)]
        error = sum(errors) / len(errors)
        
        if error < best_error:
            best_error = error
            best_threshold = threshold
    
    return float(best_threshold)


def irt_binary_normalize(x: float, threshold: float, higher_is_better: bool = True) -> float:
    """
    IRT-style binary normalization using pre-computed optimal threshold.
    
    Args:
        x: Raw score to normalize
        threshold: Pre-computed optimal threshold for this scenario
        higher_is_better: Whether higher scores are better
        
    Returns:
        Binary score (0.0 or 1.0)
    """
    is_correct = float(x) > threshold
    if not higher_is_better:
        is_correct = not is_correct
    return 1.0 if is_correct else 0.0


