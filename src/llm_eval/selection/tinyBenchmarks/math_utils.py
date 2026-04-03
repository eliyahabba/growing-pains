"""
Mathematical utility functions for IRT computations.

This module contains core mathematical functions used across the tinyBenchmarks
implementation, including sigmoid and item response curve calculations.
"""

import numpy as np
from scipy.optimize import minimize


def sigmoid(z):
    """
    Compute the sigmoid function for the input z.
    
    Parameters:
    - z: A numeric value or numpy array.
    
    Returns:
    - The sigmoid of z.
    """
    return 1 / (1 + np.exp(-z))


def item_curve(theta, a, b):
    """
    Compute the item response curve for given parameters.
    
    This function handles different input shapes for compatibility across
    different parts of the codebase.

    Parameters:
    - theta: The ability parameter of the subject
             Can be [n_models, n_dims, 1], [n_models, 1], or [n_models] format
    - a: The discrimination parameter of the item - various shapes supported
    - b: The difficulty parameter of the item - various shapes supported

    Returns:
    - The probability of a correct response given the item parameters and subject ability.
    """
    # Handle different input shapes for theta
    if theta.ndim == 3:  # [n_models, n_dims, 1] format from original code
        theta_squeezed = theta.squeeze(axis=2)  # [n_models, n_dims]
        if theta_squeezed.shape[1] == 1:  # Single dimension case
            theta_val = theta_squeezed  # Keep as [n_models, 1]
        else:
            theta_val = theta_squeezed  # Keep as [n_models, n_dims]
    elif theta.ndim == 1:  # [n_models] format
        theta_val = theta.reshape(-1, 1)  # [n_models, 1]
    else:  # [n_models, 1] or [n_models, n_dims] format
        theta_val = theta
    
    # Handle different parameter shapes
    if a.ndim == 3 and b.ndim == 3:
        # Multi-dimensional IRT case: a=[n_models, n_dims, n_items], b=[n_models, n_dims, n_items]
        # theta_val should be [n_models, n_dims]
        if theta_val.ndim == 2 and theta_val.shape[1] == a.shape[1]:
            # Expand theta for broadcasting: [n_models, n_dims] -> [n_models, n_dims, 1]
            theta_expanded = theta_val[:, :, None]
            # Compute z = a*theta - b: [n_models, n_dims, n_items]
            z = a * theta_expanded - b
            # Sum over dimensions: [n_models, n_items]
            z = z.sum(axis=1)
        else:
            raise ValueError(f"Incompatible shapes: theta_val={theta_val.shape}, a={a.shape}")
    
    elif a.ndim == 2 and b.ndim == 2:
        # Standard 2D case: a=[n_models, n_items], b=[n_models, n_items]
        # theta_val should be [n_models, 1] or [n_models, n_dims]
        if theta_val.shape[1] == 1:
            # Single dimension case
            z = a * theta_val - b  # Broadcasting: [n_models, n_items]
        else:
            # Multi-dimension case - need to sum
            # Expand theta: [n_models, n_dims] -> [n_models, n_dims, 1]
            # But a and b are 2D, so this shouldn't happen normally
            # Fall back to using first dimension
            theta_first = theta_val[:, 0:1]  # [n_models, 1]
            z = a * theta_first - b
    
    else:
        # Fallback for other cases
        try:
            z = a * theta_val - b
        except ValueError as e:
            raise ValueError(f"Broadcasting error in item_curve: theta_val={theta_val.shape}, a={a.shape}, b={b.shape}") from e
    
    z = np.clip(z, -30, 30)  # Prevent overflow
    return sigmoid(z)


def simple_item_curve(theta, a, b):
    """
    Simple version of item_curve for basic use cases.
    
    This is the original simpler implementation kept for backward compatibility.
    
    Parameters:
    - theta: The ability parameter of the subject.
    - a: The discrimination parameter of the item.
    - b: The difficulty parameter of the item.
    
    Returns:
    - The probability of a correct response given the item parameters and subject ability.
    """
    z = np.clip(a * theta - b, -30, 30)
    if z.ndim > 1:
        z = z.sum(axis=1)
    return sigmoid(z)


def estimate_ability_parameters(responses_test, A, B, theta_init=None, eps=1e-10, optimizer="BFGS"):
    """
    Estimates the ability parameters for a new set of test responses.
    
    This is a unified implementation that handles both single-dimensional and 
    multi-dimensional cases automatically.
    
    Parameters:
    - responses_test: A 1D array of the test subject's responses.
    - A: The discrimination parameters of the IRT model - shape [n_models, n_dims, n_items] or [1, n_items]
    - B: The difficulty parameters of the IRT model - shape [n_models, n_dims, n_items] or [1, n_items]
    - theta_init: Initial guess for the ability parameters.
    - eps: A small value to avoid division by zero and log of zero errors.
    - optimizer: The optimization method to use.
    
    Returns: 
    - optimal_theta: The estimated ability parameters for the test subject.
                    Format depends on input: [n_models, n_dims, 1] or [1, 1, 1]
    """
    
    # Determine dimensionality
    if A.ndim == 3:
        # Multi-dimensional case: [n_models, n_dims, n_items]
        D = A.shape[1]
        is_multidim = True
    elif A.ndim == 2:
        # Single-dimensional case: [1, n_items] -> treat as [1, 1, n_items]
        D = 1
        is_multidim = False
        # Reshape to 3D format
        A = A[:, None, :]  # [1, 1, n_items]
        B = B[:, None, :]  # [1, 1, n_items]
    else:
        raise ValueError(f"Unsupported A shape: {A.shape}")

    # Define the negative log likelihood function
    def neg_log_like(x):
        # Reshape theta to [1, D, 1] format
        theta_reshaped = np.array(x).reshape(1, D, 1)
        P = item_curve(theta_reshaped, A, B).squeeze()  # [n_items]
        log_likelihood = np.sum(responses_test * np.log(P + eps) + (1 - responses_test) * np.log(1 - P + eps))
        return -log_likelihood

    # Set initial theta
    if theta_init is not None:
        if isinstance(theta_init, np.ndarray):
            if theta_init.size == D:
                theta_init_val = theta_init.flatten()
            elif theta_init.size == 1:
                theta_init_val = np.full(D, float(theta_init.flatten()[0]))
            else:
                theta_init_val = np.zeros(D)
        else:
            theta_init_val = np.full(D, float(theta_init))
    else:
        theta_init_val = np.zeros(D)

    # Use the minimize function to find the ability parameters
    result = minimize(neg_log_like, theta_init_val, method=optimizer)
    
    if is_multidim:
        # Return in multi-dimensional format: [1, D, 1]
        optimal_theta = result.x[None, :, None]
    else:
        # Return in single-dimensional format: [1, 1, 1]
        optimal_theta = np.array([[[result.x[0]]]])

    return optimal_theta
