"""Metrics and evaluation utilities."""

from typing import Dict, List

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }
    
    return metrics


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        MAPE value
    """
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return float('inf')
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary into readable string.
    
    Args:
        metrics: Dictionary of metric names and values
        precision: Number of decimal places
    
    Returns:
        Formatted string
    """
    lines = []
    for name, value in metrics.items():
        lines.append(f"{name.upper()}: {value:.{precision}f}")
    
    return "\n".join(lines)


def normalize_views(
    views: np.ndarray,
    days_since_publish: np.ndarray,
    method: str = "per_day"
) -> np.ndarray:
    """
    Normalize view counts by time since publication.
    
    Args:
        views: Array of view counts
        days_since_publish: Array of days since publication
        method: Normalization method ('per_day', 'log', 'sqrt')
    
    Returns:
        Normalized view counts
    """
    if method == "per_day":
        # Views per day (avoid division by zero)
        days = np.maximum(days_since_publish, 1)
        return views / days
    elif method == "log":
        # Log-normalized views
        return np.log1p(views)
    elif method == "sqrt":
        # Square root normalization
        return np.sqrt(views)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def remove_outliers(
    data: np.ndarray,
    n_std: float = 2.0,
    method: str = "zscore"
) -> np.ndarray:
    """
    Remove outliers from data.
    
    Args:
        data: Input data array
        n_std: Number of standard deviations for outlier threshold
        method: Outlier detection method ('zscore', 'iqr')
    
    Returns:
        Boolean mask where True indicates non-outliers
    """
    if method == "zscore":
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        return z_scores < n_std
    
    elif method == "iqr":
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - (n_std * iqr)
        upper_bound = q3 + (n_std * iqr)
        return (data >= lower_bound) & (data <= upper_bound)
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
