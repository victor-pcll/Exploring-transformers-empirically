import numpy as np

def safe_mean(data, divisor=1.0):
    """
    Compute mean safely, returning 0.0 if data is empty.
    Args:
        data: list or array of numerical values
        divisor: value to divide the mean by
    Returns:
        mean value as a float
    """
    return float(np.mean(data)/divisor if len(data) > 0 else 0.0)

def safe_std(data, divisor=1.0):
    """
    Compute standard deviation safely, returning 0.0 if data has less than 2 elements.
    Args:
        data: list or array of numerical values
        divisor: value to divide the standard deviation by
    Returns:
        standard deviation value as a float
    """
    return float(np.std(data, ddof=1)/divisor if len(data) > 1 else 0.0)