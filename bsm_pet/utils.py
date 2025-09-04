"""Utility functions for option pricing."""
import numpy as np
from scipy.stats import norm

def norm_cdf(x):
    """
    Cumulative distribution function for standard normal distribution.
    
    Args:
        x (float): Input value
        
    Returns:
        float: Probability
    """
    return norm.cdf(x)

def norm_pdf(x):
    """
    Probability density function for standard normal distribution.
    
    Args:
        x (float): Input value
        
    Returns:
        float: Probability density
    """
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

def days_to_years(days, day_count_convention=365):
    """
    Convert days to year fraction.
    
    Args:
        days (int): Number of days
        day_count_convention (int): Days in year
        
    Returns:
        float: Year fraction
    """
    return days / day_count_convention