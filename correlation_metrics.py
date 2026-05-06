"""
Correlation metrics for evaluating predictor-target relationships.

Extracted and adapted from:
https://github.com/xu-ji/information-bottleneck (ICML 2023)

Provides three standard correlation measures:
- Pearson: Linear correlation coefficient
- Spearman: Rank-based (non-parametric) correlation
- Kendall: Tau statistic, ranked concordance pairs
"""

import numpy as np
from typing import Tuple, Union
import warnings

# Optional scipy import for p-values (falls back to None if not available)
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_stats = None


def compute_pearson(x: Union[np.ndarray, list], 
                    y: Union[np.ndarray, list]) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient.
    
    Measures linear correlation between x and y.
    Assumes normal distribution; sensitive to outliers.
    
    Args:
        x: First variable (array-like)
        y: Second variable (array-like)
        
    Returns:
        (correlation: float, p_value: float)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Filter out NaN and inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        warnings.warn(f"Not enough valid pairs ({len(x_clean)}) for Pearson correlation")
        return np.nan, np.nan
    
    if HAS_SCIPY:
        corr, pval = scipy_stats.pearsonr(x_clean, y_clean)
        return float(corr), float(pval)
    else:
        # Pure numpy implementation
        mean_x = np.mean(x_clean)
        mean_y = np.mean(y_clean)
        cov = np.mean((x_clean - mean_x) * (y_clean - mean_y))
        std_x = np.std(x_clean, ddof=1)
        std_y = np.std(y_clean, ddof=1)
        if std_x == 0 or std_y == 0:
            return np.nan, np.nan
        corr = cov / (std_x * std_y)
        return float(corr), np.nan


def compute_spearman(x: Union[np.ndarray, list],
                     y: Union[np.ndarray, list]) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation coefficient.
    
    Measures monotonic (rank-based) relationship.
    Non-parametric; more robust to outliers than Pearson.
    
    Args:
        x: First variable (array-like)
        y: Second variable (array-like)
        
    Returns:
        (correlation: float, p_value: float)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Filter out NaN and inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        warnings.warn(f"Not enough valid pairs ({len(x_clean)}) for Spearman correlation")
        return np.nan, np.nan
    
    if HAS_SCIPY:
        corr, pval = scipy_stats.spearmanr(x_clean, y_clean)
        return float(corr), float(pval)
    else:
        # Rank-based Pearson correlation
        rank_x = np.argsort(np.argsort(x_clean)) + 1
        rank_y = np.argsort(np.argsort(y_clean)) + 1
        
        mean_rank_x = np.mean(rank_x)
        mean_rank_y = np.mean(rank_y)
        cov = np.mean((rank_x - mean_rank_x) * (rank_y - mean_rank_y))
        std_rank_x = np.std(rank_x, ddof=1)
        std_rank_y = np.std(rank_y, ddof=1)
        if std_rank_x == 0 or std_rank_y == 0:
            return np.nan, np.nan
        corr = cov / (std_rank_x * std_rank_y)
        return float(corr), np.nan


def compute_kendall_tau(x: Union[np.ndarray, list],
                        y: Union[np.ndarray, list]) -> Tuple[float, float]:
    """
    Compute Kendall's Tau rank correlation.
    
    Measures correlation based on concordant/discordant pairs.
    Non-parametric; good for small sample sizes.
    
    Args:
        x: First variable (array-like)
        y: Second variable (array-like)
        
    Returns:
        (tau: float, p_value: float)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Filter out NaN and inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        warnings.warn(f"Not enough valid pairs ({len(x_clean)}) for Kendall Tau correlation")
        return np.nan, np.nan
    
    if HAS_SCIPY:
        tau, pval = scipy_stats.kendalltau(x_clean, y_clean)
        return float(tau), float(pval)
    else:
        # Pure numpy implementation of Kendall's Tau
        n = len(x_clean)
        # Count concordant and discordant pairs
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                x_diff = (x_clean[i] - x_clean[j])
                y_diff = (y_clean[i] - y_clean[j])
                if x_diff != 0 and y_diff != 0:
                    if np.sign(x_diff) == np.sign(y_diff):
                        concordant += 1
                    else:
                        discordant += 1
        
        total = concordant + discordant
        if total == 0:
            return np.nan, np.nan
        
        tau = (concordant - discordant) / total
        return float(tau), np.nan


def compute_all_correlations(x: Union[np.ndarray, list],
                            y: Union[np.ndarray, list],
                            verbose: bool = False) -> dict:
    """
    Compute all three correlation metrics at once.
    
    Args:
        x: First variable (array-like)
        y: Second variable (array-like)
        verbose: If True, print results
        
    Returns:
        Dictionary with keys:
        - 'pearson': (coefficient, p_value)
        - 'spearman': (coefficient, p_value)
        - 'kendall': (tau, p_value)
    """
    results = {
        'pearson': compute_pearson(x, y),
        'spearman': compute_spearman(x, y),
        'kendall': compute_kendall_tau(x, y),
    }
    
    if verbose:
        print("\n=== Correlation Metrics ===")
        p_val_str = f"p = {results['pearson'][1]:.2e}" if np.isfinite(results['pearson'][1]) else "p = N/A"
        print(f"Pearson  r = {results['pearson'][0]:.4f}, {p_val_str}")
        
        s_val_str = f"p = {results['spearman'][1]:.2e}" if np.isfinite(results['spearman'][1]) else "p = N/A"
        print(f"Spearman r = {results['spearman'][0]:.4f}, {s_val_str}")
        
        k_val_str = f"p = {results['kendall'][1]:.2e}" if np.isfinite(results['kendall'][1]) else "p = N/A"
        print(f"Kendall  τ = {results['kendall'][0]:.4f}, {k_val_str}")
        print()
    
    return results


def format_correlation_table(results: dict, metric_name: str = "") -> str:
    """
    Format correlation results as a LaTeX table row (as in paper).
    
    Args:
        results: Output from compute_all_correlations()
        metric_name: Name of the metric being evaluated
        
    Returns:
        LaTeX table row string
    """
    p, p_pval = results['pearson']
    r, r_pval = results['spearman']
    tau, tau_pval = results['kendall']
    
    # Remove NaN values for display
    p_str = f"{p:.4f}" if np.isfinite(p) else "N/A"
    r_str = f"{r:.4f}" if np.isfinite(r) else "N/A"
    tau_str = f"{tau:.4f}" if np.isfinite(tau) else "N/A"
    
    return f"{metric_name} & {r_str} & {p_str} & {tau_str} \\\\"


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic data with correlation
    x = np.random.randn(50)
    y = x + 0.3 * np.random.randn(50)
    
    print("Example: Synthetic data with positive correlation")
    results = compute_all_correlations(x, y, verbose=True)
    print("LaTeX table format:")
    print(format_correlation_table(results, "Example"))
