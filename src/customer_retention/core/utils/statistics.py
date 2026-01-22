import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Union
from customer_retention.core.compat import pd, Series


def compute_psi_numeric(current: Series, reference_hist_edges: List[float], reference_hist_counts: List[int], epsilon: float = 1e-10) -> float:
    """Calculate Population Stability Index for numeric column using reference histogram bins."""
    edges = np.array(reference_hist_edges)
    baseline_counts = np.array(reference_hist_counts)
    current_counts, _ = np.histogram(current.dropna(), bins=edges)
    baseline_prop = baseline_counts / baseline_counts.sum()
    current_prop = current_counts / current_counts.sum() if current_counts.sum() > 0 else np.zeros_like(current_counts, dtype=float)
    baseline_prop = np.maximum(baseline_prop, epsilon)
    current_prop = np.maximum(current_prop, epsilon)
    return float(np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop)))


def compute_psi_from_series(reference: Series, current: Series, n_bins: int = 10, epsilon: float = 1e-10) -> float:
    """Calculate PSI from two series directly (auto-generates bins)."""
    ref_clean, curr_clean = reference.dropna(), current.dropna()
    if ref_clean.dtype in ['object', 'category', 'bool'] or curr_clean.dtype in ['object', 'category', 'bool']:
        return compute_psi_categorical(ref_clean, curr_clean, epsilon)
    min_val, max_val = min(ref_clean.min(), curr_clean.min()), max(ref_clean.max(), curr_clean.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)
    ref_hist, _ = np.histogram(ref_clean, bins=bins)
    curr_hist, _ = np.histogram(curr_clean, bins=bins)
    ref_pct = ref_hist / len(ref_clean) + epsilon
    curr_pct = curr_hist / len(curr_clean) + epsilon if len(curr_clean) > 0 else np.full_like(ref_hist, epsilon, dtype=float)
    return float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))


def compute_psi_categorical(reference: Series, current: Series, epsilon: float = 1e-10) -> float:
    """Calculate Population Stability Index for categorical column."""
    ref_counts = reference.value_counts(normalize=True)
    curr_counts = current.value_counts(normalize=True)
    all_categories = set(ref_counts.index) | set(curr_counts.index)
    psi = 0.0
    for cat in all_categories:
        ref_pct = ref_counts.get(cat, epsilon)
        curr_pct = curr_counts.get(cat, epsilon)
        psi += (curr_pct - ref_pct) * np.log((curr_pct + epsilon) / (ref_pct + epsilon))
    return float(psi)


def compute_ks_statistic(reference: Series, current: Series) -> Tuple[float, float]:
    """Calculate Kolmogorov-Smirnov statistic and p-value."""
    ref_clean, curr_clean = reference.dropna(), current.dropna()
    statistic, pvalue = stats.ks_2samp(ref_clean, curr_clean)
    return float(statistic), float(pvalue)


def compute_chi_square(current: Series, baseline_proportions: Dict[str, float]) -> Tuple[float, float]:
    """Calculate Chi-square test statistic comparing current distribution to baseline proportions."""
    current_counts = current.value_counts()
    all_categories = sorted(set(list(current_counts.index) + list(baseline_proportions.keys())))
    observed, expected = [], []
    total_current = len(current)
    for cat in all_categories:
        observed.append(current_counts.get(cat, 0))
        expected.append(max(baseline_proportions.get(cat, 0) * total_current, 1e-10))
    expected_arr = np.array(expected)
    expected_arr = expected_arr * (sum(observed) / sum(expected_arr)) if sum(expected_arr) > 0 else expected_arr
    chi_square, pvalue = stats.chisquare(observed, expected_arr)
    return float(chi_square), float(pvalue)
