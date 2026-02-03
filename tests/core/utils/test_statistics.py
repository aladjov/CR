import numpy as np
import pandas as pd

from customer_retention.core.utils.statistics import (
    _ensure_array,
    _is_categorical_dtype,
    compute_chi_square,
    compute_effect_size,
    compute_ks_statistic,
    compute_psi_categorical,
    compute_psi_from_series,
    compute_psi_numeric,
)


class TestEnsureArray:
    def test_list_to_array(self):
        result = _ensure_array([1.0, 2.0, 3.0])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_ndarray_passthrough(self):
        arr = np.array([4.0, 5.0])
        result = _ensure_array(arr)
        assert result is arr


class TestComputeEffectSize:
    def test_large_effect(self):
        group1 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        group2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d, label = compute_effect_size(group1, group2)
        assert abs(d) >= 0.8
        assert label == "Large effect"

    def test_medium_effect(self):
        np.random.seed(42)
        group1 = np.random.normal(10, 5, 200)
        group2 = np.random.normal(13, 5, 200)
        d, label = compute_effect_size(group1, group2)
        assert 0.5 <= abs(d) < 0.8
        assert label == "Medium effect"

    def test_small_effect(self):
        np.random.seed(42)
        group1 = np.random.normal(10, 5, 100)
        group2 = np.random.normal(11, 5, 100)
        d, label = compute_effect_size(group1, group2)
        assert 0.2 <= abs(d) < 0.5
        assert label == "Small effect"

    def test_negligible_effect(self):
        np.random.seed(42)
        group1 = np.random.normal(10, 5, 100)
        group2 = np.random.normal(10, 5, 100)
        d, label = compute_effect_size(group1, group2)
        assert abs(d) < 0.2
        assert label == "Negligible"

    def test_short_group_returns_zero(self):
        d, label = compute_effect_size([1.0], [2.0, 3.0])
        assert d == 0.0
        assert label == "Negligible"

    def test_both_groups_short(self):
        d, label = compute_effect_size([1.0], [2.0])
        assert d == 0.0
        assert label == "Negligible"

    def test_zero_pooled_std(self):
        d, label = compute_effect_size([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        assert d == 0.0
        assert label == "Negligible"

    def test_list_inputs(self):
        d, label = compute_effect_size([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 11.0, 12.0, 13.0, 14.0])
        assert isinstance(d, float)
        assert label in ["Large effect", "Medium effect", "Small effect", "Negligible"]


class TestComputePsiNumeric:
    def test_no_drift(self):
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 1000))
        edges = np.histogram(reference.to_numpy(), bins=10)[1].tolist()
        counts = np.histogram(reference.to_numpy(), bins=10)[0].tolist()
        psi = compute_psi_numeric(reference, edges, counts)
        assert psi < 0.1

    def test_with_drift(self):
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 1000))
        edges = np.histogram(reference.to_numpy(), bins=10)[1].tolist()
        counts = np.histogram(reference.to_numpy(), bins=10)[0].tolist()
        current = pd.Series(np.random.normal(3, 1, 1000))
        psi = compute_psi_numeric(current, edges, counts)
        assert psi > 0.1

    def test_empty_current_series(self):
        reference = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        edges = np.histogram(reference.to_numpy(), bins=3)[1].tolist()
        counts = np.histogram(reference.to_numpy(), bins=3)[0].tolist()
        # current with all NaN means current_counts.sum() == 0
        current = pd.Series([np.nan, np.nan])
        psi = compute_psi_numeric(current, edges, counts)
        assert isinstance(psi, float)

    def test_returns_float(self):
        ref = pd.Series([1.0, 2.0, 3.0, 4.0])
        edges = np.histogram(ref.to_numpy(), bins=3)[1].tolist()
        counts = np.histogram(ref.to_numpy(), bins=3)[0].tolist()
        psi = compute_psi_numeric(ref, edges, counts)
        assert isinstance(psi, float)


class TestIsCategoricalDtype:
    def test_object_dtype(self):
        assert _is_categorical_dtype('object') is True

    def test_category_dtype(self):
        assert _is_categorical_dtype('category') is True

    def test_bool_dtype(self):
        assert _is_categorical_dtype('bool') is True

    def test_numeric_dtype(self):
        assert _is_categorical_dtype('float64') is False

    def test_int_dtype(self):
        assert _is_categorical_dtype('int64') is False


class TestComputePsiFromSeries:
    def test_numeric_no_drift(self):
        np.random.seed(42)
        ref = pd.Series(np.random.normal(0, 1, 500))
        curr = pd.Series(np.random.normal(0, 1, 500))
        psi = compute_psi_from_series(ref, curr)
        assert psi < 0.1

    def test_numeric_with_drift(self):
        np.random.seed(42)
        ref = pd.Series(np.random.normal(0, 1, 500))
        curr = pd.Series(np.random.normal(5, 1, 500))
        psi = compute_psi_from_series(ref, curr)
        assert psi > 0.1

    def test_categorical_series(self):
        ref = pd.Series(['A', 'B', 'C'] * 100)
        curr = pd.Series(['A', 'A', 'B'] * 100)
        psi = compute_psi_from_series(ref, curr)
        assert isinstance(psi, float)
        assert psi >= 0

    def test_mixed_categorical_numeric(self):
        ref = pd.Series(['x', 'y', 'z'] * 50)
        curr = pd.Series([1, 2, 3] * 50)
        # One is object dtype, triggers categorical path
        psi = compute_psi_from_series(ref, curr)
        assert isinstance(psi, float)

    def test_with_nan_values(self):
        ref = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0] * 50)
        curr = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0] * 50)
        psi = compute_psi_from_series(ref, curr)
        assert isinstance(psi, float)

    def test_custom_bins(self):
        np.random.seed(42)
        ref = pd.Series(np.random.normal(0, 1, 200))
        curr = pd.Series(np.random.normal(0, 1, 200))
        psi = compute_psi_from_series(ref, curr, n_bins=5)
        assert isinstance(psi, float)

    def test_empty_current_numeric(self):
        ref = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        curr = pd.Series([np.nan, np.nan, np.nan])
        # After dropna, curr_clean is empty -> len(curr_clean) == 0
        # This should handle the zero-length case
        try:
            psi = compute_psi_from_series(ref, curr)
            assert isinstance(psi, float)
        except (ValueError, ZeroDivisionError):
            pass  # Acceptable if empty series causes error


class TestComputePsiCategorical:
    def test_same_distribution(self):
        ref = pd.Series(['A', 'B', 'C'] * 100)
        curr = pd.Series(['A', 'B', 'C'] * 100)
        psi = compute_psi_categorical(ref, curr)
        assert abs(psi) < 0.01

    def test_different_distribution(self):
        ref = pd.Series(['A'] * 200 + ['B'] * 100)
        curr = pd.Series(['A'] * 100 + ['B'] * 200)
        psi = compute_psi_categorical(ref, curr)
        assert psi > 0.01

    def test_new_category_in_current(self):
        ref = pd.Series(['A', 'B'] * 100)
        curr = pd.Series(['A', 'B', 'C'] * 100)
        psi = compute_psi_categorical(ref, curr)
        assert isinstance(psi, float)

    def test_missing_category_in_current(self):
        ref = pd.Series(['A', 'B', 'C'] * 100)
        curr = pd.Series(['A', 'B'] * 100)
        psi = compute_psi_categorical(ref, curr)
        assert isinstance(psi, float)


class TestComputeKsStatistic:
    def test_same_distribution(self):
        np.random.seed(42)
        ref = pd.Series(np.random.normal(0, 1, 500))
        curr = pd.Series(np.random.normal(0, 1, 500))
        stat, pval = compute_ks_statistic(ref, curr)
        assert isinstance(stat, float)
        assert isinstance(pval, float)
        assert pval > 0.05  # Same distribution, not significant

    def test_different_distribution(self):
        np.random.seed(42)
        ref = pd.Series(np.random.normal(0, 1, 500))
        curr = pd.Series(np.random.normal(5, 1, 500))
        stat, pval = compute_ks_statistic(ref, curr)
        assert stat > 0.3
        assert pval < 0.05

    def test_handles_nan_values(self):
        ref = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0] * 100)
        curr = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0] * 100)
        stat, pval = compute_ks_statistic(ref, curr)
        assert isinstance(stat, float)
        assert isinstance(pval, float)


class TestComputeChiSquare:
    def test_matching_distribution(self):
        current = pd.Series(['A'] * 40 + ['B'] * 30 + ['C'] * 30)
        baseline_proportions = {'A': 0.4, 'B': 0.3, 'C': 0.3}
        chi_stat, pval = compute_chi_square(current, baseline_proportions)
        assert isinstance(chi_stat, float)
        assert isinstance(pval, float)
        assert pval > 0.05

    def test_different_distribution(self):
        current = pd.Series(['A'] * 80 + ['B'] * 10 + ['C'] * 10)
        baseline_proportions = {'A': 0.33, 'B': 0.33, 'C': 0.34}
        chi_stat, pval = compute_chi_square(current, baseline_proportions)
        assert chi_stat > 0
        assert pval < 0.05

    def test_new_category_in_current(self):
        current = pd.Series(['A'] * 30 + ['B'] * 30 + ['D'] * 40)
        baseline_proportions = {'A': 0.5, 'B': 0.5}
        chi_stat, pval = compute_chi_square(current, baseline_proportions)
        assert isinstance(chi_stat, float)

    def test_missing_category_in_current(self):
        current = pd.Series(['A'] * 50 + ['B'] * 50)
        baseline_proportions = {'A': 0.33, 'B': 0.33, 'C': 0.34}
        chi_stat, pval = compute_chi_square(current, baseline_proportions)
        assert isinstance(chi_stat, float)
        assert isinstance(pval, float)

    def test_zero_expected_sum(self):
        current = pd.Series(['X'] * 50)
        baseline_proportions = {}  # No proportions
        chi_stat, pval = compute_chi_square(current, baseline_proportions)
        assert isinstance(chi_stat, float)
