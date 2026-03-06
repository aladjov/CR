from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat.bulk_profiling import (
    BulkStats,
    NumericColumnStats,
    PerColumnStats,
    _pandas_bulk_stats,
    _safe_float,
    _safe_int,
    bulk_future_fractions,
    compute_bulk_stats,
)


class TestPerColumnStats:
    def test_fields(self):
        s = PerColumnStats(null_count=5, distinct_count=10)
        assert s.null_count == 5
        assert s.distinct_count == 10

    def test_mode_defaults(self):
        s = PerColumnStats(null_count=0, distinct_count=1)
        assert s.most_common_value is None
        assert s.most_common_frequency is None

    def test_mode_fields(self):
        s = PerColumnStats(
            null_count=0, distinct_count=3,
            most_common_value="apple", most_common_frequency=42,
        )
        assert s.most_common_value == "apple"
        assert s.most_common_frequency == 42


class TestNumericColumnStats:
    def test_defaults(self):
        s = NumericColumnStats()
        assert s.mean is None
        assert s.std is None
        assert s.zero_count == 0
        assert s.outlier_count_iqr == 0
        assert s.non_null_count == 0
        assert s.histogram_bins == []

    def test_all_fields(self):
        s = NumericColumnStats(
            mean=1.5,
            std=0.5,
            min_val=0.0,
            max_val=3.0,
            q1=1.0,
            median=1.5,
            q3=2.0,
            skewness=0.1,
            kurtosis=-0.2,
            zero_count=2,
            negative_count=0,
            inf_count=0,
            outlier_count_iqr=1,
            outlier_count_zscore=0,
            non_null_count=100,
            histogram_bins=[(0.0, 0.3, 5), (0.3, 0.6, 10)],
        )
        assert s.mean == 1.5
        assert s.q3 == 2.0
        assert s.outlier_count_iqr == 1
        assert s.non_null_count == 100
        assert len(s.histogram_bins) == 2


class TestBulkStats:
    def test_empty(self):
        b = BulkStats(total_count=0)
        assert b.total_count == 0
        assert b.columns == {}
        assert b.numeric == {}


class TestSafeFloat:
    def test_none(self):
        assert _safe_float(None) is None

    def test_nan(self):
        assert _safe_float(float("nan")) is None

    def test_normal(self):
        assert _safe_float(3.14) == 3.14

    def test_int(self):
        assert _safe_float(7) == 7.0

    def test_string(self):
        assert _safe_float("not_a_number") is None


class TestSafeInt:
    def test_none(self):
        assert _safe_int(None) == 0

    def test_normal(self):
        assert _safe_int(42) == 42

    def test_float_to_int(self):
        assert _safe_int(3.7) == 3

    def test_string(self):
        assert _safe_int("bad") == 0


class TestPandasBulkStatsBasic:
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = _pandas_bulk_stats(df)
        assert result.total_count == 0
        assert result.columns == {}
        assert result.numeric == {}

    def test_single_column_no_nulls(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = _pandas_bulk_stats(df)
        assert result.total_count == 5
        assert result.columns["a"].null_count == 0
        assert result.columns["a"].distinct_count == 5

    def test_total_count(self):
        df = pd.DataFrame({"x": range(100)})
        result = _pandas_bulk_stats(df)
        assert result.total_count == 100


class TestPandasBulkStatsNullsAndDistinct:
    def test_null_counts(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, None, 4, None],
                "b": ["x", "y", "z", None, "w"],
            }
        )
        result = _pandas_bulk_stats(df)
        assert result.columns["a"].null_count == 2
        assert result.columns["b"].null_count == 1

    def test_distinct_counts(self):
        df = pd.DataFrame(
            {
                "a": [1, 1, 2, 2, 3],
                "b": ["x", "x", "x", "x", "x"],
            }
        )
        result = _pandas_bulk_stats(df)
        assert result.columns["a"].distinct_count == 3
        assert result.columns["b"].distinct_count == 1

    def test_all_null_column(self):
        df = pd.DataFrame({"a": [None, None, None]})
        result = _pandas_bulk_stats(df)
        assert result.columns["a"].null_count == 3
        assert result.columns["a"].distinct_count == 0


class TestPandasBulkStatsNumeric:
    def test_numeric_basic_stats(self):
        np.random.seed(42)
        df = pd.DataFrame({"val": np.random.normal(50, 10, 1000)})
        result = _pandas_bulk_stats(df)

        assert "val" in result.numeric
        ns = result.numeric["val"]
        assert ns.mean is not None
        assert 45 < ns.mean < 55
        assert ns.std is not None
        assert 8 < ns.std < 12
        assert ns.min_val is not None
        assert ns.max_val is not None
        assert ns.q1 is not None
        assert ns.median is not None
        assert ns.q3 is not None

    def test_numeric_quartiles_order(self):
        df = pd.DataFrame({"val": range(100)})
        result = _pandas_bulk_stats(df)
        ns = result.numeric["val"]
        assert ns.q1 < ns.median < ns.q3

    def test_numeric_skewness_and_kurtosis(self):
        np.random.seed(42)
        df = pd.DataFrame({"val": np.random.exponential(2, 500)})
        result = _pandas_bulk_stats(df)
        ns = result.numeric["val"]
        assert ns.skewness is not None
        assert ns.skewness > 0  # exponential is right-skewed
        assert ns.kurtosis is not None

    def test_zero_count(self):
        df = pd.DataFrame({"val": [0, 0, 1, 2, 3]})
        result = _pandas_bulk_stats(df)
        assert result.numeric["val"].zero_count == 2

    def test_negative_count(self):
        df = pd.DataFrame({"val": [-5, -3, 0, 2, 4]})
        result = _pandas_bulk_stats(df)
        assert result.numeric["val"].negative_count == 2

    def test_inf_count(self):
        df = pd.DataFrame({"val": [1.0, float("inf"), 3.0, float("-inf"), 5.0]})
        result = _pandas_bulk_stats(df)
        assert result.numeric["val"].inf_count == 2

    def test_outlier_iqr_count(self):
        data = list(range(100)) + [1000, -500]
        df = pd.DataFrame({"val": data})
        result = _pandas_bulk_stats(df)
        assert result.numeric["val"].outlier_count_iqr > 0

    def test_outlier_zscore_count(self):
        np.random.seed(42)
        data = np.random.normal(50, 1, 200).tolist()
        data.extend([200, -100])
        df = pd.DataFrame({"val": data})
        result = _pandas_bulk_stats(df)
        assert result.numeric["val"].outlier_count_zscore > 0

    def test_constant_column_zero_std(self):
        df = pd.DataFrame({"val": [5, 5, 5, 5, 5]})
        result = _pandas_bulk_stats(df)
        ns = result.numeric["val"]
        assert ns.std == 0.0
        assert ns.outlier_count_zscore == 0

    def test_all_null_numeric_column(self):
        df = pd.DataFrame({"val": pd.array([None, None, None], dtype="Float64")})
        result = _pandas_bulk_stats(df)
        ns = result.numeric["val"]
        assert ns.mean is None


class TestPandasBulkStatsMixed:
    def test_mixed_columns(self):
        df = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0, 4.0, 5.0],
                "cat": ["a", "b", "c", "a", "b"],
                "dt": pd.date_range("2020-01-01", periods=5),
            }
        )
        result = _pandas_bulk_stats(df)
        assert result.total_count == 5
        assert len(result.columns) == 3
        assert "num" in result.numeric
        assert "cat" not in result.numeric
        assert "dt" not in result.numeric

    def test_non_numeric_columns_excluded(self):
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "active": [True, False, True],
            }
        )
        result = _pandas_bulk_stats(df)
        assert "name" not in result.numeric


class TestPandasBulkStatsMatchesProfiler:
    def test_null_count_matches(self):
        df = pd.DataFrame({"a": [1, 2, None, 4, None, 6]})
        result = _pandas_bulk_stats(df)
        expected_null = int(df["a"].isna().sum())
        assert result.columns["a"].null_count == expected_null

    def test_distinct_count_matches(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3, 3, 4]})
        result = _pandas_bulk_stats(df)
        expected_distinct = int(df["a"].nunique())
        assert result.columns["a"].distinct_count == expected_distinct

    def test_mean_matches_describe(self):
        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.normal(100, 15, 500)})
        result = _pandas_bulk_stats(df)
        expected = float(df["a"].describe()["mean"])
        assert abs(result.numeric["a"].mean - expected) < 1e-10


class TestComputeBulkStatsDispatch:
    def test_pandas_dispatch(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = compute_bulk_stats(df)
        assert isinstance(result, BulkStats)
        assert result.total_count == 3

    def test_spark_dispatch(self):
        mock_df = MagicMock()
        mock_df.to_spark = MagicMock()

        with patch("customer_retention.core.compat.bulk_profiling._spark_bulk_stats") as mock_spark:
            mock_spark.return_value = BulkStats(total_count=10)
            result = compute_bulk_stats(mock_df)
            mock_spark.assert_called_once_with(mock_df)
            assert result.total_count == 10


class TestPandasBulkStatsEdgeCases:
    def test_single_row(self):
        df = pd.DataFrame({"a": [42], "b": ["hello"]})
        result = _pandas_bulk_stats(df)
        assert result.total_count == 1
        assert result.columns["a"].null_count == 0
        assert result.columns["a"].distinct_count == 1
        assert result.numeric["a"].mean == 42.0

    def test_large_values(self):
        df = pd.DataFrame({"a": [1e15, 2e15, 3e15]})
        result = _pandas_bulk_stats(df)
        assert result.numeric["a"].mean is not None

    def test_boolean_as_numeric(self):
        df = pd.DataFrame({"flag": [True, False, True, True, False]})
        result = _pandas_bulk_stats(df)
        assert result.columns["flag"].distinct_count == 2


class TestPandasBulkStatsModeValues:
    def test_most_common_value_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 2, 3, 2]})
        result = _pandas_bulk_stats(df)
        assert result.columns["a"].most_common_value == "2"
        assert result.columns["a"].most_common_frequency == 3

    def test_most_common_value_string(self):
        df = pd.DataFrame({"cat": ["a", "b", "b", "c", "b"]})
        result = _pandas_bulk_stats(df)
        assert result.columns["cat"].most_common_value == "b"
        assert result.columns["cat"].most_common_frequency == 3

    def test_most_common_value_all_unique(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = _pandas_bulk_stats(df)
        assert result.columns["a"].most_common_value is not None
        assert result.columns["a"].most_common_frequency == 1

    def test_most_common_value_all_null(self):
        df = pd.DataFrame({"a": [None, None, None]})
        result = _pandas_bulk_stats(df)
        assert result.columns["a"].most_common_value is None
        assert result.columns["a"].most_common_frequency is None

    def test_most_common_value_mixed_types(self):
        df = pd.DataFrame({
            "num": [1, 2, 2, 3],
            "cat": ["x", "x", "y", "z"],
        })
        result = _pandas_bulk_stats(df)
        assert result.columns["num"].most_common_value == "2"
        assert result.columns["cat"].most_common_value == "x"


class TestPandasBulkStatsHistogram:
    def test_histogram_bins_count(self):
        df = pd.DataFrame({"val": range(100)})
        result = _pandas_bulk_stats(df)
        assert len(result.numeric["val"].histogram_bins) == 10

    def test_histogram_bins_cover_range(self):
        df = pd.DataFrame({"val": list(range(100))})
        result = _pandas_bulk_stats(df)
        bins = result.numeric["val"].histogram_bins
        assert bins[0][0] <= 0.0
        assert bins[-1][1] >= 99.0

    def test_histogram_bins_sum_to_total(self):
        df = pd.DataFrame({"val": np.random.normal(50, 10, 200)})
        result = _pandas_bulk_stats(df)
        bins = result.numeric["val"].histogram_bins
        total = sum(b[2] for b in bins)
        assert total == 200

    def test_histogram_constant_column(self):
        df = pd.DataFrame({"val": [5, 5, 5, 5, 5]})
        result = _pandas_bulk_stats(df)
        assert result.numeric["val"].histogram_bins == []

    def test_histogram_with_inf(self):
        df = pd.DataFrame({"val": [1.0, 2.0, float("inf"), 4.0, float("-inf")]})
        result = _pandas_bulk_stats(df)
        bins = result.numeric["val"].histogram_bins
        total = sum(b[2] for b in bins)
        assert total == 3  # only finite values

    def test_histogram_all_null(self):
        df = pd.DataFrame({"val": pd.array([None, None, None], dtype="Float64")})
        result = _pandas_bulk_stats(df)
        assert result.numeric["val"].histogram_bins == []

    def test_non_null_count(self):
        df = pd.DataFrame({"val": [1, 2, None, 4, None]})
        result = _pandas_bulk_stats(df)
        assert result.numeric["val"].non_null_count == 3

    def test_non_null_count_no_nulls(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        result = _pandas_bulk_stats(df)
        assert result.numeric["val"].non_null_count == 5


class TestBulkFutureFractions:
    def test_basic_future_detection(self):
        df = pd.DataFrame({
            "ref": pd.to_datetime(["2024-01-10", "2024-01-10", "2024-01-10", "2024-01-10"]),
            "col_a": pd.to_datetime(["2024-01-11", "2024-01-09", "2024-01-11", "2024-01-09"]),
            "col_b": pd.to_datetime(["2024-01-09", "2024-01-09", "2024-01-09", "2024-01-09"]),
        })
        result = bulk_future_fractions(df, "ref", ["col_a", "col_b"])
        assert result["col_a"] == 0.5
        assert result["col_b"] == 0.0

    def test_all_future(self):
        df = pd.DataFrame({
            "ref": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "col_a": pd.to_datetime(["2024-06-01", "2024-06-01"]),
        })
        result = bulk_future_fractions(df, "ref", ["col_a"])
        assert result["col_a"] == 1.0

    def test_no_future(self):
        df = pd.DataFrame({
            "ref": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "col_a": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        })
        result = bulk_future_fractions(df, "ref", ["col_a"])
        assert result["col_a"] == 0.0

    def test_empty_check_cols(self):
        df = pd.DataFrame({
            "ref": pd.to_datetime(["2024-01-01"]),
        })
        result = bulk_future_fractions(df, "ref", [])
        assert result == {}

    def test_missing_reference_col(self):
        df = pd.DataFrame({
            "col_a": pd.to_datetime(["2024-01-01"]),
        })
        result = bulk_future_fractions(df, "ref", ["col_a"])
        assert result == {}

    def test_nonexistent_check_col_skipped(self):
        df = pd.DataFrame({
            "ref": pd.to_datetime(["2024-01-01"]),
            "col_a": pd.to_datetime(["2024-06-01"]),
        })
        result = bulk_future_fractions(df, "ref", ["col_a", "nonexistent"])
        assert "col_a" in result
        assert "nonexistent" not in result

    def test_empty_dataframe(self):
        df = pd.DataFrame({"ref": pd.Series(dtype="datetime64[ns]"), "col_a": pd.Series(dtype="datetime64[ns]")})
        result = bulk_future_fractions(df, "ref", ["col_a"])
        assert result["col_a"] == 0.0
