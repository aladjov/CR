from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat.bulk_profiling import (
    BulkStats,
    CategoricalDistributionBulkResult,
    DatetimeAnalysisBulkResult,
    DatetimeDiscoveryCandidateStats,
    DistributionBulkResult,
    HistogramData,
    NumericColumnStats,
    PerColumnStats,
    RangeValidationBulkResult,
    _chunked,
    _pandas_bulk_categorical_stats,
    _pandas_bulk_datetime_analysis,
    _pandas_bulk_distribution_stats,
    _pandas_bulk_stats,
    _pandas_bulk_validate_ranges,
    _safe_float,
    _safe_int,
    bulk_categorical_distribution_stats,
    bulk_datetime_analysis_stats,
    bulk_datetime_discovery_stats,
    bulk_distribution_stats,
    bulk_future_fractions,
    bulk_histogram,
    bulk_monthly_counts,
    bulk_nunique,
    bulk_validate_ranges,
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


class TestSparkBulkStatsNullAggregates:
    """Spark agg results may return None despite F.coalesce wrapping."""

    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def _run_with_row1(self, row1_dict, mode_dict, mode_count_dict, cols, numeric_fields=None):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_stats

        mock_row1 = MagicMock()
        mock_row1.__getitem__ = lambda self, k: row1_dict[k]
        mock_mode_row = MagicMock()
        mock_mode_row.__getitem__ = lambda self, k: mode_dict[k]
        mock_mode_count_row = MagicMock()
        mock_mode_count_row.__getitem__ = lambda self, k: mode_count_dict[k]

        schema_fields = []
        for c in cols:
            field = MagicMock()
            field.name = c
            field.dataType = MagicMock()
            field.dataType.__class__ = type("StringType", (), {})
            schema_fields.append(field)

        mock_spark_df = MagicMock()
        mock_spark_df.columns = cols
        mock_spark_df.schema.fields = schema_fields
        mock_spark_df.agg.return_value.collect.side_effect = [
            [mock_row1], [mock_mode_row], [mock_mode_count_row],
        ]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_spark_df):
            return _spark_bulk_stats(MagicMock())

    def test_null_count_none_from_spark(self):
        result = self._run_with_row1(
            row1_dict={"__total_count__": 5, "__null__a": None, "__dist__a": 3},
            mode_dict={"__mode__a": "x"},
            mode_count_dict={"__mcount__a": 2},
            cols=["a"],
        )
        assert result.columns["a"].null_count == 0

    def test_distinct_count_none_from_spark(self):
        result = self._run_with_row1(
            row1_dict={"__total_count__": 5, "__null__a": 2, "__dist__a": None},
            mode_dict={"__mode__a": "x"},
            mode_count_dict={"__mcount__a": 2},
            cols=["a"],
        )
        assert result.columns["a"].distinct_count == 0

    def test_both_null_and_distinct_none(self):
        result = self._run_with_row1(
            row1_dict={"__total_count__": 0, "__null__a": None, "__dist__a": None},
            mode_dict={"__mode__a": None},
            mode_count_dict={"__mcount__a": 0},
            cols=["a"],
        )
        assert result.columns["a"].null_count == 0
        assert result.columns["a"].distinct_count == 0
        assert result.columns["a"].most_common_value is None


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


class TestBulkHistogram:
    def test_basic_histogram(self):
        df = pd.DataFrame({"x": np.arange(100, dtype=float)})
        hist = bulk_histogram(df, "x", nbins=10)
        assert len(hist.counts) == 10
        assert len(hist.bin_edges) == 11
        assert sum(hist.counts) == 100
        assert hist.bin_edges[0] == pytest.approx(0.0, abs=0.01)
        assert hist.bin_edges[-1] == pytest.approx(99.0, abs=0.01)

    def test_bin_centers(self):
        hist = HistogramData(bin_edges=[0.0, 1.0, 2.0], counts=[5, 3])
        assert hist.bin_centers == [0.5, 1.5]

    def test_missing_column(self):
        df = pd.DataFrame({"y": [1, 2, 3]})
        hist = bulk_histogram(df, "x", nbins=5)
        assert hist.counts == []
        assert hist.bin_edges == []

    def test_all_nulls(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        hist = bulk_histogram(df, "x", nbins=5)
        assert hist.counts == []

    def test_constant_column(self):
        df = pd.DataFrame({"x": [5.0] * 50})
        hist = bulk_histogram(df, "x", nbins=10)
        assert hist.counts == []

    def test_with_inf_and_nan(self):
        values = list(range(50)) + [np.nan, np.inf, -np.inf]
        df = pd.DataFrame({"x": [float(v) for v in values]})
        hist = bulk_histogram(df, "x", nbins=10)
        assert sum(hist.counts) == 50

    def test_nbins_respected(self):
        df = pd.DataFrame({"x": np.arange(1000, dtype=float)})
        hist = bulk_histogram(df, "x", nbins=20)
        assert len(hist.counts) == 20
        assert len(hist.bin_edges) == 21


class TestBulkMonthlyCounts:
    def test_basic_monthly_counts(self):
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        df = pd.DataFrame({"dt": dates})
        result = bulk_monthly_counts(df, "dt")
        assert len(result) == 12
        assert result[0][0] == "2023-01"
        assert result[-1][0] == "2023-12"
        assert sum(c for _, c in result) == 365

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1, 2]})
        result = bulk_monthly_counts(df, "dt")
        assert result == []

    def test_all_nulls(self):
        df = pd.DataFrame({"dt": pd.Series([pd.NaT, pd.NaT])})
        result = bulk_monthly_counts(df, "dt")
        assert result == []

    def test_sorted_output(self):
        dates = pd.to_datetime(["2023-06-01", "2023-01-01", "2023-03-01", "2023-01-15"])
        df = pd.DataFrame({"dt": dates})
        result = bulk_monthly_counts(df, "dt")
        months = [m for m, _ in result]
        assert months == sorted(months)

    def test_single_month(self):
        dates = pd.to_datetime(["2023-05-01", "2023-05-15", "2023-05-28"])
        df = pd.DataFrame({"dt": dates})
        result = bulk_monthly_counts(df, "dt")
        assert len(result) == 1
        assert result[0] == ("2023-05", 3)


class TestBulkNunique:
    def test_basic(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "x", "y"]})
        result = bulk_nunique(df, ["a", "b"])
        assert result == {"a": 3, "b": 2}

    def test_none_columns_uses_all(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 3]})
        result = bulk_nunique(df)
        assert result == {"a": 2, "b": 1}

    def test_empty_columns(self):
        df = pd.DataFrame({"a": [1]})
        result = bulk_nunique(df, [])
        assert result == {}


class TestBulkDatetimeDiscoveryStats:
    def test_basic(self):
        df = pd.DataFrame({"dt": pd.date_range("2020-01-01", periods=10)})
        result = bulk_datetime_discovery_stats(df, ["dt"])
        assert "dt" in result
        assert result["dt"].coverage == 1.0
        assert result["dt"].future_fraction == 0.0
        assert result["dt"].min_date is not None

    def test_empty_columns(self):
        df = pd.DataFrame({"dt": pd.date_range("2020-01-01", periods=5)})
        result = bulk_datetime_discovery_stats(df, [])
        assert result == {}

    def test_with_nulls(self):
        df = pd.DataFrame({"dt": [pd.Timestamp("2020-01-01"), pd.NaT, pd.NaT]})
        result = bulk_datetime_discovery_stats(df, ["dt"])
        assert result["dt"].coverage == pytest.approx(1 / 3)


class TestPandasDatetimeStatsEdgeCases:
    """Cover _pandas_datetime_stats error branches."""

    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_non_datetime_conversion_failure(self):
        from customer_retention.core.compat.bulk_profiling import _pandas_datetime_stats

        series = pd.Series(["not", "dates", "at", "all"])
        result = _pandas_datetime_stats(series)
        # All values coerce to NaT → returns empty stats
        assert result.min_date is None or result.date_range_days is None

    def test_overflow_values(self):
        from customer_retention.core.compat.bulk_profiling import _pandas_datetime_stats

        series = pd.Series([10**18, 10**19, 10**20])
        result = _pandas_datetime_stats(series)
        # Should handle gracefully without raising
        assert isinstance(result, type(result))


class TestPandasBinaryStatsEdgeCases:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_all_null_binary(self):
        from customer_retention.core.compat.bulk_profiling import _pandas_binary_stats

        series = pd.Series([None, None, None], dtype="object")
        result = _pandas_binary_stats(series, None)
        assert result.true_count == 0
        assert result.false_count == 0

    def test_custom_binary_values(self):
        """Cover the fallback when no TRUE/FALSE values matched."""
        from customer_retention.core.compat.bulk_profiling import _pandas_binary_stats

        series = pd.Series(["active", "active", "inactive"])
        result = _pandas_binary_stats(series, None)
        assert result.true_count + result.false_count > 0


class TestPandasTextStatsEdgeCases:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_all_null_text(self):
        from customer_retention.core.compat.bulk_profiling import _pandas_text_stats

        series = pd.Series([None, None, None], dtype="object")
        result = _pandas_text_stats(series, 3)
        assert result.length_min == 0
        assert result.length_max == 0


class TestSparkBulkHelpersMocked:
    """Mock-based tests for Spark helper functions to improve coverage."""

    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def _make_mock_spark_df(self, agg_result_dict):
        """Create a mock Spark DataFrame that returns given agg results."""
        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: agg_result_dict.get(key, 0)
        mock_df.agg.return_value.collect.return_value = [mock_row]
        mock_df.count.return_value = len(agg_result_dict)
        return mock_df

    def test_spark_bulk_datetime_discovery(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_datetime_discovery

        result_map = {
            "__min__dt": pd.Timestamp("2020-01-01"),
            "__max__dt": pd.Timestamp("2023-12-31"),
            "__cnt__dt": 100,
            "__fut__dt": 5,
        }
        mock_df = self._make_mock_spark_df(result_map)
        mock_df.count.return_value = 100

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_df):
            result = _spark_bulk_datetime_discovery(MagicMock(), ["dt"])

        assert "dt" in result
        assert result["dt"].coverage == 1.0
        assert result["dt"].future_fraction == 0.05

    def test_spark_bulk_future_fractions(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_future_fractions

        result_map = {"__total__": 200, "__fut__col_a": 10, "__fut__col_b": 0}
        mock_df = self._make_mock_spark_df(result_map)

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_df):
            result = _spark_bulk_future_fractions(MagicMock(), "ref", ["col_a", "col_b"])

        assert result["col_a"] == pytest.approx(10 / 200)
        assert result["col_b"] == 0.0

    def test_spark_bulk_histogram(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_histogram

        bounds_row = MagicMock()
        bounds_row.__getitem__ = lambda self, key: {"__lo__": 0.0, "__hi__": 100.0}.get(key)
        hist_row = MagicMock()
        hist_row.__getitem__ = lambda self, key: 5  # 5 per bin

        mock_df = MagicMock()
        mock_df.agg.return_value.collect.side_effect = [[bounds_row], [hist_row]]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_df):
            result = _spark_bulk_histogram(MagicMock(), "val", nbins=10)

        assert len(result.bin_edges) == 11
        assert len(result.counts) == 10

    def test_spark_bulk_histogram_empty(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_histogram

        bounds_row = MagicMock()
        bounds_row.__getitem__ = lambda self, key: None

        mock_df = MagicMock()
        mock_df.agg.return_value.collect.return_value = [bounds_row]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_df):
            result = _spark_bulk_histogram(MagicMock(), "val", nbins=20)

        assert result.bin_edges == []
        assert result.counts == []

    def test_spark_bulk_monthly_counts(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_monthly_counts

        mock_rows = [
            MagicMock(**{"__getitem__": lambda s, k: {"month": "2023-01", "cnt": 10}[k]}),
            MagicMock(**{"__getitem__": lambda s, k: {"month": "2023-02", "cnt": 20}[k]}),
        ]
        mock_df = MagicMock()
        mock_df.filter.return_value.groupBy.return_value.agg.return_value.orderBy.return_value.collect.return_value = mock_rows

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_df):
            result = _spark_bulk_monthly_counts(MagicMock(), "dt")

        assert len(result) == 2

    def test_spark_bulk_nunique(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_nunique

        result_map = {"__dist__a": 5, "__dist__b": 10}
        mock_df = self._make_mock_spark_df(result_map)

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_df):
            result = _spark_bulk_nunique(MagicMock(), ["a", "b"])

        assert result == {"a": 5, "b": 10}


class TestChunked:
    def test_exact_chunks(self):
        cols = ["a", "b", "c", "d"]
        assert _chunked(cols, 2) == [["a", "b"], ["c", "d"]]

    def test_remainder(self):
        cols = ["a", "b", "c", "d", "e"]
        result = _chunked(cols, 2)
        assert result == [["a", "b"], ["c", "d"], ["e"]]

    def test_empty(self):
        assert _chunked([], 10) == []

    def test_single_chunk(self):
        cols = ["a", "b"]
        assert _chunked(cols, 100) == [["a", "b"]]

    def test_chunk_size_one(self):
        cols = ["a", "b", "c"]
        assert _chunked(cols, 1) == [["a"], ["b"], ["c"]]


class TestBulkValidateRanges:
    def test_percentage_rule(self):
        df = pd.DataFrame({"pct": [50.0, 110.0, -5.0, 80.0, 99.0]})
        rules = {"pct": {"type": "percentage", "min": 0, "max": 100}}
        result = bulk_validate_ranges(df, rules)
        assert "pct" in result
        assert result["pct"].invalid_count == 2
        assert result["pct"].non_null_count == 5

    def test_binary_rule(self):
        df = pd.DataFrame({"flag": [0, 1, 0, 1, 2, -1]})
        rules = {"flag": {"type": "binary", "valid_values": [0, 1]}}
        result = bulk_validate_ranges(df, rules)
        assert result["flag"].invalid_count == 2

    def test_non_negative_rule(self):
        df = pd.DataFrame({"count": [5, 10, -1, 0, 20]})
        rules = {"count": {"type": "non_negative"}}
        result = bulk_validate_ranges(df, rules)
        assert result["count"].invalid_count == 1

    def test_rate_rule(self):
        df = pd.DataFrame({"rate": [0.0, 0.5, 1.0, 1.5, -0.1]})
        rules = {"rate": {"type": "rate"}}
        result = bulk_validate_ranges(df, rules)
        assert result["rate"].invalid_count == 2

    def test_range_rule(self):
        df = pd.DataFrame({"val": [1.0, 5.0, 15.0, -2.0]})
        rules = {"val": {"type": "range", "min": 0, "max": 10}}
        result = bulk_validate_ranges(df, rules)
        assert result["val"].invalid_count == 2

    def test_null_handling(self):
        df = pd.DataFrame({"pct": [50.0, None, 110.0, None, 80.0]})
        rules = {"pct": {"type": "percentage", "min": 0, "max": 100}}
        result = bulk_validate_ranges(df, rules)
        assert result["pct"].non_null_count == 3
        assert result["pct"].invalid_count == 1

    def test_empty_rules(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert bulk_validate_ranges(df, {}) == {}

    def test_all_valid(self):
        df = pd.DataFrame({"pct": [10.0, 50.0, 90.0]})
        rules = {"pct": {"type": "percentage", "min": 0, "max": 100}}
        result = bulk_validate_ranges(df, rules)
        assert result["pct"].invalid_count == 0

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"x": [1, 2]})
        rules = {"y": {"type": "non_negative"}}
        result = bulk_validate_ranges(df, rules)
        assert "y" not in result

    def test_actual_range(self):
        df = pd.DataFrame({"val": [1.0, 5.0, 10.0]})
        rules = {"val": {"type": "range", "min": 0, "max": 100}}
        result = bulk_validate_ranges(df, rules)
        assert result["val"].actual_min == pytest.approx(1.0)
        assert result["val"].actual_max == pytest.approx(10.0)

    def test_all_null_column(self):
        df = pd.DataFrame({"val": pd.array([None, None, None], dtype="Float64")})
        rules = {"val": {"type": "non_negative"}}
        result = bulk_validate_ranges(df, rules)
        assert result["val"].non_null_count == 0

    def test_non_numeric_column_skipped(self):
        df = pd.DataFrame({"label": ["a", "b", "c"], "val": [1.0, 5.0, 10.0]})
        rules = {
            "label": {"type": "binary", "valid_values": [0, 1]},
            "val": {"type": "range", "min": 0, "max": 100},
        }
        result = bulk_validate_ranges(df, rules)
        assert "label" not in result
        assert "val" in result
        assert result["val"].invalid_count == 0

    def test_spark_dispatch(self):
        mock_df = MagicMock()
        mock_df.to_spark = MagicMock()
        rules = {"a": {"type": "non_negative"}}

        with patch("customer_retention.core.compat.bulk_profiling._spark_bulk_validate_ranges") as mock_spark:
            mock_spark.return_value = {"a": RangeValidationBulkResult(non_null_count=10, invalid_count=1)}
            result = bulk_validate_ranges(mock_df, rules)
            mock_spark.assert_called_once()
            assert result["a"].invalid_count == 1


class TestBulkDistributionStats:
    def test_normal_distribution(self):
        np.random.seed(42)
        df = pd.DataFrame({"val": np.random.normal(100, 15, 1000)})
        result = bulk_distribution_stats(df, ["val"])
        assert "val" in result
        br = result["val"]
        assert br.non_null_count == 1000
        assert 95 < br.mean < 105
        assert 12 < br.std < 18
        assert br.q1 is not None
        assert br.q3 is not None
        assert br.q1 < br.median < br.q3

    def test_skewed_distribution(self):
        np.random.seed(42)
        df = pd.DataFrame({"val": np.random.lognormal(3, 1, 1000)})
        result = bulk_distribution_stats(df, ["val"])
        br = result["val"]
        assert br.skewness > 0  # log-normal is right-skewed

    def test_zero_inflated(self):
        np.random.seed(42)
        values = np.random.exponential(10, 1000)
        mask = np.random.random(1000) < 0.4
        values[mask] = 0
        df = pd.DataFrame({"val": values})
        result = bulk_distribution_stats(df, ["val"])
        br = result["val"]
        assert br.zero_count > 300

    def test_outlier_counts(self):
        data = list(range(100)) + [1000, -500]
        df = pd.DataFrame({"val": [float(x) for x in data]})
        result = bulk_distribution_stats(df, ["val"])
        assert result["val"].outlier_count_iqr > 0

    def test_percentiles(self):
        df = pd.DataFrame({"val": list(range(1000))})
        result = bulk_distribution_stats(df, ["val"])
        pct = result["val"].percentiles
        assert "p1" in pct
        assert "p50" in pct
        assert "p99" in pct
        assert pct["p1"] < pct["p50"] < pct["p99"]

    def test_empty_column(self):
        df = pd.DataFrame({"val": pd.array([None, None], dtype="Float64")})
        result = bulk_distribution_stats(df, ["val"])
        assert result["val"].non_null_count == 0

    def test_empty_columns_list(self):
        df = pd.DataFrame({"val": [1, 2, 3]})
        assert bulk_distribution_stats(df, []) == {}

    def test_missing_column(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = bulk_distribution_stats(df, ["y"])
        assert "y" not in result

    def test_negative_counts(self):
        df = pd.DataFrame({"val": [-5.0, -3.0, 0.0, 2.0, 4.0]})
        result = bulk_distribution_stats(df, ["val"])
        assert result["val"].negative_count == 2

    def test_multiple_columns(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.normal(0, 1, 100),
            "b": np.random.normal(50, 10, 100),
        })
        result = bulk_distribution_stats(df, ["a", "b"])
        assert "a" in result
        assert "b" in result
        assert result["b"].mean > result["a"].mean

    def test_spark_dispatch(self):
        mock_df = MagicMock()
        mock_df.to_spark = MagicMock()
        mock_df.columns = ["a"]

        with patch("customer_retention.core.compat.bulk_profiling._spark_bulk_distribution_stats") as mock_spark:
            mock_spark.return_value = {"a": DistributionBulkResult(non_null_count=50, mean=10.0)}
            result = bulk_distribution_stats(mock_df, ["a"])
            mock_spark.assert_called_once()
            assert result["a"].mean == 10.0


class TestBulkCategoricalDistributionStats:
    def test_basic(self):
        df = pd.DataFrame({"cat": ["a", "b", "b", "c", "c", "c"]})
        result = bulk_categorical_distribution_stats(df, ["cat"])
        assert "cat" in result
        br = result["cat"]
        assert br.total_count == 6
        assert br.category_count == 3
        assert br.value_counts["c"] == 3
        assert br.value_counts["b"] == 2
        assert br.top1_concentration == pytest.approx(50.0)

    def test_imbalance_ratio(self):
        df = pd.DataFrame({"cat": ["a"] * 100 + ["b"]})
        result = bulk_categorical_distribution_stats(df, ["cat"])
        assert result["cat"].imbalance_ratio == 100.0

    def test_entropy(self):
        df = pd.DataFrame({"cat": ["a", "b", "c", "d"]})
        result = bulk_categorical_distribution_stats(df, ["cat"])
        assert result["cat"].normalized_entropy == pytest.approx(1.0, abs=0.01)

    def test_rare_categories(self):
        df = pd.DataFrame({"cat": ["a"] * 100 + ["b"]})
        result = bulk_categorical_distribution_stats(df, ["cat"], rare_threshold=0.05)
        assert result["cat"].rare_category_count == 1
        assert "b" in result["cat"].rare_category_names

    def test_top3_concentration(self):
        df = pd.DataFrame({"cat": ["a"] * 50 + ["b"] * 30 + ["c"] * 20})
        result = bulk_categorical_distribution_stats(df, ["cat"])
        assert result["cat"].top3_concentration == 100.0

    def test_empty_column(self):
        df = pd.DataFrame({"cat": [None, None, None]})
        result = bulk_categorical_distribution_stats(df, ["cat"])
        assert result["cat"].total_count == 0

    def test_multiple_columns(self):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "p", "q"]})
        result = bulk_categorical_distribution_stats(df, ["a", "b"])
        assert result["a"].category_count == 3
        assert result["b"].category_count == 2

    def test_empty_columns_list(self):
        df = pd.DataFrame({"x": ["a"]})
        assert bulk_categorical_distribution_stats(df, []) == {}

    def test_missing_column(self):
        df = pd.DataFrame({"x": ["a"]})
        result = bulk_categorical_distribution_stats(df, ["y"])
        assert "y" not in result

    def test_top_n_limits_value_counts(self):
        cats = [f"cat_{i}" for i in range(50)]
        df = pd.DataFrame({"cat": cats})
        result = bulk_categorical_distribution_stats(df, ["cat"], top_n=10)
        assert len(result["cat"].value_counts) == 10

    def test_spark_dispatch(self):
        mock_df = MagicMock()
        mock_df.to_spark = MagicMock()
        mock_df.columns = ["a"]

        with patch("customer_retention.core.compat.bulk_profiling._spark_bulk_categorical_stats") as mock_spark:
            mock_spark.return_value = {"a": CategoricalDistributionBulkResult(
                total_count=100, category_count=5,
            )}
            result = bulk_categorical_distribution_stats(mock_df, ["a"])
            mock_spark.assert_called_once()
            assert result["a"].category_count == 5


class TestBulkDatetimeAnalysisStats:
    def test_basic(self):
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        df = pd.DataFrame({"dt": dates})
        result = bulk_datetime_analysis_stats(df, ["dt"])
        assert "dt" in result
        br = result["dt"]
        assert br.total_count == 365
        assert br.null_count == 0
        assert br.span_days == 364
        assert br.min_date is not None
        assert br.max_date is not None
        assert len(br.monthly_counts) == 12
        assert sum(br.dow_counts) == 365

    def test_with_nulls(self):
        df = pd.DataFrame({"dt": [pd.Timestamp("2023-01-01"), pd.NaT, pd.NaT]})
        result = bulk_datetime_analysis_stats(df, ["dt"])
        assert result["dt"].null_count == 2
        assert result["dt"].total_count == 3

    def test_all_null(self):
        df = pd.DataFrame({"dt": pd.Series([pd.NaT, pd.NaT])})
        result = bulk_datetime_analysis_stats(df, ["dt"])
        br = result["dt"]
        assert br.min_date is None
        assert br.monthly_counts == []

    def test_multiple_columns(self):
        df = pd.DataFrame({
            "created": pd.date_range("2023-01-01", periods=10),
            "updated": pd.date_range("2023-06-01", periods=10),
        })
        result = bulk_datetime_analysis_stats(df, ["created", "updated"])
        assert result["created"].min_date < result["updated"].min_date

    def test_empty_columns(self):
        df = pd.DataFrame({"dt": pd.date_range("2020-01-01", periods=5)})
        assert bulk_datetime_analysis_stats(df, []) == {}

    def test_dow_counts_sum(self):
        dates = pd.date_range("2023-01-02", periods=7, freq="D")  # Mon-Sun
        df = pd.DataFrame({"dt": dates})
        result = bulk_datetime_analysis_stats(df, ["dt"])
        assert sum(result["dt"].dow_counts) == 7
        assert all(c == 1 for c in result["dt"].dow_counts)

    def test_spark_dispatch(self):
        mock_df = MagicMock()
        mock_df.to_spark = MagicMock()
        mock_df.columns = ["dt"]

        with patch("customer_retention.core.compat.bulk_profiling._spark_bulk_datetime_analysis") as mock_spark:
            mock_spark.return_value = {"dt": DatetimeAnalysisBulkResult(total_count=100)}
            result = bulk_datetime_analysis_stats(mock_df, ["dt"])
            mock_spark.assert_called_once()
            assert result["dt"].total_count == 100
