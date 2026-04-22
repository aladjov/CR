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
    bulk_binary_flags,
    bulk_categorical_distribution_stats,
    bulk_datetime_analysis_stats,
    bulk_datetime_discovery_stats,
    bulk_distribution_stats,
    bulk_future_fractions,
    bulk_histogram,
    bulk_histograms,
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

    def test_empty_dataframe_skips_mode_count_batch(self, caplog):
        """Regression: empty Spark DataFrame made batch1c agg of pure F.lit(0) literals
        return zero rows, causing IndexError on collect()[0]. Must skip the batch."""
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_stats

        mock_row1 = MagicMock()
        mock_row1.__getitem__ = lambda self, k: {
            "__total_count__": 0, "__null__a": 0, "__dist__a": 0,
            "__null__b": 0, "__dist__b": 0,
        }[k]
        mock_mode_row = MagicMock()
        mock_mode_row.__getitem__ = lambda self, k: {"__mode__a": None, "__mode__b": None}[k]

        mock_spark_df = MagicMock()
        mock_spark_df.columns = ["a", "b"]
        mock_spark_df.schema.fields = []
        mock_spark_df.agg.return_value.collect.side_effect = [
            [mock_row1], [mock_mode_row],
            AssertionError("batch1c must be skipped when all modes are None"),
        ]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_spark_df):
            with caplog.at_level("WARNING"):
                result = _spark_bulk_stats(MagicMock())

        assert result.total_count == 0
        assert result.columns["a"].most_common_value is None
        assert result.columns["a"].most_common_frequency is None
        assert result.columns["b"].most_common_value is None
        assert result.columns["b"].most_common_frequency is None
        assert any("empty DataFrame reached profiling" in r.message for r in caplog.records)

    def test_empty_dataframe_with_mixed_modes_handles_empty_collect(self):
        """If Spark returns [] for batch1c (defensive), fall back to mode_freq=None."""
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_stats

        mock_row1 = MagicMock()
        mock_row1.__getitem__ = lambda self, k: {
            "__total_count__": 0, "__null__a": 0, "__dist__a": 0,
        }[k]
        mock_mode_row = MagicMock()
        mock_mode_row.__getitem__ = lambda self, k: {"__mode__a": "x"}[k]

        mock_spark_df = MagicMock()
        mock_spark_df.columns = ["a"]
        mock_spark_df.schema.fields = []
        mock_spark_df.agg.return_value.collect.side_effect = [
            [mock_row1], [mock_mode_row], [],
        ]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_spark_df):
            result = _spark_bulk_stats(MagicMock())

        assert result.columns["a"].most_common_value == "x"
        assert result.columns["a"].most_common_frequency is None


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


class TestBulkHistograms:
    def test_multiple_columns(self):
        df = pd.DataFrame({"a": np.arange(100, dtype=float), "b": np.arange(100, 200, dtype=float)})
        result = bulk_histograms(df, ["a", "b"], nbins=10)
        assert "a" in result and "b" in result
        assert len(result["a"].counts) == 10
        assert len(result["b"].counts) == 10
        assert sum(result["a"].counts) == 100
        assert sum(result["b"].counts) == 100

    def test_empty_columns_list(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        result = bulk_histograms(df, [], nbins=5)
        assert result == {}

    def test_missing_column_excluded(self):
        df = pd.DataFrame({"x": np.arange(50, dtype=float)})
        result = bulk_histograms(df, ["x", "missing"], nbins=5)
        assert len(result["x"].counts) == 5
        assert "missing" not in result

    def test_constant_column_returns_empty(self):
        df = pd.DataFrame({"a": np.arange(50, dtype=float), "const": [3.0] * 50})
        result = bulk_histograms(df, ["a", "const"], nbins=5)
        assert len(result["a"].counts) == 5
        assert result["const"].counts == []

    def test_all_null_column(self):
        df = pd.DataFrame({"a": np.arange(50, dtype=float), "nul": [np.nan] * 50})
        result = bulk_histograms(df, ["a", "nul"], nbins=5)
        assert len(result["a"].counts) == 5
        assert result["nul"].counts == []

    def test_matches_single_column_bulk_histogram(self):
        df = pd.DataFrame({"x": np.random.normal(0, 1, 500)})
        single = bulk_histogram(df, "x", nbins=20)
        batch = bulk_histograms(df, ["x"], nbins=20)
        assert batch["x"].counts == single.counts
        assert batch["x"].bin_edges == single.bin_edges

    def test_non_numeric_column_returns_empty(self):
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
        result = bulk_histograms(df, ["num", "cat"], nbins=5)
        assert len(result["num"].counts) == 5
        assert result["cat"].counts == []


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


class TestBulkBinaryFlags:
    def test_pure_binary_ints(self):
        df = pd.DataFrame({"a": [0, 1, 1, 0], "b": [0, 0, 0, 0], "c": [1, 1, 1, 1]})
        result = bulk_binary_flags(df, ["a", "b", "c"])
        assert result == {"a": True, "b": True, "c": True}

    def test_binary_floats(self):
        df = pd.DataFrame({"a": [0.0, 1.0, 0.0], "b": [0.5, 0.5, 0.5]})
        result = bulk_binary_flags(df, ["a", "b"])
        assert result["a"] is True
        assert result["b"] is False

    def test_not_binary_more_than_two_values(self):
        df = pd.DataFrame({"a": [0, 1, 2]})
        result = bulk_binary_flags(df, ["a"])
        assert result == {"a": False}

    def test_not_binary_out_of_range(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = bulk_binary_flags(df, ["a"])
        assert result == {"a": False}

    def test_all_null(self):
        df = pd.DataFrame({"a": [None, None, None]})
        result = bulk_binary_flags(df, ["a"])
        assert result == {"a": False}

    def test_ignores_nulls(self):
        df = pd.DataFrame({"a": [0, 1, None, 1, None]})
        result = bulk_binary_flags(df, ["a"])
        assert result == {"a": True}

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"a": [0, 1]})
        result = bulk_binary_flags(df, ["a", "missing"])
        assert result == {"a": True}

    def test_empty_columns(self):
        df = pd.DataFrame({"a": [0, 1]})
        assert bulk_binary_flags(df, []) == {}


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

    def test_spark_bulk_nunique_batched(self):
        from customer_retention.core.compat.bulk_profiling import _NUNIQUE_BATCH, _spark_bulk_nunique

        columns = [f"c{i}" for i in range(_NUNIQUE_BATCH * 2 + 3)]
        result_map = {f"__dist__{c}": i + 1 for i, c in enumerate(columns)}
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, k: result_map.get(k, 0)
        mock_df = MagicMock()
        mock_df.agg.return_value.collect.return_value = [mock_row]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_df):
            result = _spark_bulk_nunique(MagicMock(), columns)

        assert mock_df.agg.call_count == 3
        assert result[columns[0]] == 1
        assert result[columns[-1]] == len(columns)

    def test_spark_bulk_binary_flags(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_binary_flags

        result_map = {
            "__min__a": 0.0, "__max__a": 1.0, "__dist__a": 2,
            "__min__b": 0.0, "__max__b": 0.5, "__dist__b": 2,
            "__min__c": 0.0, "__max__c": 0.0, "__dist__c": 1,
            "__min__d": None, "__max__d": None, "__dist__d": 0,
        }
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, k: result_map.get(k)
        mock_df = MagicMock()
        mock_df.agg.return_value.collect.return_value = [mock_row]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_df):
            result = _spark_bulk_binary_flags(MagicMock(), ["a", "b", "c", "d"])

        assert result == {"a": True, "b": False, "c": True, "d": False}

    def test_spark_bulk_binary_flags_batched(self):
        from customer_retention.core.compat.bulk_profiling import _BINARY_BATCH, _spark_bulk_binary_flags

        columns = [f"c{i}" for i in range(_BINARY_BATCH + 5)]
        result_map = {}
        for c in columns:
            result_map[f"__min__{c}"] = 0.0
            result_map[f"__max__{c}"] = 1.0
            result_map[f"__dist__{c}"] = 2
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, k: result_map.get(k)
        mock_df = MagicMock()
        mock_df.agg.return_value.collect.return_value = [mock_row]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=mock_df):
            result = _spark_bulk_binary_flags(MagicMock(), columns)

        assert mock_df.agg.call_count == 2
        assert all(result[c] for c in columns)


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

    def test_unparseable_strings_coerced(self):
        df = pd.DataFrame({"dt": ["2024-01-01", "N/A", "not-a-date", None, "2024-06-15"]})
        result = bulk_datetime_analysis_stats(df, ["dt"])
        br = result["dt"]
        assert br.total_count == 5
        assert br.null_count == 3
        assert br.min_date is not None
        assert br.max_date is not None
        assert br.span_days > 0

    def test_all_unparseable_strings(self):
        df = pd.DataFrame({"dt": ["N/A", "UNKNOWN", "---"]})
        result = bulk_datetime_analysis_stats(df, ["dt"])
        br = result["dt"]
        assert br.null_count == 3
        assert br.min_date is None
        assert br.monthly_counts == []

    def test_placeholder_count_detects_old_dates(self):
        df = pd.DataFrame({"dt": [
            pd.Timestamp("1900-01-01"), pd.Timestamp("1970-01-01"),
            pd.Timestamp("2023-06-15"), pd.Timestamp("2023-07-20"),
        ]})
        result = bulk_datetime_analysis_stats(df, ["dt"])
        assert result["dt"].placeholder_count == 2

    def test_placeholder_count_zero_when_no_old_dates(self):
        df = pd.DataFrame({"dt": pd.date_range("2023-01-01", periods=10)})
        result = bulk_datetime_analysis_stats(df, ["dt"])
        assert result["dt"].placeholder_count == 0

    def test_placeholder_count_all_null(self):
        df = pd.DataFrame({"dt": pd.Series([pd.NaT, pd.NaT])})
        result = bulk_datetime_analysis_stats(df, ["dt"])
        assert result["dt"].placeholder_count == 0

    def test_spark_dispatch(self):
        mock_df = MagicMock()
        mock_df.to_spark = MagicMock()
        mock_df.columns = ["dt"]

        with patch("customer_retention.core.compat.bulk_profiling._spark_bulk_datetime_analysis") as mock_spark:
            mock_spark.return_value = {"dt": DatetimeAnalysisBulkResult(total_count=100)}
            result = bulk_datetime_analysis_stats(mock_df, ["dt"])
            mock_spark.assert_called_once()
            assert result["dt"].total_count == 100


class TestBatchAdversarialDiffs:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_identical_features_no_drifts(self):
        from customer_retention.core.compat.bulk_profiling import batch_adversarial_diffs

        gold = pd.DataFrame({
            "eid": ["a", "b", "c", "d"],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [10.0, 20.0, 30.0, 40.0],
            "target": [1, 0, pd.NA, pd.NA],
            "original_target": [pd.NA, pd.NA, 1, 0],
        })
        score = gold[gold["original_target"].notna()].copy()
        n, diffs = batch_adversarial_diffs(gold, score, "eid", "target", "original_target")
        assert n == 2
        assert len(diffs) == 0

    def test_detects_numeric_drift(self):
        from customer_retention.core.compat.bulk_profiling import batch_adversarial_diffs

        gold = pd.DataFrame({
            "eid": ["a", "b", "c"],
            "f1": [1.0, 2.0, 3.0],
            "target": [1, pd.NA, pd.NA],
            "original_target": [pd.NA, 0, 1],
        })
        score = pd.DataFrame({
            "eid": ["b", "c"],
            "f1": [2.5, 3.0],
            "target": [pd.NA, pd.NA],
            "original_target": [0, 1],
        })
        n, diffs = batch_adversarial_diffs(gold, score, "eid", "target", "original_target")
        assert n == 2
        assert "f1" in diffs
        assert diffs["f1"][0] == pytest.approx(0.5)

    def test_detects_categorical_drift(self):
        from customer_retention.core.compat.bulk_profiling import batch_adversarial_diffs

        gold = pd.DataFrame({
            "eid": ["a", "b"],
            "cat": ["x", "y"],
            "target": [pd.NA, pd.NA],
            "original_target": [0, 1],
        })
        score = pd.DataFrame({
            "eid": ["a", "b"],
            "cat": ["x", "CHANGED"],
            "target": [pd.NA, pd.NA],
            "original_target": [0, 1],
        })
        n, diffs = batch_adversarial_diffs(gold, score, "eid", "target", "original_target")
        assert n == 2
        assert "cat" in diffs
        assert diffs["cat"][2] == 1

    def test_empty_holdout(self):
        from customer_retention.core.compat.bulk_profiling import batch_adversarial_diffs

        gold = pd.DataFrame({
            "eid": ["a", "b"],
            "f1": [1.0, 2.0],
            "target": [0, 1],
            "original_target": [pd.NA, pd.NA],
        })
        n, diffs = batch_adversarial_diffs(gold, gold, "eid", "target", "original_target")
        assert n == 0
        assert len(diffs) == 0

    def test_spark_dispatch(self):
        from customer_retention.core.compat.bulk_profiling import batch_adversarial_diffs

        mock_df = MagicMock()
        mock_df.to_spark = MagicMock()

        with patch("customer_retention.core.compat.bulk_profiling._spark_adversarial_diffs") as mock_spark:
            mock_spark.return_value = (5, {"f1": (0.1, 0.05, 3)})
            n, diffs = batch_adversarial_diffs(mock_df, pd.DataFrame(), "eid", "target", "orig")
            mock_spark.assert_called_once()
            assert n == 5
            assert "f1" in diffs

    def test_spark_path_uses_as_spark_df_for_distributed_score(self):
        """score_df may be pyspark.pandas — must use as_spark_df, not pandas_dtype_to_spark_schema."""
        from customer_retention.core.compat.bulk_profiling import batch_adversarial_diffs

        score_ps = MagicMock()
        score_ps.to_spark = MagicMock()
        score_ps.columns = ["eid", "f1"]

        with (
            patch("customer_retention.core.compat.bulk_profiling._spark_adversarial_diffs") as mock_spark,
            patch("customer_retention.core.compat.pandas_dtype_to_spark_schema") as mock_schema,
        ):
            mock_spark.return_value = (2, {})
            batch_adversarial_diffs(score_ps, score_ps, "eid", "target", "orig")
            mock_spark.assert_called_once()
            mock_schema.assert_not_called()


# ===========================================================================
# Coverage-driven tests for uncovered Spark and pandas paths
# ===========================================================================


def _row(d):
    """Build a MagicMock row that subscripts into a dict."""
    r = MagicMock()
    r.__getitem__ = lambda self, key: d.get(key)
    r.asDict = MagicMock(return_value=d)
    return r


def _agg_returning(rows):
    """Build a MagicMock spark DataFrame whose .agg(...).collect() returns rows."""
    sdf = MagicMock()
    sdf.agg.return_value.collect.return_value = rows
    return sdf


class TestSparkBulkHistogramsBatched:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_batched_histograms_for_multiple_columns(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_histograms

        # Two columns: col_a (lo=0, hi=10), col_b (lo=5, hi=15)
        bounds_row = _row({
            "__lo__col_a": 0.0, "__hi__col_a": 10.0,
            "__lo__col_b": 5.0, "__hi__col_b": 15.0,
        })
        # Histogram counts: 5 columns × nbins => use uniform 7 each
        hist_row = _row({f"__hb_{c}_{i}__": 7 for c in ("col_a", "col_b") for i in range(5)})

        sdf = MagicMock()
        sdf.agg.return_value.collect.side_effect = [[bounds_row], [hist_row]]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_histograms(MagicMock(), ["col_a", "col_b"], nbins=5)

        assert "col_a" in result
        assert "col_b" in result
        assert len(result["col_a"].counts) == 5
        assert all(c == 7 for c in result["col_a"].counts)

    def test_skip_degenerate_columns(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_histograms

        bounds_row = _row({
            "__lo__col_a": None, "__hi__col_a": None,
            "__lo__col_b": 5.0, "__hi__col_b": 5.0,  # lo >= hi
            "__lo__col_c": 0.0, "__hi__col_c": 10.0,  # valid
        })
        hist_row = _row({f"__hb_col_c_{i}__": 3 for i in range(4)})
        sdf = MagicMock()
        sdf.agg.return_value.collect.side_effect = [[bounds_row], [hist_row]]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_histograms(MagicMock(), ["col_a", "col_b", "col_c"], nbins=4)

        assert result["col_a"].counts == []
        assert result["col_b"].counts == []
        assert result["col_c"].counts == [3, 3, 3, 3]

    def test_all_degenerate_short_circuits(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_histograms

        bounds_row = _row({"__lo__col_a": None, "__hi__col_a": None})
        sdf = MagicMock()
        sdf.agg.return_value.collect.return_value = [bounds_row]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_histograms(MagicMock(), ["col_a"], nbins=10)

        assert result["col_a"].counts == []
        # Only bounds agg should run; no histogram agg
        sdf.agg.assert_called_once()


class TestSparkBulkStatsNumeric:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_full_numeric_path(self):
        pytest.importorskip("pyspark")
        from pyspark.sql.types import DoubleType, StringType, StructField

        from customer_retention.core.compat.bulk_profiling import _spark_bulk_stats

        sdf = MagicMock()
        sdf.columns = ["num1", "txt1"]
        sdf.schema.fields = [
            StructField("num1", DoubleType()),
            StructField("txt1", StringType()),
        ]

        # Batch 1: count + null + distinct
        row1 = _row({
            "__total_count__": 100,
            "__null__num1": 5, "__dist__num1": 50,
            "__null__txt1": 0, "__dist__txt1": 10,
        })
        # Batch 1b: mode values
        mode_row = _row({"__mode__num1": 1.0, "__mode__txt1": "common"})
        # Batch 1c: mode counts
        mode_count = _row({"__mcount__num1": 30, "__mcount__txt1": 25})
        # Batch 2: numeric stats
        row2 = _row({
            "__mean__num1": 1.5, "__std__num1": 0.5,
            "__min__num1": 0.0, "__max__num1": 10.0,
            "__q1__num1": 0.5, "__med__num1": 1.5, "__q3__num1": 2.5,
            "__skew__num1": 0.1, "__kurt__num1": 0.2,
        })
        # Batch 3: counts/outliers
        row3 = _row({
            "__zero__num1": 3, "__neg__num1": 2, "__inf__num1": 0,
            "__oiqr__num1": 4, "__ozscore__num1": 6,
        })
        # Batch 4: histograms
        hist_row = _row({f"__hist_{i}__num1": 10 for i in range(10)})

        sdf.agg.return_value.collect.side_effect = [
            [row1], [mode_row], [mode_count], [row2], [row3], [hist_row],
        ]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_stats(MagicMock())

        assert result.total_count == 100
        assert result.columns["num1"].null_count == 5
        assert result.columns["num1"].most_common_value == "1.0"
        assert result.columns["num1"].most_common_frequency == 30
        assert "num1" in result.numeric
        n_stats = result.numeric["num1"]
        assert n_stats.mean == 1.5
        assert n_stats.zero_count == 3
        assert n_stats.outlier_count_iqr == 4
        assert n_stats.non_null_count == 95  # 100 - 5 nulls
        assert len(n_stats.histogram_bins) == 10

    def test_no_numeric_cols_short_circuit(self):
        pytest.importorskip("pyspark")
        from pyspark.sql.types import StringType, StructField

        from customer_retention.core.compat.bulk_profiling import _spark_bulk_stats

        sdf = MagicMock()
        sdf.columns = ["txt"]
        sdf.schema.fields = [StructField("txt", StringType())]

        row1 = _row({"__total_count__": 50, "__null__txt": 0, "__dist__txt": 5})
        mode_row = _row({"__mode__txt": "x"})
        mode_count = _row({"__mcount__txt": 10})
        sdf.agg.return_value.collect.side_effect = [[row1], [mode_row], [mode_count]]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_stats(MagicMock())

        assert result.numeric == {}
        # Mode value present even for non-numeric
        assert result.columns["txt"].most_common_value == "x"

    def test_mode_none_yields_no_frequency(self):
        pytest.importorskip("pyspark")
        from pyspark.sql.types import StringType, StructField

        from customer_retention.core.compat.bulk_profiling import _spark_bulk_stats

        sdf = MagicMock()
        sdf.columns = ["all_null"]
        sdf.schema.fields = [StructField("all_null", StringType())]

        row1 = _row({"__total_count__": 0, "__null__all_null": 0, "__dist__all_null": 0})
        mode_row = _row({"__mode__all_null": None})
        mode_count = _row({"__mcount__all_null": 0})
        sdf.agg.return_value.collect.side_effect = [[row1], [mode_row], [mode_count]]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_stats(MagicMock())

        assert result.columns["all_null"].most_common_value is None
        assert result.columns["all_null"].most_common_frequency is None

    def test_degenerate_min_max_skips_histogram(self):
        pytest.importorskip("pyspark")
        from pyspark.sql.types import DoubleType, StructField

        from customer_retention.core.compat.bulk_profiling import _spark_bulk_stats

        sdf = MagicMock()
        sdf.columns = ["c"]
        sdf.schema.fields = [StructField("c", DoubleType())]

        row1 = _row({"__total_count__": 10, "__null__c": 0, "__dist__c": 1})
        mode_row = _row({"__mode__c": 5.0})
        mode_count = _row({"__mcount__c": 10})
        # min == max → degenerate
        row2 = _row({
            "__mean__c": 5.0, "__std__c": 0.0,
            "__min__c": 5.0, "__max__c": 5.0,
            "__q1__c": 5.0, "__med__c": 5.0, "__q3__c": 5.0,
            "__skew__c": None, "__kurt__c": None,
        })
        row3 = _row({
            "__zero__c": 0, "__neg__c": 0, "__inf__c": 0,
            "__oiqr__c": 0, "__ozscore__c": 0,
        })
        # No histogram batch since all degenerate
        sdf.agg.return_value.collect.side_effect = [[row1], [mode_row], [mode_count], [row2], [row3]]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_stats(MagicMock())

        assert result.numeric["c"].histogram_bins == []
        # 5 batches called (no histogram)
        assert sdf.agg.return_value.collect.call_count == 5


class TestSparkBulkValidateRanges:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_range_rule_validation(self):
        from pyspark.sql.types import DoubleType, StructField

        from customer_retention.core.compat.bulk_profiling import _spark_bulk_validate_ranges

        sdf = MagicMock()
        sdf.columns = ["age"]
        sdf.schema.fields = [StructField("age", DoubleType())]
        sdf.agg.return_value.collect.return_value = [_row({
            "__cnt__age": 100, "__min__age": 0, "__max__age": 150, "__inv__age": 5,
        })]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_validate_ranges(
                MagicMock(),
                {"age": {"type": "range", "min": 0, "max": 120}},
                chunk_size=200,
            )

        assert result["age"].invalid_count == 5
        assert result["age"].non_null_count == 100

    def test_returns_empty_when_no_numeric_rules(self):
        from pyspark.sql.types import StringType, StructField

        from customer_retention.core.compat.bulk_profiling import _spark_bulk_validate_ranges

        sdf = MagicMock()
        sdf.columns = ["name"]
        sdf.schema.fields = [StructField("name", StringType())]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_validate_ranges(
                MagicMock(),
                {"name": {"type": "range", "min": 0, "max": 100}},
                chunk_size=200,
            )

        assert result == {}


class TestSparkInvalidExpr:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_percentage_rule(self):
        import pyspark.sql.functions as F  # noqa: N812

        from customer_retention.core.compat.bulk_profiling import _spark_invalid_expr

        c = F.col("pct")
        expr = _spark_invalid_expr(c, "percentage", {}, F)
        assert "0" in str(expr)
        assert "100" in str(expr)

    def test_binary_rule(self):
        import pyspark.sql.functions as F  # noqa: N812

        from customer_retention.core.compat.bulk_profiling import _spark_invalid_expr

        c = F.col("flag")
        expr = _spark_invalid_expr(c, "binary", {"valid_values": [0, 1]}, F)
        assert expr is not None

    def test_non_negative_rule(self):
        import pyspark.sql.functions as F  # noqa: N812

        from customer_retention.core.compat.bulk_profiling import _spark_invalid_expr

        c = F.col("count")
        expr = _spark_invalid_expr(c, "non_negative", {}, F)
        # Spark Connect renders operators in prefix form: <(count, 0)
        s = str(expr)
        assert "0" in s
        assert "isNotNull" in s

    def test_rate_rule(self):
        import pyspark.sql.functions as F  # noqa: N812

        from customer_retention.core.compat.bulk_profiling import _spark_invalid_expr

        c = F.col("ratio")
        expr = _spark_invalid_expr(c, "rate", {}, F)
        s = str(expr)
        # Should reference both bound checks (0 and 1)
        assert "0" in s
        assert "1" in s
        assert "or" in s.lower()

    def test_general_range_with_min_only(self):
        import pyspark.sql.functions as F  # noqa: N812

        from customer_retention.core.compat.bulk_profiling import _spark_invalid_expr

        c = F.col("v")
        expr = _spark_invalid_expr(c, "range", {"min": 5}, F)
        s = str(expr)
        assert "5" in s
        assert "isNotNull" in s

    def test_general_range_with_max_only(self):
        import pyspark.sql.functions as F  # noqa: N812

        from customer_retention.core.compat.bulk_profiling import _spark_invalid_expr

        c = F.col("v")
        expr = _spark_invalid_expr(c, "range", {"max": 99}, F)
        s = str(expr)
        assert "99" in s
        assert "isNotNull" in s

    def test_general_range_no_bounds_returns_false_lit(self):
        import pyspark.sql.functions as F  # noqa: N812

        from customer_retention.core.compat.bulk_profiling import _spark_invalid_expr

        c = F.col("v")
        expr = _spark_invalid_expr(c, "range", {}, F)
        assert "false" in str(expr).lower()


class TestSparkBulkDistributionStats:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_distribution_stats_full_path(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_distribution_stats

        # Pass 1 row: full stats for col 'x'
        pct_arr = [0.01, 0.05, 0.10, 1.0, 2.0, 3.0, 4.0, 5.0, 9.99]
        pass1_row = _row({
            "__cnt__x": 100, "__avg__x": 2.5, "__std__x": 1.0,
            "__min__x": 0.0, "__max__x": 10.0,
            "__pct__x": pct_arr,
            "__skw__x": 0.1, "__krt__x": 0.2,
            "__zer__x": 5, "__neg__x": 0,
        })
        # Pass 2 row: outlier counts (the asDict path)
        pass2_row = MagicMock()
        pass2_dict = {"__out__x": 7}
        pass2_row.__getitem__ = lambda self, key: pass2_dict.get(key)
        pass2_row.asDict = MagicMock(return_value=pass2_dict)

        sdf = MagicMock()
        sdf.agg.return_value.collect.side_effect = [[pass1_row], [pass2_row]]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_distribution_stats(MagicMock(), ["x"], chunk_size=200)

        assert result["x"].non_null_count == 100
        assert result["x"].mean == 2.5
        assert result["x"].q1 == 1.0
        assert result["x"].q3 == 3.0
        assert result["x"].outlier_count_iqr == 7
        assert "p1" in result["x"].percentiles

    def test_zero_count_columns_yield_empty_results(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_distribution_stats

        pass1_row = _row({
            "__cnt__y": 0, "__avg__y": None, "__std__y": None,
            "__min__y": None, "__max__y": None, "__pct__y": [None] * 9,
            "__skw__y": None, "__krt__y": None, "__zer__y": 0, "__neg__y": 0,
        })

        sdf = MagicMock()
        sdf.agg.return_value.collect.return_value = [pass1_row]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_distribution_stats(MagicMock(), ["y"], chunk_size=200)

        assert result["y"].non_null_count == 0
        assert result["y"].mean is None


class TestSparkBulkCategoricalStats:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_categorical_stats_basic(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_categorical_stats

        # Pass 1: counts row
        count_row = _row({"__cnt__color": 100})
        # Pass 2 vc rows
        vc_rows = [
            _row({"__col__": "color", "__val__": "red", "count": 60}),
            _row({"__col__": "color", "__val__": "blue", "count": 30}),
            _row({"__col__": "color", "__val__": "green", "count": 10}),
        ]

        # First select for stack_part (returns intermediate select)
        stack_select = MagicMock()
        stack_select.filter.return_value = stack_select
        stack_select.unionAll.return_value = stack_select

        sdf = MagicMock()
        sdf.agg.return_value.collect.return_value = [count_row]
        sdf.select.return_value = stack_select

        # The stacked.groupBy().count().collect() returns vc_rows
        stack_select.groupBy.return_value.count.return_value.collect.return_value = vc_rows

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_categorical_stats(MagicMock(), ["color"], top_n=5, rare_threshold=0.05)

        assert result["color"].total_count == 100
        assert result["color"].category_count == 3
        assert "red" in result["color"].value_counts

    def test_categorical_stats_zero_count_returns_empty(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_categorical_stats

        count_row = _row({"__cnt__empty_col": 0})
        sdf = MagicMock()
        sdf.agg.return_value.collect.return_value = [count_row]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_categorical_stats(MagicMock(), ["empty_col"], top_n=10, rare_threshold=0.01)

        assert result["empty_col"].total_count == 0
        assert result["empty_col"].category_count == 0


class TestSparkBulkDatetimeAnalysis:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_datetime_analysis_with_data(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_datetime_analysis

        mn = pd.Timestamp("2023-01-01")
        mx = pd.Timestamp("2023-12-31")
        pass1_row = _row({
            "__min__dt": mn, "__max__dt": mx, "__null__dt": 5, "__plc__dt": 0,
        })

        # Stacked select returns intermediate
        stack = MagicMock()
        stack.filter.return_value = stack
        stack.unionAll.return_value = stack

        # groupBy results for monthly + dow
        month_rows = [
            _row({"__col__": "dt", "__month__": "2023-01", "count": 10}),
            _row({"__col__": "dt", "__month__": "2023-02", "count": 15}),
        ]
        dow_rows = [
            _row({"__col__": "dt", "__dow__": 2, "count": 50}),  # Monday in Spark dayofweek
            _row({"__col__": "dt", "__dow__": 6, "count": 20}),  # Friday in Spark dayofweek
        ]
        stack.groupBy.return_value.count.return_value.collect.side_effect = [month_rows, dow_rows]

        sdf = MagicMock()
        sdf.count.return_value = 365
        sdf.agg.return_value.collect.return_value = [pass1_row]
        sdf.select.return_value = stack

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_datetime_analysis(MagicMock(), ["dt"])

        assert result["dt"].total_count == 365
        assert result["dt"].null_count == 5
        assert result["dt"].span_days >= 364
        assert len(result["dt"].monthly_counts) == 2
        # Spark dow 2 (Mon) → Python idx 0, Spark dow 6 (Fri) → Python idx 4
        assert result["dt"].dow_counts[0] == 50
        assert result["dt"].dow_counts[4] == 20

    def test_datetime_analysis_skips_columns_with_no_data(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_datetime_analysis

        # pass1 returns None for min — column has no data → skip stack_parts
        pass1_row = _row({
            "__min__dt": None, "__max__dt": None, "__null__dt": 100, "__plc__dt": 0,
        })

        sdf = MagicMock()
        sdf.count.return_value = 100
        sdf.agg.return_value.collect.return_value = [pass1_row]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            result = _spark_bulk_datetime_analysis(MagicMock(), ["dt"])

        assert result["dt"].total_count == 100
        assert result["dt"].null_count == 100
        assert result["dt"].monthly_counts == []
        assert result["dt"].dow_counts == [0] * 7


class TestPandasDatetimeStatsErrorHandling:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_returns_empty_on_unconvertible_data(self):
        from customer_retention.core.compat.bulk_profiling import _pandas_datetime_stats

        # Force the to_datetime conversion exception path
        bad = pd.Series(["not_a_date"] * 5, dtype=object)
        result = _pandas_datetime_stats(bad)
        # Either coerced to NaT (returns empty) or returns proper empty stats
        assert result.future_date_count == 0

    def test_weekend_count_exception_handled(self):
        from customer_retention.core.compat.bulk_profiling import _pandas_datetime_stats

        clean = pd.to_datetime(["2024-01-01", "2024-01-06", "2024-01-07"])
        result = _pandas_datetime_stats(pd.Series(clean))
        # Mon, Sat, Sun → 2 weekend
        assert result.weekend_count == 2


class TestPandasDatetimeAnalysisExtras:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_pandas_no_data_returns_empty_stats(self):
        df = pd.DataFrame({"dt": pd.to_datetime([None, None])})
        result = _pandas_bulk_datetime_analysis(df, ["dt"])
        assert result["dt"].total_count == 2
        assert result["dt"].null_count == 2
        assert result["dt"].monthly_counts == []


class TestBulkNuniqueDispatcher:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_pandas_default_columns(self):
        from customer_retention.core.compat.bulk_profiling import bulk_nunique

        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
        result = bulk_nunique(df)
        assert result == {"a": 2, "b": 3}

    def test_empty_columns_returns_empty(self):
        from customer_retention.core.compat.bulk_profiling import bulk_nunique

        df = pd.DataFrame({"a": [1]})
        assert bulk_nunique(df, columns=[]) == {}

    def test_spark_dispatch(self):
        from customer_retention.core.compat.bulk_profiling import bulk_nunique

        mock_df = MagicMock()
        mock_df.to_spark = MagicMock()
        mock_df.columns = ["a", "b"]
        with patch("customer_retention.core.compat.bulk_profiling._spark_bulk_nunique") as mock_spark:
            mock_spark.return_value = {"a": 5, "b": 3}
            result = bulk_nunique(mock_df, columns=["a", "b"])
        assert result == {"a": 5, "b": 3}
        mock_spark.assert_called_once()


class TestBulkNullCountsDispatcher:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_pandas_with_progress_fn(self):
        from customer_retention.core.compat.bulk_profiling import bulk_null_counts

        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 5]})
        messages = []
        result = bulk_null_counts(df, progress_fn=lambda m: messages.append(m))
        assert result == {"a": 1, "b": 2}
        assert any("pandas" in m for m in messages)

    def test_empty_columns_returns_empty(self):
        from customer_retention.core.compat.bulk_profiling import bulk_null_counts

        df = pd.DataFrame({"a": [1, 2]})
        assert bulk_null_counts(df, columns=[]) == {}

    def test_invalid_columns_filtered_out(self):
        from customer_retention.core.compat.bulk_profiling import bulk_null_counts

        df = pd.DataFrame({"a": [1, None]})
        result = bulk_null_counts(df, columns=["a", "missing"])
        assert "a" in result
        assert "missing" not in result

    def test_spark_dispatch(self):
        from customer_retention.core.compat.bulk_profiling import bulk_null_counts

        mock_df = MagicMock()
        mock_df.to_spark = MagicMock()
        mock_df.columns = ["a", "b"]
        with patch("customer_retention.core.compat.bulk_profiling._spark_bulk_null_counts") as mock_spark:
            mock_spark.return_value = {"a": 0, "b": 5}
            result = bulk_null_counts(mock_df)
        assert result == {"a": 0, "b": 5}


class TestSparkBulkNullCounts:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_batched_null_counts(self):
        from customer_retention.core.compat.bulk_profiling import _spark_bulk_null_counts

        # 250 columns → 2 batches of 200 + 50
        cols = [f"c{i}" for i in range(250)]
        batch_a = _row({f"__null__c{i}": i for i in range(200)})
        batch_b = _row({f"__null__c{i}": i for i in range(200, 250)})

        sdf = MagicMock()
        sdf.agg.return_value.collect.side_effect = [[batch_a], [batch_b]]

        with patch("customer_retention.core.compat.bulk_profiling.as_spark_df", return_value=sdf):
            messages = []
            result = _spark_bulk_null_counts(MagicMock(), cols, log=lambda m: messages.append(m))

        assert len(result) == 250
        assert result["c0"] == 0
        assert result["c249"] == 249
        # Two batch progress messages
        assert len([m for m in messages if "batch" in m]) == 2


class TestBulkHistogramsDispatcher:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_empty_valid_columns_returns_empty(self):
        from customer_retention.core.compat.bulk_profiling import bulk_histograms

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = bulk_histograms(df, columns=["missing"], nbins=10)
        assert result == {}

    def test_spark_dispatch(self):
        from customer_retention.core.compat.bulk_profiling import bulk_histograms

        mock_df = MagicMock()
        mock_df.to_spark = MagicMock()
        mock_df.columns = ["a"]
        with patch("customer_retention.core.compat.bulk_profiling._spark_bulk_histograms") as mock_spark:
            mock_spark.return_value = {"a": HistogramData()}
            result = bulk_histograms(mock_df, columns=["a"], nbins=5)
        assert "a" in result
        mock_spark.assert_called_once()


class TestPandasBulkHistogramsEdgeCases:
    @pytest.fixture(autouse=True)
    def _skip_without_pyspark(self):
        pytest.importorskip("pyspark")

    def test_non_numeric_column_returns_empty_histogram(self):
        from customer_retention.core.compat.bulk_profiling import _pandas_bulk_histograms

        df = pd.DataFrame({"text": ["a", "b", "c"], "num": [1.0, 2.0, 3.0]})
        result = _pandas_bulk_histograms(df, ["text", "num"], nbins=5)
        assert result["text"].counts == []
        assert len(result["num"].counts) == 5

    def test_all_inf_yields_empty_histogram(self):
        from customer_retention.core.compat.bulk_profiling import _pandas_bulk_histograms

        df = pd.DataFrame({"v": [float("inf"), float("-inf"), float("nan")]})
        result = _pandas_bulk_histograms(df, ["v"], nbins=10)
        assert result["v"].counts == []

    def test_constant_column_yields_empty_histogram(self):
        from customer_retention.core.compat.bulk_profiling import _pandas_bulk_histograms

        df = pd.DataFrame({"c": [5.0, 5.0, 5.0]})
        result = _pandas_bulk_histograms(df, ["c"], nbins=10)
        assert result["c"].counts == []
