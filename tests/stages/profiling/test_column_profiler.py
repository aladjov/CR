import pytest
import pandas as pd
import numpy as np
from customer_retention.stages.profiling import (
    ColumnProfiler, ProfilerFactory,
    UniversalMetrics, IdentifierMetrics, TargetMetrics,
    NumericMetrics, CategoricalMetrics, DatetimeMetrics, BinaryMetrics
)
from customer_retention.core.config import ColumnType


class TestUniversalMetrics:
    def test_compute_universal_metrics_basic(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([1, 2, 3, 4, 5, 5, 5])
        metrics = profiler.compute_universal_metrics(series)

        assert metrics.total_count == 7
        assert metrics.null_count == 0
        assert metrics.null_percentage == 0.0
        assert metrics.distinct_count == 5
        assert metrics.distinct_percentage == 71.43
        assert metrics.most_common_value == 5
        assert metrics.most_common_frequency == 3
        assert metrics.memory_size_bytes > 0

    def test_compute_universal_metrics_with_nulls(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([1, 2, None, 4, None, 5])
        metrics = profiler.compute_universal_metrics(series)

        assert metrics.total_count == 6
        assert metrics.null_count == 2
        assert metrics.null_percentage == 33.33
        assert metrics.distinct_count == 4

    def test_compute_universal_metrics_empty_series(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([], dtype=float)
        metrics = profiler.compute_universal_metrics(series)

        assert metrics.total_count == 0
        assert metrics.null_count == 0
        assert metrics.most_common_value is None
        assert metrics.most_common_frequency is None


class TestIdentifierProfiler:
    def test_profile_unique_identifiers(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.IDENTIFIER)
        series = pd.Series(["ABC123", "DEF456", "GHI789", "JKL012"])
        result = profiler.profile(series)

        assert "identifier_metrics" in result
        metrics = result["identifier_metrics"]
        assert metrics.is_unique is True
        assert metrics.duplicate_count == 0
        assert metrics.duplicate_values == []
        assert metrics.length_min == 6
        assert metrics.length_max == 6
        assert metrics.length_mode == 6

    def test_profile_with_duplicates(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.IDENTIFIER)
        series = pd.Series(["ID001", "ID002", "ID001", "ID003", "ID002"])
        result = profiler.profile(series)

        metrics = result["identifier_metrics"]
        assert metrics.is_unique is False
        assert metrics.duplicate_count == 2
        assert "ID001" in metrics.duplicate_values
        assert "ID002" in metrics.duplicate_values

    def test_profile_varying_length_ids(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.IDENTIFIER)
        series = pd.Series(["A1", "ABC123", "XY", "DEFGHI789"])
        result = profiler.profile(series)

        metrics = result["identifier_metrics"]
        assert metrics.length_min == 2
        assert metrics.length_max == 9


class TestTargetProfiler:
    def test_profile_binary_target(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        series = pd.Series([0, 1, 0, 1, 1, 0, 0])
        result = profiler.profile(series)

        assert "target_metrics" in result
        metrics = result["target_metrics"]
        assert metrics.n_classes == 2
        assert metrics.class_distribution == {"0": 4, "1": 3}
        assert metrics.class_percentages == {"0": 57.14, "1": 42.86}
        assert metrics.minority_class in [0, 1]
        assert metrics.imbalance_ratio == 1.33

    def test_profile_multiclass_target(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        series = pd.Series(["low", "medium", "high", "low", "low", "medium"])
        result = profiler.profile(series)

        metrics = result["target_metrics"]
        assert metrics.n_classes == 3
        assert metrics.class_distribution["low"] == 3
        assert metrics.class_distribution["medium"] == 2
        assert metrics.class_distribution["high"] == 1
        assert metrics.minority_class == "high"
        assert metrics.minority_percentage == 16.67

    def test_profile_imbalanced_target(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        series = pd.Series([0] * 95 + [1] * 5)
        result = profiler.profile(series)

        metrics = result["target_metrics"]
        assert metrics.imbalance_ratio == 19.0
        assert metrics.minority_percentage == 5.0


class TestNumericProfiler:
    def test_profile_continuous_numeric(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([1.5, 2.3, 3.7, 4.2, 5.8, 6.1, 7.9, 8.4, 9.2, 10.6])
        result = profiler.profile(series)

        assert "numeric_metrics" in result
        metrics = result["numeric_metrics"]
        assert metrics.mean > 0
        assert metrics.std > 0
        assert metrics.min_value == 1.5
        assert metrics.max_value == 10.6
        assert metrics.median > 0
        assert metrics.q1 > 0
        assert metrics.q3 > 0
        assert metrics.iqr > 0
        assert metrics.zero_count == 0
        assert metrics.negative_count == 0

    def test_profile_with_zeros_and_negatives(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([-5, -2, 0, 0, 0, 1, 3, 5, 8, 10])
        result = profiler.profile(series)

        metrics = result["numeric_metrics"]
        assert metrics.zero_count == 3
        assert metrics.zero_percentage == 30.0
        assert metrics.negative_count == 2
        assert metrics.negative_percentage == 20.0

    def test_profile_with_outliers(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = profiler.profile(series)

        metrics = result["numeric_metrics"]
        assert metrics.outlier_count_iqr > 0
        assert metrics.outlier_percentage > 0

    def test_profile_skewness_and_kurtosis(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series(np.random.normal(0, 1, 100))
        result = profiler.profile(series)

        metrics = result["numeric_metrics"]
        assert metrics.skewness is not None
        assert metrics.kurtosis is not None

    def test_profile_histogram_bins(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series(range(100))
        result = profiler.profile(series)

        metrics = result["numeric_metrics"]
        assert metrics.histogram_bins is not None
        assert len(metrics.histogram_bins) == 10

    def test_profile_with_infinite_values(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([1.0, 2.0, np.inf, 4.0, -np.inf, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = profiler.profile(series)

        metrics = result["numeric_metrics"]
        assert metrics.inf_count == 2
        assert metrics.inf_percentage == 20.0

    def test_profile_without_infinite_values(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = profiler.profile(series)

        metrics = result["numeric_metrics"]
        assert metrics.inf_count == 0
        assert metrics.inf_percentage == 0.0

    def test_profile_empty_numeric_series(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([], dtype=float)
        result = profiler.profile(series)

        assert result["numeric_metrics"] is None


class TestCategoricalProfiler:
    def test_profile_low_cardinality(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        series = pd.Series(["red", "blue", "green", "red", "blue", "red"])
        result = profiler.profile(series)

        assert "categorical_metrics" in result
        metrics = result["categorical_metrics"]
        assert metrics.cardinality == 3
        assert metrics.cardinality_ratio == 0.5
        assert metrics.value_counts["red"] == 3
        assert metrics.value_counts["blue"] == 2
        assert len(metrics.top_categories) == 3
        assert metrics.encoding_recommendation == "one_hot"

    def test_profile_high_cardinality(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        series = pd.Series([f"cat_{i}" for i in range(60)])
        result = profiler.profile(series)

        metrics = result["categorical_metrics"]
        assert metrics.cardinality == 60
        assert len(metrics.top_categories) == 10
        assert metrics.encoding_recommendation == "hashing_or_embedding"

    def test_profile_rare_categories(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        series = pd.Series(["common"] * 195 + ["rare1", "rare2", "rare3", "rare4", "rare5"])
        result = profiler.profile(series)

        metrics = result["categorical_metrics"]
        assert metrics.rare_category_count == 5
        assert metrics.rare_category_percentage <= 3.0

    def test_profile_contains_unknown_values(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        series = pd.Series(["red", "blue", "unknown", "green", "n/a", "red"])
        result = profiler.profile(series)

        metrics = result["categorical_metrics"]
        assert metrics.contains_unknown is True

    def test_encoding_recommendations(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)

        series_low = pd.Series(["a", "b", "c"] * 10)
        result_low = profiler.profile(series_low)
        assert result_low["categorical_metrics"].encoding_recommendation == "one_hot"

        series_medium = pd.Series([f"cat_{i % 12}" for i in range(100)])
        result_medium = profiler.profile(series_medium)
        assert result_medium["categorical_metrics"].encoding_recommendation == "one_hot_or_target"

        series_high = pd.Series([f"cat_{i % 40}" for i in range(100)])
        result_high = profiler.profile(series_high)
        assert result_high["categorical_metrics"].encoding_recommendation == "target_or_embedding"

    def test_profile_empty_categorical_series(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        series = pd.Series([], dtype=object)
        result = profiler.profile(series)

        assert result["categorical_metrics"] is None


class TestDatetimeProfiler:
    def test_profile_datetime_series(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        series = pd.Series(pd.date_range("2023-01-01", periods=365))
        result = profiler.profile(series)

        assert "datetime_metrics" in result
        metrics = result["datetime_metrics"]
        assert metrics.min_date == "2023-01-01 00:00:00"
        assert metrics.max_date == "2023-12-31 00:00:00"
        assert metrics.date_range_days == 364
        assert metrics.future_date_count == 0
        assert metrics.weekend_percentage > 0

    def test_profile_with_future_dates(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        future_date = pd.Timestamp.now() + pd.Timedelta(days=365)
        series = pd.Series([pd.Timestamp.now(), future_date])
        result = profiler.profile(series)

        metrics = result["datetime_metrics"]
        assert metrics.future_date_count == 1

    def test_profile_with_placeholder_dates(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        series = pd.Series([
            pd.Timestamp("1970-01-01"),
            pd.Timestamp("1900-01-01"),
            pd.Timestamp("2023-06-15"),
            pd.Timestamp("9999-12-31")
        ])
        result = profiler.profile(series)

        metrics = result["datetime_metrics"]
        assert metrics.placeholder_count == 3

    def test_profile_weekend_percentage(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        series = pd.Series([
            pd.Timestamp("2023-01-02"),  # Monday
            pd.Timestamp("2023-01-07"),  # Saturday
            pd.Timestamp("2023-01-08"),  # Sunday
        ])
        result = profiler.profile(series)

        metrics = result["datetime_metrics"]
        assert metrics.weekend_percentage == 66.67

    def test_profile_string_datetime(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        series = pd.Series(["2023-01-01", "2023-06-15", "2023-12-31"])
        result = profiler.profile(series)

        metrics = result["datetime_metrics"]
        assert metrics.date_range_days > 0

    def test_profile_empty_datetime_series(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        series = pd.Series([], dtype=object)
        result = profiler.profile(series)

        assert result["datetime_metrics"] is None


class TestBinaryProfiler:
    def test_profile_binary_zero_one(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.BINARY)
        series = pd.Series([0, 1, 0, 1, 1, 0, 0, 0])
        result = profiler.profile(series)

        assert "binary_metrics" in result
        metrics = result["binary_metrics"]
        assert metrics.true_count == 3
        assert metrics.false_count == 5
        assert metrics.true_percentage == 37.5
        assert metrics.balance_ratio == 1.67

    def test_profile_binary_boolean(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.BINARY)
        series = pd.Series([True, False, True, False, False])
        result = profiler.profile(series)

        metrics = result["binary_metrics"]
        assert metrics.true_count == 2
        assert metrics.false_count == 3
        assert metrics.is_boolean is True

    def test_profile_binary_yes_no(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.BINARY)
        series = pd.Series(["yes", "no", "yes", "yes", "no"])
        result = profiler.profile(series)

        metrics = result["binary_metrics"]
        assert metrics.true_count == 3
        assert metrics.false_count == 2
        assert "yes" in metrics.values_found or "Yes" in metrics.values_found

    def test_profile_binary_true_false_strings(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.BINARY)
        series = pd.Series(["true", "false", "true", "false"])
        result = profiler.profile(series)

        metrics = result["binary_metrics"]
        assert metrics.true_count == 2
        assert metrics.false_count == 2
        assert metrics.balance_ratio == 1.0

    def test_profile_imbalanced_binary(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.BINARY)
        series = pd.Series([0] * 90 + [1] * 10)
        result = profiler.profile(series)

        metrics = result["binary_metrics"]
        assert metrics.balance_ratio == 9.0

    def test_profile_empty_binary_series(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.BINARY)
        series = pd.Series([], dtype=object)
        result = profiler.profile(series)

        assert result["binary_metrics"] is None


class TestProfilerFactory:
    def test_get_profiler_for_all_types(self):
        assert ProfilerFactory.get_profiler(ColumnType.IDENTIFIER) is not None
        assert ProfilerFactory.get_profiler(ColumnType.TARGET) is not None
        assert ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS) is not None
        assert ProfilerFactory.get_profiler(ColumnType.NUMERIC_DISCRETE) is not None
        assert ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL) is not None
        assert ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_ORDINAL) is not None
        assert ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_CYCLICAL) is not None
        assert ProfilerFactory.get_profiler(ColumnType.DATETIME) is not None
        assert ProfilerFactory.get_profiler(ColumnType.BINARY) is not None
        assert ProfilerFactory.get_profiler(ColumnType.TEXT) is not None

    def test_get_profiler_for_unknown_type(self):
        assert ProfilerFactory.get_profiler(ColumnType.UNKNOWN) is None

    def test_numeric_types_share_profiler(self):
        profiler_cont = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        profiler_disc = ProfilerFactory.get_profiler(ColumnType.NUMERIC_DISCRETE)
        assert type(profiler_cont) == type(profiler_disc)

    def test_categorical_types_share_profiler(self):
        profiler_nom = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        profiler_ord = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_ORDINAL)
        profiler_cyc = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_CYCLICAL)
        assert type(profiler_nom) == type(profiler_ord) == type(profiler_cyc)
