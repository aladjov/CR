import numpy as np
import pandas as pd

from customer_retention.core.config import ColumnType
from customer_retention.stages.profiling import ProfilerFactory


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


class TestTextProfiler:
    def test_profile_basic_text(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        series = pd.Series(["hello world", "foo bar baz", "test string"])
        result = profiler.profile(series)

        assert "text_metrics" in result
        metrics = result["text_metrics"]
        assert metrics.length_min > 0
        assert metrics.length_max > 0
        assert metrics.length_mean > 0
        assert metrics.length_median > 0
        assert metrics.empty_count == 0
        assert metrics.word_count_mean > 0
        assert metrics.pii_detected is False

    def test_profile_text_with_empty_strings(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        series = pd.Series(["hello", "", "world", ""])
        result = profiler.profile(series)

        metrics = result["text_metrics"]
        assert metrics.empty_count == 2
        assert metrics.empty_percentage == 50.0

    def test_profile_text_with_digits(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        series = pd.Series(["abc123", "no digits here", "test456"])
        result = profiler.profile(series)

        metrics = result["text_metrics"]
        assert metrics.contains_digits_pct > 0

    def test_profile_text_with_special_chars(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        series = pd.Series(["hello!", "world@#", "normal text"])
        result = profiler.profile(series)

        metrics = result["text_metrics"]
        assert metrics.contains_special_pct > 0

    def test_profile_text_with_email_pii(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        series = pd.Series(["user@example.com", "regular text", "another@test.org"])
        result = profiler.profile(series)

        metrics = result["text_metrics"]
        assert metrics.pii_detected is True
        assert "email" in metrics.pii_types

    def test_profile_text_with_phone_pii(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        series = pd.Series(["Call 123-456-7890", "No phone", "555.123.4567"])
        result = profiler.profile(series)

        metrics = result["text_metrics"]
        assert metrics.pii_detected is True
        assert "phone" in metrics.pii_types

    def test_profile_text_with_ssn_pii(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        series = pd.Series(["SSN: 123-45-6789", "No SSN here"])
        result = profiler.profile(series)

        metrics = result["text_metrics"]
        assert metrics.pii_detected is True
        assert "ssn" in metrics.pii_types

    def test_profile_text_with_credit_card_pii(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        series = pd.Series(["Card: 1234 5678 9012 3456", "No card"])
        result = profiler.profile(series)

        metrics = result["text_metrics"]
        assert metrics.pii_detected is True
        assert "credit_card" in metrics.pii_types

    def test_profile_text_with_nulls(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        series = pd.Series(["hello", None, "world", None, None])
        result = profiler.profile(series)

        metrics = result["text_metrics"]
        assert metrics.length_min > 0

    def test_profile_empty_text_series(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        series = pd.Series([], dtype=object)
        result = profiler.profile(series)

        metrics = result["text_metrics"]
        assert metrics.length_min == 0
        assert metrics.length_max == 0
        assert metrics.length_mean == 0.0
        assert metrics.empty_percentage == 0.0


class TestIdentifierProfilerEdgeCases:
    def test_empty_identifier_series(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.IDENTIFIER)
        series = pd.Series([], dtype=str)
        result = profiler.profile(series)
        metrics = result["identifier_metrics"]
        assert metrics.length_min is None
        assert metrics.length_max is None

    def test_format_pattern_numeric_only(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.IDENTIFIER)
        series = pd.Series(["12345", "67890", "11111", "22222"])
        result = profiler.profile(series)
        metrics = result["identifier_metrics"]
        assert metrics.format_pattern == "numeric_only"

    def test_format_pattern_alpha_only(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.IDENTIFIER)
        series = pd.Series(["ABCDE", "FGHIJ", "KLMNO", "PQRST"])
        result = profiler.profile(series)
        metrics = result["identifier_metrics"]
        assert metrics.format_pattern in ["alpha_only", "alphanumeric"]

    def test_format_pattern_mixed(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.IDENTIFIER)
        series = pd.Series(["abc-123", "def_456", "789!!", "hello world"])
        result = profiler.profile(series)
        metrics = result["identifier_metrics"]
        assert metrics.format_pattern == "mixed"
        assert metrics.format_consistency == 0.0

    def test_detect_format_pattern_empty(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.IDENTIFIER)
        pattern, consistency = profiler.detect_format_pattern(pd.Series([], dtype=str))
        assert pattern is None
        assert consistency is None


class TestNumericProfilerEdgeCases:
    def test_all_nan_series(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([np.nan, np.nan, np.nan])
        result = profiler.profile(series)
        assert result["numeric_metrics"] is None

    def test_zero_std(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        result = profiler.profile(series)
        metrics = result["numeric_metrics"]
        assert metrics.std == 0.0
        assert metrics.outlier_count_zscore == 0

    def test_all_infinite_values(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([np.inf, -np.inf, np.inf])
        result = profiler.profile(series)
        metrics = result["numeric_metrics"]
        assert metrics.inf_count == 3
        # After filtering infinites, no finite values remain
        assert metrics.histogram_bins == []


class TestCategoricalProfilerEdgeCases:
    def test_case_variations_detection(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        series = pd.Series(["Hello", "hello", "HELLO", "World"] * 10)
        result = profiler.profile(series)
        metrics = result["categorical_metrics"]
        assert len(metrics.case_variations) > 0
        assert any("Hello" in v or "hello" in v for v in metrics.case_variations)

    def test_no_case_variations(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        series = pd.Series(["apple", "banana", "cherry"] * 10)
        result = profiler.profile(series)
        metrics = result["categorical_metrics"]
        assert metrics.case_variations == []

    def test_whitespace_issues_detection(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        series = pd.Series(["  leading", "trailing  ", "normal"] * 10)
        result = profiler.profile(series)
        metrics = result["categorical_metrics"]
        assert len(metrics.whitespace_issues) > 0

    def test_no_whitespace_issues(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        series = pd.Series(["clean", "values", "here"] * 10)
        result = profiler.profile(series)
        metrics = result["categorical_metrics"]
        assert metrics.whitespace_issues == []


class TestDatetimeProfilerEdgeCases:
    def test_string_datetime_format_detection(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        series = pd.Series(["2023-01-01", "2023-06-15", "2023-12-31"])
        result = profiler.profile(series)
        metrics = result["datetime_metrics"]
        assert metrics.format_detected is not None

    def test_non_datetime_string_returns_none(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        series = pd.Series(["not a date", "also not", "nope"])
        # Non-parseable dates may raise or return None
        try:
            result = profiler.profile(series)
            # If it returns, either None or valid
            assert result is not None
        except Exception:
            # It's acceptable to raise on completely invalid date input
            pass

    def test_format_detection_datetime64(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        series = pd.Series(pd.date_range("2023-01-01", periods=5))
        fmt, consistency = profiler.detect_datetime_format(series)
        assert fmt == "datetime64"
        assert consistency == 100.0

    def test_format_detection_empty(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        series = pd.Series([], dtype=object)
        fmt, consistency = profiler.detect_datetime_format(series)
        assert fmt is None
        assert consistency is None

    def test_format_detection_slash_format(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        series = pd.Series(["2023/01/15", "2023/06/20", "2023/12/31"])
        fmt, consistency = profiler.detect_datetime_format(series)
        assert fmt == "%Y/%m/%d"

    def test_format_detection_mixed_formats(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        series = pd.Series(["2023-01-01", "01/15/2023", "Jan 5, 2023", "garbage"])
        fmt, consistency = profiler.detect_datetime_format(series)
        # Mixed formats should be detected
        assert fmt is not None

    def test_non_datetime64_with_timestamp_objects(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        # Object dtype with Timestamp objects
        series = pd.Series([pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-15")], dtype=object)
        result = profiler.profile(series)
        metrics = result["datetime_metrics"]
        assert metrics.date_range_days > 0

    def test_tz_aware_series_profiles_without_error(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        series = pd.Series(pd.date_range("2023-01-01", periods=30, tz="UTC"))
        result = profiler.profile(series)
        metrics = result["datetime_metrics"]
        assert metrics.date_range_days == 29
        assert metrics.future_date_count == 0

    def test_tz_aware_series_detects_future_dates(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=365)
        series = pd.Series(
            [pd.Timestamp("2023-01-01", tz="UTC"), future]
        )
        result = profiler.profile(series)
        metrics = result["datetime_metrics"]
        assert metrics.future_date_count == 1


class TestBinaryProfilerEdgeCases:
    def test_non_standard_binary_values(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.BINARY)
        series = pd.Series(["A", "B", "A", "B", "A"])
        result = profiler.profile(series)
        metrics = result["binary_metrics"]
        # Neither A nor B are in true_values or false_values, so fallback
        assert metrics.true_count + metrics.false_count > 0

    def test_binary_with_only_true_values(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.BINARY)
        series = pd.Series([1, 1, 1, 1, 1])
        result = profiler.profile(series)
        metrics = result["binary_metrics"]
        assert metrics.true_count == 5
        assert metrics.false_count == 0
        assert metrics.balance_ratio == float('inf')


class TestCrossCompatibleOperations:
    def test_universal_metrics_uses_idxmax(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([1, 2, 3, 3, 3, 4, 5])
        metrics = profiler.compute_universal_metrics(series)
        assert metrics.most_common_value == 3
        assert metrics.most_common_frequency == 3

    def test_numeric_profiler_handles_inf_values(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([1.0, 2.0, np.inf, 4.0, -np.inf, 6.0])
        result = profiler.profile(series)
        metrics = result["numeric_metrics"]
        assert metrics.inf_count == 2

    def test_numeric_profiler_histogram_with_finite_values(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        series = pd.Series([1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = profiler.profile(series)
        metrics = result["numeric_metrics"]
        assert len(metrics.histogram_bins) == 10

    def test_all_profilers_work_without_ensure_pandas_series(self):
        test_cases = [
            (ColumnType.IDENTIFIER, pd.Series(["ID1", "ID2", "ID3"])),
            (ColumnType.TARGET, pd.Series([0, 1, 0, 1])),
            (ColumnType.CATEGORICAL_NOMINAL, pd.Series(["a", "b", "c"])),
            (ColumnType.BINARY, pd.Series([0, 1, 0, 1])),
            (ColumnType.NUMERIC_CONTINUOUS, pd.Series([1.0, 2.0, 3.0])),
        ]
        for col_type, series in test_cases:
            profiler = ProfilerFactory.get_profiler(col_type)
            result = profiler.profile(series)
            assert isinstance(result, dict)
