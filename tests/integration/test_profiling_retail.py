from pathlib import Path

import pandas as pd
import pytest

from customer_retention.core.config import ColumnType
from customer_retention.stages.profiling import ProfilerFactory, TypeDetector


@pytest.fixture
def retail_data():
    retail_path = Path(__file__).parent.parent / "fixtures" / "customer_retention_retail.csv"
    return pd.read_csv(retail_path)


class TestTypeDetectionRetailDataset:
    def test_detect_custid_as_identifier(self, retail_data):
        detector = TypeDetector()
        result = detector.detect_type(retail_data["custid"], "custid")

        assert result.inferred_type == ColumnType.IDENTIFIER
        assert result.confidence.value == "high"

    def test_detect_retained_as_target(self, retail_data):
        detector = TypeDetector()
        result = detector.detect_type(retail_data["retained"], "retained")

        assert result.inferred_type == ColumnType.TARGET
        assert result.confidence.value == "high"

    def test_detect_avgorder_as_numeric(self, retail_data):
        detector = TypeDetector()
        result = detector.detect_type(retail_data["avgorder"], "avgorder")

        assert result.inferred_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]

    def test_detect_created_as_datetime(self, retail_data):
        detector = TypeDetector()
        result = detector.detect_type(retail_data["created"], "created")

        assert result.inferred_type == ColumnType.DATETIME

    def test_detect_paperless_as_binary(self, retail_data):
        detector = TypeDetector()
        result = detector.detect_type(retail_data["paperless"], "paperless")

        assert result.inferred_type == ColumnType.BINARY

    def test_detect_city_as_categorical(self, retail_data):
        detector = TypeDetector()
        result = detector.detect_type(retail_data["city"], "city")

        assert result.inferred_type in [ColumnType.CATEGORICAL_NOMINAL, ColumnType.CATEGORICAL_ORDINAL]


class TestProfilingRetailDataset:
    def test_profile_identifier_column(self, retail_data):
        profiler = ProfilerFactory.get_profiler(ColumnType.IDENTIFIER)
        universal = profiler.compute_universal_metrics(retail_data["custid"])
        specific = profiler.profile(retail_data["custid"])

        assert universal.total_count == 30801
        # Note: retail data has 20 null custid values and 31 duplicates
        assert universal.null_count == 20
        assert universal.distinct_count == 30769
        # Not unique due to duplicates in data
        assert specific["identifier_metrics"].is_unique is False

    def test_profile_target_column(self, retail_data):
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        universal = profiler.compute_universal_metrics(retail_data["retained"])
        specific = profiler.profile(retail_data["retained"])

        metrics = specific["target_metrics"]
        assert metrics.n_classes == 2
        assert "0" in metrics.class_distribution
        assert "1" in metrics.class_distribution
        assert metrics.minority_percentage > 0
        assert metrics.imbalance_ratio > 0

    def test_profile_numeric_column(self, retail_data):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        universal = profiler.compute_universal_metrics(retail_data["avgorder"])
        specific = profiler.profile(retail_data["avgorder"])

        metrics = specific["numeric_metrics"]
        assert metrics.mean > 0
        assert metrics.std > 0
        assert metrics.min_value >= 0
        assert metrics.max_value > metrics.min_value
        assert metrics.median > 0
        assert metrics.histogram_bins is not None
        assert len(metrics.histogram_bins) == 10

    def test_profile_datetime_column(self, retail_data):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        universal = profiler.compute_universal_metrics(retail_data["created"])
        specific = profiler.profile(retail_data["created"])

        metrics = specific["datetime_metrics"]
        assert metrics.date_range_days > 0
        assert metrics.min_date is not None
        assert metrics.max_date is not None
        assert metrics.future_date_count >= 0
        assert metrics.weekend_percentage >= 0

    def test_profile_binary_column(self, retail_data):
        profiler = ProfilerFactory.get_profiler(ColumnType.BINARY)
        universal = profiler.compute_universal_metrics(retail_data["paperless"])
        specific = profiler.profile(retail_data["paperless"])

        metrics = specific["binary_metrics"]
        assert metrics.true_count + metrics.false_count > 0
        assert metrics.true_percentage >= 0
        assert metrics.true_percentage <= 100
        assert metrics.balance_ratio > 0

    def test_profile_categorical_column(self, retail_data):
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        universal = profiler.compute_universal_metrics(retail_data["city"])
        specific = profiler.profile(retail_data["city"])

        metrics = specific["categorical_metrics"]
        assert metrics.cardinality > 0
        assert metrics.cardinality_ratio > 0
        assert len(metrics.value_counts) > 0
        assert len(metrics.top_categories) > 0
        assert metrics.encoding_recommendation in ["one_hot", "one_hot_or_target", "target_or_embedding", "hashing_or_embedding"]


class TestEndToEndProfilingWorkflow:
    def test_profile_all_columns_in_retail_dataset(self, retail_data):
        column_mappings = {
            "custid": ColumnType.IDENTIFIER,
            "created": ColumnType.DATETIME,
            "firstorder": ColumnType.DATETIME,
            "lastorder": ColumnType.DATETIME,
            "esent": ColumnType.NUMERIC_CONTINUOUS,
            "eopenrate": ColumnType.NUMERIC_CONTINUOUS,
            "eclickrate": ColumnType.NUMERIC_CONTINUOUS,
            "avgorder": ColumnType.NUMERIC_CONTINUOUS,
            "ordfreq": ColumnType.NUMERIC_CONTINUOUS,
            "paperless": ColumnType.BINARY,
            "refill": ColumnType.BINARY,
            "doorstep": ColumnType.BINARY,
            "favday": ColumnType.CATEGORICAL_NOMINAL,  # Day names as strings
            "city": ColumnType.CATEGORICAL_NOMINAL,
            "retained": ColumnType.TARGET
        }

        profiles = {}
        for col_name, col_type in column_mappings.items():
            profiler = ProfilerFactory.get_profiler(col_type)
            if profiler:
                universal = profiler.compute_universal_metrics(retail_data[col_name])
                specific = profiler.profile(retail_data[col_name])
                profiles[col_name] = {
                    "universal": universal,
                    "specific": specific
                }

        assert len(profiles) == 15

        for col_name, profile in profiles.items():
            assert profile["universal"].total_count == 30801
            assert profile["specific"] is not None

    def test_detect_and_profile_workflow(self, retail_data):
        detector = TypeDetector()
        column_name = "avgorder"

        type_result = detector.detect_type(retail_data[column_name], column_name)

        profiler = ProfilerFactory.get_profiler(type_result.inferred_type)
        assert profiler is not None

        universal = profiler.compute_universal_metrics(retail_data[column_name])
        specific = profiler.profile(retail_data[column_name])

        assert universal.total_count == 30801
        assert specific["numeric_metrics"] is not None
        assert specific["numeric_metrics"].mean > 0

    def test_profiling_with_nulls(self, retail_data):
        series_with_nulls = retail_data["avgorder"].copy()
        series_with_nulls.iloc[:100] = None

        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        universal = profiler.compute_universal_metrics(series_with_nulls)
        specific = profiler.profile(series_with_nulls)

        assert universal.null_count == 100
        assert universal.null_percentage > 0
        assert specific["numeric_metrics"] is not None

    def test_all_column_types_have_profilers_except_unknown(self, retail_data):
        for column_type in ColumnType:
            profiler = ProfilerFactory.get_profiler(column_type)

            if column_type == ColumnType.UNKNOWN:
                assert profiler is None
            else:
                assert profiler is not None

    def test_type_detection_confidence_levels(self, retail_data):
        detector = TypeDetector()

        high_confidence_columns = ["custid", "retained", "paperless"]
        for col in high_confidence_columns:
            result = detector.detect_type(retail_data[col], col)
            assert result.confidence.value == "high"

    def test_profile_quality_metrics(self, retail_data):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        universal = profiler.compute_universal_metrics(retail_data["avgorder"])

        assert universal.distinct_percentage > 0
        assert universal.memory_size_bytes > 0
        assert universal.most_common_value is not None
        assert universal.most_common_frequency is not None

    def test_numeric_outlier_detection(self, retail_data):
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        specific = profiler.profile(retail_data["avgorder"])

        metrics = specific["numeric_metrics"]
        assert metrics.outlier_count_iqr >= 0
        assert metrics.outlier_count_zscore >= 0
        assert metrics.outlier_percentage >= 0

    def test_categorical_encoding_recommendations(self, retail_data):
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)

        result_city = profiler.profile(retail_data["city"])

        assert result_city["categorical_metrics"].encoding_recommendation is not None

    def test_datetime_range_and_weekends(self, retail_data):
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        specific = profiler.profile(retail_data["created"])

        metrics = specific["datetime_metrics"]
        assert metrics.date_range_days > 0
        assert 0 <= metrics.weekend_percentage <= 100

    def test_binary_balance_ratio(self, retail_data):
        profiler = ProfilerFactory.get_profiler(ColumnType.BINARY)

        result_paperless = profiler.profile(retail_data["paperless"])
        result_refill = profiler.profile(retail_data["refill"])

        assert result_paperless["binary_metrics"].balance_ratio > 0
        assert result_refill["binary_metrics"].balance_ratio > 0

    def test_target_class_imbalance(self, retail_data):
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        specific = profiler.profile(retail_data["retained"])

        metrics = specific["target_metrics"]
        assert metrics.imbalance_ratio >= 1.0
        assert metrics.minority_percentage <= 50.0
