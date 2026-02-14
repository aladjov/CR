from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import safe_to_list
from customer_retention.core.config import ColumnType, DatasetGranularity
from customer_retention.stages.profiling import TypeConfidence, TypeDetector


class TestTypeDetector:
    def test_detect_identifier_by_name(self):
        detector = TypeDetector()
        series = pd.Series([1, 2, 3, 4, 5])
        result = detector.detect_type(series, "customer_id")

        assert result.inferred_type == ColumnType.IDENTIFIER
        assert result.confidence == TypeConfidence.HIGH
        assert any("identifier pattern" in e.lower() for e in result.evidence)

    def test_detect_identifier_by_uniqueness(self):
        detector = TypeDetector()
        series = pd.Series(["ABC123", "DEF456", "GHI789", "JKL012"])
        result = detector.detect_type(series, "random_col")

        assert result.inferred_type == ColumnType.IDENTIFIER
        assert result.confidence == TypeConfidence.HIGH
        assert any("unique" in e.lower() for e in result.evidence)

    def test_detect_target_by_name(self):
        detector = TypeDetector()
        series = pd.Series([0, 1, 0, 1, 1])
        result = detector.detect_type(series, "retained")

        assert result.inferred_type == ColumnType.TARGET
        assert result.confidence == TypeConfidence.HIGH

    @pytest.mark.parametrize("column_name,pattern_type", [
        ("churned", "primary"),
        ("is_churned", "primary"),
        ("customer_churned", "primary"),
        ("retained", "primary"),
        ("is_retained", "primary"),
        ("churn", "primary"),
        ("churn_flag", "primary"),
        ("retention", "primary"),
        ("attrition", "primary"),
    ])
    def test_detect_target_primary_patterns(self, column_name, pattern_type):
        detector = TypeDetector()
        series = pd.Series([0, 1, 0, 1, 1])
        result = detector.detect_type(series, column_name)
        assert result.inferred_type == ColumnType.TARGET
        assert "primary target pattern" in result.evidence[0].lower()

    @pytest.mark.parametrize("column_name", [
        "unsubscribed", "is_unsubscribed", "unsubscribe",
        "terminated", "is_terminated", "terminate",
        "cancelled", "is_cancelled", "cancel",
        "closed", "is_closed", "close", "account_closed",
        "discontinued", "discontinue",
        "exited", "exit", "customer_exit",
        "left", "leave", "customer_left",
    ])
    def test_detect_target_secondary_patterns(self, column_name):
        detector = TypeDetector()
        series = pd.Series([0, 1, 0, 1, 1])
        result = detector.detect_type(series, column_name)
        assert result.inferred_type == ColumnType.TARGET
        assert "secondary target pattern" in result.evidence[0].lower()

    @pytest.mark.parametrize("column_name", [
        "target", "is_target", "target_flag",
        "label", "class_label",
        "outcome", "outcome_flag",
        "class", "target_class",
        "flag",
    ])
    def test_detect_target_generic_patterns(self, column_name):
        detector = TypeDetector()
        series = pd.Series([0, 1, 0, 1, 1])
        result = detector.detect_type(series, column_name)
        assert result.inferred_type == ColumnType.TARGET
        assert "generic target pattern" in result.evidence[0].lower()

    def test_target_pattern_priority_primary_over_secondary(self):
        detector = TypeDetector()
        series = pd.Series([0, 1, 0, 1])
        result = detector.detect_type(series, "churned_and_unsubscribed")
        assert result.inferred_type == ColumnType.TARGET
        assert "primary" in result.evidence[0].lower()

    def test_target_pattern_priority_secondary_over_generic(self):
        detector = TypeDetector()
        series = pd.Series([0, 1, 0, 1])
        result = detector.detect_type(series, "unsubscribed_target")
        assert result.inferred_type == ColumnType.TARGET
        assert "secondary" in result.evidence[0].lower()

    def test_detect_binary_zero_one(self):
        detector = TypeDetector()
        series = pd.Series([0, 1, 0, 1, 1, 0])
        result = detector.detect_type(series, "is_active")

        assert result.inferred_type == ColumnType.BINARY
        assert result.confidence == TypeConfidence.HIGH

    def test_detect_binary_true_false(self):
        detector = TypeDetector()
        series = pd.Series([True, False, True, False])
        result = detector.detect_type(series, "active")

        assert result.inferred_type == ColumnType.BINARY
        assert result.confidence == TypeConfidence.HIGH

    def test_detect_binary_yes_no(self):
        detector = TypeDetector()
        series = pd.Series(["yes", "no", "yes", "no"])
        result = detector.detect_type(series, "subscribe")

        assert result.inferred_type == ColumnType.BINARY
        assert result.confidence == TypeConfidence.HIGH

    def test_detect_datetime_dtype(self):
        detector = TypeDetector()
        series = pd.Series(pd.date_range("2023-01-01", periods=5))
        result = detector.detect_type(series, "created_date")

        assert result.inferred_type == ColumnType.DATETIME
        assert result.confidence == TypeConfidence.HIGH

    def test_detect_datetime_string_parseable(self):
        detector = TypeDetector()
        series = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
        result = detector.detect_type(series, "date_col")

        assert result.inferred_type == ColumnType.DATETIME
        assert result.confidence == TypeConfidence.HIGH

    def test_detect_numeric_discrete(self):
        detector = TypeDetector()
        series = pd.Series([1, 2, 3, 4, 5, 1, 2, 3])
        result = detector.detect_type(series, "count_col")

        assert result.inferred_type == ColumnType.NUMERIC_DISCRETE
        assert result.confidence == TypeConfidence.MEDIUM
        assert ColumnType.NUMERIC_CONTINUOUS in result.alternatives

    def test_detect_numeric_continuous(self):
        detector = TypeDetector()
        series = pd.Series(np.random.rand(100))
        result = detector.detect_type(series, "amount")

        assert result.inferred_type == ColumnType.NUMERIC_CONTINUOUS
        assert result.confidence == TypeConfidence.HIGH

    def test_detect_categorical_nominal_low_cardinality(self):
        detector = TypeDetector()
        series = pd.Series(["red", "blue", "green", "red", "blue"])
        result = detector.detect_type(series, "color")

        assert result.inferred_type == ColumnType.CATEGORICAL_NOMINAL
        assert result.confidence == TypeConfidence.HIGH

    def test_detect_categorical_cyclical_days(self):
        detector = TypeDetector()
        series = pd.Series(["Monday", "Tuesday", "Wednesday", "Monday"])
        result = detector.detect_type(series, "day_col")

        assert result.inferred_type == ColumnType.CATEGORICAL_CYCLICAL
        assert result.confidence == TypeConfidence.MEDIUM

    def test_detect_categorical_cyclical_months(self):
        detector = TypeDetector()
        series = pd.Series(["January", "February", "March", "January"])
        result = detector.detect_type(series, "month_col")

        assert result.inferred_type == ColumnType.CATEGORICAL_CYCLICAL
        assert result.confidence == TypeConfidence.MEDIUM

    def test_address_column_not_detected_as_cyclical(self):
        detector = TypeDetector()
        base_addresses = [
            "8366 Market St, Suite 422, Phoenix, FL 42449",
            "9978 Market St, Suite 461, Miami, NJ 14259",
            "9080 Market St, San Jose, AZ 41704",
            "6417 Main St, Dallas, NJ 24236",
            "3832 Broadway, Chicago, WA 89782",
            "9748 Industrial Rd, Suite 996, Phoenix, AZ 82129",
            "6770 Main St, Suite 538, San Jose, AZ 30945",
            "8520 Main St, Dallas, CA 23279",
            "8390 Broadway, San Jose, TX 62069",
            "1200 Market St, March Field, FL 33101",
        ]
        series = pd.Series([f"{a[:-3]}{i:03d}" for i, a in enumerate(base_addresses * 20)])
        result = detector.detect_type(series, "address")

        assert result.inferred_type != ColumnType.CATEGORICAL_CYCLICAL

    def test_short_month_abbreviations_still_detected(self):
        detector = TypeDetector()
        series = pd.Series(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jan", "Feb"])
        result = detector.detect_type(series, "month")

        assert result.inferred_type == ColumnType.CATEGORICAL_CYCLICAL

    def test_short_day_abbreviations_still_detected(self):
        detector = TypeDetector()
        series = pd.Series(["Mon", "Tue", "Wed", "Thu", "Fri", "Mon", "Tue"])
        result = detector.detect_type(series, "day")

        assert result.inferred_type == ColumnType.CATEGORICAL_CYCLICAL

    def test_detect_text_high_cardinality(self):
        detector = TypeDetector()
        series = pd.Series([f"text_{i}" for i in range(150)])
        result = detector.detect_type(series, "comment")

        assert result.inferred_type == ColumnType.TEXT
        assert result.confidence == TypeConfidence.MEDIUM

    def test_detect_unknown_empty_series(self):
        detector = TypeDetector()
        series = pd.Series([], dtype=object)
        result = detector.detect_type(series, "empty_col")

        assert result.inferred_type == ColumnType.UNKNOWN

    def test_detect_identifier_priority_over_target(self):
        detector = TypeDetector()
        series = pd.Series([1, 2, 3, 4, 5])
        result = detector.detect_type(series, "customer_id")

        assert result.inferred_type == ColumnType.IDENTIFIER

    def test_detect_target_priority_over_binary(self):
        detector = TypeDetector()
        series = pd.Series([0, 1, 0, 1])
        result = detector.detect_type(series, "churn_label")

        assert result.inferred_type == ColumnType.TARGET

    def test_evidence_populated(self):
        detector = TypeDetector()
        series = pd.Series([0, 1, 0, 1])
        result = detector.detect_type(series, "flag")

        assert len(result.evidence) > 0
        assert all(isinstance(e, str) for e in result.evidence)


class TestDatasetGranularityDetection:
    """Tests for dataset granularity detection."""

    def test_detect_entity_level_unique_ids(self):
        """Entity-level: each ID appears once."""
        from customer_retention.core.config import DatasetGranularity

        detector = TypeDetector()
        df = pd.DataFrame({
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "age": [25, 30, 35, 40, 45],
        })

        result = detector.detect_granularity(df)
        assert result.granularity == DatasetGranularity.ENTITY_LEVEL

    def test_detect_event_level_repeated_ids(self):
        """Event-level: IDs repeat with temporal ordering."""
        from customer_retention.core.config import DatasetGranularity

        detector = TypeDetector()
        df = pd.DataFrame({
            "transaction_id": ["T001", "T002", "T003", "T004", "T005"],
            "customer_id": ["C001", "C001", "C002", "C001", "C002"],
            "transaction_date": pd.date_range("2023-01-01", periods=5),
            "amount": [100, 200, 150, 300, 50],
        })

        result = detector.detect_granularity(df)
        assert result.granularity == DatasetGranularity.EVENT_LEVEL

    def test_detect_event_level_with_datetime_column(self):
        """Event-level detection requires datetime column."""
        from customer_retention.core.config import DatasetGranularity

        detector = TypeDetector()
        df = pd.DataFrame({
            "email_id": [f"E{i:03d}" for i in range(100)],
            "customer_id": [f"C{i % 20:03d}" for i in range(100)],  # 20 customers, 5 emails each
            "sent_date": pd.date_range("2023-01-01", periods=100),
            "opened": [1, 0] * 50,
        })

        result = detector.detect_granularity(df)
        assert result.granularity == DatasetGranularity.EVENT_LEVEL
        assert result.entity_column == "customer_id"
        assert result.time_column == "sent_date"

    def test_detect_entity_column(self):
        """Should identify the entity column (ID with repetitions)."""

        detector = TypeDetector()
        df = pd.DataFrame({
            "order_id": [f"O{i:03d}" for i in range(50)],
            "cust_id": [f"C{i % 10:03d}" for i in range(50)],
            "order_date": pd.date_range("2023-01-01", periods=50),
        })

        result = detector.detect_granularity(df)
        assert result.entity_column == "cust_id"

    def test_entity_column_low_ratio_high_event_dataset(self):
        import numpy as np

        detector = TypeDetector()
        n = 5000
        df = pd.DataFrame({
            "transaction_id": [f"T{i:06d}" for i in range(n)],
            "customer_id": [f"C{i % 40:03d}" for i in range(n)],
            "event_timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
            "amount": np.random.rand(n),
        })
        assert df["customer_id"].nunique() / len(df) < 0.01

        result = detector.detect_granularity(df)
        assert result.entity_column == "customer_id"
        assert result.granularity == DatasetGranularity.EVENT_LEVEL

    def test_entity_name_preferred_over_generic_id(self):

        detector = TypeDetector()
        df = pd.DataFrame({
            "record_id": [f"R{i:03d}" for i in range(200)],
            "user_id": [f"U{i % 20:03d}" for i in range(200)],
            "error_code": [f"E{i % 3:01d}" for i in range(200)],
            "ts": pd.date_range("2023-01-01", periods=200, freq="h"),
        })

        result = detector.detect_granularity(df)
        assert result.entity_column == "user_id"

    def test_event_level_with_many_events_per_entity(self):
        import numpy as np

        detector = TypeDetector()
        n = 10000
        df = pd.DataFrame({
            "txn_id": [f"T{i:06d}" for i in range(n)],
            "account_id": [f"A{i % 50:03d}" for i in range(n)],
            "event_timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
            "value": np.random.rand(n),
        })

        result = detector.detect_granularity(df)
        assert result.granularity == DatasetGranularity.EVENT_LEVEL
        assert result.entity_column == "account_id"
        assert result.avg_events_per_entity == pytest.approx(200.0, rel=0.1)

    def test_detect_time_column(self):
        """Should identify the time column."""

        detector = TypeDetector()
        df = pd.DataFrame({
            "id": range(20),
            "user_id": [f"U{i % 5:03d}" for i in range(20)],
            "event_timestamp": pd.date_range("2023-01-01", periods=20),
            "value": range(20),
        })

        result = detector.detect_granularity(df)
        assert result.time_column == "event_timestamp"

    def test_unknown_granularity_no_ids(self):
        """Unknown when no clear ID pattern exists (all numeric, no ID-like columns)."""
        from customer_retention.core.config import DatasetGranularity

        detector = TypeDetector()
        # All numeric columns with no ID-like names
        df = pd.DataFrame({
            "value1": range(10),
            "value2": range(10, 20),
            "value3": [1.5, 2.5] * 5,
        })

        result = detector.detect_granularity(df)
        assert result.granularity == DatasetGranularity.UNKNOWN

    def test_granularity_result_has_stats(self):
        """Result should include useful statistics."""

        detector = TypeDetector()
        df = pd.DataFrame({
            "txn_id": range(100),
            "customer_id": [f"C{i % 25:03d}" for i in range(100)],
            "date": pd.date_range("2023-01-01", periods=100),
        })

        result = detector.detect_granularity(df)
        assert result.avg_events_per_entity is not None
        assert result.avg_events_per_entity == pytest.approx(4.0, rel=0.1)

    def test_detect_granularity_empty_dataframe(self):
        """Handle empty DataFrame gracefully."""
        from customer_retention.core.config import DatasetGranularity

        detector = TypeDetector()
        df = pd.DataFrame()

        result = detector.detect_granularity(df)
        assert result.granularity == DatasetGranularity.UNKNOWN


class TestTypeDetectorSparkCompat:
    """Regression: pyspark.pandas Series.__iter__() raises PandasNotImplementedError.
    Verify safe_to_list is used instead of direct iteration in all code paths."""

    PATCH_TARGET = "customer_retention.stages.profiling.type_detector.safe_to_list"

    def test_is_binary_uses_safe_to_list(self):
        detector = TypeDetector()
        series = pd.Series([0, 1, 0, 1])
        with patch(self.PATCH_TARGET, wraps=safe_to_list) as mock:
            detector.is_binary(series, series.nunique())
            mock.assert_called_once()

    def test_is_datetime_string_uses_safe_to_list(self):
        detector = TypeDetector()
        with patch(self.PATCH_TARGET, wraps=safe_to_list) as mock:
            detector.is_datetime(pd.Series(["2023-01-01", "2023-01-02"]))
            mock.assert_called_once()

    def test_is_cyclical_pattern_uses_safe_to_list(self):
        detector = TypeDetector()
        with patch(self.PATCH_TARGET, wraps=safe_to_list) as mock:
            detector.is_cyclical_pattern(pd.Series(["Monday", "Tuesday", "Wednesday"]))
            mock.assert_called_once()

    def test_is_identifier_datetime_check_uses_safe_to_list(self):
        detector = TypeDetector()
        series = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
        with patch(self.PATCH_TARGET, wraps=safe_to_list) as mock:
            detector.is_identifier(series, "random_col", series.nunique())
            assert mock.called

    def test_detect_time_column_uses_safe_to_list(self):
        detector = TypeDetector()
        df = pd.DataFrame({
            "customer_id": ["C001", "C001", "C002"],
            "event_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        })
        with patch(self.PATCH_TARGET, wraps=safe_to_list) as mock:
            detector._detect_time_column(df)
            assert mock.called
