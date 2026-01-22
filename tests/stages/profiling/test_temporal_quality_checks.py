"""Tests for temporal quality checks - TDD approach."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from customer_retention.stages.profiling.temporal_quality_checks import (
    TemporalQualityCheck,
    DuplicateEventCheck,
    TemporalGapCheck,
    FutureDateCheck,
    EventOrderCheck,
    TemporalQualityResult,
)
from customer_retention.core.components.enums import Severity


@pytest.fixture
def clean_transactions():
    """Create clean transaction data with no issues."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "transaction_id": range(100),
        "customer_id": [f"C{i % 20:03d}" for i in range(100)],
        "transaction_date": dates,
        "amount": np.random.uniform(10, 500, 100),
    })


@pytest.fixture
def transactions_with_duplicates():
    """Create transactions with duplicate events."""
    df = pd.DataFrame({
        "transaction_id": [1, 2, 3, 4, 5],
        "customer_id": ["C001", "C001", "C001", "C002", "C002"],
        "transaction_date": pd.to_datetime([
            "2023-01-01", "2023-01-01",  # Duplicate for C001
            "2023-01-02",
            "2023-01-01", "2023-01-01",  # Duplicate for C002
        ]),
        "amount": [100, 100, 200, 150, 150],
    })
    return df


class TestDuplicateEventCheck:
    """Tests for detecting duplicate events."""

    def test_no_duplicates_passes(self, clean_transactions):
        check = DuplicateEventCheck(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = check.run(clean_transactions)

        assert result.passed
        assert result.duplicate_count == 0

    def test_detects_duplicates(self, transactions_with_duplicates):
        check = DuplicateEventCheck(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = check.run(transactions_with_duplicates)

        assert not result.passed
        assert result.duplicate_count == 2  # C001 has 1 dup, C002 has 1 dup
        assert result.severity == Severity.MEDIUM

    def test_result_contains_duplicate_examples(self, transactions_with_duplicates):
        check = DuplicateEventCheck(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = check.run(transactions_with_duplicates)

        assert "duplicate_examples" in result.details
        assert len(result.details["duplicate_examples"]) > 0


class TestTemporalGapCheck:
    """Tests for detecting temporal gaps."""

    def test_regular_data_passes(self):
        """Daily data with no gaps should pass."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        df = pd.DataFrame({
            "id": range(30),
            "date": dates,
        })

        check = TemporalGapCheck(
            time_column="date",
            expected_frequency="D",
            max_gap_multiple=2
        )
        result = check.run(df)

        assert result.passed
        assert result.gap_count == 0

    def test_detects_gaps(self):
        """Should detect missing days in daily data."""
        dates = pd.to_datetime([
            "2023-01-01", "2023-01-02", "2023-01-03",
            # Gap: 4, 5, 6, 7 missing
            "2023-01-08", "2023-01-09", "2023-01-10",
        ])
        df = pd.DataFrame({"id": range(6), "date": dates})

        check = TemporalGapCheck(
            time_column="date",
            expected_frequency="D",
            max_gap_multiple=2
        )
        result = check.run(df)

        assert not result.passed
        assert result.gap_count >= 1
        assert result.max_gap_days >= 4

    def test_weekly_frequency(self):
        """Should work with weekly frequency."""
        dates = pd.date_range("2023-01-01", periods=10, freq="W")
        df = pd.DataFrame({"id": range(10), "date": dates})

        check = TemporalGapCheck(
            time_column="date",
            expected_frequency="W",
            max_gap_multiple=2
        )
        result = check.run(df)

        assert result.passed


class TestFutureDateCheck:
    """Tests for detecting future dates."""

    def test_all_past_dates_passes(self):
        """Data with all past dates should pass."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({"id": range(10), "date": dates})

        check = FutureDateCheck(time_column="date")
        result = check.run(df)

        assert result.passed
        assert result.future_count == 0

    def test_detects_future_dates(self):
        """Should detect dates in the future."""
        future_date = datetime.now() + timedelta(days=30)
        dates = pd.to_datetime([
            "2023-01-01", "2023-01-02",
            future_date,  # Future
        ])
        df = pd.DataFrame({"id": range(3), "date": dates})

        check = FutureDateCheck(time_column="date")
        result = check.run(df)

        assert not result.passed
        assert result.future_count == 1
        assert result.severity == Severity.HIGH

    def test_with_reference_date(self):
        """Should use reference date if provided."""
        dates = pd.to_datetime(["2023-01-01", "2023-06-01", "2023-12-31"])
        df = pd.DataFrame({"id": range(3), "date": dates})

        check = FutureDateCheck(
            time_column="date",
            reference_date=pd.Timestamp("2023-03-01")
        )
        result = check.run(df)

        assert not result.passed
        assert result.future_count == 2  # Jun and Dec are after Mar


class TestEventOrderCheck:
    """Tests for detecting out-of-order events."""

    def test_ordered_data_passes(self):
        """Properly ordered data should pass."""
        df = pd.DataFrame({
            "entity": ["A", "A", "A", "B", "B"],
            "date": pd.to_datetime([
                "2023-01-01", "2023-01-15", "2023-01-30",
                "2023-02-01", "2023-02-15",
            ]),
        })

        check = EventOrderCheck(
            entity_column="entity",
            time_column="date"
        )
        result = check.run(df)

        # Note: check verifies the data is sortable, not that it's pre-sorted
        assert result.passed

    def test_detects_ordering_issues(self):
        """Should detect when events have timestamp collisions causing ambiguity."""
        # This tests for potential ordering issues - exact same timestamps
        df = pd.DataFrame({
            "entity": ["A", "A", "A"],
            "date": pd.to_datetime([
                "2023-01-01 10:00:00",
                "2023-01-01 10:00:00",  # Same timestamp
                "2023-01-01 10:00:00",
            ]),
            "sequence": [1, 2, 3],
        })

        check = EventOrderCheck(
            entity_column="entity",
            time_column="date"
        )
        result = check.run(df)

        # Multiple events at same timestamp is an ordering concern
        assert result.ambiguous_count >= 2


class TestTemporalQualityResult:
    """Tests for the result dataclass."""

    def test_result_fields(self):
        result = TemporalQualityResult(
            check_id="TQ001",
            check_name="Test Check",
            passed=True,
            severity=Severity.LOW,
            message="All good",
            details={"key": "value"},
        )

        assert result.check_id == "TQ001"
        assert result.passed
        assert result.severity == Severity.LOW


class TestTemporalQualityCheckBase:
    """Tests for the base class."""

    def test_base_class_attributes(self):
        check = TemporalQualityCheck(
            check_id="TQ999",
            check_name="Test",
            severity=Severity.MEDIUM
        )

        assert check.check_id == "TQ999"
        assert check.check_name == "Test"
        assert check.severity == Severity.MEDIUM
