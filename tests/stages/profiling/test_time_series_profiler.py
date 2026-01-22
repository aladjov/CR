"""Tests for TimeSeriesProfiler - TDD approach."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.time_series_profiler import (
    DistributionStats,
    TimeSeriesProfile,
    TimeSeriesProfiler,
)


@pytest.fixture
def sample_transactions():
    """Create sample transaction data."""
    np.random.seed(42)
    n_customers = 50
    dates = []
    customers = []
    amounts = []

    base_date = datetime(2023, 1, 1)
    for cust_id in range(n_customers):
        n_txns = np.random.randint(1, 20)
        for _ in range(n_txns):
            days_offset = np.random.randint(0, 365)
            dates.append(base_date + timedelta(days=days_offset))
            customers.append(f"C{cust_id:03d}")
            amounts.append(np.random.uniform(10, 500))

    return pd.DataFrame({
        "transaction_id": range(len(dates)),
        "customer_id": customers,
        "transaction_date": dates,
        "amount": amounts,
    })


@pytest.fixture
def sample_emails():
    """Create sample email event data."""
    np.random.seed(42)
    data = []
    base_date = datetime(2023, 1, 1)

    for cust_id in range(30):
        n_emails = np.random.randint(5, 50)
        for i in range(n_emails):
            days_offset = np.random.randint(0, 180)
            data.append({
                "email_id": f"E{len(data):05d}",
                "customer_id": f"C{cust_id:03d}",
                "sent_date": base_date + timedelta(days=days_offset),
                "opened": np.random.choice([0, 1], p=[0.7, 0.3]),
            })

    return pd.DataFrame(data)


class TestTimeSeriesProfiler:
    """Tests for the main profiler class."""

    def test_profile_returns_time_series_profile(self, sample_transactions):
        profiler = TimeSeriesProfiler(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = profiler.profile(sample_transactions)

        assert isinstance(result, TimeSeriesProfile)

    def test_profile_basic_counts(self, sample_transactions):
        profiler = TimeSeriesProfiler(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = profiler.profile(sample_transactions)

        assert result.total_events == len(sample_transactions)
        assert result.unique_entities == sample_transactions["customer_id"].nunique()

    def test_profile_entity_column_stored(self, sample_transactions):
        profiler = TimeSeriesProfiler(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = profiler.profile(sample_transactions)

        assert result.entity_column == "customer_id"
        assert result.time_column == "transaction_date"

    def test_profile_time_span(self, sample_transactions):
        profiler = TimeSeriesProfiler(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = profiler.profile(sample_transactions)

        assert result.time_span_days > 0
        assert result.time_span_days <= 365

    def test_profile_events_per_entity_stats(self, sample_transactions):
        profiler = TimeSeriesProfiler(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = profiler.profile(sample_transactions)

        stats = result.events_per_entity
        assert isinstance(stats, DistributionStats)
        assert stats.min >= 1
        assert stats.max <= 20
        assert stats.mean > 0
        assert stats.median > 0

    def test_profile_with_string_dates(self):
        """Should handle string dates by parsing them."""
        df = pd.DataFrame({
            "user_id": ["U1", "U1", "U2", "U2", "U2"],
            "event_date": ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15", "2023-03-01"],
            "value": [100, 200, 150, 250, 300],
        })

        profiler = TimeSeriesProfiler(
            entity_column="user_id",
            time_column="event_date"
        )
        result = profiler.profile(df)

        assert result.total_events == 5
        assert result.unique_entities == 2


class TestEntityLifecycles:
    """Tests for entity lifecycle analysis."""

    def test_lifecycles_dataframe_structure(self, sample_transactions):
        profiler = TimeSeriesProfiler(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = profiler.profile(sample_transactions)

        lifecycles = result.entity_lifecycles
        assert isinstance(lifecycles, pd.DataFrame)
        assert "entity" in lifecycles.columns
        assert "first_event" in lifecycles.columns
        assert "last_event" in lifecycles.columns
        assert "duration_days" in lifecycles.columns
        assert "event_count" in lifecycles.columns

    def test_lifecycles_count_matches_entities(self, sample_transactions):
        profiler = TimeSeriesProfiler(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = profiler.profile(sample_transactions)

        assert len(result.entity_lifecycles) == result.unique_entities

    def test_lifecycles_event_counts_sum_to_total(self, sample_transactions):
        profiler = TimeSeriesProfiler(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = profiler.profile(sample_transactions)

        total_from_lifecycles = result.entity_lifecycles["event_count"].sum()
        assert total_from_lifecycles == result.total_events

    def test_lifecycle_duration_calculation(self):
        """Duration should be days between first and last event."""
        df = pd.DataFrame({
            "cust": ["A", "A", "A"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-15", "2023-01-31"]),
            "val": [1, 2, 3],
        })

        profiler = TimeSeriesProfiler(entity_column="cust", time_column="date")
        result = profiler.profile(df)

        lifecycle = result.entity_lifecycles.iloc[0]
        assert lifecycle["duration_days"] == 30  # Jan 1 to Jan 31

    def test_single_event_entity_has_zero_duration(self):
        """Entity with one event should have duration 0."""
        df = pd.DataFrame({
            "cust": ["A", "B", "B"],
            "date": pd.to_datetime(["2023-01-15", "2023-01-01", "2023-01-31"]),
            "val": [1, 2, 3],
        })

        profiler = TimeSeriesProfiler(entity_column="cust", time_column="date")
        result = profiler.profile(df)

        a_lifecycle = result.entity_lifecycles[
            result.entity_lifecycles["entity"] == "A"
        ].iloc[0]
        assert a_lifecycle["duration_days"] == 0
        assert a_lifecycle["event_count"] == 1


class TestInterEventTiming:
    """Tests for inter-event timing statistics."""

    def test_avg_inter_event_time(self, sample_transactions):
        profiler = TimeSeriesProfiler(
            entity_column="customer_id",
            time_column="transaction_date"
        )
        result = profiler.profile(sample_transactions)

        assert result.avg_inter_event_days is not None
        assert result.avg_inter_event_days >= 0

    def test_inter_event_time_for_regular_events(self):
        """Regular weekly events should have ~7 day inter-event time."""
        dates = pd.date_range("2023-01-01", periods=10, freq="7D")
        df = pd.DataFrame({
            "entity": ["A"] * 10,
            "date": dates,
            "val": range(10),
        })

        profiler = TimeSeriesProfiler(entity_column="entity", time_column="date")
        result = profiler.profile(df)

        assert result.avg_inter_event_days == pytest.approx(7.0, rel=0.1)


class TestDistributionStats:
    """Tests for DistributionStats dataclass."""

    def test_distribution_stats_fields(self):
        stats = DistributionStats(
            min=1.0,
            max=100.0,
            mean=50.0,
            median=45.0,
            std=20.0,
            q25=30.0,
            q75=70.0,
        )

        assert stats.min == 1.0
        assert stats.max == 100.0
        assert stats.mean == 50.0
        assert stats.median == 45.0
        assert stats.std == 20.0
        assert stats.q25 == 30.0
        assert stats.q75 == 70.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["entity", "date", "value"])

        profiler = TimeSeriesProfiler(entity_column="entity", time_column="date")
        result = profiler.profile(df)

        assert result.total_events == 0
        assert result.unique_entities == 0

    def test_single_row_dataframe(self):
        df = pd.DataFrame({
            "entity": ["A"],
            "date": [pd.Timestamp("2023-01-01")],
            "value": [100],
        })

        profiler = TimeSeriesProfiler(entity_column="entity", time_column="date")
        result = profiler.profile(df)

        assert result.total_events == 1
        assert result.unique_entities == 1
        assert result.events_per_entity.mean == 1.0

    def test_missing_entity_column_raises(self, sample_transactions):
        profiler = TimeSeriesProfiler(
            entity_column="nonexistent",
            time_column="transaction_date"
        )

        with pytest.raises(KeyError):
            profiler.profile(sample_transactions)

    def test_missing_time_column_raises(self, sample_transactions):
        profiler = TimeSeriesProfiler(
            entity_column="customer_id",
            time_column="nonexistent"
        )

        with pytest.raises(KeyError):
            profiler.profile(sample_transactions)
