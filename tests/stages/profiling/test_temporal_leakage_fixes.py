"""Tests for temporal leakage fixes in profiling stage."""

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.temporal_feature_analyzer import (
    TemporalFeatureAnalyzer,
)
from customer_retention.stages.profiling.time_window_aggregator import (
    TimeWindowAggregator,
    save_aggregated_parquet,
)


@pytest.fixture
def sample_event_data_with_target():
    """Create sample event-level data with target column."""
    np.random.seed(42)
    ref_date = pd.Timestamp("2024-01-31")
    data = []
    for entity_id in range(20):
        target = 1 if entity_id % 2 == 0 else 0
        for day in range(30):
            data.append(
                {
                    "entity_id": entity_id,
                    "event_date": ref_date - timedelta(days=day),
                    "metric_value": np.random.uniform(50, 150),
                    "target": target,
                }
            )
    return pd.DataFrame(data)


class TestTargetExclusionFromMomentumAnalysis:
    """Tests for Issue #1: Target column included in momentum analysis."""

    def test_compare_cohorts_rejects_event_level_data(self, sample_event_data_with_target):
        """compare_cohorts should reject event-level data (multiple rows per entity).

        Target comparisons are only valid at entity level. Event-level data
        should be aggregated first using TimeWindowAggregator.
        """
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        # Event-level data should raise ValueError
        with pytest.raises(ValueError, match="event-level data"):
            analyzer.compare_cohorts(
                sample_event_data_with_target, value_columns=["metric_value", "target"], target_column="target"
            )

    def test_compare_cohorts_works_on_entity_level_data(self):
        """compare_cohorts should work on entity-level data (one row per entity)."""
        np.random.seed(42)
        # Create entity-level data (one row per entity)
        entity_data = pd.DataFrame({
            "entity_id": range(20),
            "event_date": pd.Timestamp("2024-01-31"),
            "metric_value": np.random.uniform(50, 150, 20),
            "target": [1 if i % 2 == 0 else 0 for i in range(20)],
        })

        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        # Entity-level data should work
        result = analyzer.compare_cohorts(
            entity_data, value_columns=["metric_value", "target"], target_column="target"
        )

        # Target should NOT be in the results (it's excluded from value columns)
        assert "target" not in result, "Target column should be excluded from momentum analysis"
        assert "metric_value" in result

    def test_calculate_momentum_excludes_target_when_specified(self, sample_event_data_with_target):
        """Momentum calculation should allow target exclusion."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        # Should work without target in value_columns
        result = analyzer.calculate_momentum(
            sample_event_data_with_target, value_columns=["metric_value"], short_window=7, long_window=30
        )

        assert "metric_value" in result
        assert "target" not in result

    def test_calculate_velocity_excludes_target_when_specified(self, sample_event_data_with_target):
        """Velocity calculation should allow target exclusion."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        result = analyzer.calculate_velocity(
            sample_event_data_with_target, value_columns=["metric_value"], window_days=7
        )

        assert "metric_value" in result
        assert "target" not in result


class TestReferenceDateWarning:
    """Tests for Issue #4: Reference date defaults silently to max date."""

    def test_aggregate_warns_when_reference_date_is_none(self):
        """Aggregation should warn when reference_date defaults to max date."""
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C001", "C002"],
                "date": pd.date_range("2023-12-01", periods=3, freq="D"),
                "amount": [100, 200, 300],
            }
        )

        aggregator = TimeWindowAggregator(entity_column="customer_id", time_column="date")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            aggregator.aggregate(
                df,
                windows=["30d"],
                value_columns=["amount"],
                agg_funcs=["sum"],
                reference_date=None,  # Should trigger warning
            )

            # Check that a warning was issued
            assert len(w) == 1
            assert "reference_date not provided" in str(w[0].message)
            assert "PIT correctness" in str(w[0].message)

    def test_aggregate_no_warning_when_reference_date_provided(self):
        """No warning when reference_date is explicitly provided."""
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C001"],
                "date": pd.date_range("2023-12-01", periods=2, freq="D"),
                "amount": [100, 200],
            }
        )

        aggregator = TimeWindowAggregator(entity_column="customer_id", time_column="date")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            aggregator.aggregate(
                df,
                windows=["30d"],
                value_columns=["amount"],
                agg_funcs=["sum"],
                reference_date=pd.Timestamp("2023-12-31"),
            )

            # Filter for our specific warning
            pit_warnings = [x for x in w if "reference_date" in str(x.message)]
            assert len(pit_warnings) == 0


class TestTemporalMetadataStorage:
    """Tests for Issue #3: Aggregated data lacks temporal metadata."""

    def test_aggregate_stores_reference_date_in_attrs(self):
        """Aggregated DataFrame should store reference_date in attrs."""
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C001", "C002"],
                "date": pd.date_range("2023-12-01", periods=3, freq="D"),
                "amount": [100, 200, 300],
            }
        )

        aggregator = TimeWindowAggregator(entity_column="customer_id", time_column="date")

        ref_date = pd.Timestamp("2023-12-31")
        result = aggregator.aggregate(
            df, windows=["30d"], value_columns=["amount"], agg_funcs=["sum"], reference_date=ref_date
        )

        assert "aggregation_reference_date" in result.attrs
        # Reference date is stored as ISO string for serialization compatibility
        stored_ref = result.attrs["aggregation_reference_date"]
        if isinstance(stored_ref, str):
            assert stored_ref == ref_date.isoformat()
        else:
            assert stored_ref == ref_date

    def test_aggregate_stores_timestamp_in_attrs(self):
        """Aggregated DataFrame should store aggregation timestamp in attrs."""
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002"],
                "date": pd.date_range("2023-12-01", periods=2, freq="D"),
                "amount": [100, 200],
            }
        )

        aggregator = TimeWindowAggregator(entity_column="customer_id", time_column="date")

        result = aggregator.aggregate(
            df, windows=["30d"], value_columns=["amount"], agg_funcs=["sum"], reference_date=pd.Timestamp("2023-12-31")
        )

        assert "aggregation_timestamp" in result.attrs
        # Timestamp should be ISO format string
        assert isinstance(result.attrs["aggregation_timestamp"], str)

    def test_save_aggregated_parquet_preserves_metadata(self, tmp_path):
        """save_aggregated_parquet should preserve temporal metadata."""
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002"],
                "feature_1": [1.0, 2.0],
            }
        )
        ref_date = pd.Timestamp("2023-12-31")
        df.attrs["aggregation_reference_date"] = ref_date
        df.attrs["aggregation_timestamp"] = "2024-01-01T12:00:00"

        output_path = tmp_path / "test_aggregated.parquet"
        metadata = save_aggregated_parquet(df, output_path)

        # Metadata should be returned
        assert "aggregation_reference_date" in metadata
        assert "aggregation_timestamp" in metadata

        # File should exist
        assert output_path.exists()

        # Metadata should be recoverable from parquet
        import pyarrow.parquet as pq

        table = pq.read_table(output_path)
        schema_metadata = table.schema.metadata
        assert b"aggregation_reference_date" in schema_metadata

    def test_save_aggregated_parquet_with_no_attrs(self, tmp_path):
        """save_aggregated_parquet should handle DataFrames without attrs."""
        df = pd.DataFrame(
            {
                "customer_id": ["C001"],
                "feature_1": [1.0],
            }
        )

        output_path = tmp_path / "test_no_attrs.parquet"
        metadata = save_aggregated_parquet(df, output_path)

        # Should still work, just with empty metadata
        assert output_path.exists()
        assert isinstance(metadata, dict)
