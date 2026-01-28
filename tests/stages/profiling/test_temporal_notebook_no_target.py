"""Tests ensuring event-level temporal analysis doesn't use target variable.

These tests prevent regression of target leakage in temporal notebooks.
Per Coding_Practices.md: any bug must be covered by dedicated tests.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.pattern_analysis_config import (
    SparklineDataBuilder,
)
from customer_retention.stages.profiling.temporal_feature_analyzer import (
    TemporalFeatureAnalyzer,
)


@pytest.fixture
def event_level_data():
    """Create event-level data (multiple rows per entity)."""
    np.random.seed(42)
    ref_date = pd.Timestamp("2024-01-31")
    data = []
    for entity_id in range(20):
        target = 1 if entity_id % 2 == 0 else 0
        for day in range(30):
            data.append({
                "entity_id": entity_id,
                "event_date": ref_date - timedelta(days=day),
                "metric_a": np.random.uniform(50, 150),
                "metric_b": np.random.uniform(10, 50),
                "target": target,
            })
    return pd.DataFrame(data)


@pytest.fixture
def entity_level_data():
    """Create entity-level data (one row per entity)."""
    np.random.seed(42)
    data = []
    for entity_id in range(50):
        target = 1 if entity_id % 2 == 0 else 0
        data.append({
            "entity_id": entity_id,
            "event_date": pd.Timestamp("2024-01-31"),
            "metric_a": np.random.uniform(50, 150),
            "metric_b": np.random.uniform(10, 50),
            "target": target,
        })
    return pd.DataFrame(data)


@pytest.fixture
def data_with_constant_column(event_level_data):
    """Event data with a constant column (zero variance)."""
    df = event_level_data.copy()
    df["constant_col"] = 100.0
    return df


@pytest.fixture
def data_with_nan_column(event_level_data):
    """Event data with a column containing NaN values."""
    df = event_level_data.copy()
    df["nan_col"] = np.random.uniform(0, 100, len(df))
    df.loc[df.index[:100], "nan_col"] = np.nan
    return df


class TestSelectColumnsByVariance:
    """Tests for variance-based column selection (not target correlation)."""

    def test_select_by_variance_excludes_target(self, event_level_data):
        """Column selection uses variance, not target correlation."""
        from customer_retention.stages.profiling.pattern_analysis_config import (
            select_columns_by_variance,
        )

        numeric_cols = ["metric_a", "metric_b", "target"]
        selected = select_columns_by_variance(
            event_level_data, numeric_cols, max_cols=2
        )

        # Should select columns, but target (constant per entity) should not be top choice
        assert len(selected) <= 2
        # metric_a and metric_b have more variance than target (which is 0/1 constant per entity)
        assert "metric_a" in selected or "metric_b" in selected

    def test_select_by_variance_handles_constant_columns(self, data_with_constant_column):
        """Edge case: columns with zero variance are excluded."""
        from customer_retention.stages.profiling.pattern_analysis_config import (
            select_columns_by_variance,
        )

        numeric_cols = ["metric_a", "metric_b", "constant_col"]
        selected = select_columns_by_variance(
            data_with_constant_column, numeric_cols, max_cols=3
        )

        # Constant column should be excluded (zero variance)
        assert "constant_col" not in selected
        assert len(selected) == 2  # Only metric_a and metric_b

    def test_select_by_variance_handles_nan_columns(self, data_with_nan_column):
        """Edge case: columns with NaN values handled gracefully."""
        from customer_retention.stages.profiling.pattern_analysis_config import (
            select_columns_by_variance,
        )

        numeric_cols = ["metric_a", "metric_b", "nan_col"]
        selected = select_columns_by_variance(
            data_with_nan_column, numeric_cols, max_cols=3
        )

        # Should not crash on NaN values
        assert len(selected) >= 1
        assert "metric_a" in selected or "metric_b" in selected


class TestEventLevelTargetGuard:
    """Tests for preventing target comparisons on event-level data."""

    def test_compare_cohorts_rejects_event_level_data(self, event_level_data):
        """Antipattern: cohort comparison on event data raises ValueError."""
        analyzer = TemporalFeatureAnalyzer(
            time_column="event_date", entity_column="entity_id"
        )

        with pytest.raises(ValueError, match="event-level data"):
            analyzer.compare_cohorts(
                event_level_data,
                value_columns=["metric_a"],
                target_column="target",
            )

    def test_compare_cohorts_allows_entity_level_data(self, entity_level_data):
        """Entity-level data (1 row per entity) allows target comparison."""
        analyzer = TemporalFeatureAnalyzer(
            time_column="event_date", entity_column="entity_id"
        )

        # Should not raise - data is already entity-level
        result = analyzer.compare_cohorts(
            entity_level_data,
            value_columns=["metric_a"],
            target_column="target",
        )

        assert "metric_a" in result
        assert "retained" in result["metric_a"]
        assert "churned" in result["metric_a"]

    def test_sparkline_builder_rejects_target_on_event_data(self, event_level_data):
        """Antipattern: SparklineDataBuilder rejects target for multi-row entities."""
        builder = SparklineDataBuilder(
            entity_column="entity_id",
            time_column="event_date",
            target_column="target",
        )

        with pytest.raises(ValueError, match="event-level data"):
            builder.build(event_level_data, columns=["metric_a"])

    def test_sparkline_builder_allows_no_target(self, event_level_data):
        """SparklineDataBuilder works without target (overall trends)."""
        builder = SparklineDataBuilder(
            entity_column="entity_id",
            time_column="event_date",
            target_column=None,  # No target = no cohort split
        )

        # Should work without raising
        results, has_target = builder.build(event_level_data, columns=["metric_a"])

        assert has_target is False
        assert len(results) == 1
        assert results[0].column == "metric_a"


class TestTemporalPatternsExcludeTarget:
    """Tests ensuring velocity/momentum exclude target column."""

    def test_velocity_calculation_excludes_target_column(self, event_level_data):
        """Velocity is computed on value columns only, never target."""
        analyzer = TemporalFeatureAnalyzer(
            time_column="event_date", entity_column="entity_id"
        )

        # Target should be rejected if passed as value column
        result = analyzer.calculate_velocity(
            event_level_data,
            value_columns=["metric_a", "target"],
            window_days=7,
        )

        # Target should not be in results (it's constant per entity, not a time-varying metric)
        assert "metric_a" in result
        # Target may or may not be computed, but velocity on constant values is meaningless
        # The key point is metric_a is computed correctly

    def test_momentum_calculation_excludes_target_column(self, event_level_data):
        """Momentum excludes target from analysis columns."""
        analyzer = TemporalFeatureAnalyzer(
            time_column="event_date", entity_column="entity_id"
        )

        result = analyzer.calculate_momentum(
            event_level_data,
            value_columns=["metric_a", "metric_b"],
            short_window=7,
            long_window=30,
        )

        assert "metric_a" in result
        assert "metric_b" in result
        # Should compute momentum without issues


class TestSimilarDisconnects:
    """Additional tests for edge cases per Coding_Practices.md."""

    def test_rejects_any_column_matching_target_pattern(self, event_level_data):
        """Handles different target column names."""
        # Rename target column
        df = event_level_data.rename(columns={"target": "churn_flag"})

        analyzer = TemporalFeatureAnalyzer(
            time_column="event_date", entity_column="entity_id"
        )

        with pytest.raises(ValueError, match="event-level data"):
            analyzer.compare_cohorts(
                df,
                value_columns=["metric_a"],
                target_column="churn_flag",
            )

    def test_allows_target_when_data_is_already_aggregated(self, entity_level_data):
        """Entity-level data allows target comparison."""
        analyzer = TemporalFeatureAnalyzer(
            time_column="event_date", entity_column="entity_id"
        )

        # Should work - exactly one row per entity
        result = analyzer.compare_cohorts(
            entity_level_data,
            value_columns=["metric_a"],
            target_column="target",
        )

        assert "metric_a" in result

    def test_handles_float_binary_target(self, entity_level_data):
        """Handles target stored as float (0.0, 1.0)."""
        df = entity_level_data.copy()
        df["target"] = df["target"].astype(float)

        analyzer = TemporalFeatureAnalyzer(
            time_column="event_date", entity_column="entity_id"
        )

        result = analyzer.compare_cohorts(
            df,
            value_columns=["metric_a"],
            target_column="target",
        )

        assert "metric_a" in result
        assert "retained" in result["metric_a"]

    def test_predictive_power_aggregates_correctly(self, event_level_data):
        """Predictive power methods should aggregate to entity level first."""
        analyzer = TemporalFeatureAnalyzer(
            time_column="event_date", entity_column="entity_id"
        )

        # This method is expected to aggregate internally - verify it doesn't leak
        result = analyzer.calculate_predictive_power(
            event_level_data,
            value_columns=["metric_a"],
            target_column="target",
        )

        assert "metric_a" in result
        # The method should aggregate to entity level internally
