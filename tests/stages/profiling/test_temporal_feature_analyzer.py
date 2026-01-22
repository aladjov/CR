"""Tests for TemporalFeatureAnalyzer."""
import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.temporal_feature_analyzer import (
    TemporalFeatureAnalyzer,
    VelocityResult,
    MomentumResult,
    LagCorrelationResult,
    PredictivePowerResult,
    FeatureRecommendation,
)


@pytest.fixture
def sample_event_data():
    """Create sample event-level data with clear patterns."""
    np.random.seed(42)
    n_entities = 100
    events_per_entity = 50

    data = []
    for entity_id in range(n_entities):
        # Churned entities (0) have declining metrics
        # Retained entities (1) have stable/increasing metrics
        is_retained = entity_id % 2 == 0
        target = 1 if is_retained else 0

        base_date = pd.Timestamp("2024-01-01")
        for day in range(events_per_entity):
            if is_retained:
                # Retained: stable with slight increase
                value = 100 + day * 0.5 + np.random.normal(0, 5)
            else:
                # Churned: declining over time
                value = 100 - day * 0.8 + np.random.normal(0, 5)

            data.append({
                "entity_id": entity_id,
                "event_date": base_date + pd.Timedelta(days=day),
                "metric_value": max(0, value),
                "target": target
            })

    return pd.DataFrame(data)


@pytest.fixture
def analyzer():
    """Create analyzer instance."""
    return TemporalFeatureAnalyzer(
        time_column="event_date",
        entity_column="entity_id"
    )


class TestVelocityCalculation:
    """Tests for velocity (rate of change) calculation."""

    def test_calculates_velocity_for_single_column(self, analyzer, sample_event_data):
        result = analyzer.calculate_velocity(
            sample_event_data,
            value_columns=["metric_value"],
            window_days=7
        )

        assert "metric_value" in result
        assert isinstance(result["metric_value"], VelocityResult)
        assert result["metric_value"].window_days == 7

    def test_velocity_detects_increasing_trend(self, analyzer):
        # Create clearly increasing data
        df = pd.DataFrame({
            "entity_id": [1] * 30,
            "event_date": pd.date_range("2024-01-01", periods=30),
            "value": range(30)  # Strictly increasing
        })

        result = analyzer.calculate_velocity(df, ["value"], window_days=7)
        assert result["value"].mean_velocity > 0

    def test_velocity_detects_decreasing_trend(self, analyzer):
        # Create clearly decreasing data
        df = pd.DataFrame({
            "entity_id": [1] * 30,
            "event_date": pd.date_range("2024-01-01", periods=30),
            "value": list(range(30, 0, -1))  # Strictly decreasing
        })

        result = analyzer.calculate_velocity(df, ["value"], window_days=7)
        assert result["value"].mean_velocity < 0


class TestMomentumCalculation:
    """Tests for momentum (window ratio) calculation."""

    def test_calculates_momentum_ratios(self, analyzer, sample_event_data):
        result = analyzer.calculate_momentum(
            sample_event_data,
            value_columns=["metric_value"],
            short_window=7,
            long_window=30
        )

        assert "metric_value" in result
        assert isinstance(result["metric_value"], MomentumResult)
        assert result["metric_value"].short_window == 7
        assert result["metric_value"].long_window == 30

    def test_momentum_above_one_for_increasing(self, analyzer):
        # Recent values higher than historical
        df = pd.DataFrame({
            "entity_id": [1] * 60,
            "event_date": pd.date_range("2024-01-01", periods=60),
            "value": list(range(60))  # Increasing
        })

        result = analyzer.calculate_momentum(df, ["value"], 7, 30)
        assert result["value"].mean_momentum > 1.0

    def test_momentum_below_one_for_decreasing(self, analyzer):
        # Recent values lower than historical
        df = pd.DataFrame({
            "entity_id": [1] * 60,
            "event_date": pd.date_range("2024-01-01", periods=60),
            "value": list(range(60, 0, -1))  # Decreasing
        })

        result = analyzer.calculate_momentum(df, ["value"], 7, 30)
        assert result["value"].mean_momentum < 1.0


class TestLagCorrelation:
    """Tests for lag correlation analysis."""

    def test_calculates_lag_correlations(self, analyzer, sample_event_data):
        result = analyzer.calculate_lag_correlations(
            sample_event_data,
            value_columns=["metric_value"],
            max_lag=14
        )

        assert "metric_value" in result
        assert isinstance(result["metric_value"], LagCorrelationResult)
        assert len(result["metric_value"].correlations) == 14

    def test_identifies_best_lag(self, analyzer, sample_event_data):
        result = analyzer.calculate_lag_correlations(
            sample_event_data,
            value_columns=["metric_value"],
            max_lag=14
        )

        assert result["metric_value"].best_lag >= 1
        assert result["metric_value"].best_lag <= 14


class TestPredictivePower:
    """Tests for IV and KS statistic calculation."""

    def test_calculates_information_value(self, analyzer, sample_event_data):
        result = analyzer.calculate_predictive_power(
            sample_event_data,
            value_columns=["metric_value"],
            target_column="target"
        )

        assert "metric_value" in result
        assert isinstance(result["metric_value"], PredictivePowerResult)
        assert result["metric_value"].information_value >= 0

    def test_calculates_ks_statistic(self, analyzer, sample_event_data):
        result = analyzer.calculate_predictive_power(
            sample_event_data,
            value_columns=["metric_value"],
            target_column="target"
        )

        assert 0 <= result["metric_value"].ks_statistic <= 1
        assert result["metric_value"].ks_pvalue >= 0

    def test_high_iv_for_discriminative_feature(self, analyzer, sample_event_data):
        # Our sample data has clear separation between retained/churned
        result = analyzer.calculate_predictive_power(
            sample_event_data,
            value_columns=["metric_value"],
            target_column="target"
        )

        # Should detect significant predictive power
        assert result["metric_value"].information_value > 0.02


class TestCohortComparison:
    """Tests for retained vs churned comparison."""

    def test_compares_cohorts(self, analyzer, sample_event_data):
        result = analyzer.compare_cohorts(
            sample_event_data,
            value_columns=["metric_value"],
            target_column="target"
        )

        assert "metric_value" in result
        assert "retained" in result["metric_value"]
        assert "churned" in result["metric_value"]

    def test_detects_divergent_velocity(self, analyzer, sample_event_data):
        result = analyzer.compare_cohorts(
            sample_event_data,
            value_columns=["metric_value"],
            target_column="target"
        )

        retained_vel = result["metric_value"]["retained"].velocity
        churned_vel = result["metric_value"]["churned"].velocity

        # Retained should have positive velocity, churned negative
        assert retained_vel > churned_vel


class TestFeatureRecommendations:
    """Tests for feature engineering recommendations."""

    def test_generates_recommendations(self, analyzer, sample_event_data):
        recommendations = analyzer.get_feature_recommendations(
            sample_event_data,
            value_columns=["metric_value"],
            target_column="target"
        )

        assert len(recommendations) > 0
        assert all(isinstance(r, FeatureRecommendation) for r in recommendations)

    def test_recommendations_include_formula(self, analyzer, sample_event_data):
        recommendations = analyzer.get_feature_recommendations(
            sample_event_data,
            value_columns=["metric_value"],
            target_column="target"
        )

        for rec in recommendations:
            assert rec.feature_name
            assert rec.formula
            assert rec.rationale
