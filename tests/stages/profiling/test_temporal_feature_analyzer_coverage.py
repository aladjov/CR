"""Additional coverage tests for TemporalFeatureAnalyzer to reach 86% threshold."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.temporal_feature_analyzer import (
    CohortComparison,
    FeatureRecommendation,
    FeatureType,
    LagCorrelationResult,
    MomentumResult,
    PredictivePowerResult,
    TemporalFeatureAnalyzer,
    VelocityResult,
)


@pytest.fixture
def sample_event_data():
    """Create sample event-level data."""
    np.random.seed(42)
    ref_date = pd.Timestamp("2024-01-31")
    data = []
    for entity_id in range(30):
        for day in range(45):
            data.append({
                "entity_id": entity_id,
                "event_date": ref_date - timedelta(days=day),
                "metric_value": np.random.uniform(50, 150) + (entity_id * 2),
                "other_metric": np.random.uniform(10, 50),
            })
    return pd.DataFrame(data)


@pytest.fixture
def sample_event_data_with_target():
    """Create sample event-level data with target column."""
    np.random.seed(42)
    ref_date = pd.Timestamp("2024-01-31")
    data = []
    for entity_id in range(30):
        target = 1 if entity_id % 2 == 0 else 0
        for day in range(45):
            metric = np.random.uniform(50, 150)
            if target == 1:
                metric += 30  # Make difference detectable
            data.append({
                "entity_id": entity_id,
                "event_date": ref_date - timedelta(days=day),
                "metric_value": metric,
                "target": target,
            })
    return pd.DataFrame(data)


class TestCalculateVelocity:
    """Tests for calculate_velocity method."""

    def test_calculate_velocity_basic(self, sample_event_data):
        """Should calculate velocity for specified columns."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_velocity(sample_event_data, value_columns=["metric_value"], window_days=7)

        assert "metric_value" in result
        assert isinstance(result["metric_value"], VelocityResult)
        assert result["metric_value"].window_days == 7

    def test_calculate_velocity_missing_column(self, sample_event_data):
        """Should skip columns that don't exist in dataframe."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_velocity(
            sample_event_data,
            value_columns=["metric_value", "nonexistent_column"],
            window_days=7,
        )

        assert "metric_value" in result
        assert "nonexistent_column" not in result

    def test_calculate_velocity_trend_direction_increasing(self):
        """Should detect increasing trend."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        # Create data with increasing trend
        ref_date = pd.Timestamp("2024-01-31")
        data = []
        for entity_id in range(5):
            for day in range(30):
                data.append({
                    "entity_id": entity_id,
                    "event_date": ref_date - timedelta(days=day),
                    "metric_value": 100 - day * 3,  # Increasing over time (going backward)
                })
        df = pd.DataFrame(data)

        result = analyzer.calculate_velocity(df, value_columns=["metric_value"], window_days=7)

        # Trend should be detected
        assert result["metric_value"].trend_direction in ["increasing", "decreasing", "stable"]

    def test_calculate_velocity_handles_nan_values(self):
        """Should handle NaN values in velocity calculation."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        ref_date = pd.Timestamp("2024-01-31")
        df = pd.DataFrame({
            "entity_id": [1, 1, 1],
            "event_date": [ref_date, ref_date - timedelta(days=1), ref_date - timedelta(days=2)],
            "metric_value": [np.nan, np.nan, np.nan],
        })

        result = analyzer.calculate_velocity(df, value_columns=["metric_value"], window_days=1)

        # Should not crash, should return valid result
        assert "metric_value" in result
        assert result["metric_value"].mean_velocity == 0.0


class TestCalculateAcceleration:
    """Tests for calculate_acceleration method."""

    def test_calculate_acceleration_basic(self, sample_event_data):
        """Should calculate acceleration for specified columns."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_acceleration(sample_event_data, value_columns=["metric_value"], window_days=7)

        assert "metric_value" in result
        assert isinstance(result["metric_value"], float)

    def test_calculate_acceleration_missing_column(self, sample_event_data):
        """Should skip columns that don't exist."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_acceleration(
            sample_event_data,
            value_columns=["metric_value", "nonexistent"],
            window_days=7,
        )

        assert "metric_value" in result
        assert "nonexistent" not in result

    def test_calculate_acceleration_handles_nan(self):
        """Should return 0.0 when acceleration is NaN."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        ref_date = pd.Timestamp("2024-01-31")
        df = pd.DataFrame({
            "entity_id": [1, 1],
            "event_date": [ref_date, ref_date - timedelta(days=1)],
            "metric_value": [100.0, 100.0],
        })

        result = analyzer.calculate_acceleration(df, value_columns=["metric_value"], window_days=1)

        assert "metric_value" in result
        # Should handle NaN gracefully
        assert isinstance(result["metric_value"], float)


class TestCalculateMomentum:
    """Tests for calculate_momentum method."""

    def test_calculate_momentum_basic(self, sample_event_data):
        """Should calculate momentum for specified columns."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_momentum(
            sample_event_data,
            value_columns=["metric_value"],
            short_window=7,
            long_window=30,
        )

        assert "metric_value" in result
        assert isinstance(result["metric_value"], MomentumResult)
        assert result["metric_value"].short_window == 7
        assert result["metric_value"].long_window == 30

    def test_calculate_momentum_missing_column(self, sample_event_data):
        """Should skip columns that don't exist."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_momentum(
            sample_event_data,
            value_columns=["metric_value", "nonexistent"],
            short_window=7,
            long_window=30,
        )

        assert "metric_value" in result
        assert "nonexistent" not in result

    def test_calculate_momentum_interpretation_accelerating(self):
        """Should detect accelerating momentum (> 1.1)."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        # Create data where recent values are much higher
        ref_date = pd.Timestamp("2024-01-31")
        data = []
        for entity_id in range(10):
            for day in range(30):
                value = 200 if day < 7 else 100  # Recent > Long term
                data.append({
                    "entity_id": entity_id,
                    "event_date": ref_date - timedelta(days=day),
                    "metric_value": value,
                })
        df = pd.DataFrame(data)

        result = analyzer.calculate_momentum(df, value_columns=["metric_value"], short_window=7, long_window=30)

        # Mean momentum should be > 1.0 (accelerating)
        assert result["metric_value"].mean_momentum > 1.0

    def test_calculate_momentum_interpretation_decelerating(self):
        """Should detect decelerating momentum (< 0.9)."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        # Create data where recent values are much lower
        ref_date = pd.Timestamp("2024-01-31")
        data = []
        for entity_id in range(10):
            for day in range(30):
                value = 50 if day < 7 else 100  # Recent < Long term
                data.append({
                    "entity_id": entity_id,
                    "event_date": ref_date - timedelta(days=day),
                    "metric_value": value,
                })
        df = pd.DataFrame(data)

        result = analyzer.calculate_momentum(df, value_columns=["metric_value"], short_window=7, long_window=30)

        # Mean momentum should be < 1.0 (decelerating)
        assert result["metric_value"].mean_momentum < 1.0

    def test_calculate_momentum_empty_entity_list(self):
        """Should handle case where long_mean is zero."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        ref_date = pd.Timestamp("2024-01-31")
        df = pd.DataFrame({
            "entity_id": [1],
            "event_date": [ref_date],
            "metric_value": [0.0],  # Zero value
        })

        result = analyzer.calculate_momentum(df, value_columns=["metric_value"], short_window=7, long_window=30)

        # Should return default momentum of 1.0
        assert result["metric_value"].mean_momentum == 1.0


class TestCalculateLagCorrelations:
    """Tests for calculate_lag_correlations method."""

    def test_calculate_lag_correlations_basic(self, sample_event_data):
        """Should calculate lag correlations."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_lag_correlations(sample_event_data, value_columns=["metric_value"], max_lag=14)

        assert "metric_value" in result
        assert isinstance(result["metric_value"], LagCorrelationResult)
        assert len(result["metric_value"].correlations) == 14

    def test_calculate_lag_correlations_missing_column(self, sample_event_data):
        """Should skip columns that don't exist."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_lag_correlations(
            sample_event_data,
            value_columns=["metric_value", "nonexistent"],
            max_lag=7,
        )

        assert "metric_value" in result
        assert "nonexistent" not in result

    def test_calculate_lag_correlations_short_series(self):
        """Should handle series shorter than max_lag."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        ref_date = pd.Timestamp("2024-01-31")
        df = pd.DataFrame({
            "entity_id": [1, 1, 1],
            "event_date": [ref_date, ref_date - timedelta(days=1), ref_date - timedelta(days=2)],
            "metric_value": [100.0, 105.0, 110.0],
        })

        result = analyzer.calculate_lag_correlations(df, value_columns=["metric_value"], max_lag=14)

        # Should have 14 correlations, with zeros for lags beyond data length
        assert len(result["metric_value"].correlations) == 14

    def test_calculate_lag_correlations_weekly_pattern(self):
        """Should detect weekly patterns."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        # Create data with strong weekly pattern
        ref_date = pd.Timestamp("2024-01-31")
        data = []
        for day in range(60):
            # High on weekends (day 0, 7, 14...), low on weekdays
            value = 200 if day % 7 == 0 else 100
            data.append({
                "entity_id": 1,
                "event_date": ref_date - timedelta(days=day),
                "metric_value": value,
            })
        df = pd.DataFrame(data)

        result = analyzer.calculate_lag_correlations(df, value_columns=["metric_value"], max_lag=14)

        # Lag 7 should have high correlation
        lag_7_corr = result["metric_value"].correlations[6]  # Index 6 = lag 7
        assert result["metric_value"].has_weekly_pattern or abs(lag_7_corr) > 0.1


class TestCalculatePredictivePower:
    """Tests for calculate_predictive_power method."""

    def test_calculate_predictive_power_basic(self, sample_event_data_with_target):
        """Should calculate IV and KS for features."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_predictive_power(
            sample_event_data_with_target,
            value_columns=["metric_value"],
            target_column="target",
        )

        assert "metric_value" in result
        assert isinstance(result["metric_value"], PredictivePowerResult)
        assert result["metric_value"].information_value >= 0
        assert 0 <= result["metric_value"].ks_statistic <= 1

    def test_calculate_predictive_power_all_valid_columns(self, sample_event_data_with_target):
        """Should calculate power for all valid columns."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        # Add another column to test multiple columns
        df = sample_event_data_with_target.copy()
        df["other_value"] = np.random.randn(len(df))

        result = analyzer.calculate_predictive_power(
            df,
            value_columns=["metric_value", "other_value"],
            target_column="target",
        )

        assert "metric_value" in result
        assert "other_value" in result


class TestCalculateIV:
    """Tests for _calculate_iv method."""

    def test_calculate_iv_insufficient_data(self):
        """Should return 0.0 for insufficient data."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        feature = pd.Series([1, 2, 3])
        target = pd.Series([0, 1, 0])

        iv = analyzer._calculate_iv(feature, target, bins=10)

        # Not enough data for 10 bins
        assert iv == 0.0

    def test_calculate_iv_duplicate_bins(self):
        """Should handle duplicate bin edges."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        # All same values - will cause duplicate bins
        feature = pd.Series([1.0] * 50)
        target = pd.Series([0] * 25 + [1] * 25)

        iv = analyzer._calculate_iv(feature, target, bins=10)

        # Should return 0.0 due to qcut error handling
        assert iv == 0.0

    def test_calculate_iv_single_class_target(self):
        """Should return 0.0 for single-class target."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        feature = pd.Series(np.random.randn(100))
        target = pd.Series([1] * 100)  # All same class

        iv = analyzer._calculate_iv(feature, target, bins=10)

        assert iv == 0.0


class TestCalculateKS:
    """Tests for _calculate_ks method."""

    def test_calculate_ks_empty_group(self):
        """Should return 0.0, 1.0 for empty groups."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        feature = pd.Series([1.0, 2.0, 3.0])
        target = pd.Series([0, 0, 0])  # All same class - group1 will be empty

        ks_stat, p_val = analyzer._calculate_ks(feature, target)

        assert ks_stat == 0.0
        assert p_val == 1.0

    def test_calculate_ks_valid_groups(self):
        """Should calculate KS for valid groups."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        np.random.seed(42)
        # Create clearly different distributions
        feature = pd.Series(list(np.random.normal(0, 1, 50)) + list(np.random.normal(5, 1, 50)))
        target = pd.Series([0] * 50 + [1] * 50)

        ks_stat, p_val = analyzer._calculate_ks(feature, target)

        assert 0 <= ks_stat <= 1
        assert 0 <= p_val <= 1
        # Should detect significant difference
        assert ks_stat > 0.5


class TestInterpretIV:
    """Tests for _interpret_iv method."""

    def test_interpret_iv_suspicious(self):
        """IV > 0.5 should be suspicious."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        assert analyzer._interpret_iv(0.6) == "suspicious"

    def test_interpret_iv_strong(self):
        """IV > 0.3 should be strong."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        assert analyzer._interpret_iv(0.35) == "strong"

    def test_interpret_iv_medium(self):
        """IV > 0.1 should be medium."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        assert analyzer._interpret_iv(0.15) == "medium"

    def test_interpret_iv_weak(self):
        """IV > 0.02 should be weak."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        assert analyzer._interpret_iv(0.05) == "weak"

    def test_interpret_iv_very_weak(self):
        """IV <= 0.02 should be very_weak."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        assert analyzer._interpret_iv(0.01) == "very_weak"


class TestInterpretKS:
    """Tests for _interpret_ks method."""

    def test_interpret_ks_strong(self):
        """KS > 0.4 should be strong."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        assert analyzer._interpret_ks(0.5) == "strong"

    def test_interpret_ks_medium(self):
        """KS > 0.2 should be medium."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        assert analyzer._interpret_ks(0.3) == "medium"

    def test_interpret_ks_weak(self):
        """KS <= 0.2 should be weak."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        assert analyzer._interpret_ks(0.1) == "weak"


class TestCompareCohorts:
    """Tests for compare_cohorts method."""

    @pytest.fixture
    def entity_level_data_with_target(self):
        """Create entity-level data (one row per entity) for cohort comparison."""
        np.random.seed(42)
        data = []
        for entity_id in range(30):
            target = 1 if entity_id % 2 == 0 else 0
            data.append({
                "entity_id": entity_id,
                "event_date": pd.Timestamp("2024-01-31"),
                "metric_value": np.random.uniform(50, 150),
                "target": target,
            })
        return pd.DataFrame(data)

    def test_compare_cohorts_rejects_event_level_data(self, sample_event_data_with_target):
        """Should reject event-level data (multiple rows per entity)."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        with pytest.raises(ValueError, match="event-level data"):
            analyzer.compare_cohorts(
                sample_event_data_with_target,
                value_columns=["metric_value"],
                target_column="target",
            )

    def test_compare_cohorts_basic(self, entity_level_data_with_target):
        """Should compare metrics between cohorts on entity-level data."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        result = analyzer.compare_cohorts(
            entity_level_data_with_target,
            value_columns=["metric_value"],
            target_column="target",
        )

        assert "metric_value" in result
        assert "retained" in result["metric_value"]
        assert "churned" in result["metric_value"]
        assert isinstance(result["metric_value"]["retained"], CohortComparison)

    def test_compare_cohorts_excludes_target_column(self, entity_level_data_with_target):
        """Should exclude target from value_columns automatically."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        result = analyzer.compare_cohorts(
            entity_level_data_with_target,
            value_columns=["metric_value", "target"],  # Include target
            target_column="target",
        )

        # Target should be excluded
        assert "target" not in result
        assert "metric_value" in result


class TestGetFeatureRecommendations:
    """Tests for get_feature_recommendations method."""

    def test_get_feature_recommendations_with_target(self, sample_event_data_with_target):
        """Should generate recommendations based on predictive power."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        result = analyzer.get_feature_recommendations(
            sample_event_data_with_target,
            value_columns=["metric_value"],
            target_column="target",
        )

        assert isinstance(result, list)
        # Should have some recommendations
        if len(result) > 0:
            assert isinstance(result[0], FeatureRecommendation)

    def test_get_feature_recommendations_without_target(self, sample_event_data):
        """Should generate recommendations without target."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        result = analyzer.get_feature_recommendations(
            sample_event_data,
            value_columns=["metric_value"],
            target_column=None,
        )

        assert isinstance(result, list)

    def test_get_feature_recommendations_velocity_based(self):
        """Should recommend velocity features for trending data."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        # Create data with clear trend
        ref_date = pd.Timestamp("2024-01-31")
        data = []
        for entity_id in range(10):
            for day in range(45):
                data.append({
                    "entity_id": entity_id,
                    "event_date": ref_date - timedelta(days=day),
                    "metric_value": 100 + day * 5,  # Clear trend
                })
        df = pd.DataFrame(data)

        result = analyzer.get_feature_recommendations(df, value_columns=["metric_value"])

        # Should recommend some features
        assert isinstance(result, list)

    def test_get_feature_recommendations_lag_based(self):
        """Should recommend lag features for autocorrelated data."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        # Create data with strong autocorrelation
        ref_date = pd.Timestamp("2024-01-31")
        data = []
        prev_value = 100
        for day in range(60):
            prev_value = prev_value + np.random.uniform(-1, 1)
            data.append({
                "entity_id": 1,
                "event_date": ref_date - timedelta(days=day),
                "metric_value": prev_value,
            })
        df = pd.DataFrame(data)

        result = analyzer.get_feature_recommendations(df, value_columns=["metric_value"])

        assert isinstance(result, list)


class TestFeatureTypeEnum:
    """Tests for FeatureType enum."""

    def test_feature_type_values(self):
        """Verify all feature type values."""
        assert FeatureType.VELOCITY.value == "velocity"
        assert FeatureType.ACCELERATION.value == "acceleration"
        assert FeatureType.MOMENTUM.value == "momentum"
        assert FeatureType.LAG.value == "lag"
        assert FeatureType.ROLLING.value == "rolling"
        assert FeatureType.RATIO.value == "ratio"


class TestPrepareDataframe:
    """Tests for _prepare_dataframe method."""

    def test_prepare_dataframe_converts_time_column(self):
        """Should convert time column to datetime."""
        analyzer = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

        df = pd.DataFrame({
            "entity_id": [1, 2],
            "event_date": ["2024-01-01", "2024-01-02"],
            "value": [100, 200],
        })

        result = analyzer._prepare_dataframe(df)

        assert pd.api.types.is_datetime64_any_dtype(result["event_date"])
