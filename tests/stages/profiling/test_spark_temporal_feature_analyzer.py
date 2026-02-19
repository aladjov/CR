import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.spark_temporal_feature_analyzer import (
    SparkTemporalFeatureAnalyzer,
)
from customer_retention.stages.profiling.temporal_feature_analyzer import (
    LagCorrelationResult,
    MomentumResult,
    TemporalFeatureAnalyzer,
    VelocityResult,
)

BOTH_ANALYZERS = [TemporalFeatureAnalyzer, SparkTemporalFeatureAnalyzer]


@pytest.fixture
def event_data():
    np.random.seed(42)
    data = []
    base_date = pd.Timestamp("2024-01-01")
    for entity_id in range(50):
        is_retained = entity_id % 2 == 0
        for day in range(30):
            value = (100 + day * 0.5 if is_retained else 100 - day * 0.8)
            value += np.random.normal(0, 3)
            data.append({
                "entity_id": entity_id,
                "event_date": base_date + pd.Timedelta(days=day),
                "metric_value": max(0, value),
                "target": 1 if is_retained else 0,
            })
    return pd.DataFrame(data)


@pytest.fixture
def entity_level_data():
    np.random.seed(42)
    data = []
    for entity_id in range(100):
        is_retained = entity_id % 2 == 0
        value = (120 if is_retained else 80) + np.random.normal(0, 10)
        data.append({
            "entity_id": entity_id,
            "event_date": pd.Timestamp("2024-01-31"),
            "metric_value": max(0, value),
            "target": 1 if is_retained else 0,
        })
    return pd.DataFrame(data)


class TestSparkAnalyzerMatchesLocal:

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_velocity_returns_result(self, event_data, cls):
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_velocity(event_data, ["metric_value"], window_days=7)
        assert "metric_value" in result
        assert isinstance(result["metric_value"], VelocityResult)

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_acceleration_returns_float(self, event_data, cls):
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_acceleration(event_data, ["metric_value"], window_days=7)
        assert "metric_value" in result
        assert isinstance(result["metric_value"], float)

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_momentum_returns_result(self, event_data, cls):
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_momentum(event_data, ["metric_value"], 7, 30)
        assert "metric_value" in result
        assert isinstance(result["metric_value"], MomentumResult)

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_lag_correlations_returns_result(self, event_data, cls):
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_lag_correlations(event_data, ["metric_value"], max_lag=7)
        assert "metric_value" in result
        assert isinstance(result["metric_value"], LagCorrelationResult)
        assert len(result["metric_value"].correlations) == 7

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_predictive_power_returns_result(self, event_data, cls):
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_predictive_power(
            event_data, ["metric_value"], "target")
        assert "metric_value" in result
        assert result["metric_value"].information_value >= 0

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_cohort_velocity_signals(self, event_data, cls):
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        result = analyzer.compute_cohort_velocity_signals(
            event_data, ["metric_value"], "target", windows=[7, 14])
        assert "metric_value" in result
        assert len(result["metric_value"]) == 2

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_cohort_momentum_signals(self, event_data, cls):
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        result = analyzer.compute_cohort_momentum_signals(
            event_data, ["metric_value"], "target", window_pairs=[(7, 30)])
        assert "metric_value" in result
        assert len(result["metric_value"]) == 1

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_compare_cohorts(self, entity_level_data, cls):
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        result = analyzer.compare_cohorts(
            entity_level_data, ["metric_value"], "target")
        assert "metric_value" in result
        assert "retained" in result["metric_value"]
        assert "churned" in result["metric_value"]

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_feature_recommendations(self, event_data, cls):
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        recs = analyzer.get_feature_recommendations(
            event_data, ["metric_value"], target_column="target")
        assert isinstance(recs, list)


class TestSparkAnalyzerNumericMatch:

    def test_velocity_values_match(self, event_data):
        local = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        spark = SparkTemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        r_local = local.calculate_velocity(event_data, ["metric_value"], window_days=7)
        r_spark = spark.calculate_velocity(event_data, ["metric_value"], window_days=7)
        np.testing.assert_allclose(
            r_local["metric_value"].mean_velocity,
            r_spark["metric_value"].mean_velocity,
            rtol=1e-6, atol=1e-10,
        )
        np.testing.assert_allclose(
            r_local["metric_value"].std_velocity,
            r_spark["metric_value"].std_velocity,
            rtol=1e-6, atol=1e-10,
        )

    def test_acceleration_values_match(self, event_data):
        local = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        spark = SparkTemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        r_local = local.calculate_acceleration(event_data, ["metric_value"], window_days=7)
        r_spark = spark.calculate_acceleration(event_data, ["metric_value"], window_days=7)
        np.testing.assert_allclose(
            r_local["metric_value"], r_spark["metric_value"],
            rtol=1e-6, atol=1e-10,
        )

    def test_momentum_values_match(self, event_data):
        local = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        spark = SparkTemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        r_local = local.calculate_momentum(event_data, ["metric_value"], 7, 30)
        r_spark = spark.calculate_momentum(event_data, ["metric_value"], 7, 30)
        np.testing.assert_allclose(
            r_local["metric_value"].mean_momentum,
            r_spark["metric_value"].mean_momentum,
            rtol=1e-2,
        )

    def test_lag_correlations_match(self, event_data):
        local = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        spark = SparkTemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        r_local = local.calculate_lag_correlations(event_data, ["metric_value"], max_lag=7)
        r_spark = spark.calculate_lag_correlations(event_data, ["metric_value"], max_lag=7)
        np.testing.assert_allclose(
            r_local["metric_value"].correlations,
            r_spark["metric_value"].correlations,
            rtol=1e-6, atol=1e-10,
        )

    def test_cohort_velocity_effect_sizes_match(self, event_data):
        local = TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        spark = SparkTemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")
        r_local = local.compute_cohort_velocity_signals(
            event_data, ["metric_value"], "target", windows=[7])
        r_spark = spark.compute_cohort_velocity_signals(
            event_data, ["metric_value"], "target", windows=[7])
        np.testing.assert_allclose(
            r_local["metric_value"][0].velocity_effect_size,
            r_spark["metric_value"][0].velocity_effect_size,
            rtol=1e-2,
        )


class TestSparkAnalyzerEdgeCases:

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_empty_dataframe(self, cls):
        df = pd.DataFrame(columns=["entity_id", "event_date", "value"])
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        vel, accel = analyzer._velocity_accel_series(df, "value", 7)
        assert vel == []
        assert accel == []

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_single_entity_single_day(self, cls):
        df = pd.DataFrame({
            "entity_id": [1],
            "event_date": [pd.Timestamp("2024-01-01")],
            "value": [100.0],
        })
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_velocity(df, ["value"], window_days=7)
        assert "value" in result

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_all_nan_column(self, cls):
        df = pd.DataFrame({
            "entity_id": [1] * 30,
            "event_date": pd.date_range("2024-01-01", periods=30),
            "value": [np.nan] * 30,
        })
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_velocity(df, ["value"], window_days=7)
        assert "value" in result

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_missing_value_column_skipped(self, cls):
        df = pd.DataFrame({
            "entity_id": [1] * 10,
            "event_date": pd.date_range("2024-01-01", periods=10),
        })
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_velocity(df, ["nonexistent"], window_days=7)
        assert result == {}

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_momentum_zero_long_mean(self, cls):
        df = pd.DataFrame({
            "entity_id": [1] * 60,
            "event_date": pd.date_range("2024-01-01", periods=60),
            "value": [0.0] * 60,
        })
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_momentum(df, ["value"], 7, 30)
        assert result["value"].mean_momentum == 1.0

    @pytest.mark.parametrize("cls", BOTH_ANALYZERS)
    def test_tz_aware_timestamps(self, cls):
        df = pd.DataFrame({
            "entity_id": [1] * 30,
            "event_date": pd.date_range("2024-01-01", periods=30, tz="UTC"),
            "value": list(range(30)),
        })
        analyzer = cls(time_column="event_date", entity_column="entity_id")
        result = analyzer.calculate_velocity(df, ["value"], window_days=7)
        assert "value" in result
