import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.temporal_feature_analyzer import (
    CohortVelocityResult,
    FeatureRecommendation,
    LagCorrelationResult,
    MomentumResult,
    PredictivePowerResult,
    TemporalFeatureAnalyzer,
    VelocityResult,
)


@pytest.fixture
def sample_event_data():
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
    return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")


class TestVelocityCalculation:
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

    def test_handles_entity_level_data_without_time_column(self, analyzer):
        np.random.seed(42)
        # Entity-level data without time column
        entity_data = pd.DataFrame({
            "entity_id": range(100),
            "metric_value": [100 + (i % 2) * 40 + np.random.normal(0, 5) for i in range(100)],
            "target": [i % 2 for i in range(100)]
        })

        result = analyzer.calculate_predictive_power(
            entity_data, value_columns=["metric_value"], target_column="target"
        )

        assert "metric_value" in result
        assert result["metric_value"].information_value > 0.02


class TestCohortComparison:

    @pytest.fixture
    def entity_level_data(self):
        np.random.seed(42)
        n_entities = 100

        data = []
        for entity_id in range(n_entities):
            is_retained = entity_id % 2 == 0
            target = 1 if is_retained else 0

            # Single aggregated row per entity
            if is_retained:
                value = 120 + np.random.normal(0, 10)  # Higher value for retained
            else:
                value = 80 + np.random.normal(0, 10)  # Lower value for churned

            data.append({
                "entity_id": entity_id,
                "event_date": pd.Timestamp("2024-01-31"),
                "metric_value": max(0, value),
                "target": target
            })

        return pd.DataFrame(data)

    def test_rejects_event_level_data(self, analyzer, sample_event_data):
        with pytest.raises(ValueError, match="event-level data"):
            analyzer.compare_cohorts(
                sample_event_data,
                value_columns=["metric_value"],
                target_column="target"
            )

    def test_compares_cohorts_on_entity_level_data(self, analyzer, entity_level_data):
        result = analyzer.compare_cohorts(
            entity_level_data,
            value_columns=["metric_value"],
            target_column="target"
        )

        assert "metric_value" in result
        assert "retained" in result["metric_value"]
        assert "churned" in result["metric_value"]

    def test_detects_divergent_mean_values(self, analyzer, entity_level_data):
        result = analyzer.compare_cohorts(
            entity_level_data,
            value_columns=["metric_value"],
            target_column="target"
        )

        retained_mean = result["metric_value"]["retained"].mean_value
        churned_mean = result["metric_value"]["churned"].mean_value

        # Retained should have higher mean value than churned
        assert retained_mean > churned_mean


class TestFeatureRecommendations:

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


class TestCohortVelocitySignals:

    @pytest.fixture
    def divergent_cohort_data(self):
        np.random.seed(42)
        data = []
        base_date = pd.Timestamp("2024-01-01")
        for entity_id in range(100):
            is_retained = entity_id % 2 == 0
            target = 1 if is_retained else 0
            for day in range(60):
                value = (100 + day * 0.8 if is_retained else 100 - day * 0.6)
                value += np.random.normal(0, 3)
                data.append({
                    "entity_id": entity_id,
                    "event_date": base_date + pd.Timedelta(days=day),
                    "metric_value": max(0, value),
                    "target": target
                })
        return pd.DataFrame(data)

    def test_computes_velocity_signals_for_multiple_windows(self, analyzer, divergent_cohort_data):
        result = analyzer.compute_cohort_velocity_signals(
            divergent_cohort_data,
            value_columns=["metric_value"],
            target_column="target",
            windows=[7, 14, 30]
        )
        assert "metric_value" in result
        assert len(result["metric_value"]) == 3
        windows_in_result = [r.window_days for r in result["metric_value"]]
        assert set(windows_in_result) == {7, 14, 30}

    def test_returns_cohort_velocity_result_dataclass(self, analyzer, divergent_cohort_data):
        result = analyzer.compute_cohort_velocity_signals(
            divergent_cohort_data, ["metric_value"], "target", windows=[7]
        )
        r = result["metric_value"][0]
        assert isinstance(r, CohortVelocityResult)
        assert r.column == "metric_value"
        assert r.window_days == 7

    def test_computes_effect_sizes_between_cohorts(self, analyzer, divergent_cohort_data):
        result = analyzer.compute_cohort_velocity_signals(
            divergent_cohort_data, ["metric_value"], "target", windows=[7]
        )
        r = result["metric_value"][0]
        assert r.velocity_effect_size != 0
        assert r.velocity_effect_interp in ["Large effect", "Medium effect", "Small effect", "Negligible"]

    def test_detects_divergent_velocity_between_cohorts(self, analyzer, divergent_cohort_data):
        result = analyzer.compute_cohort_velocity_signals(
            divergent_cohort_data, ["metric_value"], "target", windows=[14]
        )
        r = result["metric_value"][0]
        assert abs(r.velocity_effect_size) >= 0.5

    def test_velocity_series_have_correct_sign(self, analyzer, divergent_cohort_data):
        result = analyzer.compute_cohort_velocity_signals(
            divergent_cohort_data, ["metric_value"], "target", windows=[7]
        )
        r = result["metric_value"][0]
        retained_mean_vel = np.mean(r.retained_velocity) if r.retained_velocity else 0
        churned_mean_vel = np.mean(r.churned_velocity) if r.churned_velocity else 0
        assert retained_mean_vel > 0
        assert churned_mean_vel < 0

    def test_handles_missing_target_column(self, analyzer, divergent_cohort_data):
        df = divergent_cohort_data.drop(columns=["target"])
        with pytest.raises(ValueError, match="target_column"):
            analyzer.compute_cohort_velocity_signals(df, ["metric_value"], "target", windows=[7])

    def test_handles_empty_cohort(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1, 1, 1],
            "event_date": pd.date_range("2024-01-01", periods=3),
            "metric_value": [1, 2, 3],
            "target": [1, 1, 1]
        })
        result = analyzer.compute_cohort_velocity_signals(df, ["metric_value"], "target", windows=[1])
        r = result["metric_value"][0]
        assert r.churned_velocity == []

    def test_computes_acceleration_effect_size(self, analyzer, divergent_cohort_data):
        result = analyzer.compute_cohort_velocity_signals(
            divergent_cohort_data, ["metric_value"], "target", windows=[7]
        )
        r = result["metric_value"][0]
        assert hasattr(r, "accel_effect_size")
        assert r.accel_effect_interp in ["Large effect", "Medium effect", "Small effect", "Negligible"]

    def test_default_windows(self, analyzer, divergent_cohort_data):
        result = analyzer.compute_cohort_velocity_signals(
            divergent_cohort_data, ["metric_value"], "target"
        )
        windows_in_result = [r.window_days for r in result["metric_value"]]
        assert set(windows_in_result) == {7, 14, 30, 90, 180, 365}

    def test_longer_windows_produce_smoother_series(self, analyzer, divergent_cohort_data):
        result = analyzer.compute_cohort_velocity_signals(
            divergent_cohort_data, ["metric_value"], "target", windows=[7, 30]
        )
        weekly = result["metric_value"][0]
        monthly = result["metric_value"][1]
        assert len(weekly.retained_velocity) > len(monthly.retained_velocity)

    def test_includes_overall_velocity(self, analyzer, divergent_cohort_data):
        result = analyzer.compute_cohort_velocity_signals(
            divergent_cohort_data, ["metric_value"], "target", windows=[7]
        )
        r = result["metric_value"][0]
        assert r.overall_velocity is not None
        assert len(r.overall_velocity) > 0

    def test_includes_period_label(self, analyzer, divergent_cohort_data):
        result = analyzer.compute_cohort_velocity_signals(
            divergent_cohort_data, ["metric_value"], "target", windows=[7, 30, 365]
        )
        labels = [r.period_label for r in result["metric_value"]]
        assert "Weekly" in labels
        assert "Monthly" in labels
        assert "Yearly" in labels


class TestVelocityRecommendations:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    @pytest.fixture
    def strong_signal_results(self):
        from customer_retention.stages.profiling.temporal_feature_analyzer import CohortVelocityResult
        return {
            "metric_a": [
                CohortVelocityResult(
                    column="metric_a", window_days=7,
                    retained_velocity=[0.5, 0.6, 0.7], churned_velocity=[-0.5, -0.6, -0.7],
                    overall_velocity=[0.0, 0.0, 0.0],
                    retained_accel=[0.1, 0.1], churned_accel=[-0.1, -0.1],
                    overall_accel=[0.0, 0.0],
                    velocity_effect_size=0.85, velocity_effect_interp="Large effect",
                    accel_effect_size=0.6, accel_effect_interp="Medium effect",
                    period_label="Weekly"
                ),
            ]
        }

    @pytest.fixture
    def weak_signal_results(self):
        from customer_retention.stages.profiling.temporal_feature_analyzer import CohortVelocityResult
        return {
            "metric_b": [
                CohortVelocityResult(
                    column="metric_b", window_days=30,
                    retained_velocity=[0.1, 0.1], churned_velocity=[0.05, 0.08],
                    overall_velocity=[0.08, 0.09],
                    retained_accel=[0.0, 0.0], churned_accel=[0.0, 0.0],
                    overall_accel=[0.0, 0.0],
                    velocity_effect_size=0.15, velocity_effect_interp="Negligible",
                    accel_effect_size=0.05, accel_effect_interp="Negligible",
                    period_label="Monthly"
                ),
            ]
        }

    def test_generates_recommendations_for_strong_signals(self, analyzer, strong_signal_results):
        recs = analyzer.generate_velocity_recommendations(strong_signal_results)
        assert len(recs) > 0
        assert any(r.action == "add_velocity_feature" for r in recs)

    def test_recommends_best_window_for_feature(self, analyzer, strong_signal_results):
        recs = analyzer.generate_velocity_recommendations(strong_signal_results)
        velocity_rec = next((r for r in recs if r.action == "add_velocity_feature"), None)
        assert velocity_rec is not None
        assert "Weekly" in velocity_rec.description or "7" in velocity_rec.description

    def test_no_recommendations_for_weak_signals(self, analyzer, weak_signal_results):
        recs = analyzer.generate_velocity_recommendations(weak_signal_results)
        velocity_recs = [r for r in recs if r.action == "add_velocity_feature"]
        assert len(velocity_recs) == 0

    def test_generates_interpretation_notes(self, analyzer, strong_signal_results):
        notes = analyzer.generate_velocity_interpretation(strong_signal_results)
        assert len(notes) > 0
        assert any("metric_a" in note for note in notes)

    def test_recommendation_has_required_fields(self, analyzer, strong_signal_results):
        recs = analyzer.generate_velocity_recommendations(strong_signal_results)
        for rec in recs:
            assert rec.source_column is not None
            assert rec.action is not None
            assert rec.description is not None
            assert rec.params is not None


class TestCohortMomentumSignals:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    @pytest.fixture
    def divergent_momentum_data(self):
        np.random.seed(42)
        data = []
        base_date = pd.Timestamp("2024-01-01")
        for entity_id in range(100):
            is_retained = entity_id % 2 == 0
            target = 1 if is_retained else 0
            for day in range(90):
                if is_retained:
                    value = 100 + (day / 90) * 50 + np.random.normal(0, 5)
                else:
                    value = 100 - (day / 90) * 30 + np.random.normal(0, 5)
                data.append({
                    "entity_id": entity_id,
                    "event_date": base_date + pd.Timedelta(days=day),
                    "metric_value": max(0, value),
                    "target": target
                })
        return pd.DataFrame(data)

    def test_computes_cohort_momentum_for_multiple_pairs(self, analyzer, divergent_momentum_data):
        result = analyzer.compute_cohort_momentum_signals(
            divergent_momentum_data, ["metric_value"], "target",
            window_pairs=[(7, 30), (30, 90)]
        )
        assert "metric_value" in result
        assert len(result["metric_value"]) == 2

    def test_returns_cohort_momentum_result(self, analyzer, divergent_momentum_data):
        from customer_retention.stages.profiling.temporal_feature_analyzer import CohortMomentumResult
        result = analyzer.compute_cohort_momentum_signals(
            divergent_momentum_data, ["metric_value"], "target", window_pairs=[(7, 30)]
        )
        r = result["metric_value"][0]
        assert isinstance(r, CohortMomentumResult)
        assert r.short_window == 7
        assert r.long_window == 30

    def test_computes_effect_size_between_cohorts(self, analyzer, divergent_momentum_data):
        result = analyzer.compute_cohort_momentum_signals(
            divergent_momentum_data, ["metric_value"], "target", window_pairs=[(7, 30)]
        )
        r = result["metric_value"][0]
        assert r.effect_size != 0
        assert r.effect_interp in ["Large effect", "Medium effect", "Small effect", "Negligible"]

    def test_generates_momentum_interpretation(self, analyzer, divergent_momentum_data):
        result = analyzer.compute_cohort_momentum_signals(
            divergent_momentum_data, ["metric_value"], "target", window_pairs=[(7, 30)]
        )
        notes = analyzer.generate_momentum_interpretation(result)
        assert len(notes) > 0
        assert any("metric_value" in note for note in notes)

    def test_momentum_recommendations_use_actual_window_params(self, analyzer, divergent_momentum_data):
        result = analyzer.compute_cohort_momentum_signals(
            divergent_momentum_data, ["metric_value"], "target", window_pairs=[(30, 90)]
        )
        recs = analyzer.generate_momentum_recommendations(result)
        for rec in recs:
            assert rec.params["short_window"] == 30
            assert rec.params["long_window"] == 90
            assert "30d/90d" in rec.description

    def test_momentum_recommendations_feature_name_reflects_windows(self, analyzer):
        from customer_retention.stages.profiling.temporal_feature_analyzer import CohortMomentumResult
        mock_results = {
            "opened": [CohortMomentumResult(
                column="opened", short_window=180, long_window=365,
                retained_momentum=1.2, churned_momentum=0.8, overall_momentum=1.0,
                effect_size=0.9, effect_interp="Large effect", window_label="180d/365d"
            )]
        }
        recs = analyzer.generate_momentum_recommendations(mock_results)
        assert len(recs) == 1
        assert recs[0].params == {"short_window": 180, "long_window": 365}
        assert "180d/365d" in recs[0].description


class TestMomentumFeatureNameDerivation:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_add_momentum_recommendations_derives_name_from_result(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1] * 60,
            "event_date": pd.date_range("2024-01-01", periods=60),
            "value": list(range(60))
        })
        recs = analyzer.get_feature_recommendations(df, value_columns=["value"])
        momentum_recs = [r for r in recs if r.feature_type.value == "momentum"]
        for rec in momentum_recs:
            assert "_momentum_" in rec.feature_name
            parts = rec.feature_name.split("_momentum_")[1]
            short_w, long_w = parts.split("_")
            assert short_w.isdigit()
            assert long_w.isdigit()
            assert f"mean_{short_w}d" in rec.formula
            assert f"mean_{long_w}d" in rec.formula

    def test_momentum_feature_name_not_hardcoded_7_30(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1] * 60,
            "event_date": pd.date_range("2024-01-01", periods=60),
            "value": list(range(60))
        })
        result = analyzer.calculate_momentum(df, ["value"], short_window=30, long_window=90)
        if "value" in result:
            r = result["value"]
            expected_name = f"value_momentum_{r.short_window}_{r.long_window}"
            assert r.short_window == 30
            assert r.long_window == 90
            assert expected_name == "value_momentum_30_90"
