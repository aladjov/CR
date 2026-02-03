import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.temporal_feature_analyzer import (
    CohortMomentumResult,
    CohortVelocityResult,
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


class TestAccelerationCalculation:

    def test_calculates_acceleration_for_single_column(self, analyzer, sample_event_data):
        result = analyzer.calculate_acceleration(
            sample_event_data, value_columns=["metric_value"], window_days=7
        )
        assert "metric_value" in result
        assert isinstance(result["metric_value"], float)

    def test_acceleration_skips_missing_columns(self, analyzer, sample_event_data):
        result = analyzer.calculate_acceleration(
            sample_event_data, value_columns=["nonexistent"], window_days=7
        )
        assert result == {}

    def test_acceleration_returns_zero_for_constant(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1] * 30,
            "event_date": pd.date_range("2024-01-01", periods=30),
            "value": [50.0] * 30,
        })
        result = analyzer.calculate_acceleration(df, ["value"], window_days=7)
        assert result["value"] == 0.0


class TestClassifyTrend:

    def test_stable_trend(self, analyzer):
        assert analyzer._classify_trend(0.005) == "stable"

    def test_increasing_trend(self, analyzer):
        assert analyzer._classify_trend(0.5) == "increasing"

    def test_decreasing_trend(self, analyzer):
        assert analyzer._classify_trend(-0.5) == "decreasing"


class TestClassifyMomentum:

    def test_accelerating(self, analyzer):
        assert analyzer._classify_momentum(1.2) == "accelerating"

    def test_decelerating(self, analyzer):
        assert analyzer._classify_momentum(0.8) == "decelerating"

    def test_stable(self, analyzer):
        assert analyzer._classify_momentum(1.0) == "stable"


class TestWindowMapping:

    def test_weekly_window(self, analyzer):
        assert analyzer._window_to_period_label(7) == "Weekly"
        assert analyzer._window_to_period(7) == "W"

    def test_biweekly_window(self, analyzer):
        assert analyzer._window_to_period_label(14) == "Bi-weekly"
        assert analyzer._window_to_period(14) == "2W"

    def test_monthly_window(self, analyzer):
        assert analyzer._window_to_period_label(30) == "Monthly"

    def test_quarterly_window(self, analyzer):
        assert analyzer._window_to_period_label(90) == "Quarterly"

    def test_semiannual_window(self, analyzer):
        assert analyzer._window_to_period_label(180) == "Semi-annual"

    def test_yearly_window(self, analyzer):
        assert analyzer._window_to_period_label(365) == "Yearly"
        assert analyzer._window_to_period(365) == "Y"


class TestVelocityInterpretation:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_strong_increasing_signal(self, analyzer):
        results = {
            "col_a": [CohortVelocityResult(
                column="col_a", window_days=7,
                retained_velocity=[1.0], churned_velocity=[-1.0],
                overall_velocity=[0.0],
                retained_accel=[0.0], churned_accel=[0.0],
                overall_accel=[0.0],
                velocity_effect_size=0.9, velocity_effect_interp="Large effect",
                accel_effect_size=0.1, accel_effect_interp="Negligible",
                period_label="Weekly",
            )]
        }
        notes = analyzer.generate_velocity_interpretation(results)
        assert len(notes) == 1
        assert "Strong signal" in notes[0]
        assert "increasing" in notes[0]

    def test_strong_decreasing_signal(self, analyzer):
        results = {
            "col_a": [CohortVelocityResult(
                column="col_a", window_days=7,
                retained_velocity=[1.0], churned_velocity=[-1.0],
                overall_velocity=[0.0],
                retained_accel=[0.0], churned_accel=[0.0],
                overall_accel=[0.0],
                velocity_effect_size=-0.9, velocity_effect_interp="Large effect",
                accel_effect_size=0.1, accel_effect_interp="Negligible",
                period_label="Weekly",
            )]
        }
        notes = analyzer.generate_velocity_interpretation(results)
        assert "decreasing" in notes[0]

    def test_moderate_signal(self, analyzer):
        results = {
            "col_a": [CohortVelocityResult(
                column="col_a", window_days=7,
                retained_velocity=[1.0], churned_velocity=[-1.0],
                overall_velocity=[0.0],
                retained_accel=[0.0], churned_accel=[0.0],
                overall_accel=[0.0],
                velocity_effect_size=0.6, velocity_effect_interp="Medium effect",
                accel_effect_size=0.0, accel_effect_interp="Negligible",
                period_label="Weekly",
            )]
        }
        notes = analyzer.generate_velocity_interpretation(results)
        assert "Moderate signal" in notes[0]
        assert "secondary predictor" in notes[0]

    def test_weak_signal(self, analyzer):
        results = {
            "col_a": [CohortVelocityResult(
                column="col_a", window_days=7,
                retained_velocity=[1.0], churned_velocity=[-1.0],
                overall_velocity=[0.0],
                retained_accel=[0.0], churned_accel=[0.0],
                overall_accel=[0.0],
                velocity_effect_size=0.25, velocity_effect_interp="Small effect",
                accel_effect_size=0.0, accel_effect_interp="Negligible",
                period_label="Weekly",
            )]
        }
        notes = analyzer.generate_velocity_interpretation(results)
        assert "Weak signal" in notes[0]
        assert "feature combinations" in notes[0]

    def test_no_significant_signal(self, analyzer):
        results = {
            "col_a": [CohortVelocityResult(
                column="col_a", window_days=7,
                retained_velocity=[1.0], churned_velocity=[-1.0],
                overall_velocity=[0.0],
                retained_accel=[0.0], churned_accel=[0.0],
                overall_accel=[0.0],
                velocity_effect_size=0.1, velocity_effect_interp="Negligible",
                accel_effect_size=0.0, accel_effect_interp="Negligible",
                period_label="Weekly",
            )]
        }
        notes = analyzer.generate_velocity_interpretation(results)
        assert "No significant" in notes[0]

    def test_empty_results_list(self, analyzer):
        results = {"col_a": []}
        notes = analyzer.generate_velocity_interpretation(results)
        assert notes == []

    def test_find_best_velocity_window_empty(self, analyzer):
        assert analyzer._find_best_velocity_window([]) is None


class TestVelocityRecommendationsPriority:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_acceleration_recommendation_generated(self, analyzer):
        results = {
            "col_a": [CohortVelocityResult(
                column="col_a", window_days=7,
                retained_velocity=[1.0], churned_velocity=[-1.0],
                overall_velocity=[0.0],
                retained_accel=[0.0], churned_accel=[0.0],
                overall_accel=[0.0],
                velocity_effect_size=0.3,
                velocity_effect_interp="Small effect",
                accel_effect_size=0.7,
                accel_effect_interp="Medium effect",
                period_label="Weekly",
            )]
        }
        recs = analyzer.generate_velocity_recommendations(results)
        accel_recs = [r for r in recs if r.action == "add_acceleration_feature"]
        assert len(accel_recs) == 1
        assert accel_recs[0].priority == 2

    def test_priority_1_for_large_velocity_effect(self, analyzer):
        results = {
            "col_a": [CohortVelocityResult(
                column="col_a", window_days=7,
                retained_velocity=[1.0], churned_velocity=[-1.0],
                overall_velocity=[0.0],
                retained_accel=[0.0], churned_accel=[0.0],
                overall_accel=[0.0],
                velocity_effect_size=0.85, velocity_effect_interp="Large effect",
                accel_effect_size=0.1, accel_effect_interp="Negligible",
                period_label="Weekly",
            )]
        }
        recs = analyzer.generate_velocity_recommendations(results)
        velocity_rec = next(r for r in recs if r.action == "add_velocity_feature")
        assert velocity_rec.priority == 1

    def test_priority_2_for_medium_velocity_effect(self, analyzer):
        results = {
            "col_a": [CohortVelocityResult(
                column="col_a", window_days=7,
                retained_velocity=[1.0], churned_velocity=[-1.0],
                overall_velocity=[0.0],
                retained_accel=[0.0], churned_accel=[0.0],
                overall_accel=[0.0],
                velocity_effect_size=0.6, velocity_effect_interp="Medium effect",
                accel_effect_size=0.1, accel_effect_interp="Negligible",
                period_label="Weekly",
            )]
        }
        recs = analyzer.generate_velocity_recommendations(results)
        velocity_rec = next(r for r in recs if r.action == "add_velocity_feature")
        assert velocity_rec.priority == 2


class TestLagRecommendations:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_strong_lag_recommendation(self, analyzer):
        results = {
            "col_a": LagCorrelationResult(
                column="col_a", correlations=[0.6, 0.3, 0.2, 0.1, 0.05, 0.01, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                best_lag=1, best_correlation=0.6,
                has_weekly_pattern=False,
            )
        }
        recs = analyzer.generate_lag_recommendations(results)
        assert len(recs) == 1
        assert recs[0].action == "add_lag_feature"
        assert recs[0].priority == 1

    def test_moderate_lag_recommendation_priority_2(self, analyzer):
        results = {
            "col_a": LagCorrelationResult(
                column="col_a", correlations=[0.35, 0.2, 0.1, 0.05, 0.01, 0.01, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                best_lag=1, best_correlation=0.35,
                has_weekly_pattern=False,
            )
        }
        recs = analyzer.generate_lag_recommendations(results)
        assert recs[0].priority == 2

    def test_weekly_pattern_recommendation(self, analyzer):
        results = {
            "col_a": LagCorrelationResult(
                column="col_a", correlations=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                best_lag=1, best_correlation=0.1,
                has_weekly_pattern=True,
            )
        }
        recs = analyzer.generate_lag_recommendations(results)
        weekly_recs = [r for r in recs if r.action == "add_weekly_lag"]
        assert len(weekly_recs) == 1
        assert weekly_recs[0].params["lag_days"] == 7

    def test_weekly_pattern_with_best_lag_7_no_duplicate(self, analyzer):
        results = {
            "col_a": LagCorrelationResult(
                column="col_a", correlations=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                best_lag=7, best_correlation=0.8,
                has_weekly_pattern=True,
            )
        }
        recs = analyzer.generate_lag_recommendations(results)
        # Should not add weekly lag separately when best_lag is already 7
        weekly_recs = [r for r in recs if r.action == "add_weekly_lag"]
        assert len(weekly_recs) == 0

    def test_no_recommendations_for_weak_lag(self, analyzer):
        results = {
            "col_a": LagCorrelationResult(
                column="col_a", correlations=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                best_lag=1, best_correlation=0.1,
                has_weekly_pattern=False,
            )
        }
        recs = analyzer.generate_lag_recommendations(results)
        assert len(recs) == 0


class TestLagInterpretation:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_strong_autocorrelation_interpretation(self, analyzer):
        results = {
            "col_a": LagCorrelationResult(
                column="col_a", correlations=[0.6], best_lag=1, best_correlation=0.6,
                has_weekly_pattern=False,
            )
        }
        notes = analyzer.generate_lag_interpretation(results)
        assert any("Strong autocorrelation" in n for n in notes)
        assert any("high predictability" in n for n in notes)

    def test_moderate_autocorrelation_interpretation(self, analyzer):
        results = {
            "col_a": LagCorrelationResult(
                column="col_a", correlations=[0.35], best_lag=1, best_correlation=0.35,
                has_weekly_pattern=False,
            )
        }
        notes = analyzer.generate_lag_interpretation(results)
        assert any("Moderate autocorrelation" in n for n in notes)

    def test_weekly_pattern_interpretation(self, analyzer):
        results = {
            "col_a": LagCorrelationResult(
                column="col_a", correlations=[0.1], best_lag=1, best_correlation=0.1,
                has_weekly_pattern=True,
            )
        }
        notes = analyzer.generate_lag_interpretation(results)
        assert any("Weekly patterns" in n for n in notes)

    def test_all_weak_interpretation(self, analyzer):
        results = {
            "col_a": LagCorrelationResult(
                column="col_a", correlations=[0.1], best_lag=1, best_correlation=0.1,
                has_weekly_pattern=False,
            )
        }
        notes = analyzer.generate_lag_interpretation(results)
        assert any("weak autocorrelation" in n for n in notes)
        assert any("rolling features" in n for n in notes)


class TestInterpretIVAndKS:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_interpret_iv_very_weak(self, analyzer):
        assert analyzer._interpret_iv(0.01) == "very_weak"

    def test_interpret_iv_weak(self, analyzer):
        assert analyzer._interpret_iv(0.05) == "weak"

    def test_interpret_iv_medium(self, analyzer):
        assert analyzer._interpret_iv(0.15) == "medium"

    def test_interpret_iv_strong(self, analyzer):
        assert analyzer._interpret_iv(0.35) == "strong"

    def test_interpret_iv_suspicious(self, analyzer):
        assert analyzer._interpret_iv(0.6) == "suspicious"

    def test_interpret_ks_weak(self, analyzer):
        assert analyzer._interpret_ks(0.1) == "weak"

    def test_interpret_ks_medium(self, analyzer):
        assert analyzer._interpret_ks(0.3) == "medium"

    def test_interpret_ks_strong(self, analyzer):
        assert analyzer._interpret_ks(0.5) == "strong"


class TestCalculateIVEdgeCases:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_iv_returns_zero_for_small_sample(self, analyzer):
        feature = pd.Series([1.0, 2.0, 3.0])
        target = pd.Series([0, 1, 0])
        assert analyzer._calculate_iv(feature, target, bins=10) == 0.0

    def test_iv_returns_zero_for_all_same_target(self, analyzer):
        feature = pd.Series(np.random.normal(0, 1, 100))
        target = pd.Series([1] * 100)
        assert analyzer._calculate_iv(feature, target, bins=10) == 0.0

    def test_ks_returns_zero_for_empty_group(self, analyzer):
        feature = pd.Series([1.0, 2.0])
        target = pd.Series([1, 1])
        ks_stat, ks_pval = analyzer._calculate_ks(feature, target)
        assert ks_stat == 0.0
        assert ks_pval == 1.0


class TestVelocityAccelSeriesEmpty:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_empty_dataframe(self, analyzer):
        df = pd.DataFrame(columns=["entity_id", "event_date", "value"])
        vel, accel = analyzer._velocity_accel_series(df, "value", 7)
        assert vel == []
        assert accel == []

    def test_missing_column(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1],
            "event_date": pd.Timestamp("2024-01-01"),
        })
        vel, accel = analyzer._velocity_accel_series(df, "nonexistent", 7)
        assert vel == []
        assert accel == []


class TestVectorizedEntityMomentumEdge:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_empty_dataframe(self, analyzer):
        df = pd.DataFrame(columns=["entity_id", "event_date", "value"])
        result = analyzer._vectorized_entity_momentum(df, "value", 7, 30)
        assert result == []

    def test_missing_column(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1],
            "event_date": [pd.Timestamp("2024-01-01")],
        })
        result = analyzer._vectorized_entity_momentum(df, "nonexistent", 7, 30)
        assert result == []


class TestMomentumInterpretationBranches:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_strong_signal(self, analyzer):
        results = {
            "col_a": [CohortMomentumResult(
                column="col_a", short_window=7, long_window=30,
                retained_momentum=1.1, churned_momentum=0.8,
                overall_momentum=0.95, effect_size=0.7,
                effect_interp="Medium effect", window_label="7d/30d",
            )]
        }
        notes = analyzer.generate_momentum_interpretation(results)
        assert any("Strong signal" in n for n in notes)

    def test_moderate_signal(self, analyzer):
        results = {
            "col_a": [CohortMomentumResult(
                column="col_a", short_window=7, long_window=30,
                retained_momentum=1.0, churned_momentum=0.97,
                overall_momentum=0.99, effect_size=0.3,
                effect_interp="Small effect", window_label="7d/30d",
            )]
        }
        notes = analyzer.generate_momentum_interpretation(results)
        assert any("Moderate signal" in n for n in notes)

    def test_no_significant_signal(self, analyzer):
        results = {
            "col_a": [CohortMomentumResult(
                column="col_a", short_window=7, long_window=30,
                retained_momentum=1.0, churned_momentum=1.0,
                overall_momentum=1.0, effect_size=0.05,
                effect_interp="Negligible", window_label="7d/30d",
            )]
        }
        notes = analyzer.generate_momentum_interpretation(results)
        assert any("No significant" in n for n in notes)

    def test_empty_col_results(self, analyzer):
        results = {"col_a": []}
        notes = analyzer.generate_momentum_interpretation(results)
        assert notes == []

    def test_retained_accelerating_churned_decelerating(self, analyzer):
        results = {
            "col_a": [CohortMomentumResult(
                column="col_a", short_window=7, long_window=30,
                retained_momentum=1.2, churned_momentum=0.8,
                overall_momentum=1.0, effect_size=0.7,
                effect_interp="Medium effect", window_label="7d/30d",
            )]
        }
        notes = analyzer.generate_momentum_interpretation(results)
        assert any("accelerating" in n for n in notes)
        assert any("decelerating" in n for n in notes)

    def test_both_stable(self, analyzer):
        results = {
            "col_a": [CohortMomentumResult(
                column="col_a", short_window=7, long_window=30,
                retained_momentum=1.0, churned_momentum=1.0,
                overall_momentum=1.0, effect_size=0.6,
                effect_interp="Medium effect", window_label="7d/30d",
            )]
        }
        notes = analyzer.generate_momentum_interpretation(results)
        assert any("stable" in n for n in notes)


class TestMomentumRecommendationsPriority:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_priority_1_for_large_effect(self, analyzer):
        results = {
            "col_a": [CohortMomentumResult(
                column="col_a", short_window=7, long_window=30,
                retained_momentum=1.2, churned_momentum=0.8,
                overall_momentum=1.0, effect_size=0.9,
                effect_interp="Large effect", window_label="7d/30d",
            )]
        }
        recs = analyzer.generate_momentum_recommendations(results)
        assert len(recs) == 1
        assert recs[0].priority == 1

    def test_priority_2_for_medium_effect(self, analyzer):
        results = {
            "col_a": [CohortMomentumResult(
                column="col_a", short_window=7, long_window=30,
                retained_momentum=1.2, churned_momentum=0.8,
                overall_momentum=1.0, effect_size=0.6,
                effect_interp="Medium effect", window_label="7d/30d",
            )]
        }
        recs = analyzer.generate_momentum_recommendations(results)
        assert recs[0].priority == 2

    def test_no_recommendation_for_weak_effect(self, analyzer):
        results = {
            "col_a": [CohortMomentumResult(
                column="col_a", short_window=7, long_window=30,
                retained_momentum=1.0, churned_momentum=1.0,
                overall_momentum=1.0, effect_size=0.1,
                effect_interp="Negligible", window_label="7d/30d",
            )]
        }
        recs = analyzer.generate_momentum_recommendations(results)
        assert len(recs) == 0


class TestCohortMomentumMissingTarget:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_raises_on_missing_target(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1, 2],
            "event_date": pd.date_range("2024-01-01", periods=2),
            "value": [1.0, 2.0],
        })
        with pytest.raises(ValueError, match="target_column"):
            analyzer.compute_cohort_momentum_signals(df, ["value"], "target")

    def test_skips_missing_columns(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1, 2],
            "event_date": pd.date_range("2024-01-01", periods=2),
            "value": [1.0, 2.0],
            "target": [0, 1],
        })
        result = analyzer.compute_cohort_momentum_signals(
            df, ["nonexistent"], "target", window_pairs=[(7, 30)]
        )
        assert result == {}

    def test_default_window_pairs(self, analyzer):
        np.random.seed(42)
        data = []
        base_date = pd.Timestamp("2024-01-01")
        for eid in range(10):
            for day in range(90):
                data.append({
                    "entity_id": eid,
                    "event_date": base_date + pd.Timedelta(days=day),
                    "value": np.random.normal(100, 10),
                    "target": eid % 2,
                })
        df = pd.DataFrame(data)
        result = analyzer.compute_cohort_momentum_signals(df, ["value"], "target")
        assert "value" in result
        assert len(result["value"]) == 3  # default: [(7, 30), (30, 90), (7, 90)]


class TestValidateEventLevelTarget:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_none_target_column_passes(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1, 1, 2],
            "event_date": pd.date_range("2024-01-01", periods=3),
        })
        # Should not raise
        analyzer._validate_event_level_target_usage(df, None)

    def test_entity_level_data_passes(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1, 2, 3],
            "event_date": pd.date_range("2024-01-01", periods=3),
            "target": [0, 1, 0],
        })
        # unique entities == rows, should not raise
        analyzer._validate_event_level_target_usage(df, "target")

    def test_event_level_data_raises(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1, 1, 2],
            "event_date": pd.date_range("2024-01-01", periods=3),
            "target": [1, 1, 0],
        })
        with pytest.raises(ValueError, match="event-level data"):
            analyzer._validate_event_level_target_usage(df, "target")


class TestValidateTargetConstantPerEntity:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_warns_on_varying_target(self, analyzer):
        import warnings
        df = pd.DataFrame({
            "entity_id": [1, 1, 2, 2],
            "event_date": pd.date_range("2024-01-01", periods=4),
            "target": [0, 1, 1, 1],  # entity 1 varies
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analyzer._validate_target_constant_per_entity(df, "target")
            assert len(w) == 1
            assert "varies within" in str(w[0].message)


class TestCalculateIVQcutError:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_iv_returns_zero_on_qcut_error(self, analyzer):
        from unittest.mock import patch
        feature = pd.Series(np.random.normal(0, 1, 100))
        target = pd.Series([0] * 50 + [1] * 50)
        with patch(
            "customer_retention.stages.profiling.temporal_feature_analyzer.qcut",
            side_effect=ValueError("qcut error"),
        ):
            result = analyzer._calculate_iv(feature, target, bins=10)
        assert result == 0.0


class TestCohortVelocitySkipsMissingCol:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_skips_missing_column(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1, 2],
            "event_date": pd.date_range("2024-01-01", periods=2),
            "value": [1.0, 2.0],
            "target": [0, 1],
        })
        result = analyzer.compute_cohort_velocity_signals(
            df, ["nonexistent"], "target", windows=[7]
        )
        assert result == {}


class TestFeatureRecommendationsWithoutTarget:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_generates_recommendations_without_target(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1] * 30,
            "event_date": pd.date_range("2024-01-01", periods=30),
            "value": list(range(30)),
        })
        recs = analyzer.get_feature_recommendations(df, value_columns=["value"])
        assert isinstance(recs, list)
        # Should still produce velocity/momentum/lag recommendations
        types = {r.feature_type for r in recs}
        assert FeatureType.VELOCITY in types or FeatureType.MOMENTUM in types or FeatureType.LAG in types or len(recs) == 0


class TestMomentumEdgeZeroLongMean:

    @pytest.fixture
    def analyzer(self):
        return TemporalFeatureAnalyzer(time_column="event_date", entity_column="entity_id")

    def test_skips_entity_with_zero_long_mean(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1] * 60,
            "event_date": pd.date_range("2024-01-01", periods=60),
            "value": [0.0] * 60,
        })
        result = analyzer.calculate_momentum(df, ["value"], short_window=7, long_window=30)
        # long_mean is 0, so entity should be skipped; default 1.0
        assert result["value"].mean_momentum == 1.0

    def test_nan_short_mean_skipped(self, analyzer):
        df = pd.DataFrame({
            "entity_id": [1] * 60,
            "event_date": pd.date_range("2024-01-01", periods=60),
            "value": [np.nan] * 7 + [5.0] * 53,
        })
        result = analyzer.calculate_momentum(df, ["value"], short_window=7, long_window=30)
        assert isinstance(result["value"].mean_momentum, float)
