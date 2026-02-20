from __future__ import annotations

from customer_retention.analysis.auto_explorer.intent_defaults import (
    IntentDefaultsEngine,
    IntentSuggestion,
    format_dataset_preview,
)
from customer_retention.analysis.auto_explorer.project_context import (
    CadenceInterval,
    IntentConfig,
    PredictionObjective,
    SplitStrategy,
    TemporalPosture,
)


def _engine():
    return IntentDefaultsEngine()


class TestImmediateRisk:
    def test_reactive(self):
        s = _engine().suggest(PredictionObjective.IMMEDIATE_RISK, TemporalPosture.REACTIVE)
        assert s.config.recent_window_days == 4 * 90
        assert s.config.purge_gap_days == 90 + 14

    def test_stable(self):
        s = _engine().suggest(PredictionObjective.IMMEDIATE_RISK, TemporalPosture.STABLE)
        assert s.config.recent_window_days == max(180, 3 * 90)
        assert s.config.purge_gap_days == 90 + 14


class TestDisengagement:
    def test_reactive(self):
        s = _engine().suggest(PredictionObjective.DISENGAGEMENT, TemporalPosture.REACTIVE)
        assert s.config.recent_window_days == 2 * 90
        assert s.config.purge_gap_days == 90 + 21

    def test_stable_with_span(self):
        s = _engine().suggest(
            PredictionObjective.DISENGAGEMENT,
            TemporalPosture.STABLE,
            data_span_days=1000,
        )
        expected = max(180, min(int(0.2 * 1000), 365))
        assert s.config.recent_window_days == expected
        assert s.config.purge_gap_days == 90 + 21

    def test_stable_no_span(self):
        s = _engine().suggest(PredictionObjective.DISENGAGEMENT, TemporalPosture.STABLE)
        assert s.config.recent_window_days == 270
        assert s.config.purge_gap_days == 90 + 21


class TestRenewalRisk:
    def test_reactive_with_cycle(self):
        s = _engine().suggest(
            PredictionObjective.RENEWAL_RISK,
            TemporalPosture.REACTIVE,
            renewal_cycle_days=365,
        )
        assert s.config.recent_window_days == 365
        assert s.config.label_window_days == 365
        assert s.config.purge_gap_days == 365 + 21

    def test_reactive_no_cycle(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.REACTIVE)
        assert s.config.recent_window_days == 365
        assert s.config.label_window_days == 180
        assert s.config.purge_gap_days == 180 + 21

    def test_stable_with_cycle(self):
        s = _engine().suggest(
            PredictionObjective.RENEWAL_RISK,
            TemporalPosture.STABLE,
            renewal_cycle_days=365,
        )
        expected = max(365, min(2 * 365, 730))
        assert s.config.recent_window_days == expected
        assert s.config.label_window_days == 365
        assert s.config.purge_gap_days == 365 + 21

    def test_stable_no_cycle(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.STABLE)
        assert s.config.recent_window_days == 548
        assert s.config.label_window_days == 180
        assert s.config.purge_gap_days == 180 + 21


class TestReturnType:
    def test_returns_intent_suggestion(self):
        s = _engine().suggest(PredictionObjective.IMMEDIATE_RISK, TemporalPosture.STABLE)
        assert isinstance(s, IntentSuggestion)
        assert isinstance(s.config, IntentConfig)
        assert isinstance(s.formula_explanations, dict)

    def test_formula_explanations_present(self):
        s = _engine().suggest(PredictionObjective.IMMEDIATE_RISK, TemporalPosture.STABLE)
        for field in [
            "prediction_horizons",
            "recent_window_days",
            "observation_window_days",
            "purge_gap_days",
            "label_window_days",
        ]:
            assert field in s.formula_explanations
            assert len(s.formula_explanations[field]) > 0


class TestHorizon:
    def test_default_horizon_is_90(self):
        s = _engine().suggest(PredictionObjective.IMMEDIATE_RISK, TemporalPosture.STABLE)
        assert s.config.label_window_days == 90

    def test_custom_horizon_propagates(self):
        s = _engine().suggest(
            PredictionObjective.IMMEDIATE_RISK,
            TemporalPosture.STABLE,
            prediction_horizon=30,
        )
        assert s.config.label_window_days == 30
        assert s.config.purge_gap_days == 30 + 14

    def test_prediction_horizons_flanking(self):
        s = _engine().suggest(
            PredictionObjective.IMMEDIATE_RISK,
            TemporalPosture.STABLE,
            prediction_horizon=90,
        )
        assert s.config.prediction_horizons == [30, 60, 90]

    def test_prediction_horizons_short(self):
        s = _engine().suggest(
            PredictionObjective.IMMEDIATE_RISK,
            TemporalPosture.STABLE,
            prediction_horizon=30,
        )
        assert s.config.prediction_horizons == [15, 30]


class TestConsistency:
    def test_temporal_split_always_true(self):
        for obj in PredictionObjective:
            for posture in [TemporalPosture.STABLE, TemporalPosture.REACTIVE]:
                s = _engine().suggest(obj, posture)
                assert s.config.temporal_split is True

    def test_observation_window_equals_recent(self):
        for obj in PredictionObjective:
            for posture in [TemporalPosture.STABLE, TemporalPosture.REACTIVE]:
                s = _engine().suggest(obj, posture)
                assert s.config.observation_window_days == s.config.recent_window_days


class TestCadenceImmediateRisk:
    def test_reactive_short_horizon(self):
        s = _engine().suggest(PredictionObjective.IMMEDIATE_RISK, TemporalPosture.REACTIVE, prediction_horizon=30)
        assert s.config.cadence_interval == CadenceInterval.DAILY
        assert s.config.split_strategy == SplitStrategy.TEMPORAL

    def test_reactive_medium_horizon(self):
        s = _engine().suggest(PredictionObjective.IMMEDIATE_RISK, TemporalPosture.REACTIVE, prediction_horizon=60)
        assert s.config.cadence_interval == CadenceInterval.WEEKLY
        assert s.config.split_strategy == SplitStrategy.TEMPORAL

    def test_reactive_long_horizon(self):
        s = _engine().suggest(PredictionObjective.IMMEDIATE_RISK, TemporalPosture.REACTIVE, prediction_horizon=120)
        assert s.config.cadence_interval == CadenceInterval.WEEKLY
        assert s.config.split_strategy == SplitStrategy.TEMPORAL

    def test_stable_any_horizon(self):
        for h in [30, 60, 90, 120]:
            s = _engine().suggest(PredictionObjective.IMMEDIATE_RISK, TemporalPosture.STABLE, prediction_horizon=h)
            assert s.config.cadence_interval == CadenceInterval.WEEKLY
            assert s.config.split_strategy == SplitStrategy.TEMPORAL


class TestCadenceDisengagement:
    def test_reactive_medium_horizon(self):
        s = _engine().suggest(PredictionObjective.DISENGAGEMENT, TemporalPosture.REACTIVE, prediction_horizon=90)
        assert s.config.cadence_interval == CadenceInterval.WEEKLY
        assert s.config.split_strategy == SplitStrategy.TEMPORAL

    def test_reactive_long_horizon(self):
        s = _engine().suggest(PredictionObjective.DISENGAGEMENT, TemporalPosture.REACTIVE, prediction_horizon=150)
        assert s.config.cadence_interval == CadenceInterval.WEEKLY
        assert s.config.split_strategy == SplitStrategy.TEMPORAL

    def test_stable_medium_horizon(self):
        s = _engine().suggest(PredictionObjective.DISENGAGEMENT, TemporalPosture.STABLE, prediction_horizon=120)
        assert s.config.cadence_interval == CadenceInterval.MONTHLY
        assert s.config.split_strategy == SplitStrategy.TEMPORAL

    def test_stable_long_horizon(self):
        s = _engine().suggest(PredictionObjective.DISENGAGEMENT, TemporalPosture.STABLE, prediction_horizon=200)
        assert s.config.cadence_interval == CadenceInterval.MONTHLY
        assert s.config.split_strategy == SplitStrategy.TEMPORAL


class TestCadenceRenewalReactive:
    def test_monthly_cycle(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.REACTIVE, renewal_cycle_days=30)
        assert s.config.cadence_interval == CadenceInterval.WEEKLY
        assert s.config.split_strategy == SplitStrategy.COHORT_BASED

    def test_quarterly_cycle(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.REACTIVE, renewal_cycle_days=90)
        assert s.config.cadence_interval == CadenceInterval.BIWEEKLY
        assert s.config.split_strategy == SplitStrategy.COHORT_BASED

    def test_yearly_cycle(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.REACTIVE, renewal_cycle_days=365)
        assert s.config.cadence_interval == CadenceInterval.MONTHLY
        assert s.config.split_strategy == SplitStrategy.COHORT_BASED

    def test_mixed_no_cycle(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.REACTIVE)
        assert s.config.cadence_interval == CadenceInterval.WEEKLY
        assert s.config.split_strategy == SplitStrategy.COHORT_BASED


class TestCadenceRenewalStable:
    def test_monthly_cycle(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.STABLE, renewal_cycle_days=30)
        assert s.config.cadence_interval == CadenceInterval.WEEKLY
        assert s.config.split_strategy == SplitStrategy.COHORT_BASED

    def test_quarterly_cycle(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.STABLE, renewal_cycle_days=90)
        assert s.config.cadence_interval == CadenceInterval.MONTHLY
        assert s.config.split_strategy == SplitStrategy.COHORT_BASED

    def test_yearly_cycle(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.STABLE, renewal_cycle_days=365)
        assert s.config.cadence_interval == CadenceInterval.MONTHLY
        assert s.config.split_strategy == SplitStrategy.COHORT_BASED

    def test_mixed_no_cycle(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.STABLE)
        assert s.config.cadence_interval == CadenceInterval.MONTHLY
        assert s.config.split_strategy == SplitStrategy.COHORT_BASED


class TestCadenceFormulaExplanations:
    def test_cadence_explanations_present(self):
        s = _engine().suggest(PredictionObjective.IMMEDIATE_RISK, TemporalPosture.STABLE)
        assert "cadence_interval" in s.formula_explanations
        assert "split_strategy" in s.formula_explanations
        assert len(s.formula_explanations["cadence_interval"]) > 0
        assert len(s.formula_explanations["split_strategy"]) > 0

    def test_renewal_cadence_explanation_mentions_cycle(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.REACTIVE, renewal_cycle_days=90)
        assert "quarterly" in s.formula_explanations["cadence_interval"].lower()


class TestCadenceBoundaryValues:
    def test_immediate_reactive_at_30(self):
        s = _engine().suggest(PredictionObjective.IMMEDIATE_RISK, TemporalPosture.REACTIVE, prediction_horizon=30)
        assert s.config.cadence_interval == CadenceInterval.DAILY

    def test_immediate_reactive_at_31(self):
        s = _engine().suggest(PredictionObjective.IMMEDIATE_RISK, TemporalPosture.REACTIVE, prediction_horizon=31)
        assert s.config.cadence_interval == CadenceInterval.WEEKLY

    def test_renewal_cycle_boundary_31(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.REACTIVE, renewal_cycle_days=31)
        assert s.config.cadence_interval == CadenceInterval.WEEKLY

    def test_renewal_cycle_boundary_32(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.REACTIVE, renewal_cycle_days=32)
        assert s.config.cadence_interval == CadenceInterval.BIWEEKLY

    def test_renewal_cycle_boundary_92(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.REACTIVE, renewal_cycle_days=92)
        assert s.config.cadence_interval == CadenceInterval.BIWEEKLY

    def test_renewal_cycle_boundary_93(self):
        s = _engine().suggest(PredictionObjective.RENEWAL_RISK, TemporalPosture.REACTIVE, renewal_cycle_days=93)
        assert s.config.cadence_interval == CadenceInterval.MONTHLY


class TestFormatDatasetPreview:
    def test_weekly_cadence(self):
        result = format_dataset_preview(364, CadenceInterval.WEEKLY, 1200, 4.2)
        assert "52 weekly periods" in result
        assert "1,200 unique entities" in result
        assert "4.2 events" in result
        assert "5,040 rows" in result

    def test_monthly_cadence(self):
        result = format_dataset_preview(365, CadenceInterval.MONTHLY, 500, 10.0)
        assert "12 monthly periods" in result
        assert "500 unique entities" in result
        assert "5,000 rows" in result

    def test_daily_cadence(self):
        result = format_dataset_preview(30, CadenceInterval.DAILY, 100, 1.0)
        assert "30 daily periods" in result
        assert "100 unique entities" in result

    def test_zero_avg_rows(self):
        result = format_dataset_preview(365, CadenceInterval.WEEKLY, 1000, 0.0)
        assert "0 rows" in result

    def test_returns_bold_prefix(self):
        result = format_dataset_preview(365, CadenceInterval.WEEKLY, 100, 1.0)
        assert result.startswith("**Dataset preview:**")
