"""Tests for pattern analysis configuration."""
import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.pattern_analysis_config import (
    AggregationFeatureConfig,
    FindingsValidationResult,
    PatternAnalysisConfig,
    PatternAnalysisResult,
    SparklineData,
    SparklineDataBuilder,
    _extract_text_pca_columns,
    create_momentum_ratio_features,
    create_recency_bucket_feature,
    deduplicate_events,
    get_analysis_frequency,
    get_duplicate_event_count,
    get_sparkline_frequency,
    select_columns_by_variance,
    validate_not_event_level,
    validate_temporal_findings,
)


class TestPatternAnalysisConfig:
    def test_default_init(self):
        config = PatternAnalysisConfig(entity_column="customer_id", time_column="date")
        assert config.entity_column == "customer_id"
        assert config.time_column == "date"
        assert config.target_column is None
        assert config.velocity_window_days == 7
        assert config.short_momentum_window == 7
        assert config.long_momentum_window == 30

    def test_with_target(self):
        config = PatternAnalysisConfig(
            entity_column="customer_id",
            time_column="date",
            target_column="churned",
        )
        assert config.target_column == "churned"

    def test_derive_window_settings_from_aggregation_windows(self):
        config = PatternAnalysisConfig(
            entity_column="customer_id",
            time_column="date",
            aggregation_windows=["7d", "30d", "90d"],
        )
        config._derive_window_settings()
        assert config.velocity_window_days == 7
        assert config.short_momentum_window == 7
        assert config.long_momentum_window == 30

    def test_derive_window_settings_single_window(self):
        config = PatternAnalysisConfig(
            entity_column="customer_id",
            time_column="date",
            aggregation_windows=["14d"],
        )
        config._derive_window_settings()
        assert config.velocity_window_days == 14
        assert config.long_momentum_window == 14 * 4

    def test_derive_window_settings_empty(self):
        config = PatternAnalysisConfig(
            entity_column="customer_id",
            time_column="date",
            aggregation_windows=[],
        )
        config._derive_window_settings()
        assert config.velocity_window_days == 7

    def test_parse_window_to_days_days(self):
        config = PatternAnalysisConfig(entity_column="e", time_column="t")
        assert config._parse_window_to_days("7d") == 7
        assert config._parse_window_to_days("30d") == 30

    def test_parse_window_to_days_weeks(self):
        config = PatternAnalysisConfig(entity_column="e", time_column="t")
        assert config._parse_window_to_days("2w") == 14
        assert config._parse_window_to_days("4W") == 28

    def test_parse_window_to_days_months(self):
        config = PatternAnalysisConfig(entity_column="e", time_column="t")
        assert config._parse_window_to_days("1m") == 30
        assert config._parse_window_to_days("3M") == 90

    def test_parse_window_to_days_plain_number(self):
        config = PatternAnalysisConfig(entity_column="e", time_column="t")
        assert config._parse_window_to_days("14") == 14

    def test_parse_window_to_days_invalid(self):
        config = PatternAnalysisConfig(entity_column="e", time_column="t")
        assert config._parse_window_to_days("") is None
        assert config._parse_window_to_days("invalid") is None
        assert config._parse_window_to_days("xd") is None
        assert config._parse_window_to_days("xw") is None
        assert config._parse_window_to_days("xm") is None

    def test_get_momentum_pairs_multiple_windows(self):
        config = PatternAnalysisConfig(
            entity_column="e",
            time_column="t",
            aggregation_windows=["7d", "30d", "90d"],
        )
        pairs = config.get_momentum_pairs()
        assert (7, 30) in pairs
        assert (30, 90) in pairs

    def test_get_momentum_pairs_single_window(self):
        config = PatternAnalysisConfig(
            entity_column="e",
            time_column="t",
            aggregation_windows=["7d"],
            short_momentum_window=7,
            long_momentum_window=30,
        )
        pairs = config.get_momentum_pairs()
        assert pairs == [(7, 30)]

    def test_configure_sparklines_explicit_columns(self):
        config = PatternAnalysisConfig(entity_column="customer_id", time_column="date")
        df = pd.DataFrame({"customer_id": [1], "date": ["2023-01-01"], "value": [100]})
        config.configure_sparklines(df, columns=["value"])
        assert config.sparkline_columns == ["value"]

    def test_configure_sparklines_auto_detect(self):
        config = PatternAnalysisConfig(
            entity_column="customer_id",
            time_column="date",
            target_column="target",
        )
        df = pd.DataFrame({
            "customer_id": [1, 2],
            "date": ["2023-01-01", "2023-01-02"],
            "target": [0, 1],
            "amount": [100.0, 200.0],
            "count": [1, 2],
        })
        config.configure_sparklines(df, max_columns=2)
        assert "amount" in config.sparkline_columns or "count" in config.sparkline_columns
        assert "customer_id" not in config.sparkline_columns
        assert "target" not in config.sparkline_columns

    def test_print_config(self, capsys):
        config = PatternAnalysisConfig(
            entity_column="customer_id",
            time_column="date",
            aggregation_windows=["7d", "30d"],
        )
        config.print_config()
        captured = capsys.readouterr()
        assert "PATTERN ANALYSIS CONFIGURATION" in captured.out
        assert "customer_id" in captured.out

    def test_print_config_with_sparklines(self, capsys):
        config = PatternAnalysisConfig(
            entity_column="customer_id",
            time_column="date",
            sparkline_columns=["amount"],
        )
        config.print_config()
        captured = capsys.readouterr()
        assert "Sparkline Config" in captured.out


class TestPatternAnalysisResult:
    def test_default_values(self):
        result = PatternAnalysisResult()
        assert not result.trend_detected
        assert result.trend_strength == 0.0
        assert not result.seasonality_detected

    def test_with_patterns(self):
        result = PatternAnalysisResult(
            trend_detected=True,
            trend_direction="increasing",
            trend_strength=0.85,
            seasonality_detected=True,
            seasonality_periods=["weekly"],
        )
        assert result.trend_detected
        assert result.trend_direction == "increasing"

    def test_print_summary_with_patterns(self, capsys):
        result = PatternAnalysisResult(
            trend_detected=True,
            trend_direction="increasing",
            trend_strength=0.8,
            recency_effect=True,
            recency_correlation=0.5,
            velocity_features_recommended=["velocity_7d"],
        )
        result.print_summary()
        captured = capsys.readouterr()
        assert "PATTERN ANALYSIS SUMMARY" in captured.out
        assert "Trend" in captured.out

    def test_print_summary_no_patterns(self, capsys):
        result = PatternAnalysisResult()
        result.print_summary()
        captured = capsys.readouterr()
        assert "No significant patterns detected" in captured.out


class TestSparklineFunctions:
    def test_get_sparkline_frequency_short_span(self):
        assert get_sparkline_frequency(30) == "D"
        assert get_sparkline_frequency(60) == "D"

    def test_get_sparkline_frequency_medium_span(self):
        assert get_sparkline_frequency(90) == "W"
        assert get_sparkline_frequency(365) == "W"

    def test_get_sparkline_frequency_long_span(self):
        assert get_sparkline_frequency(400) == "ME"
        assert get_sparkline_frequency(730) == "ME"

    def test_get_analysis_frequency_short(self):
        freq, label = get_analysis_frequency(60)
        assert freq == "D"
        assert label == "Daily"

    def test_get_analysis_frequency_medium(self):
        freq, label = get_analysis_frequency(200)
        assert freq == "W"
        assert label == "Weekly"

    def test_get_analysis_frequency_long(self):
        freq, label = get_analysis_frequency(400)
        assert freq == "ME"
        assert label == "Monthly"


class TestSparklineData:
    def test_divergence_score_no_target(self):
        data = SparklineData(
            column="amount",
            weeks=[1, 2, 3],
            retained_values=[100, 110, 120],
            has_target_split=False,
        )
        assert data.divergence_score == 0.0

    def test_divergence_score_with_target(self):
        data = SparklineData(
            column="amount",
            weeks=[1, 2, 3],
            retained_values=[100.0, 110.0, 120.0],
            churned_values=[50.0, 55.0, 60.0],
            has_target_split=True,
        )
        assert data.divergence_score > 0

    def test_divergence_score_empty_values(self):
        data = SparklineData(
            column="amount",
            weeks=[],
            retained_values=[],
            churned_values=[],
            has_target_split=True,
        )
        assert data.divergence_score == 0.0

    def test_divergence_score_with_nans(self):
        data = SparklineData(
            column="amount",
            weeks=[1, 2, 3],
            retained_values=[100.0, np.nan, 120.0],
            churned_values=[50.0, np.nan, 60.0],
            has_target_split=True,
        )
        score = data.divergence_score
        assert score >= 0


class TestSparklineDataBuilder:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "customer_id": ["A", "A", "A", "B", "B", "B"],
            "date": pd.to_datetime([
                "2023-01-01", "2023-01-08", "2023-01-15",
                "2023-01-01", "2023-01-08", "2023-01-15",
            ]),
            "amount": [100, 110, 120, 50, 55, 60],
            "target": [1, 1, 1, 0, 0, 0],
        })

    def test_build_without_target(self, sample_df):
        builder = SparklineDataBuilder(
            entity_column="customer_id",
            time_column="date",
            freq="W",
        )
        results, has_target = builder.build(sample_df, columns=["amount"])
        assert not has_target
        assert len(results) == 1
        assert results[0].column == "amount"

    def test_build_with_target_rejects_event_level_data(self, sample_df):
        """SparklineDataBuilder rejects target on event-level data.

        Target comparisons are only valid at entity level. Event-level data
        (multiple rows per entity) should use target_column=None.
        """
        builder = SparklineDataBuilder(
            entity_column="customer_id",
            time_column="date",
            target_column="target",
            freq="W",
        )
        with pytest.raises(ValueError, match="event-level data"):
            builder.build(sample_df, columns=["amount"])

    def test_build_with_target_on_entity_level_data(self):
        """SparklineDataBuilder works with target on entity-level data."""
        # Entity-level data: one row per entity
        entity_df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "amount": [100, 50],
            "target": [1, 0],
        })
        builder = SparklineDataBuilder(
            entity_column="customer_id",
            time_column="date",
            target_column="target",
            freq="W",
        )
        results, has_target = builder.build(entity_df, columns=["amount"])
        assert has_target
        assert len(results) == 1
        assert results[0].has_target_split

    def test_build_skips_missing_columns(self, sample_df):
        builder = SparklineDataBuilder(
            entity_column="customer_id",
            time_column="date",
            freq="W",
        )
        results, _ = builder.build(sample_df, columns=["nonexistent"])
        assert len(results) == 0

    def test_print_summary_with_target(self, capsys):
        """Test print_summary with pre-built target-split data."""
        # Manually create SparklineData with target split
        # (simulating entity-level data that would pass validation)
        sparkline_data = [SparklineData(
            column="amount",
            weeks=[pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-08")],
            retained_values=[100.0, 110.0],
            churned_values=[50.0, 55.0],
            has_target_split=True,
        )]
        builder = SparklineDataBuilder(
            entity_column="customer_id",
            time_column="date",
            target_column="target",
            freq="W",
        )
        builder.print_summary(sparkline_data, has_target=True)
        captured = capsys.readouterr()
        assert "Retained vs Churned" in captured.out

    def test_print_summary_without_target(self, sample_df, capsys):
        builder = SparklineDataBuilder(
            entity_column="customer_id",
            time_column="date",
            freq="W",
        )
        results, has_target = builder.build(sample_df, columns=["amount"])
        builder.print_summary(results, has_target)
        captured = capsys.readouterr()
        assert "Overall Patterns" in captured.out


class TestPatternAnalysisConfigFromFindings:

    def _make_ts_meta(self, entity_column="cust_id", time_column="date",
                      suggested_aggregations=None):
        class TSMeta:
            pass
        ts = TSMeta()
        ts.entity_column = entity_column
        ts.time_column = time_column
        ts.suggested_aggregations = suggested_aggregations
        return ts

    def _make_findings(self, ts_meta=None, target_column=None):
        class Findings:
            pass
        f = Findings()
        f.time_series_metadata = ts_meta
        f.target_column = target_column
        return f

    def test_from_findings_basic(self):
        ts_meta = self._make_ts_meta(suggested_aggregations=["7d", "30d"])
        findings = self._make_findings(ts_meta=ts_meta, target_column="churn")
        config = PatternAnalysisConfig.from_findings(findings)
        assert config.entity_column == "cust_id"
        assert config.time_column == "date"
        assert config.target_column == "churn"
        assert config.has_target is True
        assert config.aggregation_windows == ["7d", "30d"]

    def test_from_findings_no_ts_metadata_raises(self):
        findings = self._make_findings(ts_meta=None)
        with pytest.raises(ValueError, match="time series metadata"):
            PatternAnalysisConfig.from_findings(findings)

    def test_from_findings_uses_window_override(self):
        ts_meta = self._make_ts_meta(suggested_aggregations=["7d"])
        findings = self._make_findings(ts_meta=ts_meta)
        config = PatternAnalysisConfig.from_findings(findings, window_override=["14d", "60d"])
        assert config.aggregation_windows == ["14d", "60d"]

    def test_from_findings_uses_default_windows(self):
        ts_meta = self._make_ts_meta(suggested_aggregations=None)
        findings = self._make_findings(ts_meta=ts_meta)
        config = PatternAnalysisConfig.from_findings(findings)
        assert config.aggregation_windows == ["7d", "30d", "90d"]

    def test_from_findings_target_override(self):
        ts_meta = self._make_ts_meta()
        findings = self._make_findings(ts_meta=ts_meta, target_column="churn")
        config = PatternAnalysisConfig.from_findings(findings, target_column="retained")
        assert config.target_column == "retained"

    def test_from_findings_no_target(self):
        ts_meta = self._make_ts_meta()
        findings = self._make_findings(ts_meta=ts_meta, target_column=None)
        config = PatternAnalysisConfig.from_findings(findings)
        assert config.target_column is None
        assert config.has_target is False


class TestDeriveWindowSettingsEdge:

    def test_unparseable_windows(self):
        config = PatternAnalysisConfig(
            entity_column="e", time_column="t",
            aggregation_windows=["abc", "xyz"],
        )
        config._derive_window_settings()
        # All unparseable -> empty list -> returns early
        assert config.velocity_window_days == 7  # unchanged default


class TestGetMomentumPairsEdge:

    def test_empty_aggregation_windows(self):
        config = PatternAnalysisConfig(
            entity_column="e", time_column="t",
            aggregation_windows=[],
            short_momentum_window=10,
            long_momentum_window=40,
        )
        pairs = config.get_momentum_pairs()
        assert pairs == [(10, 40)]

    def test_all_unparseable_windows(self):
        config = PatternAnalysisConfig(
            entity_column="e", time_column="t",
            aggregation_windows=["abc", "xyz"],
            short_momentum_window=7,
            long_momentum_window=30,
        )
        pairs = config.get_momentum_pairs()
        assert pairs == [(7, 30)]


class TestPatternAnalysisResultSummary:

    def test_summary_with_seasonality(self):
        result = PatternAnalysisResult(
            seasonality_detected=True,
            seasonality_periods=["weekly", "monthly"],
        )
        summary = result.format_summary()
        assert "Seasonality" in summary
        assert "weekly" in summary

    def test_summary_with_cohort_effect(self):
        result = PatternAnalysisResult(
            cohort_effect=True,
            cohort_trend="improving",
        )
        summary = result.format_summary()
        assert "Cohort effect" in summary
        assert "improving" in summary

    def test_summary_with_momentum(self):
        result = PatternAnalysisResult(
            momentum_features_recommended=["momentum_7_30"],
        )
        summary = result.format_summary()
        assert "momentum" in summary.lower()


class TestSelectColumnsByVariance:

    def test_basic_selection(self):
        df = pd.DataFrame({
            "high_var": [1, 100, 200, 300, 400],
            "low_var": [10, 10, 10, 10, 10],
            "medium_var": [50, 60, 70, 80, 90],
        })
        result = select_columns_by_variance(df, ["high_var", "low_var", "medium_var"], max_cols=2)
        assert len(result) <= 2
        assert "high_var" in result

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = select_columns_by_variance(df, ["a", "nonexistent"])
        assert "a" in result

    def test_empty_column_skipped(self):
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        result = select_columns_by_variance(df, ["a"])
        assert result == []

    def test_zero_std_skipped(self):
        df = pd.DataFrame({"a": [5, 5, 5, 5, 5]})
        result = select_columns_by_variance(df, ["a"])
        assert result == []

    def test_near_zero_mean_skipped(self):
        df = pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.0, 0.0]})
        result = select_columns_by_variance(df, ["a"])
        assert result == []

    def test_nan_cv_handled(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, 1.0]})
        result = select_columns_by_variance(df, ["a"])
        # Only 1 non-nan value -> std could be nan
        assert isinstance(result, list)


class TestValidateNotEventLevel:

    def test_none_target_passes(self):
        df = pd.DataFrame({"entity": [1, 1, 2], "val": [1, 2, 3]})
        validate_not_event_level(df, "entity", None)  # Should not raise

    def test_entity_level_passes(self):
        df = pd.DataFrame({"entity": [1, 2, 3], "target": [0, 1, 0]})
        validate_not_event_level(df, "entity", "target")

    def test_event_level_raises(self):
        df = pd.DataFrame({"entity": [1, 1, 2], "target": [1, 1, 0]})
        with pytest.raises(ValueError, match="event-level data"):
            validate_not_event_level(df, "entity", "target")


class TestFindingsValidationResult:

    def test_valid_result(self):
        r = FindingsValidationResult(valid=True)
        summary = r.format_summary()
        assert "MISSING" not in summary

    def test_invalid_result(self):
        r = FindingsValidationResult(
            valid=False,
            missing_sections=["time_series_metadata"],
            warnings=["No aggregation windows"],
        )
        summary = r.format_summary()
        assert "MISSING" in summary
        assert "time_series_metadata" in summary
        assert "Warning" in summary

    def test_print_summary(self, capsys):
        r = FindingsValidationResult(valid=True, warnings=["test warning"])
        r.print_summary()
        captured = capsys.readouterr()
        assert "Warning" in captured.out


class TestValidateTemporalFindings:

    def _make_findings(self, ts_meta=None, metadata=None):
        class Findings:
            pass
        f = Findings()
        f.time_series_metadata = ts_meta
        f.metadata = metadata
        return f

    def _make_ts_meta(self, suggested_aggregations=None):
        class TSMeta:
            pass
        ts = TSMeta()
        ts.suggested_aggregations = suggested_aggregations
        return ts

    def test_missing_ts_metadata(self):
        findings = self._make_findings()
        result = validate_temporal_findings(findings)
        assert not result.valid
        assert any("time_series_metadata" in m for m in result.missing_sections)

    def test_no_suggested_aggregations_warning(self):
        ts_meta = self._make_ts_meta(suggested_aggregations=[])
        findings = self._make_findings(ts_meta=ts_meta, metadata={})
        result = validate_temporal_findings(findings)
        assert any("defaults" in w for w in result.warnings)

    def test_missing_temporal_patterns(self):
        ts_meta = self._make_ts_meta(suggested_aggregations=["7d"])
        findings = self._make_findings(ts_meta=ts_meta, metadata={})
        result = validate_temporal_findings(findings)
        assert not result.valid

    def test_missing_subsections(self):
        ts_meta = self._make_ts_meta(suggested_aggregations=["7d"])
        findings = self._make_findings(
            ts_meta=ts_meta,
            metadata={"temporal_patterns": {"trend": {}}},
        )
        result = validate_temporal_findings(findings)
        assert result.valid
        # Should warn about missing recency and momentum
        assert any("recency" in w for w in result.warnings)
        assert any("momentum" in w for w in result.warnings)

    def test_complete_findings(self):
        ts_meta = self._make_ts_meta(suggested_aggregations=["7d"])
        findings = self._make_findings(
            ts_meta=ts_meta,
            metadata={"temporal_patterns": {"trend": {}, "recency": {}, "momentum": {}}},
        )
        result = validate_temporal_findings(findings)
        assert result.valid
        assert len(result.warnings) == 0

    def test_none_metadata(self):
        ts_meta = self._make_ts_meta(suggested_aggregations=["7d"])
        findings = self._make_findings(ts_meta=ts_meta, metadata=None)
        result = validate_temporal_findings(findings)
        assert not result.valid


class TestAggregationFeatureConfig:

    def _make_findings(self, metadata=None, text_processing=None):
        class Findings:
            pass
        f = Findings()
        f.metadata = metadata
        f.text_processing = text_processing
        return f

    def test_from_findings_empty(self):
        findings = self._make_findings(metadata={})
        config = AggregationFeatureConfig.from_findings(findings)
        assert config.trend_features == []
        assert config.priority_features == []

    def test_from_findings_with_features(self):
        findings = self._make_findings(metadata={
            "temporal_patterns": {
                "trend": {"recommendations": [{"features": ["trend_1"]}]},
                "velocity": {"recommendations": [{"features": ["vel_1"]}]},
                "momentum": {
                    "recommendations": [{"features": ["mom_1"]}],
                    "_divergent_columns": ["col_a"],
                },
                "lag": {"recommendations": [{"features": ["lag_1"]}]},
                "seasonality": {"recommendations": [{"features": ["season_1"]}]},
                "recency": {"recommendations": [{"features": ["rec_1"]}]},
                "categorical": {"recommendations": [{"features": ["cat_1"]}]},
                "sparkline": {
                    "recommendations": [
                        {"features": ["spark_1"]},
                        {"action": "robust_scale", "features": []},
                    ],
                },
                "cohort": {"recommendations": [
                    {"action": "add_feature", "features": ["cohort_1"]},
                ]},
                "effect_size": {"recommendations": [
                    {"action": "prioritize_feature", "feature": "important_1", "priority": "high", "features": ["es_1"]},
                ]},
                "predictive_power": {"recommendations": [
                    {"action": "include_feature", "feature": "pp_1", "priority": 1, "features": ["pp_feat"]},
                ]},
                "feature_flags": {"use_rolling": True},
            }
        })
        config = AggregationFeatureConfig.from_findings(findings)
        assert "trend_1" in config.trend_features
        assert "vel_1" in config.velocity_features
        assert "mom_1" in config.momentum_features
        assert "lag_1" in config.lag_features
        assert "season_1" in config.seasonality_features
        assert "rec_1" in config.recency_features
        assert "cat_1" in config.categorical_features
        assert "spark_1" in config.sparkline_features
        assert "cohort_1" in config.cohort_features
        assert "col_a" in config.divergent_columns
        assert "important_1" in config.priority_features
        assert "pp_1" in config.priority_features
        assert len(config.scaling_recommendations) == 1
        assert config.feature_flags == {"use_rolling": True}

    def test_from_findings_skip_cohort_action(self):
        findings = self._make_findings(metadata={
            "temporal_patterns": {
                "cohort": {"recommendations": [
                    {"action": "skip_cohort_features", "features": ["skip_me"]},
                ]},
            }
        })
        config = AggregationFeatureConfig.from_findings(findings)
        assert config.cohort_features == []

    def test_get_all_features(self):
        config = AggregationFeatureConfig(
            trend_features=["a"], velocity_features=["b"], momentum_features=["a"],
        )
        all_feats = config.get_all_features()
        assert "a" in all_feats
        assert "b" in all_feats
        # deduplication
        assert all_feats.count("a") == 1

    def test_get_priority_features(self):
        config = AggregationFeatureConfig(priority_features=["x", "y"])
        assert config.get_priority_features() == ["x", "y"]

    def test_format_summary(self):
        config = AggregationFeatureConfig(
            trend_features=["t1"],
            seasonality_features=["s1"],
            cohort_features=["c1"],
            recency_features=["r1"],
            categorical_features=["cat1"],
            velocity_features=["v1"],
            momentum_features=["m1"],
            lag_features=["l1"],
            sparkline_features=["sp1"],
            priority_features=["p1"],
            scaling_recommendations=[{"action": "scale"}],
            divergent_columns=["dc1"],
            text_pca_columns=["tpc1"],
            feature_flags={"flag1": True},
        )
        summary = config.format_summary()
        assert "AGGREGATION FEATURE CONFIG" in summary
        assert "Trend features" in summary
        assert "Seasonality" in summary
        assert "Cohort" in summary
        assert "Recency" in summary
        assert "Categorical" in summary
        assert "Velocity" in summary
        assert "Momentum" in summary
        assert "Lag" in summary
        assert "Sparkline" in summary
        assert "Priority" in summary
        assert "Scaling" in summary
        assert "Divergent" in summary
        assert "Text PCA" in summary
        assert "Feature flags" in summary

    def test_format_recommendation_summary(self):
        config = AggregationFeatureConfig(
            trend_features=["t1"],
            velocity_features=["v1"],
            feature_flags={"f": True},
            scaling_recommendations=[{"a": 1}],
        )
        summary = config.format_recommendation_summary()
        assert "RECOMMENDATION APPLICATION SUMMARY" in summary
        assert "Total" in summary
        assert "Feature flags" in summary
        assert "Scaling recs" in summary

    def test_print_summary(self, capsys):
        config = AggregationFeatureConfig()
        config.print_summary()
        captured = capsys.readouterr()
        assert "AGGREGATION FEATURE CONFIG" in captured.out

    def test_print_recommendation_summary(self, capsys):
        config = AggregationFeatureConfig()
        config.print_recommendation_summary()
        captured = capsys.readouterr()
        assert "RECOMMENDATION APPLICATION SUMMARY" in captured.out

    def test_from_findings_with_text_processing(self):
        class TextMeta:
            component_columns = ["pca_1", "pca_2"]
        findings = self._make_findings(
            metadata={},
            text_processing={"col_a": TextMeta()},
        )
        config = AggregationFeatureConfig.from_findings(findings)
        assert "pca_1" in config.text_pca_columns
        assert "pca_2" in config.text_pca_columns

    def test_none_metadata_from_findings(self):
        findings = self._make_findings(metadata=None)
        config = AggregationFeatureConfig.from_findings(findings)
        assert config.trend_features == []


class TestExtractTextPcaColumns:

    def test_no_text_processing(self):
        class F:
            text_processing = None
        assert _extract_text_pca_columns(F()) == []

    def test_with_text_processing(self):
        class Meta:
            component_columns = ["pca_1"]
        class F:
            text_processing = {"col": Meta()}
        assert _extract_text_pca_columns(F()) == ["pca_1"]

    def test_text_processing_no_component_columns(self):
        class Meta:
            pass
        class F:
            text_processing = {"col": Meta()}
        assert _extract_text_pca_columns(F()) == []


class TestGetDuplicateEventCount:

    def test_returns_count(self):
        class F:
            metadata = {"temporal_quality": {"issues": {"duplicate_events": 5}}}
        assert get_duplicate_event_count(F()) == 5

    def test_missing_metadata(self):
        class F:
            metadata = None
        assert get_duplicate_event_count(F()) == 0

    def test_no_issues(self):
        class F:
            metadata = {"temporal_quality": None}
        assert get_duplicate_event_count(F()) == 0

    def test_no_duplicate_events_key(self):
        class F:
            metadata = {"temporal_quality": {"issues": {}}}
        assert get_duplicate_event_count(F()) == 0


class TestDeduplicateEvents:

    def test_no_duplicates_needed(self):
        df = pd.DataFrame({"e": [1, 2], "t": ["a", "b"]})
        result, removed = deduplicate_events(df, "e", "t", duplicate_count=0)
        assert removed == 0
        assert len(result) == 2

    def test_negative_count(self):
        df = pd.DataFrame({"e": [1, 2], "t": ["a", "b"]})
        result, removed = deduplicate_events(df, "e", "t", duplicate_count=-1)
        assert removed == 0

    def test_with_duplicates(self):
        df = pd.DataFrame({"e": [1, 1, 2], "t": ["a", "a", "b"], "v": [10, 20, 30]})
        result, removed = deduplicate_events(df, "e", "t", duplicate_count=1)
        assert removed == 1
        assert len(result) == 2


class TestCreateRecencyBucketFeature:

    def test_basic_buckets(self):
        df = pd.DataFrame({"days_since_last_event": [3, 15, 60, 120, 200]})
        result = create_recency_bucket_feature(df)
        assert "recency_bucket" in result.columns
        assert result.iloc[0]["recency_bucket"] == "0-7d"
        assert result.iloc[1]["recency_bucket"] == "8-30d"
        assert result.iloc[2]["recency_bucket"] == "31-90d"
        assert result.iloc[3]["recency_bucket"] == "91-180d"
        assert result.iloc[4]["recency_bucket"] == ">180d"

    def test_missing_column_returns_unchanged(self):
        df = pd.DataFrame({"other": [1, 2]})
        result = create_recency_bucket_feature(df)
        assert "recency_bucket" not in result.columns

    def test_nan_values(self):
        df = pd.DataFrame({"days_since_last_event": [5, np.nan, 100]})
        result = create_recency_bucket_feature(df)
        assert pd.isna(result.iloc[1]["recency_bucket"])


class TestCreateMomentumRatioFeatures:

    def test_basic_ratio(self):
        df = pd.DataFrame({
            "val_mean_7d": [10.0, 20.0],
            "val_mean_30d": [8.0, 25.0],
        })
        recs = [{"source_column": "val", "params": {"short_window": 7, "long_window": 30}}]
        result = create_momentum_ratio_features(df, recs)
        assert "val_momentum_7_30" in result.columns
        assert result.iloc[0]["val_momentum_7_30"] == pytest.approx(10.0 / 8.0)

    def test_division_by_zero_replaced(self):
        df = pd.DataFrame({
            "val_mean_7d": [10.0],
            "val_mean_30d": [0.0],
        })
        recs = [{"source_column": "val", "params": {"short_window": 7, "long_window": 30}}]
        result = create_momentum_ratio_features(df, recs)
        assert result.iloc[0]["val_momentum_7_30"] == 1.0

    def test_missing_columns_skipped(self):
        df = pd.DataFrame({"other": [1]})
        recs = [{"source_column": "val", "params": {"short_window": 7, "long_window": 30}}]
        result = create_momentum_ratio_features(df, recs)
        assert "val_momentum_7_30" not in result.columns

    def test_incomplete_rec_skipped(self):
        df = pd.DataFrame({"val_mean_7d": [10.0], "val_mean_30d": [8.0]})
        recs = [{"source_column": "", "params": {"short_window": 7}}]
        result = create_momentum_ratio_features(df, recs)
        assert len(result.columns) == 2  # no new columns

    def test_empty_recs(self):
        df = pd.DataFrame({"a": [1]})
        result = create_momentum_ratio_features(df, [])
        assert len(result) == 1
