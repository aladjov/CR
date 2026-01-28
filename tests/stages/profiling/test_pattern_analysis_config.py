"""Tests for pattern analysis configuration."""
import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.pattern_analysis_config import (
    PatternAnalysisConfig,
    PatternAnalysisResult,
    SparklineData,
    SparklineDataBuilder,
    get_analysis_frequency,
    get_sparkline_frequency,
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
