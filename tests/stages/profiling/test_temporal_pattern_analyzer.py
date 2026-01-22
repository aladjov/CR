"""Tests for TemporalPatternAnalyzer - TDD approach."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from customer_retention.stages.profiling.temporal_pattern_analyzer import (
    TemporalPatternAnalyzer,
    TemporalPatternAnalysis,
    TrendResult,
    TrendDirection,
    SeasonalityPeriod,
    RecencyResult,
)


@pytest.fixture
def trending_up_data():
    """Data with clear upward trend."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    # Clear upward trend: base + day_number * slope
    values = 100 + np.arange(365) * 0.5 + np.random.normal(0, 10, 365)
    return pd.DataFrame({
        "date": dates,
        "value": values,
        "entity": ["E001"] * 365,
    })


@pytest.fixture
def trending_down_data():
    """Data with clear downward trend."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    values = 300 - np.arange(365) * 0.5 + np.random.normal(0, 10, 365)
    return pd.DataFrame({
        "date": dates,
        "value": values,
        "entity": ["E001"] * 365,
    })


@pytest.fixture
def seasonal_weekly_data():
    """Data with weekly seasonality (higher on weekends)."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    # Higher values on weekends
    base = 100
    weekend_boost = [30 if d.dayofweek >= 5 else 0 for d in dates]
    values = base + np.array(weekend_boost) + np.random.normal(0, 5, 365)
    return pd.DataFrame({
        "date": dates,
        "value": values,
        "entity": ["E001"] * 365,
    })


@pytest.fixture
def cohort_data():
    """Data with multiple customer cohorts."""
    np.random.seed(42)
    data = []

    # Cohort 1: Signed up in Jan 2023, high activity
    for i in range(100):
        signup = pd.Timestamp("2023-01-15")
        n_events = np.random.randint(5, 15)
        for j in range(n_events):
            data.append({
                "customer_id": f"C1_{i:03d}",
                "signup_date": signup,
                "event_date": signup + timedelta(days=np.random.randint(0, 180)),
                "retained": 1,
            })

    # Cohort 2: Signed up in Apr 2023, medium activity
    for i in range(100):
        signup = pd.Timestamp("2023-04-15")
        n_events = np.random.randint(2, 8)
        for j in range(n_events):
            data.append({
                "customer_id": f"C2_{i:03d}",
                "signup_date": signup,
                "event_date": signup + timedelta(days=np.random.randint(0, 120)),
                "retained": np.random.choice([0, 1], p=[0.3, 0.7]),
            })

    return pd.DataFrame(data)


@pytest.fixture
def recency_data():
    """Data for recency analysis with target."""
    np.random.seed(42)
    data = []

    reference_date = pd.Timestamp("2023-12-31")

    for i in range(200):
        # Churned customers: last activity long ago
        if i < 80:
            last_activity = reference_date - timedelta(days=np.random.randint(60, 180))
            retained = 0
        # Retained customers: recent activity
        else:
            last_activity = reference_date - timedelta(days=np.random.randint(1, 30))
            retained = 1

        data.append({
            "customer_id": f"C{i:03d}",
            "last_event_date": last_activity,
            "retained": retained,
        })

    return pd.DataFrame(data)


class TestTrendDetection:
    """Tests for trend detection."""

    def test_detects_upward_trend(self, trending_up_data):
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_trend(trending_up_data, value_column="value")

        assert isinstance(result, TrendResult)
        assert result.direction == TrendDirection.INCREASING
        assert result.strength > 0.5

    def test_detects_downward_trend(self, trending_down_data):
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_trend(trending_down_data, value_column="value")

        assert result.direction == TrendDirection.DECREASING
        assert result.strength > 0.5

    def test_stable_data_shows_stable_trend(self):
        """Random noise around constant mean should be stable."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.normal(100, 5, 100),  # Constant mean
        })

        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_trend(df, value_column="value")

        assert result.direction == TrendDirection.STABLE

    def test_trend_result_has_slope(self, trending_up_data):
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_trend(trending_up_data, value_column="value")

        assert result.slope is not None
        assert result.slope > 0  # Upward slope


class TestSeasonalityDetection:
    """Tests for seasonality detection."""

    def test_detects_weekly_pattern(self, seasonal_weekly_data):
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_seasonality(seasonal_weekly_data, value_column="value")

        assert isinstance(result, list)
        # Should find weekly pattern
        periods = [r.period for r in result]
        assert any(6 <= p <= 8 for p in periods)  # ~7 days

    def test_returns_empty_for_non_seasonal_data(self):
        """Pure random data should not show seasonality."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.normal(100, 5, 365),
        })

        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.detect_seasonality(df, value_column="value")

        # Should return empty or low-strength results
        strong_patterns = [r for r in result if r.strength > 0.5]
        assert len(strong_patterns) == 0


class TestCohortAnalysis:
    """Tests for cohort analysis."""

    def test_identifies_cohorts(self, cohort_data):
        analyzer = TemporalPatternAnalyzer(time_column="event_date")
        result = analyzer.analyze_cohorts(
            cohort_data,
            entity_column="customer_id",
            cohort_column="signup_date",
            period="M"
        )

        assert isinstance(result, pd.DataFrame)
        assert "cohort" in result.columns
        assert "entity_count" in result.columns

    def test_cohort_metrics(self, cohort_data):
        analyzer = TemporalPatternAnalyzer(time_column="event_date")
        result = analyzer.analyze_cohorts(
            cohort_data,
            entity_column="customer_id",
            cohort_column="signup_date",
            target_column="retained",
            period="M"
        )

        # Should have retention metrics
        assert "retention_rate" in result.columns or "avg_target" in result.columns


class TestRecencyAnalysis:
    """Tests for recency analysis."""

    def test_computes_recency_stats(self, recency_data):
        analyzer = TemporalPatternAnalyzer(time_column="last_event_date")
        result = analyzer.analyze_recency(
            recency_data,
            entity_column="customer_id",
            reference_date=pd.Timestamp("2023-12-31")
        )

        assert isinstance(result, RecencyResult)
        assert result.avg_recency_days is not None
        assert result.median_recency_days is not None

    def test_recency_target_correlation(self, recency_data):
        analyzer = TemporalPatternAnalyzer(time_column="last_event_date")
        result = analyzer.analyze_recency(
            recency_data,
            entity_column="customer_id",
            target_column="retained",
            reference_date=pd.Timestamp("2023-12-31")
        )

        # Should show negative correlation (recent = more likely retained)
        assert result.target_correlation is not None
        assert result.target_correlation < 0


class TestFullAnalysis:
    """Tests for complete temporal pattern analysis."""

    def test_analyze_returns_complete_result(self, trending_up_data):
        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.analyze(
            trending_up_data,
            value_column="value"
        )

        assert isinstance(result, TemporalPatternAnalysis)
        assert result.trend is not None

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame(columns=["date", "value"])

        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.analyze(df, value_column="value")

        assert result.trend is None or result.trend.direction == TrendDirection.UNKNOWN


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_row_data(self):
        df = pd.DataFrame({
            "date": [pd.Timestamp("2023-01-01")],
            "value": [100],
        })

        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.analyze(df, value_column="value")

        # Should not crash, return unknown/empty results
        assert result is not None

    def test_handles_null_values(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": [100, None, 102, None, 104, 105, None, 107, 108, 109],
        })

        analyzer = TemporalPatternAnalyzer(time_column="date")
        result = analyzer.analyze(df, value_column="value")

        # Should handle nulls gracefully
        assert result is not None
