"""Tests for TemporalTargetAnalyzer."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling import TemporalTargetAnalyzer


class TestTemporalTargetAnalyzerInit:
    def test_default_init(self):
        analyzer = TemporalTargetAnalyzer()
        assert analyzer is not None


class TestRetentionByYear:
    @pytest.fixture
    def yearly_data(self):
        """Data with different retention rates by year."""
        np.random.seed(42)
        dates_2020 = pd.date_range('2020-01-01', '2020-12-31', periods=200)
        dates_2021 = pd.date_range('2021-01-01', '2021-12-31', periods=200)
        dates_2022 = pd.date_range('2022-01-01', '2022-12-31', periods=200)

        targets_2020 = np.random.choice([0, 1], 200, p=[0.3, 0.7])  # 70% retention
        targets_2021 = np.random.choice([0, 1], 200, p=[0.2, 0.8])  # 80% retention
        targets_2022 = np.random.choice([0, 1], 200, p=[0.1, 0.9])  # 90% retention

        return pd.DataFrame({
            'signup_date': list(dates_2020) + list(dates_2021) + list(dates_2022),
            'retained': list(targets_2020) + list(targets_2021) + list(targets_2022)
        })

    def test_yearly_retention_calculated(self, yearly_data):
        analyzer = TemporalTargetAnalyzer()
        result = analyzer.analyze(yearly_data, 'signup_date', 'retained')

        assert result.yearly_stats is not None
        assert len(result.yearly_stats) == 3  # 2020, 2021, 2022

    def test_yearly_trend_detected(self, yearly_data):
        analyzer = TemporalTargetAnalyzer()
        result = analyzer.analyze(yearly_data, 'signup_date', 'retained')

        # Retention increases each year
        assert result.yearly_trend in ['improving', 'stable', 'declining']


class TestRetentionByMonth:
    @pytest.fixture
    def monthly_data(self):
        """Data with seasonal patterns."""
        np.random.seed(42)
        dates = []
        targets = []

        # Higher retention in summer months
        for month in range(1, 13):
            n = 100
            month_dates = pd.date_range(f'2022-{month:02d}-01', periods=n, freq='D')
            dates.extend(month_dates[:n])

            # Summer (6,7,8) has 90% retention, winter (12,1,2) has 60%
            if month in [6, 7, 8]:
                targets.extend(np.random.choice([0, 1], n, p=[0.1, 0.9]))
            elif month in [12, 1, 2]:
                targets.extend(np.random.choice([0, 1], n, p=[0.4, 0.6]))
            else:
                targets.extend(np.random.choice([0, 1], n, p=[0.2, 0.8]))

        return pd.DataFrame({'date': dates[:1200], 'retained': targets[:1200]})

    def test_monthly_stats_calculated(self, monthly_data):
        analyzer = TemporalTargetAnalyzer()
        result = analyzer.analyze(monthly_data, 'date', 'retained')

        assert result.monthly_stats is not None
        assert len(result.monthly_stats) == 12

    def test_best_worst_months_identified(self, monthly_data):
        analyzer = TemporalTargetAnalyzer()
        result = analyzer.analyze(monthly_data, 'date', 'retained')

        assert result.best_month is not None
        assert result.worst_month is not None
        assert result.seasonal_spread >= 0


class TestRetentionByDayOfWeek:
    @pytest.fixture
    def dow_data(self):
        """Data with day-of-week patterns."""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
        targets = []

        for date in dates:
            # Weekend signups have lower retention
            if date.dayofweek >= 5:  # Saturday, Sunday
                targets.append(np.random.choice([0, 1], p=[0.4, 0.6]))
            else:
                targets.append(np.random.choice([0, 1], p=[0.15, 0.85]))

        return pd.DataFrame({'date': dates, 'retained': targets})

    def test_dow_stats_calculated(self, dow_data):
        analyzer = TemporalTargetAnalyzer()
        result = analyzer.analyze(dow_data, 'date', 'retained')

        assert result.dow_stats is not None
        assert len(result.dow_stats) == 7

    def test_weekday_weekend_difference(self, dow_data):
        analyzer = TemporalTargetAnalyzer()
        result = analyzer.analyze(dow_data, 'date', 'retained')

        weekday_rates = result.dow_stats[result.dow_stats['day_name'].isin(
            ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        )]['retention_rate'].mean()

        weekend_rates = result.dow_stats[result.dow_stats['day_name'].isin(
            ['Sat', 'Sun']
        )]['retention_rate'].mean()

        assert weekday_rates > weekend_rates


class TestOverallMetrics:
    def test_overall_retention_calculated(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=100),
            'retained': [1] * 75 + [0] * 25
        })
        analyzer = TemporalTargetAnalyzer()
        result = analyzer.analyze(df, 'date', 'retained')

        assert abs(result.overall_rate - 0.75) < 0.01

    def test_date_range_captured(self):
        df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', '2022-12-31', periods=100),
            'retained': [1] * 50 + [0] * 50
        })
        analyzer = TemporalTargetAnalyzer()
        result = analyzer.analyze(df, 'date', 'retained')

        assert result.min_date.year == 2022
        assert result.max_date.year == 2022


class TestEdgeCases:
    def test_empty_dataframe(self):
        analyzer = TemporalTargetAnalyzer()
        df = pd.DataFrame({'date': [], 'target': []})
        result = analyzer.analyze(df, 'date', 'target')

        assert result.overall_rate == 0
        assert len(result.monthly_stats) == 0

    def test_single_date(self):
        analyzer = TemporalTargetAnalyzer()
        df = pd.DataFrame({
            'date': [datetime(2022, 6, 15)] * 100,
            'target': [1] * 80 + [0] * 20
        })
        result = analyzer.analyze(df, 'date', 'target')

        assert result.overall_rate == 0.8
        assert len(result.monthly_stats) == 1

    def test_missing_dates_handled(self):
        analyzer = TemporalTargetAnalyzer()
        df = pd.DataFrame({
            'date': [datetime(2022, 1, 1), None, datetime(2022, 6, 15), pd.NaT],
            'target': [1, 0, 1, 0]
        })
        result = analyzer.analyze(df, 'date', 'target')

        assert result.n_valid_dates == 2

    def test_string_dates_parsed(self):
        analyzer = TemporalTargetAnalyzer()
        df = pd.DataFrame({
            'date': ['2022-01-01', '2022-06-15', '2022-12-31'],
            'target': [1, 1, 0]
        })
        result = analyzer.analyze(df, 'date', 'target')

        assert result.n_valid_dates == 3

    def test_all_unparseable_dates(self):
        """Test that all unparseable dates returns empty result."""
        analyzer = TemporalTargetAnalyzer()
        df = pd.DataFrame({
            'date': ['not-a-date', 'also-bad', 'nope'],
            'target': [1, 0, 1]
        })
        result = analyzer.analyze(df, 'date', 'target')

        assert result.n_valid_dates == 0
        assert result.overall_rate == 0.0

    def test_missing_datetime_column(self):
        """Test with a datetime column that does not exist."""
        analyzer = TemporalTargetAnalyzer()
        df = pd.DataFrame({'other': [1, 2, 3], 'target': [1, 0, 1]})
        result = analyzer.analyze(df, 'nonexistent', 'target')

        assert result.n_valid_dates == 0

    def test_missing_target_column(self):
        """Test with a target column that does not exist."""
        analyzer = TemporalTargetAnalyzer()
        df = pd.DataFrame({'date': pd.date_range('2022-01-01', periods=3), 'other': [1, 0, 1]})
        result = analyzer.analyze(df, 'date', 'nonexistent')

        assert result.n_valid_dates == 0

    def test_yearly_trend_stable(self):
        """Test that stable trend is returned when slope is within threshold."""
        analyzer = TemporalTargetAnalyzer(min_samples_per_period=1)
        # Same retention rate across 3 years
        data = []
        for year in [2020, 2021, 2022]:
            for _ in range(50):
                data.append({'date': pd.Timestamp(f'{year}-06-15'), 'target': 1})
            for _ in range(50):
                data.append({'date': pd.Timestamp(f'{year}-06-16'), 'target': 0})
        df = pd.DataFrame(data)
        result = analyzer.analyze(df, 'date', 'target')

        assert result.yearly_trend == 'stable'

    def test_yearly_trend_declining(self):
        """Test detection of declining yearly trend."""
        analyzer = TemporalTargetAnalyzer(min_samples_per_period=1)
        data = []
        # 2020: 90% retention, 2021: 70%, 2022: 50%
        for year, rate in [(2020, 0.9), (2021, 0.7), (2022, 0.5)]:
            n = 100
            targets = [1] * int(n * rate) + [0] * (n - int(n * rate))
            dates = pd.date_range(f'{year}-01-01', periods=n, freq='D')
            for d, t in zip(dates, targets):
                data.append({'date': d, 'target': t})
        df = pd.DataFrame(data)
        result = analyzer.analyze(df, 'date', 'target')

        assert result.yearly_trend == 'declining'

    def test_analyze_multiple(self):
        """Test analyze_multiple method."""
        analyzer = TemporalTargetAnalyzer(min_samples_per_period=1)
        df = pd.DataFrame({
            'signup_date': pd.date_range('2022-01-01', periods=100, freq='D'),
            'last_login': pd.date_range('2022-06-01', periods=100, freq='D'),
            'retained': [1] * 70 + [0] * 30
        })
        summary = analyzer.analyze_multiple(df, ['signup_date', 'last_login'], 'retained')

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert 'feature' in summary.columns
        assert 'yearly_trend' in summary.columns
        assert 'best_month' in summary.columns
        assert 'worst_month' in summary.columns
        assert 'seasonal_spread' in summary.columns

    def test_seasonal_extremes_empty_monthly_stats(self):
        """Test _find_seasonal_extremes with empty stats."""
        analyzer = TemporalTargetAnalyzer(min_samples_per_period=9999)
        # With very high min_samples_per_period, all months will be filtered out
        df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=10, freq='D'),
            'target': [1] * 5 + [0] * 5
        })
        result = analyzer.analyze(df, 'date', 'target')

        assert result.best_month is None
        assert result.worst_month is None
        assert result.seasonal_spread == 0.0
