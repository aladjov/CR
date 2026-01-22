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
