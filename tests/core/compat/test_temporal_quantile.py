import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import temporal_quantile


class TestTemporalQuantile:
    @pytest.fixture
    def hourly_series(self):
        return pd.Series(pd.date_range("2023-01-01", periods=1000, freq="h"))

    @pytest.fixture
    def daily_series(self):
        return pd.Series(pd.date_range("2023-01-01", periods=365, freq="D"))

    def test_returns_timestamp(self, hourly_series):
        result = temporal_quantile(hourly_series, 0.8)
        assert isinstance(result, pd.Timestamp)

    def test_quantile_zero_returns_min(self, hourly_series):
        result = temporal_quantile(hourly_series, 0.0)
        assert result == hourly_series.min()

    def test_quantile_one_returns_max(self, hourly_series):
        result = temporal_quantile(hourly_series, 1.0)
        assert result == hourly_series.max()

    def test_quantile_half_returns_median(self, daily_series):
        result = temporal_quantile(daily_series, 0.5)
        expected = daily_series.quantile(0.5)
        assert abs((result - expected).total_seconds()) < 86400

    def test_80_pct_splits_near_80_20(self, hourly_series):
        cutoff = temporal_quantile(hourly_series, 0.8)
        before = (hourly_series < cutoff).sum()
        after = (hourly_series >= cutoff).sum()
        ratio = before / len(hourly_series)
        assert 0.78 <= ratio <= 0.82

    def test_monotonic_quantiles(self, hourly_series):
        q1 = temporal_quantile(hourly_series, 0.2)
        q2 = temporal_quantile(hourly_series, 0.5)
        q3 = temporal_quantile(hourly_series, 0.8)
        assert q1 <= q2 <= q3

    def test_single_element_series(self):
        series = pd.Series([pd.Timestamp("2023-06-15")])
        result = temporal_quantile(series, 0.5)
        assert result == pd.Timestamp("2023-06-15")

    def test_two_element_series(self):
        series = pd.Series([pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")])
        result = temporal_quantile(series, 0.5)
        assert pd.Timestamp("2023-01-01") <= result <= pd.Timestamp("2023-12-31")

    def test_unsorted_input_still_correct(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        shuffled = pd.Series(dates).sample(frac=1, random_state=42).reset_index(drop=True)
        result = temporal_quantile(shuffled, 0.8)
        expected = temporal_quantile(pd.Series(dates), 0.8)
        assert abs((result - expected).total_seconds()) < 86400

    def test_with_duplicate_dates(self):
        dates = [pd.Timestamp("2023-01-01")] * 50 + [pd.Timestamp("2023-06-01")] * 50
        series = pd.Series(dates)
        result = temporal_quantile(series, 0.4)
        assert result == pd.Timestamp("2023-01-01")
        result_high = temporal_quantile(series, 0.6)
        assert result_high == pd.Timestamp("2023-06-01")

    def test_named_series_works(self, hourly_series):
        named = hourly_series.rename("as_of_date")
        result = temporal_quantile(named, 0.8)
        assert isinstance(result, pd.Timestamp)
