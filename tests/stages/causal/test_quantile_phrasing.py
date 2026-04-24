"""Tests for the quantile-based phrase renderer."""
from __future__ import annotations

import pytest

from customer_retention.stages.causal.interpretation.quantile_phrasing import (
    PopulationStats,
    quantile_phrase,
)


@pytest.fixture
def stats():
    return PopulationStats(q05=5.0, q25=10.0, q50=20.0, q75=30.0, q95=50.0)


class TestQuantilePhraseBands:
    def test_below_q05_is_very_low(self, stats):
        assert quantile_phrase(1.0, stats) == "very low"

    def test_at_q05_is_low(self, stats):
        assert quantile_phrase(5.0, stats) == "low"

    def test_between_q05_q25_is_low(self, stats):
        assert quantile_phrase(7.0, stats) == "low"

    def test_between_q25_q75_is_typical(self, stats):
        assert quantile_phrase(20.0, stats) == "typical"

    def test_at_q75_is_typical(self, stats):
        assert quantile_phrase(30.0, stats) == "typical"

    def test_between_q75_q95_is_elevated(self, stats):
        assert quantile_phrase(40.0, stats) == "elevated"

    def test_above_q95_is_very_high(self, stats):
        assert quantile_phrase(99.0, stats) == "very high"

    def test_at_q95_is_elevated(self, stats):
        assert quantile_phrase(50.0, stats) == "elevated"


class TestPolarityInversion:
    def test_high_is_bad_inverts(self, stats):
        assert quantile_phrase(1.0, stats, polarity="high_is_bad") == "very high"
        assert quantile_phrase(99.0, stats, polarity="high_is_bad") == "very low"

    def test_high_is_bad_keeps_typical(self, stats):
        assert quantile_phrase(20.0, stats, polarity="high_is_bad") == "typical"

    def test_neutral_polarity_does_not_invert(self, stats):
        assert quantile_phrase(99.0, stats, polarity="neutral") == "very high"

    def test_high_is_good_does_not_invert(self, stats):
        assert quantile_phrase(99.0, stats, polarity="high_is_good") == "very high"

    def test_unknown_polarity_does_not_invert(self, stats):
        assert quantile_phrase(99.0, stats, polarity="completely_bogus") == "very high"


class TestFallbacks:
    def test_none_value_returns_unknown(self, stats):
        assert quantile_phrase(None, stats) == "unknown"

    def test_missing_q05_returns_unknown(self):
        stats = PopulationStats(q25=10.0, q50=20.0, q75=30.0, q95=50.0)
        assert quantile_phrase(5.0, stats) == "unknown"

    def test_missing_q95_returns_unknown(self):
        stats = PopulationStats(q05=5.0, q25=10.0, q50=20.0, q75=30.0)
        assert quantile_phrase(5.0, stats) == "unknown"

    def test_all_quantiles_none_returns_unknown(self):
        assert quantile_phrase(5.0, PopulationStats()) == "unknown"
