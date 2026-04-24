"""Tests for the deterministic business-phrase renderer."""
from __future__ import annotations

import pytest

from customer_retention.stages.causal.interpretation.business_phrase import (
    render_business_phrase,
    render_window_phrase,
)


class TestRenderBusinessPhrase:
    def test_count_with_window_uses_over_suffix(self):
        assert (
            render_business_phrase("count", "missed payments", "last 30 days")
            == "count of missed payments over last 30 days"
        )

    def test_avg_and_mean_map_to_average(self):
        assert render_business_phrase("avg", "NPS score", "last 90 days") == "average NPS score over last 90 days"
        assert render_business_phrase("mean", "NPS score", "last 90 days") == "average NPS score over last 90 days"

    def test_recency_days_omits_window_suffix(self):
        assert render_business_phrase("recency_days", "login", "last 90 days") == "days since most recent login"

    def test_recency_days_without_window_still_valid(self):
        assert render_business_phrase("recency_days", "login") == "days since most recent login"

    def test_passthrough_returns_business_name_only(self):
        assert render_business_phrase("passthrough", "NPS score", "last 30 days") == "NPS score"

    def test_derived_datetime_returns_business_name_only(self):
        assert render_business_phrase("derived_datetime", "tenure days", "last 365 days") == "tenure days"

    def test_unknown_kind_falls_back_to_business_name(self):
        assert render_business_phrase("totally_made_up", "NPS score", "last 30 days") == "NPS score"

    def test_none_kind_falls_back_to_business_name(self):
        assert render_business_phrase(None, "NPS score", "last 30 days") == "NPS score"

    def test_empty_business_name_returns_empty_string(self):
        assert render_business_phrase("count", "", "last 30 days") == ""

    def test_whitespace_only_business_name_returns_empty_string(self):
        assert render_business_phrase("count", "   ", "last 30 days") == ""

    def test_none_business_name_returns_empty_string(self):
        assert render_business_phrase("count", None, "last 30 days") == ""  # type: ignore[arg-type]

    def test_strips_business_name_whitespace(self):
        assert (
            render_business_phrase("count", "  missed payments  ", "last 30 days")
            == "count of missed payments over last 30 days"
        )

    def test_count_without_window_returns_no_over_suffix(self):
        assert render_business_phrase("count", "missed payments") == "count of missed payments"

    def test_count_with_empty_window_returns_no_over_suffix(self):
        assert render_business_phrase("count", "missed payments", "") == "count of missed payments"

    @pytest.mark.parametrize(
        "kind, expected_verb",
        [
            ("count_distinct", "distinct count of"),
            ("sum", "sum of"),
            ("max", "maximum"),
            ("min", "minimum"),
            ("last", "most recent"),
            ("first", "first observed"),
            ("ratio", "share of"),
        ],
    )
    def test_every_known_verb_rendered(self, kind, expected_verb):
        phrase = render_business_phrase(kind, "value")
        assert phrase.startswith(expected_verb)


class TestRenderWindowPhrase:
    def test_positive_days_returns_last_n_days(self):
        assert render_window_phrase(30) == "last 30 days"
        assert render_window_phrase(365) == "last 365 days"

    def test_none_returns_lifetime(self):
        assert render_window_phrase(None) == "lifetime"

    def test_zero_returns_lifetime(self):
        assert render_window_phrase(0) == "lifetime"

    def test_negative_returns_lifetime(self):
        assert render_window_phrase(-1) == "lifetime"
