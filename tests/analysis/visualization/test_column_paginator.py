"""Tests for ColumnPaginator — pagination, filtering, sorting logic."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

from customer_retention.analysis.visualization.attention_scorer import AttentionScore
from customer_retention.analysis.visualization.column_paginator import (
    ColumnEntry,
    ColumnPaginator,
    has_interactive_widgets,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(name: str, score: float, col_type: str = "numeric") -> ColumnEntry:
    return ColumnEntry(
        name=name,
        column_type=col_type,
        attention_score=AttentionScore(
            column_name=name, column_type=col_type, score=score
        ),
    )


def _noop_render(entry, output):
    pass


def _noop_summary(entry):
    return {"Column": entry.name, "Score": entry.attention_score.score}


# ===================================================================
# ColumnEntry
# ===================================================================


class TestColumnEntry:
    def test_basic_construction(self):
        e = _entry("col_a", 50.0)
        assert e.name == "col_a"
        assert e.column_type == "numeric"
        assert e.attention_score.score == 50.0

    def test_extra_defaults_to_empty(self):
        e = _entry("col_a", 50.0)
        assert e.extra == {}

    def test_analysis_data_none(self):
        e = _entry("col_a", 50.0)
        assert e.analysis_data is None

    def test_recommendation_data_none(self):
        e = _entry("col_a", 50.0)
        assert e.recommendation_data is None


# ===================================================================
# Filtering logic
# ===================================================================


class TestFiltering:
    def _paginator(self, entries=None, page_size=5):
        if entries is None:
            entries = [
                _entry("alpha", 80),
                _entry("beta", 60),
                _entry("gamma", 40),
                _entry("delta", 20),
                _entry("epsilon", 10),
            ]
        return ColumnPaginator(
            entries=entries,
            render_callback=_noop_render,
            summary_callback=_noop_summary,
            page_size=page_size,
        )

    def test_initial_sorted_by_score(self):
        p = self._paginator()
        assert [e.name for e in p._filtered_entries] == [
            "alpha", "beta", "gamma", "delta", "epsilon"
        ]

    def test_min_score_filter(self):
        p = self._paginator()
        p._min_score = 50
        p._apply_filters()
        assert [e.name for e in p._filtered_entries] == ["alpha", "beta"]

    def test_search_filter(self):
        p = self._paginator()
        p._search_text = "alph"
        p._apply_filters()
        names = [e.name for e in p._filtered_entries]
        assert names == ["alpha"]

    def test_search_case_insensitive(self):
        entries = [_entry("MyColumn", 50)]
        p = self._paginator(entries=entries)
        p._search_text = "mycolumn"
        p._apply_filters()
        assert len(p._filtered_entries) == 1

    def test_sort_by_name(self):
        p = self._paginator()
        p._sort_key = "name"
        p._apply_filters()
        assert [e.name for e in p._filtered_entries] == [
            "alpha", "beta", "delta", "epsilon", "gamma"
        ]

    def test_sort_by_type(self):
        entries = [
            _entry("a", 50, "numeric"),
            _entry("b", 60, "categorical"),
            _entry("c", 40, "numeric"),
        ]
        p = self._paginator(entries=entries)
        p._sort_key = "type"
        p._apply_filters()
        assert p._filtered_entries[0].column_type == "categorical"

    def test_combined_search_and_min_score(self):
        p = self._paginator()
        p._search_text = "alph"
        p._min_score = 50
        p._apply_filters()
        assert [e.name for e in p._filtered_entries] == ["alpha"]

    def test_empty_after_filter(self):
        p = self._paginator()
        p._min_score = 99
        p._apply_filters()
        assert len(p._filtered_entries) == 0


# ===================================================================
# Pagination math
# ===================================================================


class TestPaginationMath:
    def test_page_count(self):
        entries = [_entry(f"c{i}", i) for i in range(23)]
        p = ColumnPaginator(entries, _noop_render, _noop_summary, page_size=10)
        total_pages = math.ceil(len(p._filtered_entries) / p._page_size)
        assert total_pages == 3

    def test_partial_last_page(self):
        entries = [_entry(f"c{i}", i) for i in range(23)]
        p = ColumnPaginator(entries, _noop_render, _noop_summary, page_size=10)
        # Last page has 3 entries
        start = 2 * 10
        last_page = p._filtered_entries[start : start + 10]
        assert len(last_page) == 3

    def test_single_entry(self):
        entries = [_entry("only", 50)]
        p = ColumnPaginator(entries, _noop_render, _noop_summary, page_size=10)
        assert len(p._filtered_entries) == 1

    def test_empty_entries(self):
        p = ColumnPaginator([], _noop_render, _noop_summary, page_size=10)
        assert len(p._filtered_entries) == 0

    def test_page_size_larger_than_entries(self):
        entries = [_entry(f"c{i}", i) for i in range(3)]
        p = ColumnPaginator(entries, _noop_render, _noop_summary, page_size=10)
        total_pages = math.ceil(len(p._filtered_entries) / p._page_size)
        assert total_pages == 1


# ===================================================================
# Render callback invocation
# ===================================================================


class TestRenderCallback:
    def test_render_called_for_page_entries_only(self):
        entries = [_entry(f"c{i}", 100 - i) for i in range(15)]
        render_mock = MagicMock()
        p = ColumnPaginator(entries, render_mock, _noop_summary, page_size=5)
        p._render_page(0)
        assert render_mock.call_count == 5

    def test_render_partial_last_page(self):
        entries = [_entry(f"c{i}", 100 - i) for i in range(12)]
        render_mock = MagicMock()
        p = ColumnPaginator(entries, render_mock, _noop_summary, page_size=5)
        p._render_page(2)
        assert render_mock.call_count == 2

    def test_render_empty_page(self):
        render_mock = MagicMock()
        p = ColumnPaginator([], render_mock, _noop_summary, page_size=5)
        p._render_page(0)
        assert render_mock.call_count == 0


# ===================================================================
# Summary callback invocation
# ===================================================================


class TestSummaryCallback:
    def test_summary_called_for_all_entries(self):
        entries = [_entry(f"c{i}", i) for i in range(7)]
        summary_mock = MagicMock(return_value={"Column": "x", "Score": 0})
        p = ColumnPaginator(entries, _noop_render, summary_mock, page_size=3)
        p._show_summary_table()
        assert summary_mock.call_count == 7


# ===================================================================
# Fallback path
# ===================================================================


class TestFallback:
    @patch("customer_retention.analysis.visualization.column_paginator.has_interactive_widgets", return_value=False)
    @patch("customer_retention.analysis.visualization.column_paginator.display_table")
    def test_fallback_renders_top_n(self, mock_table, mock_widgets):
        entries = [_entry(f"c{i}", 100 - i) for i in range(30)]
        render_mock = MagicMock()
        p = ColumnPaginator(
            entries, render_mock, _noop_summary, page_size=5, fallback_top_n=10
        )
        p.show()
        # Fallback should render top 10 entries
        assert render_mock.call_count == 10

    @patch("customer_retention.analysis.visualization.column_paginator.has_interactive_widgets", return_value=False)
    @patch("customer_retention.analysis.visualization.column_paginator.display_table")
    def test_fallback_fewer_than_top_n(self, mock_table, mock_widgets):
        entries = [_entry(f"c{i}", 50) for i in range(3)]
        render_mock = MagicMock()
        p = ColumnPaginator(
            entries, render_mock, _noop_summary, fallback_top_n=20
        )
        p.show()
        assert render_mock.call_count == 3


# ===================================================================
# has_interactive_widgets
# ===================================================================


class TestHasInteractiveWidgets:
    @patch("customer_retention.analysis.visualization.column_paginator.HAS_WIDGETS", False)
    def test_no_widgets_installed(self):
        assert has_interactive_widgets() is False

    @patch("customer_retention.analysis.visualization.column_paginator.HAS_WIDGETS", True)
    @patch("customer_retention.analysis.visualization.column_paginator.detect_environment", return_value="terminal")
    def test_terminal_environment(self, mock_env):
        assert has_interactive_widgets() is False

    @patch("customer_retention.analysis.visualization.column_paginator.HAS_WIDGETS", True)
    @patch("customer_retention.analysis.visualization.column_paginator.detect_environment", return_value="jupyter")
    def test_jupyter_environment(self, mock_env):
        assert has_interactive_widgets() is True

    @patch("customer_retention.analysis.visualization.column_paginator.HAS_WIDGETS", True)
    @patch("customer_retention.analysis.visualization.column_paginator.detect_environment", return_value="databricks")
    def test_databricks_environment(self, mock_env):
        assert has_interactive_widgets() is True
