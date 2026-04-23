"""Tests for the dashboard view loader + publisher."""

from __future__ import annotations

from unittest.mock import MagicMock

from customer_retention.stages.causal.dashboard_views import (
    DASHBOARD_VIEW_NAMES,
    load_dashboard_view_sql,
    publish_dashboard_views,
    render_dashboard_view_sql,
    split_view_statements,
)


class TestLoadDashboardViewSql:
    def test_returns_non_empty_text(self):
        text = load_dashboard_view_sql()
        assert isinstance(text, str)
        assert len(text) > 200

    def test_contains_every_named_view(self):
        text = load_dashboard_view_sql()
        for name in DASHBOARD_VIEW_NAMES:
            assert name in text


class TestRenderDashboardViewSql:
    def test_substitutes_catalog_and_schema(self):
        rendered = render_dashboard_view_sql("my_catalog", "my_schema")
        assert "{catalog}" not in rendered
        assert "{schema}" not in rendered
        assert "my_catalog.my_schema." in rendered

    def test_substitution_idempotent_on_re_render(self):
        rendered1 = render_dashboard_view_sql("c", "s")
        rendered2 = render_dashboard_view_sql("c", "s")
        assert rendered1 == rendered2


class TestSplitViewStatements:
    def test_splits_one_statement_per_named_view(self):
        rendered = render_dashboard_view_sql("c", "s")
        statements = split_view_statements(rendered)
        assert len(statements) == len(DASHBOARD_VIEW_NAMES)
        for stmt in statements:
            assert "CREATE OR REPLACE VIEW" in stmt

    def test_strips_empty_trailing_segment(self):
        statements = split_view_statements("SELECT 1; SELECT 2;")
        assert statements == ["SELECT 1", "SELECT 2"]

    def test_skips_blank_lines(self):
        statements = split_view_statements("SELECT 1;\n\n\n;\nSELECT 2")
        assert statements == ["SELECT 1", "SELECT 2"]


class TestPublishDashboardViews:
    def test_publishes_one_statement_per_named_view(self):
        spark = MagicMock()
        statements = publish_dashboard_views(spark, "c", "s")
        expected = len(DASHBOARD_VIEW_NAMES)
        assert len(statements) == expected
        assert spark.sql.call_count == expected
        for call in spark.sql.call_args_list:
            assert "CREATE OR REPLACE VIEW" in call.args[0]

    def test_substitutes_catalog_and_schema_into_sql(self):
        spark = MagicMock()
        publish_dashboard_views(spark, "alpha", "beta")
        joined = "\n".join(call.args[0] for call in spark.sql.call_args_list)
        assert "alpha.beta." in joined
