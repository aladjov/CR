"""Tests for the dashboard view loader + publisher."""

from __future__ import annotations

from unittest.mock import MagicMock

from customer_retention.stages.causal.dashboard_views import (
    DASHBOARD_PROVENANCE_VIEW_NAMES,
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


def _view_block(rendered: str, view_name: str) -> str:
    """Return the SQL chunk that defines ``view_name`` (CREATE OR REPLACE
    VIEW ... up to its terminating ``;``)."""
    import re as _re

    m = _re.search(
        rf"CREATE OR REPLACE VIEW [^\s]*\.{view_name} AS(.*?);",
        rendered,
        _re.DOTALL,
    )
    assert m, f"view {view_name!r} not found in rendered SQL"
    return m.group(0)


class TestEntityGrainSemantics:
    """Pin the per-account semantics of the dashboard surface.

    ``eligibility_snapshot`` is keyed on ``(scoring_run_id, entity_id,
    playbook_id)`` -- naively summing rows multi-counts every account that
    matches more than one play. The dashboard collapses to the primary
    play per entity via ``v_account_primary_recommendation`` and routes
    every entity-grain rollup through it. These tests catch the regression
    where a future edit reverts to row-grain.
    """

    def test_primary_recommendation_view_exists(self):
        rendered = render_dashboard_view_sql("c", "s")
        assert "c.s.v_account_primary_recommendation" in rendered

    def test_primary_recommendation_view_partitions_by_entity(self):
        block = _view_block(
            render_dashboard_view_sql("c", "s"),
            "v_account_primary_recommendation",
        )
        # Entity-grain collapse: ROW_NUMBER() OVER (PARTITION BY entity_id)
        # picks one row per entity, then WHERE entity_rank = 1 enforces it.
        assert "PARTITION BY entity_id" in block
        assert "entity_rank = 1" in block

    def test_primary_recommendation_view_carries_alternates_array(self):
        block = _view_block(
            render_dashboard_view_sql("c", "s"),
            "v_account_primary_recommendation",
        )
        # Plays the account also passed eligibility for ride along on the
        # primary row as an array of structs (playbook_name + scoring) so
        # the dashboard never has to fan out to multi-grain rows.
        assert "alternates" in block
        assert "alternate_count" in block
        assert "COLLECT_LIST" in block
        for field in (
            "'playbook_id'",
            "'playbook_name'",
            "'churn_probability'",
            "'fit_score'",
            "'expected_uplift_pct'",
        ):
            assert field in block, f"alternates struct missing field {field}"

    def test_primary_recommendation_ranks_recommended_first(self):
        block = _view_block(
            render_dashboard_view_sql("c", "s"),
            "v_account_primary_recommendation",
        )
        # When an account is recommended for one play and merely eligible
        # for another, the recommended one wins as primary.
        assert "CASE WHEN recommended THEN 0 ELSE 1 END" in block

    def test_portfolio_risk_matrix_sources_from_primary_view(self):
        # The L1 KPI tiles route through the entity-grain view so summing
        # eligible_count == DISTINCT entity count, never row count.
        block = _view_block(
            render_dashboard_view_sql("c", "s"),
            "v_portfolio_risk_matrix",
        )
        assert "v_account_primary_recommendation" in block
        # Must NOT touch eligibility_snapshot directly -- doing so would
        # reintroduce the multi-counting bug.
        assert "eligibility_snapshot" not in block

    def test_playbook_archetype_rollup_sources_from_primary_view(self):
        block = _view_block(
            render_dashboard_view_sql("c", "s"),
            "v_playbook_archetype_rollup",
        )
        assert "v_account_primary_recommendation" in block
        assert "eligibility_snapshot" not in block

    def test_eligible_all_playbooks_sources_from_primary_view(self):
        # L3 cohort drill is also entity-grain: drilling into a playbook
        # shows accounts whose PRIMARY play is that playbook (others ride
        # on the alternates array). The user-facing question "which
        # customers should I act on under play X" is per-account.
        block = _view_block(
            render_dashboard_view_sql("c", "s"),
            "v_eligible_all_playbooks",
        )
        assert "v_account_primary_recommendation" in block
        assert "eligibility_snapshot" not in block


class TestSplitViewStatements:
    def test_splits_one_statement_per_named_view(self):
        rendered = render_dashboard_view_sql("c", "s")
        statements = split_view_statements(rendered)
        # Default render includes the (gated-at-publish-time) provenance view.
        assert len(statements) == len(DASHBOARD_VIEW_NAMES) + len(
            DASHBOARD_PROVENANCE_VIEW_NAMES
        )
        for stmt in statements:
            assert "CREATE OR REPLACE VIEW" in stmt

    def test_strips_empty_trailing_segment(self):
        statements = split_view_statements("SELECT 1; SELECT 2;")
        assert statements == ["SELECT 1", "SELECT 2"]

    def test_skips_blank_lines(self):
        statements = split_view_statements("SELECT 1;\n\n\n;\nSELECT 2")
        assert statements == ["SELECT 1", "SELECT 2"]

    def test_ignores_semicolon_inside_line_comment(self):
        # Operator-supplied SQL (e.g. the ECH profile override) puts English
        # prose with semicolons inside ``--`` comments.  The splitter must
        # not cut mid-comment -- doing so hands Spark a truncated DDL and
        # surfaces ``[PARSE_SYNTAX_ERROR] Syntax error at or near end of input``.
        sql = (
            "CREATE VIEW v AS\n"
            "SELECT 1\n"
            "-- NB03 materialises and the scoring path consumes; reading from it\n"
            "FROM t;"
        )
        statements = split_view_statements(sql)
        assert len(statements) == 1
        assert "FROM t" in statements[0]
        assert "consumes; reading from it" in statements[0]

    def test_ignores_semicolon_inside_string_literal(self):
        sql = "INSERT INTO t VALUES ('a;b'); SELECT 1;"
        statements = split_view_statements(sql)
        assert statements == ["INSERT INTO t VALUES ('a;b')", "SELECT 1"]

    def test_ignores_semicolon_inside_backticks(self):
        # Delta path notation uses backticks: ``delta.`/Volumes/.../foo```.
        sql = "SELECT * FROM delta.`/Volumes/x;y/z`; SELECT 2;"
        statements = split_view_statements(sql)
        assert statements == [
            "SELECT * FROM delta.`/Volumes/x;y/z`",
            "SELECT 2",
        ]


class TestPublishDashboardViews:
    def test_publishes_one_statement_per_named_view(self):
        spark = MagicMock()
        statements = publish_dashboard_views(spark, "c", "s")
        # MagicMock's tableExists returns truthy for the provenance prereqs
        # (feature_meta + column_descriptions), so the publisher includes the
        # provenance view too.
        expected = len(DASHBOARD_VIEW_NAMES) + len(DASHBOARD_PROVENANCE_VIEW_NAMES)
        assert len(statements) == expected
        # publish_dashboard_views also issues a CREATE TABLE IF NOT EXISTS for
        # run_context (the v_run_context view references it and Spark validates
        # the view body at DDL time), so spark.sql is called once extra.
        assert spark.sql.call_count == expected + 1
        view_calls = [
            call for call in spark.sql.call_args_list
            if "CREATE OR REPLACE VIEW" in call.args[0]
        ]
        assert len(view_calls) == expected

    def test_ensures_run_context_table_before_publishing_views(self):
        spark = MagicMock()
        publish_dashboard_views(spark, "c", "s")
        first_sql = spark.sql.call_args_list[0].args[0]
        assert "CREATE TABLE IF NOT EXISTS c.s.run_context" in first_sql

    def test_substitutes_catalog_and_schema_into_sql(self):
        spark = MagicMock()
        publish_dashboard_views(spark, "alpha", "beta")
        joined = "\n".join(call.args[0] for call in spark.sql.call_args_list)
        assert "alpha.beta." in joined
