"""Tests for the dashboard view loader + publisher."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from customer_retention.stages.causal.dashboard_views import (
    _MATERIALIZED_VIEW_SPECS,
    DASHBOARD_PROVENANCE_VIEW_NAMES,
    DASHBOARD_VIEW_NAMES,
    _resolve_gold_dedup_order_by,
    _synthetic_placeholder_column_descriptions,
    load_dashboard_view_sql,
    publish_dashboard_views,
    render_dashboard_view_sql,
    split_view_statements,
)

# Number of materialization specs that fire when no composite_name is passed
# (i.e. specs whose ``requires_composite`` is False). Each spec adds three
# spark.sql calls: CTAS, OPTIMIZE, and a final CREATE OR REPLACE VIEW that
# re-points the public view at the materialized Delta table. Only the third
# of those three is a CREATE OR REPLACE VIEW.
_MATERIALIZE_NONCOMPOSITE_SPECS = sum(
    1 for s in _MATERIALIZED_VIEW_SPECS if not s.requires_composite
)
_MATERIALIZE_NONCOMPOSITE_SQL_CALLS = _MATERIALIZE_NONCOMPOSITE_SPECS * 3


class TestResolveGoldDedupOrderBy:
    def test_prefers_as_of_date(self):
        assert _resolve_gold_dedup_order_by(
            ["entity_id", "as_of_date", "inference_point_in_time", "feature_x"]
        ) == "`as_of_date` DESC"

    def test_falls_back_to_inference_point_in_time(self):
        assert _resolve_gold_dedup_order_by(
            ["entity_id", "inference_point_in_time", "feature_x"]
        ) == "`inference_point_in_time` DESC"

    def test_falls_back_to_scoring_run_id(self):
        assert _resolve_gold_dedup_order_by(
            ["entity_id", "scoring_run_id", "feature_x"]
        ) == "`scoring_run_id` DESC"

    def test_returns_literal_one_when_no_timestamp_column(self):
        # User's email-aggregated gold has only entity_id + feature columns,
        # no timestamp. Must not break view publication — degrades to
        # ``ORDER BY 1`` which sorts by the first selected column
        # (deterministic, even if arbitrary).
        assert _resolve_gold_dedup_order_by(
            ["entity_id", "dow_cos", "active_span_days", "event_count_365d"]
        ) == "1"

    def test_returns_literal_one_for_empty_or_none(self):
        assert _resolve_gold_dedup_order_by(None) == "1"
        assert _resolve_gold_dedup_order_by([]) == "1"


class TestSyntheticPlaceholderColumnDescriptions:
    def test_includes_event_placeholder(self):
        rows = _synthetic_placeholder_column_descriptions()
        names = {r.column_name for r in rows}
        assert "event" in names, (
            "event_count_* features need a column_descriptions row "
            "for the synthetic 'event' source-column placeholder so "
            "v_feature_provenance can render a business_definition under "
            "each event-derived SHAP driver in the dashboard"
        )

    def test_includes_event_gap_placeholder(self):
        rows = _synthetic_placeholder_column_descriptions()
        names = {r.column_name for r in rows}
        assert "event_gap" in names, (
            "inter_event_gap_* features use the 'event_gap' synthetic "
            "source-column placeholder; without a row for it, the "
            "dashboard panel shows null business_definition for those features"
        )

    def test_placeholder_rows_have_business_definition(self):
        for row in _synthetic_placeholder_column_descriptions():
            assert row.business_name, f"placeholder {row.column_name} missing business_name"
            assert row.business_definition, f"placeholder {row.column_name} missing business_definition"
            assert row.source == "framework_synthetic"


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
        # Default render includes the (gated-at-publish-time) provenance view
        # AND the always-on dashboard-template store: one CREATE TABLE for
        # ``dashboard_template_overrides`` plus one CREATE VIEW for
        # ``v_dashboard_template_active``.
        from customer_retention.stages.causal.dashboard_views import (
            DASHBOARD_TEMPLATE_TABLE_NAMES,
            DASHBOARD_TEMPLATE_VIEW_NAMES,
        )
        expected = (
            len(DASHBOARD_VIEW_NAMES)
            + len(DASHBOARD_PROVENANCE_VIEW_NAMES)
            + len(DASHBOARD_TEMPLATE_TABLE_NAMES)
            + len(DASHBOARD_TEMPLATE_VIEW_NAMES)
        )
        assert len(statements) == expected
        # Every statement is either a CREATE VIEW or the template's CREATE TABLE.
        for stmt in statements:
            assert (
                "CREATE OR REPLACE VIEW" in stmt
                or "CREATE TABLE IF NOT EXISTS" in stmt
            )

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
        pytest.importorskip("pyspark")
        spark = MagicMock()
        statements = publish_dashboard_views(spark, "c", "s")
        # MagicMock's tableExists returns truthy for the provenance prereqs
        # (feature_meta + column_descriptions), so the publisher includes the
        # provenance view too. The template-store CREATE TABLE +
        # v_dashboard_template_active CREATE VIEW always render.
        from customer_retention.stages.causal.dashboard_views import (
            DASHBOARD_TEMPLATE_TABLE_NAMES,
            DASHBOARD_TEMPLATE_VIEW_NAMES,
        )
        expected = (
            len(DASHBOARD_VIEW_NAMES)
            + len(DASHBOARD_PROVENANCE_VIEW_NAMES)
            + len(DASHBOARD_TEMPLATE_TABLE_NAMES)
            + len(DASHBOARD_TEMPLATE_VIEW_NAMES)
        )
        assert len(statements) == expected
        # publish_dashboard_views also issues a CREATE TABLE IF NOT EXISTS for
        # run_context (the v_run_context view references it and Spark validates
        # the view body at DDL time), and one extra CREATE TABLE IF NOT EXISTS
        # for ``column_descriptions`` issued by the synthetic-placeholder seed
        # (the merge it executes after fails on MagicMock and is swallowed,
        # but the CREATE TABLE call count is still recorded). It then issues
        # the materialization pass: per non-composite spec, three more calls
        # (CREATE OR REPLACE TABLE, OPTIMIZE ZORDER, CREATE OR REPLACE VIEW)
        # that snapshot the hot-path view into an indexed Delta table so the
        # app's per-click reads collapse to point lookups. Each CREATE VIEW
        # statement is now preceded by a DROP VIEW IF EXISTS so stored view
        # schemas reset cleanly on every publish (Unity Catalog otherwise
        # holds onto stale nullability metadata that trips downstream casts).
        view_create_count = (
            len(DASHBOARD_VIEW_NAMES)
            + len(DASHBOARD_PROVENANCE_VIEW_NAMES)
            + len(DASHBOARD_TEMPLATE_VIEW_NAMES)
        )
        assert spark.sql.call_count == (
            expected + 2 + _MATERIALIZE_NONCOMPOSITE_SQL_CALLS + view_create_count
        )
        drop_calls = [
            call for call in spark.sql.call_args_list
            if call.args[0].startswith("DROP VIEW IF EXISTS")
        ]
        assert len(drop_calls) == view_create_count
        view_calls = [
            call for call in spark.sql.call_args_list
            if "CREATE OR REPLACE VIEW" in call.args[0]
        ]
        assert len(view_calls) == (
            len(DASHBOARD_VIEW_NAMES)
            + len(DASHBOARD_PROVENANCE_VIEW_NAMES)
            + len(DASHBOARD_TEMPLATE_VIEW_NAMES)
            + _MATERIALIZE_NONCOMPOSITE_SPECS
        )

    def test_ensures_run_context_table_before_publishing_views(self):
        pytest.importorskip("pyspark")
        spark = MagicMock()
        publish_dashboard_views(spark, "c", "s")
        first_sql = spark.sql.call_args_list[0].args[0]
        assert "CREATE TABLE IF NOT EXISTS c.s.run_context" in first_sql

    def test_substitutes_catalog_and_schema_into_sql(self):
        pytest.importorskip("pyspark")
        spark = MagicMock()
        publish_dashboard_views(spark, "alpha", "beta")
        joined = "\n".join(call.args[0] for call in spark.sql.call_args_list)
        assert "alpha.beta." in joined


class TestMaterializeHotViews:
    """The materialization pass at the end of publish_dashboard_views.

    The four hot-path views (v_account_primary_recommendation,
    v_account_explanation, v_account_feature_deviation,
    v_account_feature_deviation_topn) are the SQL bodies the Streamlit
    app re-executes on every L1->L4 click. The publisher snapshots each
    into a Delta table (Z-ORDERED on entity_id) and re-points the public
    view at the materialized table so per-click reads become indexed
    point lookups. These tests assert the CTAS + OPTIMIZE + re-point
    sequence happens and that failure modes degrade rather than crash.
    """

    def _sqls(self, spark) -> list[str]:
        return [call.args[0] for call in spark.sql.call_args_list]

    def test_emits_ctas_for_each_noncomposite_spec(self):
        pytest.importorskip("pyspark")
        spark = MagicMock()
        publish_dashboard_views(spark, "c", "s")
        sqls = self._sqls(spark)
        # Two non-composite specs: v_account_primary_recommendation and
        # v_account_explanation. Each emits a CREATE OR REPLACE TABLE ...
        # AS SELECT * FROM v_<name>.
        ctas = [s for s in sqls if "CREATE OR REPLACE TABLE c.s.dashboard_" in s]
        assert len(ctas) == 2
        assert any(
            "dashboard_account_primary_recommendation USING DELTA" in s
            and "FROM c.s.v_account_primary_recommendation" in s
            for s in ctas
        )
        assert any(
            "dashboard_account_explanation USING DELTA" in s
            and "FROM c.s.v_account_explanation" in s
            for s in ctas
        )

    def test_emits_optimize_zorder_on_entity_id(self):
        pytest.importorskip("pyspark")
        spark = MagicMock()
        publish_dashboard_views(spark, "c", "s")
        sqls = self._sqls(spark)
        zorders = [
            s for s in sqls
            if s.startswith("OPTIMIZE c.s.dashboard_") and "ZORDER BY" in s
        ]
        assert len(zorders) == 2
        for s in zorders:
            assert "(`entity_id`)" in s

    def test_repoints_view_at_materialized_table(self):
        pytest.importorskip("pyspark")
        spark = MagicMock()
        publish_dashboard_views(spark, "c", "s")
        sqls = self._sqls(spark)
        # The final pass re-creates the view as SELECT * FROM the table.
        repoints = [
            s for s in sqls
            if "CREATE OR REPLACE VIEW c.s.v_account_explanation AS"
               " SELECT * FROM c.s.dashboard_account_explanation" in s
            or "CREATE OR REPLACE VIEW c.s.v_account_primary_recommendation AS"
               " SELECT * FROM c.s.dashboard_account_primary_recommendation" in s
        ]
        assert len(repoints) == 2

    def test_emits_deviation_specs_when_composite_supplied(self):
        pytest.importorskip("pyspark")
        spark = MagicMock()

        class _F:
            def __init__(self, name, type_name):
                self.name = name
                self.dataType = type(type_name, (), {})()

        # Gold schema with one numeric column so the deviation block fires.
        spark.table.return_value.schema = [
            _F("entity_id", "StringType"),
            _F("feat_a", "DoubleType"),
        ]
        publish_dashboard_views(spark, "c", "s", composite_name="cn1")
        sqls = self._sqls(spark)
        # All four specs ran: two non-composite + two deviation.
        ctas = [s for s in sqls if "CREATE OR REPLACE TABLE c.s.dashboard_" in s]
        assert len(ctas) == 4
        assert any("dashboard_account_feature_deviation USING DELTA" in s for s in ctas)
        assert any("dashboard_account_feature_deviation_topn USING DELTA" in s for s in ctas)

    def test_skips_deviation_materialization_without_composite(self):
        pytest.importorskip("pyspark")
        spark = MagicMock()
        publish_dashboard_views(spark, "c", "s")
        sqls = self._sqls(spark)
        # Deviation tables never get materialized when no composite_name is in
        # effect (effective_composite is None on the no-composite-name path).
        assert not any("dashboard_account_feature_deviation" in s for s in sqls)

    def test_skips_deviation_materialization_when_prereqs_missing(self):
        # The publisher's effective_composite resolves to None when the
        # deviation prerequisites are absent (e.g. gold_features_cn1 does
        # not exist). In that case the deviation specs must be skipped at
        # the materialization pass too -- otherwise the CTAS would fail
        # against a view body the publisher already stripped out.
        pytest.importorskip("pyspark")
        spark = MagicMock()
        spark.catalog.tableExists.side_effect = lambda fqn: "gold_features_" not in fqn
        publish_dashboard_views(spark, "c", "s", composite_name="cn1")
        sqls = self._sqls(spark)
        assert not any("dashboard_account_feature_deviation" in s for s in sqls)
        # Non-composite specs still materialize.
        assert any("dashboard_account_explanation USING DELTA" in s for s in sqls)

    def test_optimize_failure_does_not_block_view_repoint(self):
        # Some Delta deployments do not support OPTIMIZE ZORDER. The
        # publisher must tolerate that failure: the table is still useful
        # (a snapshot of the view's body), and re-pointing the view at it
        # still wins back the per-click latency from CTE re-execution.
        pytest.importorskip("pyspark")
        spark = MagicMock()

        def _side_effect(stmt):
            if stmt.startswith("OPTIMIZE "):
                raise RuntimeError("OPTIMIZE not supported on this engine")
            return MagicMock()

        spark.sql.side_effect = _side_effect
        # No assertion error from publish itself; warnings logged for OPTIMIZE.
        publish_dashboard_views(spark, "c", "s")
        sqls = [call.args[0] for call in spark.sql.call_args_list]
        # CTAS still ran, view re-point still ran for both specs.
        ctas = [s for s in sqls if "CREATE OR REPLACE TABLE c.s.dashboard_" in s]
        repoints = [
            s for s in sqls
            if "CREATE OR REPLACE VIEW c.s.v_account_explanation AS"
               " SELECT * FROM c.s.dashboard_account_explanation" in s
            or "CREATE OR REPLACE VIEW c.s.v_account_primary_recommendation AS"
               " SELECT * FROM c.s.dashboard_account_primary_recommendation" in s
        ]
        assert len(ctas) == 2
        assert len(repoints) == 2

    def test_ctas_failure_does_not_block_remaining_specs(self):
        # If one CTAS errors out (e.g. transient permission issue on the
        # destination table), the publisher must still attempt the other
        # specs rather than fail the whole publish. Each spec is
        # independent: a failure on one leaves its view at the pre-call
        # body but the others still get materialized.
        pytest.importorskip("pyspark")
        spark = MagicMock()

        def _side_effect(stmt):
            if (
                stmt.startswith("CREATE OR REPLACE TABLE ")
                and "dashboard_account_primary_recommendation" in stmt
            ):
                raise RuntimeError("simulated CTAS failure")
            return MagicMock()

        spark.sql.side_effect = _side_effect
        publish_dashboard_views(spark, "c", "s")
        sqls = [call.args[0] for call in spark.sql.call_args_list]
        # v_account_explanation still materialized successfully.
        assert any(
            "CREATE OR REPLACE TABLE c.s.dashboard_account_explanation" in s
            for s in sqls
        )
        # v_account_primary_recommendation CTAS attempted (and failed) so
        # its OPTIMIZE / re-point are skipped (CTAS returns False before
        # the downstream steps).
        assert not any(
            "OPTIMIZE c.s.dashboard_account_primary_recommendation" in s
            for s in sqls
        )
