"""Structural tests for the Phase 3 deviation views.

The framework's deviation block is gated on a ``composite_name`` parameter:
when omitted the SQL is parseable as before, when supplied two extra views
appear and reference ``gold_features_{composite_name}``.
"""
from __future__ import annotations

import re
from unittest.mock import MagicMock

from customer_retention.stages.causal.dashboard_views import (
    DASHBOARD_DEVIATION_VIEW_NAMES,
    DASHBOARD_PROVENANCE_VIEW_NAMES,
    DASHBOARD_TEMPLATE_TABLE_NAMES,
    DASHBOARD_TEMPLATE_VIEW_NAMES,
    DASHBOARD_VIEW_NAMES,
    publish_dashboard_views,
    render_dashboard_view_sql,
    split_view_statements,
)

# Statements always emitted on top of the regular view set:
#   - DASHBOARD_PROVENANCE_VIEW_NAMES (gated at publish-time on prereqs;
#     the renderer always includes them)
#   - DASHBOARD_TEMPLATE_TABLE_NAMES (CREATE TABLE -- always emitted)
#   - DASHBOARD_TEMPLATE_VIEW_NAMES  (CREATE VIEW  -- always emitted)
_BASE_NON_DEVIATION_STATEMENTS = (
    len(DASHBOARD_VIEW_NAMES)
    + len(DASHBOARD_PROVENANCE_VIEW_NAMES)
    + len(DASHBOARD_TEMPLATE_TABLE_NAMES)
    + len(DASHBOARD_TEMPLATE_VIEW_NAMES)
)
# Subset that becomes a CREATE OR REPLACE VIEW (everything except the
# template table's CREATE TABLE).
_BASE_NON_DEVIATION_VIEW_CALLS = (
    len(DASHBOARD_VIEW_NAMES)
    + len(DASHBOARD_PROVENANCE_VIEW_NAMES)
    + len(DASHBOARD_TEMPLATE_VIEW_NAMES)
)


def _block(rendered: str, view_name: str) -> str:
    statements = re.split(r";\s*(?=CREATE OR REPLACE VIEW)", rendered)
    matches = [s for s in statements if view_name in s]
    assert matches, f"view {view_name!r} missing from rendered SQL"
    return matches[0]


def test_render_without_composite_name_excludes_deviation_views():
    rendered = render_dashboard_view_sql("c", "s")
    for name in DASHBOARD_DEVIATION_VIEW_NAMES:
        assert name not in rendered
    assert "{composite_name}" not in rendered
    assert "@cr:deviation-block" not in rendered


def test_render_with_composite_name_includes_deviation_views():
    rendered = render_dashboard_view_sql("c", "s", composite_name="cn1")
    for name in DASHBOARD_DEVIATION_VIEW_NAMES:
        assert name in rendered
    assert "{composite_name}" not in rendered
    assert "@cr:deviation-block" not in rendered
    assert "gold_features_cn1" in rendered


def test_render_substitutes_catalog_and_schema_with_composite():
    rendered = render_dashboard_view_sql("alpha", "beta", composite_name="cn1")
    assert "alpha.beta." in rendered
    assert "{catalog}" not in rendered
    assert "{schema}" not in rendered


def test_statement_count_matches_expected_with_and_without():
    # Provenance view is included in render output by default; the publisher
    # is what gates it on prerequisite-table existence. The template-store
    # CREATE TABLE + v_dashboard_template_active CREATE VIEW always render.
    assert len(split_view_statements(render_dashboard_view_sql("c", "s"))) == _BASE_NON_DEVIATION_STATEMENTS
    assert len(split_view_statements(render_dashboard_view_sql(
        "c", "s", composite_name="cn1"
    ))) == _BASE_NON_DEVIATION_STATEMENTS + len(DASHBOARD_DEVIATION_VIEW_NAMES)


def test_deviation_view_computes_z_and_pct_dev():
    block = _block(
        render_dashboard_view_sql("c", "s", composite_name="cn1"),
        "v_account_feature_deviation",
    )
    assert "(g.feature_value - p.mean) / p.stddev" in block
    assert "(g.feature_value - p.mean) / p.mean" in block
    assert "p.stddev IS NOT NULL" in block
    assert "ABS(p.mean) > 1e-9" in block


def test_deviation_view_uses_latest_population_stats():
    block = _block(
        render_dashboard_view_sql("c", "s", composite_name="cn1"),
        "v_account_feature_deviation",
    )
    assert "ROW_NUMBER()" in block
    assert "ORDER BY computed_at DESC" in block
    assert "dtype = 'numeric'" in block


def test_deviation_view_explodes_gold_features_into_long_form():
    # When ``gold_struct_cols`` is supplied, the renderer substitutes the
    # explicit numeric column list into ``STRUCT(...)`` so FROM_JSON's
    # MAP<STRING,DOUBLE> parse stays strict (any non-double would NULL the
    # whole map and silently zero the view).
    block = _block(
        render_dashboard_view_sql(
            "c", "s",
            composite_name="cn1",
            gold_struct_cols=["feat_a", "feat_b"],
        ),
        "v_account_feature_deviation",
    )
    assert "FROM_JSON(TO_JSON(STRUCT(`feat_a`, `feat_b`))" in block
    # ``STRUCT(*)`` only appears in explanatory comments, never in
    # executable DDL (which would re-introduce the FROM_JSON-NULL trap).
    code_only = "\n".join(line for line in block.splitlines()
                          if not line.lstrip().startswith("--"))
    assert "STRUCT(*)" not in code_only
    assert "LATERAL VIEW EXPLODE" in block


def test_deviation_view_struct_args_default_to_dummy_when_no_cols():
    # ``render_dashboard_view_sql`` should never emit invalid SQL even when
    # the caller forgets to pass ``gold_struct_cols`` -- substitute ``1`` so
    # the view DDL stays parseable. The publisher catches this case earlier
    # (logs a warning + skips deviation) but the renderer must not crash.
    block = _block(
        render_dashboard_view_sql("c", "s", composite_name="cn1"),
        "v_account_feature_deviation",
    )
    assert "STRUCT(1)" in block


def test_deviation_topn_caps_at_12():
    block = _block(
        render_dashboard_view_sql("c", "s", composite_name="cn1"),
        "v_account_feature_deviation_topn",
    )
    assert "ABS(COALESCE(d.z, 0.0)) DESC" in block
    assert "deviation_rank <= 12" in block


def _spark_with_numeric_gold():
    """MagicMock spark whose gold_features schema reports two DoubleType
    columns plus an entity_id (string) so ``_gold_numeric_columns`` returns
    a non-empty list and the publisher emits the deviation block."""
    spark = MagicMock()

    class _F:
        def __init__(self, name, type_name):
            self.name = name
            self.dataType = type(type_name, (), {})()

    fields = [_F("entity_id", "StringType"),
              _F("feat_a", "DoubleType"),
              _F("feat_b", "DoubleType")]
    spark.table.return_value.schema = fields
    return spark


def test_publish_with_composite_name_runs_extra_statements():
    spark = _spark_with_numeric_gold()
    publish_dashboard_views(spark, "c", "s", composite_name="cn1")
    # MagicMock's tableExists returns truthy for every prereq, so the publisher
    # includes both the deviation block and the provenance view, plus the
    # always-on template-store view.
    expected = _BASE_NON_DEVIATION_VIEW_CALLS + len(DASHBOARD_DEVIATION_VIEW_NAMES)
    view_calls = [
        call for call in spark.sql.call_args_list
        if "CREATE OR REPLACE VIEW" in call.args[0]
    ]
    assert len(view_calls) == expected
    submitted = "\n".join(call.args[0] for call in view_calls)
    # The numeric-only struct projection landed in the rendered DDL --
    # without this, FROM_JSON would NULL the whole map and the view would
    # silently return zero rows on every dashboard page-hit.
    assert "STRUCT(`feat_a`, `feat_b`)" in submitted
    # ``STRUCT(*)`` only appears in the explanatory CTE comment, not in
    # executable DDL.
    code_only = "\n".join(line for line in submitted.splitlines()
                          if not line.lstrip().startswith("--"))
    assert "STRUCT(*)" not in code_only


def test_publish_without_composite_name_keeps_original_count():
    spark = MagicMock()
    publish_dashboard_views(spark, "c", "s")
    view_calls = [
        call for call in spark.sql.call_args_list
        if "CREATE OR REPLACE VIEW" in call.args[0]
    ]
    assert len(view_calls) == _BASE_NON_DEVIATION_VIEW_CALLS


def test_publish_skips_deviation_when_gold_has_no_numeric_columns():
    # Defensive: if the gold table exists but its schema is metadata-only
    # (entity_id, as_of_date), STRUCT(...) would be empty and FROM_JSON
    # would return an empty map. Publisher detects this and skips the
    # deviation block instead of emitting a view that silently produces 0
    # rows on every dashboard page-hit.
    spark = MagicMock()

    class _F:
        def __init__(self, name, type_name):
            self.name = name
            self.dataType = type(type_name, (), {})()

    spark.table.return_value.schema = [
        _F("entity_id", "StringType"),
        _F("as_of_date", "TimestampType"),
    ]

    publish_dashboard_views(spark, "c", "s", composite_name="cn1")
    view_calls = [
        call for call in spark.sql.call_args_list
        if "CREATE OR REPLACE VIEW" in call.args[0]
    ]
    submitted = "\n".join(call.args[0] for call in view_calls)
    # Deviation views absent because gold had no numerics to project.
    for name in DASHBOARD_DEVIATION_VIEW_NAMES:
        assert name not in submitted


def test_publish_skips_deviation_when_population_stats_table_missing():
    # Cluster shape where ``feature_population_stats`` lives as a JSON
    # sidecar on Volume rather than as a UC Delta table. The deviation
    # views reference the UC table and would crash with
    # ``[TABLE_OR_VIEW_NOT_FOUND]`` at CREATE time. The publisher must
    # detect the missing prerequisite and silently skip just the deviation
    # block, leaving the rest of the dashboard publishable.
    spark = MagicMock()

    def _exists(fqn):
        return "feature_population_stats" not in fqn

    spark.catalog.tableExists.side_effect = _exists

    publish_dashboard_views(spark, "c", "s", composite_name="cn1")
    view_calls = [
        call for call in spark.sql.call_args_list
        if "CREATE OR REPLACE VIEW" in call.args[0]
    ]
    submitted = "\n".join(call.args[0] for call in view_calls)
    # Non-deviation views still publish; provenance + template-store views
    # pass the side_effect filter so they ship.
    assert len(view_calls) == _BASE_NON_DEVIATION_VIEW_CALLS
    # Neither deviation view body reaches Spark.
    for name in DASHBOARD_DEVIATION_VIEW_NAMES:
        assert name not in submitted


def test_publish_skips_deviation_when_gold_features_table_missing():
    # Same skip behaviour when the per-run gold table is absent (e.g.
    # composite_name was supplied but the gold step never ran).
    spark = MagicMock()

    def _exists(fqn):
        return "gold_features_" not in fqn

    spark.catalog.tableExists.side_effect = _exists

    publish_dashboard_views(spark, "c", "s", composite_name="cn1")
    view_calls = [
        call for call in spark.sql.call_args_list
        if "CREATE OR REPLACE VIEW" in call.args[0]
    ]
    submitted = "\n".join(call.args[0] for call in view_calls)
    assert len(view_calls) == _BASE_NON_DEVIATION_VIEW_CALLS
    for name in DASHBOARD_DEVIATION_VIEW_NAMES:
        assert name not in submitted
