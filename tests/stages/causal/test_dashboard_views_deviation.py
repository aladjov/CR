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
    DASHBOARD_VIEW_NAMES,
    publish_dashboard_views,
    render_dashboard_view_sql,
    split_view_statements,
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
    # is what gates it on prerequisite-table existence.
    base_with_prov = len(DASHBOARD_VIEW_NAMES) + len(DASHBOARD_PROVENANCE_VIEW_NAMES)
    assert len(split_view_statements(render_dashboard_view_sql("c", "s"))) == base_with_prov
    assert len(split_view_statements(render_dashboard_view_sql(
        "c", "s", composite_name="cn1"
    ))) == base_with_prov + len(DASHBOARD_DEVIATION_VIEW_NAMES)


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
    block = _block(
        render_dashboard_view_sql("c", "s", composite_name="cn1"),
        "v_account_feature_deviation",
    )
    assert "FROM_JSON(TO_JSON(STRUCT(*))" in block
    assert "LATERAL VIEW EXPLODE" in block
    assert "kv_key NOT IN ('entity_id', 'as_of_date')" in block


def test_deviation_topn_caps_at_12():
    block = _block(
        render_dashboard_view_sql("c", "s", composite_name="cn1"),
        "v_account_feature_deviation_topn",
    )
    assert "ABS(COALESCE(d.z, 0.0)) DESC" in block
    assert "deviation_rank <= 12" in block


def test_publish_with_composite_name_runs_extra_statements():
    spark = MagicMock()
    publish_dashboard_views(spark, "c", "s", composite_name="cn1")
    # MagicMock's tableExists returns truthy for every prereq, so the publisher
    # includes both the deviation block and the provenance view.
    expected = (
        len(DASHBOARD_VIEW_NAMES)
        + len(DASHBOARD_DEVIATION_VIEW_NAMES)
        + len(DASHBOARD_PROVENANCE_VIEW_NAMES)
    )
    view_calls = [
        call for call in spark.sql.call_args_list
        if "CREATE OR REPLACE VIEW" in call.args[0]
    ]
    assert len(view_calls) == expected


def test_publish_without_composite_name_keeps_original_count():
    spark = MagicMock()
    publish_dashboard_views(spark, "c", "s")
    view_calls = [
        call for call in spark.sql.call_args_list
        if "CREATE OR REPLACE VIEW" in call.args[0]
    ]
    assert len(view_calls) == len(DASHBOARD_VIEW_NAMES) + len(
        DASHBOARD_PROVENANCE_VIEW_NAMES
    )


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
    # Non-deviation views still publish; provenance prereqs (feature_meta /
    # column_descriptions) pass the side_effect filter so its view ships too.
    assert len(view_calls) == len(DASHBOARD_VIEW_NAMES) + len(
        DASHBOARD_PROVENANCE_VIEW_NAMES
    )
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
    assert len(view_calls) == len(DASHBOARD_VIEW_NAMES) + len(
        DASHBOARD_PROVENANCE_VIEW_NAMES
    )
    for name in DASHBOARD_DEVIATION_VIEW_NAMES:
        assert name not in submitted
