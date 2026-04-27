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
    assert len(split_view_statements(render_dashboard_view_sql("c", "s"))) == len(
        DASHBOARD_VIEW_NAMES
    )
    assert len(split_view_statements(render_dashboard_view_sql(
        "c", "s", composite_name="cn1"
    ))) == len(DASHBOARD_VIEW_NAMES) + len(DASHBOARD_DEVIATION_VIEW_NAMES)


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
    expected = len(DASHBOARD_VIEW_NAMES) + len(DASHBOARD_DEVIATION_VIEW_NAMES)
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
    assert len(view_calls) == len(DASHBOARD_VIEW_NAMES)
