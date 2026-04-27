"""Tests for the generic dashboard-profile-override helper."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from customer_retention.stages.causal.dashboard_profile_override import (
    ProfileOverrideResult,
    apply_profile_override,
    render_profile_sql,
)


def test_render_profile_sql_substitutes_catalog_and_schema():
    out = render_profile_sql(
        "SELECT * FROM {catalog}.{schema}.foo",
        catalog="cat", schema="sch",
    )
    assert out == "SELECT * FROM cat.sch.foo"


def test_render_profile_sql_substitutes_composite_name_when_supplied():
    out = render_profile_sql(
        "SELECT * FROM {catalog}.{schema}.gold_features_{composite_name}",
        catalog="c", schema="s", composite_name="cn1",
    )
    assert "gold_features_cn1" in out
    assert "{composite_name}" not in out


def test_render_profile_sql_leaves_placeholder_when_no_composite():
    out = render_profile_sql(
        "SELECT * FROM {catalog}.{schema}.gold_features_{composite_name}",
        catalog="c", schema="s",
    )
    assert "{composite_name}" in out


def test_apply_publishes_each_statement_in_sql(tmp_path):
    spark = MagicMock()
    sql = (
        "CREATE OR REPLACE VIEW {catalog}.{schema}.v_account_profile AS SELECT 1;\n"
        "CREATE OR REPLACE VIEW {catalog}.{schema}.v_account_profile_extra AS SELECT 2;\n"
    )
    template_path = tmp_path / "profile.html"
    res = apply_profile_override(
        spark, "cat", "sch",
        profile_sql=sql,
        profile_html="<article></article>",
        template_volume_path=str(template_path),
    )
    assert spark.sql.call_count == 2
    submitted = "\n".join(call.args[0] for call in spark.sql.call_args_list)
    assert "cat.sch.v_account_profile" in submitted
    assert "cat.sch.v_account_profile_extra" in submitted
    assert template_path.read_text() == "<article></article>"
    assert isinstance(res, ProfileOverrideResult)
    assert res.published_views == [
        "cat.sch.v_account_profile",
        "cat.sch.v_account_profile_extra",
    ]
    assert "CR_PROFILE_TEMPLATE_PATH=" in res.env_hint


def test_apply_writes_html_verbatim(tmp_path):
    spark = MagicMock()
    html = "---\ndata: {}\n---\n<article>{{entity_id}}</article>\n"
    template_path = tmp_path / "x.html"
    apply_profile_override(
        spark, "c", "s",
        profile_sql="CREATE OR REPLACE VIEW {catalog}.{schema}.v AS SELECT 1;",
        profile_html=html,
        template_volume_path=str(template_path),
    )
    assert template_path.read_text() == html


def test_apply_passes_composite_name_through_to_sql(tmp_path):
    spark = MagicMock()
    sql = "CREATE OR REPLACE VIEW {catalog}.{schema}.v AS SELECT * FROM {catalog}.{schema}.gold_features_{composite_name};"
    apply_profile_override(
        spark, "c", "s",
        profile_sql=sql,
        profile_html="x",
        template_volume_path=str(tmp_path / "t.html"),
        composite_name="cust_emai",
    )
    submitted = spark.sql.call_args_list[0].args[0]
    assert "gold_features_cust_emai" in submitted
    assert "{composite_name}" not in submitted


def test_apply_raises_when_composite_referenced_but_missing(tmp_path):
    spark = MagicMock()
    sql = "CREATE OR REPLACE VIEW {catalog}.{schema}.v AS SELECT * FROM {catalog}.{schema}.gold_features_{composite_name};"
    with pytest.raises(ValueError, match="composite_name"):
        apply_profile_override(
            spark, "c", "s",
            profile_sql=sql,
            profile_html="x",
            template_volume_path=str(tmp_path / "t.html"),
        )
    spark.sql.assert_not_called()


def test_apply_raises_when_template_parent_missing(tmp_path):
    spark = MagicMock()
    nonexistent_parent = tmp_path / "no" / "such" / "dir" / "t.html"
    with pytest.raises(FileNotFoundError, match="parent directory"):
        apply_profile_override(
            spark, "c", "s",
            profile_sql="CREATE OR REPLACE VIEW {catalog}.{schema}.v AS SELECT 1;",
            profile_html="x",
            template_volume_path=str(nonexistent_parent),
        )


def test_apply_raises_when_sql_has_no_statements(tmp_path):
    spark = MagicMock()
    with pytest.raises(ValueError, match="no executable statements"):
        apply_profile_override(
            spark, "c", "s",
            profile_sql="-- only comments\n;",
            profile_html="x",
            template_volume_path=str(tmp_path / "t.html"),
        )


def test_apply_is_no_op_when_spark_is_none(tmp_path):
    res = apply_profile_override(
        None, "c", "s",
        profile_sql="CREATE OR REPLACE VIEW {catalog}.{schema}.v AS SELECT 1;",
        profile_html="x",
        template_volume_path=str(tmp_path / "t.html"),
    )
    assert res.published_views == []
    assert res.template_path == ""
    assert not (tmp_path / "t.html").exists()


def test_result_str_lists_views_and_env_hint(tmp_path):
    spark = MagicMock()
    res = apply_profile_override(
        spark, "c", "s",
        profile_sql=(
            "CREATE OR REPLACE VIEW {catalog}.{schema}.v_a AS SELECT 1;\n"
            "CREATE OR REPLACE VIEW {catalog}.{schema}.v_b AS SELECT 2;\n"
        ),
        profile_html="x",
        template_volume_path=str(tmp_path / "t.html"),
    )
    rendered = str(res)
    assert "c.s.v_a" in rendered
    assert "c.s.v_b" in rendered
    assert "CR_PROFILE_TEMPLATE_PATH=" in rendered
