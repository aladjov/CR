"""Tests for the generic dashboard-profile-override helper.

The override path used to write the HTML body to a Volume FUSE path; it
now appends a row into ``{catalog}.{schema}.dashboard_template_overrides``
(a Delta table the dashboard view publisher creates at the same time as
``v_dashboard_template_active``). The Streamlit App reads via SQL so it
no longer touches a Volume.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from customer_retention.stages.causal.dashboard_profile_override import (
    ProfileOverrideResult,
    apply_profile_override,
    render_profile_sql,
)
from customer_retention.stages.causal.dashboard_views import MaterializedViewSpec


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


def test_render_profile_sql_substitutes_arbitrary_placeholders():
    out = render_profile_sql(
        "SELECT * FROM delta.`{volume_run_data}/silver/silver_merged`",
        catalog="c", schema="s",
        placeholders={"volume_run_data": "/Volumes/x/y/runs/r1/data"},
    )
    assert "/Volumes/x/y/runs/r1/data/silver/silver_merged" in out
    assert "{volume_run_data}" not in out


def test_render_profile_sql_rejects_reserved_placeholder_keys():
    with pytest.raises(ValueError, match="reserved key"):
        render_profile_sql(
            "SELECT 1",
            catalog="c", schema="s",
            placeholders={"catalog": "evil"},
        )
    with pytest.raises(ValueError, match="reserved key"):
        render_profile_sql(
            "SELECT 1",
            catalog="c", schema="s",
            placeholders={"composite_name": "evil"},
        )


def _spark_mock_with_writer():
    """A Spark mock whose ``createDataFrame(...).write.mode("append").saveAsTable(...)``
    chain captures the args without exploding. Returned alongside the writer
    leaf so individual tests can assert on what was appended.
    """
    spark = MagicMock()
    appender = MagicMock()
    spark.createDataFrame.return_value.write.mode.return_value.saveAsTable.side_effect = (
        appender
    )
    # current_user() row -- shape: row.asDict-style with a "u" attribute.
    user_row = MagicMock()
    user_row.__getitem__.side_effect = lambda key: "tester@example.com" if key == "u" else None
    spark.sql.return_value.first.return_value = user_row
    return spark, appender


def test_apply_passes_placeholders_through():
    pytest.importorskip("pyspark")
    spark, _ = _spark_mock_with_writer()
    sql = "CREATE OR REPLACE VIEW {catalog}.{schema}.v AS SELECT * FROM delta.`{vol}/silver/silver_merged`;"
    apply_profile_override(
        spark, "c", "s",
        profile_sql=sql,
        profile_html="x",
        composite_name="cn1",
        placeholders={"vol": "/Volumes/foo/bar/runs/r/data"},
    )
    # The first spark.sql() call publishes a view (current_user() is also
    # called via spark.sql(...).first() and has separate args).
    submitted = [c.args[0] for c in spark.sql.call_args_list if c.args]
    view_call = next(s for s in submitted if "/silver_merged" in s)
    assert "/Volumes/foo/bar/runs/r/data/silver/silver_merged" in view_call
    assert "{vol}" not in view_call


def test_render_profile_sql_leaves_placeholder_when_no_composite():
    out = render_profile_sql(
        "SELECT * FROM {catalog}.{schema}.gold_features_{composite_name}",
        catalog="c", schema="s",
    )
    assert "{composite_name}" in out


def test_apply_publishes_each_statement_in_sql():
    pytest.importorskip("pyspark")
    spark, _ = _spark_mock_with_writer()
    sql = (
        "CREATE OR REPLACE VIEW {catalog}.{schema}.v_account_profile AS SELECT 1;\n"
        "CREATE OR REPLACE VIEW {catalog}.{schema}.v_account_profile_extra AS SELECT 2;\n"
    )
    res = apply_profile_override(
        spark, "cat", "sch",
        profile_sql=sql,
        profile_html="<article></article>",
        composite_name="cust_emai",
    )
    submitted = [c.args[0] for c in spark.sql.call_args_list if c.args]
    publish_calls = [s for s in submitted if "v_account_profile" in s]
    assert any("cat.sch.v_account_profile" in s for s in publish_calls)
    assert any("cat.sch.v_account_profile_extra" in s for s in publish_calls)
    assert isinstance(res, ProfileOverrideResult)
    assert res.published_views == [
        "cat.sch.v_account_profile",
        "cat.sch.v_account_profile_extra",
    ]
    assert res.template_table_fqn == "cat.sch.dashboard_template_overrides"
    assert res.composite_name == "cust_emai"


def test_apply_appends_html_row_to_uc_table():
    pytest.importorskip("pyspark")
    spark, _ = _spark_mock_with_writer()
    html = "---\ndata: {}\n---\n<article>{{entity_id}}</article>\n"
    apply_profile_override(
        spark, "c", "s",
        profile_sql="CREATE OR REPLACE VIEW {catalog}.{schema}.v AS SELECT 1;",
        profile_html=html,
        composite_name="cn1",
    )
    # The HTML body must appear verbatim as a column value passed to
    # spark.createDataFrame(...). Search the call's positional args (a list
    # of Row objects) for the html string.
    create_call = spark.createDataFrame.call_args
    rows = create_call.args[0]
    assert any(getattr(r, "profile_html", None) == html for r in rows)
    # And the chain must end at saveAsTable on the template-overrides FQN.
    save_call = spark.createDataFrame.return_value.write.mode.return_value.saveAsTable.call_args
    assert save_call.args[0] == "c.s.dashboard_template_overrides"


def test_apply_passes_composite_name_through_to_sql():
    pytest.importorskip("pyspark")
    spark, _ = _spark_mock_with_writer()
    sql = "CREATE OR REPLACE VIEW {catalog}.{schema}.v AS SELECT * FROM {catalog}.{schema}.gold_features_{composite_name};"
    apply_profile_override(
        spark, "c", "s",
        profile_sql=sql,
        profile_html="x",
        composite_name="cust_emai",
    )
    submitted = [c.args[0] for c in spark.sql.call_args_list if c.args]
    view_call = next(s for s in submitted if "gold_features_" in s)
    assert "gold_features_cust_emai" in view_call
    assert "{composite_name}" not in view_call


def test_apply_raises_when_composite_name_missing():
    spark = MagicMock()
    with pytest.raises(ValueError, match="composite_name"):
        apply_profile_override(
            spark, "c", "s",
            profile_sql="CREATE OR REPLACE VIEW {catalog}.{schema}.v AS SELECT 1;",
            profile_html="x",
            composite_name="",
        )
    spark.sql.assert_not_called()


def test_apply_raises_when_sql_has_no_statements():
    spark = MagicMock()
    with pytest.raises(ValueError, match="no executable statements"):
        apply_profile_override(
            spark, "c", "s",
            profile_sql="-- only comments\n;",
            profile_html="x",
            composite_name="cn1",
        )


def test_apply_is_no_op_when_spark_is_none():
    res = apply_profile_override(
        None, "c", "s",
        profile_sql="CREATE OR REPLACE VIEW {catalog}.{schema}.v AS SELECT 1;",
        profile_html="x",
        composite_name="cn1",
    )
    assert res.published_views == []
    assert res.template_table_fqn == ""


def test_apply_runs_ctas_and_zorder_for_materialize_views():
    """When ``materialize_views`` is supplied, apply_profile_override must run
    the CTAS + OPTIMIZE ZORDER + view re-point sequence for each spec so the
    L4 per-account lookups become point reads instead of multi-join scans."""
    pytest.importorskip("pyspark")
    spark, _ = _spark_mock_with_writer()
    res = apply_profile_override(
        spark, "cat", "sch",
        profile_sql="CREATE OR REPLACE VIEW {catalog}.{schema}.v_account_profile_sps AS SELECT 1;",
        profile_html="x",
        composite_name="cn1",
        materialize_views=[
            MaterializedViewSpec(
                view_name="v_account_profile_sps",
                table_name="dashboard_account_profile_sps",
                zorder_col="entity_id",
                requires_composite=False,
            )
        ],
    )
    submitted = [c.args[0] for c in spark.sql.call_args_list if c.args]
    assert any(
        "CREATE OR REPLACE TABLE cat.sch.dashboard_account_profile_sps" in s
        and "FROM cat.sch.v_account_profile_sps" in s
        for s in submitted
    ), f"CTAS not submitted; got: {submitted}"
    assert any(
        "OPTIMIZE cat.sch.dashboard_account_profile_sps" in s
        and "ZORDER BY (`entity_id`)" in s
        for s in submitted
    ), f"OPTIMIZE ZORDER not submitted; got: {submitted}"
    assert any(
        "CREATE OR REPLACE VIEW cat.sch.v_account_profile_sps AS SELECT * FROM cat.sch.dashboard_account_profile_sps"
        in s for s in submitted
    ), f"view re-point not submitted; got: {submitted}"
    assert res.materialized_views == ["v_account_profile_sps"]


def test_apply_with_no_materialize_views_kwarg_returns_empty_list():
    """Default path: no materialization requested → result.materialized_views=[]
    and no CTAS/OPTIMIZE statements are submitted."""
    pytest.importorskip("pyspark")
    spark, _ = _spark_mock_with_writer()
    res = apply_profile_override(
        spark, "c", "s",
        profile_sql="CREATE OR REPLACE VIEW {catalog}.{schema}.v AS SELECT 1;",
        profile_html="x",
        composite_name="cn1",
    )
    submitted = [c.args[0] for c in spark.sql.call_args_list if c.args]
    assert not any("CREATE OR REPLACE TABLE" in s for s in submitted)
    assert not any("OPTIMIZE" in s for s in submitted)
    assert res.materialized_views == []


def test_result_str_lists_views_and_uc_table():
    pytest.importorskip("pyspark")
    spark, _ = _spark_mock_with_writer()
    res = apply_profile_override(
        spark, "c", "s",
        profile_sql=(
            "CREATE OR REPLACE VIEW {catalog}.{schema}.v_a AS SELECT 1;\n"
            "CREATE OR REPLACE VIEW {catalog}.{schema}.v_b AS SELECT 2;\n"
        ),
        profile_html="x",
        composite_name="cn1",
    )
    rendered = str(res)
    assert "c.s.v_a" in rendered
    assert "c.s.v_b" in rendered
    # Result summary references the UC table, not a volume path.
    assert "c.s.dashboard_template_overrides" in rendered
    assert "CR_PROFILE_TEMPLATE_PATH" not in rendered
