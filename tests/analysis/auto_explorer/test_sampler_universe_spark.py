"""Regression test for the subquery-safe filter evaluator.

Reproduces the exact shape that caused the SPS engagement to return a
~4%-sized universe instead of ~19.5%: `F.expr` silently evaluates
`IN (SELECT ...)` to all-true in column-expression context. The fix
routes such predicates through `spark.sql()` which handles subqueries
correctly.

Requires a live Spark session — skipped on CI.
"""
from __future__ import annotations

import pytest

pyspark = pytest.importorskip("pyspark")
pytestmark = pytest.mark.spark

from pyspark.sql import SparkSession  # noqa: E402

from customer_retention.analysis.auto_explorer.sampling import (  # noqa: E402
    SegmentEntitySelection,
    _filter_has_subquery,
    _spark_passing_entities_sql,
    compute_sampling_universe,
)
from customer_retention.core.compat.spark_backend import _as_pandas_api  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    return (
        SparkSession.builder.master("local[*]")
        .appName("sampler_universe_subquery")
        .getOrCreate()
    )


def test_filter_has_subquery_detection():
    assert _filter_has_subquery("x in (select y from t)")
    assert _filter_has_subquery("x IN (SELECT y FROM t WHERE z=1)")
    assert _filter_has_subquery("a = 1 AND b IN (select c from d)")
    assert not _filter_has_subquery("seg in ('a', 'b')")
    assert not _filter_has_subquery("x = 5 AND y > 10")


def test_spark_sql_evaluator_handles_subquery(spark):
    accounts = spark.createDataFrame(
        [("A1", "Emerging"), ("A2", "Small"), ("A3", "Mid-Market"),
         ("A4", "Emerging"), ("A5", "Small")],
        ["ACCOUNT_ID", "REVENUE_MARKET_SEGMENT"],
    )
    contracts = spark.createDataFrame(
        [("A1", "start"), ("A2", "start"), ("A2", "terminate"),
         ("A3", "start"), ("A4", "terminate")],
        ["ACCOUNT_ID", "event_type"],
    )
    accounts.createOrReplaceTempView("account")
    contracts.createOrReplaceTempView("contract")

    query = (
        "REVENUE_MARKET_SEGMENT IN ('Emerging', 'Small') "
        "AND ACCOUNT_ID IN (SELECT ACCOUNT_ID FROM contract WHERE event_type='start')"
    )
    result = _spark_passing_entities_sql("account", query, "ACCOUNT_ID")
    ids = {row["ACCOUNT_ID"] for row in result.collect()}
    assert ids == {"A1", "A2"}


def test_compute_sampling_universe_routes_subquery_through_sql(spark):
    accounts_rows = [
        ("A1", "Emerging"), ("A2", "Small"), ("A3", "Mid-Market"),
        ("A4", "Emerging"), ("A5", "Small"), ("A6", "Enterprise"),
    ]
    contracts_rows = [
        ("A1", "start"), ("A2", "start"), ("A4", "terminate"), ("A6", "start"),
    ]
    accounts = spark.createDataFrame(accounts_rows, ["ACCOUNT_ID", "REVENUE_MARKET_SEGMENT"])
    contracts = spark.createDataFrame(contracts_rows, ["ACCOUNT_ID", "event_type"])

    frames = {"account": _as_pandas_api(accounts), "contract": _as_pandas_api(contracts)}
    entity_columns = {"account": "ACCOUNT_ID", "contract": "ACCOUNT_ID"}
    filters = {
        "account": (
            "REVENUE_MARKET_SEGMENT IN ('Emerging', 'Small') "
            "AND ACCOUNT_ID IN (SELECT ACCOUNT_ID FROM contract WHERE event_type='start')"
        ),
    }

    universe = compute_sampling_universe(
        frames=frames,
        entity_columns=entity_columns,
        primary_entity_dataset="account",
        filters=filters,
    )
    assert isinstance(universe, SegmentEntitySelection)

    collected_ids = {
        row[0] for row in universe._as_spark_df().collect()
    } if universe.is_distributed else set(universe)

    assert collected_ids == {"A1", "A2"}
    assert "A6" not in collected_ids
    assert "A3" not in collected_ids


def test_simple_predicate_still_uses_fexpr_path(spark):
    # Non-subquery predicate should work the same way; this guards against
    # accidentally regressing the simple case when touching the router.
    accounts = spark.createDataFrame(
        [("A1", "Emerging"), ("A2", "Enterprise"), ("A3", "Small")],
        ["ACCOUNT_ID", "REVENUE_MARKET_SEGMENT"],
    )
    frames = {"account": _as_pandas_api(accounts)}
    entity_columns = {"account": "ACCOUNT_ID"}
    filters = {"account": "REVENUE_MARKET_SEGMENT in ('Emerging', 'Small')"}

    universe = compute_sampling_universe(
        frames=frames,
        entity_columns=entity_columns,
        primary_entity_dataset="account",
        filters=filters,
    )
    collected = {row[0] for row in universe._as_spark_df().collect()} \
        if universe.is_distributed else set(universe)
    assert collected == {"A1", "A3"}
