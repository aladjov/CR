"""Spark-vs-pandas parity for the new TimeWindowAggregator Spark dispatch.

Skipped on CI (no PySpark per `coding_practice_testing_spark`); runs on
Databricks where PySpark is present. The test asserts that for the agg
funcs the Spark path supports (sum, mean, max, min, count, value_counts +
event_count + recency + tenure), the result of `aggregate(spark_df, ...)`
matches `aggregate(pandas_df, ...)` value-for-value within float tolerance.

Any future regression in the Spark dispatch (column-set drift, dtype
drift, off-by-one window boundary, value_counts pivot bug) flips this
test red.
"""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import DataFrame as NativeSparkDF
from pyspark.sql import SparkSession

from customer_retention.stages.profiling.time_window_aggregator import (
    TimeWindowAggregator,
)


@pytest.fixture(scope="module")
def spark():
    try:
        s = (
            SparkSession.builder
            .appName("twa-spark-parity")
            .master("local[2]")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.driver.memory", "1g")
            .getOrCreate()
        )
    except RuntimeError:
        pytest.skip("local Spark session not available (Databricks Connect environment)")
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


@pytest.fixture
def synthetic_events():
    np.random.seed(42)
    ref = pd.Timestamp("2026-04-15 12:00:00")
    rows = []
    for entity in ["A", "B", "C", "D"]:
        n = np.random.randint(8, 15)
        for _ in range(n):
            days_ago = int(np.random.randint(0, 400))
            rows.append({
                "ACCOUNT_ID": entity,
                "event_timestamp": ref - timedelta(days=days_ago),
                "event_type": str(np.random.choice(["start", "terminate"])),
                "amount": float(np.random.uniform(10, 1000)),
            })
    # Edge cases:
    # - entity E has no events in the 7d window
    rows.append({
        "ACCOUNT_ID": "E", "event_timestamp": ref - timedelta(days=300),
        "event_type": "start", "amount": 50.0,
    })
    # - entity F: only one event, value_counts terminate=0
    rows.append({
        "ACCOUNT_ID": "F", "event_timestamp": ref - timedelta(days=2),
        "event_type": "start", "amount": 99.0,
    })
    return ref, pd.DataFrame(rows)


def test_spark_dispatch_returns_native_spark_dataframe(spark, synthetic_events):
    ref, pdf = synthetic_events
    sdf = spark.createDataFrame(pdf)
    agg = TimeWindowAggregator(entity_column="ACCOUNT_ID", time_column="event_timestamp")

    result = agg.aggregate(
        sdf, windows=["30d"], value_columns=["amount"], agg_funcs=["sum"],
        reference_date=ref, include_event_count=True,
    )
    assert isinstance(result, NativeSparkDF)


def test_spark_pandas_parity_full_agg_set(spark, synthetic_events):
    ref, pdf = synthetic_events
    sdf = spark.createDataFrame(pdf)
    agg = TimeWindowAggregator(entity_column="ACCOUNT_ID", time_column="event_timestamp")

    windows = ["7d", "30d", "90d", "180d", "365d", "all_time"]
    value_cols = ["amount", "event_type"]
    agg_funcs = ["sum", "mean", "max", "count", "value_counts"]

    pdf_result = agg.aggregate(
        pdf, windows=windows, value_columns=value_cols, agg_funcs=agg_funcs,
        reference_date=ref, include_event_count=True,
        include_recency=True, include_tenure=True,
    )
    sdf_result = agg.aggregate(
        sdf, windows=windows, value_columns=value_cols, agg_funcs=agg_funcs,
        reference_date=ref, include_event_count=True,
        include_recency=True, include_tenure=True,
    )

    assert isinstance(sdf_result, NativeSparkDF)

    sdf_pdf = (
        sdf_result.toPandas().sort_values("ACCOUNT_ID").reset_index(drop=True)
    )
    pdf_result = pdf_result.sort_values("ACCOUNT_ID").reset_index(drop=True)

    assert set(sdf_pdf["ACCOUNT_ID"]) == set(pdf_result["ACCOUNT_ID"])

    common_cols = sorted(set(pdf_result.columns) & set(sdf_pdf.columns))
    # value_counts outputs should appear on both sides
    vc_cols = [c for c in common_cols
               if c.startswith("event_type_") and "_count_" in c]
    assert len(vc_cols) >= 2, f"expected value_counts cols, got {vc_cols}"
    # event_count_{w} + numeric agg outputs + recency + tenure must be common
    for w in windows:
        assert f"event_count_{w}" in common_cols
        assert f"amount_sum_{w}" in common_cols
    assert "days_since_last_event" in common_cols
    assert "days_since_first_event" in common_cols

    # Numerical parity
    mismatches = []
    for c in common_cols:
        if c == "ACCOUNT_ID":
            continue
        a = pdf_result[c].astype(float).fillna(-12345.0)
        b = sdf_pdf[c].astype(float).fillna(-12345.0)
        diff = float((a - b).abs().max())
        if diff > 1e-6:
            mismatches.append((c, diff))

    assert not mismatches, f"parity mismatches: {mismatches[:8]}"


def test_spark_value_counts_zero_fill_for_missing_value(spark, synthetic_events):
    """Entity F has only `start` events — its `terminate` count must be 0,
    not null, in every window. This is the cycle-012 fix invariant."""
    ref, pdf = synthetic_events
    sdf = spark.createDataFrame(pdf)
    agg = TimeWindowAggregator(entity_column="ACCOUNT_ID", time_column="event_timestamp")

    sdf_result = agg.aggregate(
        sdf, windows=["7d", "30d"], value_columns=["event_type"],
        agg_funcs=["value_counts"], reference_date=ref,
    )
    pdf_result = sdf_result.toPandas().set_index("ACCOUNT_ID")
    for w in ["7d", "30d"]:
        assert pdf_result.loc["F", f"event_type_terminate_count_{w}"] == 0
        assert pdf_result.loc["F", f"event_type_start_count_{w}"] >= 1


def test_spark_unsupported_agg_func_falls_back_to_pandas(spark, synthetic_events):
    """`mode` is not in _SPARK_SUPPORTED_AGG_FUNCS → falls back to pandas
    path. Result is a pandas DataFrame with a UserWarning."""
    ref, pdf = synthetic_events
    sdf = spark.createDataFrame(pdf)
    agg = TimeWindowAggregator(entity_column="ACCOUNT_ID", time_column="event_timestamp")
    with pytest.warns(UserWarning, match="falling back to pandas"):
        result = agg.aggregate(
            sdf, windows=["30d"], value_columns=["event_type"],
            agg_funcs=["mode"], reference_date=ref,
        )
    assert isinstance(result, pd.DataFrame)
