"""Tests for enrich_lifecycle_dataset.

Each test runs against pandas AND pyspark.pandas via the ``df_factory``
fixture parametrization in ``conftest.py``. The Spark-pandas variant is
skipped when pyspark is not importable.
"""
from __future__ import annotations

import logging

import pytest

from customer_retention.core.compat import native_pd
from customer_retention.stages.lifecycle import (
    LifecycleEnrichmentConfig,
    enrich_lifecycle_dataset,
)


def _base_config(**overrides) -> LifecycleEnrichmentConfig:
    base = dict(
        enriched_view_name="sps_enriched_contract",
        parent_entity_key="ACCOUNT_ID",
        sub_entity_key="CONTRACT_ID",
        valid_from_column="CONTRACT_START_DATE",
        valid_to_columns=("BILLING_TERMINATION_DATE",),
        status_column="CONTRACT_STATUS",
        terminal_status_values=("Cancelled", "Terminated", "Expired"),
        drop_columns=("BILLING_TERMINATION_DATE", "CONTRACT_END_DATE", "CONTRACT_STATUS", "TAKE_RATE"),
    )
    base.update(overrides)
    return LifecycleEnrichmentConfig(**base)


def _to_native(df) -> native_pd.DataFrame:
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return df


class TestEventGeneration:
    def test_active_contracts_emit_only_start_row(
        self, df_factory, synthetic_contract_records
    ):
        df = df_factory([r for r in synthetic_contract_records if r["ACCOUNT_ID"] == "A"])
        out = _to_native(enrich_lifecycle_dataset(df, config=_base_config()))
        assert len(out) == 10
        assert (out["event_type"] == "start").all()
        assert (out["ACCOUNT_ID"] == "A").all()

    def test_cancelled_contracts_emit_start_and_terminate(
        self, df_factory, synthetic_contract_records
    ):
        df = df_factory([r for r in synthetic_contract_records if r["ACCOUNT_ID"] == "B"])
        out = _to_native(enrich_lifecycle_dataset(df, config=_base_config()))
        assert len(out) == 17  # 10 starts + 7 terminates
        starts = out[out["event_type"] == "start"]
        terms = out[out["event_type"] == "terminate"]
        assert len(starts) == 10
        assert len(terms) == 7

    def test_status_gating_keeps_active_with_past_end_date(
        self, df_factory, synthetic_contract_records
    ):
        df = df_factory([r for r in synthetic_contract_records if r["ACCOUNT_ID"] == "C"])
        out = _to_native(enrich_lifecycle_dataset(df, config=_base_config()))
        assert len(out) == 1
        assert out["event_type"].iloc[0] == "start"

    def test_coalesce_fallback_when_btd_null(
        self, df_factory, synthetic_contract_records
    ):
        cfg = _base_config(
            valid_to_columns=("BILLING_TERMINATION_DATE", "CONTRACT_END_DATE"),
            drop_columns=("BILLING_TERMINATION_DATE", "CONTRACT_END_DATE", "CONTRACT_STATUS", "TAKE_RATE"),
        )
        df = df_factory([r for r in synthetic_contract_records if r["ACCOUNT_ID"] == "D"])
        out = _to_native(enrich_lifecycle_dataset(df, config=cfg))
        assert len(out) == 2  # 1 start + 1 terminate (via fallback)
        terminate = out[out["event_type"] == "terminate"]
        assert len(terminate) == 1
        # Terminate timestamp should equal CONTRACT_END_DATE = T0 + 60d
        ts = native_pd.Timestamp(terminate["event_timestamp"].iloc[0])
        expected = native_pd.Timestamp("2024-01-01") + native_pd.Timedelta(days=60)
        assert ts == expected


class TestCorruptRowPolicy:
    def test_corrupt_row_raise_default(self, df_factory, synthetic_contract_records):
        df = df_factory([r for r in synthetic_contract_records if r["ACCOUNT_ID"] == "E"])
        with pytest.raises(ValueError, match="inverted lifecycle"):
            enrich_lifecycle_dataset(df, config=_base_config(on_corrupt_row="raise"))

    def test_corrupt_row_skip_drops_with_warning(
        self, df_factory, synthetic_contract_records, caplog
    ):
        df = df_factory([r for r in synthetic_contract_records if r["ACCOUNT_ID"] == "E"])
        with caplog.at_level(logging.WARNING, logger="customer_retention.stages.lifecycle.enrich"):
            out = _to_native(
                enrich_lifecycle_dataset(df, config=_base_config(on_corrupt_row="skip"))
            )
        # Account E only has 1 row → with skip the inverted row is dropped
        # entirely, leaving 0 events.
        assert len(out) == 0
        assert any("inverted lifecycle" in rec.message for rec in caplog.records)

    def test_corrupt_row_warn_clamps_to_valid_from(
        self, df_factory, synthetic_contract_records, caplog
    ):
        df = df_factory([r for r in synthetic_contract_records if r["ACCOUNT_ID"] == "E"])
        with caplog.at_level(logging.WARNING, logger="customer_retention.stages.lifecycle.enrich"):
            out = _to_native(
                enrich_lifecycle_dataset(df, config=_base_config(on_corrupt_row="warn"))
            )
        # warn keeps the row: 1 start + 1 terminate clamped to valid_from
        assert len(out) == 2
        terminate = out[out["event_type"] == "terminate"]
        assert len(terminate) == 1
        start_ts = native_pd.Timestamp(out[out["event_type"] == "start"]["event_timestamp"].iloc[0])
        terminate_ts = native_pd.Timestamp(terminate["event_timestamp"].iloc[0])
        assert terminate_ts == start_ts
        assert any("inverted lifecycle" in rec.message for rec in caplog.records)


class TestSchema:
    def test_drop_columns_removed_from_output(
        self, df_factory, synthetic_contract_records
    ):
        df = df_factory(synthetic_contract_records[:11])  # Account A only (no inverted rows)
        out = _to_native(enrich_lifecycle_dataset(df, config=_base_config()))
        assert "BILLING_TERMINATION_DATE" not in out.columns
        assert "CONTRACT_STATUS" not in out.columns
        assert "TAKE_RATE" not in out.columns
        # valid_from_column is also dropped (geometry encodes it via event_timestamp)
        assert "CONTRACT_START_DATE" not in out.columns

    def test_event_type_and_timestamp_in_output(
        self, df_factory, synthetic_contract_records
    ):
        df = df_factory([r for r in synthetic_contract_records if r["ACCOUNT_ID"] == "B"])
        out = _to_native(enrich_lifecycle_dataset(df, config=_base_config()))
        assert "event_type" in out.columns
        assert "event_timestamp" in out.columns
        assert set(out["event_type"].unique()) == {"start", "terminate"}

    def test_preserved_attribute_columns(
        self, df_factory, synthetic_contract_records
    ):
        df = df_factory([r for r in synthetic_contract_records if r["ACCOUNT_ID"] == "B"])
        out = _to_native(enrich_lifecycle_dataset(df, config=_base_config()))
        assert "DOCUMENT_PLAN" in out.columns
        assert "CONTRACT_TERM" in out.columns
        # The same CONTRACT_ID has identical attribute values on START and TERMINATE
        sample_id = "B-00"
        rows = out[out["CONTRACT_ID"] == sample_id]
        assert len(rows) == 2  # one start, one terminate
        assert rows["DOCUMENT_PLAN"].nunique() == 1
        assert rows["CONTRACT_TERM"].nunique() == 1


class TestBackendDispatch:
    def test_native_pandas_routes_to_pandas_path(
        self, monkeypatch, synthetic_contract_records
    ):
        from customer_retention.stages.lifecycle import enrich as mod

        called = {}
        original_pandas = mod._enrich_pandas
        original_distributed = mod._enrich_distributed

        def stub_pandas(df, config):
            called["pandas"] = True
            return original_pandas(df, config)

        def stub_distributed(df, config):
            called["distributed"] = True
            return original_distributed(df, config)

        monkeypatch.setattr(mod, "_enrich_pandas", stub_pandas)
        monkeypatch.setattr(mod, "_enrich_distributed", stub_distributed)

        df = native_pd.DataFrame(
            [r for r in synthetic_contract_records if r["ACCOUNT_ID"] == "A"]
        )
        mod.enrich_lifecycle_dataset(df, config=_base_config())
        assert called == {"pandas": True}

    def test_pyspark_pandas_routes_to_distributed_path(self, monkeypatch):
        from customer_retention.stages.lifecycle import enrich as mod

        captured = {}

        def fake_is_spark_pandas(obj):
            return obj is fake_input

        def fake_as_spark_df(obj):
            return ("spark_df_for_", obj)

        def fake_distributed(spark_df, config):
            captured["distributed_called_with"] = spark_df
            return "spark_result"

        def fake_as_pandas_api(spark_df):
            captured["as_pandas_api_called_with"] = spark_df
            return "pyspark_pandas_result"

        monkeypatch.setattr(mod, "_is_spark_pandas", fake_is_spark_pandas)
        monkeypatch.setattr(mod, "_is_native_spark_df", lambda obj: False)
        monkeypatch.setattr(mod, "as_spark_df", fake_as_spark_df)
        monkeypatch.setattr(mod, "_enrich_distributed", fake_distributed)
        monkeypatch.setattr(mod, "as_pandas_api", fake_as_pandas_api)

        fake_input = object()
        result = mod.enrich_lifecycle_dataset(fake_input, config=_base_config())

        assert captured["distributed_called_with"] == ("spark_df_for_", fake_input)
        assert captured["as_pandas_api_called_with"] == "spark_result"
        assert result == "pyspark_pandas_result"

    def test_native_spark_routes_to_distributed_directly(self, monkeypatch):
        from customer_retention.stages.lifecycle import enrich as mod

        captured = {}

        def fake_distributed(spark_df, config):
            captured["distributed_called_with"] = spark_df
            return "spark_result"

        monkeypatch.setattr(mod, "_is_spark_pandas", lambda obj: False)
        monkeypatch.setattr(mod, "_is_native_spark_df", lambda obj: obj is fake_input)
        monkeypatch.setattr(mod, "_enrich_distributed", fake_distributed)
        monkeypatch.setattr(
            mod,
            "as_pandas_api",
            lambda obj: pytest.fail("must NOT call as_pandas_api on native spark input"),
        )

        fake_input = object()
        result = mod.enrich_lifecycle_dataset(fake_input, config=_base_config())
        assert result == "spark_result"
        assert captured["distributed_called_with"] is fake_input


class TestDistributedExpressionBuilders:
    """Validates the Spark column expressions built by the distributed
    helpers. ``pyspark.sql.functions`` works without an active Spark session
    so these tests verify the SQL semantics structurally without needing a
    cluster."""

    def test_build_effective_valid_to_status_gated_single_column(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.lifecycle.enrich import _build_effective_valid_to_expr

        class FakeDF:
            columns = ("BILLING_TERMINATION_DATE",)

        cfg = _base_config()
        expr = _build_effective_valid_to_expr(FakeDF(), cfg)
        s = str(expr)
        assert "CASE WHEN" in s
        assert "BILLING_TERMINATION_DATE" in s
        assert "CONTRACT_STATUS" in s
        assert "Cancelled" in s
        assert "Terminated" in s
        assert "Expired" in s

    def test_build_effective_valid_to_coalesces_multiple_columns(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.lifecycle.enrich import _build_effective_valid_to_expr

        class FakeDF:
            columns = ("BILLING_TERMINATION_DATE", "CONTRACT_END_DATE")

        cfg = _base_config(
            valid_to_columns=("BILLING_TERMINATION_DATE", "CONTRACT_END_DATE"),
            drop_columns=("BILLING_TERMINATION_DATE", "CONTRACT_END_DATE", "CONTRACT_STATUS", "TAKE_RATE"),
        )
        expr = _build_effective_valid_to_expr(FakeDF(), cfg)
        s = str(expr)
        assert "coalesce" in s
        assert "BILLING_TERMINATION_DATE" in s
        assert "CONTRACT_END_DATE" in s

    def test_build_effective_valid_to_no_status_gating(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.lifecycle.enrich import _build_effective_valid_to_expr

        class FakeDF:
            columns = ("BILLING_TERMINATION_DATE",)

        cfg = _base_config(
            status_column=None,
            terminal_status_values=(),
            drop_columns=(),
        )
        expr = _build_effective_valid_to_expr(FakeDF(), cfg)
        s = str(expr)
        assert "CASE WHEN" not in s  # no status gate
        assert "BILLING_TERMINATION_DATE" in s


class TestRegisterTempViewRoundTrip:
    def test_round_trip_to_temp_view_preserves_event_schema(
        self, synthetic_contract_records
    ):
        pytest.importorskip("pyspark")
        from pyspark.sql import SparkSession

        try:
            spark = SparkSession.builder.master("local[1]").getOrCreate()
        except Exception as exc:
            pytest.skip(f"local Spark session unavailable: {exc}")

        import pyspark.pandas as ps

        from customer_retention.core.compat import (
            as_spark_df,
            load_spark_table,
            register_temp_view,
        )

        records = [r for r in synthetic_contract_records if r["ACCOUNT_ID"] != "E"]
        df = ps.DataFrame(records)
        enriched = enrich_lifecycle_dataset(df, config=_base_config())

        view_name = "test_enriched_contract_round_trip"
        register_temp_view(as_spark_df(enriched), view_name)
        try:
            loaded = load_spark_table(f"global_temp.{view_name}")
            schema_names = {f.name for f in loaded.schema.fields}
            assert "ACCOUNT_ID" in schema_names
            assert "CONTRACT_ID" in schema_names
            assert "event_type" in schema_names
            assert "event_timestamp" in schema_names
            # Leakage / dropped columns must be absent
            assert "BILLING_TERMINATION_DATE" not in schema_names
            assert "CONTRACT_STATUS" not in schema_names
            assert "TAKE_RATE" not in schema_names
        finally:
            spark.catalog.dropGlobalTempView(view_name)


class TestParity:
    def test_pandas_and_spark_pandas_equivalence(self, synthetic_contract_records):
        pytest.importorskip("pyspark")
        from pyspark.sql import SparkSession

        try:
            SparkSession.builder.master("local[1]").getOrCreate()
        except Exception as exc:
            pytest.skip(f"local Spark session unavailable: {exc}")

        # Drop Account E so neither path raises
        records = [r for r in synthetic_contract_records if r["ACCOUNT_ID"] != "E"]
        cfg = _base_config()

        import pyspark.pandas as ps

        out_pd = _to_native(enrich_lifecycle_dataset(native_pd.DataFrame(records), config=cfg))
        out_ps = _to_native(enrich_lifecycle_dataset(ps.DataFrame(records), config=cfg))

        sort_cols = ["CONTRACT_ID", "event_type"]
        out_pd = out_pd.sort_values(sort_cols).reset_index(drop=True)
        out_ps = out_ps.sort_values(sort_cols).reset_index(drop=True)
        assert list(out_pd.columns) == list(out_ps.columns) or set(out_pd.columns) == set(out_ps.columns)
        assert len(out_pd) == len(out_ps)
        # Compare key columns row-wise
        for col in ["ACCOUNT_ID", "CONTRACT_ID", "event_type"]:
            assert list(out_pd[col]) == list(out_ps[col]), f"mismatch in {col}"
        # event_timestamp may differ in tz/precision representation; compare as Timestamp
        ts_pd = [native_pd.Timestamp(t) for t in out_pd["event_timestamp"]]
        ts_ps = [native_pd.Timestamp(t) for t in out_ps["event_timestamp"]]
        assert ts_pd == ts_ps
