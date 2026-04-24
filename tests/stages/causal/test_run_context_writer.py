"""Tests for ``run_context_writer`` — schema projection and config builder."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from customer_retention.stages.causal.run_context_writer import (
    RunContextConfig,
    _enum_value,
    _field_type_ddl,
    _schema_to_ddl,
    from_project_context,
)
from customer_retention.stages.causal.schemas import run_context_schema


class _FakeEnum:
    """Stand-in for Pydantic string-enum values (``obj.value`` is the payload)."""

    def __init__(self, value: str):
        self.value = value


def _project_context_stub(**overrides) -> SimpleNamespace:
    intent = SimpleNamespace(prediction_horizons=[30, 60, 90])
    datasets = {"accounts": SimpleNamespace(), "events": SimpleNamespace()}
    defaults = dict(
        intent=intent,
        primary_objective=_FakeEnum("immediate_risk"),
        temporal_posture=_FakeEnum("short_memory"),
        project_name="demo",
        run_id="run-abc",
        target_dataset="accounts",
        entity_column="entity_id",
        datasets=datasets,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestFromProjectContext:
    def test_pulls_horizon_objective_posture_from_context(self):
        spark = MagicMock()
        ctx = _project_context_stub()
        cfg = from_project_context(
            project_context=ctx,
            spark=spark,
            table_fqn="c.s.run_context",
            scoring_run_id="run-1",
            as_of_date=datetime(2026, 4, 24, tzinfo=timezone.utc),
            model_name="churn_classifier",
            model_version="12",
            model_type="xgboost",
        )
        assert cfg.horizon_days == 30
        assert cfg.prediction_horizons == [30, 60, 90]
        assert cfg.primary_objective == "immediate_risk"
        assert cfg.temporal_posture == "short_memory"
        assert cfg.project_name == "demo"
        assert cfg.project_run_id == "run-abc"
        assert cfg.target_dataset == "accounts"
        assert cfg.entity_column == "entity_id"
        assert cfg.dataset_names == ["accounts", "events"]
        assert cfg.model_type == "xgboost"

    def test_tolerates_missing_project_context(self):
        spark = MagicMock()
        cfg = from_project_context(
            project_context=None,
            spark=spark,
            table_fqn="c.s.run_context",
            scoring_run_id="run-2",
            as_of_date=datetime(2026, 4, 24, tzinfo=timezone.utc),
            model_name="churn_classifier",
            model_version="12",
        )
        assert cfg.horizon_days is None
        assert cfg.primary_objective is None
        assert cfg.temporal_posture is None
        assert cfg.dataset_names is None
        # Required model fields still populated
        assert cfg.model_name == "churn_classifier"
        assert cfg.model_version == "12"

    def test_empty_intent_does_not_raise(self):
        spark = MagicMock()
        ctx = _project_context_stub(intent=None)
        cfg = from_project_context(
            project_context=ctx,
            spark=spark,
            table_fqn="c.s.run_context",
            scoring_run_id="run-3",
            as_of_date=datetime(2026, 4, 24, tzinfo=timezone.utc),
            model_name="churn_classifier",
            model_version="12",
        )
        assert cfg.horizon_days is None
        assert cfg.prediction_horizons is None
        # Other context fields still populated
        assert cfg.primary_objective == "immediate_risk"

    def test_returns_runcontextconfig_instance(self):
        spark = MagicMock()
        cfg = from_project_context(
            project_context=None,
            spark=spark,
            table_fqn="c.s.run_context",
            scoring_run_id="run-4",
            as_of_date=datetime(2026, 4, 24, tzinfo=timezone.utc),
            model_name="m",
            model_version="1",
        )
        assert isinstance(cfg, RunContextConfig)
        assert cfg.spark is spark
        assert cfg.table_fqn == "c.s.run_context"


class TestEnumValue:
    def test_string_enum_returns_value_attr(self):
        assert _enum_value(_FakeEnum("short_memory")) == "short_memory"

    def test_plain_string_returns_itself(self):
        assert _enum_value("immediate_risk") == "immediate_risk"

    def test_none_returns_none(self):
        assert _enum_value(None) is None


class TestSchemaToDdl:
    def test_contains_every_run_context_column(self):
        ddl = _schema_to_ddl(run_context_schema())
        for required in (
            "scoring_run_id STRING",
            "as_of_date TIMESTAMP",
            "model_name STRING",
            "horizon_days INT",
            "prediction_horizons ARRAY<INT>",
            "primary_objective STRING",
            "temporal_posture STRING",
            "dataset_names ARRAY<STRING>",
        ):
            assert required in ddl, f"missing {required!r} in DDL: {ddl}"

    def test_field_type_ddl_handles_array_of_string(self):
        from pyspark.sql.types import ArrayType, StringType
        assert _field_type_ddl(ArrayType(StringType())) == "ARRAY<STRING>"

    def test_field_type_ddl_handles_int(self):
        from pyspark.sql.types import IntegerType
        assert _field_type_ddl(IntegerType()) == "INT"
