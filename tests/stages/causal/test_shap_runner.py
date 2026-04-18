"""Unit tests for the distributed SHAP runner — attribution-driven path.

``compute_shap_distributed`` no longer scores the model inline. It consumes a
``ShapAttribution`` artifact produced at model-training time (see
``stages.modeling.shap_attribution``) and emits a single
``spark_df.select(*expressions)`` that runs parallel across all executor
cores with no shuffle and no FE round-trip.
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pyspark", reason="PySpark required for SHAP runner tests")

from customer_retention.stages.causal import shap_runner
from customer_retention.stages.causal.shap_runner import (
    EXPECTED_VALUE_COL,
    SHAP_PREFIX,
    ShapRunResult,
    compute_shap_distributed,
    unwrap_tree_model,
)
from customer_retention.stages.modeling.shap_attribution import ShapAttribution

# ---------------------------------------------------------------------------
# Shared stubs — Spark F / Column surrogate
# ---------------------------------------------------------------------------


class _RecordingExpr:
    """Expression stub that records build structure for expression-shape tests."""

    def __init__(
        self,
        kind: str,
        payload: Any = None,
        children: List["_RecordingExpr"] | None = None,
    ) -> None:
        self.kind = kind
        self.payload = payload
        self.children = children or []
        self.alias_name: str | None = None
        self.cast_to: str | None = None

    def alias(self, name: str) -> "_RecordingExpr":
        self.alias_name = name
        return self

    def cast(self, t: str) -> "_RecordingExpr":
        self.cast_to = t
        return self

    def __sub__(self, other: Any) -> "_RecordingExpr":
        return _RecordingExpr("sub", children=[self, _wrap(other)])

    def __mul__(self, other: Any) -> "_RecordingExpr":
        return _RecordingExpr("mul", children=[self, _wrap(other)])

    def __rmul__(self, other: Any) -> "_RecordingExpr":
        return _RecordingExpr("mul", children=[_wrap(other), self])


def _wrap(value: Any) -> "_RecordingExpr":
    return value if isinstance(value, _RecordingExpr) else _RecordingExpr("const", payload=value)


@pytest.fixture
def recording_functions(monkeypatch):
    import sys

    fake = MagicMock(name="pyspark.sql.functions")
    fake.col = lambda name: _RecordingExpr("col", payload=name)
    fake.lit = lambda value: _RecordingExpr("lit", payload=value)
    fake_sql = MagicMock(name="pyspark.sql")
    fake_sql.functions = fake
    monkeypatch.setitem(sys.modules, "pyspark.sql", fake_sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", fake)
    return fake


def _make_spark_df(columns: List[str]) -> Any:
    df = MagicMock(name="SparkDF")
    df.columns = list(columns)
    df.select = MagicMock(return_value=MagicMock(name="ShapDF"))
    return df


# ---------------------------------------------------------------------------
# Module-level constants + dataclass defaults
# ---------------------------------------------------------------------------


class TestConstants:
    def test_shap_prefix(self):
        assert SHAP_PREFIX == "shap_"

    def test_expected_value_col(self):
        assert EXPECTED_VALUE_COL == "shap_expected_value"

    def test_shap_run_result_defaults(self):
        r = ShapRunResult()
        assert r.shap_df is None
        assert r.feature_columns == []
        assert r.shap_columns == []
        assert r.background_size == 0


# ---------------------------------------------------------------------------
# Attribution-select builder — exercised directly to cover expression shape
# ---------------------------------------------------------------------------


class TestBuildAttributionSelect:
    def test_emits_join_key_plus_shap_columns_plus_expected_value(self, recording_functions):
        select_exprs, shap_cols = shap_runner._build_attribution_select(
            join_key="account_id",
            feature_order=["a", "b"],
            importances={"a": 0.6, "b": 0.4},
            means={"a": 10.0, "b": 20.0},
        )
        assert shap_cols == ["shap_a", "shap_b"]
        assert len(select_exprs) == 4
        assert select_exprs[0].kind == "col"
        assert select_exprs[0].payload == "account_id"
        assert select_exprs[-1].kind == "lit"
        assert select_exprs[-1].payload == 0.0
        assert select_exprs[-1].alias_name == EXPECTED_VALUE_COL

    def test_shap_formula_is_importance_times_deviation(self, recording_functions):
        select_exprs, _ = shap_runner._build_attribution_select(
            join_key="pk",
            feature_order=["x"],
            importances={"x": 0.7},
            means={"x": 5.0},
        )
        shap_expr = select_exprs[1]
        assert shap_expr.alias_name == "shap_x"
        assert shap_expr.kind == "mul"
        left, right = shap_expr.children
        assert left.kind == "sub"
        col_expr, mean_lit = left.children
        assert col_expr.kind == "col" and col_expr.payload == "x"
        assert mean_lit.kind == "lit" and mean_lit.payload == 5.0
        assert right.kind == "lit" and right.payload == 0.7

    def test_missing_importance_defaults_to_zero_attribution(self, recording_functions):
        """Defensive against a feature missing from ``importances``. Attribution
        becomes ``(x - mean) * 0 = 0`` — a sane degenerate output rather than a
        KeyError mid-pipeline."""
        select_exprs, _ = shap_runner._build_attribution_select(
            join_key="pk",
            feature_order=["x"],
            importances={},
            means={"x": 5.0},
        )
        shap_expr = select_exprs[1]
        right = shap_expr.children[1]
        assert right.kind == "lit" and right.payload == 0.0


# ---------------------------------------------------------------------------
# compute_shap_distributed — validation
# ---------------------------------------------------------------------------


class TestComputeShapDistributedValidation:
    def test_empty_attribution_feature_columns_raises(self, recording_functions):
        df = _make_spark_df(["account_id", "a"])
        with pytest.raises(ValueError, match="feature_columns is empty"):
            compute_shap_distributed(
                spark_df=df,
                attribution=ShapAttribution(),
            )

    def test_missing_join_key_raises(self, recording_functions):
        df = _make_spark_df(["a", "b"])
        with pytest.raises(ValueError, match="join_key"):
            compute_shap_distributed(
                spark_df=df,
                attribution=ShapAttribution(
                    importances={"a": 1.0},
                    background_means={"a": 0.0},
                    feature_columns=["a"],
                    sample_size=10,
                ),
            )

    def test_attribution_feature_missing_in_spark_df_raises(self, recording_functions):
        df = _make_spark_df(["account_id", "a"])
        with pytest.raises(ValueError, match="training/scoring schema drift"):
            compute_shap_distributed(
                spark_df=df,
                attribution=ShapAttribution(
                    importances={"a": 0.5, "missing": 0.5},
                    background_means={"a": 0.0, "missing": 0.0},
                    feature_columns=["a", "missing"],
                    sample_size=10,
                ),
            )


# ---------------------------------------------------------------------------
# compute_shap_distributed — orchestration
# ---------------------------------------------------------------------------


class TestComputeShapDistributedOrchestration:
    def test_happy_path_returns_shap_run_result(self, recording_functions):
        df = _make_spark_df(["account_id", "feat_a", "feat_b"])
        attribution = ShapAttribution(
            importances={"feat_a": 0.7, "feat_b": 0.3},
            background_means={"feat_a": 10.0, "feat_b": 5.0},
            feature_columns=["feat_a", "feat_b"],
            sample_size=1234,
        )
        result = compute_shap_distributed(spark_df=df, attribution=attribution)

        assert result.feature_columns == ["feat_a", "feat_b"]
        assert result.shap_columns == ["shap_feat_a", "shap_feat_b"]
        assert result.background_size == 1234
        assert result.shap_df is not None

    def test_emits_single_select_one_spark_job(self, recording_functions):
        """Stays distributed: emission is one ``spark_df.select(*expressions)``.
        No model scoring, no FE score_batch, no shuffle."""
        df = _make_spark_df(["account_id", "a"])
        attribution = ShapAttribution(
            importances={"a": 1.0},
            background_means={"a": 0.0},
            feature_columns=["a"],
            sample_size=10,
        )
        compute_shap_distributed(spark_df=df, attribution=attribution)
        df.select.assert_called_once()

    def test_no_fe_score_batch_on_hot_path(self, monkeypatch, recording_functions):
        """Regression: the hot path must not touch Databricks Feature
        Engineering. If the runner tries to import it, the test fails."""
        import sys

        fake_fe_module = MagicMock()
        fake_fe_module.FeatureEngineeringClient = MagicMock(
            side_effect=AssertionError("fe.score_batch must not run in the SHAP hot path")
        )
        fake_databricks = MagicMock()
        fake_databricks.feature_engineering = fake_fe_module
        monkeypatch.setitem(sys.modules, "databricks", fake_databricks)
        monkeypatch.setitem(sys.modules, "databricks.feature_engineering", fake_fe_module)

        df = _make_spark_df(["account_id", "a", "b"])
        attribution = ShapAttribution(
            importances={"a": 0.5, "b": 0.5},
            background_means={"a": 0.0, "b": 0.0},
            feature_columns=["a", "b"],
            sample_size=100,
        )
        compute_shap_distributed(spark_df=df, attribution=attribution)

    def test_background_size_threads_through_from_attribution(self, recording_functions):
        df = _make_spark_df(["account_id", "a"])
        attribution = ShapAttribution(
            importances={"a": 1.0},
            background_means={"a": 0.0},
            feature_columns=["a"],
            sample_size=9999,
        )
        result = compute_shap_distributed(spark_df=df, attribution=attribution)
        assert result.background_size == 9999


# ---------------------------------------------------------------------------
# unwrap_tree_model — public helper for non-causal callers
# ---------------------------------------------------------------------------


class TestUnwrapTreeModel:
    def test_returns_non_pyfunc_unchanged(self):
        """Anything that is not an ``mlflow.pyfunc.PyFuncModel`` round-trips
        untouched — including plain sklearn / spark estimators callers may
        pass during local diagnostics."""
        model = object()
        assert unwrap_tree_model(model) is model

    def test_returns_raw_model_from_get_raw_model(self, monkeypatch):
        import sys
        import types

        class _FakePyFuncModel:
            pass

        fake_pyfunc = types.ModuleType("mlflow.pyfunc")
        fake_pyfunc.PyFuncModel = _FakePyFuncModel
        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.pyfunc = fake_pyfunc
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.pyfunc", fake_pyfunc)

        raw = object()
        impl = MagicMock()
        impl.get_raw_model = MagicMock(return_value=raw)
        model = _FakePyFuncModel()
        model._model_impl = impl

        assert unwrap_tree_model(model) is raw

    def test_falls_through_to_raw_attrs_when_no_get_raw_model(self, monkeypatch):
        import sys
        import types

        class _FakePyFuncModel:
            pass

        fake_pyfunc = types.ModuleType("mlflow.pyfunc")
        fake_pyfunc.PyFuncModel = _FakePyFuncModel
        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.pyfunc = fake_pyfunc
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.pyfunc", fake_pyfunc)

        raw = object()

        class _Impl:
            sklearn_model = raw

        model = _FakePyFuncModel()
        model._model_impl = _Impl()

        assert unwrap_tree_model(model) is raw
