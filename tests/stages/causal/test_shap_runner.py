"""Unit tests for the distributed SHAP runner.

The runner now computes SHAP as distributed linear attribution
(``importance × (x − background_mean)``) using pure Spark SQL — no
``pandas_udf``, no model pickling, no ``SparkContext`` capture. Tests
target:

1. Dataclass defaults and public-surface invariants.
2. ``freeze_background`` sampling helpers.
3. ``_resolve_feature_order`` — PipelineModel assembler, sklearn
   ``feature_names_in_``, caller fallback, missing-column fail-fast.
4. ``_extract_importances`` — Spark ML tree / LR, sklearn tree / linear,
   absent-attribute fail-fast, length-mismatch fail-fast.
5. ``_background_means`` — driver rows vs batched distributed fallback.
6. ``_build_attribution_select`` — Spark SQL expression shape.
7. ``compute_shap_distributed`` — orchestration + regression guards
   (no ``pandas_udf`` invoked, no model pickling).
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pyspark", reason="PySpark required for SHAP runner tests")

from customer_retention.stages.causal import shap_runner
from customer_retention.stages.causal.shap_runner import (
    DEFAULT_BACKGROUND_SIZE,
    DEFAULT_BATCH_SIZE,
    EXPECTED_VALUE_COL,
    SHAP_PREFIX,
    BackgroundSample,
    ShapRunResult,
    compute_shap_distributed,
    freeze_background,
    unwrap_tree_model,
)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


class _Aliasable:
    """Column-like stub used for row access and chaining in mock Spark DFs.

    Supports the PySpark ``Column`` operator surface we exercise in
    expression-builder tests: ``alias``, ``cast``, ``__eq__`` / ``__ne__``
    (returning Column-like, not bool), boolean ops ``|`` / ``&`` / ``~``,
    so mask conditions like ``mask_entry | (col == lit)`` do not trip
    over native Python booleans.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def alias(self, alias: str) -> "_Aliasable":
        return _Aliasable(alias)

    def cast(self, _t: str) -> "_Aliasable":
        return self

    def __eq__(self, other: Any) -> "_Aliasable":  # type: ignore[override]
        return _Aliasable(f"({self.name}=={other})")

    def __ne__(self, other: Any) -> "_Aliasable":  # type: ignore[override]
        return _Aliasable(f"({self.name}!={other})")

    def __or__(self, other: Any) -> "_Aliasable":
        return _Aliasable(f"({self.name}|{other})")

    def __and__(self, other: Any) -> "_Aliasable":
        return _Aliasable(f"({self.name}&{other})")

    def __invert__(self) -> "_Aliasable":
        return _Aliasable(f"~({self.name})")

    def __neg__(self) -> "_Aliasable":
        return _Aliasable(f"-({self.name})")

    def __gt__(self, other: Any) -> "_Aliasable":
        return _Aliasable(f"({self.name}>{other})")

    def __lt__(self, other: Any) -> "_Aliasable":
        return _Aliasable(f"({self.name}<{other})")

    def otherwise(self, other: Any) -> "_Aliasable":
        return _Aliasable(f"otherwise({self.name},{other})")

    def when(self, _cond: Any, _value: Any) -> "_Aliasable":
        return self

    def __truediv__(self, other: Any) -> "_Aliasable":
        return _Aliasable(f"({self.name}/{other})")

    def __sub__(self, other: Any) -> "_Aliasable":
        return _Aliasable(f"({self.name}-{other})")

    def __add__(self, other: Any) -> "_Aliasable":
        return _Aliasable(f"({self.name}+{other})")

    def __mul__(self, other: Any) -> "_Aliasable":
        return _Aliasable(f"({self.name}*{other})")

    def __hash__(self) -> int:
        return id(self)


class _RecordingExpr:
    """Column-expression stub that records how it was built.

    Supports the builder shape Spark's ``Column`` exposes (``alias``,
    ``cast``, arithmetic) so tests can reconstruct exactly how
    ``_build_attribution_select`` composed each expression without
    needing a live Spark session.
    """

    def __init__(self, kind: str, payload: Any = None, children: List["_RecordingExpr"] | None = None) -> None:
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


def _install_recording_functions(monkeypatch):
    import sys

    fake_functions = MagicMock(name="pyspark.sql.functions")
    fake_functions.col = lambda name: _RecordingExpr("col", payload=name)
    fake_functions.lit = lambda value: _RecordingExpr("lit", payload=value)
    fake_functions.mean = lambda col: _RecordingExpr("mean", children=[_wrap(col)])
    fake_functions.count = MagicMock(side_effect=lambda c: _Aliasable("__c"))
    fake_functions.struct = MagicMock(side_effect=lambda *cs: _Aliasable("struct"))
    fake_sql = MagicMock(name="pyspark.sql")
    fake_sql.functions = fake_functions
    monkeypatch.setitem(sys.modules, "pyspark.sql", fake_sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", fake_functions)
    return fake_functions


# ---------------------------------------------------------------------------
# Defaults + dataclasses
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_background_size(self):
        assert DEFAULT_BACKGROUND_SIZE == 1000

    def test_default_batch_size_constant_kept_for_back_compat(self):
        # DerivationConfig still exposes this; the runner no longer uses it
        # internally but the constant stays exported so downstream imports
        # keep working without a coordinated rename.
        assert DEFAULT_BATCH_SIZE == 10_000

    def test_shap_prefix(self):
        assert SHAP_PREFIX == "shap_"

    def test_expected_value_col(self):
        assert EXPECTED_VALUE_COL == "shap_expected_value"

    def test_background_sample_defaults(self):
        s = BackgroundSample()
        assert s.rows == []
        assert s.feature_columns == []
        assert s.target_column is None
        assert s.sample_size == 0

    def test_shap_run_result_defaults(self):
        r = ShapRunResult()
        assert r.shap_df is None
        assert r.feature_columns == []
        assert r.shap_columns == []
        assert r.background_size == 0


# ---------------------------------------------------------------------------
# freeze_background
# ---------------------------------------------------------------------------


class _FakeRow:
    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def asDict(self, recursive: bool = False) -> dict:  # noqa: N802
        return dict(self._data)


def _make_select_chain(rows: list[dict]) -> Any:
    chain = MagicMock(name="SelectChain")
    chain.limit.return_value = chain
    chain.collect.return_value = [_FakeRow(r) for r in rows]
    return chain


def _make_fake_spark_df(rows: list[dict], columns: list[str]) -> Any:
    df = MagicMock(name="SparkDF")
    df.columns = list(columns)
    df.count.return_value = len(rows)
    df.select.return_value = _make_select_chain(rows)
    df.sample.return_value = df
    df.sampleBy.return_value = df
    grouped = MagicMock(name="GroupBy")
    agg_result = MagicMock(name="AggResult")
    agg_result.collect.return_value = [
        _FakeRow({"target": 0, "__c": 60}),
        _FakeRow({"target": 1, "__c": 40}),
    ]
    grouped.agg.return_value = agg_result
    df.groupBy.return_value = grouped
    return df


@pytest.fixture
def recording_functions(monkeypatch):
    return _install_recording_functions(monkeypatch)


class TestFreezeBackground:
    def test_empty_feature_columns_raises(self):
        df = _make_fake_spark_df([], [])
        with pytest.raises(ValueError, match="at least one feature column"):
            freeze_background(df, feature_columns=[])

    def test_uniform_sample_returns_bounded_rows(self, recording_functions):
        rows = [{"a": float(i), "b": float(i + 1)} for i in range(50)]
        df = _make_fake_spark_df(rows, ["a", "b"])
        sample = freeze_background(df, feature_columns=["a", "b"], n=10)
        assert sample.feature_columns == ["a", "b"]
        assert sample.target_column is None
        assert sample.sample_size == 50

    def test_stratified_sample_uses_sampleby(self, recording_functions):
        rows = [{"a": float(i), "b": float(i + 1), "target": i % 2} for i in range(100)]
        df = _make_fake_spark_df(rows, ["a", "b", "target"])
        sample = freeze_background(df, feature_columns=["a", "b"], target_col="target", n=20)
        assert sample.target_column == "target"
        df.sampleBy.assert_called_once()

    def test_target_col_not_in_columns_falls_back_to_uniform(self, recording_functions):
        rows = [{"a": float(i)} for i in range(20)]
        df = _make_fake_spark_df(rows, ["a"])
        sample = freeze_background(df, feature_columns=["a"], target_col="missing_target", n=10)
        df.sampleBy.assert_not_called()
        assert sample.target_column is None


# ---------------------------------------------------------------------------
# Feature-order resolution
# ---------------------------------------------------------------------------


class _FakeAssembler:
    def __init__(self, input_cols: List[str]) -> None:
        self._input_cols = input_cols

    def getInputCols(self) -> List[str]:  # noqa: N802
        return list(self._input_cols)


def _pipeline_model(stages: List[Any]) -> Any:
    pm = MagicMock(name="PipelineModel")
    pm.stages = list(stages)
    return pm


class TestResolveFeatureOrder:
    def test_spark_pipeline_uses_assembler_input_cols(self):
        classifier = MagicMock(name="GBT", spec=["featureImportances"])
        classifier.featureImportances.toArray.return_value = [0.3, 0.7]
        model = _pipeline_model([_FakeAssembler(["a", "b"]), classifier])

        resolved = shap_runner._resolve_feature_order(
            model, caller_feature_columns=["a", "b", "c"], spark_columns=["a", "b", "c"]
        )
        assert resolved == ["a", "b"]

    def test_raw_spark_classifier_uses_caller_feature_columns(self):
        classifier = MagicMock(name="LR", spec=["coefficients"])
        resolved = shap_runner._resolve_feature_order(
            classifier, caller_feature_columns=["a", "b"], spark_columns=["a", "b"]
        )
        assert resolved == ["a", "b"]

    def test_sklearn_feature_names_in_preferred_over_caller(self):
        import numpy as np

        model = MagicMock(name="SklearnRF", spec=["feature_importances_", "feature_names_in_"])
        model.feature_names_in_ = np.array(["a", "b"])
        resolved = shap_runner._resolve_feature_order(
            model, caller_feature_columns=["a", "b", "c"], spark_columns=["a", "b", "c"]
        )
        assert resolved == ["a", "b"]

    def test_missing_column_in_spark_df_fails_fast(self):
        classifier = MagicMock(name="GBT", spec=["featureImportances"])
        model = _pipeline_model([_FakeAssembler(["a", "d"]), classifier])
        with pytest.raises(ValueError, match=r"feature columns not in spark_df\.columns.*\['d'\]"):
            shap_runner._resolve_feature_order(
                model, caller_feature_columns=["a", "d"], spark_columns=["a", "b", "c"]
            )


# ---------------------------------------------------------------------------
# Importance extraction
# ---------------------------------------------------------------------------


class TestExtractImportances:
    def test_tree_feature_importances_returned_positive(self):
        cl = MagicMock(name="GBT", spec=["featureImportances"])
        cl.featureImportances.toArray.return_value = [0.3, 0.5, 0.2]
        out = shap_runner._extract_importances(cl, feature_count=3)
        assert out == [0.3, 0.5, 0.2]

    def test_linear_coefficients_absolute_value(self):
        cl = MagicMock(name="LR", spec=["coefficients"])
        cl.coefficients.toArray.return_value = [-0.5, 0.3]
        out = shap_runner._extract_importances(cl, feature_count=2)
        assert out == [0.5, 0.3]

    def test_sklearn_feature_importances_(self):
        cl = MagicMock(name="SklearnRF", spec=["feature_importances_"])
        cl.feature_importances_ = [0.1, 0.4, 0.5]
        out = shap_runner._extract_importances(cl, feature_count=3)
        assert out == [0.1, 0.4, 0.5]

    def test_sklearn_coef_2d_takes_first_row(self):
        import numpy as np

        cl = MagicMock(name="SklearnLR", spec=["coef_"])
        cl.coef_ = np.array([[-0.2, 0.4, -0.1]])
        out = shap_runner._extract_importances(cl, feature_count=3)
        assert out == [0.2, 0.4, 0.1]

    def test_sklearn_coef_1d(self):
        import numpy as np

        cl = MagicMock(name="SklearnLinear", spec=["coef_"])
        cl.coef_ = np.array([-0.7, 0.2])
        out = shap_runner._extract_importances(cl, feature_count=2)
        assert out == [0.7, 0.2]

    def test_pipeline_model_uses_last_stage_classifier(self):
        cl = MagicMock(name="GBT", spec=["featureImportances"])
        cl.featureImportances.toArray.return_value = [0.6, 0.4]
        model = _pipeline_model([_FakeAssembler(["a", "b"]), cl])
        out = shap_runner._extract_importances(model, feature_count=2)
        assert out == [0.6, 0.4]

    def test_model_without_any_importance_attribute_fails_fast(self):
        cl = MagicMock(name="Mystery", spec=[])
        with pytest.raises(ValueError, match="featureImportances"):
            shap_runner._extract_importances(cl, feature_count=2)

    def test_length_mismatch_fails_fast(self):
        cl = MagicMock(name="GBT", spec=["featureImportances"])
        cl.featureImportances.toArray.return_value = [0.3, 0.5, 0.2]
        with pytest.raises(ValueError, match=r"length 3 does not match feature count 2"):
            shap_runner._extract_importances(cl, feature_count=2)


# ---------------------------------------------------------------------------
# Background means
# ---------------------------------------------------------------------------


class TestBackgroundMeans:
    def test_driver_means_from_background_rows(self):
        bg = BackgroundSample(
            rows=[{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}],
            feature_columns=["a", "b"],
            sample_size=2,
        )
        out = shap_runner._background_means(bg, ["a", "b"], spark_df=None)
        assert out == {"a": 2.0, "b": 3.0}

    def test_driver_means_missing_column_defaults_to_zero(self):
        bg = BackgroundSample(
            rows=[{"a": 1.0}, {"a": 3.0}],
            feature_columns=["a"],
            sample_size=2,
        )
        out = shap_runner._background_means(bg, ["a", "b"], spark_df=None)
        assert out == {"a": 2.0, "b": 0.0}

    def test_driver_means_skips_none_values(self):
        bg = BackgroundSample(
            rows=[{"a": 1.0}, {"a": None}, {"a": 3.0}],
            feature_columns=["a"],
            sample_size=3,
        )
        out = shap_runner._background_means(bg, ["a"], spark_df=None)
        assert out == {"a": 2.0}

    def test_empty_background_triggers_distributed_fallback_in_batches(self, recording_functions):
        bg = BackgroundSample(rows=[], feature_columns=[], sample_size=0)

        features = [f"f{i}" for i in range(250)]
        agg_calls: list[list[_RecordingExpr]] = []

        class _FakeHeadRow:
            def __getitem__(self, key: str) -> float:
                return 1.5

        spark_df = MagicMock(name="SparkDF")

        def _agg_capture(*exprs):
            agg_calls.append(list(exprs))
            result = MagicMock()
            result.head.return_value = _FakeHeadRow()
            return result

        spark_df.agg = _agg_capture
        out = shap_runner._spark_fallback_means(spark_df, features, batch_size=200)
        assert len(agg_calls) == 2
        assert len(agg_calls[0]) == 200
        assert len(agg_calls[1]) == 50
        assert len(out) == 250
        assert all(v == 1.5 for v in out.values())

    def test_fallback_null_mean_falls_back_to_zero(self, recording_functions):
        class _NullRow:
            def __getitem__(self, key: str):
                return None

        spark_df = MagicMock(name="SparkDF")
        agg_result = MagicMock()
        agg_result.head.return_value = _NullRow()
        spark_df.agg.return_value = agg_result
        out = shap_runner._spark_fallback_means(spark_df, ["a", "b"], batch_size=200)
        assert out == {"a": 0.0, "b": 0.0}


# ---------------------------------------------------------------------------
# Attribution select builder
# ---------------------------------------------------------------------------


class TestBuildAttributionSelect:
    def test_emits_join_key_plus_shap_columns_plus_expected_value(self, recording_functions):
        select_exprs, shap_cols = shap_runner._build_attribution_select(
            join_key="account_id",
            feature_order=["a", "b"],
            importances=[0.6, 0.4],
            means={"a": 10.0, "b": 20.0},
        )
        assert shap_cols == ["shap_a", "shap_b"]
        # 1 join_key + N shap cols + 1 expected_value
        assert len(select_exprs) == 4
        # First is the join key column
        assert isinstance(select_exprs[0], _RecordingExpr)
        assert select_exprs[0].kind == "col"
        assert select_exprs[0].payload == "account_id"
        # Last is a literal 0.0 aliased as shap_expected_value
        assert select_exprs[-1].kind == "lit"
        assert select_exprs[-1].payload == 0.0
        assert select_exprs[-1].alias_name == EXPECTED_VALUE_COL

    def test_shap_column_formula_is_importance_times_deviation(self, recording_functions):
        select_exprs, _ = shap_runner._build_attribution_select(
            join_key="pk",
            feature_order=["x"],
            importances=[0.7],
            means={"x": 5.0},
        )
        shap_expr = select_exprs[1]
        assert shap_expr.alias_name == "shap_x"
        # Outer expression: (col - lit) * lit
        assert shap_expr.kind == "mul"
        left, right = shap_expr.children
        assert left.kind == "sub"
        col_expr, mean_lit = left.children
        assert col_expr.kind == "col" and col_expr.payload == "x"
        assert mean_lit.kind == "lit" and mean_lit.payload == 5.0
        assert right.kind == "lit" and right.payload == 0.7


# ---------------------------------------------------------------------------
# compute_shap_distributed — validation + orchestration
# ---------------------------------------------------------------------------


class TestComputeShapDistributedValidation:
    def test_empty_feature_columns_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["account_id"]
        with pytest.raises(ValueError, match="at least one feature column"):
            compute_shap_distributed(
                spark_df=df, feature_columns=[], model=MagicMock(), background=BackgroundSample()
            )

    def test_missing_join_key_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["a", "b"]
        with pytest.raises(ValueError, match="join_key"):
            compute_shap_distributed(
                spark_df=df, feature_columns=["a"], model=MagicMock(), background=BackgroundSample()
            )


class TestComputeShapDistributedOrchestration:
    def _make_spark_df(self, columns: List[str]) -> Any:
        df = MagicMock(name="SparkDF")
        df.columns = list(columns)
        df.select.return_value = MagicMock(name="ShapDF")
        return df

    def test_spark_pipeline_model_full_path(self, recording_functions):
        classifier = MagicMock(name="GBT", spec=["featureImportances"])
        classifier.featureImportances.toArray.return_value = [0.8, 0.2]
        model = _pipeline_model([_FakeAssembler(["feat_a", "feat_b"]), classifier])

        bg = BackgroundSample(
            rows=[{"feat_a": 1.0, "feat_b": 10.0}, {"feat_a": 3.0, "feat_b": 20.0}],
            feature_columns=["feat_a", "feat_b"],
            sample_size=2,
        )
        df = self._make_spark_df(["account_id", "feat_a", "feat_b"])

        result = compute_shap_distributed(
            spark_df=df,
            feature_columns=["feat_a", "feat_b"],
            model=model,
            background=bg,
            join_key="account_id",
        )
        assert result.feature_columns == ["feat_a", "feat_b"]
        assert result.shap_columns == ["shap_feat_a", "shap_feat_b"]
        assert result.background_size == 2
        df.select.assert_called_once()
        # Captured argument list contains join_key + 2 shap cols + expected_value
        emitted = df.select.call_args.args
        assert len(emitted) == 4

    def test_no_pandas_udf_used(self, recording_functions, monkeypatch):
        """Regression: the old path built a pandas_udf with a model in the closure,
        which raised CONTEXT_ONLY_VALID_ON_DRIVER at pickle time. The new path
        must never construct a pandas_udf."""
        called: list[str] = []

        def _boom(*_a, **_kw):
            called.append("pandas_udf")
            raise AssertionError("pandas_udf must not be invoked in the new path")

        recording_functions.pandas_udf = _boom

        classifier = MagicMock(name="GBT", spec=["featureImportances"])
        classifier.featureImportances.toArray.return_value = [0.5, 0.5]
        model = _pipeline_model([_FakeAssembler(["a", "b"]), classifier])
        bg = BackgroundSample(
            rows=[{"a": 1.0, "b": 2.0}],
            feature_columns=["a", "b"],
            sample_size=1,
        )
        df = self._make_spark_df(["pk", "a", "b"])
        compute_shap_distributed(
            spark_df=df, feature_columns=["a", "b"], model=model, background=bg, join_key="pk"
        )
        assert called == []

    def test_empty_background_uses_spark_fallback_means(self, recording_functions):
        classifier = MagicMock(name="GBT", spec=["featureImportances"])
        classifier.featureImportances.toArray.return_value = [0.5, 0.5]
        model = _pipeline_model([_FakeAssembler(["a", "b"]), classifier])
        bg = BackgroundSample(rows=[], feature_columns=[], sample_size=0)

        df = self._make_spark_df(["pk", "a", "b"])
        agg_called: list[int] = []

        class _HeadRow:
            def __getitem__(self, key: str) -> float:
                return 2.0

        def _agg(*exprs):
            agg_called.append(len(exprs))
            r = MagicMock()
            r.head.return_value = _HeadRow()
            return r

        df.agg = _agg
        result = compute_shap_distributed(
            spark_df=df, feature_columns=["a", "b"], model=model, background=bg, join_key="pk"
        )
        assert agg_called == [2]
        assert result.shap_columns == ["shap_a", "shap_b"]

    def test_logistic_regression_model_end_to_end(self, recording_functions):
        classifier = MagicMock(name="LR", spec=["coefficients"])
        classifier.coefficients.toArray.return_value = [-0.5, 0.3]
        model = _pipeline_model([_FakeAssembler(["a", "b"]), classifier])
        bg = BackgroundSample(
            rows=[{"a": 0.0, "b": 0.0}, {"a": 2.0, "b": 4.0}],
            feature_columns=["a", "b"],
            sample_size=2,
        )
        df = self._make_spark_df(["pk", "a", "b"])
        result = compute_shap_distributed(
            spark_df=df, feature_columns=["a", "b"], model=model, background=bg, join_key="pk"
        )
        assert result.feature_columns == ["a", "b"]
        assert result.shap_columns == ["shap_a", "shap_b"]

    def test_model_without_importances_fails_fast(self, recording_functions):
        cl = MagicMock(name="Mystery", spec=[])
        model = _pipeline_model([_FakeAssembler(["a", "b"]), cl])
        bg = BackgroundSample(
            rows=[{"a": 1.0, "b": 2.0}], feature_columns=["a", "b"], sample_size=1
        )
        df = self._make_spark_df(["pk", "a", "b"])
        with pytest.raises(ValueError, match="featureImportances"):
            compute_shap_distributed(
                spark_df=df, feature_columns=["a", "b"], model=model, background=bg, join_key="pk"
            )


# ---------------------------------------------------------------------------
# unwrap_tree_model — unchanged surface
# ---------------------------------------------------------------------------


class TestUnwrapTreeModel:
    def test_raw_model_returned_as_is(self):
        raw_model = MagicMock(name="XGBClassifier", spec=[])
        assert unwrap_tree_model(raw_model) is raw_model

    def test_pyfunc_model_unwraps_via_model_impl(self, monkeypatch):
        import sys

        inner_model = MagicMock(name="InnerSklearnModel")
        impl = MagicMock(name="ModelImpl", spec=["python_model"])
        impl.python_model = inner_model
        pyfunc_model = MagicMock(name="PyFuncModel", spec=["_model_impl"])
        pyfunc_model._model_impl = impl
        fake_pyfunc = MagicMock(name="mlflow.pyfunc")
        fake_pyfunc.PyFuncModel = type(pyfunc_model)
        fake_mlflow = MagicMock(name="mlflow")
        fake_mlflow.pyfunc = fake_pyfunc
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.pyfunc", fake_pyfunc)
        assert unwrap_tree_model(pyfunc_model) is inner_model

    def test_pyfunc_without_impl_falls_through(self, monkeypatch):
        import sys

        pyfunc_model = MagicMock(name="PyFuncModel", spec=["predict"])
        del pyfunc_model._model_impl
        fake_pyfunc = MagicMock(name="mlflow.pyfunc")
        fake_pyfunc.PyFuncModel = type(pyfunc_model)
        fake_mlflow = MagicMock(name="mlflow")
        fake_mlflow.pyfunc = fake_pyfunc
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.pyfunc", fake_pyfunc)
        assert unwrap_tree_model(pyfunc_model) is pyfunc_model

    def test_no_mlflow_returns_model(self, monkeypatch):
        import sys

        monkeypatch.delitem(sys.modules, "mlflow", raising=False)
        monkeypatch.delitem(sys.modules, "mlflow.pyfunc", raising=False)
        model = MagicMock(name="SomeModel", spec=[])
        assert unwrap_tree_model(model) is model

    def _install_fake_mlflow(self, monkeypatch, pyfunc_model):
        import sys

        fake_pyfunc = MagicMock(name="mlflow.pyfunc")
        fake_pyfunc.PyFuncModel = type(pyfunc_model)
        fake_mlflow = MagicMock(name="mlflow")
        fake_mlflow.pyfunc = fake_pyfunc
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.pyfunc", fake_pyfunc)

    def test_pyfunc_uses_get_raw_model_when_available(self, monkeypatch):
        raw_estimator = MagicMock(name="RawSklearnEstimator", spec=[])
        impl = MagicMock(name="SklearnModelWrapper", spec=["get_raw_model"])
        impl.get_raw_model.return_value = raw_estimator
        pyfunc_model = MagicMock(name="PyFuncModel", spec=["_model_impl"])
        pyfunc_model._model_impl = impl
        self._install_fake_mlflow(monkeypatch, pyfunc_model)
        assert unwrap_tree_model(pyfunc_model) is raw_estimator
        impl.get_raw_model.assert_called_once_with()

    def test_pyfunc_with_sklearn_impl_returns_sklearn_model(self, monkeypatch):
        sklearn_model = MagicMock(name="SklearnRandomForest", spec=[])
        impl = MagicMock(name="SklearnModelWrapper", spec=["sklearn_model"])
        impl.sklearn_model = sklearn_model
        pyfunc_model = MagicMock(name="PyFuncModel", spec=["_model_impl"])
        pyfunc_model._model_impl = impl
        self._install_fake_mlflow(monkeypatch, pyfunc_model)
        assert unwrap_tree_model(pyfunc_model) is sklearn_model

    def test_pyfunc_with_xgboost_impl_returns_xgb_model(self, monkeypatch):
        xgb_model = MagicMock(name="XGBBooster", spec=[])
        impl = MagicMock(name="XGBModelWrapper", spec=["xgb_model"])
        impl.xgb_model = xgb_model
        pyfunc_model = MagicMock(name="PyFuncModel", spec=["_model_impl"])
        pyfunc_model._model_impl = impl
        self._install_fake_mlflow(monkeypatch, pyfunc_model)
        assert unwrap_tree_model(pyfunc_model) is xgb_model

    def test_pyfunc_with_lightgbm_impl_returns_lgb_model(self, monkeypatch):
        lgb_model = MagicMock(name="LGBBooster", spec=[])
        impl = MagicMock(name="LGBModelWrapper", spec=["lgb_model"])
        impl.lgb_model = lgb_model
        pyfunc_model = MagicMock(name="PyFuncModel", spec=["_model_impl"])
        pyfunc_model._model_impl = impl
        self._install_fake_mlflow(monkeypatch, pyfunc_model)
        assert unwrap_tree_model(pyfunc_model) is lgb_model

    def test_pyfunc_with_spark_impl_returns_spark_pipeline(self, monkeypatch):
        spark_pipeline = MagicMock(name="SparkPipelineModel", spec=["stages", "transform"])
        impl = MagicMock(name="SparkPyFuncModelWrapper", spec=["spark_model"])
        impl.spark_model = spark_pipeline
        pyfunc_model = MagicMock(name="PyFuncModel", spec=["_model_impl"])
        pyfunc_model._model_impl = impl
        self._install_fake_mlflow(monkeypatch, pyfunc_model)
        assert unwrap_tree_model(pyfunc_model) is spark_pipeline

    def test_unknown_flavor_returns_pyfunc_model_unchanged(self, monkeypatch):
        impl = MagicMock(name="UnknownWrapper", spec=[])
        pyfunc_model = MagicMock(name="PyFuncModel", spec=["_model_impl"])
        pyfunc_model._model_impl = impl
        self._install_fake_mlflow(monkeypatch, pyfunc_model)
        assert unwrap_tree_model(pyfunc_model) is pyfunc_model

    def test_does_not_call_unwrap_python_model_on_flavor_wrapper(self, monkeypatch):
        impl = MagicMock(name="SklearnModelWrapper", spec=["sklearn_model"])
        impl.sklearn_model = MagicMock(name="raw")
        pyfunc_model = MagicMock(name="PyFuncModel", spec=["_model_impl", "unwrap_python_model"])
        pyfunc_model._model_impl = impl
        self._install_fake_mlflow(monkeypatch, pyfunc_model)
        unwrap_tree_model(pyfunc_model)
        pyfunc_model.unwrap_python_model.assert_not_called()

    def test_python_model_wrapper_still_unwraps(self, monkeypatch):
        custom_model = MagicMock(name="CustomPythonModel", spec=[])
        impl = MagicMock(name="PyFuncImpl", spec=["python_model"])
        impl.python_model = custom_model
        pyfunc_model = MagicMock(name="PyFuncModel", spec=["_model_impl"])
        pyfunc_model._model_impl = impl
        self._install_fake_mlflow(monkeypatch, pyfunc_model)
        assert unwrap_tree_model(pyfunc_model) is custom_model


# ---------------------------------------------------------------------------
# Shapley Sampling — sampling plan (pure Python, deterministic with seed)
# ---------------------------------------------------------------------------


class TestBuildSamplingPlan:
    def test_plan_shape_num_samples_times_num_features(self):
        import numpy as np

        rng = np.random.default_rng(7)
        perms = [rng.permutation(4).tolist() for _ in range(3)]
        plan = shap_runner._build_sampling_plan(perms, num_features=4)
        assert len(plan) == 3 * 4

    def test_each_sample_feature_pair_appears_exactly_once(self):
        import numpy as np

        rng = np.random.default_rng(7)
        perms = [rng.permutation(5).tolist() for _ in range(4)]
        plan = shap_runner._build_sampling_plan(perms, num_features=5)
        seen = {(s, f) for s, f, _mask in plan}
        assert seen == {(s, f) for s in range(4) for f in range(5)}

    def test_before_mask_never_includes_the_chosen_feature(self):
        import numpy as np

        rng = np.random.default_rng(11)
        perms = [rng.permutation(6).tolist() for _ in range(2)]
        plan = shap_runner._build_sampling_plan(perms, num_features=6)
        for sample_id, feature_idx, before_mask in plan:
            assert before_mask[feature_idx] is False, (
                f"before_mask[{feature_idx}] must be False; sample_id={sample_id}"
            )

    def test_before_mask_count_equals_position_in_permutation(self):
        import numpy as np

        rng = np.random.default_rng(13)
        perms = [rng.permutation(5).tolist() for _ in range(3)]
        plan = shap_runner._build_sampling_plan(perms, num_features=5)
        for sample_id, feature_idx, before_mask in plan:
            perm = perms[sample_id]
            expected_position = perm.index(feature_idx)
            assert sum(before_mask) == expected_position

    def test_before_mask_matches_features_preceding_in_permutation(self):
        perms = [[2, 0, 3, 1]]
        plan = shap_runner._build_sampling_plan(perms, num_features=4)
        plan_by_feature = {f: m for _s, f, m in plan}
        # feature 2 is at position 0 → no features before it
        assert plan_by_feature[2] == [False, False, False, False]
        # feature 0 is at position 1 → feature 2 precedes it
        assert plan_by_feature[0] == [False, False, True, False]
        # feature 3 is at position 2 → features 2 and 0 precede
        assert plan_by_feature[3] == [True, False, True, False]
        # feature 1 is at position 3 → features 2, 0, 3 precede
        assert plan_by_feature[1] == [True, False, True, True]

    def test_deterministic_given_seed(self):
        import numpy as np

        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        perms_a = [rng_a.permutation(10).tolist() for _ in range(5)]
        perms_b = [rng_b.permutation(10).tolist() for _ in range(5)]
        plan_a = shap_runner._build_sampling_plan(perms_a, num_features=10)
        plan_b = shap_runner._build_sampling_plan(perms_b, num_features=10)
        assert plan_a == plan_b


# ---------------------------------------------------------------------------
# compute_shap_sampling_distributed — validation
# ---------------------------------------------------------------------------


class TestComputeShapSamplingValidation:
    def test_empty_feature_columns_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["account_id", "feat_a"]
        with pytest.raises(ValueError, match="at least one feature column"):
            shap_runner.compute_shap_sampling_distributed(
                spark_df=df,
                feature_columns=[],
                model_uri="models:/foo@production",
                background=BackgroundSample(),
            )

    def test_missing_join_key_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["feat_a", "feat_b"]
        with pytest.raises(ValueError, match="join_key"):
            shap_runner.compute_shap_sampling_distributed(
                spark_df=df,
                feature_columns=["feat_a"],
                model_uri="models:/foo@production",
                background=BackgroundSample(),
            )

    def test_num_samples_zero_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["account_id", "feat_a"]
        with pytest.raises(ValueError, match="num_samples"):
            shap_runner.compute_shap_sampling_distributed(
                spark_df=df,
                feature_columns=["feat_a"],
                model_uri="models:/foo@production",
                background=BackgroundSample(),
                num_samples=0,
            )

    def test_missing_feature_column_in_spark_df_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["account_id", "feat_a"]
        with pytest.raises(ValueError, match=r"feature columns not in spark_df\.columns.*feat_missing"):
            shap_runner.compute_shap_sampling_distributed(
                spark_df=df,
                feature_columns=["feat_a", "feat_missing"],
                model_uri="models:/foo@production",
                background=BackgroundSample(),
            )

    def test_empty_model_uri_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["account_id", "feat_a"]
        with pytest.raises(ValueError, match="model_uri"):
            shap_runner.compute_shap_sampling_distributed(
                spark_df=df,
                feature_columns=["feat_a"],
                model_uri="",
                background=BackgroundSample(),
            )


# ---------------------------------------------------------------------------
# compute_shap_sampling_distributed — orchestration
# ---------------------------------------------------------------------------


def _install_sampling_mocks(monkeypatch):
    """Mock mlflow.pyfunc.spark_udf + pyspark.sql.functions for the sampling path.

    Returns the mock objects so tests can inspect call sites. No real Spark
    session is required.
    """
    import sys

    # Mock mlflow.pyfunc.spark_udf to return a callable that returns a column-like
    fake_predict_udf = MagicMock(name="predict_udf")
    fake_predict_udf.return_value = _Aliasable("pred")

    fake_spark_udf_factory = MagicMock(name="spark_udf", return_value=fake_predict_udf)
    fake_pyfunc = MagicMock(name="mlflow.pyfunc")
    fake_pyfunc.spark_udf = fake_spark_udf_factory
    fake_mlflow = MagicMock(name="mlflow")
    fake_mlflow.pyfunc = fake_pyfunc
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.pyfunc", fake_pyfunc)

    # Mock pyspark.sql.functions
    fake_F = MagicMock(name="pyspark.sql.functions")
    fake_F.col = lambda name: _Aliasable(name)
    fake_F.lit = lambda value: _Aliasable(f"lit({value})")
    fake_F.when = lambda _cond, _value: _Aliasable("when")
    fake_F.first = MagicMock(return_value=_Aliasable("first"))
    fake_F.avg = MagicMock(return_value=_Aliasable("avg"))
    fake_F.sum = MagicMock(return_value=_Aliasable("sum"))
    fake_F.element_at = MagicMock(return_value=_Aliasable("element_at"))
    fake_F.array = MagicMock(side_effect=lambda *a: _Aliasable("array"))
    fake_F.struct = MagicMock(side_effect=lambda *a: _Aliasable("struct"))
    fake_sql = MagicMock(name="pyspark.sql")
    fake_sql.functions = fake_F

    # Mock pyspark.sql.types — only the types we actually use
    fake_types = MagicMock(name="pyspark.sql.types")
    fake_types.IntegerType = lambda: "IntegerType"
    fake_types.BooleanType = lambda: "BooleanType"
    fake_types.ArrayType = lambda inner: ("ArrayType", inner)
    fake_types.StructType = lambda fields: ("StructType", tuple(fields))
    fake_types.StructField = lambda name, dtype, nullable=True: ("StructField", name, dtype, nullable)
    fake_sql.types = fake_types

    monkeypatch.setitem(sys.modules, "pyspark.sql", fake_sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", fake_F)
    monkeypatch.setitem(sys.modules, "pyspark.sql.types", fake_types)

    return {
        "spark_udf_factory": fake_spark_udf_factory,
        "predict_udf": fake_predict_udf,
        "pyfunc_module": fake_pyfunc,
        "F": fake_F,
    }


def _make_chainable_spark_df(columns):
    """Return a MagicMock Spark DataFrame whose every method returns the same
    mock so long chains like df.crossJoin(...).select(...).union(...).withColumn(...)
    collapse to a single inspectable object."""
    df = MagicMock(name="SparkDF")
    df.columns = list(columns)

    def _chain(*_a, **_kw):
        return df

    df.crossJoin.side_effect = _chain
    df.select.side_effect = _chain
    df.union.side_effect = _chain
    df.withColumn.side_effect = _chain
    df.withColumnRenamed.side_effect = _chain
    df.groupBy.return_value = df
    df.pivot.return_value = df
    df.agg.return_value = df
    return df


class TestComputeShapSamplingOrchestration:
    def test_invokes_mlflow_pyfunc_spark_udf_with_model_uri(self, monkeypatch):
        mocks = _install_sampling_mocks(monkeypatch)
        df = _make_chainable_spark_df(["account_id", "feat_a", "feat_b"])
        df.sparkSession = MagicMock(name="SparkSession")

        bg = BackgroundSample(
            rows=[{"feat_a": 1.0, "feat_b": 2.0}],
            feature_columns=["feat_a", "feat_b"],
            sample_size=1,
        )
        shap_runner.compute_shap_sampling_distributed(
            spark_df=df,
            feature_columns=["feat_a", "feat_b"],
            model_uri="models:/churn@production",
            background=bg,
            num_samples=3,
        )
        mocks["spark_udf_factory"].assert_called_once()
        call = mocks["spark_udf_factory"].call_args
        # The model URI is passed positionally or via kwargs
        assert "models:/churn@production" in (list(call.args) + list(call.kwargs.values()))

    def test_returns_shap_run_result_with_expected_columns(self, monkeypatch):
        _install_sampling_mocks(monkeypatch)
        df = _make_chainable_spark_df(["account_id", "feat_a", "feat_b"])
        df.sparkSession = MagicMock(name="SparkSession")

        bg = BackgroundSample(
            rows=[{"feat_a": 1.0, "feat_b": 2.0}],
            feature_columns=["feat_a", "feat_b"],
            sample_size=1,
        )
        result = shap_runner.compute_shap_sampling_distributed(
            spark_df=df,
            feature_columns=["feat_a", "feat_b"],
            model_uri="models:/churn@production",
            background=bg,
            num_samples=3,
        )
        assert result.feature_columns == ["feat_a", "feat_b"]
        assert result.shap_columns == ["shap_feat_a", "shap_feat_b"]
        assert result.background_size == 1
        assert result.shap_df is not None

    def test_no_pandas_udf_invoked_on_sampling_path(self, monkeypatch):
        """Regression guard: the sampling path must never use pandas_udf
        (that was the original CONTEXT_ONLY_VALID_ON_DRIVER bug)."""
        mocks = _install_sampling_mocks(monkeypatch)
        pandas_udf_calls: list = []
        mocks["F"].pandas_udf = lambda *a, **kw: pandas_udf_calls.append((a, kw))

        df = _make_chainable_spark_df(["account_id", "feat_a"])
        df.sparkSession = MagicMock(name="SparkSession")

        bg = BackgroundSample(
            rows=[{"feat_a": 1.0}], feature_columns=["feat_a"], sample_size=1
        )
        shap_runner.compute_shap_sampling_distributed(
            spark_df=df,
            feature_columns=["feat_a"],
            model_uri="models:/m@production",
            background=bg,
            num_samples=2,
        )
        assert pandas_udf_calls == []

    def test_does_not_pickle_model_object(self, monkeypatch):
        """The sampling path should never serialize a model; it uses
        mlflow.pyfunc.spark_udf which loads the model on each executor
        from the MLflow registry, avoiding cloudpickle entirely."""
        mocks = _install_sampling_mocks(monkeypatch)
        df = _make_chainable_spark_df(["account_id", "feat_a"])
        df.sparkSession = MagicMock(name="SparkSession")

        bg = BackgroundSample(
            rows=[{"feat_a": 1.0}], feature_columns=["feat_a"], sample_size=1
        )
        shap_runner.compute_shap_sampling_distributed(
            spark_df=df,
            feature_columns=["feat_a"],
            model_uri="models:/m@production",
            background=bg,
            num_samples=2,
        )
        # spark_udf is called with a model_uri string, not a model object
        call = mocks["spark_udf_factory"].call_args
        uri_arg = None
        for arg in list(call.args) + list(call.kwargs.values()):
            if isinstance(arg, str) and arg.startswith("models:"):
                uri_arg = arg
                break
        assert uri_arg == "models:/m@production"

    def test_crossjoin_expansion_happens(self, monkeypatch):
        _install_sampling_mocks(monkeypatch)
        df = _make_chainable_spark_df(["account_id", "feat_a", "feat_b"])
        df.sparkSession = MagicMock(name="SparkSession")

        bg = BackgroundSample(
            rows=[{"feat_a": 1.0, "feat_b": 2.0}],
            feature_columns=["feat_a", "feat_b"],
            sample_size=1,
        )
        shap_runner.compute_shap_sampling_distributed(
            spark_df=df,
            feature_columns=["feat_a", "feat_b"],
            model_uri="models:/m@production",
            background=bg,
            num_samples=3,
        )
        df.crossJoin.assert_called()


# ---------------------------------------------------------------------------
# extract_top_drivers_per_account
# ---------------------------------------------------------------------------


class TestExtractTopDriversPerAccount:
    def test_validates_join_key_in_spark_df(self):
        spark_df = MagicMock(name="SparkDF")
        spark_df.columns = ["a", "b"]
        shap_df = MagicMock(name="ShapDF")
        shap_df.columns = ["account_id", "shap_a", "shap_b"]
        with pytest.raises(ValueError, match="spark_df"):
            shap_runner.extract_top_drivers_per_account(
                spark_df=spark_df,
                shap_df=shap_df,
                feature_columns=["a", "b"],
                join_key="account_id",
                k=5,
            )

    def test_validates_join_key_in_shap_df(self):
        spark_df = MagicMock(name="SparkDF")
        spark_df.columns = ["account_id", "a", "b"]
        shap_df = MagicMock(name="ShapDF")
        shap_df.columns = ["shap_a", "shap_b"]
        with pytest.raises(ValueError, match="shap_df"):
            shap_runner.extract_top_drivers_per_account(
                spark_df=spark_df,
                shap_df=shap_df,
                feature_columns=["a", "b"],
                join_key="account_id",
                k=5,
            )

    def test_validates_feature_columns_present_in_spark_df(self):
        spark_df = MagicMock(name="SparkDF")
        spark_df.columns = ["account_id", "a"]
        shap_df = MagicMock(name="ShapDF")
        shap_df.columns = ["account_id", "shap_a", "shap_missing"]
        with pytest.raises(ValueError, match=r"feature columns not in spark_df\.columns"):
            shap_runner.extract_top_drivers_per_account(
                spark_df=spark_df,
                shap_df=shap_df,
                feature_columns=["a", "missing"],
                join_key="account_id",
                k=2,
            )

    def test_validates_shap_columns_present_in_shap_df(self):
        spark_df = MagicMock(name="SparkDF")
        spark_df.columns = ["account_id", "a", "b"]
        shap_df = MagicMock(name="ShapDF")
        shap_df.columns = ["account_id", "shap_a"]
        with pytest.raises(ValueError, match=r"shap columns not in shap_df\.columns"):
            shap_runner.extract_top_drivers_per_account(
                spark_df=spark_df,
                shap_df=shap_df,
                feature_columns=["a", "b"],
                join_key="account_id",
                k=2,
            )

    def test_validates_k_positive(self):
        spark_df = MagicMock(name="SparkDF")
        spark_df.columns = ["account_id", "a"]
        shap_df = MagicMock(name="ShapDF")
        shap_df.columns = ["account_id", "shap_a"]
        with pytest.raises(ValueError, match="k"):
            shap_runner.extract_top_drivers_per_account(
                spark_df=spark_df,
                shap_df=shap_df,
                feature_columns=["a"],
                join_key="account_id",
                k=0,
            )

    def test_happy_path_returns_dataframe(self, monkeypatch):
        import sys

        fake_F = MagicMock(name="F")
        fake_F.col = lambda name: _Aliasable(name)
        fake_F.lit = lambda v: _Aliasable(f"lit({v})")
        fake_F.abs = lambda c: _Aliasable("abs")
        fake_F.array = MagicMock(return_value=_Aliasable("array"))
        fake_F.struct = MagicMock(return_value=_Aliasable("struct"))
        fake_F.when = MagicMock(return_value=_Aliasable("when"))
        fake_F.slice = MagicMock(return_value=_Aliasable("slice"))
        fake_F.sort_array = MagicMock(return_value=_Aliasable("sort_array"))
        fake_F.transform = MagicMock(return_value=_Aliasable("transform"))
        fake_sql = MagicMock(name="pyspark.sql")
        fake_sql.functions = fake_F
        monkeypatch.setitem(sys.modules, "pyspark.sql", fake_sql)
        monkeypatch.setitem(sys.modules, "pyspark.sql.functions", fake_F)

        spark_df = _make_chainable_spark_df(["account_id", "a", "b", "c"])
        shap_df = _make_chainable_spark_df(["account_id", "shap_a", "shap_b", "shap_c"])
        out = shap_runner.extract_top_drivers_per_account(
            spark_df=spark_df,
            shap_df=shap_df,
            feature_columns=["a", "b", "c"],
            join_key="account_id",
            k=2,
        )
        assert out is not None

    def test_sort_key_uses_abs_shap_as_first_struct_field(self, monkeypatch):
        """Regression for the lexicographic-sort bug: ``F.sort_array`` orders
        struct arrays by the first field, so ``abs_shap`` must be field #1.
        Capture the struct-builder call order and verify."""
        import sys

        struct_calls: list = []

        def _struct_record(*args):
            struct_calls.append(args)
            return _Aliasable("struct")

        fake_F = MagicMock(name="F")
        fake_F.col = lambda name: _Aliasable(name)
        fake_F.lit = lambda v: _Aliasable(f"lit({v})")
        fake_F.abs = lambda c: _Aliasable(f"abs({c.name})")
        fake_F.array = MagicMock(return_value=_Aliasable("array"))
        fake_F.struct = _struct_record
        fake_F.when = MagicMock(return_value=_Aliasable("when"))
        fake_F.slice = MagicMock(return_value=_Aliasable("slice"))
        fake_F.sort_array = MagicMock(return_value=_Aliasable("sort_array"))
        fake_F.transform = MagicMock(return_value=_Aliasable("transform"))
        fake_sql = MagicMock(name="pyspark.sql")
        fake_sql.functions = fake_F
        monkeypatch.setitem(sys.modules, "pyspark.sql", fake_sql)
        monkeypatch.setitem(sys.modules, "pyspark.sql.functions", fake_F)

        spark_df = _make_chainable_spark_df(["account_id", "feat_a"])
        shap_df = _make_chainable_spark_df(["account_id", "shap_feat_a"])
        shap_runner.extract_top_drivers_per_account(
            spark_df=spark_df,
            shap_df=shap_df,
            feature_columns=["feat_a"],
            join_key="account_id",
            k=1,
        )
        # The outer sortable struct is the first F.struct call
        sortable = struct_calls[0]
        # Field #1 must be the abs_shap alias so F.sort_array orders by it
        first_field = sortable[0]
        assert first_field.name == "abs_shap"

    def test_output_schema_matches_delta_schema(self, monkeypatch):
        """Regression: top_shap_drivers Delta schema requires struct fields
        ``{feature, value, shap_contribution, direction}`` — not the earlier
        ``{feature, shap_value}`` shape."""
        import sys

        struct_calls: list = []

        def _struct_record(*args):
            struct_calls.append(args)
            return _Aliasable("struct")

        fake_F = MagicMock(name="F")
        fake_F.col = lambda name: _Aliasable(name)
        fake_F.lit = lambda v: _Aliasable(f"lit({v})")
        fake_F.abs = lambda c: _Aliasable("abs")
        fake_F.array = MagicMock(return_value=_Aliasable("array"))
        fake_F.struct = _struct_record
        fake_F.when = MagicMock(return_value=_Aliasable("when"))
        fake_F.slice = MagicMock(return_value=_Aliasable("slice"))
        fake_F.sort_array = MagicMock(return_value=_Aliasable("sort_array"))
        fake_F.transform = MagicMock(return_value=_Aliasable("transform"))
        fake_sql = MagicMock(name="pyspark.sql")
        fake_sql.functions = fake_F
        monkeypatch.setitem(sys.modules, "pyspark.sql", fake_sql)
        monkeypatch.setitem(sys.modules, "pyspark.sql.functions", fake_F)

        spark_df = _make_chainable_spark_df(["account_id", "x"])
        shap_df = _make_chainable_spark_df(["account_id", "shap_x"])
        shap_runner.extract_top_drivers_per_account(
            spark_df=spark_df,
            shap_df=shap_df,
            feature_columns=["x"],
            join_key="account_id",
            k=1,
        )
        # The sortable struct fields (aliased) in order:
        # [abs_shap, feature, value, shap_contribution, direction]
        sortable_aliases = [a.name if isinstance(a, _Aliasable) else None for a in struct_calls[0]]
        # Extract alias targets — the Aliasable chain yields the alias name
        assert any("feature" in a for a in sortable_aliases)
        assert any("value" in a for a in sortable_aliases)
        assert any("shap_contribution" in a for a in sortable_aliases)
        assert any("direction" in a for a in sortable_aliases)
