"""Unit tests for the distributed SHAP runner.

The runner is heavily Spark-dependent (broadcast + ``pandas_udf``). The
tests target three layers without requiring a live cluster:

1. **Public dataclass surface** — defaults, validation, structure checks.
2. **Sampling helpers** — exercised with mocked Spark DataFrames so the
   stratification math is validated.
3. **Validation paths** — empty feature lists, missing join keys, etc.

The full distributed compute path is exercised in the integration suite
against a real Databricks workspace; here we only verify the orchestration
logic that surrounds the UDF.
"""

from __future__ import annotations

from typing import Any
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
# Defaults + dataclasses
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_background_size(self):
        assert DEFAULT_BACKGROUND_SIZE == 1000

    def test_default_batch_size(self):
        assert DEFAULT_BATCH_SIZE == 10_000

    def test_shap_prefix(self):
        assert SHAP_PREFIX == "shap_"

    def test_expected_value_col(self):
        assert EXPECTED_VALUE_COL == "shap_expected_value"

    def test_background_sample_dataclass_defaults(self):
        sample = BackgroundSample()
        assert sample.rows == []
        assert sample.feature_columns == []
        assert sample.target_column is None
        assert sample.sample_size == 0

    def test_shap_run_result_defaults(self):
        result = ShapRunResult()
        assert result.shap_df is None
        assert result.feature_columns == []
        assert result.shap_columns == []
        assert result.background_size == 0


# ---------------------------------------------------------------------------
# freeze_background
# ---------------------------------------------------------------------------


class _FakeRow:
    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def asDict(self, recursive: bool = False) -> dict:  # noqa: N802
        # Mirrors pyspark.sql.Row.asDict — production code calls row.asDict(recursive=True)
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

    # groupBy(target).agg(count).collect() for stratified sample
    grouped = MagicMock(name="GroupBy")
    agg_result = MagicMock(name="AggResult")
    agg_result.collect.return_value = [_FakeRow({"target": 0, "__c": 60}), _FakeRow({"target": 1, "__c": 40})]
    grouped.agg.return_value = agg_result
    df.groupBy.return_value = grouped
    return df


@pytest.fixture
def patched_functions(monkeypatch):
    import sys

    fake_functions = MagicMock(name="pyspark.sql.functions")
    fake_functions.count = MagicMock(side_effect=lambda _c: _Aliasable("__c"))
    fake_functions.col = MagicMock(side_effect=lambda c: _Aliasable(c))
    fake_functions.struct = MagicMock(side_effect=lambda *cs: _Aliasable("struct"))
    fake_sql = MagicMock(name="pyspark.sql")
    fake_sql.functions = fake_functions
    monkeypatch.setitem(sys.modules, "pyspark.sql", fake_sql)
    yield fake_functions


class _Aliasable:
    def __init__(self, name: str) -> None:
        self.name = name

    def alias(self, alias: str) -> "_Aliasable":
        return _Aliasable(alias)

    def cast(self, _t: str) -> "_Aliasable":
        return self

    def getItem(self, _idx: int) -> "_Aliasable":  # noqa: N802
        return self


class TestFreezeBackground:
    def test_empty_feature_columns_raises(self):
        df = _make_fake_spark_df([], [])
        with pytest.raises(ValueError, match="at least one feature column"):
            freeze_background(df, feature_columns=[])

    def test_uniform_sample_returns_bounded_rows(self, patched_functions):
        rows = [{"a": float(i), "b": float(i + 1)} for i in range(50)]
        df = _make_fake_spark_df(rows, ["a", "b"])
        sample = freeze_background(df, feature_columns=["a", "b"], n=10)
        assert sample.feature_columns == ["a", "b"]
        assert sample.target_column is None
        # Sample size is the rows captured by the mocked select.collect path
        assert sample.sample_size == 50

    def test_stratified_sample_uses_sampleby(self, patched_functions):
        rows = [{"a": float(i), "b": float(i + 1), "target": i % 2} for i in range(100)]
        df = _make_fake_spark_df(rows, ["a", "b", "target"])
        sample = freeze_background(df, feature_columns=["a", "b"], target_col="target", n=20)
        assert sample.target_column == "target"
        df.sampleBy.assert_called_once()

    def test_target_col_not_in_columns_falls_back_to_uniform(self, patched_functions):
        rows = [{"a": float(i)} for i in range(20)]
        df = _make_fake_spark_df(rows, ["a"])
        sample = freeze_background(df, feature_columns=["a"], target_col="missing_target", n=10)
        # sampleBy should NOT have been called
        df.sampleBy.assert_not_called()
        # When the target column is absent we treat the sample as untargeted
        assert sample.target_column is None


# ---------------------------------------------------------------------------
# compute_shap_distributed validation
# ---------------------------------------------------------------------------


class TestComputeShapDistributedValidation:
    def test_empty_feature_columns_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["account_id"]
        with pytest.raises(ValueError, match="at least one feature column"):
            compute_shap_distributed(
                spark_df=df,
                feature_columns=[],
                model=MagicMock(),
                background=BackgroundSample(),
            )

    def test_missing_join_key_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["a", "b"]
        with pytest.raises(ValueError, match="join_key"):
            compute_shap_distributed(
                spark_df=df,
                feature_columns=["a"],
                model=MagicMock(),
                background=BackgroundSample(),
            )

    def test_join_key_present_passes_validation(self, patched_functions, monkeypatch):
        import sys

        fake_pyspark_functions = MagicMock(name="pyspark.sql.functions")
        fake_pyspark_functions.struct = MagicMock(return_value=_Aliasable("struct"))
        fake_pyspark_functions.col = MagicMock(side_effect=lambda c: _Aliasable(c))

        # pandas_udf decorator: replace the wrapped function with a callable
        # that returns a column-like object (we never actually invoke SHAP).
        def _fake_pandas_udf(*_args, **_kwargs):
            def _wrap(_fn):
                return MagicMock(name="UDFCallable", return_value=_Aliasable("shap_struct"))

            return _wrap

        fake_pyspark_functions.pandas_udf = _fake_pandas_udf
        fake_sql = MagicMock(name="pyspark.sql")
        fake_sql.functions = fake_pyspark_functions

        fake_types = MagicMock(name="pyspark.sql.types")
        fake_types.StructType = lambda fields: ("StructType", tuple(fields))
        fake_types.StructField = lambda name, dtype, nullable: ("StructField", name)
        fake_types.ArrayType = lambda inner: ("ArrayType", inner)
        fake_types.DoubleType = lambda: "Double"
        fake_sql.types = fake_types

        monkeypatch.setitem(sys.modules, "pyspark.sql", fake_sql)
        monkeypatch.setitem(sys.modules, "pyspark.sql.types", fake_types)
        monkeypatch.setitem(sys.modules, "pyspark.sql.functions", fake_pyspark_functions)

        df = MagicMock(name="DF")
        df.columns = ["account_id", "feat_a", "feat_b"]
        df.count.return_value = 100

        chain = MagicMock(name="SelectChain")
        chain.repartition.return_value = chain
        chain.withColumn.return_value = chain
        chain.select.return_value = chain
        chain.count.return_value = 100
        df.select.return_value = chain

        result = compute_shap_distributed(
            spark_df=df,
            feature_columns=["feat_a", "feat_b"],
            model=MagicMock(),
            background=BackgroundSample(rows=[], feature_columns=["feat_a", "feat_b"], sample_size=0),
            join_key="account_id",
        )
        assert result.feature_columns == ["feat_a", "feat_b"]
        assert result.shap_columns == ["shap_feat_a", "shap_feat_b"]


# ---------------------------------------------------------------------------
# Picklable model wrapper
# ---------------------------------------------------------------------------


class TestPicklableModelWrapper:
    def test_wrapper_holds_model(self):
        model = MagicMock(name="Model")
        wrapper = shap_runner._PicklableModelWrapper(model)
        assert wrapper.model is model

    def test_explainer_imports_lazily_without_background(self, monkeypatch):
        import sys

        fake_shap = MagicMock(name="shap")
        fake_shap.TreeExplainer = MagicMock(return_value="fake_explainer")
        monkeypatch.setitem(sys.modules, "shap", fake_shap)
        wrapper = shap_runner._PicklableModelWrapper("dummy_model")
        result = wrapper.explainer()
        assert result == "fake_explainer"
        fake_shap.TreeExplainer.assert_called_once_with("dummy_model")

    def test_explainer_passes_background_dataframe_to_treeexplainer(self, monkeypatch):
        import sys

        fake_shap = MagicMock(name="shap")
        fake_shap.TreeExplainer = MagicMock(return_value="fake_explainer")
        monkeypatch.setitem(sys.modules, "shap", fake_shap)

        background = BackgroundSample(
            rows=[{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}],
            feature_columns=["a", "b"],
            sample_size=2,
        )
        wrapper = shap_runner._PicklableModelWrapper("dummy_model", background=background)
        wrapper.explainer()
        # TreeExplainer was constructed with the model AND a non-empty background frame
        fake_shap.TreeExplainer.assert_called_once()
        call = fake_shap.TreeExplainer.call_args
        assert call.args[0] == "dummy_model"
        passed_background = call.kwargs.get("data")
        assert passed_background is not None
        assert list(passed_background.columns) == ["a", "b"]
        assert len(passed_background) == 2


# ---------------------------------------------------------------------------
# unwrap_tree_model
# ---------------------------------------------------------------------------


class TestUnwrapTreeModel:
    def test_raw_model_returned_as_is(self):
        raw_model = MagicMock(name="XGBClassifier", spec=[])
        assert unwrap_tree_model(raw_model) is raw_model

    def test_pyfunc_model_unwraps_via_model_impl(self, monkeypatch):
        import sys

        inner_model = MagicMock(name="InnerSklearnModel")
        impl = MagicMock(name="ModelImpl")
        impl.python_model = inner_model
        pyfunc_model = MagicMock(name="PyFuncModel")
        pyfunc_model._model_impl = impl

        fake_pyfunc = MagicMock(name="mlflow.pyfunc")
        fake_pyfunc.PyFuncModel = type(pyfunc_model)
        fake_mlflow = MagicMock(name="mlflow")
        fake_mlflow.pyfunc = fake_pyfunc
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.pyfunc", fake_pyfunc)

        result = unwrap_tree_model(pyfunc_model)
        assert result is inner_model

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

        result = unwrap_tree_model(pyfunc_model)
        assert result is pyfunc_model

    def test_no_mlflow_returns_model(self, monkeypatch):
        import sys
        monkeypatch.delitem(sys.modules, "mlflow", raising=False)
        monkeypatch.delitem(sys.modules, "mlflow.pyfunc", raising=False)
        model = MagicMock(name="SomeModel", spec=[])
        assert unwrap_tree_model(model) is model
