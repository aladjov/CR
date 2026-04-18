"""Unit tests for the distributed SHAP runner.

The runner computes per-row SHAP as linear attribution:
``shap_i(row) = importance_i * (x_i - background_mean_i)``.

The importance vector is derived from model **behaviour** (correlation
between feature values and model predictions over a scored sample), NOT
from model internals. This makes the code robust to MLflow / Feature
Engineering wrapper changes that have historically broken introspection
paths. The scoring primitive is Databricks' ``fe.score_batch``.
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pyspark", reason="PySpark required for SHAP runner tests")

from customer_retention.stages.causal import shap_runner
from customer_retention.stages.causal.shap_runner import (
    DEFAULT_BACKGROUND_SIZE,
    EXPECTED_VALUE_COL,
    IMPORTANCE_SAMPLE_SIZE,
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
    """Column-like stub mirroring the PySpark ``Column`` operator surface we
    exercise. Supports ``alias``, ``cast``, comparison returning Column-like
    (not bool), boolean ops, and arithmetic — so chained Spark expressions
    don't trip over native Python semantics in tests."""

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

    def __sub__(self, other: Any) -> "_Aliasable":
        return _Aliasable(f"({self.name}-{other})")

    def __add__(self, other: Any) -> "_Aliasable":
        return _Aliasable(f"({self.name}+{other})")

    def __mul__(self, other: Any) -> "_Aliasable":
        return _Aliasable(f"({self.name}*{other})")

    def __rmul__(self, other: Any) -> "_Aliasable":
        return _Aliasable(f"({other}*{self.name})")

    def __hash__(self) -> int:
        return id(self)


class _RecordingExpr:
    """Expression stub that records build structure for expression-shape tests."""

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

    fake = MagicMock(name="pyspark.sql.functions")
    fake.col = lambda name: _RecordingExpr("col", payload=name)
    fake.lit = lambda value: _RecordingExpr("lit", payload=value)
    fake.mean = lambda col: _RecordingExpr("mean", children=[_wrap(col)])
    fake.corr = lambda a, b: _RecordingExpr("corr", children=[_wrap(a), _wrap(b)])
    fake.count = MagicMock(side_effect=lambda c: _Aliasable("__c"))
    fake_sql = MagicMock(name="pyspark.sql")
    fake_sql.functions = fake
    monkeypatch.setitem(sys.modules, "pyspark.sql", fake_sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", fake)
    return fake


# ---------------------------------------------------------------------------
# Defaults + dataclasses
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_background_size(self):
        assert DEFAULT_BACKGROUND_SIZE == 1000

    def test_importance_sample_size_is_bounded(self):
        assert 1000 <= IMPORTANCE_SAMPLE_SIZE <= 50_000

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
# Background means (driver vs distributed fallback)
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

    def test_empty_background_triggers_distributed_batched_fallback(self, recording_functions):
        features = [f"f{i}" for i in range(250)]
        agg_calls: list[list] = []

        class _Row:
            def __getitem__(self, key):
                return 1.5

        spark_df = MagicMock(name="SparkDF")

        def _agg(*exprs):
            agg_calls.append(list(exprs))
            r = MagicMock()
            r.head.return_value = _Row()
            return r

        spark_df.agg = _agg
        out = shap_runner._spark_fallback_means(spark_df, features, batch_size=200)
        # 250 in batches of 200 → 2 .agg() calls (Coding_Practices.md batched pattern)
        assert len(agg_calls) == 2
        assert len(agg_calls[0]) == 200
        assert len(agg_calls[1]) == 50
        assert len(out) == 250
        assert all(v == 1.5 for v in out.values())

    def test_fallback_null_mean_falls_back_to_zero(self, recording_functions):
        class _Row:
            def __getitem__(self, key):
                return None

        spark_df = MagicMock(name="SparkDF")
        agg_result = MagicMock()
        agg_result.head.return_value = _Row()
        spark_df.agg.return_value = agg_result
        out = shap_runner._spark_fallback_means(spark_df, ["a", "b"], batch_size=200)
        assert out == {"a": 0.0, "b": 0.0}


# ---------------------------------------------------------------------------
# Importance computation via behaviour (fe.score_batch + batched F.corr)
# ---------------------------------------------------------------------------


def _install_aliasable_functions(monkeypatch):
    import sys

    fake = MagicMock(name="F")
    fake.col = lambda name: _Aliasable(name)
    fake.lit = lambda v: _Aliasable(f"lit({v})")
    fake.corr = lambda a, b: _Aliasable("corr")
    fake.mean = lambda c: _Aliasable("mean")
    fake_sql = MagicMock()
    fake_sql.functions = fake
    monkeypatch.setitem(sys.modules, "pyspark.sql", fake_sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", fake)
    return fake


def _install_fe_and_spark_mocks(monkeypatch, corrs_per_batch: list[dict]):
    """Install the full mock surface for ``_compute_importances_from_behaviour``.

    The production flow sample-caches ``spark_df``, passes entity-keys-only to
    ``fe.score_batch``, then inner-joins ``prediction`` back onto the cached
    feature sample before running batched correlations on the **joined** DF.
    Each successive ``.agg(...)`` on the joined DF returns a row dict-indexable
    by alias name (``c_0``, ``c_1`` …) per the batched-corr pattern.
    """
    import sys

    joined_df = MagicMock(name="JoinedDF")
    call_idx = {"i": 0}

    def _agg(*_exprs):
        idx = call_idx["i"]
        call_idx["i"] += 1
        row_dict = corrs_per_batch[idx] if idx < len(corrs_per_batch) else {}

        class _Row:
            def __getitem__(self, key):
                return row_dict.get(key, 0.0)

        r = MagicMock()
        r.head.return_value = _Row()
        return r

    joined_df.agg = _agg

    entity_sample = MagicMock(name="EntitySample")
    sampled_df = MagicMock(name="SampledDF")
    sampled_df.count = MagicMock(return_value=1000)
    sampled_df.select = MagicMock(return_value=entity_sample)
    sampled_df.join = MagicMock(return_value=joined_df)
    sampled_df.unpersist = MagicMock()

    spark_df = MagicMock(name="SparkDF")
    spark_df.select.return_value.limit.return_value.cache.return_value = sampled_df

    scored_pred = MagicMock(name="ScoredPred")
    scored_df = MagicMock(name="ScoredDF")
    scored_df.select = MagicMock(return_value=scored_pred)
    fake_client = MagicMock()
    fake_client.score_batch = MagicMock(return_value=scored_df)
    fake_fe_module = MagicMock()
    fake_fe_module.FeatureEngineeringClient = MagicMock(return_value=fake_client)
    fake_databricks = MagicMock()
    fake_databricks.feature_engineering = fake_fe_module
    monkeypatch.setitem(sys.modules, "databricks", fake_databricks)
    monkeypatch.setitem(sys.modules, "databricks.feature_engineering", fake_fe_module)

    return {
        "spark_df": spark_df,
        "sampled_df": sampled_df,
        "entity_sample": entity_sample,
        "scored_df": scored_df,
        "scored_pred": scored_pred,
        "joined_df": joined_df,
        "fe_client": fake_client,
        "agg_call_idx": call_idx,
    }


def _install_safe_corr_stub(monkeypatch):
    """Replace ``_safe_corr_expr`` with an ``_Aliasable``-returning stub so
    ``.alias(...)`` still works in the test harness and we can record calls."""
    import customer_retention.core.compat as _compat

    calls: list[tuple[Any, Any]] = []

    def _stub(col_a, col_b):
        calls.append((col_a, col_b))
        return _Aliasable("safe_corr")

    monkeypatch.setattr(_compat, "_safe_corr_expr", _stub)
    return calls


class TestComputeImportancesFromBehaviour:
    def test_fe_receives_only_entity_keys_not_features(self, monkeypatch):
        """FE's ``score_batch`` contract: the input DF must carry the entity
        lookup keys — NOT the feature columns. Passing features in causes a
        name collision with the feature-store wrapper's lookup output and
        also wastes scoring-side memory."""
        mocks = _install_fe_and_spark_mocks(monkeypatch, [{"c_0": 0.5, "c_1": 0.3}])
        _install_aliasable_functions(monkeypatch)
        _install_safe_corr_stub(monkeypatch)

        shap_runner._compute_importances_from_behaviour(
            spark_df=mocks["spark_df"],
            feature_columns=["a", "b"],
            entity_key_cols=["account_id", "point_in_time"],
            model_uri="models:/m@prod",
            sample_size=5000,
        )

        score_call = mocks["fe_client"].score_batch.call_args
        assert score_call.kwargs.get("model_uri") == "models:/m@prod"
        assert score_call.kwargs.get("result_type") == "double"
        assert score_call.kwargs.get("df") is mocks["entity_sample"]
        assert list(mocks["sampled_df"].select.call_args.args) == ["account_id", "point_in_time"]

    def test_feature_sample_is_cached_limited_and_unpersisted(self, monkeypatch):
        """The sampled feature frame is materialized once (``cache`` + ``count``)
        so FE scoring and the join-back see the same rows — ``.limit`` alone is
        non-deterministic across reads. ``.unpersist`` runs in ``finally`` to
        release executor memory even on error."""
        mocks = _install_fe_and_spark_mocks(monkeypatch, [{"c_0": 0.4}])
        _install_aliasable_functions(monkeypatch)
        _install_safe_corr_stub(monkeypatch)

        shap_runner._compute_importances_from_behaviour(
            spark_df=mocks["spark_df"],
            feature_columns=["a"],
            entity_key_cols=["account_id"],
            model_uri="models:/m@prod",
            sample_size=7500,
        )

        spark_df = mocks["spark_df"]
        spark_df.select.return_value.limit.assert_called_with(7500)
        spark_df.select.return_value.limit.return_value.cache.assert_called_once()
        mocks["sampled_df"].count.assert_called_once()
        mocks["sampled_df"].unpersist.assert_called_once()

    def test_correlates_against_prediction_joined_back_on_entity_keys(self, monkeypatch):
        """Root-cause fix: ``scored`` from ``fe.score_batch`` contains only
        entity keys + prediction (plus on-demand features), NOT the store-
        resident feature columns. Correlating features against prediction on
        ``scored`` raises ``UNRESOLVED_COLUMN``. The fix projects
        ``[entity_keys, prediction]`` from ``scored`` and inner-joins onto the
        cached feature sample so both sides share a single DF."""
        mocks = _install_fe_and_spark_mocks(monkeypatch, [{"c_0": 0.6}])
        _install_aliasable_functions(monkeypatch)
        _install_safe_corr_stub(monkeypatch)

        shap_runner._compute_importances_from_behaviour(
            spark_df=mocks["spark_df"],
            feature_columns=["unsubscribed"],
            entity_key_cols=["account_id"],
            model_uri="models:/m@prod",
            sample_size=1000,
        )

        scored_select_args = list(mocks["scored_df"].select.call_args.args)
        assert "account_id" in scored_select_args
        assert "prediction" in scored_select_args

        join_call = mocks["sampled_df"].join.call_args
        assert join_call.args[0] is mocks["scored_pred"]
        assert join_call.kwargs.get("on") == ["account_id"]
        assert join_call.kwargs.get("how") == "inner"

    def test_uses_safe_corr_expr_not_bare_f_corr(self, monkeypatch):
        """ANSI-safe correlation: bare ``F.corr`` raises ``DIVIDE_BY_ZERO`` on
        zero-variance columns when ``spark.sql.ansi.enabled = true``.
        Coding_Practices.md mandates ``_safe_corr_expr`` from ``core.compat``."""
        mocks = _install_fe_and_spark_mocks(monkeypatch, [{"c_0": 0.4, "c_1": 0.2}])
        fake_F = _install_aliasable_functions(monkeypatch)
        fake_F.corr = MagicMock(side_effect=AssertionError("bare F.corr is banned"))
        calls = _install_safe_corr_stub(monkeypatch)

        shap_runner._compute_importances_from_behaviour(
            spark_df=mocks["spark_df"],
            feature_columns=["a", "b"],
            entity_key_cols=["account_id"],
            model_uri="models:/m@prod",
            sample_size=1000,
        )
        assert len(calls) == 2
        fake_F.corr.assert_not_called()

    def test_uniform_fallback_when_all_correlations_zero(self, monkeypatch):
        """Regression: if prediction is constant (degenerate model), all
        correlations are 0. Must return uniform weights (not all-zero)
        so the downstream attribution formula emits meaningful values."""
        mocks = _install_fe_and_spark_mocks(monkeypatch, [{}])
        _install_aliasable_functions(monkeypatch)
        _install_safe_corr_stub(monkeypatch)

        out = shap_runner._compute_importances_from_behaviour(
            spark_df=mocks["spark_df"],
            feature_columns=["a", "b", "c"],
            entity_key_cols=["account_id"],
            model_uri="models:/m@prod",
            sample_size=1000,
        )
        expected = 1.0 / 3
        assert all(abs(v - expected) < 1e-9 for v in out.values())
        assert sum(out.values()) == pytest.approx(1.0)

    def test_importances_normalize_to_unit_sum(self, monkeypatch):
        mocks = _install_fe_and_spark_mocks(
            monkeypatch, [{"c_0": 0.6, "c_1": -0.4, "c_2": 0.0}]
        )
        _install_aliasable_functions(monkeypatch)
        _install_safe_corr_stub(monkeypatch)

        out = shap_runner._compute_importances_from_behaviour(
            spark_df=mocks["spark_df"],
            feature_columns=["a", "b", "c"],
            entity_key_cols=["account_id"],
            model_uri="models:/m@prod",
            sample_size=1000,
        )
        # |0.6| + |0.4| + 0 = 1.0 → identity-normalize; correlation signs absorbed
        assert out["a"] == pytest.approx(0.6)
        assert out["b"] == pytest.approx(0.4)
        assert out["c"] == pytest.approx(0.0)

    def test_batched_corr_uses_100_per_batch_on_joined_df(self, monkeypatch):
        """Coding_Practices.md: batch up to 100 expressions per ``.agg()`` on
        the **joined** DF so Catalyst plans stay O(100²). Large feature counts
        otherwise blow up plan size or force per-column Spark jobs."""
        mocks = _install_fe_and_spark_mocks(monkeypatch, [])
        _install_aliasable_functions(monkeypatch)
        _install_safe_corr_stub(monkeypatch)

        agg_sizes: list[int] = []

        class _Row:
            def __getitem__(self, key):
                return 0.1

        def _agg(*exprs):
            agg_sizes.append(len(exprs))
            r = MagicMock()
            r.head.return_value = _Row()
            return r

        mocks["joined_df"].agg = _agg

        features = [f"f{i}" for i in range(250)]
        shap_runner._compute_importances_from_behaviour(
            spark_df=mocks["spark_df"],
            feature_columns=features,
            entity_key_cols=["account_id"],
            model_uri="models:/m@prod",
            sample_size=1000,
        )
        assert agg_sizes == [100, 100, 50]

    def test_unpersist_called_even_when_fe_raises(self, monkeypatch):
        """Cleanup contract: ``unpersist`` runs in ``finally`` so a scoring
        failure does not leak the cached feature sample."""
        mocks = _install_fe_and_spark_mocks(monkeypatch, [{"c_0": 0.1}])
        _install_aliasable_functions(monkeypatch)
        _install_safe_corr_stub(monkeypatch)

        mocks["fe_client"].score_batch.side_effect = RuntimeError("FE down")

        with pytest.raises(RuntimeError, match="FE down"):
            shap_runner._compute_importances_from_behaviour(
                spark_df=mocks["spark_df"],
                feature_columns=["a"],
                entity_key_cols=["account_id"],
                model_uri="models:/m@prod",
                sample_size=1000,
            )
        mocks["sampled_df"].unpersist.assert_called_once()


# ---------------------------------------------------------------------------
# Attribution select builder
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


# ---------------------------------------------------------------------------
# compute_shap_distributed — validation
# ---------------------------------------------------------------------------


class TestComputeShapDistributedValidation:
    def test_empty_feature_columns_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["account_id"]
        with pytest.raises(ValueError, match="at least one feature column"):
            compute_shap_distributed(
                spark_df=df, feature_columns=[], model_uri="models:/m@p",
                background=BackgroundSample(), entity_key_cols=["account_id"],
            )

    def test_empty_model_uri_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["account_id", "a"]
        with pytest.raises(ValueError, match="model_uri"):
            compute_shap_distributed(
                spark_df=df, feature_columns=["a"], model_uri="",
                background=BackgroundSample(), entity_key_cols=["account_id"],
            )

    def test_missing_join_key_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["a", "b"]
        with pytest.raises(ValueError, match="join_key"):
            compute_shap_distributed(
                spark_df=df, feature_columns=["a"], model_uri="models:/m@p",
                background=BackgroundSample(), entity_key_cols=["account_id"],
            )

    def test_empty_entity_key_cols_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["account_id", "a"]
        with pytest.raises(ValueError, match="entity_key_cols"):
            compute_shap_distributed(
                spark_df=df, feature_columns=["a"], model_uri="models:/m@p",
                background=BackgroundSample(), entity_key_cols=[],
            )

    def test_missing_feature_column_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["account_id", "a"]
        with pytest.raises(ValueError, match=r"feature columns not in spark_df\.columns"):
            compute_shap_distributed(
                spark_df=df, feature_columns=["a", "missing"], model_uri="models:/m@p",
                background=BackgroundSample(), entity_key_cols=["account_id"],
            )

    def test_missing_entity_key_raises(self):
        df = MagicMock(name="DF")
        df.columns = ["account_id", "a"]
        with pytest.raises(ValueError, match=r"entity_key_cols not in spark_df\.columns"):
            compute_shap_distributed(
                spark_df=df, feature_columns=["a"], model_uri="models:/m@p",
                background=BackgroundSample(), entity_key_cols=["account_id", "missing_ts"],
            )


# ---------------------------------------------------------------------------
# Numeric-feature filter — regression for
# "TypeError: unsupported operand type(s) for +: 'int' and 'datetime.datetime'"
# on the _driver_means path when feature_columns includes a timestamp column.
# ---------------------------------------------------------------------------


class TestNumericFeatureFilter:
    def test_returns_numeric_columns_unchanged(self):
        df = _make_chainable_spark_df(["a", "b", "c"])
        out = shap_runner._numeric_feature_columns(df, ["a", "b", "c"])
        assert out == ["a", "b", "c"]

    def test_drops_timestamp_column(self):
        df = _make_chainable_spark_df(
            ["account_id", "numeric_feat", "created_at"],
            non_numeric_cols=["created_at"],
        )
        out = shap_runner._numeric_feature_columns(df, ["numeric_feat", "created_at"])
        assert out == ["numeric_feat"]

    def test_keeps_boolean_columns(self):
        """Booleans are usable in linear attribution (0/1 semantics)."""
        from pyspark.sql.types import BooleanType, DoubleType

        df = MagicMock(name="DF")
        df.columns = ["flag", "score"]

        class _FakeSchema:
            def __getitem__(self, name):
                field = MagicMock()
                field.dataType = BooleanType() if name == "flag" else DoubleType()
                return field

        df.schema = _FakeSchema()
        out = shap_runner._numeric_feature_columns(df, ["flag", "score"])
        assert out == ["flag", "score"]

    def test_compute_shap_distributed_drops_non_numeric_with_warning(self, monkeypatch, caplog):
        """Regression: c02's `feature_columns = training_df.columns - metadata`
        often includes datetime columns. Those must be auto-filtered with a
        visible warning — not silently passed through to _driver_means where
        ``sum([datetime, datetime, ...])`` crashes with TypeError."""
        import logging as _logging

        _install_orchestration_mocks(monkeypatch)
        df = _make_chainable_spark_df(
            ["account_id", "feat_a", "created_at", "feat_b"],
            non_numeric_cols=["created_at"],
        )
        bg = BackgroundSample(
            rows=[{"feat_a": 1.0, "feat_b": 2.0}],
            feature_columns=["feat_a", "feat_b"],
            sample_size=1,
        )
        with caplog.at_level(_logging.WARNING, logger="customer_retention.stages.causal.shap_runner"):
            result = compute_shap_distributed(
                spark_df=df,
                feature_columns=["feat_a", "created_at", "feat_b"],
                model_uri="models:/m@prod",
                background=bg,
                entity_key_cols=["account_id"],
            )
        # Non-numeric column is dropped from feature_columns and shap_columns
        assert result.feature_columns == ["feat_a", "feat_b"]
        assert result.shap_columns == ["shap_feat_a", "shap_feat_b"]
        # Warning mentions the dropped column so the user can see what happened
        assert any("created_at" in rec.getMessage() for rec in caplog.records)

    def test_compute_shap_distributed_raises_when_no_numeric_features(self, monkeypatch):
        """Fail-fast per Coding_Practices.md: if every feature is non-numeric,
        the SHAP attribution formula has nothing to produce — surface a clear
        error instead of emitting an empty ShapRunResult."""
        _install_orchestration_mocks(monkeypatch)
        df = _make_chainable_spark_df(
            ["account_id", "created_at"],
            non_numeric_cols=["created_at"],
        )
        bg = BackgroundSample()
        with pytest.raises(ValueError, match="No numeric features"):
            compute_shap_distributed(
                spark_df=df,
                feature_columns=["created_at"],
                model_uri="models:/m@prod",
                background=bg,
                entity_key_cols=["account_id"],
            )

    def test_no_pandas_udf_invoked(self, monkeypatch):
        """Regression for the CONTEXT_ONLY_VALID_ON_DRIVER bug."""
        mocks = _install_orchestration_mocks(monkeypatch)
        pandas_udf_calls: list = []
        mocks["F"].pandas_udf = lambda *a, **kw: pandas_udf_calls.append((a, kw))

# ---------------------------------------------------------------------------
# compute_shap_distributed — orchestration
# ---------------------------------------------------------------------------


def _make_chainable_spark_df(columns, non_numeric_cols=()):
    """Build a mock Spark DataFrame.

    ``non_numeric_cols`` lets tests mark specific columns as TimestampType
    so the numeric-feature filter exercises the drop path. All other
    columns default to DoubleType.
    """
    from pyspark.sql.types import DoubleType, TimestampType

    df = MagicMock(name="SparkDF")
    df.columns = list(columns)

    schema_map = {}
    for c in columns:
        field = MagicMock(name=f"Field({c})")
        field.dataType = TimestampType() if c in non_numeric_cols else DoubleType()
        schema_map[c] = field

    class _FakeSchema:
        def __getitem__(self, name):
            return schema_map[name]

    df.schema = _FakeSchema()

    def _chain(*_a, **_kw):
        return df

    df.crossJoin.side_effect = _chain
    df.select.side_effect = _chain
    df.union.side_effect = _chain
    df.withColumn.side_effect = _chain
    df.withColumnRenamed.side_effect = _chain
    df.groupBy.return_value = df
    df.pivot.return_value = df
    df.limit.return_value = df
    df.cache.side_effect = _chain
    df.join.side_effect = _chain
    df.count.return_value = len(columns)
    df.unpersist.return_value = None

    class _Row:
        def __getitem__(self, key):
            return 0.5

    agg_result = MagicMock()
    agg_result.head.return_value = _Row()
    df.agg.return_value = agg_result
    return df


def _install_orchestration_mocks(monkeypatch):
    import sys

    scored_df = MagicMock(name="ScoredDF")

    class _Row:
        def __getitem__(self, key):
            return 0.3

    def _agg(*_exprs):
        r = MagicMock()
        r.head.return_value = _Row()
        return r

    scored_df.agg = _agg
    fake_client = MagicMock()
    fake_client.score_batch = MagicMock(return_value=scored_df)
    fake_fe_module = MagicMock()
    fake_fe_module.FeatureEngineeringClient = MagicMock(return_value=fake_client)
    fake_databricks = MagicMock()
    fake_databricks.feature_engineering = fake_fe_module
    monkeypatch.setitem(sys.modules, "databricks", fake_databricks)
    monkeypatch.setitem(sys.modules, "databricks.feature_engineering", fake_fe_module)

    fake_F = MagicMock(name="F")
    fake_F.col = lambda name: _Aliasable(name)
    fake_F.lit = lambda v: _Aliasable(f"lit({v})")
    fake_F.corr = lambda a, b: _Aliasable("corr")
    fake_F.mean = lambda c: _Aliasable("mean")
    fake_sql = MagicMock()
    fake_sql.functions = fake_F
    monkeypatch.setitem(sys.modules, "pyspark.sql", fake_sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", fake_F)

    import customer_retention.core.compat as _compat
    monkeypatch.setattr(_compat, "_safe_corr_expr", lambda a, b: _Aliasable("safe_corr"))

    return {"fe_client": fake_client, "F": fake_F}


class TestComputeShapDistributedOrchestration:
    def test_happy_path_returns_shap_run_result(self, monkeypatch):
        _install_orchestration_mocks(monkeypatch)
        df = _make_chainable_spark_df(["account_id", "feat_a", "feat_b"])
        bg = BackgroundSample(
            rows=[{"feat_a": 1.0, "feat_b": 2.0}],
            feature_columns=["feat_a", "feat_b"],
            sample_size=1,
        )
        result = compute_shap_distributed(
            spark_df=df,
            feature_columns=["feat_a", "feat_b"],
            model_uri="models:/churn@production",
            background=bg,
            entity_key_cols=["account_id"],
        )
        assert result.feature_columns == ["feat_a", "feat_b"]
        assert result.shap_columns == ["shap_feat_a", "shap_feat_b"]
        assert result.background_size == 1
        assert result.shap_df is not None

    def test_calls_fe_score_batch(self, monkeypatch):
        mocks = _install_orchestration_mocks(monkeypatch)
        df = _make_chainable_spark_df(["account_id", "inference_point_in_time", "feat_a"])
        bg = BackgroundSample(
            rows=[{"feat_a": 1.0}], feature_columns=["feat_a"], sample_size=1
        )
        compute_shap_distributed(
            spark_df=df,
            feature_columns=["feat_a"],
            model_uri="models:/m@prod",
            background=bg,
            entity_key_cols=["account_id", "inference_point_in_time"],
        )
        mocks["fe_client"].score_batch.assert_called_once()

    def test_no_pandas_udf_invoked(self, monkeypatch):
        """Regression for the CONTEXT_ONLY_VALID_ON_DRIVER bug."""
        mocks = _install_orchestration_mocks(monkeypatch)
        pandas_udf_calls: list = []
        mocks["F"].pandas_udf = lambda *a, **kw: pandas_udf_calls.append((a, kw))

        df = _make_chainable_spark_df(["account_id", "feat_a"])
        bg = BackgroundSample(
            rows=[{"feat_a": 1.0}], feature_columns=["feat_a"], sample_size=1
        )
        compute_shap_distributed(
            spark_df=df,
            feature_columns=["feat_a"],
            model_uri="models:/m@prod",
            background=bg,
            entity_key_cols=["account_id"],
        )
        assert pandas_udf_calls == []

    def test_never_calls_mlflow_load_model(self, monkeypatch):
        """Regression for KeyError('spark') and PyFuncModel-introspection
        failures: the durable path must never call any mlflow load_model
        variant. Scoring is exclusively via fe.score_batch."""
        import sys

        _install_orchestration_mocks(monkeypatch)
        load_calls: list = []

        fake_mlflow_spark = MagicMock()
        fake_mlflow_spark.load_model = lambda *a, **kw: load_calls.append(("spark", a, kw))
        fake_mlflow_pyfunc = MagicMock()
        fake_mlflow_pyfunc.load_model = lambda *a, **kw: load_calls.append(("pyfunc", a, kw))
        fake_mlflow = MagicMock()
        fake_mlflow.spark = fake_mlflow_spark
        fake_mlflow.pyfunc = fake_mlflow_pyfunc
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.spark", fake_mlflow_spark)
        monkeypatch.setitem(sys.modules, "mlflow.pyfunc", fake_mlflow_pyfunc)

        df = _make_chainable_spark_df(["account_id", "feat_a"])
        bg = BackgroundSample(
            rows=[{"feat_a": 1.0}], feature_columns=["feat_a"], sample_size=1
        )
        compute_shap_distributed(
            spark_df=df,
            feature_columns=["feat_a"],
            model_uri="models:/m@prod",
            background=bg,
            entity_key_cols=["account_id"],
        )
        assert load_calls == []

    def test_select_emits_attribution_expressions(self, monkeypatch):
        _install_orchestration_mocks(monkeypatch)
        df = _make_chainable_spark_df(["account_id", "feat_a", "feat_b"])
        bg = BackgroundSample(
            rows=[{"feat_a": 1.0, "feat_b": 2.0}],
            feature_columns=["feat_a", "feat_b"],
            sample_size=1,
        )
        compute_shap_distributed(
            spark_df=df,
            feature_columns=["feat_a", "feat_b"],
            model_uri="models:/m@prod",
            background=bg,
            entity_key_cols=["account_id"],
        )
        df.select.assert_called()


# ---------------------------------------------------------------------------
# unwrap_tree_model — public helper retained for other callers, not used in
# the hot path anymore
# ---------------------------------------------------------------------------


class TestUnwrapTreeModel:
    """Retained as a public helper for non-causal callers (diagnostics,
    local SHAP). Not used in the causal hot path since the switch to
    behaviour-based importance."""

    def test_raw_model_returned_as_is(self):
        raw_model = MagicMock(name="XGBClassifier", spec=[])
        assert unwrap_tree_model(raw_model) is raw_model

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

    def test_pyfunc_without_impl_falls_through(self, monkeypatch):
        pyfunc_model = MagicMock(name="PyFuncModel", spec=["predict"])
        del pyfunc_model._model_impl
        self._install_fake_mlflow(monkeypatch, pyfunc_model)
        assert unwrap_tree_model(pyfunc_model) is pyfunc_model

    def test_pyfunc_uses_get_raw_model_when_available(self, monkeypatch):
        raw = MagicMock(name="RawEstimator", spec=[])
        impl = MagicMock(name="Impl", spec=["get_raw_model"])
        impl.get_raw_model.return_value = raw
        pyfunc_model = MagicMock(name="PyFuncModel", spec=["_model_impl"])
        pyfunc_model._model_impl = impl
        self._install_fake_mlflow(monkeypatch, pyfunc_model)
        assert unwrap_tree_model(pyfunc_model) is raw
        impl.get_raw_model.assert_called_once_with()

    def test_pyfunc_with_sklearn_impl(self, monkeypatch):
        sklearn_model = MagicMock(name="SklearnRF", spec=[])
        impl = MagicMock(name="Impl", spec=["sklearn_model"])
        impl.sklearn_model = sklearn_model
        pyfunc_model = MagicMock(name="PyFuncModel", spec=["_model_impl"])
        pyfunc_model._model_impl = impl
        self._install_fake_mlflow(monkeypatch, pyfunc_model)
        assert unwrap_tree_model(pyfunc_model) is sklearn_model

    def test_pyfunc_unknown_impl_returns_pyfunc(self, monkeypatch):
        impl = MagicMock(name="UnknownWrapper", spec=[])
        pyfunc_model = MagicMock(name="PyFuncModel", spec=["_model_impl"])
        pyfunc_model._model_impl = impl
        self._install_fake_mlflow(monkeypatch, pyfunc_model)
        assert unwrap_tree_model(pyfunc_model) is pyfunc_model

    def test_non_pyfunc_model_returned_unchanged(self, monkeypatch):
        import sys

        fake_pyfunc = MagicMock(name="mlflow.pyfunc")
        # Some unrelated class so isinstance check returns False
        fake_pyfunc.PyFuncModel = type("UnrelatedBase", (), {})
        fake_mlflow = MagicMock(name="mlflow")
        fake_mlflow.pyfunc = fake_pyfunc
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.pyfunc", fake_pyfunc)
        arbitrary_model = MagicMock(name="ArbitraryModel", spec=[])
        assert unwrap_tree_model(arbitrary_model) is arbitrary_model
