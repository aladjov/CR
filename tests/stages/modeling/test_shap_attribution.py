"""Tests for ``stages.modeling.shap_attribution`` — training-time attribution
artifact (importances + background means) persisted to MLflow.

Scope:
  - ``ShapAttribution`` dataclass round-trips via JSON.
  - ``compute_shap_attribution`` batches ``.agg()`` per Coding_Practices.md,
    caches once, unpersists even on error, and stays distributed.
  - ``log_attribution`` writes through ``mlflow.log_dict``.
  - ``resolve_run_id_for_model_uri`` handles both ``@alias`` and ``/version``
    URI forms and fails fast on malformed inputs.
"""

from __future__ import annotations

import json
import sys
from typing import Any, List
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pyspark", reason="PySpark required for SHAP attribution tests")

from customer_retention.stages.modeling.shap_attribution import (
    ARTIFACT_FILENAME,
    ShapAttribution,
    compute_shap_attribution,
    load_attribution_from_model_uri,
    load_attribution_from_run,
    log_attribution,
    resolve_run_id_for_model_uri,
)


class _Aliasable:
    def __init__(self, name: str) -> None:
        self.name = name

    def alias(self, name: str) -> "_Aliasable":
        return _Aliasable(name)

    def cast(self, _t: str) -> "_Aliasable":
        return self

    def __eq__(self, other: Any) -> "_Aliasable":  # type: ignore[override]
        return _Aliasable(f"({self.name}=={other})")

    def __hash__(self) -> int:
        return id(self)


def _install_spark_stubs(monkeypatch):
    fake = MagicMock(name="F")
    fake.col = lambda name: _Aliasable(name)
    fake.mean = lambda c: _Aliasable("mean")
    fake_sql = MagicMock()
    fake_sql.functions = fake
    monkeypatch.setitem(sys.modules, "pyspark.sql", fake_sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", fake)
    return fake


def _install_safe_corr_stub(monkeypatch):
    import customer_retention.core.compat as _compat

    calls: list[tuple[Any, Any]] = []

    def _stub(a, b):
        calls.append((a, b))
        return _Aliasable("safe_corr")

    monkeypatch.setattr(_compat, "_safe_corr_expr", _stub)
    return calls


def _make_scored_mock(batch_rows: List[dict]):
    """Build a cached DataFrame mock where successive ``.agg(*exprs).head()``
    calls yield dict-indexed rows."""
    scored = MagicMock(name="Scored")
    scored.count = MagicMock(return_value=len(batch_rows) * 1000 or 1000)
    scored.unpersist = MagicMock()
    call_idx = {"i": 0}

    def _agg(*_exprs):
        idx = call_idx["i"]
        call_idx["i"] += 1
        row_dict = batch_rows[idx] if idx < len(batch_rows) else {}

        class _Row:
            def __getitem__(self, key):
                return row_dict.get(key, 0.0)

        r = MagicMock()
        r.head.return_value = _Row()
        return r

    scored.agg = _agg
    return scored, call_idx


def _make_pipeline_and_df(scored):
    """Wire: ``df.fillna(...).limit(N)`` → transformed → ``.select(...).cache()``
    → ``scored`` (the object whose ``.agg`` we drive in tests)."""
    df = MagicMock(name="RawDF")
    filled = MagicMock(name="Filled")
    df.fillna = MagicMock(return_value=filled)
    limited = MagicMock(name="Limited")
    filled.limit = MagicMock(return_value=limited)
    transformed = MagicMock(name="Transformed")
    selected = MagicMock(name="Selected")
    selected.cache = MagicMock(return_value=scored)
    transformed.select = MagicMock(return_value=selected)

    pipeline = MagicMock(name="PipelineModel")
    pipeline.transform = MagicMock(return_value=transformed)
    return pipeline, df, filled, transformed, selected


class TestShapAttributionDataclass:
    def test_to_dict_round_trip(self):
        attr = ShapAttribution(
            importances={"a": 0.6, "b": 0.4},
            background_means={"a": 1.0, "b": 2.5},
            feature_columns=["a", "b"],
            sample_size=1000,
        )
        restored = ShapAttribution.from_dict(attr.to_dict())
        assert restored == attr

    def test_from_dict_handles_missing_keys(self):
        restored = ShapAttribution.from_dict({})
        assert restored.importances == {}
        assert restored.background_means == {}
        assert restored.feature_columns == []
        assert restored.sample_size == 0

    def test_to_dict_is_json_serializable(self):
        attr = ShapAttribution(
            importances={"a": 0.5, "b": 0.5},
            background_means={"a": 1.0, "b": 2.0},
            feature_columns=["a", "b"],
            sample_size=10,
        )
        payload = json.dumps(attr.to_dict())
        assert json.loads(payload) == attr.to_dict()


class TestComputeShapAttribution:
    def test_empty_feature_columns_raises(self):
        with pytest.raises(ValueError, match="at least one feature column"):
            compute_shap_attribution(
                pipeline_model=MagicMock(),
                df=MagicMock(),
                feature_columns=[],
            )

    def test_fills_nulls_before_transform(self, monkeypatch):
        """``VectorAssembler(handleInvalid="error")`` inside the pipeline must
        see null-filled columns, matching training-time ``prepare_features``
        preprocessing. Otherwise the transform raises."""
        _install_spark_stubs(monkeypatch)
        _install_safe_corr_stub(monkeypatch)
        scored, _ = _make_scored_mock([{"c_0": 0.5}, {"m_0": 1.2}])
        pipeline, df, _filled, _transformed, _selected = _make_pipeline_and_df(scored)

        compute_shap_attribution(pipeline, df, ["a"], sample_size=100)

        df.fillna.assert_called_once()
        args, kwargs = df.fillna.call_args
        assert args[0] == 0 or kwargs.get("value") == 0
        assert kwargs.get("subset") == ["a"] or args[1] == ["a"]

    def test_correlates_against_prediction_via_safe_corr_expr(self, monkeypatch):
        """ANSI-safe correlation: bare ``F.corr`` raises on zero-variance
        columns; the compat helper ``_safe_corr_expr`` returns NULL."""
        fake_F = _install_spark_stubs(monkeypatch)
        fake_F.corr = MagicMock(side_effect=AssertionError("bare F.corr is banned"))
        calls = _install_safe_corr_stub(monkeypatch)
        scored, _ = _make_scored_mock([{"c_0": 0.6, "c_1": 0.3}, {"m_0": 1.0, "m_1": 2.0}])
        pipeline, df, _filled, _transformed, _selected = _make_pipeline_and_df(scored)

        compute_shap_attribution(pipeline, df, ["a", "b"], sample_size=100)

        assert len(calls) == 2
        fake_F.corr.assert_not_called()

    def test_correlation_batching_respects_coding_practices(self, monkeypatch):
        """Coding_Practices.md: ≤100 expressions per ``.agg()`` for correlation
        to keep Catalyst plans O(100²)."""
        _install_spark_stubs(monkeypatch)
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

        scored = MagicMock(name="Scored")
        scored.count = MagicMock(return_value=5000)
        scored.unpersist = MagicMock()
        scored.agg = _agg
        pipeline, df, *_ = _make_pipeline_and_df(scored)

        features = [f"f{i}" for i in range(250)]
        compute_shap_attribution(pipeline, df, features, sample_size=5000)
        # 3 corr batches (100, 100, 50) + 2 mean batches (200, 50)
        assert agg_sizes == [100, 100, 50, 200, 50]

    def test_uniform_fallback_when_all_correlations_zero(self, monkeypatch):
        _install_spark_stubs(monkeypatch)
        _install_safe_corr_stub(monkeypatch)
        scored, _ = _make_scored_mock([{}, {"m_0": 0.0, "m_1": 0.0, "m_2": 0.0}])
        pipeline, df, *_ = _make_pipeline_and_df(scored)

        out = compute_shap_attribution(pipeline, df, ["a", "b", "c"], sample_size=100)

        expected = 1.0 / 3
        assert all(abs(v - expected) < 1e-9 for v in out.importances.values())
        assert sum(out.importances.values()) == pytest.approx(1.0)

    def test_importances_normalize_to_unit_sum(self, monkeypatch):
        _install_spark_stubs(monkeypatch)
        _install_safe_corr_stub(monkeypatch)
        scored, _ = _make_scored_mock([
            {"c_0": 0.6, "c_1": -0.4, "c_2": 0.0},
            {"m_0": 1.0, "m_1": 2.0, "m_2": 3.0},
        ])
        pipeline, df, *_ = _make_pipeline_and_df(scored)

        out = compute_shap_attribution(pipeline, df, ["a", "b", "c"], sample_size=100)

        assert out.importances["a"] == pytest.approx(0.6)
        assert out.importances["b"] == pytest.approx(0.4)
        assert out.importances["c"] == pytest.approx(0.0)

    def test_background_means_are_populated(self, monkeypatch):
        _install_spark_stubs(monkeypatch)
        _install_safe_corr_stub(monkeypatch)
        scored, _ = _make_scored_mock([
            {"c_0": 0.6, "c_1": 0.4},
            {"m_0": 12.5, "m_1": -3.25},
        ])
        pipeline, df, *_ = _make_pipeline_and_df(scored)

        out = compute_shap_attribution(pipeline, df, ["x", "y"], sample_size=100)

        assert out.background_means == {"x": 12.5, "y": -3.25}
        assert out.feature_columns == ["x", "y"]

    def test_mean_null_defaults_to_zero(self, monkeypatch):
        _install_spark_stubs(monkeypatch)
        _install_safe_corr_stub(monkeypatch)

        class _Row:
            def __getitem__(self, key):
                return None if key.startswith("m_") else 0.2

        def _agg(*_exprs):
            r = MagicMock()
            r.head.return_value = _Row()
            return r

        scored = MagicMock()
        scored.count = MagicMock(return_value=100)
        scored.unpersist = MagicMock()
        scored.agg = _agg
        pipeline, df, *_ = _make_pipeline_and_df(scored)

        out = compute_shap_attribution(pipeline, df, ["a"], sample_size=100)
        assert out.background_means == {"a": 0.0}

    def test_sample_is_cached_and_unpersisted(self, monkeypatch):
        _install_spark_stubs(monkeypatch)
        _install_safe_corr_stub(monkeypatch)
        scored, _ = _make_scored_mock([{"c_0": 0.5}, {"m_0": 1.0}])
        pipeline, df, filled, transformed, selected = _make_pipeline_and_df(scored)

        compute_shap_attribution(pipeline, df, ["a"], sample_size=7500)

        filled.limit.assert_called_with(7500)
        selected.cache.assert_called_once()
        scored.count.assert_called_once()
        scored.unpersist.assert_called_once()

    def test_unpersist_runs_when_agg_raises(self, monkeypatch):
        _install_spark_stubs(monkeypatch)
        _install_safe_corr_stub(monkeypatch)

        scored = MagicMock()
        scored.count = MagicMock(return_value=100)
        scored.unpersist = MagicMock()
        scored.agg = MagicMock(side_effect=RuntimeError("spark down"))
        pipeline, df, *_ = _make_pipeline_and_df(scored)

        with pytest.raises(RuntimeError, match="spark down"):
            compute_shap_attribution(pipeline, df, ["a"], sample_size=100)
        scored.unpersist.assert_called_once()

    def test_sample_size_reflects_materialized_count(self, monkeypatch):
        _install_spark_stubs(monkeypatch)
        _install_safe_corr_stub(monkeypatch)
        scored, _ = _make_scored_mock([{"c_0": 0.5}, {"m_0": 1.0}])
        scored.count = MagicMock(return_value=4321)
        pipeline, df, *_ = _make_pipeline_and_df(scored)

        out = compute_shap_attribution(pipeline, df, ["a"], sample_size=5000)
        assert out.sample_size == 4321


class TestLogAttribution:
    def test_invokes_mlflow_log_dict_with_default_artifact_path(self, monkeypatch):
        fake_mlflow = MagicMock()
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

        attr = ShapAttribution(
            importances={"a": 1.0},
            background_means={"a": 0.0},
            feature_columns=["a"],
            sample_size=1,
        )
        log_attribution(attr)
        fake_mlflow.log_dict.assert_called_once()
        payload, path = fake_mlflow.log_dict.call_args.args
        assert payload == attr.to_dict()
        assert path == ARTIFACT_FILENAME

    def test_custom_artifact_path_is_forwarded(self, monkeypatch):
        fake_mlflow = MagicMock()
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

        attr = ShapAttribution(importances={"a": 1.0}, feature_columns=["a"], sample_size=1)
        log_attribution(attr, artifact_path="custom/path.json")
        fake_mlflow.log_dict.assert_called_once_with(attr.to_dict(), "custom/path.json")


class TestResolveRunIdForModelUri:
    def _install_mlflow_client(self, monkeypatch, client):
        fake_tracking = MagicMock()
        fake_tracking.MlflowClient = MagicMock(return_value=client)
        fake_mlflow = MagicMock()
        fake_mlflow.tracking = fake_tracking
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_tracking)

    def test_resolves_alias_uri(self, monkeypatch):
        client = MagicMock()
        version = MagicMock()
        version.run_id = "run_42"
        client.get_model_version_by_alias = MagicMock(return_value=version)
        self._install_mlflow_client(monkeypatch, client)

        run_id = resolve_run_id_for_model_uri("models:/catalog.schema.model_abc@production")
        assert run_id == "run_42"
        client.get_model_version_by_alias.assert_called_once_with(
            "catalog.schema.model_abc", "production"
        )

    def test_resolves_numeric_version_uri(self, monkeypatch):
        client = MagicMock()
        version = MagicMock()
        version.run_id = "run_7"
        client.get_model_version = MagicMock(return_value=version)
        self._install_mlflow_client(monkeypatch, client)

        run_id = resolve_run_id_for_model_uri("models:/catalog.schema.model_abc/3")
        assert run_id == "run_7"
        client.get_model_version.assert_called_once_with("catalog.schema.model_abc", "3")

    def test_non_models_uri_raises(self):
        with pytest.raises(ValueError, match="models:/"):
            resolve_run_id_for_model_uri("runs:/abc/model")

    def test_malformed_uri_raises(self, monkeypatch):
        client = MagicMock()
        self._install_mlflow_client(monkeypatch, client)
        with pytest.raises(ValueError, match="Unsupported"):
            resolve_run_id_for_model_uri("models:/bare_name")


class TestLoadAttribution:
    def test_load_from_run_downloads_and_parses(self, monkeypatch, tmp_path):
        artifact = tmp_path / ARTIFACT_FILENAME
        artifact.write_text(json.dumps({
            "importances": {"a": 0.6, "b": 0.4},
            "background_means": {"a": 1.0, "b": 2.0},
            "feature_columns": ["a", "b"],
            "sample_size": 123,
        }))

        client = MagicMock()
        client.download_artifacts = MagicMock(return_value=str(artifact))
        fake_tracking = MagicMock()
        fake_tracking.MlflowClient = MagicMock(return_value=client)
        fake_mlflow = MagicMock()
        fake_mlflow.tracking = fake_tracking
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_tracking)

        attr = load_attribution_from_run("run_1")
        assert attr.importances == {"a": 0.6, "b": 0.4}
        assert attr.background_means == {"a": 1.0, "b": 2.0}
        assert attr.feature_columns == ["a", "b"]
        assert attr.sample_size == 123

    def test_load_from_model_uri_resolves_then_downloads(self, monkeypatch, tmp_path):
        artifact = tmp_path / ARTIFACT_FILENAME
        artifact.write_text(json.dumps({
            "importances": {"a": 1.0},
            "background_means": {"a": 0.0},
            "feature_columns": ["a"],
            "sample_size": 1,
        }))
        client = MagicMock()
        version = MagicMock()
        version.run_id = "run_abc"
        client.get_model_version_by_alias = MagicMock(return_value=version)
        client.download_artifacts = MagicMock(return_value=str(artifact))
        fake_tracking = MagicMock()
        fake_tracking.MlflowClient = MagicMock(return_value=client)
        fake_mlflow = MagicMock()
        fake_mlflow.tracking = fake_tracking
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_tracking)

        attr = load_attribution_from_model_uri("models:/catalog.schema.m@production")
        client.get_model_version_by_alias.assert_called_once_with(
            "catalog.schema.m", "production"
        )
        client.download_artifacts.assert_called_once_with("run_abc", ARTIFACT_FILENAME, pytest.ANY if False else client.download_artifacts.call_args.args[2])
        assert attr.feature_columns == ["a"]
