"""Tests for the refactored callable batch inference framework module."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from customer_retention.stages.scoring.batch_inference import (
    BatchInferenceConfig,
    BatchInferenceResult,
    apply_risk_tiers_pandas,
    run_batch_inference,
)


class TestBatchInferenceConfig:
    def test_default_thresholds(self):
        config = BatchInferenceConfig()
        assert config.threshold == 0.5
        assert config.risk_tier_high == 0.6
        assert config.risk_tier_medium == 0.3
        assert config.target_table == "predictions"
        assert config.audit_table == "inference_audit_log"

    def test_custom_thresholds(self):
        config = BatchInferenceConfig(
            threshold=0.4, risk_tier_high=0.75, risk_tier_medium=0.4
        )
        assert config.threshold == 0.4
        assert config.risk_tier_high == 0.75
        assert config.risk_tier_medium == 0.4

    def test_filter_expression_default_none(self):
        assert BatchInferenceConfig().filter_expression is None

    def test_filter_expression_is_passthrough_string(self):
        config = BatchInferenceConfig(filter_expression="plan_type == 'enterprise'")
        assert config.filter_expression == "plan_type == 'enterprise'"


class TestScopeFilterApplication:
    """Scope filter (NB00 ``ProjectContext.sample_filters``) must narrow the
    customer population before the entity projection on both execution paths.
    Applied as a narrow projection (``df.filter`` / ``safe_query``) — no
    shuffle, no extra Spark jobs, stays distributed."""

    def test_databricks_path_calls_df_filter_when_expression_set(self, monkeypatch):
        pytest.importorskip("pyspark", reason="PySpark required")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        filter_calls: list[str] = []

        class _FakeSparkDF:
            def __init__(self, name="root"):
                self._name = name

            def filter(self, expr):
                filter_calls.append(expr)
                return _FakeSparkDF(name="filtered")

            def select(self, *_a, **_kw):
                return self

            def withColumn(self, *_a, **_kw):  # noqa: N802 — mirrors Spark API
                return self

            def agg(self, *_a, **_kw):
                r = MagicMock()
                r.collect.return_value = [{"total": 0, "churners": 0, "avg_prob": 0.0}]
                return r

            def groupBy(self, *_a, **_kw):  # noqa: N802 — mirrors Spark API
                r = MagicMock()
                cnt = MagicMock()
                cnt.collect.return_value = []
                r.count.return_value = cnt
                return r

            def write(self, *_a, **_kw):
                return MagicMock()

        # Short-circuit at the first point past filter application so we can
        # assert the filter was invoked without wiring the full Spark plan.
        class _BoomError(RuntimeError):
            pass
        _Boom = _BoomError  # noqa: N806 — short alias used below

        def stub_score(spark, entity_df, feature_table, model_uri):  # noqa: ARG001
            raise _Boom("stop after filter")

        monkeypatch.setattr(batch_inference, "_score_with_feature_store", stub_score)

        fake_spark = MagicMock()
        fake_spark.table.return_value = _FakeSparkDF()
        monkeypatch.setattr(batch_inference, "_get_spark", lambda: fake_spark)

        cfg = BatchInferenceConfig(
            catalog="c", schema="s", model_name="m",
            filter_expression="plan_type == 'enterprise'",
        )
        with pytest.raises(_Boom):
            _run_databricks(cfg)
        assert filter_calls == ["plan_type == 'enterprise'"]

    def test_databricks_path_skips_filter_when_expression_missing(self, monkeypatch):
        pytest.importorskip("pyspark", reason="PySpark required")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        filter_calls: list[str] = []

        class _FakeSparkDF:
            def filter(self, expr):
                filter_calls.append(expr)
                return self

            def select(self, *_a, **_kw):
                return self

            def withColumn(self, *_a, **_kw):  # noqa: N802 — mirrors Spark API
                return self

        class _BoomError(RuntimeError):
            pass
        _Boom = _BoomError  # noqa: N806 — short alias used below

        monkeypatch.setattr(
            batch_inference,
            "_score_with_feature_store",
            lambda *a, **kw: (_ for _ in ()).throw(_Boom()),
        )
        fake_spark = MagicMock()
        fake_spark.table.return_value = _FakeSparkDF()
        monkeypatch.setattr(batch_inference, "_get_spark", lambda: fake_spark)

        with pytest.raises(_Boom):
            _run_databricks(BatchInferenceConfig(catalog="c", schema="s", model_name="m"))
        assert filter_calls == []

    def test_local_path_applies_safe_query_when_expression_set(self, monkeypatch, tmp_path):
        import pandas as _pd

        from customer_retention.core import compat
        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_local,
        )

        captured: list[str] = []

        def fake_safe_query(df, expr):
            captured.append(expr)
            return df.iloc[:1]

        monkeypatch.setattr(compat, "safe_query", fake_safe_query)

        class _BoomError(RuntimeError):
            pass
        _Boom = _BoomError  # noqa: N806 — short alias used below

        monkeypatch.setattr(
            batch_inference,
            "_load_local_customers",
            lambda *_a, **_kw: _pd.DataFrame(
                {"customer_id": [1, 2, 3], "plan_type": ["free", "enterprise", "enterprise"]}
            ),
        )

        # Stub feature-store-manager so `get_inference_features` raises,
        # which sits *after* the filter application — letting us prove the
        # filter was applied.
        import sys
        fake_fs = type(sys)("customer_retention.integrations.feature_store")

        class _StubRegistry:
            @staticmethod
            def load(_p):
                class _R:
                    @staticmethod
                    def list_features():
                        return ["f1"]
                return _R()

        class _BoomManager:
            def get_inference_features(self, *_a, **_kw):
                raise _Boom("stop after filter")

        fake_fs.FeatureRegistry = _StubRegistry
        fake_fs.get_feature_store_manager = lambda *a, **kw: _BoomManager()
        monkeypatch.setitem(sys.modules, "customer_retention.integrations.feature_store", fake_fs)

        # Stub the delta factory too — same short-circuit style.
        fake_factory = type(sys)("customer_retention.integrations.adapters.factory")
        fake_factory.get_delta = lambda: object()
        monkeypatch.setitem(
            sys.modules,
            "customer_retention.integrations.adapters.factory",
            fake_factory,
        )

        cfg = BatchInferenceConfig(
            model_path=tmp_path / "stub_model.joblib",
            filter_expression="plan_type == 'enterprise'",
        )
        cfg.model_path.write_bytes(b"")
        import joblib

        class _StubModel:
            feature_names_in_ = ["f1"]

        monkeypatch.setattr(joblib, "load", lambda _p: _StubModel())

        with pytest.raises(_Boom):
            _run_local(cfg)
        assert captured == ["plan_type == 'enterprise'"]


class TestBatchInferenceResult:
    def test_summary_one_liner(self):
        result = BatchInferenceResult(
            inference_id="batch_20260408_120000",
            inference_timestamp=datetime(2026, 4, 8, 12, 0, 0),
            total_scored=10000,
            predicted_churners=1234,
            avg_probability=0.234,
            risk_distribution={"High": 500, "Medium": 2000, "Low": 7500},
            model_uri="models:/cat.sch.mdl@production",
        )
        text = result.summary()
        assert "10,000" in text
        assert "1,234" in text
        assert "0.234" in text

    def test_long_summary_includes_all_risk_tiers(self):
        result = BatchInferenceResult(
            inference_id="batch_20260408_120000",
            inference_timestamp=datetime(2026, 4, 8, 12, 0, 0),
            total_scored=100,
            predicted_churners=20,
            avg_probability=0.2,
            risk_distribution={"High": 5, "Medium": 15, "Low": 80},
            model_uri="models:/cat.sch.mdl@production",
        )
        text = result.long_summary()
        assert "High" in text and "5" in text
        assert "Medium" in text and "15" in text
        assert "Low" in text and "80" in text
        assert "models:/cat.sch.mdl@production" in text


class TestApplyRiskTiersPandas:
    def test_three_tier_buckets(self):
        df = pd.DataFrame({"p": [0.05, 0.25, 0.30, 0.45, 0.59, 0.60, 0.95]})
        out = apply_risk_tiers_pandas(df, "p", high=0.6, medium=0.3)
        tiers = out["risk_tier"].tolist()
        # bins=[0, medium=0.3, high=0.6, 1.0] with right-inclusive cuts
        assert tiers[0] == "Low"  # 0.05
        assert tiers[1] == "Low"  # 0.25
        assert tiers[2] == "Low"  # 0.30 (right edge of Low bucket)
        assert tiers[3] == "Medium"  # 0.45
        assert tiers[4] == "Medium"  # 0.59
        assert tiers[5] == "Medium"  # 0.60 (right edge of Medium bucket)
        assert tiers[6] == "High"  # 0.95

    def test_returns_new_dataframe_not_mutating(self):
        df = pd.DataFrame({"p": [0.5]})
        out = apply_risk_tiers_pandas(df, "p", high=0.6, medium=0.3)
        assert "risk_tier" not in df.columns
        assert "risk_tier" in out.columns

    def test_custom_thresholds(self):
        df = pd.DataFrame({"p": [0.1, 0.5, 0.9]})
        out = apply_risk_tiers_pandas(df, "p", high=0.8, medium=0.4)
        tiers = out["risk_tier"].tolist()
        assert tiers[0] == "Low"
        assert tiers[1] == "Medium"
        assert tiers[2] == "High"


class TestApplyRiskTiersSpark:
    """The Spark variant of risk-tier bucketing — tested via a mock DataFrame
    that records the column expression that gets applied.
    """

    def test_invokes_when_chain_with_thresholds(self):
        pytest.importorskip("pyspark", reason="PySpark required for Spark risk-tier tests")
        from unittest.mock import MagicMock

        from customer_retention.stages.scoring.batch_inference import apply_risk_tiers_spark

        # Mock DataFrame whose withColumn returns itself
        mock_df = MagicMock()
        mock_df.withColumn.return_value = mock_df

        result = apply_risk_tiers_spark(mock_df, "churn_probability", high=0.7, medium=0.4)

        # withColumn was called once with "risk_tier" + a Column expression
        assert mock_df.withColumn.call_count == 1
        call_args = mock_df.withColumn.call_args
        assert call_args[0][0] == "risk_tier"
        assert result is mock_df


class TestGetSpark:
    def test_raises_when_no_session(self, monkeypatch):
        from customer_retention.core.compat import detection
        from customer_retention.stages.scoring import batch_inference

        monkeypatch.setattr(detection, "get_spark_session", lambda: None)
        with pytest.raises(RuntimeError, match="No active Spark session"):
            batch_inference._get_spark()

    def test_returns_session_when_present(self, monkeypatch):
        from unittest.mock import MagicMock

        from customer_retention.core.compat import detection
        from customer_retention.stages.scoring import batch_inference

        fake_spark = MagicMock(name="SparkSession")
        monkeypatch.setattr(detection, "get_spark_session", lambda: fake_spark)
        assert batch_inference._get_spark() is fake_spark


class TestRunDatabricksRequiresIdentity:
    def test_missing_catalog_schema_or_model_raises(self, monkeypatch):
        pytest.importorskip("pyspark", reason="PySpark required for Databricks inference tests")
        from customer_retention.stages.scoring import batch_inference
        from customer_retention.stages.scoring.batch_inference import (
            BatchInferenceConfig,
            _run_databricks,
        )

        # Stub _get_spark so we get past it; the validation error should fire
        # before any Spark operations
        monkeypatch.setattr(batch_inference, "_get_spark", lambda: object())

        with pytest.raises(ValueError, match="catalog, schema, and model_name"):
            _run_databricks(BatchInferenceConfig())  # all three None


class TestRunBatchInferenceDispatch:
    def test_dispatcher_picks_local_when_not_databricks(self, monkeypatch):
        # Force is_databricks() False and replace _run_local with a stub
        from customer_retention.core.compat import detection
        from customer_retention.stages.scoring import batch_inference

        monkeypatch.setattr(detection, "is_databricks", lambda: False)

        sentinel = BatchInferenceResult(
            inference_id="local_stub",
            inference_timestamp=datetime(2026, 4, 8),
            total_scored=0,
            predicted_churners=0,
            avg_probability=0.0,
            risk_distribution={},
            model_uri="stub",
        )
        called = {}

        def stub_local(config):
            called["local"] = True
            return sentinel

        monkeypatch.setattr(batch_inference, "_run_local", stub_local)
        result = run_batch_inference(BatchInferenceConfig())
        assert called == {"local": True}
        assert result.inference_id == "local_stub"

    def test_dispatcher_picks_databricks_when_is_databricks_true(self, monkeypatch):
        from customer_retention.core.compat import detection
        from customer_retention.stages.scoring import batch_inference

        monkeypatch.setattr(detection, "is_databricks", lambda: True)

        sentinel = BatchInferenceResult(
            inference_id="dbx_stub",
            inference_timestamp=datetime(2026, 4, 8),
            total_scored=0,
            predicted_churners=0,
            avg_probability=0.0,
            risk_distribution={},
            model_uri="stub",
        )
        called = {}

        def stub_dbx(config):
            called["dbx"] = True
            return sentinel

        monkeypatch.setattr(batch_inference, "_run_databricks", stub_dbx)
        result = run_batch_inference(BatchInferenceConfig(catalog="c", schema="s", model_name="m"))
        assert called == {"dbx": True}
        assert result.inference_id == "dbx_stub"

    def test_dispatcher_defaults_inference_timestamp(self, monkeypatch):
        from customer_retention.core.compat import detection
        from customer_retention.stages.scoring import batch_inference

        monkeypatch.setattr(detection, "is_databricks", lambda: False)
        captured = {}

        def stub_local(config):
            captured["ts"] = config.inference_timestamp
            return BatchInferenceResult(
                inference_id="x",
                inference_timestamp=config.inference_timestamp,
                total_scored=0,
                predicted_churners=0,
                avg_probability=0.0,
                risk_distribution={},
                model_uri="x",
            )

        monkeypatch.setattr(batch_inference, "_run_local", stub_local)
        config = BatchInferenceConfig()  # inference_timestamp=None
        run_batch_inference(config)
        assert isinstance(captured["ts"], datetime), (
            "dispatcher must default inference_timestamp to datetime.now()"
        )


class TestS10CellsCallFramework:
    """Confirm the refactored s10 generator emits cells that import the
    framework module rather than inlining business logic.
    """

    @staticmethod
    def _make_stage():
        from unittest.mock import MagicMock

        from customer_retention.generators.notebook_generator.stages.s10_batch_inference import (
            BatchInferenceStage,
        )

        config = MagicMock()
        config.threshold = 0.5
        config.feature_store.catalog = "test_catalog"
        config.feature_store.schema = "test_schema"
        config.mlflow.model_name = "test_model"
        stage = BatchInferenceStage(config=config, findings=None)
        stage.header_cells = lambda: []
        stage.get_dataset_name = lambda: "test_dataset"
        stage.get_identifier_columns = lambda: ["customer_id"]
        return stage

    def _local_code(self):
        return "\n".join(c.source for c in self._make_stage().generate_local_cells())

    def _databricks_code(self):
        return "\n".join(c.source for c in self._make_stage().generate_databricks_cells())

    def _local_code_only(self):
        # Only code cells (excludes markdown explanations that may legitimately
        # mention the lifted helper names in prose).
        return "\n".join(
            c.source for c in self._make_stage().generate_local_cells() if c.cell_type == "code"
        )

    def _databricks_code_only(self):
        return "\n".join(
            c.source for c in self._make_stage().generate_databricks_cells() if c.cell_type == "code"
        )

    def test_local_cells_import_framework_callable(self):
        code = self._local_code()
        assert "from customer_retention.stages.scoring.batch_inference import" in code
        assert "BatchInferenceConfig" in code
        assert "run_batch_inference" in code

    def test_databricks_cells_import_framework_callable(self):
        code = self._databricks_code()
        assert "from customer_retention.stages.scoring.batch_inference import" in code
        assert "BatchInferenceConfig" in code
        assert "run_batch_inference" in code

    def test_local_cells_pass_dataset_name_through(self):
        code = self._local_code()
        # The dataset_name from the stage flows into BatchInferenceConfig
        assert "test_dataset" in code

    def test_databricks_cells_pass_catalog_schema_model(self):
        code = self._databricks_code()
        assert "test_catalog" in code
        assert "test_schema" in code
        assert "test_model" in code

    def test_no_inline_pd_cut_business_logic(self):
        # Risk-tier bucketing must NOT live in cell *code* — that was the
        # exact business logic that needed to move to the framework.
        # (Markdown cells may explain the bucketing in prose; that's fine.)
        code_local = self._local_code_only()
        code_dbx = self._databricks_code_only()
        assert "pd.cut" not in code_local
        assert "when(col" not in code_dbx
        assert "bins=[0, 0.3, 0.6, 1.0]" not in code_local

    def test_no_inline_fe_score_batch_call(self):
        # The fe.score_batch call now lives inside _run_databricks. Check
        # only code cells; markdown explanations may name the lifted call.
        code = self._databricks_code_only()
        assert "fe.score_batch(" not in code
        assert "FeatureEngineeringClient(" not in code

    def test_no_inline_save_table_calls(self):
        # Both write paths moved to framework
        code_local = self._local_code_only()
        code_dbx = self._databricks_code_only()
        assert ".saveAsTable(" not in code_dbx
        assert "storage.write" not in code_local


class TestS10ScoringReplaySplice:
    """Phase 7 splice point — BatchInferenceStage emits a replay cell iff
    any harvested function has replay_at_scoring=True. Empty / missing
    harvest preserves byte-parity with the pre-Phase-7 notebook."""

    @staticmethod
    def _make_stage(harvest_result=None):
        from unittest.mock import MagicMock

        from customer_retention.generators.notebook_generator.stages.s10_batch_inference import (
            BatchInferenceStage,
        )

        config = MagicMock()
        config.threshold = 0.5
        config.feature_store.catalog = "c"
        config.feature_store.schema = "s"
        config.mlflow.model_name = "m"
        stage = BatchInferenceStage(config=config, findings=None)
        stage.header_cells = lambda: []
        stage.get_dataset_name = lambda: "d"
        stage.get_identifier_columns = lambda: ["id"]
        stage.harvest_result = harvest_result
        return stage

    @staticmethod
    def _harvest_with_replay(names):
        from customer_retention.runtime.harvest import HarvestResult
        from customer_retention.runtime.registry import RegisteredFunction

        hr = HarvestResult.empty()
        for n in names:
            rf = RegisteredFunction(
                name=n,
                source=f"def {n}(df): return df",
                scope="dataset",
                dataset="request",
                replay_at_scoring=True,
                inferred_stage="landing_post",
            )
            hr.functions_by_target.setdefault(("landing_post", "request"), []).append(rf)
        return hr

    def test_no_harvest_emits_no_replay_cell_local(self):
        code = "\n".join(c.source for c in self._make_stage().generate_local_cells())
        assert "from user_extensions import" not in code
        assert "User-Extension Replay" not in code

    def test_no_harvest_emits_no_replay_cell_databricks(self):
        code = "\n".join(c.source for c in self._make_stage().generate_databricks_cells())
        assert "from user_extensions import" not in code

    def test_replay_function_adds_import_cell_local(self):
        stage = self._make_stage(self._harvest_with_replay(["enrich_req"]))
        code = "\n".join(c.source for c in stage.generate_local_cells())
        assert "from user_extensions import enrich_req" in code
        assert "TODO wire replay call-site for enrich_req" in code

    def test_replay_function_adds_import_cell_databricks(self):
        stage = self._make_stage(self._harvest_with_replay(["enrich_req"]))
        code = "\n".join(c.source for c in stage.generate_databricks_cells())
        assert "from user_extensions import enrich_req" in code

    def test_replay_cell_inserted_before_run_batch_inference(self):
        stage = self._make_stage(self._harvest_with_replay(["enrich_req"]))
        sources = [c.source for c in stage.generate_local_cells()]
        replay_idx = next(i for i, s in enumerate(sources) if "enrich_req" in s)
        run_idx = next(i for i, s in enumerate(sources) if "run_batch_inference(config)" in s)
        assert replay_idx < run_idx
