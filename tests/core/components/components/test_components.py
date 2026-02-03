import sys
from unittest.mock import MagicMock, patch

import pandas as pd

from customer_retention.core.components.base import Component, ComponentResult
from customer_retention.generators.orchestration.context import PipelineContext


class TestIngester:
    def test_ingester_is_component(self):
        from customer_retention.core.components.components import Ingester
        assert issubclass(Ingester, Component)

    def test_ingester_has_chapter_1(self):
        from customer_retention.core.components.components import Ingester
        comp = Ingester()
        assert 1 in comp.chapters

    def test_ingester_validate_requires_raw_data_path(self):
        from customer_retention.core.components.components import Ingester
        comp = Ingester()
        context = PipelineContext()
        errors = comp.validate_inputs(context)
        assert any("raw_data_path" in e for e in errors)

    def test_ingester_run_returns_result(self, tmp_path):
        from customer_retention.core.components.components import Ingester
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"id": [1, 2], "value": [10, 20]}).to_csv(csv_path, index=False)
        comp = Ingester()
        context = PipelineContext(raw_data_path=str(csv_path))
        context.bronze_path = str(tmp_path / "bronze")
        result = comp.run(context)
        assert isinstance(result, ComponentResult)


class TestProfiler:
    def test_profiler_is_component(self):
        from customer_retention.core.components.components import Profiler
        assert issubclass(Profiler, Component)

    def test_profiler_has_chapter_2(self):
        from customer_retention.core.components.components import Profiler
        comp = Profiler()
        assert 2 in comp.chapters

    def test_profiler_validate_requires_dataframe(self):
        from customer_retention.core.components.components import Profiler
        comp = Profiler()
        context = PipelineContext()
        errors = comp.validate_inputs(context)
        assert len(errors) > 0

    def test_profiler_validate_no_errors_when_valid(self):
        from customer_retention.core.components.components import Profiler
        comp = Profiler()
        context = PipelineContext()
        context.current_df = pd.DataFrame({"a": [1]})
        errors = comp.validate_inputs(context)
        assert len(errors) == 0

    @patch("customer_retention.stages.profiling.column_profiler.ColumnProfiler")
    @patch("customer_retention.stages.profiling.type_detector.TypeDetector")
    def test_profiler_run_success(self, mock_type_cls, mock_profiler_cls):
        from customer_retention.core.components.components import Profiler
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        mock_type = MagicMock()
        mock_type.detect_all.return_value = {"a": "numeric", "b": "categorical"}
        mock_type_cls.return_value = mock_type
        mock_profiler = MagicMock()
        mock_profiler.profile_all.return_value = {"a": {"mean": 1.5}, "b": {"unique": 2}}
        mock_profiler_cls.return_value = mock_profiler
        comp = Profiler()
        context = PipelineContext()
        context.current_df = df
        result = comp.run(context)
        assert isinstance(result, ComponentResult)
        assert result.success is True
        assert result.metrics["columns_profiled"] == 2
        assert context.profiling_results is not None
        assert "types" in context.profiling_results
        assert "profile" in context.profiling_results

    @patch("customer_retention.stages.profiling.type_detector.TypeDetector")
    def test_profiler_run_exception(self, mock_type_cls):
        from customer_retention.core.components.components import Profiler
        mock_type_cls.side_effect = RuntimeError("type detection failed")
        comp = Profiler()
        context = PipelineContext()
        context.current_df = pd.DataFrame({"a": [1]})
        result = comp.run(context)
        assert result.success is False
        assert any("type detection failed" in e for e in result.errors)


class TestTransformer:
    def test_transformer_is_component(self):
        from customer_retention.core.components.components import Transformer
        assert issubclass(Transformer, Component)

    def test_transformer_has_chapter_3(self):
        from customer_retention.core.components.components import Transformer
        comp = Transformer()
        assert 3 in comp.chapters

    def test_transformer_validate_requires_dataframe(self):
        from customer_retention.core.components.components import Transformer
        comp = Transformer()
        context = PipelineContext()
        context.current_df = None
        errors = comp.validate_inputs(context)
        assert any("No DataFrame" in e for e in errors)

    def test_transformer_validate_no_errors_when_valid(self):
        from customer_retention.core.components.components import Transformer
        comp = Transformer()
        context = PipelineContext()
        context.current_df = pd.DataFrame({"a": [1]})
        errors = comp.validate_inputs(context)
        assert len(errors) == 0

    def test_transformer_run_success_with_silver_path(self):
        from customer_retention.core.components.components import Transformer
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        mock_missing_mod = MagicMock()
        mock_missing_handler = MagicMock()
        mock_missing_handler.handle.return_value = df
        mock_missing_mod.MissingHandler.return_value = mock_missing_handler
        mock_outlier_mod = MagicMock()
        mock_outlier_handler = MagicMock()
        mock_outlier_handler.handle.return_value = df
        mock_outlier_mod.OutlierHandler.return_value = mock_outlier_handler
        with patch.dict(sys.modules, {
            "customer_retention.stages.cleaning.missing_handler": mock_missing_mod,
            "customer_retention.stages.cleaning.outlier_handler": mock_outlier_mod,
        }):
            comp = Transformer()
            context = PipelineContext()
            context.current_df = df
            context.silver_path = "/tmp/silver"
            result = comp.run(context)
            assert isinstance(result, ComponentResult)
            assert result.success is True
            assert result.metrics["row_count"] == 3
            assert result.artifacts["silver_data"] == "/tmp/silver"
            assert context.current_stage == "silver"

    def test_transformer_run_success_without_silver_path(self):
        from customer_retention.core.components.components import Transformer
        df = pd.DataFrame({"a": [1, 2]})
        mock_missing_mod = MagicMock()
        mock_missing_handler = MagicMock()
        mock_missing_handler.handle.return_value = df
        mock_missing_mod.MissingHandler.return_value = mock_missing_handler
        mock_outlier_mod = MagicMock()
        mock_outlier_handler = MagicMock()
        mock_outlier_handler.handle.return_value = df
        mock_outlier_mod.OutlierHandler.return_value = mock_outlier_handler
        with patch.dict(sys.modules, {
            "customer_retention.stages.cleaning.missing_handler": mock_missing_mod,
            "customer_retention.stages.cleaning.outlier_handler": mock_outlier_mod,
        }):
            comp = Transformer()
            context = PipelineContext()
            context.current_df = df
            context.silver_path = None
            result = comp.run(context)
            assert result.success is True
            assert result.artifacts == {}

    def test_transformer_run_exception(self):
        from customer_retention.core.components.components import Transformer
        comp = Transformer()
        context = PipelineContext()
        context.current_df = pd.DataFrame({"a": [1]})
        # The import of MissingHandler will fail with ImportError
        result = comp.run(context)
        assert result.success is False
        assert len(result.errors) > 0


class TestFeatureEngineer:
    def test_feature_eng_is_component(self):
        from customer_retention.core.components.components import FeatureEngineer
        assert issubclass(FeatureEngineer, Component)

    def test_feature_eng_has_chapter_4(self):
        from customer_retention.core.components.components import FeatureEngineer
        comp = FeatureEngineer()
        assert 4 in comp.chapters

    def test_feature_eng_validate_requires_dataframe(self):
        from customer_retention.core.components.components import FeatureEngineer
        comp = FeatureEngineer()
        context = PipelineContext()
        context.current_df = None
        errors = comp.validate_inputs(context)
        assert any("No DataFrame" in e for e in errors)

    def test_feature_eng_validate_no_errors_when_valid(self):
        from customer_retention.core.components.components import FeatureEngineer
        comp = FeatureEngineer()
        context = PipelineContext()
        context.current_df = pd.DataFrame({"a": [1]})
        errors = comp.validate_inputs(context)
        assert len(errors) == 0

    @patch("customer_retention.stages.features.feature_engineer.FeatureEngineer")
    def test_feature_eng_run_success_with_gold_path(self, mock_fe_cls):
        from customer_retention.core.components.components import FeatureEngineer
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result_df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "a_sq": [1, 4]})
        mock_fe = MagicMock()
        mock_fe.engineer_all.return_value = result_df
        mock_fe_cls.return_value = mock_fe
        comp = FeatureEngineer()
        context = PipelineContext()
        context.current_df = df
        context.gold_path = "/tmp/gold"
        result = comp.run(context)
        assert isinstance(result, ComponentResult)
        assert result.success is True
        assert result.metrics["feature_count"] == 3
        assert result.artifacts["gold_data"] == "/tmp/gold"
        assert context.current_stage == "gold"

    @patch("customer_retention.stages.features.feature_engineer.FeatureEngineer")
    def test_feature_eng_run_success_without_gold_path(self, mock_fe_cls):
        from customer_retention.core.components.components import FeatureEngineer
        df = pd.DataFrame({"a": [1, 2]})
        mock_fe = MagicMock()
        mock_fe.engineer_all.return_value = df
        mock_fe_cls.return_value = mock_fe
        comp = FeatureEngineer()
        context = PipelineContext()
        context.current_df = df
        context.gold_path = None
        result = comp.run(context)
        assert result.success is True
        assert result.artifacts == {}

    @patch("customer_retention.stages.features.feature_engineer.FeatureEngineer")
    def test_feature_eng_run_exception(self, mock_fe_cls):
        from customer_retention.core.components.components import FeatureEngineer
        mock_fe_cls.side_effect = RuntimeError("feature engineering failed")
        comp = FeatureEngineer()
        context = PipelineContext()
        context.current_df = pd.DataFrame({"a": [1]})
        result = comp.run(context)
        assert result.success is False
        assert any("feature engineering failed" in e for e in result.errors)


class TestTrainer:
    def test_trainer_is_component(self):
        from customer_retention.core.components.components import Trainer
        assert issubclass(Trainer, Component)

    def test_trainer_has_chapter_5(self):
        from customer_retention.core.components.components import Trainer
        comp = Trainer()
        assert 5 in comp.chapters

    def test_trainer_validate_requires_dataframe(self):
        from customer_retention.core.components.components import Trainer
        comp = Trainer()
        context = PipelineContext()
        context.current_df = None
        errors = comp.validate_inputs(context)
        assert any("No DataFrame" in e for e in errors)

    def test_trainer_validate_requires_target_column(self):
        from customer_retention.core.components.components import Trainer
        comp = Trainer()
        context = PipelineContext()
        context.current_df = pd.DataFrame({"a": [1]})
        # target_column comes from exploration_findings, which is None
        errors = comp.validate_inputs(context)
        assert any("target_column" in e for e in errors)

    def test_trainer_validate_no_errors_when_valid(self):
        from customer_retention.core.components.components import Trainer
        comp = Trainer()
        context = PipelineContext()
        context.current_df = pd.DataFrame({"a": [1], "target": [0]})
        context.exploration_findings = MagicMock(target_column="target")
        errors = comp.validate_inputs(context)
        assert len(errors) == 0

    @patch("customer_retention.stages.modeling.baseline_trainer.BaselineTrainer")
    @patch("customer_retention.stages.modeling.data_splitter.DataSplitter")
    def test_trainer_run_success(self, mock_splitter_cls, mock_trainer_cls):
        from customer_retention.core.components.components import Trainer
        df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        mock_splitter = MagicMock()
        mock_splitter.split.return_value = (df[["a"]], df[["a"]], df["target"], df["target"])
        mock_splitter_cls.return_value = mock_splitter
        mock_trainer = MagicMock()
        mock_trainer.train_all.return_value = {
            "logistic": {"pr_auc": 0.85},
            "random_forest": {"pr_auc": 0.90},
        }
        mock_trainer_cls.return_value = mock_trainer
        comp = Trainer()
        context = PipelineContext()
        context.current_df = df
        context.exploration_findings = MagicMock(target_column="target")
        result = comp.run(context)
        assert isinstance(result, ComponentResult)
        assert result.success is True
        assert result.metrics["best_model"] == "random_forest"
        assert result.metrics["pr_auc"] == 0.90
        assert context.model_results is not None

    @patch("customer_retention.stages.modeling.data_splitter.DataSplitter")
    def test_trainer_run_exception(self, mock_splitter_cls):
        from customer_retention.core.components.components import Trainer
        mock_splitter_cls.side_effect = RuntimeError("split failed")
        comp = Trainer()
        context = PipelineContext()
        context.current_df = pd.DataFrame({"a": [1]})
        context.exploration_findings = MagicMock(target_column="target")
        result = comp.run(context)
        assert isinstance(result, ComponentResult)
        assert result.success is False
        assert any("split failed" in e for e in result.errors)


class TestValidator:
    def test_validator_is_component(self):
        from customer_retention.core.components.components import Validator
        assert issubclass(Validator, Component)

    def test_validator_has_chapter_6(self):
        from customer_retention.core.components.components import Validator
        comp = Validator()
        assert 6 in comp.chapters

    def test_validator_validate_requires_model_results(self):
        from customer_retention.core.components.components import Validator
        comp = Validator()
        context = PipelineContext()
        context.model_results = None
        errors = comp.validate_inputs(context)
        assert any("No model results" in e for e in errors)

    def test_validator_validate_no_errors_when_valid(self):
        from customer_retention.core.components.components import Validator
        comp = Validator()
        context = PipelineContext()
        context.model_results = {"logistic": {"pr_auc": 0.85}}
        errors = comp.validate_inputs(context)
        assert len(errors) == 0

    @patch("customer_retention.analysis.diagnostics.calibration_analyzer.CalibrationAnalyzer")
    @patch("customer_retention.analysis.diagnostics.overfitting_analyzer.OverfittingAnalyzer")
    @patch("customer_retention.analysis.diagnostics.leakage_detector.LeakageDetector")
    def test_validator_run_success(self, mock_leakage, mock_overfit, mock_calib):
        from customer_retention.core.components.components import Validator
        comp = Validator()
        context = PipelineContext()
        context.model_results = {"logistic": {"pr_auc": 0.85}}
        result = comp.run(context)
        assert isinstance(result, ComponentResult)
        assert result.success is True
        assert result.metrics["diagnostics_run"] == 3
        assert context.validation_results is not None
        assert context.validation_results["leakage"] == "checked"
        assert context.validation_results["overfitting"] == "checked"
        assert context.validation_results["calibration"] == "checked"

    @patch("customer_retention.analysis.diagnostics.leakage_detector.LeakageDetector")
    def test_validator_run_exception(self, mock_leakage):
        from customer_retention.core.components.components import Validator
        mock_leakage.side_effect = RuntimeError("leakage check failed")
        comp = Validator()
        context = PipelineContext()
        context.model_results = {"logistic": {"pr_auc": 0.85}}
        result = comp.run(context)
        assert result.success is False
        assert any("leakage check failed" in e for e in result.errors)


class TestExplainer:
    def test_explainer_is_component(self):
        from customer_retention.core.components.components import Explainer
        assert issubclass(Explainer, Component)

    def test_explainer_has_chapter_7(self):
        from customer_retention.core.components.components import Explainer
        comp = Explainer()
        assert 7 in comp.chapters

    def test_explainer_validate_requires_model_results(self):
        from customer_retention.core.components.components import Explainer
        comp = Explainer()
        context = PipelineContext()
        context.model_results = None
        errors = comp.validate_inputs(context)
        assert any("No model results" in e for e in errors)

    def test_explainer_validate_no_errors_when_valid(self):
        from customer_retention.core.components.components import Explainer
        comp = Explainer()
        context = PipelineContext()
        context.model_results = {"logistic": {"pr_auc": 0.85}}
        errors = comp.validate_inputs(context)
        assert len(errors) == 0

    def test_explainer_run_success(self):
        from customer_retention.core.components.components import Explainer
        comp = Explainer()
        context = PipelineContext()
        context.model_results = {"logistic": {"pr_auc": 0.85}}
        result = comp.run(context)
        assert isinstance(result, ComponentResult)
        assert result.success is True
        assert result.metrics["explanations_generated"] == 1

    def test_explainer_run_exception(self):
        from customer_retention.core.components.components import Explainer
        comp = Explainer()
        context = PipelineContext()
        context.model_results = {"logistic": {"pr_auc": 0.85}}
        fail_result = ComponentResult(
            success=False,
            status=comp.create_result(success=False).status,
            errors=["explain error"],
        )
        with patch.object(comp, "create_result", side_effect=[
            RuntimeError("explain error"),
            fail_result,
        ]):
            result = comp.run(context)
            assert result.success is False
            assert any("explain error" in e for e in result.errors)


class TestDeployer:
    def test_deployer_is_component(self):
        from customer_retention.core.components.components import Deployer
        assert issubclass(Deployer, Component)

    def test_deployer_has_chapter_8(self):
        from customer_retention.core.components.components import Deployer
        comp = Deployer()
        assert 8 in comp.chapters

    def test_deployer_validate_requires_model_results(self):
        from customer_retention.core.components.components import Deployer
        comp = Deployer()
        context = PipelineContext()
        context.model_results = None
        errors = comp.validate_inputs(context)
        assert any("No model results" in e for e in errors)

    def test_deployer_validate_no_errors_when_valid(self):
        from customer_retention.core.components.components import Deployer
        comp = Deployer()
        context = PipelineContext()
        context.model_results = {"logistic": {"pr_auc": 0.85}}
        errors = comp.validate_inputs(context)
        assert len(errors) == 0

    def test_deployer_run_success(self):
        from customer_retention.core.components.components import Deployer
        comp = Deployer()
        context = PipelineContext()
        context.model_results = {"logistic": {"pr_auc": 0.85}}
        result = comp.run(context)
        assert isinstance(result, ComponentResult)
        assert result.success is True
        assert result.metrics["models_registered"] == 1

    def test_deployer_run_exception(self):
        from customer_retention.core.components.components import Deployer
        comp = Deployer()
        context = PipelineContext()
        context.model_results = {"logistic": {"pr_auc": 0.85}}
        fail_result = ComponentResult(
            success=False,
            status=comp.create_result(success=False).status,
            errors=["deploy error"],
        )
        with patch.object(comp, "create_result", side_effect=[
            RuntimeError("deploy error"),
            fail_result,
        ]):
            result = comp.run(context)
            assert result.success is False
            assert any("deploy error" in e for e in result.errors)
