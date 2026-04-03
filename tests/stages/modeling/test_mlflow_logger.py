from unittest.mock import MagicMock, patch

import pytest

from customer_retention.stages.modeling import ExperimentConfig, MLflowLogger


class TestExperimentConfig:
    def test_config_has_required_fields(self):
        config = ExperimentConfig(
            experiment_name="test_experiment",
            run_name="test_run",
        )
        assert config.experiment_name == "test_experiment"
        assert config.run_name == "test_run"


class TestMLflowLoggerParameters:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_model_hyperparameters(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")
        params = {"n_estimators": 100, "max_depth": 10}

        logger.log_params(params)

        mock_mlflow.log_params.assert_called_once_with(params)
        call_args = mock_mlflow.log_params.call_args[0][0]
        assert call_args["n_estimators"] == 100
        assert call_args["max_depth"] == 10

    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_multiple_parameters(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")
        params = {
            "model_type": "random_forest",
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.1,
        }

        logger.log_params(params)

        mock_mlflow.log_params.assert_called_once()
        call_args = mock_mlflow.log_params.call_args[0][0]
        assert call_args["model_type"] == "random_forest"
        assert call_args["learning_rate"] == 0.1


class TestMLflowLoggerMetrics:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_evaluation_metrics(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")
        metrics = {
            "pr_auc_test": 0.65,
            "roc_auc_test": 0.75,
            "f1_test": 0.60,
        }

        logger.log_metrics(metrics)

        mock_mlflow.log_metrics.assert_called_once_with(metrics)
        call_args = mock_mlflow.log_metrics.call_args[0][0]
        assert call_args["pr_auc_test"] == 0.65
        assert call_args["roc_auc_test"] == 0.75

    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_train_and_test_metrics(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")
        metrics = {
            "pr_auc_train": 0.70,
            "pr_auc_test": 0.65,
            "roc_auc_train": 0.80,
            "roc_auc_test": 0.75,
        }

        logger.log_metrics(metrics)

        mock_mlflow.log_metrics.assert_called_once()
        call_args = mock_mlflow.log_metrics.call_args[0][0]
        assert call_args["pr_auc_train"] == 0.70
        assert call_args["roc_auc_test"] == 0.75


class TestMLflowLoggerArtifacts:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_model_artifact(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")

        logger.log_artifact("/path/to/model.pkl")

        mock_mlflow.log_artifact.assert_called_once_with("/path/to/model.pkl", None)
        assert mock_mlflow.log_artifact.call_args[0][0] == "/path/to/model.pkl"

    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_scaler_artifact(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")

        logger.log_artifact("/path/to/scaler.pkl")

        mock_mlflow.log_artifact.assert_called_once()
        assert mock_mlflow.log_artifact.call_args[0][0] == "/path/to/scaler.pkl"


class TestMLflowLoggerTags:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_model_tags(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")
        tags = {
            "model_type": "xgboost",
            "dataset_version": "v1.0",
            "feature_set_version": "1.0.0",
        }

        logger.set_tags(tags)

        mock_mlflow.set_tags.assert_called_once_with(tags)
        call_args = mock_mlflow.set_tags.call_args[0][0]
        assert call_args["model_type"] == "xgboost"
        assert call_args["dataset_version"] == "v1.0"

    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_validation_gate_tag(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")

        logger.set_tags({"validation_gate_passed": "true"})

        mock_mlflow.set_tags.assert_called_once()
        call_args = mock_mlflow.set_tags.call_args[0][0]
        assert call_args["validation_gate_passed"] == "true"


class TestMLflowLoggerExperiment:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_sets_experiment_on_start(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = MagicMock()
        logger = MLflowLogger(experiment_name="new_experiment")
        logger.start_run()

        mock_mlflow.set_experiment.assert_called_once_with("new_experiment")
        mock_mlflow.start_run.assert_called()

    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_sets_tracking_uri_before_experiment(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = MagicMock()
        logger = MLflowLogger(experiment_name="exp", tracking_uri="sqlite:///test.db")
        logger.start_run()

        mock_mlflow.set_tracking_uri.assert_called_once_with("sqlite:///test.db")
        mock_mlflow.set_experiment.assert_called_once_with("exp")

    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_creates_experiment_with_artifact_location_for_sqlite(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        logger = MLflowLogger(experiment_name="exp", tracking_uri="sqlite:////data/experiments/mlruns.db")
        logger.start_run()

        mock_mlflow.create_experiment.assert_called_once_with("exp", artifact_location="/data/experiments/mlruns/artifacts")

    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_skips_create_when_experiment_exists(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = MagicMock()
        logger = MLflowLogger(experiment_name="exp", tracking_uri="sqlite:///test.db")
        logger.start_run()

        mock_mlflow.create_experiment.assert_not_called()


class TestMLflowLoggerContextManager:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_context_manager_starts_and_ends_run(self, mock_mlflow):
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        logger = MLflowLogger(experiment_name="test")

        with logger:
            pass

        mock_mlflow.start_run.assert_called()
        mock_mlflow.end_run.assert_called()


class TestMLflowLoggerDict:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_dict_as_json(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")
        feature_manifest = {
            "features": ["f1", "f2", "f3"],
            "version": "1.0.0",
        }

        logger.log_dict(feature_manifest, "feature_manifest.json")

        mock_mlflow.log_dict.assert_called_once()
        call_args = mock_mlflow.log_dict.call_args[0]
        assert call_args[0] == feature_manifest
        assert call_args[1] == "feature_manifest.json"


class TestMLflowLoggerModel:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow.sklearn")
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_sklearn_model(self, mock_mlflow, mock_sklearn):
        logger = MLflowLogger(experiment_name="test")
        mock_model = MagicMock(spec=[])

        logger.log_model(mock_model, "model")

        mock_sklearn.log_model.assert_called_once()
        call_args = mock_sklearn.log_model.call_args
        assert call_args[0][0] is mock_model
        assert call_args[0][1] == "model"

    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_spark_wrapper_uses_spark_flavor(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")
        mock_model = MagicMock()
        mock_model.as_pipeline_model = MagicMock(return_value=MagicMock())
        mock_model._fitted_model = MagicMock()
        mock_model.spark_model_class = "LogisticRegression"
        mock_model.spark_model_params = {"maxIter": 10}
        mock_model.feature_names = ["f1", "f2"]
        mock_model.class_weight = "balanced"

        logger.log_model(mock_model, "logistic_regression")

        mock_model.as_pipeline_model.assert_called_once()
        mock_mlflow.spark.log_model.assert_called_once()
        mock_mlflow.set_tag.assert_called_with("logistic_regression.model_flavor", "spark")
        mock_mlflow.log_dict.assert_called_once()

    @patch("customer_retention.stages.modeling.mlflow_logger.get_mlflow_dfs_tmpdir", return_value="/Volumes/cat/sch/mlflow_tmp")
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_spark_model_passes_dfs_tmpdir_on_databricks(self, mock_mlflow, _mock_tmpdir):
        logger = MLflowLogger(experiment_name="test")
        mock_model = MagicMock()
        mock_model.as_pipeline_model = MagicMock(return_value=MagicMock())
        mock_model._fitted_model = MagicMock()
        mock_model.spark_model_class = "LR"
        mock_model.spark_model_params = {}
        mock_model.feature_names = ["f1"]
        mock_model.class_weight = None

        logger.log_model(mock_model, "lr")

        call_kwargs = mock_mlflow.spark.log_model.call_args
        assert call_kwargs[1]["dfs_tmpdir"] == "/Volumes/cat/sch/mlflow_tmp"

    @patch("customer_retention.stages.modeling.mlflow_logger.get_mlflow_dfs_tmpdir", return_value=None)
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_spark_model_no_dfs_tmpdir_locally(self, mock_mlflow, _mock_tmpdir):
        logger = MLflowLogger(experiment_name="test")
        mock_model = MagicMock()
        mock_model.as_pipeline_model = MagicMock(return_value=MagicMock())
        mock_model._fitted_model = MagicMock()
        mock_model.spark_model_class = "LR"
        mock_model.spark_model_params = {}
        mock_model.feature_names = ["f1"]
        mock_model.class_weight = None

        logger.log_model(mock_model, "lr")

        call_kwargs = mock_mlflow.spark.log_model.call_args
        assert "dfs_tmpdir" not in call_kwargs[1]

    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow.sklearn")
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_unfitted_spark_wrapper_uses_sklearn(self, mock_mlflow, mock_sklearn):
        logger = MLflowLogger(experiment_name="test")
        mock_model = MagicMock()
        mock_model.as_pipeline_model = MagicMock()
        mock_model._fitted_model = None

        logger.log_model(mock_model, "model")

        mock_sklearn.log_model.assert_called_once()

    @patch("customer_retention.stages.modeling.mlflow_logger.get_mlflow_dfs_tmpdir", return_value=None)
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_spark_model_passes_pip_requirements(self, mock_mlflow, _mock_tmpdir):
        logger = MLflowLogger(experiment_name="test")
        mock_model = MagicMock()
        mock_model.as_pipeline_model = MagicMock(return_value=MagicMock())
        mock_model._fitted_model = MagicMock()
        mock_model.spark_model_class = "LR"
        mock_model.spark_model_params = {}
        mock_model.feature_names = ["f1"]
        mock_model.class_weight = None

        logger.log_model(mock_model, "lr")

        call_kwargs = mock_mlflow.spark.log_model.call_args[1]
        assert "pip_requirements" in call_kwargs
        assert any("pyspark==" in r for r in call_kwargs["pip_requirements"])

    @patch("customer_retention.stages.modeling.mlflow_logger.get_mlflow_dfs_tmpdir", return_value=None)
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_spark_model_passes_signature(self, mock_mlflow, _mock_tmpdir):
        logger = MLflowLogger(experiment_name="test")
        mock_model = MagicMock()
        mock_model.as_pipeline_model = MagicMock(return_value=MagicMock())
        mock_model._fitted_model = MagicMock()
        mock_model.spark_model_class = "LR"
        mock_model.spark_model_params = {}
        mock_model.feature_names = ["f1", "f2"]
        mock_model.class_weight = None

        logger.log_model(mock_model, "lr")

        call_kwargs = mock_mlflow.spark.log_model.call_args[1]
        sig = call_kwargs["signature"]
        input_names = [col.name for col in sig.inputs.inputs]
        assert input_names == ["f1", "f2"]
        output_names = [col.name for col in sig.outputs.inputs]
        assert output_names == ["prediction"]


class TestSparkPipRequirements:
    def test_returns_pyspark_pinned_version(self):
        pytest.importorskip("pyspark")
        from customer_retention.stages.modeling.mlflow_logger import _spark_pip_requirements
        reqs = _spark_pip_requirements()
        assert len(reqs) == 1
        assert reqs[0].startswith("pyspark==")

    def test_version_matches_installed(self):
        pyspark = pytest.importorskip("pyspark")
        from customer_retention.stages.modeling.mlflow_logger import _spark_pip_requirements
        reqs = _spark_pip_requirements()
        assert reqs[0] == f"pyspark=={pyspark.__version__}"


class TestSparkClassificationSignature:
    def test_signature_input_matches_feature_names(self):
        from customer_retention.stages.modeling.mlflow_logger import _spark_classification_signature
        sig = _spark_classification_signature(["f1", "f2", "f3"])
        input_names = [col.name for col in sig.inputs.inputs]
        assert input_names == ["f1", "f2", "f3"]

    def test_signature_output_is_prediction(self):
        from customer_retention.stages.modeling.mlflow_logger import _spark_classification_signature
        sig = _spark_classification_signature(["f1"])
        output_names = [col.name for col in sig.outputs.inputs]
        assert output_names == ["prediction"]

    def test_signature_all_doubles(self):
        from mlflow.types.schema import DataType

        from customer_retention.stages.modeling.mlflow_logger import _spark_classification_signature
        sig = _spark_classification_signature(["a", "b"])
        for col in sig.inputs.inputs:
            assert col.type == DataType.double
        for col in sig.outputs.inputs:
            assert col.type == DataType.double


class TestMLflowLoggerRunId:
    def test_run_id_none_before_start(self):
        logger = MLflowLogger(experiment_name="test")
        assert logger.run_id is None

    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_run_id_returns_id_after_start(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "abc123"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id="1")

        logger = MLflowLogger(experiment_name="test")
        logger.start_run()

        assert logger.run_id == "abc123"

    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_last_run_id_survives_end_run(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "abc123"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id="1")

        logger = MLflowLogger(experiment_name="test")
        logger.start_run()
        logger.end_run()

        assert logger.run_id is None
        assert logger.last_run_id == "abc123"


class TestMLflowLoggerNestedRun:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_nested_run_calls_start_run_with_nested_true(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")

        with logger.nested_run("child_run"):
            pass

        mock_mlflow.start_run.assert_called_once_with(run_name="child_run", nested=True)

    @patch("customer_retention.stages.modeling.mlflow_logger.MLFLOW_AVAILABLE", False)
    def test_nested_run_noop_when_unavailable(self):
        logger = MLflowLogger(experiment_name="test")
        with logger.nested_run("child"):
            pass  # should not raise


class TestMLflowLoggerEndStaleRuns:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_end_stale_runs_calls_end_run(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")
        logger.end_stale_runs()
        mock_mlflow.end_run.assert_called_once()


class TestMLflowLoggerTrackingUri:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_get_tracking_uri_delegates_to_mlflow(self, mock_mlflow):
        mock_mlflow.get_tracking_uri.return_value = "databricks"
        logger = MLflowLogger(experiment_name="test")
        assert logger.get_tracking_uri() == "databricks"
        mock_mlflow.get_tracking_uri.assert_called_once()

    @patch("customer_retention.stages.modeling.mlflow_logger.MLFLOW_AVAILABLE", False)
    def test_get_tracking_uri_returns_configured_uri_when_unavailable(self):
        logger = MLflowLogger(experiment_name="test", tracking_uri="http://local:5000")
        assert logger.get_tracking_uri() == "http://local:5000"

    @patch("customer_retention.stages.modeling.mlflow_logger.MLFLOW_AVAILABLE", False)
    def test_get_tracking_uri_returns_none_when_unavailable_no_uri(self):
        logger = MLflowLogger(experiment_name="test")
        assert logger.get_tracking_uri() is None


class TestMLflowLoggerDisableAutolog:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_disable_autolog_calls_autolog_disable(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")
        logger.disable_autolog()
        mock_mlflow.autolog.assert_called_once_with(disable=True)
