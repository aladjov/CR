from unittest.mock import MagicMock, patch

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


class TestMLflowLoggerArtifacts:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_model_artifact(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")

        logger.log_artifact("/path/to/model.pkl")

        mock_mlflow.log_artifact.assert_called_once_with("/path/to/model.pkl", None)

    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_scaler_artifact(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")

        logger.log_artifact("/path/to/scaler.pkl")

        mock_mlflow.log_artifact.assert_called_once()


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

    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_validation_gate_tag(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")

        logger.set_tags({"validation_gate_passed": "true"})

        mock_mlflow.set_tags.assert_called_once()


class TestMLflowLoggerExperiment:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_creates_experiment_if_not_exists(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None

        logger = MLflowLogger(experiment_name="new_experiment")
        logger.start_run()

        mock_mlflow.create_experiment.assert_called()

    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_uses_existing_experiment(self, mock_mlflow):
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        logger = MLflowLogger(experiment_name="existing_experiment")
        logger.start_run()

        mock_mlflow.start_run.assert_called()


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


class TestMLflowLoggerModel:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow.sklearn")
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_logs_sklearn_model(self, mock_mlflow, mock_sklearn):
        logger = MLflowLogger(experiment_name="test")
        mock_model = MagicMock()

        logger.log_model(mock_model, "model")

        mock_sklearn.log_model.assert_called_once()
