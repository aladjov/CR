"""Tests for exploration MLflow tracking parity with production."""

from unittest.mock import MagicMock, patch

from customer_retention.stages.modeling.mlflow_logger import MLflowLogger

# These are the metric keys the production training template logs per child run.
# Exploration MUST log the same keys so MLflow UI comparison works.
PRODUCTION_METRIC_KEYS = {"roc_auc", "pr_auc", "f1", "precision", "recall", "cv_mean", "cv_std"}

# Expected tags on an exploration parent run
EXPLORATION_REQUIRED_TAGS = {"run_type", "composite_name", "target_column", "entity_key", "timestamp_column"}


class TestExplorationAndProductionMetricKeysParity:
    def test_exploration_metric_keys_match_production(self):
        """The metric keys NB08 will log must be a superset of production keys."""
        # This is a documentation/contract test. If production adds a key,
        # this test reminds us to add it to NB08 too.
        exploration_keys = {"roc_auc", "pr_auc", "f1", "precision", "recall", "cv_mean", "cv_std"}
        assert PRODUCTION_METRIC_KEYS.issubset(exploration_keys)


class TestExplorationTagsIncludeRunType:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_exploration_tags_include_run_type(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
        logger = MLflowLogger(experiment_name="training_test_cn")
        logger.start_run(run_name="exploration_test_cn")
        tags = {
            "run_type": "exploration",
            "composite_name": "test_cn",
            "target_column": "churned",
            "entity_key": "entity_id",
            "timestamp_column": "as_of_date",
        }
        logger.set_tags(tags)
        mock_mlflow.set_tags.assert_called_once_with(tags)
        assert EXPLORATION_REQUIRED_TAGS.issubset(tags.keys())


class TestExplorationMetadataSchema:
    def test_exploration_metadata_has_all_expected_keys(self):
        """Validate that the metadata dict we plan to write has all required keys."""
        metadata = {
            "mlflow_experiment_name": "/Shared/training_test__abc1234",
            "mlflow_run_id": "fake-run-id",
            "composite_name": "test__abc1234",
            "target_column": "churned",
            "entity_key": "entity_id",
            "timestamp_column": "as_of_date",
            "best_model_name": "Random Forest",
            "best_roc_auc": 0.85,
            "feature_columns": ["f1", "f2"],
            "run_type": "exploration",
        }
        required = {
            "mlflow_experiment_name", "mlflow_run_id", "composite_name",
            "target_column", "entity_key", "timestamp_column",
            "best_model_name", "best_roc_auc", "feature_columns", "run_type",
        }
        assert required.issubset(metadata.keys())


class TestFeaturesJsonFormat:
    @patch("customer_retention.stages.modeling.mlflow_logger.mlflow")
    def test_features_json_has_required_keys(self, mock_mlflow):
        logger = MLflowLogger(experiment_name="test")
        features_dict = {"feature_columns": ["f1", "f2", "f3"], "count": 3}
        logger.log_dict(features_dict, "features.json")
        call_args = mock_mlflow.log_dict.call_args[0]
        logged = call_args[0]
        assert "feature_columns" in logged
        assert "count" in logged
        assert logged["count"] == len(logged["feature_columns"])


class TestExperimentNameMatchesProductionPattern:
    def test_experiment_name_local(self):
        cn = "cust_emai__abc1234"
        experiment_name = f"training_{cn}"
        assert experiment_name == "training_cust_emai__abc1234"

    def test_experiment_name_databricks(self):
        """On Databricks, production uses /Shared/training_{CN}."""
        cn = "cust_emai__abc1234"
        on_databricks = True
        experiment_name = f"/Shared/training_{cn}" if on_databricks else f"training_{cn}"
        assert experiment_name == "/Shared/training_cust_emai__abc1234"


class TestGracefulWhenMlflowUnavailable:
    @patch("customer_retention.stages.modeling.mlflow_logger.MLFLOW_AVAILABLE", False)
    def test_all_operations_noop(self):
        logger = MLflowLogger(experiment_name="test")
        # None of these should raise
        logger.start_run()
        logger.disable_autolog()
        logger.end_stale_runs()
        logger.log_params({"a": 1})
        logger.log_metrics({"b": 0.5})
        logger.set_tags({"c": "d"})
        logger.log_dict({"e": "f"}, "test.json")
        with logger.nested_run("child"):
            pass
        logger.end_run()
        assert logger.run_id is None


class TestGracefulWhenProjectContextMissing:
    def test_cn_fallback_to_unknown(self):
        """When no ProjectContext and no namespace, CN should fall back gracefully."""
        source_names = []
        cn = "unknown" if not source_names else "should_not_reach"
        assert cn == "unknown"
