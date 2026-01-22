from abc import ABC

import pytest

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

requires_mlflow = pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="mlflow not installed")


class TestMLflowAdapterInterface:
    def test_mlflow_adapter_is_abstract(self):
        from customer_retention.integrations.adapters.mlflow import MLflowAdapter
        assert issubclass(MLflowAdapter, ABC)

    def test_local_mlflow_implements_interface(self):
        from customer_retention.integrations.adapters.mlflow import LocalMLflow, MLflowAdapter
        assert issubclass(LocalMLflow, MLflowAdapter)

    def test_databricks_mlflow_implements_interface(self):
        from customer_retention.integrations.adapters.mlflow import DatabricksMLflow, MLflowAdapter
        assert issubclass(DatabricksMLflow, MLflowAdapter)


@requires_mlflow
class TestLocalMLflowRun:
    def test_start_run_returns_run_id(self, tmp_path):
        from customer_retention.integrations.adapters.mlflow import LocalMLflow
        adapter = LocalMLflow(tracking_uri=str(tmp_path / "mlruns"))
        run_id = adapter.start_run(experiment_name="test_exp")
        assert isinstance(run_id, str)
        assert len(run_id) > 0
        adapter.end_run()

    def test_end_run_completes_without_error(self, tmp_path):
        from customer_retention.integrations.adapters.mlflow import LocalMLflow
        adapter = LocalMLflow(tracking_uri=str(tmp_path / "mlruns"))
        adapter.start_run(experiment_name="test_exp")
        adapter.end_run()


@requires_mlflow
class TestLocalMLflowLogging:
    def test_log_params_stores_params(self, tmp_path):
        from customer_retention.integrations.adapters.mlflow import LocalMLflow
        adapter = LocalMLflow(tracking_uri=str(tmp_path / "mlruns"))
        adapter.start_run(experiment_name="test_exp")
        adapter.log_params({"learning_rate": 0.01, "epochs": 10})
        adapter.end_run()

    def test_log_metrics_stores_metrics(self, tmp_path):
        from customer_retention.integrations.adapters.mlflow import LocalMLflow
        adapter = LocalMLflow(tracking_uri=str(tmp_path / "mlruns"))
        adapter.start_run(experiment_name="test_exp")
        adapter.log_metrics({"accuracy": 0.95, "loss": 0.05})
        adapter.end_run()


@requires_mlflow
class TestLocalMLflowModel:
    def test_log_model_returns_uri(self, tmp_path):
        from sklearn.linear_model import LogisticRegression

        from customer_retention.integrations.adapters.mlflow import LocalMLflow
        adapter = LocalMLflow(tracking_uri=str(tmp_path / "mlruns"))
        adapter.start_run(experiment_name="test_exp")
        model = LogisticRegression()
        uri = adapter.log_model(model, artifact_path="model")
        assert isinstance(uri, str)
        assert "model" in uri
        adapter.end_run()

    def test_load_model_returns_model(self, tmp_path):
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        from customer_retention.integrations.adapters.mlflow import LocalMLflow
        adapter = LocalMLflow(tracking_uri=str(tmp_path / "mlruns"))
        adapter.start_run(experiment_name="test_exp")
        model = LogisticRegression()
        model.fit(np.array([[1], [2], [3]]), [0, 1, 0])
        uri = adapter.log_model(model, artifact_path="model")
        adapter.end_run()
        loaded = adapter.load_model(uri)
        assert hasattr(loaded, "predict")


@requires_mlflow
class TestLocalMLflowRegistry:
    def test_log_model_with_registered_name(self, tmp_path):
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        from customer_retention.integrations.adapters.mlflow import LocalMLflow
        adapter = LocalMLflow(tracking_uri=str(tmp_path / "mlruns"))
        adapter.start_run(experiment_name="test_exp")
        model = LogisticRegression()
        model.fit(np.array([[1], [2], [3]]), [0, 1, 0])
        uri = adapter.log_model(model, artifact_path="model", registered_name="test_model")
        assert isinstance(uri, str)
        adapter.end_run()


class TestDatabricksMLflowMocked:
    def test_databricks_mlflow_requires_spark(self):
        from customer_retention.core.compat.detection import is_spark_available
        from customer_retention.integrations.adapters.mlflow import DatabricksMLflow
        if not is_spark_available():
            with pytest.raises(ImportError):
                DatabricksMLflow()


@requires_mlflow
class TestExperimentTrackerInit:
    def test_creates_experiment_on_init(self, tmp_path):
        from customer_retention.integrations.adapters.mlflow import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=str(tmp_path / "mlruns"), experiment_name="test_exp")
        assert tracker.experiment_name == "test_exp"

    def test_default_experiment_name(self, tmp_path):
        from customer_retention.integrations.adapters.mlflow import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=str(tmp_path / "mlruns"))
        assert tracker.experiment_name == "customer_retention"


@requires_mlflow
class TestExperimentTrackerExploration:
    @pytest.fixture
    def sample_findings(self):
        from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
        from customer_retention.core.config.column_config import ColumnType
        return ExplorationFindings(
            source_path="test_data.csv",
            source_format="csv",
            row_count=1000,
            column_count=5,
            memory_usage_mb=10.5,
            overall_quality_score=85.0,
            modeling_ready=True,
            columns={
                "id": ColumnFinding(
                    name="id", inferred_type=ColumnType.IDENTIFIER,
                    confidence=0.99, evidence=["unique"]
                ),
                "age": ColumnFinding(
                    name="age", inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                    confidence=0.95, evidence=["numeric"], cleaning_needed=True,
                    cleaning_recommendations=["impute_median"]
                ),
                "target": ColumnFinding(
                    name="target", inferred_type=ColumnType.TARGET,
                    confidence=0.99, evidence=["binary"]
                ),
            },
            target_column="target",
            identifier_columns=["id"],
        )

    def test_log_exploration_returns_run_id(self, tmp_path, sample_findings):
        from customer_retention.integrations.adapters.mlflow import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=str(tmp_path / "mlruns"))
        run_id = tracker.log_exploration(sample_findings)
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_log_exploration_logs_metrics(self, tmp_path, sample_findings):
        from customer_retention.integrations.adapters.mlflow import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=str(tmp_path / "mlruns"))
        run_id = tracker.log_exploration(sample_findings)
        run = mlflow.get_run(run_id)
        assert run.data.metrics["row_count"] == 1000
        assert run.data.metrics["column_count"] == 5
        assert run.data.metrics["overall_quality_score"] == 85.0

    def test_log_exploration_logs_params(self, tmp_path, sample_findings):
        from customer_retention.integrations.adapters.mlflow import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=str(tmp_path / "mlruns"))
        run_id = tracker.log_exploration(sample_findings)
        run = mlflow.get_run(run_id)
        assert run.data.params["source_path"] == "test_data.csv"
        assert run.data.params["target_column"] == "target"

    def test_log_exploration_sets_tags(self, tmp_path, sample_findings):
        from customer_retention.integrations.adapters.mlflow import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=str(tmp_path / "mlruns"))
        run_id = tracker.log_exploration(sample_findings)
        run = mlflow.get_run(run_id)
        assert run.data.tags["stage"] == "exploration"
        assert run.data.tags["modeling_ready"] == "True"

    def test_log_exploration_logs_column_type_counts(self, tmp_path, sample_findings):
        from customer_retention.integrations.adapters.mlflow import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=str(tmp_path / "mlruns"))
        run_id = tracker.log_exploration(sample_findings)
        run = mlflow.get_run(run_id)
        assert run.data.metrics["columns_needing_cleaning"] == 1


@requires_mlflow
class TestExperimentTrackerPipeline:
    @pytest.fixture
    def sample_pipeline(self):
        from customer_retention.analysis.recommendations import ImputeRecommendation, RecommendationPipeline
        pipeline = RecommendationPipeline([
            ImputeRecommendation(columns=["age"], strategy="median"),
            ImputeRecommendation(columns=["income"], strategy="mean"),
        ])
        return pipeline

    def test_log_pipeline_execution_returns_run_id(self, tmp_path, sample_pipeline):
        from customer_retention.integrations.adapters.mlflow import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=str(tmp_path / "mlruns"))
        run_id = tracker.log_pipeline_execution(sample_pipeline)
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_log_pipeline_logs_recommendation_count(self, tmp_path, sample_pipeline):
        from customer_retention.integrations.adapters.mlflow import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=str(tmp_path / "mlruns"))
        run_id = tracker.log_pipeline_execution(sample_pipeline)
        run = mlflow.get_run(run_id)
        assert run.data.params["recommendation_count"] == "2"

    def test_log_pipeline_logs_generated_code(self, tmp_path, sample_pipeline):
        import os

        from customer_retention.integrations.adapters.mlflow import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=str(tmp_path / "mlruns"))
        run_id = tracker.log_pipeline_execution(sample_pipeline)
        artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)
        assert os.path.exists(os.path.join(artifacts_path, "generated_code_local.py"))


@requires_mlflow
class TestExperimentTrackerSearch:
    def test_list_exploration_runs(self, tmp_path):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        from customer_retention.integrations.adapters.mlflow import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=str(tmp_path / "mlruns"))
        findings = ExplorationFindings(
            source_path="test.csv", source_format="csv", row_count=100, column_count=3
        )
        tracker.log_exploration(findings)
        runs = tracker.list_exploration_runs()
        assert len(runs) == 1
        assert runs[0]["data"]["tags"]["stage"] == "exploration"

    def test_get_best_run(self, tmp_path):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        from customer_retention.integrations.adapters.mlflow import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=str(tmp_path / "mlruns"))
        for score in [70, 90, 80]:
            findings = ExplorationFindings(
                source_path=f"test_{score}.csv", source_format="csv",
                row_count=100, column_count=3, overall_quality_score=float(score)
            )
            tracker.log_exploration(findings)
        best = tracker.get_best_run(metric="overall_quality_score")
        assert best["data"]["metrics"]["overall_quality_score"] == 90.0
