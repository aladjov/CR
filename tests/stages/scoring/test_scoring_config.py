import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from customer_retention.stages.scoring.config import ScoringConfig


@pytest.fixture
def databricks_env(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
    monkeypatch.setenv("CR_CATALOG", "analytics")
    monkeypatch.setenv("CR_SCHEMA", "churn")
    monkeypatch.setenv("CR_EXPERIMENT_NAME", "customer_churn")
    monkeypatch.setenv("CR_EXPERIMENTS_DIR", "/Volumes/analytics/churn/experiments")


@pytest.fixture
def mock_mlflow_client():
    client = MagicMock()
    experiment = MagicMock()
    experiment.experiment_id = "123"
    client.get_experiment_by_name.return_value = experiment
    run = MagicMock()
    run.data.tags = {
        "recommendations_hash": "abc123",
        "target_column": "unsubscribed",
        "entity_key": "customer_id",
        "timestamp_column": "event_timestamp",
        "composite_name": "cust_emails_prof__a1b2c3d",
    }
    run.data.params = {}
    client.search_runs.return_value = [run]
    return client


@pytest.fixture
def local_pipeline_dir(tmp_path):
    pipeline_dir = tmp_path / "customer_churn"
    pipeline_dir.mkdir()
    config_content = textwrap.dedent("""\
        from pathlib import Path
        PIPELINE_NAME = "customer_churn"
        COMPOSITE_NAME = "cust_emails_prof__a1b2c3d"
        TARGET_COLUMN = "unsubscribed"
        RECOMMENDATIONS_HASH = "4131c25b"
        FEAST_ENTITY_KEY = "customer_id"
        FEAST_TIMESTAMP_COL = "event_timestamp"
        FEAST_REPO_PATH = "/tmp/feast_repo"
        FEAST_FEATURE_VIEW = "featureset_cust_emails_prof__a1b2c3d"
        MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
        EXPERIMENTS_DIR = Path("/tmp/experiments")
        PRODUCTION_DIR = Path("/tmp/production")
        ARTIFACTS_PATH = "/tmp/production/artifacts/4131c25b"
    """)
    (pipeline_dir / "config.py").write_text(config_content)
    return pipeline_dir


class TestFromLocalConfig:
    def test_reads_generated_config(self, local_pipeline_dir):
        config = ScoringConfig.from_local_config(local_pipeline_dir)
        assert config.pipeline_name == "customer_churn"
        assert config.composite_name == "cust_emails_prof__a1b2c3d"
        assert config.target_column == "unsubscribed"
        assert config.entity_key == "customer_id"
        assert config.timestamp_column == "event_timestamp"
        assert config.recommendations_hash == "4131c25b"
        assert config.feast_repo_path == "/tmp/feast_repo"
        assert config.feast_feature_view == "featureset_cust_emails_prof__a1b2c3d"
        assert config.mlflow_tracking_uri == "sqlite:///mlruns.db"
        assert config.experiments_dir == Path("/tmp/experiments")
        assert config.production_dir == Path("/tmp/production")
        assert config.artifacts_path == Path("/tmp/production/artifacts/4131c25b")

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ScoringConfig.from_local_config(tmp_path / "nonexistent")

    def test_missing_config_py_raises(self, tmp_path):
        empty_dir = tmp_path / "empty_pipeline"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            ScoringConfig.from_local_config(empty_dir)

    def test_stores_pipeline_dir(self, local_pipeline_dir):
        config = ScoringConfig.from_local_config(local_pipeline_dir)
        assert config.pipeline_dir == local_pipeline_dir

    def test_catalog_and_schema_empty_for_local(self, local_pipeline_dir):
        config = ScoringConfig.from_local_config(local_pipeline_dir)
        assert config.catalog == ""
        assert config.schema == ""

    def test_falls_back_to_pipeline_name_without_composite(self, tmp_path):
        pipeline_dir = tmp_path / "legacy"
        pipeline_dir.mkdir()
        config_content = textwrap.dedent("""\
            from pathlib import Path
            PIPELINE_NAME = "legacy_pipe"
            TARGET_COLUMN = "target"
            RECOMMENDATIONS_HASH = None
            FEAST_ENTITY_KEY = "id"
            FEAST_TIMESTAMP_COL = "event_timestamp"
            FEAST_REPO_PATH = "/tmp/feast"
            FEAST_FEATURE_VIEW = "legacy_features"
            MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
            EXPERIMENTS_DIR = Path("/tmp/exp")
            PRODUCTION_DIR = Path("/tmp/prod")
            ARTIFACTS_PATH = "/tmp/prod/artifacts/default"
        """)
        (pipeline_dir / "config.py").write_text(config_content)
        config = ScoringConfig.from_local_config(pipeline_dir)
        assert config.composite_name == "legacy_pipe"


class TestFromDatabricks:
    def test_reads_env_vars(self, databricks_env, mock_mlflow_client):
        with patch("customer_retention.stages.scoring.config.MlflowClient", return_value=mock_mlflow_client):
            config = ScoringConfig.from_databricks()
        assert config.catalog == "analytics"
        assert config.schema == "churn"
        assert config.pipeline_name == "customer_churn"
        assert config.experiments_dir == Path("/Volumes/analytics/churn/experiments")

    def test_discovers_target_from_mlflow(self, databricks_env, mock_mlflow_client):
        with patch("customer_retention.stages.scoring.config.MlflowClient", return_value=mock_mlflow_client):
            config = ScoringConfig.from_databricks()
        assert config.target_column == "unsubscribed"
        assert config.entity_key == "customer_id"
        assert config.recommendations_hash == "abc123"

    def test_discovers_composite_name_from_tags(self, databricks_env, mock_mlflow_client):
        with patch("customer_retention.stages.scoring.config.MlflowClient", return_value=mock_mlflow_client):
            config = ScoringConfig.from_databricks()
        assert config.composite_name == "cust_emails_prof__a1b2c3d"

    def test_missing_experiment_raises(self, databricks_env):
        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        client.search_experiments.return_value = []
        with patch("customer_retention.stages.scoring.config.MlflowClient", return_value=client):
            with pytest.raises(ValueError, match="not found"):
                ScoringConfig.from_databricks()

    def test_from_databricks_finds_experiment_by_search(self, databricks_env):
        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        experiment = MagicMock()
        experiment.experiment_id = "456"
        experiment.name = "/Users/someone/customer_churn"
        experiment.creation_time = 1000
        client.search_experiments.return_value = [experiment]
        run = MagicMock()
        run.data.tags = {
            "target_column": "churned",
            "entity_key": "customer_id",
            "timestamp_column": "event_timestamp",
            "recommendations_hash": "abc123",
            "composite_name": "test__abc1234",
        }
        run.data.params = {}
        client.search_runs.return_value = [run]
        with patch("customer_retention.stages.scoring.config.MlflowClient", return_value=client):
            config = ScoringConfig.from_databricks()
        assert config.pipeline_name == "customer_churn"
        assert config.target_column == "churned"

    def test_from_databricks_finds_experiment_via_persisted_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.delenv("CR_EXPERIMENT_NAME", raising=False)
        monkeypatch.setenv("CR_CATALOG", "analytics")
        monkeypatch.setenv("CR_SCHEMA", "churn")
        monkeypatch.setenv("CR_EXPERIMENTS_DIR", "/Volumes/analytics/churn/experiments")
        monkeypatch.setenv("CR_WORKSPACE_PATH", "Users/me/project")
        config_file = tmp_path / ".churnkit_config.json"
        import json
        config_file.write_text(json.dumps({"experiment_name": "/Users/me/project/customer_churn"}))
        monkeypatch.setattr(
            "customer_retention.core.config.experiments._workspace_config_path",
            lambda wp: config_file,
        )
        client = MagicMock()
        experiment = MagicMock()
        experiment.experiment_id = "789"
        client.get_experiment_by_name.return_value = experiment
        run = MagicMock()
        run.data.tags = {"target_column": "churned", "composite_name": "test__abc1234"}
        run.data.params = {}
        client.search_runs.return_value = [run]
        with patch("customer_retention.stages.scoring.config.MlflowClient", return_value=client):
            config = ScoringConfig.from_databricks()
        client.get_experiment_by_name.assert_called_with("/Users/me/project/customer_churn")
        assert config.pipeline_name == "/Users/me/project/customer_churn"

    def test_from_databricks_raises_when_no_experiment_anywhere(self, databricks_env):
        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        client.search_experiments.return_value = []
        with patch("customer_retention.stages.scoring.config.MlflowClient", return_value=client):
            with pytest.raises(ValueError, match="not found"):
                ScoringConfig.from_databricks()

    def test_empty_runs_raises(self, databricks_env):
        client = MagicMock()
        experiment = MagicMock()
        experiment.experiment_id = "123"
        client.get_experiment_by_name.return_value = experiment
        client.search_runs.return_value = []
        with patch("customer_retention.stages.scoring.config.MlflowClient", return_value=client):
            with pytest.raises(ValueError, match="No runs found"):
                ScoringConfig.from_databricks()

    def test_defaults_for_missing_tags(self, databricks_env):
        client = MagicMock()
        experiment = MagicMock()
        experiment.experiment_id = "123"
        client.get_experiment_by_name.return_value = experiment
        run = MagicMock()
        run.data.tags = {}
        run.data.params = {"target_column": "churn"}
        client.search_runs.return_value = [run]
        with patch("customer_retention.stages.scoring.config.MlflowClient", return_value=client):
            config = ScoringConfig.from_databricks()
        assert config.target_column == "churn"
        assert config.entity_key == "customer_id"
        assert config.recommendations_hash == ""
        assert config.composite_name == "customer_churn"

    def test_artifacts_path_on_volumes(self, databricks_env, mock_mlflow_client):
        with patch("customer_retention.stages.scoring.config.MlflowClient", return_value=mock_mlflow_client):
            config = ScoringConfig.from_databricks()
        assert str(config.artifacts_path).startswith("/Volumes/")

    def test_mlflow_tracking_uri_databricks(self, databricks_env, mock_mlflow_client):
        with patch("customer_retention.stages.scoring.config.MlflowClient", return_value=mock_mlflow_client):
            config = ScoringConfig.from_databricks()
        assert config.mlflow_tracking_uri == "databricks"

    def test_pipeline_dir_empty_on_databricks(self, databricks_env, mock_mlflow_client):
        with patch("customer_retention.stages.scoring.config.MlflowClient", return_value=mock_mlflow_client):
            config = ScoringConfig.from_databricks()
        assert config.pipeline_dir == Path()

    def test_feast_fields_empty_on_databricks(self, databricks_env, mock_mlflow_client):
        with patch("customer_retention.stages.scoring.config.MlflowClient", return_value=mock_mlflow_client):
            config = ScoringConfig.from_databricks()
        assert config.feast_repo_path == ""
        assert config.feast_feature_view == ""


class TestScoringConfigProperties:
    def test_original_column(self, local_pipeline_dir):
        config = ScoringConfig.from_local_config(local_pipeline_dir)
        assert config.original_column == "original_unsubscribed"

    def test_is_databricks_false_for_local(self, local_pipeline_dir):
        config = ScoringConfig.from_local_config(local_pipeline_dir)
        assert not config.is_databricks

    def test_is_databricks_true(self, databricks_env, mock_mlflow_client):
        with patch("customer_retention.stages.scoring.config.MlflowClient", return_value=mock_mlflow_client):
            config = ScoringConfig.from_databricks()
        assert config.is_databricks

    def test_scoring_output_dir(self, local_pipeline_dir):
        config = ScoringConfig.from_local_config(local_pipeline_dir)
        assert config.scoring_output_dir == Path("/tmp/experiments/data/scoring")
