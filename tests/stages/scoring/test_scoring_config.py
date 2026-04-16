import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from customer_retention.stages.scoring.config import ScoringConfig


@pytest.fixture
def databricks_env(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
    monkeypatch.setenv("CR_CATALOG", "analytics")
    monkeypatch.setenv("CR_SCHEMA", "churn")
    monkeypatch.setenv("CR_EXPERIMENT_NAME", "customer_churn")
    monkeypatch.setenv("CR_EXPERIMENTS_DIR", "/Volumes/analytics/churn/experiments")


def _make_namespace(tmp_path):
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

    ns = RunNamespace(root=tmp_path, run_id="test-run-123")
    ns.setup()
    return ns


def _write_training_meta(ns, **overrides):
    metadata = {
        "pipeline_name": "customer_churn",
        "mlflow_experiment_name": "/Shared/training_cust_emails_prof__a1b2c3d",
        "mlflow_run_id": "run_abc",
        "composite_name": "cust_emails_prof__a1b2c3d",
        "target_column": "unsubscribed",
        "entity_key": "customer_id",
        "timestamp_column": "event_timestamp",
        "recommendations_hash": "abc123",
        "best_model_name": "RandomForestClassifier",
        "best_roc_auc": 0.9,
        "feature_columns": ["f1", "f2"],
        "logged_models": [{
            "artifact_path": "model_RandomForestClassifier",
            "model_uri": "runs:/run_abc/model_RandomForestClassifier",
            "flavor": "spark",
            "run_id": "run_abc",
            "display_name": "RandomForestClassifier",
            "wrapper_meta_artifact_path": None,
        }],
        "registered_model_name": "analytics.churn.model_cust_emails_prof__a1b2c3d",
        "registered_model_version": "3",
    }
    metadata.update(overrides)
    ns.training_metadata_path.write_text(json.dumps(metadata))
    return metadata


@pytest.fixture
def databricks_ns(databricks_env, tmp_path):
    ns = _make_namespace(tmp_path)
    _write_training_meta(ns)
    with patch("customer_retention.stages.scoring.config._discover_namespace", return_value=ns):
        yield ns


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

    def test_populates_logged_models_from_namespace(self, local_pipeline_dir, tmp_path):
        ns = _make_namespace(tmp_path)
        _write_training_meta(ns, best_model_name="logistic_regression", registered_model_name="")
        with patch("customer_retention.stages.scoring.config._discover_namespace", return_value=ns):
            config = ScoringConfig.from_local_config(local_pipeline_dir)
        assert config.best_model_name == "logistic_regression"
        assert len(config.logged_models) == 1
        assert config.mlflow_run_id == "run_abc"

    def test_empty_logged_models_when_no_namespace(self, local_pipeline_dir):
        with patch("customer_retention.stages.scoring.config._discover_namespace", return_value=None):
            config = ScoringConfig.from_local_config(local_pipeline_dir)
        assert config.logged_models == []
        assert config.best_model_name == ""
        assert config.mlflow_run_id == ""

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
    def test_reads_env_vars(self, databricks_ns):
        config = ScoringConfig.from_databricks()
        assert config.catalog == "analytics"
        assert config.schema == "churn"
        assert config.experiments_dir == Path("/Volumes/analytics/churn/experiments")

    def test_pipeline_name_from_training_metadata(self, databricks_ns):
        config = ScoringConfig.from_databricks()
        assert config.pipeline_name == "customer_churn"

    def test_pipeline_name_falls_back_to_mlflow_experiment_name(self, databricks_env, tmp_path):
        # Back-compat: older training_metadata.json files without the
        # pipeline_name field should fall back to mlflow_experiment_name.
        ns = _make_namespace(tmp_path)
        _write_training_meta(ns)
        meta = json.loads(ns.training_metadata_path.read_text())
        meta.pop("pipeline_name", None)
        ns.training_metadata_path.write_text(json.dumps(meta))
        with patch("customer_retention.stages.scoring.config._discover_namespace", return_value=ns):
            config = ScoringConfig.from_databricks()
        assert config.pipeline_name == "/Shared/training_cust_emails_prof__a1b2c3d"

    def test_discovers_target_from_metadata(self, databricks_ns):
        config = ScoringConfig.from_databricks()
        assert config.target_column == "unsubscribed"
        assert config.entity_key == "customer_id"
        assert config.recommendations_hash == "abc123"

    def test_discovers_composite_name_from_metadata(self, databricks_ns):
        config = ScoringConfig.from_databricks()
        assert config.composite_name == "cust_emails_prof__a1b2c3d"

    def test_populates_logged_models_from_metadata(self, databricks_ns):
        config = ScoringConfig.from_databricks()
        assert len(config.logged_models) == 1
        assert config.logged_models[0]["model_uri"] == "runs:/run_abc/model_RandomForestClassifier"
        assert config.logged_models[0]["flavor"] == "spark"

    def test_populates_registered_model_name(self, databricks_ns):
        config = ScoringConfig.from_databricks()
        assert config.registered_model_name == "analytics.churn.model_cust_emails_prof__a1b2c3d"

    def test_populates_mlflow_run_id(self, databricks_ns):
        config = ScoringConfig.from_databricks()
        assert config.mlflow_run_id == "run_abc"

    def test_populates_best_model_name(self, databricks_ns):
        config = ScoringConfig.from_databricks()
        assert config.best_model_name == "RandomForestClassifier"

    def test_raises_when_no_namespace(self, databricks_env):
        with patch("customer_retention.stages.scoring.config._discover_namespace", return_value=None):
            with pytest.raises(ValueError, match="No training_metadata.json"):
                ScoringConfig.from_databricks()

    def test_raises_when_metadata_missing(self, databricks_env, tmp_path):
        ns = _make_namespace(tmp_path)
        with patch("customer_retention.stages.scoring.config._discover_namespace", return_value=ns):
            with pytest.raises(ValueError, match="No training_metadata.json"):
                ScoringConfig.from_databricks()

    def test_raises_when_metadata_corrupt(self, databricks_env, tmp_path):
        ns = _make_namespace(tmp_path)
        ns.training_metadata_path.write_text("not valid json{{{")
        with patch("customer_retention.stages.scoring.config._discover_namespace", return_value=ns):
            with pytest.raises(ValueError, match="No training_metadata.json"):
                ScoringConfig.from_databricks()

    def test_falls_back_to_exploration_metadata(self, databricks_env, tmp_path):
        ns = _make_namespace(tmp_path)
        ns.exploration_metadata_path.write_text(json.dumps({
            "mlflow_experiment_name": "/Users/me/exploration_exp",
            "composite_name": "expl_cn",
            "target_column": "churned",
        }))
        with patch("customer_retention.stages.scoring.config._discover_namespace", return_value=ns):
            config = ScoringConfig.from_databricks()
        assert config.pipeline_name == "/Users/me/exploration_exp"
        assert config.composite_name == "expl_cn"
        assert config.target_column == "churned"

    def test_training_metadata_takes_priority_over_exploration(self, databricks_env, tmp_path):
        ns = _make_namespace(tmp_path)
        _write_training_meta(
            ns, pipeline_name="train_pipe", composite_name="train_cn",
            mlflow_experiment_name="/Shared/training_exp",
        )
        ns.exploration_metadata_path.write_text(json.dumps({
            "pipeline_name": "expl_pipe",
            "mlflow_experiment_name": "/Users/me/exploration_exp",
            "composite_name": "expl_cn",
        }))
        with patch("customer_retention.stages.scoring.config._discover_namespace", return_value=ns):
            config = ScoringConfig.from_databricks()
        assert config.pipeline_name == "train_pipe"
        assert config.composite_name == "train_cn"

    def test_defaults_for_missing_fields(self, databricks_env, tmp_path):
        ns = _make_namespace(tmp_path)
        ns.training_metadata_path.write_text(json.dumps({
            "mlflow_experiment_name": "/Shared/training_x",
        }))
        with patch("customer_retention.stages.scoring.config._discover_namespace", return_value=ns):
            config = ScoringConfig.from_databricks()
        assert config.target_column == "target"
        assert config.entity_key == "entity_id"
        assert config.recommendations_hash == ""
        assert config.composite_name == "/Shared/training_x"
        assert config.logged_models == []
        assert config.registered_model_name == ""

    def test_artifacts_path_from_namespace(self, databricks_ns):
        config = ScoringConfig.from_databricks()
        assert config.artifacts_path == databricks_ns.artifacts_dir("abc123")

    def test_artifacts_path_on_volumes(self, databricks_ns):
        config = ScoringConfig.from_databricks()
        assert str(config.artifacts_path).startswith(str(databricks_ns.root))

    def test_mlflow_tracking_uri_databricks(self, databricks_ns):
        config = ScoringConfig.from_databricks()
        assert config.mlflow_tracking_uri == "databricks"

    def test_pipeline_dir_empty_on_databricks(self, databricks_ns):
        config = ScoringConfig.from_databricks()
        assert config.pipeline_dir == Path()

    def test_feast_fields_empty_on_databricks(self, databricks_ns):
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

    def test_is_databricks_true(self, databricks_ns):
        config = ScoringConfig.from_databricks()
        assert config.is_databricks

    def test_scoring_output_dir(self, local_pipeline_dir):
        config = ScoringConfig.from_local_config(local_pipeline_dir)
        assert config.scoring_output_dir == Path("/tmp/experiments/data/scoring")


class TestArtifactPathContract:
    """NB08 saves artifacts via namespace.artifacts_dir(hash).
    NB11 resolves via ScoringConfig.artifacts_path which uses namespace.artifacts_dir(hash).
    These must match."""

    def test_databricks_artifacts_path_matches_namespace_root(self, databricks_ns):
        config = ScoringConfig.from_databricks()
        assert config.artifacts_path == databricks_ns.artifacts_dir("abc123")

    def test_artifacts_roundtrip_via_namespace(self, tmp_path):
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
        from customer_retention.transforms.artifact_store import ArtifactStore

        ns = RunNamespace(root=tmp_path, run_id="test-run")
        ns.setup()
        recs_hash = "abc12345"

        store = ArtifactStore(ns.artifacts_dir(recs_hash))
        store.register("scaler", "amount", {"mean": 100})
        store.save_manifest()

        scoring_artifacts_path = ns.artifacts_dir(recs_hash)
        loaded = ArtifactStore.from_manifest(scoring_artifacts_path / "manifest.yaml")
        assert loaded.has("amount_scaler")
        assert loaded.load("amount_scaler") == {"mean": 100}

    def test_default_subdir_when_no_hash(self, databricks_env, tmp_path):
        ns = _make_namespace(tmp_path)
        _write_training_meta(ns, recommendations_hash="")
        with patch("customer_retention.stages.scoring.config._discover_namespace", return_value=ns):
            config = ScoringConfig.from_databricks()
        assert config.artifacts_path == ns.artifacts_dir("")
