import json
from pathlib import Path
from unittest.mock import patch

import pytest

from customer_retention.stages.scoring import ScoringConfig
from customer_retention.stages.scoring.pipeline_discovery import find_generated_pipeline_dir


@pytest.fixture
def databricks_env(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
    monkeypatch.setenv("CR_CATALOG", "analytics")
    monkeypatch.setenv("CR_SCHEMA", "churn")
    monkeypatch.setenv("CR_EXPERIMENT_NAME", "customer_churn")
    monkeypatch.setenv("CR_EXPERIMENTS_DIR", "/Volumes/analytics/churn/experiments")


def _make_namespace(tmp_path):
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

    ns = RunNamespace(root=tmp_path, run_id="test-run-discovery")
    ns.setup()
    return ns


def _write_meta(ns, **fields):
    ns.training_metadata_path.write_text(json.dumps(fields))


def _make_pipeline_tree(workspace_root: Path, pipeline_name: str, composite_name: str) -> Path:
    pipeline_dir = workspace_root / "generated_pipelines" / "databricks" / pipeline_name
    silver = pipeline_dir / "silver"
    silver.mkdir(parents=True)
    (silver / f"silver_featureset_{composite_name}.py").write_text("# stub")
    return pipeline_dir


def _scoring_config(databricks_env, ns) -> ScoringConfig:
    with patch(
        "customer_retention.stages.scoring.config._discover_namespace",
        return_value=ns,
    ):
        return ScoringConfig.from_databricks()


class TestFindGeneratedPipelineDir:
    def test_returns_pipeline_name_path_when_folder_exists(self, databricks_env, tmp_path):
        ws = tmp_path / "ws"
        expected = _make_pipeline_tree(ws, "customer_churn", "cust_emai_aggr__26e8271")
        ns = _make_namespace(tmp_path)
        _write_meta(
            ns,
            pipeline_name="customer_churn",
            mlflow_experiment_name="/Shared/training_cust_emai_aggr__26e8271",
            composite_name="cust_emai_aggr__26e8271",
        )
        config = _scoring_config(databricks_env, ns)

        result = find_generated_pipeline_dir(workspace_root=ws, scoring_config=config)

        assert result == expected

    def test_falls_back_to_composite_scan_when_pipeline_name_path_missing(
        self, databricks_env, tmp_path,
    ):
        # Old metadata: pipeline_name field absent → ScoringConfig falls back to
        # mlflow_experiment_name (a workspace path that won't match a folder).
        # Resolver must find the actual pipeline folder by scanning silver/.
        ws = tmp_path / "ws"
        expected = _make_pipeline_tree(ws, "customer_churn", "cust_emai_aggr__26e8271")
        ns = _make_namespace(tmp_path)
        _write_meta(
            ns,
            mlflow_experiment_name="/Shared/training_cust_emai_aggr__26e8271",
            composite_name="cust_emai_aggr__26e8271",
        )
        config = _scoring_config(databricks_env, ns)
        assert config.pipeline_name == "/Shared/training_cust_emai_aggr__26e8271"

        result = find_generated_pipeline_dir(workspace_root=ws, scoring_config=config)

        assert result == expected

    def test_picks_correct_folder_when_multiple_pipelines_exist(self, databricks_env, tmp_path):
        ws = tmp_path / "ws"
        _make_pipeline_tree(ws, "other_pipeline", "other_cn__abc1234")
        expected = _make_pipeline_tree(ws, "customer_churn", "cust_emai_aggr__26e8271")
        _make_pipeline_tree(ws, "third_pipeline", "third_cn__def5678")
        ns = _make_namespace(tmp_path)
        _write_meta(
            ns,
            mlflow_experiment_name="/Shared/training_cust_emai_aggr__26e8271",
            composite_name="cust_emai_aggr__26e8271",
        )
        config = _scoring_config(databricks_env, ns)

        result = find_generated_pipeline_dir(workspace_root=ws, scoring_config=config)

        assert result == expected

    def test_raises_when_base_dir_missing(self, databricks_env, tmp_path):
        ns = _make_namespace(tmp_path)
        _write_meta(
            ns,
            pipeline_name="customer_churn",
            mlflow_experiment_name="/Shared/x",
            composite_name="cust_emai_aggr__26e8271",
        )
        config = _scoring_config(databricks_env, ns)

        with pytest.raises(FileNotFoundError, match="generated_pipelines"):
            find_generated_pipeline_dir(workspace_root=tmp_path / "no_ws", scoring_config=config)

    def test_raises_when_no_match_found(self, databricks_env, tmp_path):
        ws = tmp_path / "ws"
        _make_pipeline_tree(ws, "wrong_pipeline", "wrong_cn__abcdef0")
        ns = _make_namespace(tmp_path)
        _write_meta(
            ns,
            mlflow_experiment_name="/Shared/training_other",
            composite_name="cust_emai_aggr__26e8271",
        )
        config = _scoring_config(databricks_env, ns)

        with pytest.raises(FileNotFoundError) as excinfo:
            find_generated_pipeline_dir(workspace_root=ws, scoring_config=config)
        msg = str(excinfo.value)
        assert "cust_emai_aggr__26e8271" in msg
        assert "PIPELINE_DIR" in msg

    def test_pipeline_name_with_slash_skips_direct_path_attempt(self, databricks_env, tmp_path):
        # An MLflow experiment path like "/Shared/x" must not be joined to base
        # (would produce /Workspace/.../databricks/Shared/x — the user's bug).
        ws = tmp_path / "ws"
        expected = _make_pipeline_tree(ws, "real_pipe", "cust_emai_aggr__26e8271")
        rogue = ws / "generated_pipelines" / "databricks" / "Shared" / "training_x"
        rogue.mkdir(parents=True)
        ns = _make_namespace(tmp_path)
        _write_meta(
            ns,
            pipeline_name="/Shared/training_x",
            mlflow_experiment_name="/Shared/training_x",
            composite_name="cust_emai_aggr__26e8271",
        )
        config = _scoring_config(databricks_env, ns)

        result = find_generated_pipeline_dir(workspace_root=ws, scoring_config=config)

        assert result == expected

    def test_target_parameter_switches_subdir(self, databricks_env, tmp_path):
        ws = tmp_path / "ws"
        local_dir = ws / "generated_pipelines" / "local" / "customer_churn"
        (local_dir / "silver").mkdir(parents=True)
        (local_dir / "silver" / "silver_featureset_cust_emai_aggr__26e8271.py").write_text("# stub")
        ns = _make_namespace(tmp_path)
        _write_meta(
            ns,
            pipeline_name="customer_churn",
            mlflow_experiment_name="/Shared/x",
            composite_name="cust_emai_aggr__26e8271",
        )
        config = _scoring_config(databricks_env, ns)

        result = find_generated_pipeline_dir(workspace_root=ws, scoring_config=config, target="local")

        assert result == local_dir

    def test_raises_when_composite_name_missing_and_pipeline_name_unusable(
        self, databricks_env, tmp_path,
    ):
        ws = tmp_path / "ws"
        (ws / "generated_pipelines" / "databricks").mkdir(parents=True)
        ns = _make_namespace(tmp_path)
        _write_meta(ns, mlflow_experiment_name="/Shared/x")
        config = _scoring_config(databricks_env, ns)

        with pytest.raises(FileNotFoundError, match="PIPELINE_DIR"):
            find_generated_pipeline_dir(workspace_root=ws, scoring_config=config)
