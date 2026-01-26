"""Integration tests for experiments directory configuration.

Tests that the pipeline generator correctly handles:
- Custom experiments_dir parameter
- CR_EXPERIMENTS_DIR environment variable override
- Correct path resolution for all artifacts (data, mlruns, feast, findings)
- Generated modules can load and execute with custom paths
"""
import ast
import os
import subprocess
import sys

import pandas as pd
import pytest
import yaml


@pytest.fixture
def experiments_setup(tmp_path):
    """Create a complete test setup with findings, data, and experiments structure."""
    # Create directory structure
    project_root = tmp_path / "project"
    project_root.mkdir()
    experiments_dir = project_root / "experiments"
    experiments_dir.mkdir()
    findings_dir = experiments_dir / "findings"
    findings_dir.mkdir()
    output_dir = project_root / "generated_pipelines"
    output_dir.mkdir()

    # Create a fake pyproject.toml so _find_project_root works
    (project_root / "pyproject.toml").write_text("[project]\nname = 'test'\n")

    # Create test data in findings dir
    test_df = pd.DataFrame({
        "customer_id": ["A001", "A002", "A003", "A004", "A005"],
        "revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
        "orders": [5, 10, 7, 15, 12],
        "target": [0, 1, 0, 1, 0]
    })
    data_path = findings_dir / "customers.parquet"
    test_df.to_parquet(data_path, index=False)

    # Create multi-dataset findings
    multi_dataset = {
        "datasets": {
            "customers": {
                "name": "customers",
                "findings_path": "customers_findings.yaml",
                "source_path": str(data_path),
                "granularity": "entity_level",
                "row_count": 5,
                "column_count": 4,
                "excluded": False
            }
        },
        "relationships": [],
        "primary_entity_dataset": "customers",
        "event_datasets": [],
        "excluded_datasets": []
    }
    (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

    # Create source findings
    customers_findings = {
        "source_path": str(data_path),
        "source_format": "parquet",
        "row_count": 5,
        "column_count": 4,
        "columns": {
            "customer_id": {
                "name": "customer_id",
                "inferred_type": "identifier",
                "confidence": 0.95,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": []
            },
            "revenue": {
                "name": "revenue",
                "inferred_type": "numeric_continuous",
                "confidence": 0.9,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": []
            },
            "orders": {
                "name": "orders",
                "inferred_type": "numeric_discrete",
                "confidence": 0.9,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": []
            },
            "target": {
                "name": "target",
                "inferred_type": "binary",
                "confidence": 0.99,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": []
            }
        },
        "target_column": "target",
        "identifier_columns": ["customer_id"]
    }
    (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers_findings))

    return {
        "project_root": project_root,
        "experiments_dir": experiments_dir,
        "findings_dir": findings_dir,
        "output_dir": output_dir,
        "data_path": data_path,
        "tmp_path": tmp_path
    }


class TestExperimentsDirectoryConfiguration:
    """Tests for experiments directory parameter handling."""

    def test_generator_accepts_experiments_dir_parameter(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="my_custom_experiments"
        )
        assert generator._experiments_dir == "my_custom_experiments"

    def test_generator_accepts_none_experiments_dir(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline"
        )
        assert generator._experiments_dir is None

    def test_generated_config_contains_experiments_dir(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="custom_exp"
        )
        generator.generate()

        config_content = (experiments_setup["output_dir"] / "config.py").read_text()
        assert "EXPERIMENTS_DIR" in config_content
        assert "custom_exp" in config_content
        assert "CR_EXPERIMENTS_DIR" in config_content

    def test_generated_config_default_experiments_dir(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline"
        )
        generator.generate()

        config_content = (experiments_setup["output_dir"] / "config.py").read_text()
        assert '_default_experiments = "experiments"' in config_content


class TestGeneratedConfigPaths:
    """Tests for correct path configuration in generated config.py."""

    def test_mlflow_tracking_uri_uses_experiments_dir(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        config_content = (experiments_setup["output_dir"] / "config.py").read_text()
        assert "EXPERIMENTS_DIR / \"mlruns\"" in config_content

    def test_feast_repo_path_uses_experiments_dir(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        config_content = (experiments_setup["output_dir"] / "config.py").read_text()
        assert "EXPERIMENTS_DIR / \"feature_repo\"" in config_content

    def test_data_paths_use_experiments_dir(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        config_content = (experiments_setup["output_dir"] / "config.py").read_text()
        assert 'EXPERIMENTS_DIR / "data" / "bronze"' in config_content
        assert 'EXPERIMENTS_DIR / "data" / "silver"' in config_content
        assert 'EXPERIMENTS_DIR / "data" / "gold"' in config_content

    def test_findings_dir_uses_experiments_dir(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        config_content = (experiments_setup["output_dir"] / "config.py").read_text()
        assert 'FINDINGS_DIR = EXPERIMENTS_DIR / "findings"' in config_content


class TestGeneratedModulesExecutable:
    """Tests that generated modules can be imported and executed."""

    def test_generated_config_is_valid_python(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        config_content = (experiments_setup["output_dir"] / "config.py").read_text()
        ast.parse(config_content)

    def test_all_generated_python_files_are_valid(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        py_files = list(experiments_setup["output_dir"].rglob("*.py"))
        assert len(py_files) > 0

        for py_file in py_files:
            content = py_file.read_text()
            ast.parse(content)

    def test_config_can_be_imported_and_executed(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        # Run config.py in a subprocess to verify it imports correctly
        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{experiments_setup["output_dir"]}')
import config
print(f'PIPELINE_NAME: {{config.PIPELINE_NAME}}')
print(f'EXPERIMENTS_DIR: {{config.EXPERIMENTS_DIR}}')
print(f'MLFLOW_TRACKING_URI: {{config.MLFLOW_TRACKING_URI}}')
print(f'FEAST_REPO_PATH: {{config.FEAST_REPO_PATH}}')
print(f'FINDINGS_DIR: {{config.FINDINGS_DIR}}')
"""],
            capture_output=True,
            text=True,
            cwd=str(experiments_setup["project_root"])
        )
        assert result.returncode == 0, f"Config import failed: {result.stderr}"
        assert "test_pipeline" in result.stdout
        assert "experiments" in result.stdout


class TestEnvironmentVariableOverride:
    """Tests for CR_EXPERIMENTS_DIR environment variable override."""

    def test_env_var_overrides_experiments_dir(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="default_experiments"
        )
        generator.generate()

        custom_path = "/dbfs/mnt/catalog/my_experiments"
        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{experiments_setup["output_dir"]}')
import config
print(f'EXPERIMENTS_DIR: {{config.EXPERIMENTS_DIR}}')
print(f'MLFLOW_TRACKING_URI: {{config.MLFLOW_TRACKING_URI}}')
print(f'FEAST_REPO_PATH: {{config.FEAST_REPO_PATH}}')
"""],
            capture_output=True,
            text=True,
            env={**os.environ, "CR_EXPERIMENTS_DIR": custom_path},
            cwd=str(experiments_setup["project_root"])
        )
        assert result.returncode == 0, f"Config import failed: {result.stderr}"
        assert custom_path in result.stdout
        assert f"{custom_path}/mlruns" in result.stdout or f"{custom_path}\\mlruns" in result.stdout

    def test_mlflow_tracking_uri_env_var_takes_precedence(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        custom_mlflow_uri = "databricks://my-workspace"
        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{experiments_setup["output_dir"]}')
import config
print(f'MLFLOW_TRACKING_URI: {{config.MLFLOW_TRACKING_URI}}')
"""],
            capture_output=True,
            text=True,
            env={**os.environ, "MLFLOW_TRACKING_URI": custom_mlflow_uri},
            cwd=str(experiments_setup["project_root"])
        )
        assert result.returncode == 0, f"Config import failed: {result.stderr}"
        assert custom_mlflow_uri in result.stdout


class TestRunAllSetupFunction:
    """Tests for the setup_experiments_dir function in run_all.py."""

    def test_run_all_contains_setup_experiments_dir(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        run_all_content = (experiments_setup["output_dir"] / "run_all.py").read_text()
        assert "def setup_experiments_dir():" in run_all_content
        assert "EXPERIMENTS_DIR.mkdir" in run_all_content
        assert '"data" / "bronze"' in run_all_content
        assert '"data" / "silver"' in run_all_content
        assert '"data" / "gold"' in run_all_content
        assert '"mlruns"' in run_all_content

    def test_pipeline_runner_contains_setup_experiments_dir(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        runner_content = (experiments_setup["output_dir"] / "pipeline_runner.py").read_text()
        assert "def setup_experiments_dir():" in runner_content


class TestBronzeLayerPathResolution:
    """Tests that bronze layer correctly resolves source paths from experiments dir."""

    def test_bronze_source_path_uses_findings_dir(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        # Update findings to use relative path (the expected format)
        customers_findings_path = experiments_setup["findings_dir"] / "customers_findings.yaml"
        customers_findings = yaml.safe_load(customers_findings_path.read_text())
        customers_findings["source_path"] = "customers.parquet"
        customers_findings_path.write_text(yaml.dump(customers_findings))

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        config_content = (experiments_setup["output_dir"] / "config.py").read_text()
        assert "FINDINGS_DIR" in config_content
        # Source path should use FINDINGS_DIR
        assert 'str(FINDINGS_DIR / "customers.parquet")' in config_content

    def test_bronze_can_load_source_data(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        # Update findings to use relative path from findings_dir
        customers_findings_path = experiments_setup["findings_dir"] / "customers_findings.yaml"
        customers_findings = yaml.safe_load(customers_findings_path.read_text())
        customers_findings["source_path"] = "customers.parquet"
        customers_findings_path.write_text(yaml.dump(customers_findings))

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        # Test that bronze layer can load data
        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{experiments_setup["output_dir"]}')
from config import SOURCES, FINDINGS_DIR
import pandas as pd
from pathlib import Path

for name, src in SOURCES.items():
    path = Path(src['path'])
    print(f'Source {{name}}: {{path}}')
    print(f'  Exists: {{path.exists()}}')
    if path.exists():
        df = pd.read_parquet(path)
        print(f'  Rows: {{len(df)}}')
"""],
            capture_output=True,
            text=True,
            cwd=str(experiments_setup["project_root"])
        )
        assert result.returncode == 0, f"Bronze load test failed: {result.stderr}"
        assert "Exists: True" in result.stdout
        assert "Rows: 5" in result.stdout


class TestScoringPathConfiguration:
    """Tests for scoring module path configuration."""

    def test_scoring_uses_experiments_dir_for_predictions(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        scoring_content = (experiments_setup["output_dir"] / "scoring" / "run_scoring.py").read_text()
        assert "EXPERIMENTS_DIR" in scoring_content
        assert '"data" / "scoring"' in scoring_content


class TestDatabricksCompatibility:
    """Tests for Databricks-specific path configurations."""

    def test_dbfs_path_works_with_env_var(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        dbfs_path = "/dbfs/mnt/my_catalog/experiments"
        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{experiments_setup["output_dir"]}')
import config

# Verify all paths use the DBFS location
print(f'EXPERIMENTS_DIR: {{config.EXPERIMENTS_DIR}}')
print(f'Bronze path: {{config.get_bronze_path("test")}}')
print(f'Silver path: {{config.get_silver_path()}}')
print(f'Gold path: {{config.get_gold_path()}}')
print(f'Feast path: {{config.get_feast_data_path()}}')
"""],
            capture_output=True,
            text=True,
            env={**os.environ, "CR_EXPERIMENTS_DIR": dbfs_path},
            cwd=str(experiments_setup["project_root"])
        )
        assert result.returncode == 0, f"DBFS path test failed: {result.stderr}"
        assert dbfs_path in result.stdout
        # All paths should contain the dbfs_path
        assert result.stdout.count(dbfs_path) >= 5

    def test_unity_catalog_path_works_with_env_var(self, experiments_setup):
        from customer_retention.generators.pipeline_generator import PipelineGenerator

        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        uc_path = "/Volumes/my_catalog/my_schema/experiments"
        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{experiments_setup["output_dir"]}')
import config
print(f'EXPERIMENTS_DIR: {{config.EXPERIMENTS_DIR}}')
print(f'FINDINGS_DIR: {{config.FINDINGS_DIR}}')
"""],
            capture_output=True,
            text=True,
            env={**os.environ, "CR_EXPERIMENTS_DIR": uc_path},
            cwd=str(experiments_setup["project_root"])
        )
        assert result.returncode == 0, f"Unity Catalog path test failed: {result.stderr}"
        assert uc_path in result.stdout
