import ast
import os
import subprocess
import sys

import yaml

from customer_retention.generators.pipeline_generator import PipelineGenerator

_CONFIG_ENV_VARS = {"CR_EXPERIMENTS_DIR", "CR_PRODUCTION_DIR", "CR_DOCS_BASE_URL", "MLFLOW_TRACKING_URI"}


def _subprocess_env(**overrides):
    env = {k: v for k, v in os.environ.items() if k not in _CONFIG_ENV_VARS}
    env.update(overrides)
    return env


class TestExperimentsDirectoryConfiguration:

    def test_generator_accepts_experiments_dir_parameter(self, experiments_setup):
        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="my_custom_experiments"
        )
        assert generator._experiments_dir == "my_custom_experiments"

    def test_generator_accepts_none_experiments_dir(self, experiments_setup):
        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline"
        )
        assert generator._experiments_dir is None

    def test_generated_config_contains_experiments_dir(self, experiments_setup):
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
        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline"
        )
        generator.generate()

        config_content = (experiments_setup["output_dir"] / "config.py").read_text()
        assert '_default_experiments = "experiments"' in config_content


class TestGeneratedConfigPaths:

    def test_mlflow_tracking_uri_uses_experiments_dir(self, experiments_setup):
        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        config_content = (experiments_setup["output_dir"] / "config.py").read_text()
        assert "EXPERIMENTS_DIR / \"mlruns\"" in config_content

    def test_feast_repo_path_uses_production_dir(self, experiments_setup):
        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        config_content = (experiments_setup["output_dir"] / "config.py").read_text()
        assert "PRODUCTION_DIR / \"feature_repo\"" in config_content

    def test_data_paths_use_production_dir(self, experiments_setup):
        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

        config_content = (experiments_setup["output_dir"] / "config.py").read_text()
        assert 'PRODUCTION_DIR / "data" / "bronze"' in config_content
        assert 'PRODUCTION_DIR / "data" / "silver"' in config_content
        assert 'PRODUCTION_DIR / "data" / "gold"' in config_content

    def test_findings_dir_uses_experiments_dir(self, experiments_setup):
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

    def test_generated_config_is_valid_python(self, experiments_setup):
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
        generator = PipelineGenerator(
            findings_dir=str(experiments_setup["findings_dir"]),
            output_dir=str(experiments_setup["output_dir"]),
            pipeline_name="test_pipeline",
            experiments_dir="experiments"
        )
        generator.generate()

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

    def test_env_var_overrides_experiments_dir(self, experiments_setup):
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
            env=_subprocess_env(CR_EXPERIMENTS_DIR=custom_path),
            cwd=str(experiments_setup["project_root"])
        )
        assert result.returncode == 0, f"Config import failed: {result.stderr}"
        assert custom_path in result.stdout
        assert f"{custom_path}/mlruns" in result.stdout or f"{custom_path}\\mlruns" in result.stdout

    def test_mlflow_tracking_uri_env_var_takes_precedence(self, experiments_setup):
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
            env=_subprocess_env(MLFLOW_TRACKING_URI=custom_mlflow_uri),
            cwd=str(experiments_setup["project_root"])
        )
        assert result.returncode == 0, f"Config import failed: {result.stderr}"
        assert custom_mlflow_uri in result.stdout


class TestRunAllSetupFunction:

    def test_run_all_contains_setup_experiments_dir(self, experiments_setup):
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
        assert "PRODUCTION_DIR.mkdir" in run_all_content
        assert '"data" / "bronze"' in run_all_content
        assert '"data" / "silver"' in run_all_content
        assert '"data" / "gold"' in run_all_content
        assert 'EXPERIMENTS_DIR / "mlruns"' in run_all_content

    def test_pipeline_runner_contains_setup_experiments_dir(self, experiments_setup):
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

    def test_bronze_source_path_uses_raw_source_path(self, experiments_setup):
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
        assert "customers.parquet" in config_content
        assert 'FINDINGS_DIR / "customers.parquet"' not in config_content

    def test_bronze_can_load_source_data(self, experiments_setup):
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

        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys, os
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
sys.stdout.flush()
os._exit(0)
"""],
            capture_output=True,
            text=True,
            cwd=str(experiments_setup["project_root"])
        )
        assert result.returncode == 0, f"Bronze load test failed: {result.stderr}"
        assert "Exists: True" in result.stdout
        assert "Rows: 5" in result.stdout


class TestDatabricksCompatibility:

    def test_dbfs_path_works_with_env_var(self, experiments_setup):
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

print(f'EXPERIMENTS_DIR: {{config.EXPERIMENTS_DIR}}')
print(f'Bronze path: {{config.get_bronze_path("test")}}')
print(f'Silver path: {{config.get_silver_path()}}')
print(f'Gold path: {{config.get_gold_path()}}')
print(f'Feast path: {{config.get_feast_data_path()}}')
"""],
            capture_output=True,
            text=True,
            env=_subprocess_env(CR_EXPERIMENTS_DIR=dbfs_path),
            cwd=str(experiments_setup["project_root"])
        )
        assert result.returncode == 0, f"DBFS path test failed: {result.stderr}"
        assert dbfs_path in result.stdout
        assert result.stdout.count(dbfs_path) >= 5

    def test_unity_catalog_path_works_with_env_var(self, experiments_setup):
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
            env=_subprocess_env(CR_EXPERIMENTS_DIR=uc_path),
            cwd=str(experiments_setup["project_root"])
        )
        assert result.returncode == 0, f"Unity Catalog path test failed: {result.stderr}"
        assert uc_path in result.stdout
