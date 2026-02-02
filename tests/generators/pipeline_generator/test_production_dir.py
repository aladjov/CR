import ast
import os
import subprocess
import sys

import pytest
import yaml


def _generate(experiments_setup, production_dir=None):
    from customer_retention.generators.pipeline_generator import PipelineGenerator

    generator = PipelineGenerator(
        findings_dir=str(experiments_setup["findings_dir"]),
        output_dir=str(experiments_setup["output_dir"]),
        pipeline_name="test_pipeline",
        experiments_dir="experiments",
        production_dir=production_dir,
    )
    generator.generate()
    return experiments_setup["output_dir"]


class TestProductionDirModelField:

    def test_default_is_none(self):
        from customer_retention.generators.pipeline_generator.models import PipelineConfig

        config = PipelineConfig(
            name="t", target_column="target", sources=[], bronze={},
            silver=None, gold=None, output_dir=".",
        )
        assert config.production_dir is None

    def test_accepts_custom_value(self):
        from customer_retention.generators.pipeline_generator.models import PipelineConfig

        config = PipelineConfig(
            name="t", target_column="target", sources=[], bronze={},
            silver=None, gold=None, output_dir=".",
            production_dir="/custom/prod",
        )
        assert config.production_dir == "/custom/prod"


class TestConfigTemplateProductionDir:

    def test_config_contains_production_dir(self, experiments_setup):
        out = _generate(experiments_setup)
        content = (out / "config.py").read_text()
        assert "PRODUCTION_DIR" in content

    def test_default_production_dir(self, experiments_setup):
        out = _generate(experiments_setup)
        content = (out / "config.py").read_text()
        assert 'EXPERIMENTS_DIR / "production"' in content

    def test_custom_production_dir(self, experiments_setup):
        out = _generate(experiments_setup, production_dir="/my/prod")
        content = (out / "config.py").read_text()
        assert '"/my/prod"' in content

    def test_env_var_override(self, experiments_setup):
        out = _generate(experiments_setup)
        content = (out / "config.py").read_text()
        assert "CR_PRODUCTION_DIR" in content

    def test_get_bronze_path_uses_production_dir(self, experiments_setup):
        out = _generate(experiments_setup)
        content = (out / "config.py").read_text()
        assert 'PRODUCTION_DIR / "data" / "bronze"' in content

    def test_get_silver_path_uses_production_dir(self, experiments_setup):
        out = _generate(experiments_setup)
        content = (out / "config.py").read_text()
        assert 'PRODUCTION_DIR / "data" / "silver"' in content

    def test_get_gold_path_uses_production_dir(self, experiments_setup):
        out = _generate(experiments_setup)
        content = (out / "config.py").read_text()
        assert 'PRODUCTION_DIR / "data" / "gold"' in content

    def test_exploration_artifacts_uses_experiments_dir(self, experiments_setup):
        out = _generate(experiments_setup)
        content = (out / "config.py").read_text()
        for line in content.splitlines():
            if "EXPLORATION_ARTIFACTS" in line and "PRODUCTION_DIR" in line:
                pytest.fail("EXPLORATION_ARTIFACTS must use EXPERIMENTS_DIR, not PRODUCTION_DIR")

    def test_artifacts_path_defaults_under_production_dir(self, experiments_setup):
        out = _generate(experiments_setup)
        content = (out / "config.py").read_text()
        for line in content.splitlines():
            if line.startswith("ARTIFACTS_PATH") and "PRODUCTION_DIR" in line:
                break
        else:
            pytest.fail("ARTIFACTS_PATH default should use PRODUCTION_DIR")

    def test_feast_repo_path_under_production_dir(self, experiments_setup):
        out = _generate(experiments_setup)
        content = (out / "config.py").read_text()
        assert 'PRODUCTION_DIR / "feature_repo"' in content

    def test_mlflow_stays_on_experiments_dir(self, experiments_setup):
        out = _generate(experiments_setup)
        content = (out / "config.py").read_text()
        assert 'EXPERIMENTS_DIR / \'mlruns.db\'' in content or 'EXPERIMENTS_DIR / "mlruns.db"' in content

    def test_generated_config_is_valid_python(self, experiments_setup):
        out = _generate(experiments_setup)
        content = (out / "config.py").read_text()
        ast.parse(content)

    def test_production_dir_runtime_value(self, experiments_setup):
        out = _generate(experiments_setup)
        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{out}')
import config
print(f'PRODUCTION_DIR: {{config.PRODUCTION_DIR}}')
print(f'EXPERIMENTS_DIR: {{config.EXPERIMENTS_DIR}}')
print(f'Bronze: {{config.get_bronze_path("test")}}')
print(f'Silver: {{config.get_silver_path()}}')
print(f'Gold: {{config.get_gold_path()}}')
"""],
            capture_output=True, text=True,
            cwd=str(experiments_setup["project_root"]),
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"
        assert "PRODUCTION_DIR:" in result.stdout
        assert "production" in result.stdout

    def test_env_var_overrides_production_dir_at_runtime(self, experiments_setup):
        out = _generate(experiments_setup)
        custom = "/custom/prod/override"
        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{out}')
import config
print(f'PRODUCTION_DIR: {{config.PRODUCTION_DIR}}')
"""],
            capture_output=True, text=True,
            env={**os.environ, "CR_PRODUCTION_DIR": custom},
            cwd=str(experiments_setup["project_root"]),
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"
        assert custom in result.stdout


class TestLandingTemplateProductionDir:

    def test_landing_output_uses_production_dir(self, experiments_setup):
        findings_dir = experiments_setup["findings_dir"]
        multi_path = findings_dir / "multi_dataset_findings.yaml"
        multi = yaml.safe_load(multi_path.read_text())
        multi["event_datasets"] = ["transactions"]
        multi["datasets"]["transactions"] = {
            "name": "transactions",
            "findings_path": "transactions_findings.yaml",
            "source_path": str(experiments_setup["data_path"]),
            "granularity": "event_level",
            "row_count": 5,
            "column_count": 4,
            "excluded": False,
        }
        multi_path.write_text(yaml.dump(multi))

        txn_findings = {
            "source_path": "customers.parquet",
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
                    "cleaning_recommendations": [],
                },
                "revenue": {
                    "name": "revenue",
                    "inferred_type": "numeric_continuous",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "orders": {
                    "name": "orders",
                    "inferred_type": "numeric_discrete",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "target": {
                    "name": "target",
                    "inferred_type": "binary",
                    "confidence": 0.99,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "target_column": "target",
            "identifier_columns": ["customer_id"],
        }
        (findings_dir / "transactions_findings.yaml").write_text(yaml.dump(txn_findings))

        out = _generate(experiments_setup)
        landing_files = list((out / "landing").glob("*.py"))
        assert len(landing_files) > 0
        content = landing_files[0].read_text()
        assert "PRODUCTION_DIR" in content


class TestValidationTemplateProductionDir:

    def test_validation_imports_production_dir(self, experiments_setup):
        out = _generate(experiments_setup)
        content = (out / "validation" / "validate_pipeline.py").read_text()
        assert "PRODUCTION_DIR" in content

    def test_prod_paths_use_production_dir(self, experiments_setup):
        out = _generate(experiments_setup)
        content = (out / "validation" / "validate_pipeline.py").read_text()
        assert 'PRODUCTION_DIR / "data" / "bronze"' in content
        assert 'PRODUCTION_DIR / "data" / "silver"' in content
        assert 'PRODUCTION_DIR / "data" / "gold"' in content

    def test_exploration_paths_use_experiments_dir(self, experiments_setup):
        out = _generate(experiments_setup)
        content = (out / "validation" / "validate_pipeline.py").read_text()
        assert "EXPLORATION_ARTIFACTS" in content


class TestRunnerSetupProductionDir:

    def test_runner_creates_production_dir_subdirs(self, experiments_setup):
        out = _generate(experiments_setup)
        runner_content = (out / "pipeline_runner.py").read_text()
        assert "PRODUCTION_DIR" in runner_content
        assert 'PRODUCTION_DIR / "data"' in runner_content or "PRODUCTION_DIR" in runner_content

    def test_run_all_creates_production_dir_subdirs(self, experiments_setup):
        out = _generate(experiments_setup)
        run_all_content = (out / "run_all.py").read_text()
        assert "PRODUCTION_DIR" in run_all_content

    def test_run_all_setup_creates_mlruns_under_experiments(self, experiments_setup):
        out = _generate(experiments_setup)
        run_all_content = (out / "run_all.py").read_text()
        assert 'EXPERIMENTS_DIR / "mlruns"' in run_all_content
