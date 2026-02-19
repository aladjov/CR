import ast

import pytest
import yaml

from customer_retention.generators.pipeline_generator.exploration_generator import (
    DatabricksExplorationGenerator,
)


@pytest.fixture
def project_context_path(tmp_path):
    ctx = {
        "project_name": "retention_v2",
        "target_dataset": "profiles",
        "target_column": "churned",
        "entity_column": "customer_id",
        "primary_objective": "immediate_risk",
        "objectives": [{"objective": "immediate_risk", "priority": "primary", "anchor": "now"}],
        "datasets": {
            "profiles": {
                "name": "profiles",
                "path": "/data/profiles.csv",
                "role": "target",
                "entity_column": "customer_id",
                "target_candidates": ["churned"],
            },
            "transactions": {
                "name": "transactions",
                "path": "/data/transactions.parquet",
                "role": "feature",
                "join_keys": ["customer_id"],
                "join_to": "profiles",
                "relationship": "many_to_one",
            },
        },
    }
    path = tmp_path / "project_context.yaml"
    path.write_text(yaml.dump(ctx))
    return path


class TestDatabricksExplorationGenerator:
    def test_creates_output_file(self, project_context_path, tmp_path):
        output_dir = tmp_path / "output"
        gen = DatabricksExplorationGenerator(
            str(project_context_path), str(output_dir),
        )
        result = gen.generate()
        assert result.exists()
        assert result.name == "exploration_runner.py"

    def test_output_is_valid_python(self, project_context_path, tmp_path):
        output_dir = tmp_path / "output"
        gen = DatabricksExplorationGenerator(
            str(project_context_path), str(output_dir),
        )
        result = gen.generate()
        ast.parse(result.read_text())

    def test_includes_datasets(self, project_context_path, tmp_path):
        output_dir = tmp_path / "output"
        gen = DatabricksExplorationGenerator(
            str(project_context_path), str(output_dir),
        )
        result = gen.generate()
        code = result.read_text()
        assert '"profiles"' in code
        assert '"transactions"' in code
        assert "/data/profiles.csv" in code
        assert "/data/transactions.parquet" in code

    def test_custom_notebooks_base_path(self, project_context_path, tmp_path):
        output_dir = tmp_path / "output"
        gen = DatabricksExplorationGenerator(
            str(project_context_path), str(output_dir),
            notebooks_base_path="/Workspace/exploration",
        )
        result = gen.generate()
        assert "/Workspace/exploration" in result.read_text()

    def test_custom_findings_base_path(self, project_context_path, tmp_path):
        output_dir = tmp_path / "output"
        gen = DatabricksExplorationGenerator(
            str(project_context_path), str(output_dir),
            findings_base_path="/dbfs/findings",
        )
        result = gen.generate()
        assert "/dbfs/findings" in result.read_text()

    def test_project_name_in_output(self, project_context_path, tmp_path):
        output_dir = tmp_path / "output"
        gen = DatabricksExplorationGenerator(
            str(project_context_path), str(output_dir),
        )
        result = gen.generate()
        assert "retention_v2" in result.read_text()

    def test_target_dataset_identified(self, project_context_path, tmp_path):
        output_dir = tmp_path / "output"
        gen = DatabricksExplorationGenerator(
            str(project_context_path), str(output_dir),
        )
        result = gen.generate()
        assert 'TARGET_DATASET = "profiles"' in result.read_text()

    def test_creates_parent_directories(self, project_context_path, tmp_path):
        output_dir = tmp_path / "deep" / "nested" / "output"
        gen = DatabricksExplorationGenerator(
            str(project_context_path), str(output_dir),
        )
        result = gen.generate()
        assert result.exists()

    def test_single_dataset(self, tmp_path):
        ctx = {
            "project_name": "single",
            "target_dataset": "customers",
            "target_column": "churn",
            "entity_column": "id",
            "primary_objective": "immediate_risk",
            "objectives": [{"objective": "immediate_risk", "priority": "primary", "anchor": "now"}],
            "datasets": {
                "customers": {
                    "name": "customers",
                    "path": "/data/customers.csv",
                    "role": "target",
                    "entity_column": "id",
                    "target_candidates": ["churn"],
                },
            },
        }
        ctx_path = tmp_path / "ctx.yaml"
        ctx_path.write_text(yaml.dump(ctx))
        output_dir = tmp_path / "output"
        gen = DatabricksExplorationGenerator(str(ctx_path), str(output_dir))
        result = gen.generate()
        code = result.read_text()
        ast.parse(code)
        assert '"customers"' in code
