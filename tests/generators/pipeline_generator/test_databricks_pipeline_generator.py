import ast
from pathlib import Path

import pytest
import yaml

from customer_retention.generators.pipeline_generator.databricks_generator import (
    DatabricksPipelineGenerator,
)


@pytest.fixture
def sample_findings_dir(tmp_path):
    findings_dir = tmp_path / "findings"
    findings_dir.mkdir()

    multi_dataset = {
        "datasets": {
            "customers": {
                "name": "customers",
                "findings_path": str(findings_dir / "customers_findings.yaml"),
                "source_path": "/data/customers.csv",
                "granularity": "entity_level",
                "row_count": 1000,
                "column_count": 5,
                "excluded": False,
            },
            "orders": {
                "name": "orders",
                "findings_path": str(findings_dir / "orders_findings.yaml"),
                "source_path": "/data/orders.parquet",
                "granularity": "event_level",
                "row_count": 5000,
                "column_count": 4,
                "excluded": False,
                "entity_column": "customer_id",
                "time_column": "order_date",
            },
        },
        "relationships": [
            {
                "left_dataset": "customers",
                "right_dataset": "orders",
                "left_column": "customer_id",
                "right_column": "customer_id",
                "relationship_type": "one_to_many",
                "confidence": 1.0,
            },
        ],
        "primary_entity_dataset": "customers",
        "event_datasets": ["orders"],
        "excluded_datasets": [],
    }
    (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

    customers_findings = {
        "source_path": "/data/customers.csv",
        "source_format": "csv",
        "row_count": 1000,
        "column_count": 5,
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
            "age": {
                "name": "age",
                "inferred_type": "numeric_continuous",
                "confidence": 0.9,
                "evidence": [],
                "quality_score": 85,
                "cleaning_needed": True,
                "cleaning_recommendations": ["impute_null:median"],
                "type_metrics": {"has_nulls": True},
            },
            "churn": {
                "name": "churn",
                "inferred_type": "binary",
                "confidence": 0.99,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": [],
            },
        },
        "target_column": "churn",
        "identifier_columns": ["customer_id"],
    }
    (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers_findings))

    orders_findings = {
        "source_path": "/data/orders.parquet",
        "source_format": "parquet",
        "row_count": 5000,
        "column_count": 4,
        "columns": {
            "order_id": {
                "name": "order_id",
                "inferred_type": "identifier",
                "confidence": 0.95,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": [],
            },
            "customer_id": {
                "name": "customer_id",
                "inferred_type": "identifier",
                "confidence": 0.95,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": [],
            },
            "amount": {
                "name": "amount",
                "inferred_type": "numeric_continuous",
                "confidence": 0.9,
                "evidence": [],
                "quality_score": 90,
                "cleaning_needed": True,
                "cleaning_recommendations": ["cap_outlier:iqr"],
                "type_metrics": {"has_outliers": True},
            },
            "order_date": {
                "name": "order_date",
                "inferred_type": "datetime",
                "confidence": 0.95,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": [],
            },
        },
        "identifier_columns": ["order_id"],
        "datetime_columns": ["order_date"],
        "time_series_metadata": {
            "granularity": "event_level",
            "entity_column": "customer_id",
            "time_column": "order_date",
            "aggregation_windows_used": ["7d", "30d", "90d"],
        },
    }
    (findings_dir / "orders_findings.yaml").write_text(yaml.dump(orders_findings))

    return findings_dir


class TestDatabricksPipelineGeneratorInit:
    def test_generator_takes_required_params(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
        )
        assert generator._pipeline_name == "test_pipeline"

    def test_generator_stores_catalog_and_schema(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
            catalog="ml_catalog", schema="retention",
        )
        assert generator._catalog == "ml_catalog"
        assert generator._schema == "retention"

    def test_generator_sets_output_dir(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "my_pipeline",
        )
        assert generator._output_dir == Path(tmp_path)

    def test_generator_defaults_catalog_and_schema(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
        )
        assert generator._catalog == "main"
        assert generator._schema == "default"

    def test_explicit_framework_repo_path_passed_to_renderer(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
            framework_repo_path="/Workspace/Repos/me/churnkit",
        )
        assert generator._renderer._framework_repo_path == "/Workspace/Repos/me/churnkit"

    def test_framework_repo_path_from_persisted_config(self, sample_findings_dir, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_FRAMEWORK_REPO_PATH", raising=False)
        monkeypatch.setattr(
            "customer_retention.generators.pipeline_generator.databricks_generator.get_framework_repo_path",
            lambda: "/Workspace/Repos/team/framework",
        )
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
        )
        assert generator._renderer._framework_repo_path == "/Workspace/Repos/team/framework"

    def test_explicit_framework_repo_path_overrides_persisted(self, sample_findings_dir, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "customer_retention.generators.pipeline_generator.databricks_generator.get_framework_repo_path",
            lambda: "/Workspace/Repos/team/old",
        )
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
            framework_repo_path="/Workspace/Repos/me/new",
        )
        assert generator._renderer._framework_repo_path == "/Workspace/Repos/me/new"

    def test_no_framework_repo_path_when_unset(self, sample_findings_dir, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_FRAMEWORK_REPO_PATH", raising=False)
        monkeypatch.setattr(
            "customer_retention.generators.pipeline_generator.databricks_generator.get_framework_repo_path",
            lambda: None,
        )
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
        )
        assert generator._renderer._framework_repo_path is None


class TestDatabricksPipelineGeneratorGenerate:
    def test_generate_creates_all_required_files(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
        )
        generated_files = generator.generate()
        assert len(generated_files) > 0
        file_names = [f.name for f in generated_files]
        assert "config.py" in file_names
        assert "pipeline_runner.py" in file_names

    def test_generate_creates_correct_directory_structure(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
        )
        generator.generate()
        assert (tmp_path / "bronze").exists()
        assert (tmp_path / "silver").exists()
        assert (tmp_path / "gold").exists()
        assert (tmp_path / "training").exists()

    def test_generate_returns_list_of_paths(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
        )
        files = generator.generate()
        assert isinstance(files, list)
        for f in files:
            assert isinstance(f, Path)

    def test_generated_paths_exist(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
        )
        files = generator.generate()
        for f in files:
            assert f.exists()

    def test_output_files_are_valid_python(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "valid_python",
        )
        files = generator.generate()
        for file_path in files:
            if file_path.suffix == ".py":
                content = file_path.read_text()
                ast.parse(content)


class TestDatabricksMultiDatasetGeneration:
    def test_generates_per_source_bronze_files(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "multi_source",
        )
        generator.generate()
        bronze_dir = tmp_path / "bronze"
        bronze_files = sorted(f.name for f in bronze_dir.glob("*.py"))
        assert len(bronze_files) == 3
        assert "bronze_entity_customers.py" in bronze_files
        assert "bronze_event_orders.py" in bronze_files
        assert "bronze_entity_orders_aggregated.py" in bronze_files

    def test_entity_source_gets_single_bronze(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
        )
        generator.generate()
        bronze_dir = tmp_path / "bronze"
        entity_files = [f for f in bronze_dir.glob("bronze_entity_customers*")]
        assert len(entity_files) == 1

    def test_event_source_gets_event_and_entity(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
        )
        generator.generate()
        bronze_dir = tmp_path / "bronze"
        event_file = bronze_dir / "bronze_event_orders.py"
        entity_agg_file = bronze_dir / "bronze_entity_orders_aggregated.py"
        assert event_file.exists()
        assert entity_agg_file.exists()

    def test_silver_uses_composite_name(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
        )
        generator.generate()
        silver_dir = tmp_path / "silver"
        silver_files = list(silver_dir.glob("*.py"))
        assert len(silver_files) == 1
        assert "silver_featureset_" in silver_files[0].name or "silver" in silver_files[0].name

    def test_gold_uses_composite_name(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
        )
        generator.generate()
        gold_dir = tmp_path / "gold"
        gold_files = list(gold_dir.glob("*.py"))
        assert len(gold_files) == 1
        assert "gold_features_" in gold_files[0].name or "gold" in gold_files[0].name

    def test_recommendations_applied_to_bronze(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "test_pipeline",
        )
        generator.generate()
        bronze_entity = tmp_path / "bronze" / "bronze_entity_customers.py"
        content = bronze_entity.read_text()
        assert "age" in content or "transformations" in content.lower()


class TestDatabricksE2E:
    def test_runner_references_all_notebooks(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "e2e_test",
        )
        generator.generate()
        runner_path = tmp_path / "pipeline_runner.py"
        content = runner_path.read_text()
        assert "bronze_entity_customers" in content
        assert "bronze_event_orders" in content
        assert "silver" in content
        assert "gold" in content
        assert "training" in content

    def test_no_parquet_format_in_generated_output(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "no_parquet",
        )
        files = generator.generate()
        for file_path in files:
            if file_path.suffix == ".py":
                content = file_path.read_text()
                assert 'format("parquet")' not in content, (
                    f"Found parquet format in {file_path.name}"
                )

    def test_delta_format_everywhere(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "delta_check",
        )
        files = generator.generate()
        bronze_files = [f for f in files if "bronze" in f.name and f.suffix == ".py"]
        for bf in bronze_files:
            content = bf.read_text()
            assert "delta" in content.lower() or "saveAsTable" in content

    def test_all_generated_files_valid_python(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "syntax_check",
        )
        files = generator.generate()
        for file_path in files:
            if file_path.suffix == ".py":
                content = file_path.read_text()
                ast.parse(content)

    def test_config_has_catalog_and_schema(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "config_test",
            catalog="my_catalog", schema="my_schema",
        )
        generator.generate()
        config_content = (tmp_path / "config.py").read_text()
        assert "my_catalog" in config_content
        assert "my_schema" in config_content

    def test_all_notebooks_have_sys_path_when_persisted(self, sample_findings_dir, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "customer_retention.generators.pipeline_generator.databricks_generator.get_framework_repo_path",
            lambda: "/Workspace/Repos/team/churnkit",
        )
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "repo_test",
        )
        files = generator.generate()
        for file_path in files:
            if file_path.suffix != ".py":
                continue
            content = file_path.read_text()
            assert "import sys" in content, f"{file_path.name} missing sys.path setup"
            assert 'FRAMEWORK_REPO_ROOT = "/Workspace/Repos/team/churnkit"' in content
            assert '{FRAMEWORK_REPO_ROOT}/src' in content, f"{file_path.name} missing /src suffix"

    def test_no_sys_path_when_not_persisted(self, sample_findings_dir, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_FRAMEWORK_REPO_PATH", raising=False)
        monkeypatch.setattr(
            "customer_retention.generators.pipeline_generator.databricks_generator.get_framework_repo_path",
            lambda: None,
        )
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "no_repo_test",
        )
        files = generator.generate()
        for file_path in files:
            if file_path.suffix != ".py":
                continue
            content = file_path.read_text()
            assert "FRAMEWORK_REPO_ROOT" not in content, f"{file_path.name} has unexpected sys.path"

    def test_aggregated_entity_reads_from_event_table(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "table_ref_test",
        )
        generator.generate()
        agg_path = tmp_path / "bronze" / "bronze_entity_orders_aggregated.py"
        content = agg_path.read_text()
        assert 'bronze_table("orders_events")' in content
        assert "orders_aggregated_events" not in content

    def test_all_notebooks_use_parent_config_path(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "config_path_test",
        )
        files = generator.generate()
        for file_path in files:
            if file_path.suffix == ".py" and file_path.name != "config.py" and file_path.name != "pipeline_runner.py":
                content = file_path.read_text()
                assert "%run ../config" in content, f"{file_path.name} should use ../config"
                assert "%run ./config" not in content, f"{file_path.name} should not use ./config"

    def test_generate_creates_landing_when_populated(self, sample_findings_dir, tmp_path):
        generator = DatabricksPipelineGenerator(
            str(sample_findings_dir), str(tmp_path), "landing_test",
        )
        files = generator.generate()
        file_names = [f.name for f in files]
        has_landing = any("landing_" in n for n in file_names)
        if has_landing:
            landing_dir = tmp_path / "landing"
            assert landing_dir.exists()
            landing_files = list(landing_dir.glob("*.py"))
            for lf in landing_files:
                ast.parse(lf.read_text())

    def test_single_source_generates(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        multi_dataset = {
            "datasets": {
                "test": {
                    "name": "test",
                    "findings_path": str(findings_dir / "test_findings.yaml"),
                    "source_path": "/test.csv",
                    "granularity": "entity_level",
                    "row_count": 100,
                    "column_count": 2,
                    "excluded": False,
                },
            },
            "relationships": [],
            "primary_entity_dataset": "test",
            "event_datasets": [],
            "excluded_datasets": [],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        test_findings = {
            "source_path": "/test.csv",
            "source_format": "csv",
            "row_count": 100,
            "column_count": 2,
            "columns": {
                "id": {
                    "name": "id",
                    "inferred_type": "identifier",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "target": {
                    "name": "target",
                    "inferred_type": "binary",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "target_column": "target",
            "identifier_columns": ["id"],
        }
        (findings_dir / "test_findings.yaml").write_text(yaml.dump(test_findings))

        output_dir = tmp_path / "output"
        generator = DatabricksPipelineGenerator(
            str(findings_dir), str(output_dir), "single_source",
        )
        files = generator.generate()
        assert len(files) > 0
        for file_path in files:
            if file_path.suffix == ".py":
                ast.parse(file_path.read_text())
