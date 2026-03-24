import json
from pathlib import Path

import pytest
import yaml

from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
from customer_retention.generators.pipeline_generator.generator import PipelineGenerator
from customer_retention.generators.pipeline_generator.models import PipelineTransformationType


@pytest.fixture
def sample_findings_dir(tmp_path):
    findings_dir = tmp_path / "findings"
    findings_dir.mkdir()

    multi_dataset = {
        "datasets": {
            "customers": {"name": "customers", "findings_path": str(findings_dir / "customers_findings.yaml"),
                         "source_path": "/data/customers.csv", "granularity": "entity_level",
                         "row_count": 1000, "column_count": 5, "excluded": False},
            "orders": {"name": "orders", "findings_path": str(findings_dir / "orders_findings.yaml"),
                      "source_path": "/data/orders.parquet", "granularity": "event_level",
                      "row_count": 5000, "column_count": 4, "excluded": False,
                      "entity_column": "customer_id", "time_column": "order_date"}
        },
        "relationships": [
            {"left_dataset": "customers", "right_dataset": "orders", "left_column": "customer_id",
             "right_column": "customer_id", "relationship_type": "one_to_many", "confidence": 1.0}
        ],
        "primary_entity_dataset": "customers",
        "event_datasets": ["orders"],
        "excluded_datasets": []
    }
    (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

    customers_findings = {
        "source_path": "/data/customers.csv",
        "source_format": "csv",
        "row_count": 1000,
        "column_count": 5,
        "columns": {
            "customer_id": {"name": "customer_id", "inferred_type": "identifier", "confidence": 0.95,
                          "evidence": [], "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
            "age": {"name": "age", "inferred_type": "numeric_continuous", "confidence": 0.9, "evidence": [],
                   "quality_score": 85, "cleaning_needed": True,
                   "cleaning_recommendations": ["impute_null:median"], "type_metrics": {"has_nulls": True}},
            "churn": {"name": "churn", "inferred_type": "binary", "confidence": 0.99, "evidence": [],
                     "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []}
        },
        "target_column": "churn",
        "identifier_columns": ["customer_id"]
    }
    (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers_findings))

    orders_findings = {
        "source_path": "/data/orders.parquet",
        "source_format": "parquet",
        "row_count": 5000,
        "column_count": 4,
        "columns": {
            "order_id": {"name": "order_id", "inferred_type": "identifier", "confidence": 0.95, "evidence": [],
                        "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
            "customer_id": {"name": "customer_id", "inferred_type": "identifier", "confidence": 0.95, "evidence": [],
                          "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
            "amount": {"name": "amount", "inferred_type": "numeric_continuous", "confidence": 0.9, "evidence": [],
                      "quality_score": 90, "cleaning_needed": True,
                      "cleaning_recommendations": ["cap_outlier:iqr"], "type_metrics": {"has_outliers": True}},
            "order_date": {"name": "order_date", "inferred_type": "datetime", "confidence": 0.95, "evidence": [],
                          "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []}
        },
        "identifier_columns": ["order_id"],
        "datetime_columns": ["order_date"],
        "time_series_metadata": {"granularity": "event_level", "entity_column": "customer_id", "time_column": "order_date", "aggregation_windows_used": ["7d", "30d", "90d"]}
    }
    (findings_dir / "orders_findings.yaml").write_text(yaml.dump(orders_findings))

    return findings_dir


class TestPipelineGeneratorInit:
    def test_generator_takes_required_params(self, sample_findings_dir, tmp_path):
        generator = PipelineGenerator(str(sample_findings_dir), str(tmp_path), "test_pipeline")
        assert generator._pipeline_name == "test_pipeline"

    def test_generator_sets_output_dir(self, sample_findings_dir, tmp_path):
        generator = PipelineGenerator(str(sample_findings_dir), str(tmp_path), "my_pipeline")
        assert generator._output_dir == Path(tmp_path)


class TestPipelineGeneratorGenerate:
    def test_generate_creates_all_required_files(self, sample_findings_dir, tmp_path):
        generator = PipelineGenerator(str(sample_findings_dir), str(tmp_path), "test_pipeline")
        generated_files = generator.generate()
        assert len(generated_files) > 0
        file_names = [f.name for f in generated_files]
        assert "config.py" in file_names
        assert "pipeline_runner.py" in file_names
        assert "workflow.json" in file_names

    def test_generate_creates_correct_directory_structure(self, sample_findings_dir, tmp_path):
        generator = PipelineGenerator(str(sample_findings_dir), str(tmp_path), "test_pipeline")
        generator.generate()
        assert tmp_path.exists()
        assert (tmp_path / "bronze").exists()
        assert (tmp_path / "silver").exists()
        assert (tmp_path / "gold").exists()
        assert (tmp_path / "training").exists()

    def test_generate_with_single_source(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        multi_dataset = {
            "datasets": {
                "test": {"name": "test", "findings_path": str(findings_dir / "test_findings.yaml"),
                        "source_path": "/test.csv", "granularity": "entity_level",
                        "row_count": 100, "column_count": 2, "excluded": False}
            },
            "relationships": [],
            "primary_entity_dataset": "test",
            "event_datasets": [],
            "excluded_datasets": []
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        test_findings = {
            "source_path": "/test.csv", "source_format": "csv", "row_count": 100, "column_count": 2,
            "columns": {
                "id": {"name": "id", "inferred_type": "identifier", "confidence": 0.95, "evidence": [],
                      "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
                "target": {"name": "target", "inferred_type": "binary", "confidence": 0.9, "evidence": [],
                          "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []}
            },
            "target_column": "target", "identifier_columns": ["id"]
        }
        (findings_dir / "test_findings.yaml").write_text(yaml.dump(test_findings))

        output_dir = tmp_path / "output"
        generator = PipelineGenerator(str(findings_dir), str(output_dir), "single_source")
        files = generator.generate()
        assert len(files) > 0

    def test_generate_with_multiple_sources(self, sample_findings_dir, tmp_path):
        generator = PipelineGenerator(str(sample_findings_dir), str(tmp_path), "multi_source")
        generator.generate()
        bronze_dir = tmp_path / "bronze"
        bronze_files = sorted(f.name for f in bronze_dir.glob("*.py"))
        assert len(bronze_files) == 3
        assert "bronze_entity_customers.py" in bronze_files
        assert "bronze_event_orders.py" in bronze_files
        assert "bronze_entity_orders_aggregated.py" in bronze_files

    def test_output_files_are_valid_python(self, sample_findings_dir, tmp_path):
        import ast

        generator = PipelineGenerator(str(sample_findings_dir), str(tmp_path), "valid_python")
        files = generator.generate()
        for file_path in files:
            if file_path.suffix == ".py":
                content = file_path.read_text()
                ast.parse(content)

    def test_workflow_json_is_valid(self, sample_findings_dir, tmp_path):
        generator = PipelineGenerator(str(sample_findings_dir), str(tmp_path), "workflow_test")
        generator.generate()
        workflow_path = tmp_path / "workflow.json"
        content = workflow_path.read_text()
        parsed = json.loads(content)
        assert "tasks" in parsed


class TestPipelineGeneratorReturnsFilePaths:
    def test_generate_returns_list_of_paths(self, sample_findings_dir, tmp_path):
        generator = PipelineGenerator(str(sample_findings_dir), str(tmp_path), "paths_test")
        files = generator.generate()
        assert isinstance(files, list)
        for f in files:
            assert isinstance(f, Path)

    def test_generated_paths_exist(self, sample_findings_dir, tmp_path):
        generator = PipelineGenerator(str(sample_findings_dir), str(tmp_path), "exists_test")
        files = generator.generate()
        for f in files:
            assert f.exists()


class TestBronzeDedup:

    def test_duplicate_bronze_steps_deduplicated(self, tmp_path):
        """Build config with duplicate (SEGMENT_AWARE_CAP, col) pairs — verify dedup."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        multi_dataset = {
            "datasets": {
                "customers": {
                    "name": "customers",
                    "findings_path": str(findings_dir / "customers_findings.yaml"),
                    "source_path": "/data/customers.csv",
                    "granularity": "entity_level",
                    "row_count": 100, "column_count": 3, "excluded": False,
                }
            },
            "relationships": [],
            "primary_entity_dataset": "customers",
            "event_datasets": [],
            "excluded_datasets": [],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        customers_findings = {
            "source_path": "/data/customers.csv",
            "source_format": "csv",
            "row_count": 100,
            "column_count": 3,
            "columns": {
                "id": {"name": "id", "inferred_type": "identifier", "confidence": 0.95,
                       "evidence": [], "quality_score": 100,
                       "cleaning_needed": False, "cleaning_recommendations": []},
                "revenue": {"name": "revenue", "inferred_type": "numeric_continuous", "confidence": 0.9,
                           "evidence": [], "quality_score": 80,
                           "cleaning_needed": True, "cleaning_recommendations": ["cap_outlier:iqr"]},
                "target": {"name": "target", "inferred_type": "binary", "confidence": 0.99,
                          "evidence": [], "quality_score": 100,
                          "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "target_column": "target",
            "identifier_columns": ["id"],
        }
        (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers_findings))

        # Recommendations with duplicate segment_aware_cap entries for same column
        recommendations = {
            "sources": {
                "customers": {
                    "source_file": "/data/customers.csv",
                    "null_handling": [],
                    "outlier_handling": [
                        {"id": "bronze_outlier_revenue_1", "layer": "bronze",
                         "category": "outlier", "source_notebook": "03_quality",
                         "target_column": "revenue", "action": "segment_aware_cap",
                         "parameters": {"method": "segment_iqr", "n_segments": 2},
                         "rationale": "Cap revenue outliers"},
                        {"id": "bronze_outlier_revenue_2", "layer": "bronze",
                         "category": "outlier", "source_notebook": "03_quality",
                         "target_column": "revenue", "action": "segment_aware_cap",
                         "parameters": {"method": "segment_iqr", "n_segments": 2},
                         "rationale": "Cap revenue outliers (duplicate)"},
                    ],
                }
            },
        }
        (findings_dir / "recommendations.yaml").write_text(yaml.dump(recommendations))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        bronze = config.bronze["customers"]
        segment_caps = [
            t for t in bronze.transformations
            if t.type == PipelineTransformationType.SEGMENT_AWARE_CAP and t.column == "revenue"
        ]
        assert len(segment_caps) == 1, (
            f"Expected 1 SEGMENT_AWARE_CAP for revenue, got {len(segment_caps)}"
        )
