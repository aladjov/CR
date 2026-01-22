import pytest
import yaml
from pathlib import Path


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
        "time_series_metadata": {"granularity": "event_level", "entity_column": "customer_id", "time_column": "order_date"}
    }
    (findings_dir / "orders_findings.yaml").write_text(yaml.dump(orders_findings))

    return findings_dir


class TestFindingsParserInit:
    def test_parser_takes_findings_dir(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(tmp_path))
        assert parser._findings_dir == Path(tmp_path)


class TestFindingsParserLoadMultiDataset:
    def test_load_multi_dataset_findings(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(sample_findings_dir))
        multi = parser._load_multi_dataset_findings()
        assert "customers" in multi.datasets
        assert "orders" in multi.datasets

    def test_load_multi_dataset_has_relationships(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(sample_findings_dir))
        multi = parser._load_multi_dataset_findings()
        assert len(multi.relationships) == 1
        assert multi.relationships[0].left_column == "customer_id"


class TestFindingsParserLoadSourceFindings:
    def test_load_source_findings(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(sample_findings_dir))
        findings = parser._load_source_findings(["customers", "orders"], sample_findings_dir)
        assert "customers" in findings
        assert "orders" in findings

    def test_load_source_findings_has_columns(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(sample_findings_dir))
        findings = parser._load_source_findings(["customers"], sample_findings_dir)
        assert "age" in findings["customers"].columns


class TestFindingsParserBuildPipelineConfig:
    def test_parse_returns_pipeline_config(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineConfig
        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()
        assert isinstance(config, PipelineConfig)

    def test_parse_extracts_sources(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()
        source_names = [s.name for s in config.sources]
        assert "customers" in source_names
        assert "orders" in source_names

    def test_parse_extracts_target_column(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()
        assert config.target_column == "churn"

    def test_parse_extracts_bronze_transformations(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()
        assert "customers" in config.bronze
        customers_bronze = config.bronze["customers"]
        assert len(customers_bronze.transformations) >= 1

    def test_parse_extracts_silver_joins(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()
        assert len(config.silver.joins) >= 1
        assert config.silver.joins[0]["left_key"] == "customer_id"

    def test_parse_identifies_event_level_sources(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()
        orders_source = next(s for s in config.sources if s.name == "orders")
        assert orders_source.is_event_level is True
        assert orders_source.time_column == "order_date"


class TestFindingsParserErrorHandling:
    def test_parse_raises_on_missing_multi_dataset(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            parser.parse()

    def test_handles_missing_optional_fields(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
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
            "source_path": "/test.csv",
            "source_format": "csv",
            "row_count": 100,
            "column_count": 2,
            "columns": {
                "id": {"name": "id", "inferred_type": "identifier", "confidence": 0.95, "evidence": [],
                      "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []},
                "target": {"name": "target", "inferred_type": "binary", "confidence": 0.9, "evidence": [],
                          "quality_score": 100, "cleaning_needed": False, "cleaning_recommendations": []}
            },
            "target_column": "target",
            "identifier_columns": ["id"]
        }
        (findings_dir / "test_findings.yaml").write_text(yaml.dump(test_findings))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        assert config is not None
