from pathlib import Path

import pytest
import yaml


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


@pytest.fixture
def aggregated_event_setup(tmp_path):
    """Setup where multi_dataset_findings only lists the aggregated dataset,
    but a pre-aggregation findings file exists with raw CSV metadata."""
    findings_dir = tmp_path / "findings"
    findings_dir.mkdir()

    agg_parquet = str(findings_dir / "orders_agg.parquet")
    raw_csv = "/data/raw/orders.csv"

    # Multi-dataset findings: only the aggregated entity-level dataset
    multi_dataset = {
        "datasets": {
            "customers": {
                "name": "customers",
                "findings_path": str(findings_dir / "customers_findings.yaml"),
                "source_path": "/data/customers.csv",
                "granularity": "entity_level",
                "row_count": 1000,
                "column_count": 3,
                "excluded": False,
            },
            "orders_agg": {
                "name": "orders_agg",
                "findings_path": str(findings_dir / "orders_agg_findings.yaml"),
                "source_path": agg_parquet,
                "granularity": "entity_level",
                "row_count": 500,
                "column_count": 6,
                "excluded": False,
            },
        },
        "relationships": [
            {
                "left_dataset": "customers",
                "right_dataset": "orders_agg",
                "left_column": "customer_id",
                "right_column": "customer_id",
                "relationship_type": "one_to_one",
                "confidence": 1.0,
            }
        ],
        "primary_entity_dataset": "customers",
        "event_datasets": [],
        "excluded_datasets": [],
        "notes": {
            "temporal_config": {
                "feature_groups": ["lifecycle", "recency"],
            }
        },
    }
    (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

    # Customers findings (entity-level, simple)
    customers_findings = {
        "source_path": "/data/customers.csv",
        "source_format": "csv",
        "row_count": 1000,
        "column_count": 3,
        "columns": {
            "customer_id": {
                "name": "customer_id", "inferred_type": "identifier",
                "confidence": 0.95, "evidence": [], "quality_score": 100,
                "cleaning_needed": False, "cleaning_recommendations": [],
            },
            "churn": {
                "name": "churn", "inferred_type": "binary",
                "confidence": 0.99, "evidence": [], "quality_score": 100,
                "cleaning_needed": False, "cleaning_recommendations": [],
            },
        },
        "target_column": "churn",
        "identifier_columns": ["customer_id"],
    }
    (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers_findings))

    # Aggregated findings (entity-level, no time_series_metadata)
    agg_findings = {
        "source_path": agg_parquet,
        "source_format": "parquet",
        "row_count": 500,
        "column_count": 6,
        "columns": {
            "customer_id": {
                "name": "customer_id", "inferred_type": "identifier",
                "confidence": 0.95, "evidence": [], "quality_score": 100,
                "cleaning_needed": False, "cleaning_recommendations": [],
            },
            "total_amount": {
                "name": "total_amount", "inferred_type": "numeric_continuous",
                "confidence": 0.9, "evidence": [], "quality_score": 100,
                "cleaning_needed": True, "cleaning_recommendations": ["impute_null:0"],
            },
            "order_count": {
                "name": "order_count", "inferred_type": "numeric_discrete",
                "confidence": 0.9, "evidence": [], "quality_score": 100,
                "cleaning_needed": False, "cleaning_recommendations": [],
            },
        },
        "identifier_columns": ["customer_id"],
    }
    (findings_dir / "orders_agg_findings.yaml").write_text(yaml.dump(agg_findings))

    # Pre-aggregation findings (event-level raw CSV, aggregation_executed=True)
    preagg_findings = {
        "source_path": raw_csv,
        "source_format": "csv",
        "row_count": 5000,
        "column_count": 4,
        "columns": {
            "order_id": {
                "name": "order_id", "inferred_type": "identifier",
                "confidence": 0.95, "evidence": [], "quality_score": 100,
                "cleaning_needed": False, "cleaning_recommendations": [],
            },
            "customer_id": {
                "name": "customer_id", "inferred_type": "identifier",
                "confidence": 0.95, "evidence": [], "quality_score": 100,
                "cleaning_needed": False, "cleaning_recommendations": [],
            },
            "amount": {
                "name": "amount", "inferred_type": "numeric_continuous",
                "confidence": 0.9, "evidence": [], "quality_score": 90,
                "cleaning_needed": True, "cleaning_recommendations": ["cap_outlier:iqr"],
            },
            "order_date": {
                "name": "order_date", "inferred_type": "datetime",
                "confidence": 0.95, "evidence": [], "quality_score": 100,
                "cleaning_needed": False, "cleaning_recommendations": [],
            },
        },
        "identifier_columns": ["order_id"],
        "datetime_columns": ["order_date"],
        "time_series_metadata": {
            "granularity": "event_level",
            "entity_column": "customer_id",
            "time_column": "order_date",
            "aggregation_executed": True,
            "aggregated_findings_path": str(findings_dir / "orders_agg_findings.yaml"),
            "suggested_aggregations": ["7d", "30d", "90d"],
        },
    }
    (findings_dir / "orders_raw_findings.yaml").write_text(yaml.dump(preagg_findings))

    return findings_dir


class TestEventSourceDiscovery:
    def test_parse_builds_landing_for_aggregated_event_source(self, aggregated_event_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(aggregated_event_setup))
        config = parser.parse()

        assert "orders_agg" in config.landing
        landing = config.landing["orders_agg"]
        assert landing.entity_column == "customer_id"
        assert landing.time_column == "order_date"
        assert landing.raw_source_path == "/data/raw/orders.csv"
        assert "orders_agg" in config.bronze_event
        bronze_event = config.bronze_event["orders_agg"]
        assert bronze_event.aggregation is not None
        assert "7d" in bronze_event.aggregation.windows
        assert "30d" in bronze_event.aggregation.windows

    def test_parse_marks_source_as_event_level(self, aggregated_event_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(aggregated_event_setup))
        config = parser.parse()

        source = next(s for s in config.sources if s.name == "orders_agg")
        assert source.is_event_level is True

    def test_parse_no_landing_when_no_preagg_exists(self, tmp_path):
        """Entity-level dataset with no pre-aggregation findings file -> no landing."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        multi_dataset = {
            "datasets": {
                "simple": {
                    "name": "simple",
                    "findings_path": str(findings_dir / "simple_findings.yaml"),
                    "source_path": "/data/simple.parquet",
                    "granularity": "entity_level",
                    "row_count": 100, "column_count": 2, "excluded": False,
                }
            },
            "relationships": [],
            "primary_entity_dataset": "simple",
            "event_datasets": [],
            "excluded_datasets": [],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        simple_findings = {
            "source_path": "/data/simple.parquet",
            "source_format": "parquet",
            "row_count": 100,
            "column_count": 2,
            "columns": {
                "id": {"name": "id", "inferred_type": "identifier", "confidence": 0.95,
                       "evidence": [], "quality_score": 100,
                       "cleaning_needed": False, "cleaning_recommendations": []},
                "target": {"name": "target", "inferred_type": "binary", "confidence": 0.9,
                           "evidence": [], "quality_score": 100,
                           "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "target_column": "target",
            "identifier_columns": ["id"],
        }
        (findings_dir / "simple_findings.yaml").write_text(yaml.dump(simple_findings))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        assert len(config.landing) == 0

    def test_parse_explicit_event_datasets_still_work(self, sample_findings_dir):
        """Regression: explicit event_datasets in multi_dataset_findings still produce landing."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()

        assert "orders" in config.landing
        landing = config.landing["orders"]
        assert landing.entity_column == "customer_id"
        assert landing.time_column == "order_date"

    def test_bronze_event_has_pre_shaping_transformations(self, aggregated_event_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        parser = FindingsParser(str(aggregated_event_setup))
        config = parser.parse()

        assert "orders_agg" in config.bronze_event
        bronze_event = config.bronze_event["orders_agg"]
        assert len(bronze_event.pre_shaping) >= 1
        cap_steps = [t for t in bronze_event.pre_shaping
                     if t.type == PipelineTransformationType.CAP_OUTLIER]
        assert len(cap_steps) == 1
        assert cap_steps[0].column == "amount"

    def test_bronze_has_post_agg_transformations_only(self, aggregated_event_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        parser = FindingsParser(str(aggregated_event_setup))
        config = parser.parse()

        assert "orders_agg" not in config.bronze
        assert "orders_agg" in config.bronze_event
        impute_steps = [t for t in config.bronze_event["orders_agg"].post_shaping
                        if t.type == PipelineTransformationType.IMPUTE_NULL]
        assert len(impute_steps) == 1
        assert impute_steps[0].column == "total_amount"

    def test_parse_lifecycle_attached_for_discovered_event(self, aggregated_event_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(aggregated_event_setup))
        config = parser.parse()

        assert "orders_agg" in config.bronze_event
        bronze_event = config.bronze_event["orders_agg"]
        assert bronze_event.lifecycle is not None
        assert bronze_event.lifecycle.include_lifecycle_quadrant is True
        assert bronze_event.lifecycle.include_recency_bucket is True
        assert bronze_event.entity_column == "customer_id"
        assert bronze_event.time_column == "order_date"


    def test_discovered_event_removed_from_entity_bronze(self, aggregated_event_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(aggregated_event_setup))
        config = parser.parse()

        assert "orders_agg" not in config.bronze
        assert "orders_agg" in config.bronze_event

    def test_entity_transforms_flow_to_post_shaping(self, aggregated_event_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        parser = FindingsParser(str(aggregated_event_setup))
        config = parser.parse()

        bronze_event = config.bronze_event["orders_agg"]
        impute_steps = [t for t in bronze_event.post_shaping
                        if t.type == PipelineTransformationType.IMPUTE_NULL]
        assert len(impute_steps) == 1
        assert impute_steps[0].column == "total_amount"
        cap_steps = [t for t in bronze_event.pre_shaping
                     if t.type == PipelineTransformationType.CAP_OUTLIER]
        assert len(cap_steps) == 1
        assert cap_steps[0].column == "amount"

    def test_binary_columns_included_in_aggregation_value_columns(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        agg_parquet = str(findings_dir / "emails_agg.parquet")

        multi_dataset = {
            "datasets": {
                "emails_agg": {
                    "name": "emails_agg",
                    "findings_path": str(findings_dir / "emails_agg_findings.yaml"),
                    "source_path": agg_parquet,
                    "granularity": "entity_level",
                    "row_count": 500, "column_count": 3, "excluded": False,
                }
            },
            "relationships": [],
            "primary_entity_dataset": "emails_agg",
            "event_datasets": [],
            "excluded_datasets": [],
            "aggregation_windows": ["7d", "30d"],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        agg_findings = {
            "source_path": agg_parquet,
            "source_format": "parquet",
            "row_count": 500, "column_count": 3,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["customer_id"],
            "target_column": "target",
        }
        (findings_dir / "emails_agg_findings.yaml").write_text(yaml.dump(agg_findings))

        preagg_findings = {
            "source_path": "/data/raw/emails.csv",
            "source_format": "csv",
            "row_count": 10000, "column_count": 7,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
                "sent_date": {"name": "sent_date", "inferred_type": "datetime",
                              "confidence": 0.95, "evidence": [], "quality_score": 100,
                              "cleaning_needed": False, "cleaning_recommendations": []},
                "send_hour": {"name": "send_hour", "inferred_type": "numeric_discrete",
                              "confidence": 0.7, "evidence": [], "quality_score": 100,
                              "cleaning_needed": False, "cleaning_recommendations": []},
                "opened": {"name": "opened", "inferred_type": "binary",
                           "confidence": 0.9, "evidence": [], "quality_score": 100,
                           "cleaning_needed": False, "cleaning_recommendations": []},
                "clicked": {"name": "clicked", "inferred_type": "binary",
                            "confidence": 0.9, "evidence": [], "quality_score": 100,
                            "cleaning_needed": False, "cleaning_recommendations": []},
                "bounced": {"name": "bounced", "inferred_type": "binary",
                            "confidence": 0.9, "evidence": [], "quality_score": 100,
                            "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["customer_id"],
            "datetime_columns": ["sent_date"],
            "time_series_metadata": {
                "granularity": "event_level",
                "entity_column": "customer_id",
                "time_column": "feature_timestamp",
                "aggregation_executed": True,
                "aggregated_findings_path": str(findings_dir / "emails_agg_findings.yaml"),
                "suggested_aggregations": ["7d", "30d"],
            },
        }
        (findings_dir / "emails_raw_findings.yaml").write_text(yaml.dump(preagg_findings))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "emails_agg" in config.bronze_event
        agg_config = config.bronze_event["emails_agg"].aggregation
        assert agg_config is not None
        assert "send_hour" in agg_config.value_columns
        assert "opened" in agg_config.value_columns
        assert "clicked" in agg_config.value_columns
        assert "bounced" in agg_config.value_columns


    def test_reconciliation_preserves_non_discovered_bronze(self, aggregated_event_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(aggregated_event_setup))
        config = parser.parse()

        assert "customers" in config.bronze
        assert "orders_agg" not in config.bronze

    def test_reconciliation_removes_discovered_even_without_transforms(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        agg_parquet = str(findings_dir / "clean_agg.parquet")
        multi_dataset = {
            "datasets": {
                "clean_agg": {
                    "name": "clean_agg",
                    "findings_path": str(findings_dir / "clean_agg_findings.yaml"),
                    "source_path": agg_parquet,
                    "granularity": "entity_level",
                    "row_count": 100, "column_count": 2, "excluded": False,
                }
            },
            "relationships": [],
            "primary_entity_dataset": "clean_agg",
            "event_datasets": [],
            "excluded_datasets": [],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        agg_findings = {
            "source_path": agg_parquet,
            "source_format": "parquet",
            "row_count": 100, "column_count": 2,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["customer_id"],
            "target_column": "target",
        }
        (findings_dir / "clean_agg_findings.yaml").write_text(yaml.dump(agg_findings))

        preagg_findings = {
            "source_path": "/data/raw/clean.csv",
            "source_format": "csv",
            "row_count": 5000, "column_count": 3,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
                "event_date": {"name": "event_date", "inferred_type": "datetime",
                               "confidence": 0.95, "evidence": [], "quality_score": 100,
                               "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["customer_id"],
            "datetime_columns": ["event_date"],
            "time_series_metadata": {
                "granularity": "event_level",
                "entity_column": "customer_id",
                "time_column": "event_date",
                "aggregation_executed": True,
                "aggregated_findings_path": str(findings_dir / "clean_agg_findings.yaml"),
            },
        }
        (findings_dir / "clean_raw_findings.yaml").write_text(yaml.dump(preagg_findings))

        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "clean_agg" not in config.bronze
        assert "clean_agg" in config.bronze_event
        assert config.bronze_event["clean_agg"].post_shaping == []

    def test_target_column_excluded_from_binary_aggregation(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        agg_parquet = str(findings_dir / "events_agg.parquet")
        multi_dataset = {
            "datasets": {
                "events_agg": {
                    "name": "events_agg",
                    "findings_path": str(findings_dir / "events_agg_findings.yaml"),
                    "source_path": agg_parquet,
                    "granularity": "entity_level",
                    "row_count": 100, "column_count": 2, "excluded": False,
                }
            },
            "relationships": [],
            "primary_entity_dataset": "events_agg",
            "event_datasets": [],
            "excluded_datasets": [],
            "aggregation_windows": ["7d"],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        agg_findings = {
            "source_path": agg_parquet,
            "source_format": "parquet",
            "row_count": 100, "column_count": 2,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["customer_id"],
            "target_column": "churned",
        }
        (findings_dir / "events_agg_findings.yaml").write_text(yaml.dump(agg_findings))

        preagg_findings = {
            "source_path": "/data/raw/events.csv",
            "source_format": "csv",
            "row_count": 5000, "column_count": 4,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
                "event_date": {"name": "event_date", "inferred_type": "datetime",
                               "confidence": 0.95, "evidence": [], "quality_score": 100,
                               "cleaning_needed": False, "cleaning_recommendations": []},
                "opened": {"name": "opened", "inferred_type": "binary",
                           "confidence": 0.9, "evidence": [], "quality_score": 100,
                           "cleaning_needed": False, "cleaning_recommendations": []},
                "churned": {"name": "churned", "inferred_type": "binary",
                            "confidence": 0.9, "evidence": [], "quality_score": 100,
                            "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["customer_id"],
            "datetime_columns": ["event_date"],
            "target_column": "churned",
            "time_series_metadata": {
                "granularity": "event_level",
                "entity_column": "customer_id",
                "time_column": "event_date",
                "aggregation_executed": True,
                "aggregated_findings_path": str(findings_dir / "events_agg_findings.yaml"),
            },
        }
        (findings_dir / "events_raw_findings.yaml").write_text(yaml.dump(preagg_findings))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        agg_config = config.bronze_event["events_agg"].aggregation
        assert "opened" in agg_config.value_columns
        assert "churned" not in agg_config.value_columns


class TestExplicitEventPreAggTransforms:
    def test_explicit_event_pre_shaping_in_bronze_event(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()

        assert "orders" in config.bronze_event
        bronze_event = config.bronze_event["orders"]
        cap_steps = [t for t in bronze_event.pre_shaping
                     if t.type == PipelineTransformationType.CAP_OUTLIER]
        assert len(cap_steps) == 1
        assert cap_steps[0].column == "amount"

    def test_explicit_event_not_in_entity_bronze(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()

        assert "orders" not in config.bronze
        assert "orders" in config.bronze_event


class TestFindingsParserErrorHandling:
    def test_parse_raises_when_no_findings_at_all(self, tmp_path):
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


class TestProvenanceFlowThrough:

    def test_source_notebook_flows_from_recommendations(self, tmp_path):
        """Steps created from LayeredRecommendations carry source_notebook."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

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
            "row_count": 100, "column_count": 3,
            "columns": {
                "customer_id": {
                    "name": "customer_id", "inferred_type": "identifier",
                    "confidence": 0.95, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "age": {
                    "name": "age", "inferred_type": "numeric_continuous",
                    "confidence": 0.9, "evidence": [], "quality_score": 85,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "target": {
                    "name": "target", "inferred_type": "binary",
                    "confidence": 0.99, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
            },
            "target_column": "target",
            "identifier_columns": ["customer_id"],
        }
        (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers_findings))

        recommendations = {
            "version": "1.0",
            "sources": {
                "customers": {
                    "source_file": "/data/customers.csv",
                    "null_handling": [
                        {
                            "id": "bronze_null_age",
                            "layer": "bronze",
                            "category": "null",
                            "action": "impute",
                            "target_column": "age",
                            "parameters": {"strategy": "median"},
                            "rationale": "Fill missing ages",
                            "source_notebook": "03_quality_assessment",
                            "priority": 1,
                            "dependencies": [],
                        }
                    ],
                    "outlier_handling": [
                        {
                            "id": "bronze_outlier_age",
                            "layer": "bronze",
                            "category": "outlier",
                            "action": "cap",
                            "target_column": "age",
                            "parameters": {"method": "iqr"},
                            "rationale": "Cap age outliers",
                            "source_notebook": "03_quality_assessment",
                            "priority": 1,
                            "dependencies": [],
                        }
                    ],
                    "type_casts": [],
                }
            },
        }
        (findings_dir / "recommendations.yaml").write_text(yaml.dump(recommendations))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        bronze = config.bronze["customers"]
        rec_steps = [s for s in bronze.transformations if s.source_notebook is not None]
        assert len(rec_steps) >= 1
        for step in rec_steps:
            assert step.source_notebook == "03_quality_assessment"

    def test_auto_explorer_steps_have_no_source_notebook(self, sample_findings_dir):
        """Steps from _extract_transformations (bare string recs) have source_notebook=None."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()

        # customers has cleaning_recommendations like "impute_null:median"
        # these go through _parse_cleaning_recommendation which doesn't set source_notebook
        customers_bronze = config.bronze["customers"]
        for step in customers_bronze.transformations:
            assert step.source_notebook is None


@pytest.fixture
def single_source_findings_dir(tmp_path):
    """Directory with a single findings file but no multi_dataset_findings.yaml."""
    findings_dir = tmp_path / "findings"
    findings_dir.mkdir()

    customers_findings = {
        "source_path": "/data/customers.csv",
        "source_format": "csv",
        "row_count": 1000,
        "column_count": 5,
        "columns": {
            "customer_id": {
                "name": "customer_id", "inferred_type": "identifier",
                "confidence": 0.95, "evidence": [], "quality_score": 100,
                "cleaning_needed": False, "cleaning_recommendations": [],
            },
            "age": {
                "name": "age", "inferred_type": "numeric_continuous",
                "confidence": 0.9, "evidence": [], "quality_score": 85,
                "cleaning_needed": True,
                "cleaning_recommendations": ["impute_null:median"],
                "type_metrics": {"has_nulls": True},
            },
            "churn": {
                "name": "churn", "inferred_type": "binary",
                "confidence": 0.99, "evidence": [], "quality_score": 100,
                "cleaning_needed": False, "cleaning_recommendations": [],
            },
        },
        "target_column": "churn",
        "identifier_columns": ["customer_id"],
    }
    (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers_findings))

    return findings_dir


class TestSingleSourceFallback:
    def test_parse_without_multi_dataset_uses_single_findings(self, single_source_findings_dir):
        """No multi_dataset_findings.yaml, one *_findings.yaml -> parse succeeds."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineConfig
        parser = FindingsParser(str(single_source_findings_dir))
        config = parser.parse()
        assert isinstance(config, PipelineConfig)
        source_names = [s.name for s in config.sources]
        assert "customers" in source_names

    def test_single_source_config_has_correct_target(self, single_source_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(single_source_findings_dir))
        config = parser.parse()
        assert config.target_column == "churn"

    def test_single_source_config_has_bronze_layer(self, single_source_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(single_source_findings_dir))
        config = parser.parse()
        assert "customers" in config.bronze
        assert len(config.bronze["customers"].transformations) >= 1

    def test_single_source_has_no_joins(self, single_source_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser(str(single_source_findings_dir))
        config = parser.parse()
        assert config.silver.joins == []


class TestRawSourcePathResolution:
    def test_raw_source_path_resolved_to_absolute(self, tmp_path):
        """Relative source_path in dataset info -> absolute in source config."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        multi_dataset = {
            "datasets": {
                "data": {
                    "name": "data",
                    "findings_path": str(findings_dir / "data_findings.yaml"),
                    "source_path": "../tests/fixtures/data.csv",
                    "granularity": "entity_level",
                    "row_count": 100, "column_count": 2, "excluded": False,
                }
            },
            "relationships": [],
            "primary_entity_dataset": "data",
            "event_datasets": [],
            "excluded_datasets": [],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        data_findings = {
            "source_path": "../tests/fixtures/data.csv",
            "source_format": "csv",
            "row_count": 100, "column_count": 2,
            "columns": {
                "id": {"name": "id", "inferred_type": "identifier", "confidence": 0.95,
                       "evidence": [], "quality_score": 100,
                       "cleaning_needed": False, "cleaning_recommendations": []},
                "target": {"name": "target", "inferred_type": "binary", "confidence": 0.9,
                           "evidence": [], "quality_score": 100,
                           "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "target_column": "target",
            "identifier_columns": ["id"],
        }
        (findings_dir / "data_findings.yaml").write_text(yaml.dump(data_findings))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        source = next(s for s in config.sources if s.name == "data")
        assert Path(source.raw_source_path).is_absolute()

    def test_discovered_landing_raw_source_path_is_absolute(self, aggregated_event_setup):
        """Pre-agg source_path -> absolute in landing config."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser(str(aggregated_event_setup))
        config = parser.parse()
        assert "orders_agg" in config.landing
        landing = config.landing["orders_agg"]
        assert Path(landing.raw_source_path).is_absolute()

    def test_already_absolute_path_unchanged(self, sample_findings_dir):
        """Absolute source_path stays absolute."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()
        source = next(s for s in config.sources if s.name == "customers")
        assert source.raw_source_path == "/data/customers.csv"


class TestSourcePathFilenameOnly:
    def test_source_path_is_filename_only(self, sample_findings_dir):
        """SourceConfig.path should be just the filename for template simplicity."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()
        for source in config.sources:
            assert "/" not in source.path, f"source.path should be filename only, got {source.path}"


class TestRawSourceFormatInference:
    def test_landing_raw_source_format_inferred_delta(self, sample_findings_dir):
        """Landing for orders has non-CSV raw_source_path -> format delta."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()
        assert "orders" in config.landing
        landing = config.landing["orders"]
        assert landing.raw_source_format == "delta"

    def test_landing_raw_source_format_inferred_from_csv(self, aggregated_event_setup):
        """Discovered event landing with raw CSV -> format csv."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser(str(aggregated_event_setup))
        config = parser.parse()
        assert "orders_agg" in config.landing
        landing = config.landing["orders_agg"]
        assert landing.raw_source_format == "csv"


class TestRawTimeColumnResolution:
    """Landing should use metadata time_column (standardized) and populate raw_time_column for rename."""

    def test_discovered_landing_has_standardized_time_and_raw_rename(self, tmp_path):
        """When metadata time_column differs from raw column, time_column is standardized and raw_time_column is the original."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        raw_csv = "/data/raw/emails.csv"

        multi_dataset = {
            "datasets": {
                "emails_agg": {
                    "name": "emails_agg",
                    "findings_path": str(findings_dir / "emails_agg_findings.yaml"),
                    "source_path": str(findings_dir / "emails_agg.parquet"),
                    "granularity": "entity_level",
                    "row_count": 500, "column_count": 3, "excluded": False,
                }
            },
            "relationships": [],
            "primary_entity_dataset": "emails_agg",
            "event_datasets": [],
            "excluded_datasets": [],
            "aggregation_windows": ["7d", "30d"],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        agg_findings = {
            "source_path": str(findings_dir / "emails_agg.parquet"),
            "source_format": "parquet",
            "row_count": 500,
            "column_count": 3,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["customer_id"],
            "target_column": "target",
        }
        (findings_dir / "emails_agg_findings.yaml").write_text(yaml.dump(agg_findings))

        # Pre-agg findings: time_column is "feature_timestamp" (post-rename),
        # but actual raw column is "sent_date" (in datetime_columns)
        preagg_findings = {
            "source_path": raw_csv,
            "source_format": "csv",
            "row_count": 10000,
            "column_count": 5,
            "columns": {
                "email_id": {"name": "email_id", "inferred_type": "identifier",
                             "confidence": 0.95, "evidence": [], "quality_score": 100,
                             "cleaning_needed": False, "cleaning_recommendations": []},
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
                "sent_date": {"name": "sent_date", "inferred_type": "datetime",
                              "confidence": 0.95, "evidence": [], "quality_score": 100,
                              "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["email_id"],
            "datetime_columns": ["sent_date"],
            "time_series_metadata": {
                "granularity": "event_level",
                "entity_column": "customer_id",
                "time_column": "feature_timestamp",
                "aggregation_executed": True,
                "aggregated_findings_path": str(findings_dir / "emails_agg_findings.yaml"),
                "suggested_aggregations": ["7d", "30d"],
            },
        }
        (findings_dir / "emails_raw_findings.yaml").write_text(yaml.dump(preagg_findings))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "emails_agg" in config.landing
        landing = config.landing["emails_agg"]
        # time_column uses the standardized name from metadata
        assert landing.time_column == "feature_timestamp"
        # raw_time_column has the original column for the rename step
        assert landing.raw_time_column == "sent_date"

    def test_landing_uses_metadata_time_column_when_it_exists_in_columns(self, sample_findings_dir):
        """When time_series_metadata.time_column exists in columns, no rename needed."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()

        assert "orders" in config.landing
        landing = config.landing["orders"]
        # order_date exists in both metadata and columns, so use it as-is
        assert landing.time_column == "order_date"
        # No rename needed since raw column matches metadata
        assert landing.raw_time_column is None

    def test_resolve_raw_time_column_static_method(self):
        """Unit test for _resolve_raw_time_column with synthetic findings."""
        from customer_retention.analysis.auto_explorer.findings import (
            ColumnFinding,
            ExplorationFindings,
            TimeSeriesMetadata,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        def _col(name):
            return ColumnFinding(name=name, inferred_type="datetime", confidence=0.9, evidence=[])
        findings = ExplorationFindings(
            source_path="/data/events.csv",
            source_format="csv",
            columns={
                "sent_date": _col("sent_date"),
                "customer_id": _col("customer_id"),
            },
            datetime_columns=["sent_date"],
            time_series_metadata=TimeSeriesMetadata(
                granularity="event_level",
                entity_column="customer_id",
                time_column="feature_timestamp",
            ),
        )

        result = FindingsParser._resolve_raw_time_column(findings)
        assert result == "sent_date"

    def test_resolve_raw_time_column_returns_metadata_when_column_exists(self):
        """When metadata time_column IS in columns, return it."""
        from customer_retention.analysis.auto_explorer.findings import (
            ColumnFinding,
            ExplorationFindings,
            TimeSeriesMetadata,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        def _col(name):
            return ColumnFinding(name=name, inferred_type="datetime", confidence=0.9, evidence=[])
        findings = ExplorationFindings(
            source_path="/data/events.csv",
            source_format="csv",
            columns={
                "order_date": _col("order_date"),
                "customer_id": _col("customer_id"),
            },
            datetime_columns=["order_date"],
            time_series_metadata=TimeSeriesMetadata(
                granularity="event_level",
                entity_column="customer_id",
                time_column="order_date",
            ),
        )

        result = FindingsParser._resolve_raw_time_column(findings)
        assert result == "order_date"


class TestTimestampCoalesceGuard:

    def test_no_coalesce_when_datetime_ordering_is_empty(self):
        from customer_retention.analysis.auto_explorer.findings import (
            ColumnFinding,
            ExplorationFindings,
            TimeSeriesMetadata,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        def _col(name, typ="datetime"):
            return ColumnFinding(name=name, inferred_type=typ, confidence=0.9, evidence=[])

        findings = ExplorationFindings(
            source_path="/data/events.csv",
            source_format="csv",
            columns={"sent_date": _col("sent_date"), "unsub_date": _col("unsub_date")},
            datetime_columns=["sent_date", "unsub_date"],
            datetime_ordering=[],
            time_series_metadata=TimeSeriesMetadata(
                granularity="event_level", entity_column="customer_id",
                time_column="feature_timestamp",
            ),
        )
        parser = FindingsParser.__new__(FindingsParser)
        assert parser._build_timestamp_coalesce_config(findings) is None

    def test_no_coalesce_when_single_datetime_column(self):
        from customer_retention.analysis.auto_explorer.findings import (
            ColumnFinding,
            ExplorationFindings,
            TimeSeriesMetadata,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        def _col(name, typ="datetime"):
            return ColumnFinding(name=name, inferred_type=typ, confidence=0.9, evidence=[])

        findings = ExplorationFindings(
            source_path="/data/events.csv",
            source_format="csv",
            columns={"order_date": _col("order_date")},
            datetime_columns=["order_date"],
            time_series_metadata=TimeSeriesMetadata(
                granularity="event_level", entity_column="customer_id",
                time_column="order_date",
            ),
        )
        parser = FindingsParser.__new__(FindingsParser)
        assert parser._build_timestamp_coalesce_config(findings) is None

    def test_coalesce_created_when_explicit_ordering_has_multiple_columns(self):
        from customer_retention.analysis.auto_explorer.findings import (
            ColumnFinding,
            ExplorationFindings,
            TimeSeriesMetadata,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        def _col(name, typ="datetime"):
            return ColumnFinding(name=name, inferred_type=typ, confidence=0.9, evidence=[])

        findings = ExplorationFindings(
            source_path="/data/events.csv",
            source_format="csv",
            columns={"ts1": _col("ts1"), "ts2": _col("ts2")},
            datetime_columns=["ts1", "ts2"],
            datetime_ordering=["ts1", "ts2"],
            time_series_metadata=TimeSeriesMetadata(
                granularity="event_level", entity_column="customer_id",
                time_column="feature_timestamp",
            ),
        )
        parser = FindingsParser.__new__(FindingsParser)
        result = parser._build_timestamp_coalesce_config(findings)
        assert result is not None
        assert result.datetime_columns_ordered == ["ts1", "ts2"]

    def test_discovered_landing_no_coalesce_with_empty_datetime_ordering(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        agg_parquet = str(findings_dir / "emails_agg.parquet")
        multi_dataset = {
            "datasets": {
                "emails_agg": {
                    "name": "emails_agg",
                    "findings_path": str(findings_dir / "emails_agg_findings.yaml"),
                    "source_path": agg_parquet,
                    "granularity": "entity_level",
                    "row_count": 500, "column_count": 3, "excluded": False,
                }
            },
            "relationships": [],
            "primary_entity_dataset": "emails_agg",
            "event_datasets": [],
            "excluded_datasets": [],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        agg_findings = {
            "source_path": agg_parquet,
            "source_format": "parquet",
            "row_count": 500, "column_count": 2,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["customer_id"],
            "target_column": "target",
        }
        (findings_dir / "emails_agg_findings.yaml").write_text(yaml.dump(agg_findings))

        preagg_findings = {
            "source_path": "/data/raw/emails.csv",
            "source_format": "csv",
            "row_count": 10000, "column_count": 4,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
                "sent_date": {"name": "sent_date", "inferred_type": "datetime",
                              "confidence": 0.95, "evidence": [], "quality_score": 100,
                              "cleaning_needed": False, "cleaning_recommendations": []},
                "unsub_date": {"name": "unsub_date", "inferred_type": "datetime",
                               "confidence": 0.95, "evidence": [], "quality_score": 100,
                               "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["customer_id"],
            "datetime_columns": ["sent_date", "unsub_date"],
            "datetime_ordering": [],
            "time_series_metadata": {
                "granularity": "event_level",
                "entity_column": "customer_id",
                "time_column": "feature_timestamp",
                "aggregation_executed": True,
                "aggregated_findings_path": str(findings_dir / "emails_agg_findings.yaml"),
            },
        }
        (findings_dir / "emails_raw_findings.yaml").write_text(yaml.dump(preagg_findings))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "emails_agg" in config.landing
        assert config.landing["emails_agg"].timestamp_coalesce is None


class TestBronzeEventRawTimeColumnAttachment:
    """_build_bronze_event_configs should populate raw_time_column when rename is needed."""

    def test_bronze_event_gets_raw_time_column_when_rename_needed(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        multi_dataset = {
            "datasets": {
                "emails_agg": {
                    "name": "emails_agg",
                    "findings_path": str(findings_dir / "emails_agg_findings.yaml"),
                    "source_path": str(findings_dir / "emails_agg.parquet"),
                    "granularity": "entity_level",
                    "row_count": 500, "column_count": 3, "excluded": False,
                }
            },
            "relationships": [],
            "primary_entity_dataset": "emails_agg",
            "event_datasets": [],
            "excluded_datasets": [],
            "aggregation_windows": ["7d", "30d"],
            "notes": {"temporal_config": {"feature_groups": ["recency", "lifecycle", "regularity"]}},
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        agg_findings = {
            "source_path": str(findings_dir / "emails_agg.parquet"),
            "source_format": "parquet",
            "row_count": 500,
            "column_count": 3,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["customer_id"],
            "target_column": "target",
        }
        (findings_dir / "emails_agg_findings.yaml").write_text(yaml.dump(agg_findings))

        preagg_findings = {
            "source_path": "/data/raw/emails.csv",
            "source_format": "csv",
            "row_count": 10000,
            "column_count": 5,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
                "sent_date": {"name": "sent_date", "inferred_type": "datetime",
                              "confidence": 0.95, "evidence": [], "quality_score": 100,
                              "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["customer_id"],
            "datetime_columns": ["sent_date"],
            "time_series_metadata": {
                "granularity": "event_level",
                "entity_column": "customer_id",
                "time_column": "feature_timestamp",
                "aggregation_executed": True,
                "aggregated_findings_path": str(findings_dir / "emails_agg_findings.yaml"),
                "suggested_aggregations": ["7d", "30d"],
            },
        }
        (findings_dir / "emails_raw_findings.yaml").write_text(yaml.dump(preagg_findings))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "emails_agg" in config.bronze_event
        bronze_event = config.bronze_event["emails_agg"]
        assert bronze_event.lifecycle is not None
        assert bronze_event.time_column == "feature_timestamp"
        assert bronze_event.raw_time_column == "sent_date"

    def test_bronze_event_no_raw_time_column_when_no_rename_needed(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        multi_path = sample_findings_dir / "multi_dataset_findings.yaml"
        multi = yaml.safe_load(multi_path.read_text())
        multi["notes"] = {"temporal_config": {"feature_groups": ["recency", "lifecycle"]}}
        multi_path.write_text(yaml.dump(multi))

        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()

        assert "orders" in config.bronze_event
        bronze_event = config.bronze_event["orders"]
        assert bronze_event.lifecycle is not None
        assert bronze_event.raw_time_column is None


class TestResolveOriginalTarget:

    def test_returns_original_when_differs_from_target(self):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings = ExplorationFindings(
            source_path="/data/events.csv", source_format="csv", columns={},
            metadata={"original_target_column": "unsubscribed"},
        )
        assert FindingsParser._resolve_original_target(findings, "target") == "unsubscribed"

    def test_returns_none_when_same_as_target(self):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings = ExplorationFindings(
            source_path="/data/events.csv", source_format="csv", columns={},
            metadata={"original_target_column": "target"},
        )
        assert FindingsParser._resolve_original_target(findings, "target") is None

    def test_returns_none_when_metadata_missing(self):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings = ExplorationFindings(
            source_path="/data/events.csv", source_format="csv", columns={},
        )
        assert FindingsParser._resolve_original_target(findings, "target") is None

    def test_returns_none_when_metadata_has_no_original_target(self):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings = ExplorationFindings(
            source_path="/data/events.csv", source_format="csv", columns={},
            metadata={"some_other_key": "value"},
        )
        assert FindingsParser._resolve_original_target(findings, "target") is None
