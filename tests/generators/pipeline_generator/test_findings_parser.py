from pathlib import Path

import pytest
import yaml


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
            }
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
        assert config.silver.joins[0]["left_keys"] == ["customer_id"]

    def test_silver_joins_strip_as_of_date_for_event_sources(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        multi_dataset = {
            "datasets": {
                "profiles": {
                    "name": "profiles",
                    "findings_path": str(findings_dir / "profiles_findings.yaml"),
                    "source_path": "/data/profiles.csv",
                    "granularity": "entity_level",
                    "row_count": 100,
                    "column_count": 3,
                    "excluded": False,
                },
                "transactions": {
                    "name": "transactions",
                    "findings_path": str(findings_dir / "transactions_findings.yaml"),
                    "source_path": "/data/transactions.csv",
                    "granularity": "event_level",
                    "row_count": 5000,
                    "column_count": 4,
                    "excluded": False,
                    "entity_column": "customer_id",
                    "time_column": "event_timestamp",
                },
            },
            "relationships": [
                {
                    "left_dataset": "profiles",
                    "right_dataset": "transactions",
                    "left_columns": ["customer_id", "as_of_date"],
                    "right_columns": ["customer_id", "as_of_date"],
                    "relationship_type": "one_to_many",
                    "confidence": 1.0,
                }
            ],
            "primary_entity_dataset": "profiles",
            "event_datasets": ["transactions"],
            "excluded_datasets": [],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))
        profiles_findings = {
            "source_path": "/data/profiles.csv",
            "source_format": "csv",
            "row_count": 100,
            "column_count": 3,
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
                "churned": {
                    "name": "churned",
                    "inferred_type": "binary",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "identifier_columns": ["customer_id"],
            "target_column": "churned",
        }
        (findings_dir / "profiles_findings.yaml").write_text(yaml.dump(profiles_findings))
        txn_findings = {
            "source_path": "/data/transactions.csv",
            "source_format": "csv",
            "row_count": 5000,
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
                "event_timestamp": {
                    "name": "event_timestamp",
                    "inferred_type": "datetime",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "identifier_columns": ["customer_id"],
            "datetime_columns": ["event_timestamp"],
            "time_series_metadata": {
                "granularity": "event_level",
                "entity_column": "customer_id",
                "time_column": "event_timestamp",
                "aggregation_windows_used": ["7d", "30d", "90d"],
            },
        }
        (findings_dir / "transactions_findings.yaml").write_text(yaml.dump(txn_findings))
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()
        assert len(config.silver.joins) == 1
        assert config.silver.joins[0]["left_keys"] == ["customer_id"]
        assert config.silver.joins[0]["right_keys"] == ["customer_id"]

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
                "name": "customer_id",
                "inferred_type": "identifier",
                "confidence": 0.95,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": [],
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

    # Aggregated findings (entity-level, no time_series_metadata)
    agg_findings = {
        "source_path": agg_parquet,
        "source_format": "parquet",
        "row_count": 500,
        "column_count": 6,
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
            "total_amount": {
                "name": "total_amount",
                "inferred_type": "numeric_continuous",
                "confidence": 0.9,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": True,
                "cleaning_recommendations": ["impute_null:0"],
            },
            "order_count": {
                "name": "order_count",
                "inferred_type": "numeric_discrete",
                "confidence": 0.9,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": [],
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
            "aggregation_executed": True,
            "aggregated_findings_path": str(findings_dir / "orders_agg_findings.yaml"),
            "suggested_aggregations": ["7d", "30d", "90d"],
            "aggregation_windows_used": ["7d", "30d", "90d"],
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
                    "row_count": 100,
                    "column_count": 2,
                    "excluded": False,
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
        cap_steps = [t for t in bronze_event.pre_shaping if t.type == PipelineTransformationType.CAP_OUTLIER]
        assert len(cap_steps) == 1
        assert cap_steps[0].column == "amount"

    def test_bronze_has_post_agg_transformations_only(self, aggregated_event_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType

        parser = FindingsParser(str(aggregated_event_setup))
        config = parser.parse()

        assert "orders_agg" not in config.bronze
        assert "orders_agg" in config.bronze_event
        impute_steps = [
            t
            for t in config.bronze_event["orders_agg"].post_shaping
            if t.type == PipelineTransformationType.IMPUTE_NULL
        ]
        assert len(impute_steps) == 0

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
        impute_steps = [t for t in bronze_event.post_shaping if t.type == PipelineTransformationType.IMPUTE_NULL]
        assert len(impute_steps) == 0
        cap_steps = [t for t in bronze_event.pre_shaping if t.type == PipelineTransformationType.CAP_OUTLIER]
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
                    "row_count": 500,
                    "column_count": 3,
                    "excluded": False,
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
            "row_count": 500,
            "column_count": 3,
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
            },
            "identifier_columns": ["customer_id"],
            "target_column": "target",
        }
        (findings_dir / "emails_agg_findings.yaml").write_text(yaml.dump(agg_findings))

        preagg_findings = {
            "source_path": "/data/raw/emails.csv",
            "source_format": "csv",
            "row_count": 10000,
            "column_count": 7,
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
                "sent_date": {
                    "name": "sent_date",
                    "inferred_type": "datetime",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "send_hour": {
                    "name": "send_hour",
                    "inferred_type": "numeric_discrete",
                    "confidence": 0.7,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "opened": {
                    "name": "opened",
                    "inferred_type": "binary",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "clicked": {
                    "name": "clicked",
                    "inferred_type": "binary",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "bounced": {
                    "name": "bounced",
                    "inferred_type": "binary",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
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
                "aggregation_windows_used": ["7d", "30d"],
            },
        }
        (findings_dir / "emails_raw_findings.yaml").write_text(yaml.dump(preagg_findings))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "emails_agg" in config.bronze_event
        agg_config = config.bronze_event["emails_agg"].aggregation
        assert agg_config is not None
        assert "send_hour" in agg_config.value_columns
        assert "opened" in agg_config.categorical_columns
        assert "clicked" in agg_config.categorical_columns
        assert "bounced" in agg_config.categorical_columns

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
                    "row_count": 100,
                    "column_count": 2,
                    "excluded": False,
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
            "row_count": 100,
            "column_count": 2,
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
            },
            "identifier_columns": ["customer_id"],
            "target_column": "target",
        }
        (findings_dir / "clean_agg_findings.yaml").write_text(yaml.dump(agg_findings))

        preagg_findings = {
            "source_path": "/data/raw/clean.csv",
            "source_format": "csv",
            "row_count": 5000,
            "column_count": 3,
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
                "event_date": {
                    "name": "event_date",
                    "inferred_type": "datetime",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "identifier_columns": ["customer_id"],
            "datetime_columns": ["event_date"],
            "time_series_metadata": {
                "granularity": "event_level",
                "entity_column": "customer_id",
                "time_column": "event_date",
                "aggregation_executed": True,
                "aggregated_findings_path": str(findings_dir / "clean_agg_findings.yaml"),
                "aggregation_windows_used": ["7d", "30d", "90d"],
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
                    "row_count": 100,
                    "column_count": 2,
                    "excluded": False,
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
            "row_count": 100,
            "column_count": 2,
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
            },
            "identifier_columns": ["customer_id"],
            "target_column": "churned",
        }
        (findings_dir / "events_agg_findings.yaml").write_text(yaml.dump(agg_findings))

        preagg_findings = {
            "source_path": "/data/raw/events.csv",
            "source_format": "csv",
            "row_count": 5000,
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
                "event_date": {
                    "name": "event_date",
                    "inferred_type": "datetime",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "opened": {
                    "name": "opened",
                    "inferred_type": "binary",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "churned": {
                    "name": "churned",
                    "inferred_type": "binary",
                    "confidence": 0.9,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
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
                "aggregation_windows_used": ["7d", "30d", "90d"],
            },
        }
        (findings_dir / "events_raw_findings.yaml").write_text(yaml.dump(preagg_findings))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        agg_config = config.bronze_event["events_agg"].aggregation
        assert "opened" in agg_config.categorical_columns
        assert "churned" not in agg_config.categorical_columns
        assert "churned" not in agg_config.value_columns


class TestExplicitEventPreAggTransforms:
    def test_explicit_event_pre_shaping_in_bronze_event(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType

        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()

        assert "orders" in config.bronze_event
        bronze_event = config.bronze_event["orders"]
        cap_steps = [t for t in bronze_event.pre_shaping if t.type == PipelineTransformationType.CAP_OUTLIER]
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
                "test": {
                    "name": "test",
                    "findings_path": str(findings_dir / "test_findings.yaml"),
                    "source_path": "/test.csv",
                    "granularity": "entity_level",
                    "row_count": 100,
                    "column_count": 2,
                    "excluded": False,
                }
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
                    "row_count": 100,
                    "column_count": 3,
                    "excluded": False,
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
                            "source_notebook": "02_source_integrity",
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
                            "source_notebook": "02_source_integrity",
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
            assert step.source_notebook == "02_source_integrity"

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
                    "row_count": 100,
                    "column_count": 2,
                    "excluded": False,
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
    def test_landing_raw_source_format_inferred_parquet(self, sample_findings_dir):
        """Landing for orders has .parquet raw_source_path -> format parquet."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()
        assert "orders" in config.landing
        landing = config.landing["orders"]
        assert landing.raw_source_format == "parquet"

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
                    "row_count": 500,
                    "column_count": 3,
                    "excluded": False,
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
                "customer_id": {
                    "name": "customer_id",
                    "inferred_type": "identifier",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
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
                "email_id": {
                    "name": "email_id",
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
                "sent_date": {
                    "name": "sent_date",
                    "inferred_type": "datetime",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
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
                "aggregation_windows_used": ["7d", "30d"],
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
                granularity="event_level",
                entity_column="customer_id",
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
                granularity="event_level",
                entity_column="customer_id",
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
                granularity="event_level",
                entity_column="customer_id",
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
                    "row_count": 500,
                    "column_count": 3,
                    "excluded": False,
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
            "row_count": 500,
            "column_count": 2,
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
            },
            "identifier_columns": ["customer_id"],
            "target_column": "target",
        }
        (findings_dir / "emails_agg_findings.yaml").write_text(yaml.dump(agg_findings))

        preagg_findings = {
            "source_path": "/data/raw/emails.csv",
            "source_format": "csv",
            "row_count": 10000,
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
                "sent_date": {
                    "name": "sent_date",
                    "inferred_type": "datetime",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "unsub_date": {
                    "name": "unsub_date",
                    "inferred_type": "datetime",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
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
                "aggregation_windows_used": ["7d", "30d", "90d"],
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
                    "row_count": 500,
                    "column_count": 3,
                    "excluded": False,
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
                "customer_id": {
                    "name": "customer_id",
                    "inferred_type": "identifier",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
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
                "customer_id": {
                    "name": "customer_id",
                    "inferred_type": "identifier",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
                "sent_date": {
                    "name": "sent_date",
                    "inferred_type": "datetime",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
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
                "aggregation_windows_used": ["7d", "30d"],
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


class TestLifecycleMetadataFallback:

    def _make_findings_dir(self, tmp_path, *, multi_notes=None, event_metadata=None):
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
                    "column_count": 2,
                    "excluded": False,
                },
                "events_agg": {
                    "name": "events_agg",
                    "findings_path": str(findings_dir / "events_agg_findings.yaml"),
                    "source_path": str(findings_dir / "events_agg.parquet"),
                    "granularity": "entity_level",
                    "row_count": 500,
                    "column_count": 3,
                    "excluded": False,
                },
            },
            "relationships": [
                {
                    "left_dataset": "customers",
                    "right_dataset": "events_agg",
                    "left_column": "customer_id",
                    "right_column": "customer_id",
                    "relationship_type": "one_to_one",
                    "confidence": 1.0,
                },
            ],
            "primary_entity_dataset": "customers",
            "event_datasets": [],
            "excluded_datasets": [],
            "aggregation_windows": ["7d", "30d", "90d", "365d"],
        }
        if multi_notes is not None:
            multi_dataset["notes"] = multi_notes
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        customers_findings = {
            "source_path": "/data/customers.csv",
            "source_format": "csv",
            "row_count": 1000,
            "column_count": 2,
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

        agg_findings = {
            "source_path": str(findings_dir / "events_agg.parquet"),
            "source_format": "parquet",
            "row_count": 500,
            "column_count": 3,
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
            },
            "identifier_columns": ["customer_id"],
            "target_column": "churn",
        }
        (findings_dir / "events_agg_findings.yaml").write_text(yaml.dump(agg_findings))

        preagg_findings = {
            "source_path": "/data/raw/events.csv",
            "source_format": "csv",
            "row_count": 10000,
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
                "event_date": {
                    "name": "event_date",
                    "inferred_type": "datetime",
                    "confidence": 0.95,
                    "evidence": [],
                    "quality_score": 100,
                    "cleaning_needed": False,
                    "cleaning_recommendations": [],
                },
            },
            "identifier_columns": ["customer_id"],
            "datetime_columns": ["event_date"],
            "time_series_metadata": {
                "granularity": "event_level",
                "entity_column": "customer_id",
                "time_column": "event_date",
                "aggregation_executed": True,
                "aggregated_findings_path": str(findings_dir / "events_agg_findings.yaml"),
                "aggregation_windows_used": ["7d", "30d", "90d"],
            },
        }
        if event_metadata is not None:
            preagg_findings["metadata"] = event_metadata
        (findings_dir / "events_raw_findings.yaml").write_text(yaml.dump(preagg_findings))

        return findings_dir

    def test_lifecycle_from_metadata_when_notes_empty(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = self._make_findings_dir(
            tmp_path,
            multi_notes={},
            event_metadata={
                "aggregation": {
                    "include_lifecycle_quadrant": True,
                    "include_recency": True,
                    "include_tenure": True,
                },
            },
        )
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "events_agg" in config.bronze_event
        bronze_event = config.bronze_event["events_agg"]
        assert bronze_event.lifecycle is not None
        assert bronze_event.lifecycle.include_lifecycle_quadrant is True
        assert bronze_event.lifecycle.include_recency_bucket is True

    def test_lifecycle_notes_priority_over_metadata(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = self._make_findings_dir(
            tmp_path,
            multi_notes={"temporal_config": {"feature_groups": ["recency"]}},
            event_metadata={
                "aggregation": {
                    "include_lifecycle_quadrant": True,
                    "include_recency": True,
                },
            },
        )
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "events_agg" in config.bronze_event
        bronze_event = config.bronze_event["events_agg"]
        assert bronze_event.lifecycle is not None
        assert bronze_event.lifecycle.include_recency_bucket is True
        assert bronze_event.lifecycle.include_lifecycle_quadrant is False

    def test_no_lifecycle_when_both_empty(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = self._make_findings_dir(
            tmp_path,
            multi_notes={},
            event_metadata={},
        )
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "events_agg" in config.bronze_event
        bronze_event = config.bronze_event["events_agg"]
        assert bronze_event.lifecycle is None

    def test_lifecycle_from_feature_flags(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = self._make_findings_dir(
            tmp_path,
            multi_notes={},
            event_metadata={
                "temporal_patterns": {
                    "feature_flags": {
                        "include_lifecycle_quadrant": True,
                        "include_recency": True,
                        "include_seasonality_features": True,
                    },
                },
            },
        )
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "events_agg" in config.bronze_event
        bronze_event = config.bronze_event["events_agg"]
        assert bronze_event.lifecycle is not None
        assert bronze_event.lifecycle.include_lifecycle_quadrant is True
        assert bronze_event.lifecycle.include_recency_bucket is True
        assert bronze_event.lifecycle.include_cyclical_features is True

    def test_momentum_pairs_from_aggregation_windows(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = self._make_findings_dir(
            tmp_path,
            multi_notes={},
            event_metadata={
                "aggregation": {"include_recency": True},
            },
        )
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        bronze_event = config.bronze_event["events_agg"]
        assert bronze_event.lifecycle is not None
        pairs = bronze_event.lifecycle.momentum_pairs
        assert len(pairs) > 0
        assert pairs[0]["short_window"] == "7d"
        assert pairs[0]["long_window"] == "30d"

    def test_momentum_pairs_empty_without_day_windows(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = self._make_findings_dir(tmp_path, multi_notes={}, event_metadata={})
        multi_path = findings_dir / "multi_dataset_findings.yaml"
        multi = yaml.safe_load(multi_path.read_text())
        multi["aggregation_windows"] = ["all_time"]
        multi_path.write_text(yaml.dump(multi))

        preagg_path = findings_dir / "events_raw_findings.yaml"
        preagg = yaml.safe_load(preagg_path.read_text())
        preagg["metadata"] = {"aggregation": {"include_recency": True}}
        preagg_path.write_text(yaml.dump(preagg))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        bronze_event = config.bronze_event["events_agg"]
        assert bronze_event.lifecycle is not None
        assert bronze_event.lifecycle.momentum_pairs == []


class TestFindingsParserWithNamespace:
    @pytest.fixture
    def namespace_setup(self, tmp_path):
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

        ns = RunNamespace(root=tmp_path, run_id="test-run-abc")
        ns.setup()

        cust_fd = ns.dataset_findings_dir("customers")
        cust_fd.mkdir(parents=True)
        customers_findings = {
            "source_path": "/data/customers.csv",
            "source_format": "csv",
            "row_count": 1000,
            "column_count": 3,
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
        (cust_fd / "customers_findings.yaml").write_text(yaml.dump(customers_findings))

        orders_fd = ns.dataset_findings_dir("orders")
        orders_fd.mkdir(parents=True)
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
        (orders_fd / "orders_findings.yaml").write_text(yaml.dump(orders_findings))

        multi_dataset = {
            "datasets": {
                "customers": {
                    "name": "customers",
                    "findings_path": str(cust_fd / "customers_findings.yaml"),
                    "source_path": "/data/customers.csv",
                    "granularity": "entity_level",
                    "row_count": 1000,
                    "column_count": 3,
                    "excluded": False,
                },
                "orders": {
                    "name": "orders",
                    "findings_path": str(orders_fd / "orders_findings.yaml"),
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
        ns.multi_dataset_findings_path.parent.mkdir(parents=True, exist_ok=True)
        ns.multi_dataset_findings_path.write_text(yaml.dump(multi_dataset))

        recommendations = {
            "version": "1.0",
            "sources": {
                "customers": {
                    "source_file": "/data/customers.csv",
                    "null_handling": [],
                    "outlier_handling": [],
                    "type_casts": [],
                },
            },
        }
        ns.merged_recommendations_path.write_text(yaml.dump(recommendations))

        return ns

    def test_load_multi_dataset_findings_from_namespace(self, namespace_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = namespace_setup
        parser = FindingsParser(str(ns.merged_dir), namespace=ns)
        multi = parser._load_multi_dataset_findings()
        assert "customers" in multi.datasets
        assert "orders" in multi.datasets

    def test_load_source_findings_from_namespace(self, namespace_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = namespace_setup
        parser = FindingsParser(str(ns.merged_dir), namespace=ns)
        multi = parser._load_multi_dataset_findings()
        findings = parser._load_source_findings(["customers", "orders"], ns.merged_dir, multi)
        assert "customers" in findings
        assert "orders" in findings

    def test_load_recommendations_from_namespace(self, namespace_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = namespace_setup
        parser = FindingsParser(str(ns.merged_dir), namespace=ns)
        rec = parser._load_recommendations()
        assert rec is not None
        assert "customers" in rec.sources

    def test_full_parse_with_namespace(self, namespace_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineConfig

        ns = namespace_setup
        parser = FindingsParser(str(ns.merged_dir), namespace=ns)
        config = parser.parse()
        assert isinstance(config, PipelineConfig)
        source_names = [s.name for s in config.sources]
        assert "customers" in source_names
        assert "orders" in source_names

    def test_scan_for_preagg_uses_namespace(self, namespace_setup):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = namespace_setup
        parser = FindingsParser(str(ns.merged_dir), namespace=ns)
        all_findings = ns.discover_all_findings(prefer_aggregated=False)
        assert len(all_findings) >= 1


class TestSynthesizeFromNamespace:
    def test_synthesize_uses_namespace_when_flat_dir_empty(self, tmp_path):
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineConfig

        ns = RunNamespace(root=tmp_path, run_id="synth-run")
        ns.setup()

        cust_fd = ns.dataset_findings_dir("customers")
        cust_fd.mkdir(parents=True)
        customers_findings = {
            "source_path": "/data/customers.csv",
            "source_format": "csv",
            "row_count": 500,
            "column_count": 3,
            "columns": {
                "customer_id": {
                    "name": "customer_id",
                    "inferred_type": "identifier",
                    "confidence": 0.95,
                    "evidence": [],
                },
                "churn": {"name": "churn", "inferred_type": "binary", "confidence": 0.99, "evidence": []},
            },
            "target_column": "churn",
            "identifier_columns": ["customer_id"],
        }
        (cust_fd / "customers_findings.yaml").write_text(yaml.dump(customers_findings))

        flat_dir = tmp_path / "findings"
        flat_dir.mkdir()

        parser = FindingsParser(str(flat_dir), namespace=ns)
        config = parser.parse()
        assert isinstance(config, PipelineConfig)
        source_names = [s.name for s in config.sources]
        assert "customers" in source_names

    def test_synthesize_raises_when_no_findings_anywhere(self, tmp_path):
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = RunNamespace(root=tmp_path, run_id="empty-run")
        ns.setup()

        flat_dir = tmp_path / "findings"
        flat_dir.mkdir()

        parser = FindingsParser(str(flat_dir), namespace=ns)
        with pytest.raises(FileNotFoundError):
            parser.parse()


class TestResolveOriginalTarget:
    def test_returns_original_when_differs_from_target(self):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings = ExplorationFindings(
            source_path="/data/events.csv",
            source_format="csv",
            columns={},
            metadata={"original_target_column": "unsubscribed"},
        )
        assert FindingsParser._resolve_original_target(findings, "target") == "unsubscribed"

    def test_returns_none_when_same_as_target(self):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings = ExplorationFindings(
            source_path="/data/events.csv",
            source_format="csv",
            columns={},
            metadata={"original_target_column": "target"},
        )
        assert FindingsParser._resolve_original_target(findings, "target") is None

    def test_returns_none_when_metadata_missing(self):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings = ExplorationFindings(
            source_path="/data/events.csv",
            source_format="csv",
            columns={},
        )
        assert FindingsParser._resolve_original_target(findings, "target") is None

    def test_returns_none_when_metadata_has_no_original_target(self):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings = ExplorationFindings(
            source_path="/data/events.csv",
            source_format="csv",
            columns={},
            metadata={"some_other_key": "value"},
        )
        assert FindingsParser._resolve_original_target(findings, "target") is None


class TestLabelTimestampWithIntent:
    def test_label_timestamp_uses_intent_observation_window(self):
        from customer_retention.analysis.auto_explorer.findings import (
            ColumnFinding,
            ExplorationFindings,
        )
        from customer_retention.analysis.auto_explorer.project_context import IntentConfig
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        def _col(name, typ="numeric_continuous"):
            return ColumnFinding(name=name, inferred_type=typ, confidence=0.9, evidence=[])

        findings = ExplorationFindings(
            source_path="/data/events.csv",
            source_format="csv",
            columns={"age": _col("age")},
            observation_window_days=180,
        )
        intent = IntentConfig(observation_window_days=270)
        parser = FindingsParser.__new__(FindingsParser)
        parser._intent = intent
        result = parser._build_label_timestamp_config(findings)
        assert result is not None
        assert result.fallback_window_days == 270

    def test_label_timestamp_falls_back_without_intent(self):
        from customer_retention.analysis.auto_explorer.findings import (
            ColumnFinding,
            ExplorationFindings,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        def _col(name, typ="numeric_continuous"):
            return ColumnFinding(name=name, inferred_type=typ, confidence=0.9, evidence=[])

        findings = ExplorationFindings(
            source_path="/data/events.csv",
            source_format="csv",
            columns={"age": _col("age")},
            observation_window_days=180,
        )
        parser = FindingsParser.__new__(FindingsParser)
        parser._intent = None
        result = parser._build_label_timestamp_config(findings)
        assert result is None


class TestFilterRecommendationsSkipDerivedColumns:
    def test_pre_shaping_skips_columns_not_in_raw_source(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            LayeredRecommendation,
            RecommendationRegistry,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import (
            BronzeEventConfig,
            PipelineConfig,
            SourceConfig,
        )

        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = {"events": {"customer_id", "sent_date", "opened"}}
        parser._source_findings_paths = {}

        source = SourceConfig(name="events", path="events.csv", format="csv", entity_key="customer_id", raw_source_path="/data/events.csv")
        config = PipelineConfig(name="test", target_column="churn", sources=[source], bronze={}, silver=None, gold=None, output_dir=".")
        config.bronze_event["events"] = BronzeEventConfig(source=source, entity_column="customer_id", time_column="sent_date")

        registry = RecommendationRegistry()
        registry.add_source("events", "events.csv")
        registry.sources["events"].filtering = [
            LayeredRecommendation(id="f1", layer="bronze", category="filter", action="notna", target_column="opened", parameters={"condition": "notna"}, rationale="nulls", source_notebook="04"),
            LayeredRecommendation(id="f2", layer="bronze", category="filter", action="notna", target_column="opened_velocity_pct", parameters={"condition": "notna"}, rationale="nulls", source_notebook="04"),
        ]
        parser._apply_filter_recommendations(config, registry)
        pre_cols = [s.column for s in config.bronze_event["events"].pre_shaping]
        assert "opened" in pre_cols
        assert "opened_velocity_pct" not in pre_cols

    def test_pre_shaping_skips_all_when_no_raw_index(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            LayeredRecommendation,
            RecommendationRegistry,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import (
            BronzeEventConfig,
            PipelineConfig,
            SourceConfig,
        )

        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = {}
        parser._source_findings_paths = {}

        source = SourceConfig(name="events", path="events.csv", format="csv", entity_key="customer_id", raw_source_path="/data/events.csv")
        config = PipelineConfig(name="test", target_column="churn", sources=[source], bronze={}, silver=None, gold=None, output_dir=".")
        config.bronze_event["events"] = BronzeEventConfig(source=source, entity_column="customer_id", time_column="sent_date")

        registry = RecommendationRegistry()
        registry.add_source("events", "events.csv")
        registry.sources["events"].filtering = [
            LayeredRecommendation(id="f1", layer="bronze", category="filter", action="notna", target_column="velocity_pct", parameters={"condition": "notna"}, rationale="nulls", source_notebook="04"),
        ]
        parser._apply_filter_recommendations(config, registry)
        pre_cols = [s.column for s in config.bronze_event["events"].pre_shaping]
        assert "velocity_pct" not in pre_cols

    def test_default_source_resolves_to_event_name(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            BronzeRecommendations,
            LayeredRecommendation,
            RecommendationRegistry,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import (
            BronzeEventConfig,
            BronzeLayerConfig,
            PipelineConfig,
            SourceConfig,
        )

        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = {"events": {"customer_id", "sent_date", "opened"}}
        parser._source_findings_paths = {}

        source = SourceConfig(name="events", path="events.csv", format="csv", entity_key="customer_id", raw_source_path="/data/events.csv")
        config = PipelineConfig(name="test", target_column="churn", sources=[source], bronze={}, silver=None, gold=None, output_dir=".")
        config.bronze_event["events"] = BronzeEventConfig(source=source, entity_column="customer_id", time_column="sent_date")
        config.bronze["events"] = BronzeLayerConfig(source=source)

        registry = RecommendationRegistry()
        registry.bronze = BronzeRecommendations(source_file="events.csv")
        registry.bronze.filtering = [
            LayeredRecommendation(id="f1", layer="bronze", category="filter", action="notna", target_column="opened", parameters={"condition": "notna"}, rationale="nulls", source_notebook="04"),
            LayeredRecommendation(id="f2", layer="bronze", category="filter", action="notna", target_column="opened_velocity_pct", parameters={"condition": "notna"}, rationale="nulls", source_notebook="04"),
        ]
        parser._apply_filter_recommendations(config, registry)
        pre_cols = [s.column for s in config.bronze_event["events"].pre_shaping]
        assert "opened" in pre_cols
        assert "opened_velocity_pct" not in pre_cols
        bronze_cols = [s.column for s in config.bronze["events"].transformations]
        assert "opened" in bronze_cols
        assert "opened_velocity_pct" not in bronze_cols


class TestBuildTrainingConfigWithIntent:
    def _make_parser_with_intent(self, intent):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser.__new__(FindingsParser)
        parser._intent = intent
        return parser

    def _empty_multi(self):
        from customer_retention.analysis.auto_explorer.exploration_manager import MultiDatasetFindings
        return MultiDatasetFindings()

    def test_temporal_split_strategy(self):
        from customer_retention.analysis.auto_explorer.project_context import IntentConfig, SplitStrategy
        intent = IntentConfig(split_strategy=SplitStrategy.TEMPORAL)
        parser = self._make_parser_with_intent(intent)
        result = parser._build_training_config(self._empty_multi(), {})
        assert result is not None
        assert result.split_strategy == "temporal"

    def test_cohort_based_split_strategy(self):
        from customer_retention.analysis.auto_explorer.project_context import IntentConfig, SplitStrategy
        intent = IntentConfig(split_strategy=SplitStrategy.COHORT_BASED)
        parser = self._make_parser_with_intent(intent)
        result = parser._build_training_config(self._empty_multi(), {})
        assert result is not None
        assert result.split_strategy == "cohort_based"

    def test_temporal_split_kept_when_event_datasets_exist(self):
        from customer_retention.analysis.auto_explorer.exploration_manager import MultiDatasetFindings
        from customer_retention.analysis.auto_explorer.project_context import IntentConfig, SplitStrategy
        intent = IntentConfig(split_strategy=SplitStrategy.TEMPORAL)
        parser = self._make_parser_with_intent(intent)
        multi = MultiDatasetFindings(event_datasets=["orders"])
        result = parser._build_training_config(multi, {})
        assert result is not None
        assert result.split_strategy == "temporal"

    def test_temporal_split_kept_when_no_event_datasets(self):
        from customer_retention.analysis.auto_explorer.project_context import IntentConfig, SplitStrategy
        intent = IntentConfig(split_strategy=SplitStrategy.TEMPORAL)
        parser = self._make_parser_with_intent(intent)
        result = parser._build_training_config(self._empty_multi(), {})
        assert result is not None
        assert result.split_strategy == "temporal"

    def test_default_split_without_intent_returns_none(self):
        parser = self._make_parser_with_intent(None)
        result = parser._build_training_config(self._empty_multi(), {})
        assert result is None


class TestAggregationConfigSeparatesNumericAndCategorical:
    def test_numeric_columns_only_in_value_columns(self, tmp_path):
        from customer_retention.analysis.auto_explorer.findings import (
            ColumnFinding,
            ExplorationFindings,
            TimeSeriesMetadata,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings = ExplorationFindings(
            source_path="/data/events.csv",
            source_format="csv",
            columns={
                "customer_id": ColumnFinding(
                    name="customer_id", inferred_type="identifier", confidence=0.95, evidence=[]
                ),
                "sent_date": ColumnFinding(name="sent_date", inferred_type="datetime", confidence=0.95, evidence=[]),
                "amount": ColumnFinding(name="amount", inferred_type="numeric_continuous", confidence=0.9, evidence=[]),
                "count": ColumnFinding(name="count", inferred_type="numeric_discrete", confidence=0.9, evidence=[]),
                "direction": ColumnFinding(
                    name="direction", inferred_type="categorical_nominal", confidence=0.9, evidence=[]
                ),
                "status": ColumnFinding(
                    name="status", inferred_type="categorical_ordinal", confidence=0.9, evidence=[]
                ),
                "opened": ColumnFinding(name="opened", inferred_type="binary", confidence=0.9, evidence=[]),
            },
            time_series_metadata=TimeSeriesMetadata(
                granularity="event_level",
                entity_column="customer_id",
                time_column="sent_date",
                aggregation_windows_used=["7d", "30d"],
            ),
            identifier_columns=["customer_id"],
            datetime_columns=["sent_date"],
        )

        class FakeMulti:
            aggregation_windows = ["7d", "30d"]

        parser = FindingsParser.__new__(FindingsParser)
        result = parser._build_aggregation_config(FakeMulti(), findings)

        assert result is not None
        assert "amount" in result.value_columns
        assert "count" in result.value_columns
        assert "direction" not in result.value_columns
        assert "status" not in result.value_columns
        assert "opened" not in result.value_columns

    def test_categorical_columns_in_categorical_columns(self, tmp_path):
        from customer_retention.analysis.auto_explorer.findings import (
            ColumnFinding,
            ExplorationFindings,
            TimeSeriesMetadata,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings = ExplorationFindings(
            source_path="/data/events.csv",
            source_format="csv",
            columns={
                "customer_id": ColumnFinding(
                    name="customer_id", inferred_type="identifier", confidence=0.95, evidence=[]
                ),
                "sent_date": ColumnFinding(name="sent_date", inferred_type="datetime", confidence=0.95, evidence=[]),
                "amount": ColumnFinding(name="amount", inferred_type="numeric_continuous", confidence=0.9, evidence=[]),
                "direction": ColumnFinding(
                    name="direction", inferred_type="categorical_nominal", confidence=0.9, evidence=[]
                ),
                "status": ColumnFinding(
                    name="status", inferred_type="categorical_ordinal", confidence=0.9, evidence=[]
                ),
                "opened": ColumnFinding(name="opened", inferred_type="binary", confidence=0.9, evidence=[]),
                "day_of_week": ColumnFinding(
                    name="day_of_week", inferred_type="categorical_cyclical", confidence=0.9, evidence=[]
                ),
            },
            time_series_metadata=TimeSeriesMetadata(
                granularity="event_level",
                entity_column="customer_id",
                time_column="sent_date",
                aggregation_windows_used=["7d"],
            ),
            identifier_columns=["customer_id"],
            datetime_columns=["sent_date"],
        )

        class FakeMulti:
            aggregation_windows = ["7d"]

        parser = FindingsParser.__new__(FindingsParser)
        result = parser._build_aggregation_config(FakeMulti(), findings)

        assert "direction" in result.categorical_columns
        assert "status" in result.categorical_columns
        assert "opened" in result.categorical_columns
        assert "day_of_week" in result.categorical_columns
        assert "amount" not in result.categorical_columns
        assert result.categorical_agg_funcs == ["nunique", "mode"]

    def test_target_excluded_from_both_lists(self, tmp_path):
        from customer_retention.analysis.auto_explorer.findings import (
            ColumnFinding,
            ExplorationFindings,
            TimeSeriesMetadata,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings = ExplorationFindings(
            source_path="/data/events.csv",
            source_format="csv",
            columns={
                "customer_id": ColumnFinding(
                    name="customer_id", inferred_type="identifier", confidence=0.95, evidence=[]
                ),
                "event_date": ColumnFinding(name="event_date", inferred_type="datetime", confidence=0.95, evidence=[]),
                "churned": ColumnFinding(name="churned", inferred_type="binary", confidence=0.9, evidence=[]),
                "amount": ColumnFinding(name="amount", inferred_type="numeric_continuous", confidence=0.9, evidence=[]),
            },
            target_column="churned",
            time_series_metadata=TimeSeriesMetadata(
                granularity="event_level",
                entity_column="customer_id",
                time_column="event_date",
                aggregation_windows_used=["7d"],
            ),
            identifier_columns=["customer_id"],
            datetime_columns=["event_date"],
        )

        class FakeMulti:
            aggregation_windows = ["7d"]

        parser = FindingsParser.__new__(FindingsParser)
        result = parser._build_aggregation_config(FakeMulti(), findings)

        assert "churned" not in result.value_columns
        assert "churned" not in result.categorical_columns
        assert "amount" in result.value_columns


class TestSourceConfigFormatInference:
    def test_dataframe_format_inferred_as_csv_from_csv_path(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        multi_dataset = {
            "datasets": {
                "profiles": {
                    "name": "profiles",
                    "findings_path": str(findings_dir / "profiles_findings.yaml"),
                    "source_path": "/data/raw/customer_profiles.csv",
                    "granularity": "entity_level",
                    "row_count": 100,
                    "column_count": 3,
                    "excluded": False,
                }
            },
            "relationships": [],
            "primary_entity_dataset": "profiles",
            "event_datasets": [],
            "excluded_datasets": [],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        profiles_findings = {
            "source_path": "/data/raw/customer_profiles.csv",
            "source_format": "dataframe",
            "row_count": 100,
            "column_count": 3,
            "columns": {
                "customer_id": {
                    "name": "customer_id",
                    "inferred_type": "identifier",
                    "confidence": 0.95,
                    "evidence": [],
                },
                "age": {
                    "name": "age",
                    "inferred_type": "numeric_continuous",
                    "confidence": 0.9,
                    "evidence": [],
                },
            },
            "target_column": None,
            "identifier_columns": ["customer_id"],
        }
        (findings_dir / "profiles_findings.yaml").write_text(yaml.dump(profiles_findings))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        source = config.sources[0]
        assert source.format == "csv"

    def test_parquet_source_format_inferred_from_path(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        multi_dataset = {
            "datasets": {
                "orders": {
                    "name": "orders",
                    "findings_path": str(findings_dir / "orders_findings.yaml"),
                    "source_path": "/data/raw/orders.parquet",
                    "granularity": "entity_level",
                    "row_count": 500,
                    "column_count": 3,
                    "excluded": False,
                }
            },
            "relationships": [],
            "primary_entity_dataset": "orders",
            "event_datasets": [],
            "excluded_datasets": [],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))

        orders_findings = {
            "source_path": "/data/raw/orders.parquet",
            "source_format": "parquet",
            "row_count": 500,
            "column_count": 3,
            "columns": {
                "order_id": {
                    "name": "order_id",
                    "inferred_type": "identifier",
                    "confidence": 0.95,
                    "evidence": [],
                },
            },
            "target_column": None,
            "identifier_columns": ["order_id"],
        }
        (findings_dir / "orders_findings.yaml").write_text(yaml.dump(orders_findings))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        source = config.sources[0]
        assert source.format == "parquet"

    def test_infer_format_parquet_extension(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        assert FindingsParser._infer_format("/data/file.parquet") == "parquet"
        assert FindingsParser._infer_format("/data/file.pq") == "parquet"

    def test_infer_format_csv_extension(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        assert FindingsParser._infer_format("/data/file.csv") == "csv"

    def test_infer_format_unknown_extension_defaults_to_delta(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        assert FindingsParser._infer_format("/data/file.json") == "delta"
        assert FindingsParser._infer_format("/data/delta_table") == "delta"


class TestDatetimeDerivationConfig:

    def _make_findings_dir(self, tmp_path, *, event_derivation_sources=None, entity_derivation_sources=None, entity_allow_future=None):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        entity_findings = {
            "source_path": "/data/customers.csv",
            "source_format": "csv",
            "row_count": 100,
            "column_count": 3,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
                "churn": {"name": "churn", "inferred_type": "binary",
                          "confidence": 0.99, "evidence": [], "quality_score": 100,
                          "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "target_column": "churn",
            "identifier_columns": ["customer_id"],
            "datetime_columns": ["signup_date"],
        }
        if entity_derivation_sources is not None:
            entity_findings["datetime_derivation_sources"] = entity_derivation_sources
        if entity_allow_future is not None:
            entity_findings["datetime_allow_future_columns"] = entity_allow_future
        (findings_dir / "customers_findings.yaml").write_text(yaml.dump(entity_findings))

        event_findings = {
            "source_path": "/data/events.csv",
            "source_format": "csv",
            "row_count": 500,
            "column_count": 4,
            "columns": {
                "event_id": {"name": "event_id", "inferred_type": "identifier",
                             "confidence": 0.95, "evidence": [], "quality_score": 100,
                             "cleaning_needed": False, "cleaning_recommendations": []},
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
                "event_date": {"name": "event_date", "inferred_type": "datetime",
                               "confidence": 0.95, "evidence": [], "quality_score": 100,
                               "cleaning_needed": False, "cleaning_recommendations": []},
                "response_at": {"name": "response_at", "inferred_type": "datetime",
                                "confidence": 0.9, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
                "amount": {"name": "amount", "inferred_type": "numeric_continuous",
                           "confidence": 0.9, "evidence": [], "quality_score": 90,
                           "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["event_id"],
            "datetime_columns": ["event_date", "response_at"],
            "time_series_metadata": {
                "granularity": "event_level",
                "entity_column": "customer_id",
                "time_column": "event_date",
                "aggregation_windows_used": ["7d", "30d", "90d"],
            },
        }
        if event_derivation_sources is not None:
            event_findings["datetime_derivation_sources"] = event_derivation_sources
        (findings_dir / "events_findings.yaml").write_text(yaml.dump(event_findings))

        multi_dataset = {
            "datasets": {
                "customers": {
                    "name": "customers",
                    "findings_path": str(findings_dir / "customers_findings.yaml"),
                    "source_path": "/data/customers.csv",
                    "granularity": "entity_level",
                    "row_count": 100,
                    "column_count": 3,
                },
                "events": {
                    "name": "events",
                    "findings_path": str(findings_dir / "events_findings.yaml"),
                    "source_path": "/data/events.csv",
                    "granularity": "event_level",
                    "row_count": 500,
                    "column_count": 4,
                    "entity_column": "customer_id",
                    "time_column": "event_date",
                },
            },
            "relationships": [{
                "left_dataset": "customers",
                "right_dataset": "events",
                "left_column": "customer_id",
                "right_column": "customer_id",
                "relationship_type": "one_to_many",
                "confidence": 1.0,
            }],
            "primary_entity_dataset": "customers",
            "event_datasets": ["events"],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))
        return findings_dir

    def test_builds_datetime_derivation_config_for_event_source(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = self._make_findings_dir(tmp_path, event_derivation_sources=["response_at"])
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "events" in config.bronze_event
        deriv = config.bronze_event["events"].datetime_derivation
        assert deriv is not None
        assert deriv.source_columns == ["response_at"]
        assert deriv.reference_column == "event_date"
        assert deriv.mask_future_columns == []

    def test_builds_datetime_derivation_config_for_landing(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = self._make_findings_dir(tmp_path, event_derivation_sources=["response_at"])
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "events" in config.landing
        landing = config.landing["events"]
        assert landing.datetime_derivation is not None
        assert landing.datetime_derivation.source_columns == ["response_at"]
        assert landing.datetime_derivation.reference_column == "feature_timestamp"
        assert landing.datetime_derivation.mask_future_columns == ["response_at"]

    def test_no_config_when_no_derivation_sources(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = self._make_findings_dir(tmp_path)
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "events" in config.bronze_event
        assert config.bronze_event["events"].datetime_derivation is None
        if "events" in config.landing:
            assert config.landing["events"].datetime_derivation is None

    def test_derived_columns_included_in_aggregation_value_columns(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = self._make_findings_dir(tmp_path, event_derivation_sources=["response_at"])
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        agg = config.bronze_event["events"].aggregation
        assert agg is not None
        suffixes = ["_delta_hours", "_hour", "_dow", "_is_weekend"]
        for suffix in suffixes:
            assert f"response_at{suffix}" in agg.value_columns

    def test_allow_future_columns_excludes_from_mask(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = self._make_findings_dir(tmp_path)
        event_yaml_path = findings_dir / "events_findings.yaml"
        data = yaml.safe_load(event_yaml_path.read_text())
        data["datetime_derivation_sources"] = ["response_at", "scheduled_at"]
        data["datetime_allow_future_columns"] = ["scheduled_at"]
        event_yaml_path.write_text(yaml.dump(data))

        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "events" in config.landing
        deriv = config.landing["events"].datetime_derivation
        assert deriv is not None
        assert deriv.source_columns == ["response_at", "scheduled_at"]
        assert deriv.mask_future_columns == ["response_at"]


class TestBuildHistoryWindowConfig:
    def test_returns_none_without_intent(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser.__new__(FindingsParser)
        parser._intent = None
        assert parser._build_history_window_config("event_date") is None

    def test_returns_none_when_no_window_params(self):
        from customer_retention.analysis.auto_explorer.project_context import IntentConfig
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser.__new__(FindingsParser)
        parser._intent = IntentConfig()
        assert parser._build_history_window_config("event_date") is None

    def test_returns_config_with_upper_limit(self):
        from customer_retention.analysis.auto_explorer.project_context import IntentConfig
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser.__new__(FindingsParser)
        parser._intent = IntentConfig(history_upper_limit="2024-06-30")
        result = parser._build_history_window_config("event_date")
        assert result is not None
        assert result.upper_limit == "2024-06-30"
        assert result.lookback_periods is None
        assert result.time_column == "event_date"

    def test_returns_config_with_lookback(self):
        from customer_retention.analysis.auto_explorer.project_context import CadenceInterval, IntentConfig
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser.__new__(FindingsParser)
        parser._intent = IntentConfig(lookback_periods=52, cadence_interval=CadenceInterval.WEEKLY)
        result = parser._build_history_window_config("ts")
        assert result is not None
        assert result.lookback_periods == 52
        assert result.cadence_days == 7
        assert result.upper_limit is None

    def test_returns_config_with_both(self):
        from customer_retention.analysis.auto_explorer.project_context import CadenceInterval, IntentConfig
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser.__new__(FindingsParser)
        parser._intent = IntentConfig(
            history_upper_limit="2024-12-31",
            lookback_periods=12,
            cadence_interval=CadenceInterval.MONTHLY,
        )
        result = parser._build_history_window_config("order_date")
        assert result.upper_limit == "2024-12-31"
        assert result.lookback_periods == 12
        assert result.cadence_days == 30


class TestImbalanceStrategyFromRegistry:
    def _make_findings_and_recs(self, tmp_path, strategy="class_weight", imbalance_ratio=5.0):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        customers_findings = {
            "source_path": "/data/customers.csv",
            "source_format": "csv",
            "row_count": 100,
            "column_count": 3,
            "columns": {
                "customer_id": {
                    "name": "customer_id", "inferred_type": "identifier",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "revenue": {
                    "name": "revenue", "inferred_type": "numeric_continuous",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "target": {
                    "name": "target", "inferred_type": "binary",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
            },
            "target_column": "target",
            "identifier_columns": ["customer_id"],
        }
        (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers_findings))
        multi_dataset = {
            "datasets": {
                "customers": {
                    "name": "customers",
                    "findings_path": "customers_findings.yaml",
                    "source_path": "/data/customers.csv",
                    "granularity": "entity_level",
                    "row_count": 100,
                    "column_count": 3,
                },
            },
            "relationships": [],
            "primary_entity_dataset": "customers",
            "event_datasets": [],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))
        recs = {
            "sources": {
                "customers": {
                    "source_file": "/data/customers.csv",
                    "null_handling": [],
                    "outlier_handling": [],
                    "type_conversions": [],
                    "deduplication": [],
                    "filtering": [],
                    "text_processing": [],
                    "modeling_strategy": [
                        {
                            "id": "rec_001",
                            "layer": "bronze",
                            "category": "imbalance",
                            "action": strategy,
                            "target_column": "target",
                            "parameters": {
                                "imbalance_ratio": imbalance_ratio,
                                "minority_class": 1,
                            },
                            "rationale": "Handle class imbalance",
                            "source_notebook": "02_source_integrity",
                            "priority": 1,
                        },
                    ],
                },
            },
        }
        (findings_dir / "recommendations.yaml").write_text(yaml.dump(recs))
        return findings_dir

    def test_reads_class_weight_from_registry(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = self._make_findings_and_recs(tmp_path, strategy="class_weight", imbalance_ratio=5.0)
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert config.training is not None
        assert config.training.imbalance_strategy == "class_weight"
        assert config.training.imbalance_ratio == 5.0

    def test_reads_smote_from_registry(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = self._make_findings_and_recs(tmp_path, strategy="smote", imbalance_ratio=3.0)
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert config.training is not None
        assert config.training.imbalance_strategy == "smote"
        assert config.training.imbalance_ratio == 3.0

    def test_defaults_to_class_weight_without_registry(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        customers_findings = {
            "source_path": "/data/customers.csv",
            "source_format": "csv",
            "row_count": 100,
            "column_count": 3,
            "columns": {
                "customer_id": {
                    "name": "customer_id", "inferred_type": "identifier",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "target": {
                    "name": "target", "inferred_type": "binary",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
            },
            "target_column": "target",
            "identifier_columns": ["customer_id"],
        }
        (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers_findings))
        multi_dataset = {
            "datasets": {
                "customers": {
                    "name": "customers",
                    "findings_path": "customers_findings.yaml",
                    "source_path": "/data/customers.csv",
                    "granularity": "entity_level",
                    "row_count": 100,
                    "column_count": 3,
                },
            },
            "relationships": [],
            "primary_entity_dataset": "customers",
            "event_datasets": [],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        if config.training is not None:
            assert config.training.imbalance_strategy == "class_weight"


class TestDeduplicationFromRegistry:
    def _make_event_findings_with_recs(self, tmp_path, dedup_strategy="keep_first", conflict_columns=None):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        events_findings = {
            "source_path": "/data/events.csv",
            "source_format": "csv",
            "row_count": 500,
            "column_count": 5,
            "columns": {
                "customer_id": {
                    "name": "customer_id", "inferred_type": "identifier",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "event_date": {
                    "name": "event_date", "inferred_type": "datetime",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "amount": {
                    "name": "amount", "inferred_type": "numeric_continuous",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "target": {
                    "name": "target", "inferred_type": "binary",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
            },
            "target_column": "target",
            "identifier_columns": ["customer_id"],
            "time_series_metadata": {
                "time_column": "event_date",
                "entity_column": "customer_id",
                "aggregation_windows_used": ["7d", "30d", "90d"],
            },
        }
        (findings_dir / "events_findings.yaml").write_text(yaml.dump(events_findings))
        multi_dataset = {
            "datasets": {
                "events": {
                    "name": "events",
                    "findings_path": "events_findings.yaml",
                    "source_path": "/data/events.csv",
                    "granularity": "event_level",
                    "row_count": 500,
                    "column_count": 5,
                    "entity_column": "customer_id",
                    "time_column": "event_date",
                },
            },
            "relationships": [],
            "primary_entity_dataset": "events",
            "event_datasets": ["events"],
            "aggregation_windows": ["7d", "30d"],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))
        dedup_params = {"strategy": dedup_strategy}
        if conflict_columns:
            dedup_params["conflict_columns"] = conflict_columns
        recs = {
            "sources": {
                "events": {
                    "source_file": "/data/events.csv",
                    "null_handling": [],
                    "outlier_handling": [],
                    "type_conversions": [],
                    "deduplication": [
                        {
                            "id": "rec_dedup_001",
                            "layer": "bronze",
                            "category": "deduplication",
                            "action": dedup_strategy,
                            "target_column": "customer_id",
                            "parameters": dedup_params,
                            "rationale": f"Deduplicate events using {dedup_strategy}",
                            "source_notebook": "02_source_integrity",
                            "priority": 1,
                        },
                    ],
                    "filtering": [],
                    "text_processing": [],
                    "modeling_strategy": [],
                },
            },
        }
        (findings_dir / "recommendations.yaml").write_text(yaml.dump(recs))
        return findings_dir

    def test_reads_keep_first_from_registry(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import DeduplicationConfig

        findings_dir = self._make_event_findings_with_recs(tmp_path, "keep_first")
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "events" in config.bronze_event
        dedup = config.bronze_event["events"].deduplicate
        assert isinstance(dedup, DeduplicationConfig)
        assert dedup.strategy == "keep_first"

    def test_reads_keep_most_complete_from_registry(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import DeduplicationConfig

        findings_dir = self._make_event_findings_with_recs(tmp_path, "keep_most_complete")
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        dedup = config.bronze_event["events"].deduplicate
        assert isinstance(dedup, DeduplicationConfig)
        assert dedup.strategy == "keep_most_complete"

    def test_reads_conflict_columns_from_registry(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import DeduplicationConfig

        findings_dir = self._make_event_findings_with_recs(
            tmp_path, "keep_first", conflict_columns=["customer_id", "event_date", "amount"],
        )
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        dedup = config.bronze_event["events"].deduplicate
        assert isinstance(dedup, DeduplicationConfig)
        assert dedup.conflict_columns == ["customer_id", "event_date", "amount"]

    def test_defaults_to_basic_dedup_without_registry(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        events_findings = {
            "source_path": "/data/events.csv",
            "source_format": "csv",
            "row_count": 500,
            "column_count": 3,
            "columns": {
                "customer_id": {
                    "name": "customer_id", "inferred_type": "identifier",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "event_date": {
                    "name": "event_date", "inferred_type": "datetime",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "target": {
                    "name": "target", "inferred_type": "binary",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
            },
            "target_column": "target",
            "identifier_columns": ["customer_id"],
            "time_series_metadata": {
                "time_column": "event_date",
                "entity_column": "customer_id",
                "aggregation_windows_used": ["7d", "30d", "90d"],
            },
        }
        (findings_dir / "events_findings.yaml").write_text(yaml.dump(events_findings))
        multi_dataset = {
            "datasets": {
                "events": {
                    "name": "events",
                    "findings_path": "events_findings.yaml",
                    "source_path": "/data/events.csv",
                    "granularity": "event_level",
                    "row_count": 500,
                    "column_count": 3,
                    "entity_column": "customer_id",
                    "time_column": "event_date",
                },
            },
            "relationships": [],
            "primary_entity_dataset": "events",
            "event_datasets": ["events"],
            "aggregation_windows": ["7d", "30d"],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert config.bronze_event["events"].deduplicate is True


class TestFilteringFromRegistry:
    def _make_event_findings_with_filters(self, tmp_path, filters):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        events_findings = {
            "source_path": "/data/events.csv",
            "source_format": "csv",
            "row_count": 500,
            "column_count": 5,
            "columns": {
                "customer_id": {
                    "name": "customer_id", "inferred_type": "identifier",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "event_date": {
                    "name": "event_date", "inferred_type": "datetime",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "amount": {
                    "name": "amount", "inferred_type": "numeric_continuous",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "target": {
                    "name": "target", "inferred_type": "binary",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
            },
            "target_column": "target",
            "identifier_columns": ["customer_id"],
            "time_series_metadata": {
                "time_column": "event_date",
                "entity_column": "customer_id",
                "aggregation_windows_used": ["7d", "30d", "90d"],
            },
        }
        (findings_dir / "events_findings.yaml").write_text(yaml.dump(events_findings))
        multi_dataset = {
            "datasets": {
                "events": {
                    "name": "events",
                    "findings_path": "events_findings.yaml",
                    "source_path": "/data/events.csv",
                    "granularity": "event_level",
                    "row_count": 500,
                    "column_count": 5,
                    "entity_column": "customer_id",
                    "time_column": "event_date",
                },
            },
            "relationships": [],
            "primary_entity_dataset": "events",
            "event_datasets": ["events"],
            "aggregation_windows": ["7d", "30d"],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))
        recs = {
            "sources": {
                "events": {
                    "source_file": "/data/events.csv",
                    "null_handling": [],
                    "outlier_handling": [],
                    "type_conversions": [],
                    "deduplication": [],
                    "filtering": filters,
                    "text_processing": [],
                    "modeling_strategy": [],
                },
            },
        }
        (findings_dir / "recommendations.yaml").write_text(yaml.dump(recs))
        return findings_dir

    def test_reads_non_negative_filter(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType

        filters = [{
            "id": "rec_filter_001",
            "layer": "bronze",
            "category": "filtering",
            "action": "non_negative",
            "target_column": "amount",
            "parameters": {"condition": "non_negative"},
            "rationale": "Filter negative amounts",
            "source_notebook": "04_column_deep_dive",
            "priority": 1,
        }]
        findings_dir = self._make_event_findings_with_filters(tmp_path, filters)
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        assert "events" in config.bronze_event
        filter_steps = [s for s in config.bronze_event["events"].pre_shaping
                        if s.type == PipelineTransformationType.FILTER]
        assert len(filter_steps) == 1
        assert filter_steps[0].column == "amount"
        assert filter_steps[0].parameters["condition"] == "non_negative"

    def test_reads_range_filter(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType

        filters = [{
            "id": "rec_filter_002",
            "layer": "bronze",
            "category": "filtering",
            "action": "range",
            "target_column": "amount",
            "parameters": {"condition": "range", "min_value": 0, "max_value": 10000},
            "rationale": "Cap amount range",
            "source_notebook": "04_column_deep_dive",
            "priority": 1,
        }]
        findings_dir = self._make_event_findings_with_filters(tmp_path, filters)
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        filter_steps = [s for s in config.bronze_event["events"].pre_shaping
                        if s.type == PipelineTransformationType.FILTER]
        assert len(filter_steps) == 1
        assert filter_steps[0].parameters["min_value"] == 0
        assert filter_steps[0].parameters["max_value"] == 10000

    def test_no_filter_steps_without_registry(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        events_findings = {
            "source_path": "/data/events.csv",
            "source_format": "csv",
            "row_count": 500,
            "column_count": 3,
            "columns": {
                "customer_id": {
                    "name": "customer_id", "inferred_type": "identifier",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "event_date": {
                    "name": "event_date", "inferred_type": "datetime",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
                "target": {
                    "name": "target", "inferred_type": "binary",
                    "confidence": 0.9, "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": [],
                },
            },
            "target_column": "target",
            "identifier_columns": ["customer_id"],
            "time_series_metadata": {
                "time_column": "event_date",
                "entity_column": "customer_id",
                "aggregation_windows_used": ["7d", "30d", "90d"],
            },
        }
        (findings_dir / "events_findings.yaml").write_text(yaml.dump(events_findings))
        multi_dataset = {
            "datasets": {
                "events": {
                    "name": "events",
                    "findings_path": "events_findings.yaml",
                    "source_path": "/data/events.csv",
                    "granularity": "event_level",
                    "row_count": 500,
                    "column_count": 3,
                    "entity_column": "customer_id",
                    "time_column": "event_date",
                },
            },
            "relationships": [],
            "primary_entity_dataset": "events",
            "event_datasets": ["events"],
            "aggregation_windows": ["7d", "30d"],
        }
        (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi_dataset))
        parser = FindingsParser(str(findings_dir))
        config = parser.parse()

        filter_steps = [s for s in config.bronze_event["events"].pre_shaping
                        if s.type == PipelineTransformationType.FILTER]
        assert len(filter_steps) == 0


class TestTemporalMergeMetadata:
    def _make_namespace_with_grid(self, tmp_path, grid_dates, entity_column="customer_id"):
        from customer_retention.analysis.auto_explorer.project_context import (
            ObjectivePriority,
            ObjectiveSpec,
            PredictionObjective,
            ProjectContext,
        )
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
        from customer_retention.analysis.auto_explorer.snapshot_grid import SnapshotGrid

        ns = RunNamespace.create(root=tmp_path, project_name="test")
        grid = SnapshotGrid(
            cadence_interval="weekly",
            observation_window_days=90,
            grid_dates=grid_dates,
        )
        grid.save(ns.snapshot_grid_path)
        ctx = ProjectContext(
            entity_column=entity_column,
            objectives=[ObjectiveSpec(
                objective=PredictionObjective.IMMEDIATE_RISK,
                priority=ObjectivePriority.PRIMARY,
            )],
        )
        ctx.save(ns.project_context_path)
        return ns

    def _write_findings(self, findings_dir, ns):
        multi_dataset = {
            "datasets": {
                "customers": {
                    "name": "customers",
                    "findings_path": str(findings_dir / "customers_findings.yaml"),
                    "source_path": "/data/customers.csv",
                    "granularity": "entity_level",
                    "row_count": 1000,
                    "column_count": 3,
                    "entity_column": "customer_id",
                },
                "orders": {
                    "name": "orders",
                    "findings_path": str(findings_dir / "orders_findings.yaml"),
                    "source_path": "/data/orders.parquet",
                    "granularity": "event_level",
                    "row_count": 5000,
                    "column_count": 4,
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
                }
            ],
            "primary_entity_dataset": "customers",
            "event_datasets": ["orders"],
        }
        ns.multi_dataset_findings_path.parent.mkdir(parents=True, exist_ok=True)
        ns.multi_dataset_findings_path.write_text(yaml.dump(multi_dataset))

        customers_findings = {
            "source_path": "/data/customers.csv",
            "source_format": "csv",
            "row_count": 1000,
            "column_count": 3,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
                "age": {"name": "age", "inferred_type": "numeric_continuous",
                        "confidence": 0.9, "evidence": [], "quality_score": 100,
                        "cleaning_needed": False, "cleaning_recommendations": []},
                "churn": {"name": "churn", "inferred_type": "binary",
                          "confidence": 0.99, "evidence": [], "quality_score": 100,
                          "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "target_column": "churn",
            "identifier_columns": ["customer_id"],
        }
        findings_dir.mkdir(parents=True, exist_ok=True)
        (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers_findings))

        orders_findings = {
            "source_path": "/data/orders.parquet",
            "source_format": "parquet",
            "row_count": 5000,
            "column_count": 4,
            "columns": {
                "order_id": {"name": "order_id", "inferred_type": "identifier",
                             "confidence": 0.95, "evidence": [], "quality_score": 100,
                             "cleaning_needed": False, "cleaning_recommendations": []},
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
                "amount": {"name": "amount", "inferred_type": "numeric_continuous",
                           "confidence": 0.9, "evidence": [], "quality_score": 90,
                           "cleaning_needed": False, "cleaning_recommendations": []},
                "order_date": {"name": "order_date", "inferred_type": "datetime",
                               "confidence": 0.95, "evidence": [], "quality_score": 100,
                               "cleaning_needed": False, "cleaning_recommendations": []},
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

    def test_grid_dates_populated_from_namespace(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        grid_dates = ["2024-01-01", "2024-01-08", "2024-01-15"]
        ns = self._make_namespace_with_grid(tmp_path, grid_dates)
        findings_dir = ns.multi_dataset_findings_path.parent
        self._write_findings(findings_dir, ns)

        parser = FindingsParser(str(findings_dir), namespace=ns)
        config = parser.parse()

        assert config.silver.grid_dates == grid_dates

    def test_entity_key_from_project_context(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = self._make_namespace_with_grid(tmp_path, ["2024-01-01"], entity_column="user_id")
        findings_dir = ns.multi_dataset_findings_path.parent
        self._write_findings(findings_dir, ns)

        parser = FindingsParser(str(findings_dir), namespace=ns)
        config = parser.parse()

        assert config.silver.entity_key == "user_id"

    def test_merge_sources_populated(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = self._make_namespace_with_grid(tmp_path, ["2024-01-01"])
        findings_dir = ns.multi_dataset_findings_path.parent
        self._write_findings(findings_dir, ns)

        parser = FindingsParser(str(findings_dir), namespace=ns)
        config = parser.parse()

        source_names = {s.name for s in config.silver.merge_sources}
        assert "customers" in source_names
        assert "orders" in source_names

    def test_event_source_has_timestamp_column(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = self._make_namespace_with_grid(tmp_path, ["2024-01-01"])
        findings_dir = ns.multi_dataset_findings_path.parent
        self._write_findings(findings_dir, ns)

        parser = FindingsParser(str(findings_dir), namespace=ns)
        config = parser.parse()

        orders_src = next(s for s in config.silver.merge_sources if s.name == "orders")
        assert orders_src.feature_timestamp_column == "order_date"
        assert orders_src.granularity == "event_level"

    def test_entity_source_no_timestamp(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = self._make_namespace_with_grid(tmp_path, ["2024-01-01"])
        findings_dir = ns.multi_dataset_findings_path.parent
        self._write_findings(findings_dir, ns)

        parser = FindingsParser(str(findings_dir), namespace=ns)
        config = parser.parse()

        cust_src = next(s for s in config.silver.merge_sources if s.name == "customers")
        assert cust_src.feature_timestamp_column is None
        assert cust_src.granularity == "entity_level"

    def test_no_namespace_leaves_grid_empty(self, sample_findings_dir):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser(str(sample_findings_dir))
        config = parser.parse()

        assert config.silver.grid_dates == []
        assert config.silver.merge_sources == []

    def test_no_snapshot_grid_file_leaves_grid_empty(self, tmp_path):
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = RunNamespace.create(root=tmp_path, project_name="test")
        findings_dir = ns.multi_dataset_findings_path.parent
        self._write_findings(findings_dir, ns)

        parser = FindingsParser(str(findings_dir), namespace=ns)
        config = parser.parse()

        assert config.silver.grid_dates == []

    def test_excluded_datasets_skipped(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = self._make_namespace_with_grid(tmp_path, ["2024-01-01"])
        findings_dir = ns.multi_dataset_findings_path.parent
        multi_dataset = {
            "datasets": {
                "customers": {
                    "name": "customers",
                    "findings_path": str(findings_dir / "customers_findings.yaml"),
                    "source_path": "/data/customers.csv",
                    "granularity": "entity_level",
                    "row_count": 1000,
                    "column_count": 3,
                    "entity_column": "customer_id",
                    "excluded": True,
                },
            },
            "relationships": [],
            "primary_entity_dataset": "customers",
            "event_datasets": [],
            "excluded_datasets": ["customers"],
        }
        findings_dir.mkdir(parents=True, exist_ok=True)
        ns.multi_dataset_findings_path.write_text(yaml.dump(multi_dataset))
        customers_findings = {
            "source_path": "/data/customers.csv",
            "source_format": "csv",
            "row_count": 1000,
            "column_count": 3,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
                "churn": {"name": "churn", "inferred_type": "binary",
                          "confidence": 0.99, "evidence": [], "quality_score": 100,
                          "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "target_column": "churn",
            "identifier_columns": ["customer_id"],
        }
        (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers_findings))

        parser = FindingsParser(str(findings_dir), namespace=ns)
        config = parser.parse()

        excluded_names = {s.name for s in config.silver.merge_sources}
        assert "customers" not in excluded_names

    def _make_namespace_with_key_resolution(self, tmp_path, grid_dates, key_resolution_entries):
        from customer_retention.analysis.auto_explorer.project_context import (
            DatasetRegistryEntry,
            KeyResolutionStep,
            ObjectivePriority,
            ObjectiveSpec,
            PredictionObjective,
            ProjectContext,
        )
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
        from customer_retention.analysis.auto_explorer.snapshot_grid import SnapshotGrid

        ns = RunNamespace.create(root=tmp_path, project_name="test")
        grid = SnapshotGrid(
            cadence_interval="weekly",
            observation_window_days=90,
            grid_dates=grid_dates,
        )
        grid.save(ns.snapshot_grid_path)

        datasets = {}
        for name, kr_list in key_resolution_entries.items():
            steps = [KeyResolutionStep(**kr) for kr in kr_list] if kr_list else []
            datasets[name] = DatasetRegistryEntry(
                name=name,
                path=f"/data/{name}.csv",
                entity_column="ACCOUNT_ID",
                key_resolution=steps,
            )
        datasets.setdefault("customers", DatasetRegistryEntry(
            name="customers",
            path="/data/customers.csv",
            entity_column="ACCOUNT_ID",
        ))

        ctx = ProjectContext(
            entity_column="ACCOUNT_ID",
            datasets=datasets,
            objectives=[ObjectiveSpec(
                objective=PredictionObjective.IMMEDIATE_RISK,
                priority=ObjectivePriority.PRIMARY,
            )],
        )
        ctx.save(ns.project_context_path)
        return ns

    def _write_findings_with_case_history(self, findings_dir, ns):
        multi_dataset = {
            "datasets": {
                "customers": {
                    "name": "customers",
                    "findings_path": str(findings_dir / "customers_findings.yaml"),
                    "source_path": "/data/customers.csv",
                    "granularity": "entity_level",
                    "row_count": 1000,
                    "column_count": 3,
                    "entity_column": "ACCOUNT_ID",
                },
                "case": {
                    "name": "case",
                    "findings_path": str(findings_dir / "case_findings.yaml"),
                    "source_path": "/data/case.csv",
                    "granularity": "entity_level",
                    "row_count": 500,
                    "column_count": 3,
                    "entity_column": "ACCOUNT_ID",
                },
                "case_history": {
                    "name": "case_history",
                    "findings_path": str(findings_dir / "case_history_findings.yaml"),
                    "source_path": "/data/case_history.csv",
                    "granularity": "event_level",
                    "row_count": 5000,
                    "column_count": 4,
                    "entity_column": "ACCOUNT_ID",
                    "time_column": "created_date",
                },
            },
            "relationships": [
                {
                    "left_dataset": "customers",
                    "right_dataset": "case",
                    "left_column": "ACCOUNT_ID",
                    "right_column": "ACCOUNT_ID",
                    "relationship_type": "one_to_many",
                    "confidence": 1.0,
                },
                {
                    "left_dataset": "case",
                    "right_dataset": "case_history",
                    "left_column": "CASE_ID",
                    "right_column": "CASE_ID",
                    "relationship_type": "one_to_many",
                    "confidence": 1.0,
                },
            ],
            "primary_entity_dataset": "customers",
            "event_datasets": ["case_history"],
        }
        ns.multi_dataset_findings_path.parent.mkdir(parents=True, exist_ok=True)
        ns.multi_dataset_findings_path.write_text(yaml.dump(multi_dataset))

        base_findings = {
            "row_count": 1000,
            "column_count": 3,
            "columns": {
                "ACCOUNT_ID": {"name": "ACCOUNT_ID", "inferred_type": "identifier",
                               "confidence": 0.95, "evidence": [], "quality_score": 100,
                               "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["ACCOUNT_ID"],
        }

        customers = {**base_findings, "source_path": "/data/customers.csv", "source_format": "csv",
                     "target_column": "churn",
                     "columns": {**base_findings["columns"],
                                 "churn": {"name": "churn", "inferred_type": "binary",
                                           "confidence": 0.99, "evidence": [], "quality_score": 100,
                                           "cleaning_needed": False, "cleaning_recommendations": []}}}
        case = {**base_findings, "source_path": "/data/case.csv", "source_format": "csv",
                "row_count": 500,
                "columns": {**base_findings["columns"],
                            "CASE_ID": {"name": "CASE_ID", "inferred_type": "identifier",
                                        "confidence": 0.95, "evidence": [], "quality_score": 100,
                                        "cleaning_needed": False, "cleaning_recommendations": []}},
                "identifier_columns": ["ACCOUNT_ID", "CASE_ID"]}
        case_history = {"source_path": "/data/case_history.csv", "source_format": "csv",
                        "row_count": 5000, "column_count": 4,
                        "columns": {
                            "CASE_ID": {"name": "CASE_ID", "inferred_type": "identifier",
                                        "confidence": 0.95, "evidence": [], "quality_score": 100,
                                        "cleaning_needed": False, "cleaning_recommendations": []},
                            "created_date": {"name": "created_date", "inferred_type": "datetime",
                                             "confidence": 0.95, "evidence": [], "quality_score": 100,
                                             "cleaning_needed": False, "cleaning_recommendations": []},
                        },
                        "identifier_columns": ["CASE_ID"],
                        "datetime_columns": ["created_date"],
                        "time_series_metadata": {
                            "granularity": "event_level",
                            "entity_column": "CASE_ID",
                            "time_column": "created_date",
                            "aggregation_windows_used": ["7d", "30d", "90d"],
                        }}
        findings_dir.mkdir(parents=True, exist_ok=True)
        (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers))
        (findings_dir / "case_findings.yaml").write_text(yaml.dump(case))
        (findings_dir / "case_history_findings.yaml").write_text(yaml.dump(case_history))

    def test_key_resolution_steps_from_context(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = self._make_namespace_with_key_resolution(
            tmp_path,
            ["2024-01-01"],
            {"case_history": [{"bridge_dataset": "case", "source_key": "CASE_ID",
                               "bridge_key": "CASE_ID", "resolve_column": "ACCOUNT_ID"}]},
        )
        findings_dir = ns.multi_dataset_findings_path.parent
        self._write_findings_with_case_history(findings_dir, ns)

        parser = FindingsParser(str(findings_dir), namespace=ns)
        config = parser.parse()

        ch_src = next(s for s in config.silver.merge_sources if s.name == "case_history")
        assert len(ch_src.key_resolution_steps) == 1
        assert ch_src.key_resolution_steps[0].bridge_dataset == "case"
        assert ch_src.key_resolution_steps[0].source_key == "CASE_ID"
        assert ch_src.key_resolution_steps[0].resolve_column == "ACCOUNT_ID"

    def test_no_key_resolution_leaves_empty(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = self._make_namespace_with_key_resolution(tmp_path, ["2024-01-01"], {})
        findings_dir = ns.multi_dataset_findings_path.parent
        self._write_findings_with_case_history(findings_dir, ns)

        parser = FindingsParser(str(findings_dir), namespace=ns)
        config = parser.parse()

        for src in config.silver.merge_sources:
            assert src.key_resolution_steps == []

    def test_key_resolution_multi_step(self, tmp_path):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = self._make_namespace_with_key_resolution(
            tmp_path,
            ["2024-01-01"],
            {"case_history": [
                {"bridge_dataset": "case", "source_key": "CASE_ID",
                 "bridge_key": "CASE_ID", "resolve_column": "ACCOUNT_ID"},
                {"bridge_dataset": "customers", "source_key": "ACCOUNT_ID",
                 "bridge_key": "ACCOUNT_ID", "resolve_column": "ACCOUNT_ID"},
            ]},
        )
        findings_dir = ns.multi_dataset_findings_path.parent
        self._write_findings_with_case_history(findings_dir, ns)

        parser = FindingsParser(str(findings_dir), namespace=ns)
        config = parser.parse()

        ch_src = next(s for s in config.silver.merge_sources if s.name == "case_history")
        assert len(ch_src.key_resolution_steps) == 2
        assert ch_src.key_resolution_steps[0].bridge_dataset == "case"
        assert ch_src.key_resolution_steps[1].bridge_dataset == "customers"

    def _make_namespace_with_entity_time_role(self, tmp_path, grid_dates, datasets_config):
        from customer_retention.analysis.auto_explorer.project_context import (
            DatasetRegistryEntry,
            ObjectivePriority,
            ObjectiveSpec,
            PredictionObjective,
            ProjectContext,
            RawTimeColumnRole,
        )
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
        from customer_retention.analysis.auto_explorer.snapshot_grid import SnapshotGrid
        from customer_retention.core.config.column_config import DatasetGranularity

        ns = RunNamespace.create(root=tmp_path, project_name="test")
        grid = SnapshotGrid(cadence_interval="weekly", observation_window_days=90, grid_dates=grid_dates)
        grid.save(ns.snapshot_grid_path)

        datasets = {}
        for name, cfg in datasets_config.items():
            datasets[name] = DatasetRegistryEntry(
                name=name,
                path=f"/data/{name}.csv",
                entity_column="customer_id",
                time_column=cfg.get("time_column"),
                raw_time_column_role=(
                    RawTimeColumnRole(cfg["raw_time_column_role"])
                    if cfg.get("raw_time_column_role") else None
                ),
                granularity=(
                    DatasetGranularity(cfg["granularity"])
                    if cfg.get("granularity") else None
                ),
            )

        ctx = ProjectContext(
            entity_column="customer_id",
            datasets=datasets,
            objectives=[ObjectiveSpec(
                objective=PredictionObjective.IMMEDIATE_RISK,
                priority=ObjectivePriority.PRIMARY,
            )],
        )
        ctx.save(ns.project_context_path)
        return ns

    def test_entity_update_time_uses_broadcast(self, tmp_path):
        """Entity-level dataset with ENTITY_UPDATE_TIME should NOT get feature_timestamp_column."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = self._make_namespace_with_entity_time_role(tmp_path, ["2024-01-01"], {
            "account": {
                "time_column": "LAST_MODIFIED_DATE",
                "raw_time_column_role": "entity_update_time",
                "granularity": "entity_level",
            },
            "orders": {
                "time_column": "order_date",
                "raw_time_column_role": "event_time",
                "granularity": "event_level",
            },
        })
        findings_dir = ns.multi_dataset_findings_path.parent
        self._write_entity_time_role_findings(findings_dir, ns)

        parser = FindingsParser(str(findings_dir), namespace=ns)
        config = parser.parse()

        acct_src = next(s for s in config.silver.merge_sources if s.name == "account")
        assert acct_src.feature_timestamp_column is None
        assert acct_src.granularity == "entity_level"

    def test_entity_non_update_time_uses_asof(self, tmp_path):
        """Entity-level dataset with EVENT_TIME should get feature_timestamp_column for as-of join."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = self._make_namespace_with_entity_time_role(tmp_path, ["2024-01-01"], {
            "segment_history": {
                "time_column": "effective_date",
                "raw_time_column_role": "event_time",
                "granularity": "entity_level",
            },
            "orders": {
                "time_column": "order_date",
                "raw_time_column_role": "event_time",
                "granularity": "event_level",
            },
        })
        findings_dir = ns.multi_dataset_findings_path.parent
        self._write_entity_time_role_findings(findings_dir, ns, entity_name="segment_history")

        parser = FindingsParser(str(findings_dir), namespace=ns)
        config = parser.parse()

        seg_src = next(s for s in config.silver.merge_sources if s.name == "segment_history")
        assert seg_src.feature_timestamp_column == "effective_date"
        assert seg_src.granularity == "entity_level"

    def test_entity_no_time_column_uses_broadcast(self, tmp_path):
        """Entity-level dataset without time_column should always broadcast."""
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = self._make_namespace_with_entity_time_role(tmp_path, ["2024-01-01"], {
            "account": {"granularity": "entity_level"},
            "orders": {
                "time_column": "order_date",
                "raw_time_column_role": "event_time",
                "granularity": "event_level",
            },
        })
        findings_dir = ns.multi_dataset_findings_path.parent
        self._write_entity_time_role_findings(findings_dir, ns)

        parser = FindingsParser(str(findings_dir), namespace=ns)
        config = parser.parse()

        acct_src = next(s for s in config.silver.merge_sources if s.name == "account")
        assert acct_src.feature_timestamp_column is None

    def _write_entity_time_role_findings(self, findings_dir, ns, entity_name="account"):
        multi_dataset = {
            "datasets": {
                entity_name: {
                    "name": entity_name,
                    "findings_path": str(findings_dir / f"{entity_name}_findings.yaml"),
                    "source_path": f"/data/{entity_name}.csv",
                    "granularity": "entity_level",
                    "row_count": 1000,
                    "column_count": 3,
                    "entity_column": "customer_id",
                },
                "orders": {
                    "name": "orders",
                    "findings_path": str(findings_dir / "orders_findings.yaml"),
                    "source_path": "/data/orders.parquet",
                    "granularity": "event_level",
                    "row_count": 5000,
                    "column_count": 4,
                    "entity_column": "customer_id",
                    "time_column": "order_date",
                },
            },
            "relationships": [{
                "left_dataset": entity_name, "right_dataset": "orders",
                "left_column": "customer_id", "right_column": "customer_id",
                "relationship_type": "one_to_many", "confidence": 1.0,
            }],
            "primary_entity_dataset": entity_name,
            "event_datasets": ["orders"],
        }
        findings_dir.mkdir(parents=True, exist_ok=True)
        ns.multi_dataset_findings_path.parent.mkdir(parents=True, exist_ok=True)
        ns.multi_dataset_findings_path.write_text(yaml.dump(multi_dataset))

        entity_findings = {
            "source_path": f"/data/{entity_name}.csv", "source_format": "csv",
            "row_count": 1000, "column_count": 3,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
                "churn": {"name": "churn", "inferred_type": "binary",
                          "confidence": 0.99, "evidence": [], "quality_score": 100,
                          "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "target_column": "churn",
            "identifier_columns": ["customer_id"],
        }
        (findings_dir / f"{entity_name}_findings.yaml").write_text(yaml.dump(entity_findings))

        orders_findings = {
            "source_path": "/data/orders.parquet", "source_format": "parquet",
            "row_count": 5000, "column_count": 4,
            "columns": {
                "customer_id": {"name": "customer_id", "inferred_type": "identifier",
                                "confidence": 0.95, "evidence": [], "quality_score": 100,
                                "cleaning_needed": False, "cleaning_recommendations": []},
                "amount": {"name": "amount", "inferred_type": "numeric_continuous",
                           "confidence": 0.9, "evidence": [], "quality_score": 90,
                           "cleaning_needed": False, "cleaning_recommendations": []},
                "order_date": {"name": "order_date", "inferred_type": "datetime",
                               "confidence": 0.95, "evidence": [], "quality_score": 100,
                               "cleaning_needed": False, "cleaning_recommendations": []},
            },
            "identifier_columns": ["customer_id"],
            "datetime_columns": ["order_date"],
            "time_series_metadata": {
                "granularity": "event_level",
                "entity_column": "customer_id",
                "time_column": "order_date",
                "aggregation_windows_used": ["7d", "30d", "90d"],
            },
        }
        (findings_dir / "orders_findings.yaml").write_text(yaml.dump(orders_findings))


class TestDiscoveredEventMergeSourcesReconciliation:
    """When _build_discovered_landing_configs promotes a source to event-level,
    the corresponding merge_sources entry must be updated to EVENT_LEVEL granularity."""

    def test_discovered_event_merge_source_granularity(self, tmp_path):
        from customer_retention.analysis.auto_explorer.project_context import (
            ObjectivePriority,
            ObjectiveSpec,
            PredictionObjective,
            ProjectContext,
        )
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
        from customer_retention.analysis.auto_explorer.snapshot_grid import SnapshotGrid
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        ns = RunNamespace.create(root=tmp_path, project_name="test")
        SnapshotGrid(
            cadence_interval="weekly", observation_window_days=90,
            grid_dates=["2024-01-01", "2024-01-08"],
        ).save(ns.snapshot_grid_path)
        ProjectContext(
            entity_column="customer_id",
            objectives=[ObjectiveSpec(
                objective=PredictionObjective.IMMEDIATE_RISK,
                priority=ObjectivePriority.PRIMARY,
            )],
        ).save(ns.project_context_path)

        # Write findings into per-dataset namespace dirs so discover_all_findings sees them
        cust_dir = ns.dataset_findings_dir("customers")
        cust_dir.mkdir(parents=True, exist_ok=True)
        agg_dir = ns.dataset_findings_dir("orders_agg")
        agg_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = ns.dataset_findings_dir("orders_raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

        agg_findings_path = agg_dir / "orders_agg_findings.yaml"
        agg_parquet = str(agg_dir / "orders_agg.parquet")

        findings_dir = ns.multi_dataset_findings_path.parent
        findings_dir.mkdir(parents=True, exist_ok=True)

        multi_dataset = {
            "datasets": {
                "customers": {
                    "name": "customers",
                    "findings_path": str(cust_dir / "customers_findings.yaml"),
                    "source_path": "/data/customers.csv",
                    "granularity": "entity_level",
                    "row_count": 1000, "column_count": 3,
                    "entity_column": "customer_id",
                },
                "orders_agg": {
                    "name": "orders_agg",
                    "findings_path": str(agg_findings_path),
                    "source_path": agg_parquet,
                    "granularity": "entity_level",
                    "row_count": 500, "column_count": 6,
                },
            },
            "relationships": [{
                "left_dataset": "customers", "right_dataset": "orders_agg",
                "left_column": "customer_id", "right_column": "customer_id",
                "relationship_type": "one_to_one", "confidence": 1.0,
            }],
            "primary_entity_dataset": "customers",
            "event_datasets": [],
            "excluded_datasets": [],
        }
        ns.multi_dataset_findings_path.write_text(yaml.dump(multi_dataset))

        def _col(name, typ):
            return {"name": name, "inferred_type": typ, "confidence": 0.95,
                    "evidence": [], "quality_score": 100,
                    "cleaning_needed": False, "cleaning_recommendations": []}
        (cust_dir / "customers_findings.yaml").write_text(yaml.dump({
            "source_path": "/data/customers.csv", "source_format": "csv",
            "row_count": 1000, "column_count": 3,
            "columns": {"customer_id": _col("customer_id", "identifier"),
                        "churn": _col("churn", "binary")},
            "target_column": "churn", "identifier_columns": ["customer_id"],
        }))
        agg_findings_path.write_text(yaml.dump({
            "source_path": agg_parquet, "source_format": "parquet",
            "row_count": 500, "column_count": 6,
            "columns": {"customer_id": _col("customer_id", "identifier"),
                        "total_amount": _col("total_amount", "numeric_continuous")},
            "identifier_columns": ["customer_id"],
        }))
        (raw_dir / "orders_raw_findings.yaml").write_text(yaml.dump({
            "source_path": "/data/raw/orders.csv", "source_format": "csv",
            "row_count": 5000, "column_count": 4,
            "columns": {"customer_id": _col("customer_id", "identifier"),
                        "amount": _col("amount", "numeric_continuous"),
                        "order_date": _col("order_date", "datetime")},
            "identifier_columns": ["customer_id"],
            "datetime_columns": ["order_date"],
            "time_series_metadata": {
                "granularity": "event_level", "entity_column": "customer_id",
                "time_column": "order_date", "aggregation_executed": True,
                "aggregated_findings_path": str(agg_findings_path),
                "suggested_aggregations": ["7d", "30d"],
                "aggregation_windows_used": ["7d", "30d"],
            },
        }))

        parser = FindingsParser(str(findings_dir), namespace=ns)
        config = parser.parse()

        source = next(s for s in config.sources if s.name == "orders_agg")
        assert source.is_event_level is True

        merge_src = next(
            (s for s in config.silver.merge_sources if s.name == "orders_agg"), None,
        )
        assert merge_src is not None
        assert merge_src.granularity == "event_level", (
            "Discovered event source must have event_level granularity in merge_sources "
            "to match SOURCES.is_event_level — otherwise silver loads the aggregated "
            "bronze table but tries an as-of join with a nonexistent timestamp column"
        )


class TestBuildGoldConfigColumnTypes:
    def _make_findings_with_types(self, col_types: dict):
        from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
        columns = {}
        for name, ct in col_types.items():
            columns[name] = ColumnFinding(name=name, inferred_type=ct, confidence=1.0, evidence=[])
        return ExplorationFindings(source_path="/test", source_format="csv", columns=columns, row_count=100)

    def test_numeric_continuous_gets_scaling(self):
        from customer_retention.core.config.column_config import ColumnType
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings = self._make_findings_with_types({"age": ColumnType.NUMERIC_CONTINUOUS})
        parser = FindingsParser.__new__(FindingsParser)
        gold = parser._build_gold_config({"test": findings})
        assert len(gold.scalings) == 1
        assert gold.scalings[0].column == "age"

    def test_numeric_discrete_gets_scaling(self):
        from customer_retention.core.config.column_config import ColumnType
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings = self._make_findings_with_types({"reopen_count": ColumnType.NUMERIC_DISCRETE})
        parser = FindingsParser.__new__(FindingsParser)
        gold = parser._build_gold_config({"test": findings})
        assert len(gold.scalings) == 1
        assert gold.scalings[0].column == "reopen_count"

    def test_categorical_nominal_gets_encoding(self):
        from customer_retention.core.config.column_config import ColumnType
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings = self._make_findings_with_types({"region": ColumnType.CATEGORICAL_NOMINAL})
        parser = FindingsParser.__new__(FindingsParser)
        gold = parser._build_gold_config({"test": findings})
        assert len(gold.encodings) == 1
        assert gold.encodings[0].column == "region"

    def test_categorical_ordinal_gets_encoding(self):
        from customer_retention.core.config.column_config import ColumnType
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings = self._make_findings_with_types({"priority": ColumnType.CATEGORICAL_ORDINAL})
        parser = FindingsParser.__new__(FindingsParser)
        gold = parser._build_gold_config({"test": findings})
        assert len(gold.encodings) == 1

    def test_binary_not_encoded_or_scaled(self):
        from customer_retention.core.config.column_config import ColumnType
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings = self._make_findings_with_types({"is_active": ColumnType.BINARY})
        parser = FindingsParser.__new__(FindingsParser)
        gold = parser._build_gold_config({"test": findings})
        assert len(gold.encodings) == 0
        assert len(gold.scalings) == 0

    def test_identifier_and_target_excluded(self):
        from customer_retention.core.config.column_config import ColumnType
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings = self._make_findings_with_types({
            "customer_id": ColumnType.IDENTIFIER,
            "churn": ColumnType.TARGET,
        })
        parser = FindingsParser.__new__(FindingsParser)
        gold = parser._build_gold_config({"test": findings})
        assert len(gold.encodings) == 0
        assert len(gold.scalings) == 0

    def test_mixed_types(self):
        from customer_retention.core.config.column_config import ColumnType
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings = self._make_findings_with_types({
            "age": ColumnType.NUMERIC_CONTINUOUS,
            "count": ColumnType.NUMERIC_DISCRETE,
            "region": ColumnType.CATEGORICAL_NOMINAL,
            "tier": ColumnType.CATEGORICAL_ORDINAL,
            "is_active": ColumnType.BINARY,
            "customer_id": ColumnType.IDENTIFIER,
        })
        parser = FindingsParser.__new__(FindingsParser)
        gold = parser._build_gold_config({"test": findings})
        scaled_cols = {s.column for s in gold.scalings}
        encoded_cols = {e.column for e in gold.encodings}
        assert scaled_cols == {"age", "count"}
        assert encoded_cols == {"region", "tier"}

    def test_target_column_categorical_not_encoded(self):
        from customer_retention.core.config.column_config import ColumnType
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings = self._make_findings_with_types({"unsubscribed": ColumnType.CATEGORICAL_NOMINAL, "region": ColumnType.CATEGORICAL_NOMINAL})
        findings.target_column = "unsubscribed"
        parser = FindingsParser.__new__(FindingsParser)
        gold = parser._build_gold_config({"test": findings})
        encoded_cols = {e.column for e in gold.encodings}
        assert "unsubscribed" not in encoded_cols
        assert "region" in encoded_cols

    def test_target_column_numeric_not_scaled(self):
        from customer_retention.core.config.column_config import ColumnType
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings = self._make_findings_with_types({"churn_score": ColumnType.NUMERIC_CONTINUOUS, "age": ColumnType.NUMERIC_CONTINUOUS})
        findings.target_column = "churn_score"
        parser = FindingsParser.__new__(FindingsParser)
        gold = parser._build_gold_config({"test": findings})
        scaled_cols = {s.column for s in gold.scalings}
        assert "churn_score" not in scaled_cols
        assert "age" in scaled_cols

    def test_target_column_ordinal_not_encoded(self):
        from customer_retention.core.config.column_config import ColumnType
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings = self._make_findings_with_types({"risk_level": ColumnType.CATEGORICAL_ORDINAL})
        findings.target_column = "risk_level"
        parser = FindingsParser.__new__(FindingsParser)
        gold = parser._build_gold_config({"test": findings})
        assert len(gold.encodings) == 0


class TestGoldRecommendationsSkipTarget:
    def test_apply_gold_encoding_skips_target_column(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            GoldRecommendations,
            LayeredRecommendation,
            RecommendationRegistry,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import (
            BronzeLayerConfig,
            GoldLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
        )

        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = {"data": {"customer_id", "unsubscribed", "region"}}
        parser._source_findings_paths = {}

        source = SourceConfig(name="data", path="data.csv", format="csv", entity_key="customer_id", raw_source_path="/data/data.csv")
        config = PipelineConfig(name="test", target_column="unsubscribed", sources=[source], bronze={"data": BronzeLayerConfig(source=source)}, silver=SilverLayerConfig(), gold=GoldLayerConfig(), output_dir=".")

        registry = RecommendationRegistry()
        registry.add_source("data", "data.csv")
        registry.gold = GoldRecommendations(target_column="unsubscribed")
        registry.gold.encoding = [
            LayeredRecommendation(id="e1", layer="gold", category="encoding", action="one_hot", target_column="unsubscribed", parameters={"method": "one_hot"}, rationale="encode", source_notebook="04"),
            LayeredRecommendation(id="e2", layer="gold", category="encoding", action="one_hot", target_column="region", parameters={"method": "one_hot"}, rationale="encode", source_notebook="04"),
        ]
        parser._apply_gold_recommendations(config, registry)
        encoded_cols = {e.column for e in config.gold.encodings}
        assert "unsubscribed" not in encoded_cols
        assert "region" in encoded_cols

    def test_apply_gold_scaling_skips_target_column(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            GoldRecommendations,
            LayeredRecommendation,
            RecommendationRegistry,
        )
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import (
            BronzeLayerConfig,
            GoldLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
        )

        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = {"data": {"customer_id", "churn_score", "age"}}
        parser._source_findings_paths = {}

        source = SourceConfig(name="data", path="data.csv", format="csv", entity_key="customer_id", raw_source_path="/data/data.csv")
        config = PipelineConfig(name="test", target_column="churn_score", sources=[source], bronze={"data": BronzeLayerConfig(source=source)}, silver=SilverLayerConfig(), gold=GoldLayerConfig(), output_dir=".")

        registry = RecommendationRegistry()
        registry.add_source("data", "data.csv")
        registry.gold = GoldRecommendations(target_column="churn_score")
        registry.gold.scaling = [
            LayeredRecommendation(id="s1", layer="gold", category="scaling", action="standard", target_column="churn_score", parameters={"method": "standard"}, rationale="scale", source_notebook="04"),
            LayeredRecommendation(id="s2", layer="gold", category="scaling", action="standard", target_column="age", parameters={"method": "standard"}, rationale="scale", source_notebook="04"),
        ]
        parser._apply_gold_recommendations(config, registry)
        scaled_cols = {s.column for s in config.gold.scalings}
        assert "churn_score" not in scaled_cols
        assert "age" in scaled_cols


class TestSilverDerivedColumnValidation:
    def _make_parser(self, raw_source_columns=None):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = raw_source_columns or {}
        parser._source_findings_paths = {}
        return parser

    def _make_config_with_entity_and_event(self, entity_columns, event_config):
        from customer_retention.generators.pipeline_generator.models import (
            BronzeEventConfig,
            BronzeLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
        )

        entity_source = SourceConfig(
            name="customers", path="customers.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/customers.csv",
        )
        event_source = SourceConfig(
            name="events", path="events.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/events.csv",
            is_event_level=True, time_column="sent_date",
        )
        config = PipelineConfig(
            name="test", target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": BronzeLayerConfig(source=entity_source)},
            silver=SilverLayerConfig(),
            gold=None, output_dir=".",
        )
        config.bronze_event["events"] = event_config or BronzeEventConfig(
            source=event_source, entity_column="customer_id", time_column="sent_date",
        )
        return config

    def _make_registry_with_silver_derived(self, recs):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            RecommendationRegistry,
            SilverRecommendations,
        )

        registry = RecommendationRegistry()
        registry.silver = SilverRecommendations(entity_column="customer_id", derived_columns=recs)
        return registry

    def test_ratio_valid_columns_passes(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import LayeredRecommendation
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
            SourceConfig,
        )

        event_source = SourceConfig(
            name="events", path="events.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/events.csv",
            is_event_level=True, time_column="sent_date",
        )
        event_cfg = BronzeEventConfig(
            source=event_source, entity_column="customer_id", time_column="sent_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
        )
        parser = self._make_parser(raw_source_columns={"customers": {"customer_id", "age", "income"}})
        config = self._make_config_with_entity_and_event(
            entity_columns={"customer_id", "age", "income"},
            event_config=event_cfg,
        )
        rec = LayeredRecommendation(
            id="d1", layer="silver", category="derived_column", action="ratio",
            target_column="age_to_income", parameters={"numerator": "age", "denominator": "income"},
            rationale="ratio", source_notebook="06",
        )
        registry = self._make_registry_with_silver_derived([rec])
        parser._apply_silver_recommendations(config, registry)
        assert len(config.silver.derived_columns) == 1
        assert config.silver.derived_columns[0].column == "age_to_income"

    def test_ratio_missing_columns_filtered(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import LayeredRecommendation
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
            SourceConfig,
        )

        event_source = SourceConfig(
            name="events", path="events.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/events.csv",
            is_event_level=True, time_column="sent_date",
        )
        event_cfg = BronzeEventConfig(
            source=event_source, entity_column="customer_id", time_column="sent_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
        )
        parser = self._make_parser(raw_source_columns={"customers": {"customer_id", "age", "income"}})
        config = self._make_config_with_entity_and_event(
            entity_columns={"customer_id", "age", "income"},
            event_config=event_cfg,
        )
        rec = LayeredRecommendation(
            id="d1", layer="silver", category="derived_column", action="ratio",
            target_column="click_to_open_rate",
            parameters={"numerator": "clicked_velocity_pct", "denominator": "opened_velocity_pct"},
            rationale="ratio", source_notebook="06",
        )
        registry = self._make_registry_with_silver_derived([rec])
        parser._apply_silver_recommendations(config, registry)
        assert len(config.silver.derived_columns) == 0

    def test_interaction_valid_columns_passes(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import LayeredRecommendation
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
            SourceConfig,
        )

        event_source = SourceConfig(
            name="events", path="events.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/events.csv",
            is_event_level=True, time_column="sent_date",
        )
        event_cfg = BronzeEventConfig(
            source=event_source, entity_column="customer_id", time_column="sent_date",
            aggregation=AggregationWindowConfig(
                windows=["7d"],
                value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
        )
        parser = self._make_parser(raw_source_columns={"customers": {"customer_id", "age", "income"}})
        config = self._make_config_with_entity_and_event(
            entity_columns={"customer_id", "age", "income"},
            event_config=event_cfg,
        )
        rec = LayeredRecommendation(
            id="d1", layer="silver", category="derived_column", action="interaction",
            target_column="amount_sum_7d_x_age",
            parameters={"features": ["amount_sum_7d", "age"]},
            rationale="interaction", source_notebook="06",
        )
        registry = self._make_registry_with_silver_derived([rec])
        parser._apply_silver_recommendations(config, registry)
        assert len(config.silver.derived_columns) == 1

    def test_interaction_missing_columns_filtered(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import LayeredRecommendation
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
            SourceConfig,
        )

        event_source = SourceConfig(
            name="events", path="events.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/events.csv",
            is_event_level=True, time_column="sent_date",
        )
        event_cfg = BronzeEventConfig(
            source=event_source, entity_column="customer_id", time_column="sent_date",
            aggregation=AggregationWindowConfig(
                windows=["7d"],
                value_columns=["amount"],
                agg_funcs=["sum"],
            ),
        )
        parser = self._make_parser(raw_source_columns={"customers": {"customer_id", "age"}})
        config = self._make_config_with_entity_and_event(
            entity_columns={"customer_id", "age"},
            event_config=event_cfg,
        )
        rec = LayeredRecommendation(
            id="d1", layer="silver", category="derived_column", action="interaction",
            target_column="combo",
            parameters={"features": ["clicked_velocity_pct", "age"]},
            rationale="interaction", source_notebook="06",
        )
        registry = self._make_registry_with_silver_derived([rec])
        parser._apply_silver_recommendations(config, registry)
        assert len(config.silver.derived_columns) == 0

    def test_composite_missing_columns_filtered(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import LayeredRecommendation
        from customer_retention.generators.pipeline_generator.models import BronzeEventConfig, SourceConfig

        event_source = SourceConfig(
            name="events", path="events.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/events.csv",
            is_event_level=True, time_column="sent_date",
        )
        event_cfg = BronzeEventConfig(
            source=event_source, entity_column="customer_id", time_column="sent_date",
        )
        parser = self._make_parser(raw_source_columns={"customers": {"customer_id", "age"}})
        config = self._make_config_with_entity_and_event(
            entity_columns={"customer_id", "age"},
            event_config=event_cfg,
        )
        rec = LayeredRecommendation(
            id="d1", layer="silver", category="derived_column", action="composite",
            target_column="avg_score",
            parameters={"columns": ["score_a", "score_b", "score_c"]},
            rationale="composite", source_notebook="06",
        )
        registry = self._make_registry_with_silver_derived([rec])
        parser._apply_silver_recommendations(config, registry)
        assert len(config.silver.derived_columns) == 0

    def test_composite_without_columns_key_filtered(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import LayeredRecommendation
        from customer_retention.generators.pipeline_generator.models import BronzeEventConfig, SourceConfig

        event_source = SourceConfig(
            name="events", path="events.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/events.csv",
            is_event_level=True, time_column="sent_date",
        )
        event_cfg = BronzeEventConfig(
            source=event_source, entity_column="customer_id", time_column="sent_date",
        )
        parser = self._make_parser(raw_source_columns={"customers": {"customer_id", "age"}})
        config = self._make_config_with_entity_and_event(
            entity_columns={"customer_id", "age"},
            event_config=event_cfg,
        )
        rec = LayeredRecommendation(
            id="d1", layer="silver", category="derived_column", action="composite",
            target_column="email_engagement_score",
            parameters={"expression": "0.6 * open_rate + 0.4 * click_rate", "feature_type": "composite"},
            rationale="composite", source_notebook="06",
        )
        registry = self._make_registry_with_silver_derived([rec])
        parser._apply_silver_recommendations(config, registry)
        assert len(config.silver.derived_columns) == 0

    def test_composite_with_valid_columns_passes(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import LayeredRecommendation
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
            SourceConfig,
        )

        event_source = SourceConfig(
            name="events", path="events.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/events.csv",
            is_event_level=True, time_column="sent_date",
        )
        event_cfg = BronzeEventConfig(
            source=event_source, entity_column="customer_id", time_column="sent_date",
            aggregation=AggregationWindowConfig(
                windows=["7d"], value_columns=["open_rate", "click_rate"], agg_funcs=["mean"],
            ),
        )
        parser = self._make_parser(raw_source_columns={"customers": {"customer_id", "age"}})
        config = self._make_config_with_entity_and_event(
            entity_columns={"customer_id", "age"},
            event_config=event_cfg,
        )
        rec = LayeredRecommendation(
            id="d1", layer="silver", category="derived_column", action="composite",
            target_column="engagement_score",
            parameters={"columns": ["open_rate_mean_7d", "click_rate_mean_7d"], "feature_type": "composite"},
            rationale="composite", source_notebook="06",
        )
        registry = self._make_registry_with_silver_derived([rec])
        parser._apply_silver_recommendations(config, registry)
        assert len(config.silver.derived_columns) == 1
        step = config.silver.derived_columns[0]
        assert step.parameters["columns"] == ["open_rate_mean_7d", "click_rate_mean_7d"]

    def test_collect_pipeline_columns_aggregation(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
            BronzeLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
        )

        entity_source = SourceConfig(
            name="customers", path="customers.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/customers.csv",
        )
        event_source = SourceConfig(
            name="events", path="events.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/events.csv",
            is_event_level=True, time_column="sent_date",
        )
        config = PipelineConfig(
            name="test", target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": BronzeLayerConfig(source=entity_source)},
            silver=SilverLayerConfig(), gold=None, output_dir=".",
        )
        config.bronze_event["events"] = BronzeEventConfig(
            source=event_source, entity_column="customer_id", time_column="sent_date",
            aggregation=AggregationWindowConfig(
                windows=["7d", "30d"],
                value_columns=["amount", "quantity"],
                agg_funcs=["sum", "mean"],
                categorical_columns=["category"],
                categorical_agg_funcs=["nunique", "mode"],
            ),
        )
        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = {"customers": {"customer_id", "age", "income"}}
        parser._source_findings_paths = {}
        cols = parser._collect_pipeline_columns(config)
        assert "amount_sum_7d" in cols
        assert "amount_mean_30d" in cols
        assert "quantity_sum_7d" in cols
        assert "category_nunique_7d" in cols
        assert "category_mode_30d" in cols
        assert "event_count_7d" in cols
        assert "event_count_30d" in cols
        assert "age" in cols
        assert "income" in cols

    def test_collect_pipeline_columns_entity_minus_drops(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import (
            BronzeLayerConfig,
            PipelineConfig,
            PipelineTransformationType,
            SilverLayerConfig,
            SourceConfig,
            TransformationStep,
        )

        entity_source = SourceConfig(
            name="customers", path="customers.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/customers.csv",
        )
        drop_step = TransformationStep(
            type=PipelineTransformationType.DROP_COLUMN,
            column="junk_col",
            parameters={},
            rationale="drop junk",
        )
        config = PipelineConfig(
            name="test", target_column="churn",
            sources=[entity_source],
            bronze={"customers": BronzeLayerConfig(source=entity_source, transformations=[drop_step])},
            silver=SilverLayerConfig(), gold=None, output_dir=".",
        )
        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = {"customers": {"customer_id", "age", "junk_col"}}
        parser._source_findings_paths = {}
        cols = parser._collect_pipeline_columns(config)
        assert "age" in cols
        assert "junk_col" not in cols

    def test_collect_pipeline_columns_lifecycle(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import (
            BronzeEventConfig,
            BronzeLayerConfig,
            LifecycleConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
        )

        entity_source = SourceConfig(
            name="customers", path="customers.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/customers.csv",
        )
        event_source = SourceConfig(
            name="events", path="events.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/events.csv",
            is_event_level=True, time_column="sent_date",
        )
        config = PipelineConfig(
            name="test", target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": BronzeLayerConfig(source=entity_source)},
            silver=SilverLayerConfig(), gold=None, output_dir=".",
        )
        config.bronze_event["events"] = BronzeEventConfig(
            source=event_source, entity_column="customer_id", time_column="sent_date",
            lifecycle=LifecycleConfig(
                include_lifecycle_quadrant=True,
                include_cyclical_features=True,
                include_recency_bucket=True,
                include_trend_features=True,
                include_cohort_features=True,
                include_month_cyclical=True,
                include_quarter_cyclical=True,
                momentum_pairs=[{"short_window": "7d", "long_window": "30d"}],
            ),
        )
        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = {"customers": {"customer_id", "age"}}
        parser._source_findings_paths = {}
        cols = parser._collect_pipeline_columns(config)
        assert "days_since_last" in cols
        assert "days_since_first" in cols
        assert "recency_bucket" in cols
        assert "lifecycle_quadrant" in cols
        assert "dow_sin" in cols
        assert "dow_cos" in cols
        assert "month_sin" in cols
        assert "month_cos" in cols
        assert "quarter_sin" in cols
        assert "quarter_cos" in cols
        assert "recent_vs_overall_ratio" in cols
        assert "entity_trend_slope" in cols
        assert "cohort_year" in cols
        assert "cohort_quarter" in cols
        assert "momentum_7d_30d" in cols


class TestCollectPipelineColumnsLagFeatures:
    def test_pipeline_columns_include_lag_features(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import (
            BronzeEventConfig,
            BronzeLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
            TemporalFeatureConfig,
        )

        entity_source = SourceConfig(
            name="customers", path="customers.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/customers.csv",
        )
        event_source = SourceConfig(
            name="events", path="events.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/events.csv",
            is_event_level=True, time_column="sent_date",
        )
        config = PipelineConfig(
            name="test", target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": BronzeLayerConfig(source=entity_source)},
            silver=SilverLayerConfig(), gold=None, output_dir=".",
        )
        config.bronze_event["events"] = BronzeEventConfig(
            source=event_source, entity_column="customer_id", time_column="sent_date",
            temporal_features=TemporalFeatureConfig(
                lag_columns=["amount"],
                num_lags=2,
                lag_agg_funcs=["sum", "mean"],
                feature_groups=["lagged_windows"],
            ),
        )
        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = {"customers": {"customer_id"}}
        parser._source_findings_paths = {}
        cols = parser._collect_pipeline_columns(config)
        assert "lag0_amount_sum" in cols
        assert "lag0_amount_mean" in cols
        assert "lag1_amount_sum" in cols
        assert "lag1_amount_mean" in cols

    def test_pipeline_columns_include_velocity_features(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import (
            BronzeEventConfig,
            BronzeLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
            TemporalFeatureConfig,
        )

        entity_source = SourceConfig(
            name="customers", path="customers.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/customers.csv",
        )
        event_source = SourceConfig(
            name="events", path="events.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/events.csv",
            is_event_level=True, time_column="sent_date",
        )
        config = PipelineConfig(
            name="test", target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": BronzeLayerConfig(source=entity_source)},
            silver=SilverLayerConfig(), gold=None, output_dir=".",
        )
        config.bronze_event["events"] = BronzeEventConfig(
            source=event_source, entity_column="customer_id", time_column="sent_date",
            temporal_features=TemporalFeatureConfig(
                lag_columns=["amount"],
                num_lags=2,
                lag_agg_funcs=["sum", "mean"],
                feature_groups=["lagged_windows", "velocity"],
            ),
        )
        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = {"customers": {"customer_id"}}
        parser._source_findings_paths = {}
        cols = parser._collect_pipeline_columns(config)
        assert "amount_velocity" in cols
        assert "amount_velocity_pct" in cols

    def test_pipeline_columns_no_velocity_without_feature_group(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import (
            BronzeEventConfig,
            BronzeLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
            TemporalFeatureConfig,
        )

        entity_source = SourceConfig(
            name="customers", path="customers.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/customers.csv",
        )
        event_source = SourceConfig(
            name="events", path="events.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/events.csv",
            is_event_level=True, time_column="sent_date",
        )
        config = PipelineConfig(
            name="test", target_column="churn",
            sources=[entity_source, event_source],
            bronze={"customers": BronzeLayerConfig(source=entity_source)},
            silver=SilverLayerConfig(), gold=None, output_dir=".",
        )
        config.bronze_event["events"] = BronzeEventConfig(
            source=event_source, entity_column="customer_id", time_column="sent_date",
            temporal_features=TemporalFeatureConfig(
                lag_columns=["amount"],
                num_lags=2,
                lag_agg_funcs=["sum"],
                feature_groups=["lagged_windows"],
            ),
        )
        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = {"customers": {"customer_id"}}
        parser._source_findings_paths = {}
        cols = parser._collect_pipeline_columns(config)
        assert "amount_velocity" not in cols


class TestTemporalFeatureGroupPrediction:
    @staticmethod
    def _cols_for_groups(feature_groups, lag_columns=None):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import (
            BronzeEventConfig,
            BronzeLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
            TemporalFeatureConfig,
        )
        entity_src = SourceConfig(name="cust", path="c.csv", format="csv", entity_key="cid", raw_source_path="/c.csv")
        event_src = SourceConfig(name="evt", path="e.csv", format="csv", entity_key="cid", raw_source_path="/e.csv", is_event_level=True, time_column="ts")
        config = PipelineConfig(name="t", target_column="churn", sources=[entity_src, event_src],
                                bronze={"cust": BronzeLayerConfig(source=entity_src)}, silver=SilverLayerConfig(), gold=None, output_dir=".")
        config.bronze_event["evt"] = BronzeEventConfig(
            source=event_src, entity_column="cid", time_column="ts",
            temporal_features=TemporalFeatureConfig(lag_columns=lag_columns or ["amt"], num_lags=2, lag_agg_funcs=["sum"], feature_groups=feature_groups),
        )
        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = {"cust": {"cid"}}
        parser._source_findings_paths = {}
        return parser._collect_pipeline_columns(config)

    def test_velocity_columns_predicted(self):
        cols = self._cols_for_groups(["lagged_windows", "velocity"])
        assert "amt_velocity" in cols
        assert "amt_velocity_pct" in cols

    def test_velocity_naming_matches_temporal_feature_engineer(self):
        import pandas as pd

        from customer_retention.stages.profiling.temporal_feature_engineer import (
            TemporalAggregationConfig,
            TemporalFeatureEngineer,
        )
        raw = pd.DataFrame({
            "cid": ["a"] * 120, "ts": pd.date_range("2020-01-01", periods=120, freq="D"),
            "amt": range(120),
        })
        eng = TemporalFeatureEngineer(TemporalAggregationConfig(lag_window_days=30, num_lags=2, lag_aggregations=["sum"]))
        result = eng.compute(raw, "cid", "ts", ["amt"])
        actual_velocity_cols = {c for c in result.features_df.columns if "velocity" in c.lower()}
        predicted_cols = self._cols_for_groups(["lagged_windows", "velocity"])
        predicted_velocity_cols = {c for c in predicted_cols if "velocity" in c.lower()}
        assert actual_velocity_cols == predicted_velocity_cols

    def test_acceleration_columns_predicted(self):
        cols = self._cols_for_groups(["lagged_windows", "acceleration"])
        assert "amt_acceleration" in cols
        assert "amt_momentum" in cols

    def test_lifecycle_columns_predicted(self):
        cols = self._cols_for_groups(["lagged_windows", "lifecycle"])
        for suffix in ("_beginning", "_middle", "_end", "_trend_ratio"):
            assert f"amt{suffix}" in cols

    def test_recency_columns_predicted(self):
        cols = self._cols_for_groups(["lagged_windows", "recency"])
        for name in ("days_since_last_event", "days_since_first_event", "active_span_days", "recency_ratio"):
            assert name in cols

    def test_regularity_columns_predicted(self):
        cols = self._cols_for_groups(["lagged_windows", "regularity"])
        for name in ("event_frequency", "inter_event_gap_mean", "inter_event_gap_std", "inter_event_gap_max", "regularity_score"):
            assert name in cols

    def test_cohort_columns_predicted(self):
        cols = self._cols_for_groups(["lagged_windows", "cohort_comparison"])
        assert "amt_vs_cohort_mean" in cols
        assert "amt_vs_cohort_pct" in cols
        assert "amt_cohort_zscore" in cols

    def test_disabled_group_not_predicted(self):
        cols = self._cols_for_groups(["lagged_windows"])
        assert "amt_acceleration" not in cols
        assert "amt_momentum" not in cols
        assert "amt_vs_cohort_mean" not in cols
        assert "event_frequency" not in cols
        assert "regularity_score" not in cols

    def test_build_temporal_config_uses_all_default_groups(self):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import TemporalFeatureConfig
        findings = ExplorationFindings(source_path="/tmp/test.csv", source_format="csv")
        findings.metadata = {"temporal_patterns": {
            "lag_features_computed": True,
            "lag_window_days": 30,
            "num_lags": 4,
            "lag_columns": ["amt"],
            "feature_groups": ["lagged_windows", "velocity"],
        }}
        parser = FindingsParser.__new__(FindingsParser)
        from customer_retention.analysis.auto_explorer.exploration_manager import MultiDatasetFindings
        multi = MultiDatasetFindings.__new__(MultiDatasetFindings)
        multi.notes = {}
        result = parser._build_temporal_feature_config(multi, findings)
        assert result is not None
        assert set(result.feature_groups) == set(TemporalFeatureConfig().feature_groups)


class TestGoldRecommendationFiltering:
    @staticmethod
    def _make_config_with_columns(pipeline_columns):
        from customer_retention.generators.pipeline_generator.models import (
            BronzeLayerConfig,
            GoldLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
        )

        entity_source = SourceConfig(
            name="customers", path="customers.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/customers.csv",
        )
        config = PipelineConfig(
            name="test", target_column="churn",
            sources=[entity_source],
            bronze={"customers": BronzeLayerConfig(source=entity_source)},
            silver=SilverLayerConfig(), gold=GoldLayerConfig(), output_dir=".",
        )
        return config, pipeline_columns

    @staticmethod
    def _make_parser(raw_columns):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = raw_columns
        parser._source_findings_paths = {}
        return parser

    @staticmethod
    def _make_registry_with_gold(encoding=None, scaling=None, transformations=None):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            GoldRecommendations,
            RecommendationRegistry,
        )
        registry = RecommendationRegistry()
        registry.gold = GoldRecommendations(
            target_column="churn",
            encoding=encoding or [],
            scaling=scaling or [],
            transformations=transformations or [],
        )
        return registry

    @staticmethod
    def _make_rec(target_column, action, parameters=None):
        from customer_retention.analysis.auto_explorer.layered_recommendations import LayeredRecommendation
        return LayeredRecommendation(
            id="test_rec",
            layer="gold",
            category="test",
            action=action,
            target_column=target_column,
            parameters=parameters or {},
            rationale="test rationale",
            source_notebook="nb04",
        )

    def test_gold_transformation_filtered_when_column_missing(self, caplog):
        import logging

        config, _ = self._make_config_with_columns({"entity_id", "as_of_date", "age"})
        parser = self._make_parser({"customers": {"customer_id", "age"}})
        rec = self._make_rec("nonexistent_col", "log_transform")
        registry = self._make_registry_with_gold(transformations=[rec])
        with caplog.at_level(logging.WARNING):
            parser._apply_gold_recommendations(config, registry)
        assert len(config.gold.transformations) == 0
        assert any("nonexistent_col" in m and "transformation" in m.lower() for m in caplog.messages)

    def test_gold_encoding_filtered_when_column_missing(self, caplog):
        import logging

        config, _ = self._make_config_with_columns({"entity_id", "as_of_date", "age"})
        parser = self._make_parser({"customers": {"customer_id", "age"}})
        rec = self._make_rec("nonexistent_col", "onehot", {"method": "onehot"})
        registry = self._make_registry_with_gold(encoding=[rec])
        with caplog.at_level(logging.WARNING):
            parser._apply_gold_recommendations(config, registry)
        assert len(config.gold.encodings) == 0
        assert any("nonexistent_col" in m and "encoding" in m.lower() for m in caplog.messages)

    def test_gold_scaling_filtered_when_column_missing(self, caplog):
        import logging

        config, _ = self._make_config_with_columns({"entity_id", "as_of_date", "age"})
        parser = self._make_parser({"customers": {"customer_id", "age"}})
        rec = self._make_rec("nonexistent_col", "standard", {"method": "standard"})
        registry = self._make_registry_with_gold(scaling=[rec])
        with caplog.at_level(logging.WARNING):
            parser._apply_gold_recommendations(config, registry)
        assert len(config.gold.scalings) == 0
        assert any("nonexistent_col" in m and "scaling" in m.lower() for m in caplog.messages)

    def test_gold_recommendation_kept_when_column_exists(self, caplog):
        import logging

        config, _ = self._make_config_with_columns({"entity_id", "as_of_date", "age"})
        parser = self._make_parser({"customers": {"customer_id", "age"}})
        rec = self._make_rec("age", "log_transform")
        registry = self._make_registry_with_gold(transformations=[rec])
        with caplog.at_level(logging.WARNING):
            parser._apply_gold_recommendations(config, registry)
        assert len(config.gold.transformations) == 1
        assert config.gold.transformations[0].column == "age"
        assert not any("Skipping gold" in m for m in caplog.messages)

    def test_gold_keeps_columns_from_silver_derived(self):
        from customer_retention.generators.pipeline_generator.models import (
            PipelineTransformationType,
            TransformationStep,
        )
        config, _ = self._make_config_with_columns({"entity_id", "as_of_date", "age"})
        config.silver.derived_columns.append(
            TransformationStep(
                type=PipelineTransformationType.DERIVED_COLUMN,
                column="age_income_ratio",
                parameters={"action": "ratio"},
                rationale="test",
            )
        )
        parser = self._make_parser({"customers": {"customer_id", "age"}})
        rec = self._make_rec("age_income_ratio", "log_transform")
        registry = self._make_registry_with_gold(transformations=[rec])
        parser._apply_gold_recommendations(config, registry)
        assert len(config.gold.transformations) == 1
        assert config.gold.transformations[0].column == "age_income_ratio"

    def test_silver_derived_missing_columns_emits_warning(self, caplog):
        import logging

        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        rec = self._make_rec("age_income_ratio", "ratio", {
            "numerator": "age", "denominator": "nonexistent_col",
        })
        pipeline_columns = {"entity_id", "as_of_date", "age"}
        with caplog.at_level(logging.WARNING):
            result = FindingsParser._silver_derived_sources_available(rec, pipeline_columns)
        assert result is False
        assert any("nonexistent_col" in m and "silver derived" in m.lower() for m in caplog.messages)


class TestReconcileGoldColumns:
    @staticmethod
    def _make_parser(raw_columns):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = raw_columns
        parser._source_findings_paths = {}
        return parser

    @staticmethod
    def _make_config(scaling_cols, encoding_cols=None):
        from customer_retention.generators.pipeline_generator.models import (
            BronzeLayerConfig,
            GoldLayerConfig,
            PipelineConfig,
            PipelineTransformationType,
            SilverLayerConfig,
            SourceConfig,
            TransformationStep,
        )
        entity_source = SourceConfig(
            name="customers", path="customers.csv", format="csv",
            entity_key="customer_id", raw_source_path="/data/customers.csv",
        )
        scalings = [
            TransformationStep(type=PipelineTransformationType.SCALE, column=c,
                               parameters={"method": "standard"}, rationale=f"Scale {c}")
            for c in scaling_cols
        ]
        encodings = [
            TransformationStep(type=PipelineTransformationType.ENCODE, column=c,
                               parameters={"method": "one_hot"}, rationale=f"Encode {c}")
            for c in (encoding_cols or [])
        ]
        return PipelineConfig(
            name="test", target_column="churn", sources=[entity_source],
            bronze={"customers": BronzeLayerConfig(source=entity_source)},
            silver=SilverLayerConfig(),
            gold=GoldLayerConfig(scalings=scalings, encodings=encodings),
            output_dir=".",
        )

    def test_removes_scaling_for_nonexistent_column(self, caplog):
        import logging
        parser = self._make_parser({"customers": {"customer_id", "age"}})
        config = self._make_config(["age", "phantom_col"])
        with caplog.at_level(logging.WARNING):
            parser._reconcile_gold_columns(config)
        assert len(config.gold.scalings) == 1
        assert config.gold.scalings[0].column == "age"
        assert any("phantom_col" in m for m in caplog.messages)

    def test_removes_encoding_for_nonexistent_column(self, caplog):
        import logging
        parser = self._make_parser({"customers": {"customer_id", "region"}})
        config = self._make_config([], encoding_cols=["region", "ghost"])
        with caplog.at_level(logging.WARNING):
            parser._reconcile_gold_columns(config)
        assert len(config.gold.encodings) == 1
        assert config.gold.encodings[0].column == "region"

    def test_keeps_all_valid_columns(self):
        parser = self._make_parser({"customers": {"customer_id", "age", "score"}})
        config = self._make_config(["age", "score"])
        parser._reconcile_gold_columns(config)
        assert len(config.gold.scalings) == 2
        assert {s.column for s in config.gold.scalings} == {"age", "score"}

    def test_noop_when_gold_is_none(self):
        parser = self._make_parser({"customers": {"customer_id"}})
        config = self._make_config([])
        config.gold = None
        parser._reconcile_gold_columns(config)

    def test_removes_transformation_for_nonexistent_column(self, caplog):
        import logging

        from customer_retention.generators.pipeline_generator.models import (
            PipelineTransformationType,
            TransformationStep,
        )
        parser = self._make_parser({"customers": {"customer_id", "amount"}})
        config = self._make_config([])
        config.gold.transformations = [
            TransformationStep(type=PipelineTransformationType.LOG_TRANSFORM, column="amount",
                               parameters={}, rationale="log amount"),
            TransformationStep(type=PipelineTransformationType.LOG_TRANSFORM, column="missing_col",
                               parameters={}, rationale="log missing"),
        ]
        with caplog.at_level(logging.WARNING):
            parser._reconcile_gold_columns(config)
        assert len(config.gold.transformations) == 1
        assert config.gold.transformations[0].column == "amount"


class TestEventAggregatedColumns:
    @staticmethod
    def _make_event_cfg(**overrides):
        from customer_retention.generators.pipeline_generator.models import BronzeEventConfig, SourceConfig
        src = SourceConfig(name="events", path="events.csv", format="csv",
                           entity_key="cid", raw_source_path="/data/events.csv")
        defaults = dict(source=src, entity_column="cid", time_column="ts")
        defaults.update(overrides)
        return BronzeEventConfig(**defaults)

    def test_aggregation_columns_cartesian(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import AggregationWindowConfig
        agg = AggregationWindowConfig(
            windows=["all_time", "30d"], value_columns=["amount"],
            agg_funcs=["sum", "mean"],
        )
        cols = FindingsParser._event_aggregated_columns(self._make_event_cfg(aggregation=agg))
        assert cols == {
            "amount_sum_all_time", "amount_mean_all_time",
            "amount_sum_30d", "amount_mean_30d",
            "event_count_all_time", "event_count_30d",
        }

    def test_aggregation_excludes_per_column_count(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import AggregationWindowConfig
        agg = AggregationWindowConfig(
            windows=["7d", "180d"], value_columns=["send_hour"],
            agg_funcs=["sum", "mean", "count"],
        )
        cols = FindingsParser._event_aggregated_columns(self._make_event_cfg(aggregation=agg))
        assert "send_hour_sum_7d" in cols
        assert "send_hour_mean_7d" in cols
        assert "event_count_7d" in cols
        assert "event_count_180d" in cols
        assert "send_hour_count_7d" not in cols
        assert "send_hour_count_180d" not in cols

    def test_categorical_columns(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import AggregationWindowConfig
        agg = AggregationWindowConfig(
            windows=["7d"], value_columns=[], agg_funcs=[],
            categorical_columns=["status"], categorical_agg_funcs=["nunique", "mode"],
        )
        cols = FindingsParser._event_aggregated_columns(self._make_event_cfg(aggregation=agg))
        assert {"status_nunique_7d", "status_mode_7d", "event_count_7d"} <= cols

    def test_lifecycle_columns(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import LifecycleConfig
        lc = LifecycleConfig(include_recency_bucket=True, include_lifecycle_quadrant=True,
                             include_cyclical_features=True, include_trend_features=True,
                             include_cohort_features=True, include_month_cyclical=True,
                             include_quarter_cyclical=True,
                             momentum_pairs=[{"short_window": "7d", "long_window": "30d"}])
        cols = FindingsParser._event_aggregated_columns(self._make_event_cfg(lifecycle=lc))
        assert {"days_since_last", "days_since_first", "recency_bucket",
                "lifecycle_quadrant", "dow_sin", "dow_cos", "month_sin", "month_cos",
                "quarter_sin", "quarter_cos", "recent_vs_overall_ratio",
                "entity_trend_slope", "cohort_year", "cohort_quarter",
                "momentum_7d_30d"} <= cols

    def test_empty_config_returns_empty(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        cols = FindingsParser._event_aggregated_columns(self._make_event_cfg())
        assert cols == set()


class TestReconcileEventPostShaping:
    @staticmethod
    def _make_parser():
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = {}
        parser._source_findings_paths = {}
        return parser

    @staticmethod
    def _make_config_with_event(post_shaping_cols, agg_windows=None, agg_value_cols=None, agg_funcs=None):
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
            GoldLayerConfig,
            PipelineConfig,
            PipelineTransformationType,
            SilverLayerConfig,
            SourceConfig,
            TransformationStep,
        )
        src = SourceConfig(name="events", path="events.csv", format="csv",
                           entity_key="cid", raw_source_path="/data/events.csv")
        agg = None
        if agg_windows:
            agg = AggregationWindowConfig(
                windows=agg_windows, value_columns=agg_value_cols or [],
                agg_funcs=agg_funcs or [],
            )
        steps = [
            TransformationStep(type=PipelineTransformationType.CAP_OUTLIER, column=c,
                               parameters={"method": "iqr"}, rationale=f"Cap {c}")
            for c in post_shaping_cols
        ]
        event_cfg = BronzeEventConfig(source=src, entity_column="cid", time_column="ts",
                                      aggregation=agg, post_shaping=steps)
        return PipelineConfig(
            name="test", target_column="churn", sources=[src],
            bronze={}, silver=SilverLayerConfig(),
            gold=GoldLayerConfig(), output_dir=".",
            bronze_event={"events": event_cfg},
        )

    def test_drops_steps_for_nonexistent_aggregated_columns(self, caplog):
        import logging
        parser = self._make_parser()
        config = self._make_config_with_event(
            post_shaping_cols=["amount_sum_30d", "amount_mean_all_time"],
            agg_windows=["30d"], agg_value_cols=["amount"], agg_funcs=["sum"],
        )
        with caplog.at_level(logging.WARNING):
            parser._reconcile_event_post_shaping(config)
        assert len(config.bronze_event["events"].post_shaping) == 1
        assert config.bronze_event["events"].post_shaping[0].column == "amount_sum_30d"
        assert any("amount_mean_all_time" in m for m in caplog.messages)

    def test_keeps_all_valid_columns(self):
        parser = self._make_parser()
        config = self._make_config_with_event(
            post_shaping_cols=["amount_sum_30d", "amount_mean_30d"],
            agg_windows=["30d"], agg_value_cols=["amount"], agg_funcs=["sum", "mean"],
        )
        parser._reconcile_event_post_shaping(config)
        assert len(config.bronze_event["events"].post_shaping) == 2

    def test_noop_when_no_post_shaping(self):
        parser = self._make_parser()
        config = self._make_config_with_event(
            post_shaping_cols=[], agg_windows=["30d"],
            agg_value_cols=["amount"], agg_funcs=["sum"],
        )
        parser._reconcile_event_post_shaping(config)
        assert config.bronze_event["events"].post_shaping == []

    def test_noop_when_no_aggregation(self):
        parser = self._make_parser()
        config = self._make_config_with_event(post_shaping_cols=["amount"])
        parser._reconcile_event_post_shaping(config)
        assert len(config.bronze_event["events"].post_shaping) == 1

    def test_drops_all_when_none_match(self, caplog):
        import logging
        parser = self._make_parser()
        config = self._make_config_with_event(
            post_shaping_cols=["phantom_mean_all_time", "ghost_sum_365d"],
            agg_windows=["30d"], agg_value_cols=["amount"], agg_funcs=["sum"],
        )
        with caplog.at_level(logging.WARNING):
            parser._reconcile_event_post_shaping(config)
        assert config.bronze_event["events"].post_shaping == []

    def test_lifecycle_columns_survive_reconciliation(self):
        from customer_retention.generators.pipeline_generator.models import (
            LifecycleConfig,
            PipelineTransformationType,
            TransformationStep,
        )
        parser = self._make_parser()
        config = self._make_config_with_event(
            post_shaping_cols=[], agg_windows=["30d"],
            agg_value_cols=["amount"], agg_funcs=["sum"],
        )
        event_cfg = config.bronze_event["events"]
        event_cfg.lifecycle = LifecycleConfig(include_recency_bucket=True)
        event_cfg.post_shaping = [
            TransformationStep(type=PipelineTransformationType.CAP_OUTLIER, column="days_since_last",
                               parameters={}, rationale="cap days"),
            TransformationStep(type=PipelineTransformationType.CAP_OUTLIER, column="nonexistent_col",
                               parameters={}, rationale="cap phantom"),
        ]
        parser._reconcile_event_post_shaping(config)
        assert len(event_cfg.post_shaping) == 1
        assert event_cfg.post_shaping[0].column == "days_since_last"

    def test_partial_window_match_drops_missing_combos(self):
        parser = self._make_parser()
        config = self._make_config_with_event(
            post_shaping_cols=["amount_sum_all_time", "amount_mean_all_time", "amount_sum_30d"],
            agg_windows=["all_time", "30d"], agg_value_cols=["amount"], agg_funcs=["sum"],
        )
        parser._reconcile_event_post_shaping(config)
        kept_cols = {s.column for s in config.bronze_event["events"].post_shaping}
        assert kept_cols == {"amount_sum_all_time", "amount_sum_30d"}


class TestColumnTypeDeserialization:
    def test_valid_column_type_deserializes(self):
        from customer_retention.core.config.column_config import ColumnType
        assert ColumnType("numeric_continuous") == ColumnType.NUMERIC_CONTINUOUS
        assert ColumnType("categorical_nominal") == ColumnType.CATEGORICAL_NOMINAL

    def test_bare_numeric_raises(self):
        from customer_retention.core.config.column_config import ColumnType
        with pytest.raises(ValueError):
            ColumnType("numeric")

    def test_bare_categorical_raises(self):
        from customer_retention.core.config.column_config import ColumnType
        with pytest.raises(ValueError):
            ColumnType("categorical")

    def test_bare_uppercase_numeric_raises(self):
        from customer_retention.core.config.column_config import ColumnType
        with pytest.raises(ValueError):
            ColumnType("NUMERIC")


class TestFeatureSelectionDropSkipsTarget:
    @staticmethod
    def _make_parser():
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = {"data": {"customer_id", "unsubscribed", "age", "region"}}
        parser._source_findings_paths = {}
        return parser

    @staticmethod
    def _make_config(target_column="unsubscribed"):
        from customer_retention.generators.pipeline_generator.models import (
            BronzeLayerConfig,
            GoldLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
        )
        source = SourceConfig(name="data", path="data.csv", format="csv", entity_key="customer_id", raw_source_path="/data/data.csv")
        return PipelineConfig(
            name="test", target_column=target_column, sources=[source],
            bronze={"data": BronzeLayerConfig(source=source)},
            silver=SilverLayerConfig(), gold=GoldLayerConfig(), output_dir=".",
        )

    @staticmethod
    def _make_registry(feature_selection=None):
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            GoldRecommendations,
            RecommendationRegistry,
        )
        registry = RecommendationRegistry()
        registry.gold = GoldRecommendations(target_column="unsubscribed", feature_selection=feature_selection or [])
        return registry

    @staticmethod
    def _make_rec(target_column, action):
        from customer_retention.analysis.auto_explorer.layered_recommendations import LayeredRecommendation
        return LayeredRecommendation(
            id="fs_rec", layer="gold", category="feature_selection", action=action,
            target_column=target_column, parameters={}, rationale="test", source_notebook="04",
        )

    def test_drop_weak_skips_target_column(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("unsubscribed", "drop_weak"),
            self._make_rec("age", "drop_weak"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "unsubscribed" not in config.gold.feature_selections
        assert "age" in config.gold.feature_selections

    def test_drop_multicollinear_skips_target_column(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("unsubscribed", "drop_multicollinear"),
            self._make_rec("region", "drop_multicollinear"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "unsubscribed" not in config.gold.feature_selections
        assert "region" in config.gold.feature_selections

    def test_prioritized_column_still_dropped(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("age", "prioritize"),
            self._make_rec("age", "drop_weak"),
            self._make_rec("region", "drop_weak"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "age" in config.gold.feature_selections
        assert "region" in config.gold.feature_selections


class TestAggregationFeatureExclusions:
    @staticmethod
    def _make_parser():
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        parser = FindingsParser.__new__(FindingsParser)
        return parser

    @staticmethod
    def _make_findings(**overrides):
        from customer_retention.analysis.auto_explorer.findings import (
            ColumnFinding,
            ExplorationFindings,
            TimeSeriesMetadata,
        )
        defaults = dict(
            source_path="/data/events.csv",
            source_format="csv",
            columns={
                "customer_id": ColumnFinding(
                    name="customer_id", inferred_type="identifier", confidence=0.95, evidence=[]
                ),
                "sent_date": ColumnFinding(
                    name="sent_date", inferred_type="datetime", confidence=0.95, evidence=[]
                ),
                "amount": ColumnFinding(
                    name="amount", inferred_type="numeric_continuous", confidence=0.9, evidence=[]
                ),
                "count_col": ColumnFinding(
                    name="count_col", inferred_type="numeric_discrete", confidence=0.9, evidence=[]
                ),
                "direction": ColumnFinding(
                    name="direction", inferred_type="categorical_nominal", confidence=0.9, evidence=[]
                ),
                "status": ColumnFinding(
                    name="status", inferred_type="categorical_ordinal", confidence=0.9, evidence=[]
                ),
                "opened": ColumnFinding(
                    name="opened", inferred_type="binary", confidence=0.9, evidence=[]
                ),
            },
            time_series_metadata=TimeSeriesMetadata(
                granularity="event_level",
                entity_column="customer_id",
                time_column="sent_date",
                aggregation_windows_used=["7d", "30d"],
            ),
            identifier_columns=["customer_id"],
            datetime_columns=["sent_date"],
        )
        defaults.update(overrides)
        return ExplorationFindings(**defaults)

    @staticmethod
    def _make_multi(feature_exclusions=None):
        from customer_retention.analysis.auto_explorer.exploration_manager import (
            DatasetInfo,
            MultiDatasetFindings,
        )
        from customer_retention.core.config.column_config import DatasetGranularity
        from customer_retention.generators.pipeline_generator.models import FeatureExclusion

        excl = []
        if feature_exclusions:
            excl = [FeatureExclusion(**e) for e in feature_exclusions]
        return MultiDatasetFindings(
            datasets={
                "events": DatasetInfo(
                    name="events",
                    findings_path="/tmp/events_findings.yaml",
                    source_path="/data/events.csv",
                    granularity=DatasetGranularity.EVENT_LEVEL,
                    row_count=1000,
                    column_count=7,
                    entity_column="customer_id",
                    time_column="sent_date",
                    feature_exclusions=excl,
                ),
            },
            aggregation_windows=["7d", "30d"],
            event_datasets=["events"],
        )

    def test_blocked_category_aggregation(self):
        parser = self._make_parser()
        findings = self._make_findings()
        multi = self._make_multi(feature_exclusions=[
            {"column": "amount", "blocked_categories": ["aggregation"]},
        ])
        result = parser._build_aggregation_config(multi, findings, "events")
        assert "amount" not in result.value_columns
        assert "count_col" in result.value_columns

    def test_blocked_category_categorical(self):
        parser = self._make_parser()
        findings = self._make_findings()
        multi = self._make_multi(feature_exclusions=[
            {"column": "direction", "blocked_categories": ["categorical"]},
        ])
        result = parser._build_aggregation_config(multi, findings, "events")
        assert "direction" not in result.categorical_columns
        assert "status" in result.categorical_columns

    def test_blocked_category_binary(self):
        parser = self._make_parser()
        findings = self._make_findings()
        multi = self._make_multi(feature_exclusions=[
            {"column": "opened", "blocked_categories": ["binary"]},
        ])
        result = parser._build_aggregation_config(multi, findings, "events")
        assert "opened" not in result.binary_columns

    def test_blocked_category_does_not_affect_lifecycle(self):
        parser = self._make_parser()
        findings = self._make_findings()
        multi = self._make_multi(feature_exclusions=[
            {"column": "amount", "blocked_categories": ["aggregation"]},
        ])
        result = parser._build_aggregation_config(multi, findings, "events")
        assert result.windows == ["7d", "30d"]

    def test_blocked_funcs_populates_column_blocked_funcs(self):
        parser = self._make_parser()
        findings = self._make_findings()
        multi = self._make_multi(feature_exclusions=[
            {"column": "status", "blocked_funcs": ["mode"]},
        ])
        result = parser._build_aggregation_config(multi, findings, "events")
        assert result.column_blocked_funcs == {"status": ["mode"]}

    def test_multiple_exclusions_combined(self):
        parser = self._make_parser()
        findings = self._make_findings()
        multi = self._make_multi(feature_exclusions=[
            {"column": "amount", "blocked_categories": ["aggregation"]},
            {"column": "status", "blocked_funcs": ["mode"]},
        ])
        result = parser._build_aggregation_config(multi, findings, "events")
        assert "amount" not in result.value_columns
        assert result.column_blocked_funcs == {"status": ["mode"]}

    def test_no_exclusions_empty_blocked_funcs(self):
        parser = self._make_parser()
        findings = self._make_findings()
        multi = self._make_multi()
        result = parser._build_aggregation_config(multi, findings, "events")
        assert result.column_blocked_funcs == {}

    def test_unknown_column_ignored(self):
        parser = self._make_parser()
        findings = self._make_findings()
        multi = self._make_multi(feature_exclusions=[
            {"column": "nonexistent", "blocked_categories": ["aggregation"]},
        ])
        result = parser._build_aggregation_config(multi, findings, "events")
        assert "amount" in result.value_columns
        assert "count_col" in result.value_columns


class TestEventAggregatedColumnsWithExclusions:
    @staticmethod
    def _make_event_cfg(**overrides):
        from customer_retention.generators.pipeline_generator.models import BronzeEventConfig, SourceConfig
        src = SourceConfig(name="events", path="events.csv", format="csv",
                           entity_key="cid", raw_source_path="/data/events.csv")
        defaults = dict(source=src, entity_column="cid", time_column="ts")
        defaults.update(overrides)
        return BronzeEventConfig(**defaults)

    def test_blocked_funcs_excluded_from_output_columns(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import AggregationWindowConfig
        agg = AggregationWindowConfig(
            windows=["7d"], value_columns=["amount"],
            agg_funcs=["sum", "mean"],
            categorical_columns=["status"], categorical_agg_funcs=["nunique", "mode"],
            column_blocked_funcs={"status": ["mode"]},
        )
        cols = FindingsParser._event_aggregated_columns(self._make_event_cfg(aggregation=agg))
        assert "status_nunique_7d" in cols
        assert "status_mode_7d" not in cols
        assert "amount_sum_7d" in cols

    def test_blocked_category_columns_excluded(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import AggregationWindowConfig
        agg = AggregationWindowConfig(
            windows=["7d"], value_columns=["amount"],
            agg_funcs=["sum", "mean"],
            binary_columns=["opened"], binary_agg_funcs=["rate", "count", "any"],
            column_blocked_funcs={"opened": ["rate", "count", "any"]},
        )
        cols = FindingsParser._event_aggregated_columns(self._make_event_cfg(aggregation=agg))
        assert "opened_rate_7d" not in cols
        assert "opened_count_7d" not in cols
        assert "opened_any_7d" not in cols
        assert "amount_sum_7d" in cols


class TestDropL1ZeroAction(TestFeatureSelectionDropSkipsTarget):
    def test_drop_l1_zero_collected(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("age", "drop_l1_zero"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "age" in config.gold.feature_selections

    def test_drop_l1_zero_skips_target(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("unsubscribed", "drop_l1_zero"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "unsubscribed" not in config.gold.feature_selections

    def test_drop_l1_zero_ignores_prioritize(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("age", "prioritize"),
            self._make_rec("age", "drop_l1_zero"),
            self._make_rec("region", "drop_l1_zero"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "age" in config.gold.feature_selections
        assert "region" in config.gold.feature_selections

    def test_drop_chi_squared_collected(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("age", "drop_chi_squared"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "age" in config.gold.feature_selections

    def test_drop_chi_squared_skips_target(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("unsubscribed", "drop_chi_squared"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "unsubscribed" not in config.gold.feature_selections

    def test_drop_gbdt_importance_collected(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("region", "drop_gbdt_importance"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "region" in config.gold.feature_selections

    def test_drop_gbdt_importance_skips_target(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("unsubscribed", "drop_gbdt_importance"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "unsubscribed" not in config.gold.feature_selections

    def test_drop_rescue_consensus_collected(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("age", "drop_rescue_consensus"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "age" in config.gold.feature_selections

    def test_drop_rescue_consensus_skips_target(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("unsubscribed", "drop_rescue_consensus"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "unsubscribed" not in config.gold.feature_selections

    def test_prioritize_does_not_override_drop_multicollinear(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("age", "prioritize"),
            self._make_rec("age", "drop_multicollinear"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "age" in config.gold.feature_selections

    def test_gold_generated_column_drops_survive_after_transform_loop(self):
        parser = self._make_parser()
        config = self._make_config()
        rec_transform = self._make_rec("age", "zero_inflation_handling")
        rec_drop_is_zero = self._make_rec("age_is_zero", "drop_l1_zero")
        rec_drop_log = self._make_rec("age_log", "drop_l1_zero")
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            GoldRecommendations,
            RecommendationRegistry,
        )
        registry = RecommendationRegistry()
        registry.gold = GoldRecommendations(
            target_column="unsubscribed",
            feature_selection=[rec_drop_is_zero, rec_drop_log],
            transformations=[rec_transform],
        )
        # `age` is explicitly opted in to zero-inflation derivations so the
        # transform survives the opt-in gate; the test then verifies that the
        # downstream drop recommendations for the predicted columns persist.
        parser._apply_gold_recommendations(config, registry, ["age_"])
        assert "age_is_zero" in config.gold.feature_selections
        assert "age_log" in config.gold.feature_selections


class TestFeatureSelectionDropSkipsNonPipelineColumns(TestFeatureSelectionDropSkipsTarget):
    def test_drop_skips_column_not_in_pipeline(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("nonexistent_col", "drop_weak"),
            self._make_rec("age", "drop_weak"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "nonexistent_col" not in config.gold.feature_selections
        assert "age" in config.gold.feature_selections

    def test_reconcile_removes_stale_feature_selections(self):
        parser = self._make_parser()
        config = self._make_config()
        config.gold.feature_selections = ["age", "gone_column", "region"]
        parser._reconcile_gold_columns(config)
        assert "age" in config.gold.feature_selections
        assert "region" in config.gold.feature_selections
        assert "gone_column" not in config.gold.feature_selections

    def test_all_drop_actions_skip_nonexistent(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("phantom_a", "drop_weak"),
            self._make_rec("phantom_b", "drop_multicollinear"),
            self._make_rec("phantom_c", "drop_l1_zero"),
            self._make_rec("region", "drop_l1_zero"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert config.gold.feature_selections == ["region"]

    def test_drop_availability_collected(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("age", "drop_availability"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "age" in config.gold.feature_selections

    def test_drop_zero_variance_collected(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("region", "drop_zero_variance"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "region" in config.gold.feature_selections

    def test_drop_availability_skips_target(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("unsubscribed", "drop_availability"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert "unsubscribed" not in config.gold.feature_selections

    def test_drop_zero_variance_skips_nonexistent(self):
        parser = self._make_parser()
        config = self._make_config()
        registry = self._make_registry(feature_selection=[
            self._make_rec("phantom", "drop_zero_variance"),
            self._make_rec("age", "drop_zero_variance"),
        ])
        parser._apply_gold_recommendations(config, registry)
        assert config.gold.feature_selections == ["age"]


class TestPredictGoldGeneratedColumns(TestFeatureSelectionDropSkipsTarget):
    def _make_gold_step(self, column, step_type):
        from customer_retention.generators.pipeline_generator.models import (
            TransformationStep,
        )
        return TransformationStep(type=step_type, column=column, parameters={}, rationale="test", source_notebook="05")

    def test_zero_inflation_predicts_is_zero_and_log(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        config = self._make_config()
        config.gold.transformations = [self._make_gold_step("amount", PipelineTransformationType.ZERO_INFLATION_HANDLING)]
        result = FindingsParser._predict_gold_generated_columns(config)
        assert "amount_is_zero" in result
        assert "amount_log" in result

    def test_cap_then_log_predicts_nothing(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        config = self._make_config()
        config.gold.transformations = [self._make_gold_step("price", PipelineTransformationType.CAP_THEN_LOG)]
        result = FindingsParser._predict_gold_generated_columns(config)
        assert result == set()

    def test_empty_when_no_transforms(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        config = self._make_config()
        config.gold.transformations = []
        assert FindingsParser._predict_gold_generated_columns(config) == set()

    def test_reconcile_preserves_drops_for_gold_generated_columns(self):
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        parser = self._make_parser()
        config = self._make_config()
        config.gold.transformations = [self._make_gold_step("amount", PipelineTransformationType.ZERO_INFLATION_HANDLING)]
        config.gold.feature_selections = ["amount_is_zero", "amount_log"]
        parser._reconcile_gold_columns(config)
        assert "amount_is_zero" in config.gold.feature_selections
        assert "amount_log" in config.gold.feature_selections

    def test_apply_gold_preserves_is_zero_and_log_drop(self):
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        parser = self._make_parser()
        config = self._make_config()
        config.gold.transformations = [self._make_gold_step("age", PipelineTransformationType.ZERO_INFLATION_HANDLING)]
        registry = self._make_registry(feature_selection=[
            self._make_rec("age_is_zero", "drop_weak"),
            self._make_rec("age_log", "drop_weak"),
        ])
        # Note: this exercises the case where the zero_inflation transform was
        # added directly to the config (bypassing the recommendation gate), so
        # no opt-in is required — gating only applies to RECOMMENDATIONS being
        # promoted to steps, not to steps already in the config.
        parser._apply_gold_recommendations(config, registry)
        assert "age_is_zero" in config.gold.feature_selections
        assert "age_log" in config.gold.feature_selections


class TestZeroInflationOptIn:
    """Coverage for the NB05 opt-in gate that suppresses default _is_zero/_log."""

    @staticmethod
    def _make_findings(**kwargs):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        kwargs.setdefault("source_path", "/tmp/test.csv")
        kwargs.setdefault("source_format", "csv")
        return ExplorationFindings(**kwargs)

    def test_collect_prefixes_from_per_source_findings(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        f = self._make_findings(zero_inflation_opt_in=["FIRST_RESPONSE_TIME"])
        result = FindingsParser._collect_zero_inflation_opt_in_prefixes({"case": f})
        assert "FIRST_RESPONSE_TIME_" in result
        assert "FIRST_RESPONSE_TIME" in result

    def test_collect_prefixes_from_multi_dataset(self):
        from customer_retention.analysis.auto_explorer.exploration_manager import (
            DatasetInfo,
            MultiDatasetFindings,
        )
        from customer_retention.core.config.column_config import DatasetGranularity
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        info = DatasetInfo(
            name="case",
            findings_path="",
            source_path="",
            granularity=DatasetGranularity.ENTITY_LEVEL,
            row_count=10,
            column_count=2,
            zero_inflation_opt_in=["FIRST_RESPONSE_TIME"],
        )
        multi = MultiDatasetFindings(datasets={"case": info})
        result = FindingsParser._collect_zero_inflation_opt_in_prefixes({}, multi)
        assert "FIRST_RESPONSE_TIME_" in result

    def test_empty_when_no_opt_in(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        result = FindingsParser._collect_zero_inflation_opt_in_prefixes({})
        assert result == []

    def test_matches_any_prefix_base_and_derivations(self):
        from customer_retention.generators.pipeline_generator.findings_parser import _matches_any_prefix
        prefixes = ["FIRST_RESPONSE_TIME_", "FIRST_RESPONSE_TIME"]
        assert _matches_any_prefix("FIRST_RESPONSE_TIME", prefixes)
        assert _matches_any_prefix("FIRST_RESPONSE_TIME_sum_180d", prefixes)
        assert _matches_any_prefix("FIRST_RESPONSE_TIME_count_30d_is_zero", prefixes)
        assert _matches_any_prefix("lag2_FIRST_RESPONSE_TIME_sum", prefixes)
        assert _matches_any_prefix("velocity_FIRST_RESPONSE_TIME", prefixes)
        assert not _matches_any_prefix("NET_PRICE_sum_180d", prefixes)
        assert not _matches_any_prefix("FIRST_RESPONSE_TIMER_sum", prefixes)

    def test_gate_drops_zero_inflation_when_not_opted_in(self):
        # parser drops zero_inflation_handling recs by default; the predicted
        # _is_zero/_log columns therefore never enter the pipeline.
        parser = TestDropL1ZeroAction._make_parser()
        config = TestDropL1ZeroAction._make_config()
        rec_transform = TestDropL1ZeroAction._make_rec("age", "zero_inflation_handling")
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            GoldRecommendations,
            RecommendationRegistry,
        )
        registry = RecommendationRegistry()
        registry.gold = GoldRecommendations(
            target_column="unsubscribed",
            transformations=[rec_transform],
        )
        parser._apply_gold_recommendations(config, registry)
        assert config.gold.transformations == []

    def test_gate_keeps_zero_inflation_when_opted_in(self):
        parser = TestDropL1ZeroAction._make_parser()
        config = TestDropL1ZeroAction._make_config()
        rec_transform = TestDropL1ZeroAction._make_rec("age", "zero_inflation_handling")
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            GoldRecommendations,
            RecommendationRegistry,
        )
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        registry = RecommendationRegistry()
        registry.gold = GoldRecommendations(
            target_column="unsubscribed",
            transformations=[rec_transform],
        )
        parser._apply_gold_recommendations(config, registry, ["age_", "age"])
        assert any(
            step.type == PipelineTransformationType.ZERO_INFLATION_HANDLING
            and step.column == "age"
            for step in config.gold.transformations
        )

    def test_gate_does_not_affect_log_or_sqrt_recommendations(self):
        # Only zero_inflation_handling is gated; log/sqrt/yeo_johnson recs
        # still flow through unchanged.
        parser = TestDropL1ZeroAction._make_parser()
        config = TestDropL1ZeroAction._make_config()
        rec_log = TestDropL1ZeroAction._make_rec("age", "log_transform")
        from customer_retention.analysis.auto_explorer.layered_recommendations import (
            GoldRecommendations,
            RecommendationRegistry,
        )
        from customer_retention.generators.pipeline_generator.models import PipelineTransformationType
        registry = RecommendationRegistry()
        registry.gold = GoldRecommendations(
            target_column="unsubscribed",
            transformations=[rec_log],
        )
        parser._apply_gold_recommendations(config, registry)
        assert any(
            step.type == PipelineTransformationType.LOG_TRANSFORM and step.column == "age"
            for step in config.gold.transformations
        )


class TestLeakageExclusionPrefixes:
    @staticmethod
    def _make_findings(*, columns=None, **kwargs):
        from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
        from customer_retention.stages.modeling.feature_spec import LeakageExclusion
        kwargs.setdefault("source_path", "/tmp/test.csv")
        kwargs.setdefault("source_format", "csv")
        if columns is not None:
            kwargs["excluded_leaking_features"] = [LeakageExclusion(column=c) for c in columns]
        return ExplorationFindings(**kwargs)

    @staticmethod
    def _exclusions(*columns):
        from customer_retention.stages.modeling.feature_spec import LeakageExclusion
        return [LeakageExclusion(column=c) for c in columns]

    def test_collects_prefixes_from_source_findings(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings_a = self._make_findings(columns=["BILLING_TERMINATION_DATE"])
        findings_b = self._make_findings(columns=["CONTRACT_END_DATE"])
        result = FindingsParser._collect_leakage_exclusion_prefixes({"a": findings_a, "b": findings_b})
        assert "BILLING_TERMINATION_DATE_" in result
        assert "CONTRACT_END_DATE_" in result

    def test_empty_when_no_leaking_features(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings = self._make_findings()
        result = FindingsParser._collect_leakage_exclusion_prefixes({"a": findings})
        assert result == []

    def test_deduplicates_prefixes(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        findings_a = self._make_findings(columns=["COL_A"])
        findings_b = self._make_findings(columns=["COL_A", "COL_B"])
        result = FindingsParser._collect_leakage_exclusion_prefixes({"a": findings_a, "b": findings_b})
        assert result.count("COL_A_") == 1
        assert "COL_B_" in result

    def test_collects_prefixes_from_multi_dataset(self):
        from customer_retention.analysis.auto_explorer.exploration_manager import (
            DatasetInfo,
            MultiDatasetFindings,
        )
        from customer_retention.core.config.column_config import DatasetGranularity
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        multi = MultiDatasetFindings(
            datasets={
                "subscription": DatasetInfo(
                    name="subscription",
                    findings_path="/tmp/sub_findings.yaml",
                    source_path="/tmp/sub.csv",
                    granularity=DatasetGranularity.EVENT_LEVEL,
                    row_count=100,
                    column_count=10,
                    excluded_leaking_features=self._exclusions("SUBSCRIPTION_END_DATE"),
                ),
            },
        )
        findings = self._make_findings()
        result = FindingsParser._collect_leakage_exclusion_prefixes(
            {"subscription": findings}, multi
        )
        assert "SUBSCRIPTION_END_DATE_" in result

    def test_merges_prefixes_from_both_sources(self):
        from customer_retention.analysis.auto_explorer.exploration_manager import (
            DatasetInfo,
            MultiDatasetFindings,
        )
        from customer_retention.core.config.column_config import DatasetGranularity
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        multi = MultiDatasetFindings(
            datasets={
                "contract": DatasetInfo(
                    name="contract",
                    findings_path="/tmp/c_findings.yaml",
                    source_path="/tmp/c.csv",
                    granularity=DatasetGranularity.EVENT_LEVEL,
                    row_count=100,
                    column_count=10,
                    excluded_leaking_features=self._exclusions("CONTRACT_END_DATE"),
                ),
            },
        )
        findings = self._make_findings(columns=["BILLING_TERMINATION_DATE"])
        result = FindingsParser._collect_leakage_exclusion_prefixes(
            {"contract": findings}, multi
        )
        assert "CONTRACT_END_DATE_" in result
        assert "BILLING_TERMINATION_DATE_" in result


class TestFindLeakageExcludedColumns:

    def test_matches_direct_prefixes(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        columns = ["SUBSCRIPTION_END_DATE_delta_hours_sum", "NET_PRICE_mean", "entity_id"]
        result = FindingsParser.find_leakage_excluded_columns(columns, ["SUBSCRIPTION_END_DATE_"])
        assert result == ["SUBSCRIPTION_END_DATE_delta_hours_sum"]

    def test_matches_lag_and_velocity_variants(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        columns = ["lag1_SUBSCRIPTION_END_DATE_mean", "velocity_SUBSCRIPTION_END_DATE_sum", "lag2_safe_col"]
        result = FindingsParser.find_leakage_excluded_columns(columns, ["SUBSCRIPTION_END_DATE_"])
        assert "lag1_SUBSCRIPTION_END_DATE_mean" in result
        assert "velocity_SUBSCRIPTION_END_DATE_sum" in result
        assert "lag2_safe_col" not in result

    def test_empty_prefixes_returns_empty(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        assert FindingsParser.find_leakage_excluded_columns(["col_a", "col_b"], []) == []


class TestAggregationWindowsPriority:

    def _make_findings_with_windows(self, windows_used):
        from customer_retention.analysis.auto_explorer import ExplorationFindings
        from customer_retention.analysis.auto_explorer.findings import ColumnFinding, TimeSeriesMetadata
        from customer_retention.core.config.column_config import ColumnType, DatasetGranularity

        return ExplorationFindings(
            source_path="/data/events.csv", source_format="csv",
            row_count=1000, column_count=3,
            columns={
                "customer_id": ColumnFinding("customer_id", ColumnType.IDENTIFIER, 0.95, []),
                "amount": ColumnFinding("amount", ColumnType.NUMERIC_CONTINUOUS, 0.9, []),
                "event_date": ColumnFinding("event_date", ColumnType.DATETIME, 0.9, []),
            },
            time_series_metadata=TimeSeriesMetadata(
                granularity=DatasetGranularity.EVENT_LEVEL,
                entity_column="customer_id", time_column="event_date",
                aggregation_windows_used=windows_used,
            ),
        )

    def test_prefers_per_dataset_aggregation_windows_used(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings = self._make_findings_with_windows(["180d", "365d", "all_time"])

        class FakeMulti:
            aggregation_windows = ["24h", "7d", "30d", "90d", "180d", "365d", "all_time"]

        parser = FindingsParser.__new__(FindingsParser)
        result = parser._build_aggregation_config(FakeMulti(), findings)
        assert result is not None
        assert result.windows == ["180d", "365d", "all_time"]

    def test_raises_when_no_aggregation_windows_available(self):
        from customer_retention.analysis.auto_explorer import ExplorationFindings
        from customer_retention.analysis.auto_explorer.findings import ColumnFinding, TimeSeriesMetadata
        from customer_retention.core.config.column_config import ColumnType, DatasetGranularity
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings = ExplorationFindings(
            source_path="/data/events.csv", source_format="csv",
            row_count=1000, column_count=2,
            columns={
                "customer_id": ColumnFinding("customer_id", ColumnType.IDENTIFIER, 0.95, []),
                "event_date": ColumnFinding("event_date", ColumnType.DATETIME, 0.9, []),
            },
            time_series_metadata=TimeSeriesMetadata(
                granularity=DatasetGranularity.EVENT_LEVEL,
                entity_column="customer_id", time_column="event_date",
            ),
        )

        class FakeMulti:
            aggregation_windows = []

        parser = FindingsParser.__new__(FindingsParser)
        with pytest.raises(ValueError, match="aggregation_windows_used"):
            parser._build_aggregation_config(FakeMulti(), findings)

    def test_returns_none_for_non_event_without_any_windows(self):
        from customer_retention.analysis.auto_explorer import ExplorationFindings
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser

        findings = ExplorationFindings(
            source_path="/data/customers.csv", source_format="csv",
            row_count=100, column_count=2, columns={},
        )

        class FakeMulti:
            aggregation_windows = []

        parser = FindingsParser.__new__(FindingsParser)
        result = parser._build_aggregation_config(FakeMulti(), findings)
        assert result is None


class TestReconcileBronzeColumns:
    @staticmethod
    def _make_step(col, step_type):
        from customer_retention.generators.pipeline_generator.models import (
            PipelineTransformationType,
            TransformationStep,
        )
        return TransformationStep(
            type=PipelineTransformationType(step_type), column=col,
            parameters={}, rationale="test",
        )

    @staticmethod
    def _make_config(bronze_steps):
        from customer_retention.generators.pipeline_generator.models import (
            BronzeLayerConfig,
            GoldLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
        )
        source = SourceConfig(name="data", path="data.csv", format="csv", entity_key="cid", raw_source_path="/data.csv")
        return PipelineConfig(
            name="test", target_column="target", sources=[source],
            bronze={"data": BronzeLayerConfig(source=source, transformations=bronze_steps)},
            silver=SilverLayerConfig(), gold=GoldLayerConfig(), output_dir=".",
        )

    def test_removes_filter_on_dropped_column(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        steps = [
            self._make_step("bad_col", "drop_column"),
            self._make_step("good_col", "winsorize"),
            self._make_step("bad_col", "filter"),
        ]
        config = self._make_config(steps)
        FindingsParser._reconcile_bronze_columns(config)
        remaining = [(s.column, s.type.value) for s in config.bronze["data"].transformations]
        assert ("bad_col", "drop_column") in remaining
        assert ("good_col", "winsorize") in remaining
        assert ("bad_col", "filter") not in remaining

    def test_removes_winsorize_on_dropped_column(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        steps = [
            self._make_step("col_a", "drop_column"),
            self._make_step("col_a", "winsorize"),
            self._make_step("col_b", "impute_null"),
        ]
        config = self._make_config(steps)
        FindingsParser._reconcile_bronze_columns(config)
        types = [(s.column, s.type.value) for s in config.bronze["data"].transformations]
        assert ("col_a", "winsorize") not in types
        assert ("col_a", "drop_column") in types
        assert ("col_b", "impute_null") in types

    def test_no_drops_preserves_all_steps(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        steps = [
            self._make_step("x", "winsorize"),
            self._make_step("y", "filter"),
            self._make_step("z", "impute_null"),
        ]
        config = self._make_config(steps)
        FindingsParser._reconcile_bronze_columns(config)
        assert len(config.bronze["data"].transformations) == 3

    def test_multiple_dropped_columns(self):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        steps = [
            self._make_step("a", "drop_column"),
            self._make_step("b", "drop_column"),
            self._make_step("a", "cap_outlier"),
            self._make_step("b", "filter"),
            self._make_step("c", "winsorize"),
        ]
        config = self._make_config(steps)
        FindingsParser._reconcile_bronze_columns(config)
        remaining = [(s.column, s.type.value) for s in config.bronze["data"].transformations]
        assert len(remaining) == 3
        assert ("a", "drop_column") in remaining
        assert ("b", "drop_column") in remaining
        assert ("c", "winsorize") in remaining


class TestCollectKnownPipelineColumns:
    """Parity check uses `_collect_known_pipeline_columns` to validate selected_features.

    It must cover every column the generated pipeline will actually produce —
    raw bronze, event-aggregated (temporal/lifecycle), silver DERIVED_COLUMN
    outputs, and gold zero-inflation derivatives. Missing any of these causes
    spurious `FeatureSpec parity violation` errors at generation time.
    """

    @staticmethod
    def _parser_with(event_cfg=None, silver_derived=None, gold_transforms=None, raw_source_columns=None):
        from customer_retention.generators.pipeline_generator.findings_parser import FindingsParser
        from customer_retention.generators.pipeline_generator.models import (
            BronzeLayerConfig,
            GoldLayerConfig,
            PipelineConfig,
            SilverLayerConfig,
            SourceConfig,
        )
        entity_src = SourceConfig(name="cust", path="c.csv", format="csv", entity_key="cid", raw_source_path="/c.csv")
        sources = [entity_src]
        bronze_events = {}
        if event_cfg is not None:
            event_src = SourceConfig(
                name="evt", path="e.csv", format="csv", entity_key="cid",
                raw_source_path="/e.csv", is_event_level=True, time_column="ts",
            )
            sources.append(event_src)
            event_cfg.source = event_src
            bronze_events["evt"] = event_cfg
        config = PipelineConfig(
            name="t", target_column="churn", sources=sources,
            bronze={"cust": BronzeLayerConfig(source=entity_src)},
            silver=SilverLayerConfig(derived_columns=list(silver_derived or [])),
            gold=GoldLayerConfig(transformations=list(gold_transforms or [])),
            output_dir=".",
        )
        config.bronze_event = bronze_events
        parser = FindingsParser.__new__(FindingsParser)
        parser._raw_source_columns = dict(raw_source_columns or {"cust": {"cid"}})
        parser._source_findings_paths = {}
        return parser, config

    def test_temporal_regularity_group_included(self):
        """Regression: `regularity_score`, `event_frequency`, `inter_event_gap_max`
        are emitted by event bronze via `TemporalFeatureConfig.regularity` group.
        The parity check must recognize them as pipeline columns.
        """
        from customer_retention.generators.pipeline_generator.models import (
            BronzeEventConfig,
            TemporalFeatureConfig,
        )
        event_cfg = BronzeEventConfig(
            source=None, entity_column="cid", time_column="ts",
            temporal_features=TemporalFeatureConfig(
                lag_columns=["amt"], num_lags=1, lag_agg_funcs=["sum"],
                feature_groups=["regularity"],
            ),
        )
        parser, config = self._parser_with(event_cfg=event_cfg)
        cols = parser._collect_known_pipeline_columns(config)
        assert {"event_frequency", "inter_event_gap_max", "regularity_score",
                "inter_event_gap_mean", "inter_event_gap_std"} <= cols

    def test_temporal_recency_group_included(self):
        from customer_retention.generators.pipeline_generator.models import (
            BronzeEventConfig,
            TemporalFeatureConfig,
        )
        event_cfg = BronzeEventConfig(
            source=None, entity_column="cid", time_column="ts",
            temporal_features=TemporalFeatureConfig(
                lag_columns=["amt"], num_lags=1, lag_agg_funcs=["sum"],
                feature_groups=["recency"],
            ),
        )
        parser, config = self._parser_with(event_cfg=event_cfg)
        cols = parser._collect_known_pipeline_columns(config)
        assert {"days_since_last_event", "days_since_first_event",
                "active_span_days", "recency_ratio"} <= cols

    def test_event_aggregated_columns_included(self):
        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
        )
        event_cfg = BronzeEventConfig(
            source=None, entity_column="cid", time_column="ts",
            aggregation=AggregationWindowConfig(
                windows=["30d", "180d"], value_columns=["amount"],
                agg_funcs=["sum", "mean"],
            ),
        )
        parser, config = self._parser_with(event_cfg=event_cfg)
        cols = parser._collect_known_pipeline_columns(config)
        assert {"amount_sum_30d", "amount_mean_30d", "amount_sum_180d",
                "amount_mean_180d", "event_count_30d", "event_count_180d"} <= cols

    def test_silver_derived_ratio_column_included(self):
        """Regression: ratio features added to `silver.derived_columns` must
        appear as pipeline columns. These are names like
        `event_count_180d_to_days_since_last_event_x_ratio`.
        """
        from customer_retention.generators.pipeline_generator.models import (
            PipelineTransformationType,
            TransformationStep,
        )
        ratio_step = TransformationStep(
            type=PipelineTransformationType.DERIVED_COLUMN,
            column="event_count_180d_to_days_since_last_event_x_ratio",
            parameters={"action": "ratio", "numerator": "event_count_180d", "denominator": "days_since_last_event"},
            rationale="high-mutual-info ratio",
        )
        parser, config = self._parser_with(silver_derived=[ratio_step])
        cols = parser._collect_known_pipeline_columns(config)
        assert "event_count_180d_to_days_since_last_event_x_ratio" in cols

    def test_silver_derived_interaction_and_composite_included(self):
        from customer_retention.generators.pipeline_generator.models import (
            PipelineTransformationType,
            TransformationStep,
        )
        steps = [
            TransformationStep(
                type=PipelineTransformationType.DERIVED_COLUMN,
                column="amt_x_qty", parameters={"action": "interaction"}, rationale="",
            ),
            TransformationStep(
                type=PipelineTransformationType.DERIVED_COLUMN,
                column="engagement_composite", parameters={"action": "composite"}, rationale="",
            ),
        ]
        parser, config = self._parser_with(silver_derived=steps)
        cols = parser._collect_known_pipeline_columns(config)
        assert {"amt_x_qty", "engagement_composite"} <= cols

    def test_gold_zero_inflation_derivatives_included(self):
        from customer_retention.generators.pipeline_generator.models import (
            PipelineTransformationType,
            TransformationStep,
        )
        zi_step = TransformationStep(
            type=PipelineTransformationType.ZERO_INFLATION_HANDLING,
            column="amount", parameters={}, rationale="",
        )
        parser, config = self._parser_with(gold_transforms=[zi_step])
        cols = parser._collect_known_pipeline_columns(config)
        assert {"amount_is_zero", "amount_log"} <= cols

    def test_enforce_parity_accepts_regularity_and_ratio_features(self, tmp_path):
        """End-to-end: parity check does NOT raise when selected_features
        reference temporal `regularity` features and silver ratio outputs.
        """
        from customer_retention.generators.pipeline_generator.models import (
            BronzeEventConfig,
            PipelineTransformationType,
            TemporalFeatureConfig,
            TransformationStep,
        )
        from customer_retention.stages.modeling.feature_spec import FeatureSpec, FittedTransform

        event_cfg = BronzeEventConfig(
            source=None, entity_column="cid", time_column="ts",
            temporal_features=TemporalFeatureConfig(
                lag_columns=["amt"], num_lags=1, lag_agg_funcs=["sum"],
                feature_groups=["regularity"],
            ),
        )
        ratio_step = TransformationStep(
            type=PipelineTransformationType.DERIVED_COLUMN,
            column="event_count_180d_to_days_since_last_event_x_ratio",
            parameters={"action": "ratio", "numerator": "event_count_180d", "denominator": "days_since_last_event"},
            rationale="",
        )
        parser, config = self._parser_with(event_cfg=event_cfg, silver_derived=[ratio_step])
        selected = [
            "regularity_score", "event_frequency", "inter_event_gap_max",
            "event_count_180d_to_days_since_last_event_x_ratio",
        ]
        parser._feature_spec = FeatureSpec(
            exploration_run_id="r", target_column="churn",
            entity_column="entity_id", timestamp_column="as_of_date",
            horizon_days=30, selected_features=selected,
            fitted_transforms=[FittedTransform(column=c, action="impute", method="median") for c in selected],
        )
        parser._enforce_spec_schema_parity(config)  # must not raise
