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
        impute_steps = [t for t in bronze_event.post_shaping if t.type == PipelineTransformationType.IMPUTE_NULL]
        assert len(impute_steps) == 1
        assert impute_steps[0].column == "total_amount"
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

    def test_pre_shaping_allows_all_when_no_raw_index(self):
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
        assert "velocity_pct" in pre_cols

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
