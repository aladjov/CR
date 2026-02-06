import pandas as pd
import pytest
import yaml

from customer_retention.generators.pipeline_generator.models import (
    BronzeLayerConfig,
    GoldLayerConfig,
    PipelineTransformationType,
    SilverLayerConfig,
    SourceConfig,
    TransformationStep,
)


@pytest.fixture
def entity_source():
    return SourceConfig(
        name="customers",
        path="/data/customers.csv",
        format="csv",
        entity_key="customer_id",
    )


@pytest.fixture
def event_source():
    return SourceConfig(
        name="orders",
        path="/data/orders.parquet",
        format="parquet",
        entity_key="customer_id",
        time_column="order_date",
        is_event_level=True,
    )


@pytest.fixture
def bronze_with_impute(entity_source):
    return BronzeLayerConfig(
        source=entity_source,
        transformations=[
            TransformationStep(
                type=PipelineTransformationType.IMPUTE_NULL,
                column="age",
                parameters={"value": 0},
                rationale="Fill nulls",
            ),
        ],
    )


@pytest.fixture
def silver_with_join():
    return SilverLayerConfig(
        joins=[
            {
                "left_key": "customer_id",
                "right_key": "customer_id",
                "right_source": "orders",
                "how": "left",
            }
        ],
        aggregations=[],
    )


@pytest.fixture
def gold_with_encode_scale():
    return GoldLayerConfig(
        encodings=[
            TransformationStep(
                type=PipelineTransformationType.ENCODE,
                column="category",
                parameters={"method": "one_hot"},
                rationale="Encode",
            ),
        ],
        scalings=[
            TransformationStep(
                type=PipelineTransformationType.SCALE,
                column="amount",
                parameters={"method": "standard"},
                rationale="Scale",
            ),
        ],
    )


@pytest.fixture
def experiments_setup(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    experiments_dir = project_root / "experiments"
    experiments_dir.mkdir()
    findings_dir = experiments_dir / "findings"
    findings_dir.mkdir()
    output_dir = project_root / "generated_pipelines"
    output_dir.mkdir()

    (project_root / "pyproject.toml").write_text("[project]\nname = 'test'\n")

    test_df = pd.DataFrame({
        "customer_id": ["A001", "A002", "A003", "A004", "A005"],
        "revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
        "orders": [5, 10, 7, 15, 12],
        "target": [0, 1, 0, 1, 0],
    })
    data_path = findings_dir / "customers.parquet"
    test_df.to_parquet(data_path, index=False)

    multi_dataset = {
        "datasets": {
            "customers": {
                "name": "customers",
                "findings_path": "customers_findings.yaml",
                "source_path": str(data_path),
                "granularity": "entity_level",
                "row_count": 5,
                "column_count": 4,
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
    (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers_findings))

    return {
        "project_root": project_root,
        "experiments_dir": experiments_dir,
        "findings_dir": findings_dir,
        "output_dir": output_dir,
        "data_path": data_path,
        "tmp_path": tmp_path,
    }
