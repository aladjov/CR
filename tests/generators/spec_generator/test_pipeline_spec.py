import json

import pytest
import yaml

from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.core.config.column_config import ColumnType
from customer_retention.generators.spec_generator.pipeline_spec import (
    ColumnSpec,
    FeatureSpec,
    ModelSpec,
    PipelineSpec,
    QualityGateSpec,
    SchemaSpec,
    SourceSpec,
    TransformSpec,
)


@pytest.fixture
def sample_findings() -> ExplorationFindings:
    columns = {
        "customer_id": ColumnFinding(
            name="customer_id",
            inferred_type=ColumnType.IDENTIFIER,
            confidence=0.95,
            evidence=["All unique"]
        ),
        "age": ColumnFinding(
            name="age",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.85,
            evidence=["Numeric"]
        ),
        "contract_type": ColumnFinding(
            name="contract_type",
            inferred_type=ColumnType.CATEGORICAL_NOMINAL,
            confidence=0.9,
            evidence=["Categorical"]
        ),
        "churned": ColumnFinding(
            name="churned",
            inferred_type=ColumnType.TARGET,
            confidence=0.9,
            evidence=["Binary target"]
        )
    }
    return ExplorationFindings(
        source_path="data/customers.csv",
        source_format="csv",
        row_count=10000,
        column_count=4,
        columns=columns,
        target_column="churned",
        target_type="binary",
        identifier_columns=["customer_id"]
    )


class TestSourceSpec:
    def test_creation(self):
        spec = SourceSpec(
            name="customer_data",
            path="data/customers.csv",
            format="csv",
            options={"header": True}
        )
        assert spec.name == "customer_data"
        assert spec.format == "csv"

    def test_to_dict(self):
        spec = SourceSpec(name="test", path="/path", format="parquet")
        data = spec.to_dict()
        assert data["name"] == "test"
        assert data["format"] == "parquet"


class TestColumnSpec:
    def test_creation(self):
        spec = ColumnSpec(
            name="age",
            data_type="integer",
            semantic_type="numeric_continuous",
            nullable=False
        )
        assert spec.name == "age"
        assert spec.semantic_type == "numeric_continuous"

    def test_from_column_finding(self):
        finding = ColumnFinding(
            name="age",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.9,
            evidence=[],
            universal_metrics={"null_count": 0}
        )
        spec = ColumnSpec.from_column_finding(finding)
        assert spec.name == "age"
        assert spec.semantic_type == "numeric_continuous"


class TestSchemaSpec:
    def test_creation(self):
        columns = [
            ColumnSpec(name="id", data_type="string", semantic_type="identifier"),
            ColumnSpec(name="value", data_type="float", semantic_type="numeric_continuous")
        ]
        spec = SchemaSpec(columns=columns, primary_key="id")
        assert len(spec.columns) == 2
        assert spec.primary_key == "id"

    def test_from_findings(self, sample_findings):
        spec = SchemaSpec.from_findings(sample_findings)
        assert len(spec.columns) == 4
        assert spec.primary_key == "customer_id"


class TestTransformSpec:
    def test_creation(self):
        spec = TransformSpec(
            name="scale_age",
            transform_type="standard_scaling",
            input_columns=["age"],
            output_columns=["age_scaled"],
            parameters={"with_mean": True}
        )
        assert spec.name == "scale_age"
        assert spec.transform_type == "standard_scaling"

    def test_to_dict(self):
        spec = TransformSpec(
            name="encode_contract",
            transform_type="one_hot",
            input_columns=["contract_type"],
            output_columns=["contract_type_encoded"]
        )
        data = spec.to_dict()
        assert data["transform_type"] == "one_hot"


class TestFeatureSpec:
    def test_creation(self):
        spec = FeatureSpec(
            name="customer_tenure_months",
            source_columns=["signup_date"],
            computation="months_since",
            description="Months since customer signed up"
        )
        assert spec.name == "customer_tenure_months"
        assert spec.computation == "months_since"

    def test_to_dict(self):
        spec = FeatureSpec(
            name="total_spend",
            source_columns=["monthly_charges", "tenure"],
            computation="multiply"
        )
        data = spec.to_dict()
        assert "source_columns" in data


class TestModelSpec:
    def test_creation(self):
        spec = ModelSpec(
            name="churn_classifier",
            model_type="gradient_boosting",
            target_column="churned",
            feature_columns=["age", "tenure", "monthly_charges"],
            hyperparameters={"n_estimators": 100}
        )
        assert spec.name == "churn_classifier"
        assert spec.model_type == "gradient_boosting"

    def test_to_dict(self):
        spec = ModelSpec(
            name="model",
            model_type="random_forest",
            target_column="target",
            feature_columns=["f1", "f2"]
        )
        data = spec.to_dict()
        assert data["model_type"] == "random_forest"


class TestQualityGateSpec:
    def test_creation(self):
        spec = QualityGateSpec(
            name="null_check",
            gate_type="null_percentage",
            column="age",
            threshold=5.0,
            action="warn"
        )
        assert spec.name == "null_check"
        assert spec.threshold == 5.0

    def test_to_dict(self):
        spec = QualityGateSpec(
            name="check",
            gate_type="drift",
            column="*",
            threshold=0.1
        )
        data = spec.to_dict()
        assert data["gate_type"] == "drift"


class TestPipelineSpec:
    def test_creation(self):
        spec = PipelineSpec(
            name="customer_churn_pipeline",
            version="1.0.0"
        )
        assert spec.name == "customer_churn_pipeline"
        assert spec.version == "1.0.0"

    def test_from_findings(self, sample_findings):
        spec = PipelineSpec.from_findings(sample_findings, name="churn_pipeline")
        assert spec.name == "churn_pipeline"
        assert spec.schema is not None
        assert len(spec.sources) > 0

    def test_to_dict(self, sample_findings):
        spec = PipelineSpec.from_findings(sample_findings)
        data = spec.to_dict()
        assert "name" in data
        assert "schema" in data
        assert "sources" in data

    def test_to_json(self, sample_findings):
        spec = PipelineSpec.from_findings(sample_findings)
        json_str = spec.to_json()
        parsed = json.loads(json_str)
        assert "name" in parsed

    def test_to_yaml(self, sample_findings):
        spec = PipelineSpec.from_findings(sample_findings)
        yaml_str = spec.to_yaml()
        parsed = yaml.safe_load(yaml_str)
        assert "name" in parsed

    def test_save_and_load_json(self, sample_findings, tmp_path):
        spec = PipelineSpec.from_findings(sample_findings, name="test_pipeline")
        path = tmp_path / "spec.json"
        spec.save(str(path))

        loaded = PipelineSpec.load(str(path))
        assert loaded.name == "test_pipeline"

    def test_save_and_load_yaml(self, sample_findings, tmp_path):
        spec = PipelineSpec.from_findings(sample_findings, name="test_pipeline")
        path = tmp_path / "spec.yaml"
        spec.save(str(path))

        loaded = PipelineSpec.load(str(path))
        assert loaded.name == "test_pipeline"

    def test_add_transform(self, sample_findings):
        spec = PipelineSpec.from_findings(sample_findings)
        transform = TransformSpec(
            name="custom_transform",
            transform_type="custom",
            input_columns=["age"],
            output_columns=["age_transformed"]
        )
        spec.add_transform(transform, stage="silver")
        assert any(t.name == "custom_transform" for t in spec.silver_transforms)

    def test_add_feature(self, sample_findings):
        spec = PipelineSpec.from_findings(sample_findings)
        feature = FeatureSpec(
            name="custom_feature",
            source_columns=["age"],
            computation="square"
        )
        spec.add_feature(feature)
        assert any(f.name == "custom_feature" for f in spec.feature_definitions)

    def test_add_quality_gate(self, sample_findings):
        spec = PipelineSpec.from_findings(sample_findings)
        gate = QualityGateSpec(
            name="custom_gate",
            gate_type="range_check",
            column="age",
            threshold=0
        )
        spec.add_quality_gate(gate)
        assert any(g.name == "custom_gate" for g in spec.quality_gates)
