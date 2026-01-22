import json


class TestPipelineSpecAddMethods:
    def test_add_transform_to_bronze(self):
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec, TransformSpec
        spec = PipelineSpec(name="test")
        transform = TransformSpec("test_t", "scaling", ["x"], ["y"])
        spec.add_transform(transform, stage="bronze")
        assert len(spec.bronze_transforms) == 1
        assert spec.bronze_transforms[0].name == "test_t"

    def test_add_transform_to_silver(self):
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec, TransformSpec
        spec = PipelineSpec(name="test")
        transform = TransformSpec("test_t", "scaling", ["x"], ["y"])
        spec.add_transform(transform, stage="silver")
        assert len(spec.silver_transforms) == 1

    def test_add_transform_to_gold(self):
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec, TransformSpec
        spec = PipelineSpec(name="test")
        transform = TransformSpec("test_t", "scaling", ["x"], ["y"])
        spec.add_transform(transform, stage="gold")
        assert len(spec.gold_transforms) == 1

    def test_add_feature(self):
        from customer_retention.generators.spec_generator.pipeline_spec import FeatureSpec, PipelineSpec
        spec = PipelineSpec(name="test")
        feature = FeatureSpec("new_feature", ["col1"], "sum")
        spec.add_feature(feature)
        assert len(spec.feature_definitions) == 1

    def test_add_quality_gate(self):
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec, QualityGateSpec
        spec = PipelineSpec(name="test")
        gate = QualityGateSpec("null_check", "null_percentage", "col1", 5.0)
        spec.add_quality_gate(gate)
        assert len(spec.quality_gates) == 1


class TestPipelineSpecSerialization:
    def test_to_json_and_back(self, tmp_path):
        from customer_retention.generators.spec_generator.pipeline_spec import (
            PipelineSpec,
            QualityGateSpec,
            SourceSpec,
            TransformSpec,
        )
        spec = PipelineSpec(name="json_test", version="2.0.0")
        spec.sources.append(SourceSpec("src", "/data/test.csv", "csv"))
        spec.silver_transforms.append(TransformSpec("t1", "scale", ["x"], ["y"]))
        spec.quality_gates.append(QualityGateSpec("gate1", "null", "col", 5.0))
        json_path = tmp_path / "spec.json"
        spec.save(str(json_path))
        loaded = PipelineSpec.load(str(json_path))
        assert loaded.name == "json_test"
        assert loaded.version == "2.0.0"
        assert len(loaded.sources) == 1
        assert len(loaded.silver_transforms) == 1
        assert len(loaded.quality_gates) == 1

    def test_to_yaml_and_back(self, tmp_path):
        from customer_retention.generators.spec_generator.pipeline_spec import (
            ColumnSpec,
            PipelineSpec,
            SchemaSpec,
            SourceSpec,
        )
        spec = PipelineSpec(name="yaml_test")
        spec.sources.append(SourceSpec("src", "/data/test.csv", "csv"))
        spec.schema = SchemaSpec(
            columns=[ColumnSpec("col1", "string", "identifier")],
            primary_key="col1",
            partition_columns=["date"]
        )
        yaml_path = tmp_path / "spec.yaml"
        spec.save(str(yaml_path))
        loaded = PipelineSpec.load(str(yaml_path))
        assert loaded.name == "yaml_test"
        assert loaded.schema is not None
        assert loaded.schema.primary_key == "col1"
        assert loaded.schema.partition_columns == ["date"]

    def test_load_with_bronze_and_gold_transforms(self, tmp_path):
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec, TransformSpec
        spec = PipelineSpec(name="all_layers")
        spec.bronze_transforms.append(TransformSpec("b1", "validate", ["x"], ["y"]))
        spec.gold_transforms.append(TransformSpec("g1", "aggregate", ["a"], ["b"]))
        json_path = tmp_path / "spec.json"
        spec.save(str(json_path))
        loaded = PipelineSpec.load(str(json_path))
        assert len(loaded.bronze_transforms) == 1
        assert len(loaded.gold_transforms) == 1

    def test_load_with_feature_definitions(self, tmp_path):
        from customer_retention.generators.spec_generator.pipeline_spec import FeatureSpec, PipelineSpec
        spec = PipelineSpec(name="with_features")
        spec.feature_definitions.append(FeatureSpec("f1", ["col1"], "sum"))
        json_path = tmp_path / "spec.json"
        spec.save(str(json_path))
        loaded = PipelineSpec.load(str(json_path))
        assert len(loaded.feature_definitions) == 1
        assert loaded.feature_definitions[0].name == "f1"

    def test_load_with_model_config(self, tmp_path):
        from customer_retention.generators.spec_generator.pipeline_spec import ModelSpec, PipelineSpec
        spec = PipelineSpec(name="with_model")
        spec.model_config = ModelSpec("model1", "xgboost", "target", ["f1", "f2"])
        json_path = tmp_path / "spec.json"
        spec.save(str(json_path))
        loaded = PipelineSpec.load(str(json_path))
        assert loaded.model_config is not None
        assert loaded.model_config.name == "model1"


class TestPipelineSpecFromDict:
    def test_from_dict_with_metadata(self, tmp_path):
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec
        data = {
            "name": "metadata_test",
            "version": "1.0.0",
            "description": "Test",
            "metadata": {"author": "test", "env": "dev"}
        }
        json_path = tmp_path / "spec.json"
        with open(json_path, "w") as f:
            json.dump(data, f)
        loaded = PipelineSpec.load(str(json_path))
        assert loaded.metadata == {"author": "test", "env": "dev"}

    def test_from_dict_without_optional_fields(self, tmp_path):
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec
        data = {"name": "minimal"}
        json_path = tmp_path / "spec.json"
        with open(json_path, "w") as f:
            json.dump(data, f)
        loaded = PipelineSpec.load(str(json_path))
        assert loaded.name == "minimal"
        assert loaded.schema is None
        assert loaded.model_config is None


class TestPipelineSpecFromFindingsEdgeCases:
    def test_from_findings_without_target_column(self):
        from dataclasses import dataclass, field
        from typing import Any, Dict, List

        from customer_retention.core.config import ColumnType
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec

        @dataclass
        class MockCol:
            name: str
            inferred_type: ColumnType
            universal_metrics: Dict[str, Any] = field(default_factory=dict)

        @dataclass
        class MockFindings:
            source_path: str = "/data/test.csv"
            source_format: str = "csv"
            target_column: str = None
            identifier_columns: List[str] = field(default_factory=list)
            datetime_columns: List[str] = field(default_factory=list)
            columns: Dict[str, MockCol] = field(default_factory=dict)

        findings = MockFindings()
        findings.columns = {"age": MockCol("age", ColumnType.NUMERIC_CONTINUOUS)}
        spec = PipelineSpec.from_findings(findings)
        assert spec.model_config is None

    def test_from_findings_without_datetime_columns(self):
        from dataclasses import dataclass, field
        from typing import Any, Dict, List

        from customer_retention.core.config import ColumnType
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec

        @dataclass
        class MockCol:
            name: str
            inferred_type: ColumnType
            universal_metrics: Dict[str, Any] = field(default_factory=dict)

        @dataclass
        class MockFindings:
            source_path: str = "/data/test.csv"
            source_format: str = "csv"
            target_column: str = "target"
            identifier_columns: List[str] = field(default_factory=list)
            datetime_columns: List[str] = field(default_factory=list)
            columns: Dict[str, MockCol] = field(default_factory=dict)

        findings = MockFindings()
        findings.columns = {
            "target": MockCol("target", ColumnType.TARGET),
            "age": MockCol("age", ColumnType.NUMERIC_CONTINUOUS)
        }
        spec = PipelineSpec.from_findings(findings)
        assert len(spec.feature_definitions) == 0


class TestSchemaSpecFromFindings:
    def test_creates_schema_with_identifier_column(self):
        from dataclasses import dataclass, field
        from typing import Any, Dict

        from customer_retention.core.config import ColumnType
        from customer_retention.generators.spec_generator.pipeline_spec import SchemaSpec

        @dataclass
        class MockCol:
            name: str
            inferred_type: ColumnType
            universal_metrics: Dict[str, Any] = field(default_factory=dict)

        @dataclass
        class MockFindings:
            identifier_columns: list
            columns: dict

        findings = MockFindings(
            identifier_columns=["user_id"],
            columns={
                "user_id": MockCol("user_id", ColumnType.IDENTIFIER),
                "age": MockCol("age", ColumnType.NUMERIC_CONTINUOUS, {"null_count": 5}),
            }
        )
        schema = SchemaSpec.from_findings(findings)
        assert schema.primary_key == "user_id"
        assert len(schema.columns) == 2

    def test_creates_schema_without_identifier(self):
        from dataclasses import dataclass, field
        from typing import Any, Dict

        from customer_retention.core.config import ColumnType
        from customer_retention.generators.spec_generator.pipeline_spec import SchemaSpec

        @dataclass
        class MockCol:
            name: str
            inferred_type: ColumnType
            universal_metrics: Dict[str, Any] = field(default_factory=dict)

        @dataclass
        class MockFindings:
            identifier_columns: list
            columns: dict

        findings = MockFindings(
            identifier_columns=[],
            columns={"age": MockCol("age", ColumnType.NUMERIC_CONTINUOUS)}
        )
        schema = SchemaSpec.from_findings(findings)
        assert schema.primary_key is None
