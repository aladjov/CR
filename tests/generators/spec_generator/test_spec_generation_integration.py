from pathlib import Path

import pytest


class TestSpecGenerationIntegration:
    @pytest.fixture
    def sample_findings(self):
        from dataclasses import dataclass, field
        from typing import Any, Dict, List

        from customer_retention.core.config import ColumnType

        @dataclass
        class MockCol:
            name: str
            inferred_type: ColumnType
            universal_metrics: Dict[str, Any] = field(default_factory=dict)

        @dataclass
        class MockFindings:
            source_path: str = "/data/customers.csv"
            source_format: str = "csv"
            target_column: str = "churned"
            identifier_columns: List[str] = field(default_factory=lambda: ["customer_id"])
            datetime_columns: List[str] = field(default_factory=lambda: ["signup_date"])
            columns: Dict[str, MockCol] = field(default_factory=dict)

        findings = MockFindings()
        findings.columns = {
            "customer_id": MockCol("customer_id", ColumnType.IDENTIFIER),
            "churned": MockCol("churned", ColumnType.TARGET),
            "age": MockCol("age", ColumnType.NUMERIC_CONTINUOUS, {"null_count": 10}),
            "gender": MockCol("gender", ColumnType.CATEGORICAL_NOMINAL),
            "signup_date": MockCol("signup_date", ColumnType.DATETIME),
        }
        return findings

    def test_create_spec_from_findings_and_generate_all(self, sample_findings, tmp_path):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec
        spec = PipelineSpec.from_findings(sample_findings, name="churn_pipeline")
        assert spec.name == "churn_pipeline"
        assert spec.model_config is not None
        assert spec.model_config.target_column == "churned"
        assert len(spec.silver_transforms) > 0
        assert len(spec.feature_definitions) > 0
        db_generator = DatabricksSpecGenerator(output_dir=str(tmp_path / "databricks"))
        db_files = db_generator.save_all(spec)
        assert len(db_files) == 6
        generic_generator = GenericSpecGenerator(output_dir=str(tmp_path / "generic"))
        generic_files = generic_generator.save_all(spec)
        assert len(generic_files) == 6
        dlt_content = (tmp_path / "databricks" / "churn_pipeline_dlt_pipeline.py").read_text()
        assert "dlt.table" in dlt_content
        pipeline_content = (tmp_path / "generic" / "pipeline.py").read_text()
        assert "def load_data" in pipeline_content
        assert "def train_model" in pipeline_content

    def test_spec_save_load_roundtrip(self, sample_findings, tmp_path):
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec
        spec = PipelineSpec.from_findings(sample_findings, name="roundtrip_test")
        yaml_path = tmp_path / "spec.yaml"
        spec.save(str(yaml_path))
        loaded = PipelineSpec.load(str(yaml_path))
        assert loaded.name == spec.name
        assert loaded.model_config.target_column == spec.model_config.target_column
        assert len(loaded.silver_transforms) == len(spec.silver_transforms)
        assert len(loaded.feature_definitions) == len(spec.feature_definitions)

    def test_generated_code_compiles(self, sample_findings, tmp_path):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec
        spec = PipelineSpec.from_findings(sample_findings)
        db_generator = DatabricksSpecGenerator()
        artifacts = db_generator.generate_all(spec)
        compile(artifacts["dlt_pipeline"], "<dlt>", "exec")
        compile(artifacts["feature_tables"], "<feature_tables>", "exec")
        compile(artifacts["mlflow_experiment"], "<mlflow>", "exec")
        generic_generator = GenericSpecGenerator()
        generic_artifacts = generic_generator.generate_all(spec)
        compile(generic_artifacts["pipeline.py"], "<pipeline>", "exec")
        compile(generic_artifacts["dag.py"], "<dag>", "exec")
        compile(generic_artifacts["flow.py"], "<flow>", "exec")


class TestSpecGeneratorEndToEnd:
    def test_full_databricks_artifact_generation(self, tmp_path):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
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
        spec = PipelineSpec(
            name="e2e_test",
            version="2.0.0",
            description="End-to-end test pipeline"
        )
        spec.sources.append(SourceSpec("customers", "/mnt/data/customers.parquet", "parquet"))
        spec.schema = SchemaSpec(
            columns=[
                ColumnSpec("customer_id", "string", "identifier", nullable=False),
                ColumnSpec("age", "float", "numeric_continuous"),
                ColumnSpec("tier", "string", "categorical_ordinal"),
                ColumnSpec("churned", "integer", "target"),
            ],
            primary_key="customer_id"
        )
        spec.silver_transforms = [
            TransformSpec("scale_age", "standard_scaling", ["age"], ["age_scaled"]),
            TransformSpec("encode_tier", "one_hot_encoding", ["tier"], ["tier_encoded"]),
        ]
        spec.feature_definitions = [
            FeatureSpec("age_group", ["age"], "bucket", parameters={"bins": [0, 30, 60, 100]}),
        ]
        spec.model_config = ModelSpec(
            name="churn_predictor",
            model_type="gradient_boosting",
            target_column="churned",
            feature_columns=["age_scaled", "tier_encoded"],
            hyperparameters={"max_depth": 6, "learning_rate": 0.1}
        )
        spec.quality_gates = [
            QualityGateSpec("age_nulls", "null_percentage", "age", 5.0, action="warn"),
            QualityGateSpec("row_count", "row_count_minimum", "*", 1000, action="fail"),
        ]
        generator = DatabricksSpecGenerator(
            catalog="ml_catalog",
            schema="churn_detection",
            output_dir=str(tmp_path)
        )
        files = generator.save_all(spec)
        assert len(files) == 6
        for file_path in files:
            assert Path(file_path).exists()
        dlt_code = (tmp_path / "e2e_test_dlt_pipeline.py").read_text()
        assert "ml_catalog" not in dlt_code or "e2e_test_bronze" in dlt_code
        sql_code = (tmp_path / "e2e_test_unity_catalog.sql").read_text()
        assert "ml_catalog.churn_detection" in sql_code
        assert "customer_id STRING NOT NULL" in sql_code
        mlflow_code = (tmp_path / "e2e_test_mlflow_experiment.py").read_text()
        assert "max_depth" in mlflow_code
        assert "learning_rate" in mlflow_code

    def test_full_generic_artifact_generation(self, tmp_path):
        from customer_retention.generators.spec_generator.generic_generator import GenericSpecGenerator
        from customer_retention.generators.spec_generator.pipeline_spec import (
            ColumnSpec,
            ModelSpec,
            PipelineSpec,
            QualityGateSpec,
            SchemaSpec,
            SourceSpec,
            TransformSpec,
        )
        spec = PipelineSpec(name="generic_e2e", description="Generic artifacts test")
        spec.sources.append(SourceSpec("data", "/data/input.csv", "csv"))
        spec.schema = SchemaSpec(
            columns=[
                ColumnSpec("id", "string", "identifier"),
                ColumnSpec("value", "float", "numeric_continuous"),
                ColumnSpec("category", "string", "categorical_nominal"),
            ]
        )
        spec.silver_transforms = [
            TransformSpec("scale", "standard_scaling", ["value"], ["value_scaled"]),
            TransformSpec("encode", "one_hot_encoding", ["category"], ["category_enc"]),
        ]
        spec.quality_gates = [
            QualityGateSpec("value_check", "range", "value", 100.0),
        ]
        spec.model_config = ModelSpec("test_model", "gradient_boosting", "target", ["value"])
        generator = GenericSpecGenerator(output_dir=str(tmp_path))
        files = generator.save_all(spec)
        assert "pipeline.py" in files
        assert "dag.py" in files
        assert "README.md" in files
        pipeline_code = (tmp_path / "pipeline.py").read_text()
        assert "GradientBoostingClassifier" in pipeline_code
        readme_content = (tmp_path / "README.md").read_text()
        assert "generic_e2e" in readme_content
        docker_content = (tmp_path / "docker-compose.yml").read_text()
        assert "services:" in docker_content


class TestSpecAndNotebookIntegration:
    def test_spec_informs_notebook_generation(self, tmp_path):
        from customer_retention.generators.notebook_generator import (
            NotebookConfig,
            Platform,
            generate_orchestration_notebooks,
        )
        from customer_retention.generators.spec_generator.pipeline_spec import ModelSpec, PipelineSpec, SourceSpec
        spec = PipelineSpec(name="combined_test")
        spec.sources.append(SourceSpec("data", "/data/test.csv", "csv"))
        spec.model_config = ModelSpec(
            name="test_model",
            model_type="xgboost",
            target_column="target",
            feature_columns=["f1", "f2"]
        )
        config = NotebookConfig(
            project_name=spec.name,
            model_type=spec.model_config.model_type
        )
        results = generate_orchestration_notebooks(
            output_dir=str(tmp_path),
            platforms=[Platform.LOCAL],
            config=config
        )
        assert Platform.LOCAL in results
        assert len(results[Platform.LOCAL]) == 10
