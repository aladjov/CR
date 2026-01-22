import pytest
from pathlib import Path


class TestDatabricksSpecGeneratorFullCoverage:
    @pytest.fixture
    def full_spec(self):
        from customer_retention.generators.spec_generator.pipeline_spec import (
            PipelineSpec, SourceSpec, SchemaSpec, ColumnSpec,
            TransformSpec, FeatureSpec, ModelSpec, QualityGateSpec
        )
        spec = PipelineSpec(
            name="test_pipeline",
            version="1.0.0",
            description="Test pipeline"
        )
        spec.sources.append(SourceSpec(
            name="main_source",
            path="/data/customers.csv",
            format="csv"
        ))
        spec.schema = SchemaSpec(
            columns=[
                ColumnSpec("customer_id", "string", "identifier", nullable=False),
                ColumnSpec("age", "float", "numeric_continuous", nullable=True),
                ColumnSpec("category", "string", "categorical_nominal", nullable=True),
            ],
            primary_key="customer_id"
        )
        spec.silver_transforms = [
            TransformSpec("scale_age", "standard_scaling", ["age"], ["age_scaled"]),
            TransformSpec("encode_cat", "one_hot_encoding", ["category"], ["category_encoded"]),
        ]
        spec.feature_definitions = [
            FeatureSpec("days_since_signup", ["signup_date"], "days_since_today"),
        ]
        spec.model_config = ModelSpec(
            name="churn_model",
            model_type="gradient_boosting",
            target_column="churned",
            feature_columns=["age", "category"],
            hyperparameters={"max_depth": 5, "n_estimators": 100}
        )
        spec.quality_gates = [
            QualityGateSpec("age_not_null", "null_percentage", "age", 5.0),
        ]
        return spec

    def test_generate_silver_tables_with_transforms(self, full_spec):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        generator = DatabricksSpecGenerator()
        dlt_code = generator.generate_dlt_pipeline(full_spec)
        assert "# scale_age" in dlt_code
        assert "# encode_cat" in dlt_code

    def test_generate_gold_tables_with_features(self, full_spec):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        generator = DatabricksSpecGenerator()
        dlt_code = generator.generate_dlt_pipeline(full_spec)
        assert "days_since_signup" in dlt_code
        assert "datediff" in dlt_code

    def test_generate_feature_tables_with_primary_key(self, full_spec):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        generator = DatabricksSpecGenerator()
        code = generator.generate_feature_tables(full_spec)
        assert "primary_keys=" in code
        assert "customer_id" in code
        assert "FeatureLookup" in code

    def test_generate_feature_tables_without_primary_key(self, full_spec):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        full_spec.schema.primary_key = None
        generator = DatabricksSpecGenerator()
        code = generator.generate_feature_tables(full_spec)
        assert "write_table" in code

    def test_generate_mlflow_experiment_with_hyperparameters(self, full_spec):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        generator = DatabricksSpecGenerator()
        code = generator.generate_mlflow_experiment(full_spec)
        assert "max_depth" in code
        assert "n_estimators" in code

    def test_generate_unity_catalog_schema_with_columns(self, full_spec):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        generator = DatabricksSpecGenerator()
        sql = generator.generate_unity_catalog_schema(full_spec)
        assert "customer_id STRING NOT NULL" in sql
        assert "age DOUBLE" in sql
        assert "CREATE SCHEMA" in sql

    def test_infer_connector_type_various_formats(self):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        generator = DatabricksSpecGenerator()
        assert generator._infer_connector_type("csv") == "file"
        assert generator._infer_connector_type("parquet") == "file"
        assert generator._infer_connector_type("delta") == "delta"
        assert generator._infer_connector_type("jdbc") == "jdbc"
        assert generator._infer_connector_type("kafka") == "kafka"
        assert generator._infer_connector_type("unknown") == "file"

    def test_to_spark_type_conversions(self):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        generator = DatabricksSpecGenerator()
        assert generator._to_spark_type("string") == "STRING"
        assert generator._to_spark_type("integer") == "INT"
        assert generator._to_spark_type("float") == "DOUBLE"
        assert generator._to_spark_type("timestamp") == "TIMESTAMP"
        assert generator._to_spark_type("date") == "DATE"
        assert generator._to_spark_type("boolean") == "BOOLEAN"
        assert generator._to_spark_type("unknown") == "STRING"

    def test_save_all_creates_all_files(self, full_spec, tmp_path):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        generator = DatabricksSpecGenerator(output_dir=str(tmp_path))
        files = generator.save_all(full_spec)
        assert len(files) == 6
        assert (tmp_path / "test_pipeline_lakeflow_connect.json").exists()
        assert (tmp_path / "test_pipeline_dlt_pipeline.py").exists()
        assert (tmp_path / "test_pipeline_workflow_jobs.json").exists()
        assert (tmp_path / "test_pipeline_feature_tables.py").exists()
        assert (tmp_path / "test_pipeline_mlflow_experiment.py").exists()
        assert (tmp_path / "test_pipeline_unity_catalog.sql").exists()


class TestDatabricksSpecGeneratorEdgeCases:
    def test_generate_without_schema(self, tmp_path):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec, SourceSpec
        spec = PipelineSpec(name="minimal")
        spec.sources.append(SourceSpec("src", "/data/test.csv", "csv"))
        generator = DatabricksSpecGenerator(output_dir=str(tmp_path))
        artifacts = generator.generate_all(spec)
        assert "dlt_pipeline" in artifacts

    def test_generate_without_model_config(self, tmp_path):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec, SourceSpec
        spec = PipelineSpec(name="no_model")
        spec.sources.append(SourceSpec("src", "/data/test.csv", "csv"))
        generator = DatabricksSpecGenerator(output_dir=str(tmp_path))
        code = generator.generate_mlflow_experiment(spec)
        assert "target" in code
        assert "model" in code

    def test_generate_silver_tables_without_transforms(self):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec, SourceSpec
        spec = PipelineSpec(name="no_transforms")
        spec.sources.append(SourceSpec("src", "/data/test.csv", "csv"))
        spec.silver_transforms = []
        generator = DatabricksSpecGenerator()
        code = generator.generate_dlt_pipeline(spec)
        assert "Silver Layer" in code
        assert "return df" in code

    def test_generate_gold_tables_without_features(self):
        from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
        from customer_retention.generators.spec_generator.pipeline_spec import PipelineSpec, SourceSpec
        spec = PipelineSpec(name="no_features")
        spec.sources.append(SourceSpec("src", "/data/test.csv", "csv"))
        spec.feature_definitions = []
        generator = DatabricksSpecGenerator()
        code = generator.generate_dlt_pipeline(spec)
        assert "Gold Layer" in code
