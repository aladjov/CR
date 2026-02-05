from pathlib import Path

import pytest

from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.core.config.column_config import ColumnType
from customer_retention.generators.spec_generator.databricks_generator import DatabricksSpecGenerator
from customer_retention.generators.spec_generator.pipeline_spec import (
    PipelineSpec,
)


@pytest.fixture
def sample_findings() -> ExplorationFindings:
    columns = {
        "customer_id": ColumnFinding(
            name="customer_id",
            inferred_type=ColumnType.IDENTIFIER,
            confidence=0.95,
            evidence=["All unique"],
            universal_metrics={"null_count": 0, "distinct_count": 1000}
        ),
        "age": ColumnFinding(
            name="age",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.85,
            evidence=["Numeric with many values"],
            universal_metrics={"null_count": 50, "null_percentage": 5.0},
            type_metrics={"mean": 45.2, "std": 15.3, "min_value": 18, "max_value": 85}
        ),
        "contract_type": ColumnFinding(
            name="contract_type",
            inferred_type=ColumnType.CATEGORICAL_NOMINAL,
            confidence=0.9,
            evidence=["String categorical"],
            universal_metrics={"null_count": 0, "distinct_count": 3},
            type_metrics={"top_categories": [("Monthly", 500), ("Annual", 300), ("Two-Year", 200)]}
        ),
        "churned": ColumnFinding(
            name="churned",
            inferred_type=ColumnType.TARGET,
            confidence=0.9,
            evidence=["Binary target"],
            universal_metrics={"null_count": 0, "distinct_count": 2}
        )
    }
    return ExplorationFindings(
        source_path="data/customers.csv",
        source_format="csv",
        row_count=1000,
        column_count=4,
        columns=columns,
        target_column="churned",
        target_type="binary",
        identifier_columns=["customer_id"],
        overall_quality_score=85.0,
        critical_issues=[],
        warnings=[]
    )


@pytest.fixture
def sample_spec(sample_findings) -> PipelineSpec:
    return PipelineSpec.from_findings(sample_findings, name="churn_pipeline")


class TestDatabricksGeneratorInit:
    def test_default_init(self):
        generator = DatabricksSpecGenerator()
        assert generator is not None

    def test_custom_catalog(self):
        generator = DatabricksSpecGenerator(catalog="my_catalog")
        assert generator.catalog == "my_catalog"

    def test_custom_schema(self):
        generator = DatabricksSpecGenerator(schema="my_schema")
        assert generator.schema == "my_schema"


class TestGenerateLakeflowConnect:
    def test_returns_dict(self, sample_spec):
        generator = DatabricksSpecGenerator()
        config = generator.generate_lakeflow_connect(sample_spec)

        assert isinstance(config, dict)

    def test_contains_ingestion_config(self, sample_spec):
        generator = DatabricksSpecGenerator()
        config = generator.generate_lakeflow_connect(sample_spec)

        assert "ingestion" in config

    def test_contains_source_info(self, sample_spec):
        generator = DatabricksSpecGenerator()
        config = generator.generate_lakeflow_connect(sample_spec)

        assert "sources" in config or "source" in str(config).lower()

    def test_contains_target_table(self, sample_spec):
        generator = DatabricksSpecGenerator()
        config = generator.generate_lakeflow_connect(sample_spec)

        assert "target" in str(config).lower() or "table" in str(config).lower()


class TestGenerateDLTPipeline:
    def test_returns_string(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_dlt_pipeline(sample_spec)

        assert isinstance(code, str)
        assert len(code) > 100

    def test_contains_dlt_decorator(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_dlt_pipeline(sample_spec)

        assert "@dlt.table" in code or "dlt.table" in code

    def test_contains_bronze_layer(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_dlt_pipeline(sample_spec)

        assert "bronze" in code.lower()

    def test_contains_silver_layer(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_dlt_pipeline(sample_spec)

        assert "silver" in code.lower()

    def test_contains_gold_layer(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_dlt_pipeline(sample_spec)

        assert "gold" in code.lower()

    def test_contains_expectations(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_dlt_pipeline(sample_spec)

        assert "expect" in code.lower() or "quality" in code.lower()

    def test_contains_pipeline_name(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_dlt_pipeline(sample_spec)

        assert "churn" in code.lower() or sample_spec.name in code


class TestGenerateWorkflowJobs:
    def test_returns_dict(self, sample_spec):
        generator = DatabricksSpecGenerator()
        config = generator.generate_workflow_jobs(sample_spec)

        assert isinstance(config, dict)

    def test_contains_job_name(self, sample_spec):
        generator = DatabricksSpecGenerator()
        config = generator.generate_workflow_jobs(sample_spec)

        assert "name" in config

    def test_contains_tasks(self, sample_spec):
        generator = DatabricksSpecGenerator()
        config = generator.generate_workflow_jobs(sample_spec)

        assert "tasks" in config
        assert isinstance(config["tasks"], list)
        assert len(config["tasks"]) > 0

    def test_contains_schedule(self, sample_spec):
        generator = DatabricksSpecGenerator()
        config = generator.generate_workflow_jobs(sample_spec)

        assert "schedule" in config or "trigger" in str(config).lower()

    def test_task_has_required_fields(self, sample_spec):
        generator = DatabricksSpecGenerator()
        config = generator.generate_workflow_jobs(sample_spec)

        task = config["tasks"][0]
        assert "task_key" in task
        assert "notebook_task" in task or "pipeline_task" in task or "python_wheel_task" in task


class TestGenerateFeatureTables:
    def test_returns_string(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_feature_tables(sample_spec)

        assert isinstance(code, str)
        assert len(code) > 50

    def test_contains_feature_store_import(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_feature_tables(sample_spec)

        assert "FeatureStoreClient" in code or "feature_store" in code.lower()

    def test_contains_table_creation(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_feature_tables(sample_spec)

        assert "create" in code.lower() or "write" in code.lower()

    def test_contains_primary_key(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_feature_tables(sample_spec)

        assert "primary_key" in code.lower() or "customer_id" in code


class TestGenerateMLflowExperiment:
    def test_returns_string(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_mlflow_experiment(sample_spec)

        assert isinstance(code, str)
        assert len(code) > 50

    def test_contains_mlflow_import(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_mlflow_experiment(sample_spec)

        assert "mlflow" in code.lower()

    def test_contains_experiment_setup(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_mlflow_experiment(sample_spec)

        assert "experiment" in code.lower()

    def test_contains_model_logging(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_mlflow_experiment(sample_spec)

        assert "log_model" in code or "log" in code


class TestGenerateUnityCatalogSchema:
    def test_returns_string(self, sample_spec):
        generator = DatabricksSpecGenerator()
        sql = generator.generate_unity_catalog_schema(sample_spec)

        assert isinstance(sql, str)
        assert len(sql) > 50

    def test_contains_create_table(self, sample_spec):
        generator = DatabricksSpecGenerator()
        sql = generator.generate_unity_catalog_schema(sample_spec)

        assert "CREATE TABLE" in sql.upper() or "CREATE OR REPLACE" in sql.upper()

    def test_contains_column_definitions(self, sample_spec):
        generator = DatabricksSpecGenerator()
        sql = generator.generate_unity_catalog_schema(sample_spec)

        assert "customer_id" in sql
        assert "age" in sql or "INT" in sql.upper() or "STRING" in sql.upper()


class TestGenerateConfig:
    def test_returns_string(self, sample_spec):
        generator = DatabricksSpecGenerator(catalog="analytics", schema="ml_features")
        code = generator.generate_config(sample_spec)
        assert isinstance(code, str)

    def test_contains_catalog(self, sample_spec):
        generator = DatabricksSpecGenerator(catalog="analytics", schema="ml_features")
        code = generator.generate_config(sample_spec)
        assert 'CATALOG = "analytics"' in code

    def test_contains_schema(self, sample_spec):
        generator = DatabricksSpecGenerator(catalog="analytics", schema="ml_features")
        code = generator.generate_config(sample_spec)
        assert 'SCHEMA = "ml_features"' in code

    def test_contains_pipeline_name(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_config(sample_spec)
        assert 'PIPELINE_NAME = "churn_pipeline"' in code

    def test_contains_table_paths(self, sample_spec):
        generator = DatabricksSpecGenerator(catalog="analytics", schema="ml_features")
        code = generator.generate_config(sample_spec)
        assert "analytics.ml_features.churn_pipeline_bronze" in code
        assert "analytics.ml_features.churn_pipeline_silver" in code
        assert "analytics.ml_features.churn_pipeline_gold" in code

    def test_contains_experiment_name(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_config(sample_spec)
        assert "EXPERIMENT_NAME" in code


class TestGenerateBronzeLayer:
    def test_returns_string(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_bronze_layer(sample_spec)
        assert isinstance(code, str)

    def test_has_run_function(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_bronze_layer(sample_spec)
        assert "def run(" in code

    def test_contains_pyspark_read(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_bronze_layer(sample_spec)
        assert "spark" in code.lower()

    def test_contains_delta_write(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_bronze_layer(sample_spec)
        assert "delta" in code.lower() or "write" in code.lower()

    def test_not_dlt(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_bronze_layer(sample_spec)
        assert "@dlt" not in code


class TestGenerateSilverLayer:
    def test_returns_string(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_silver_layer(sample_spec)
        assert isinstance(code, str)

    def test_has_run_function(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_silver_layer(sample_spec)
        assert "def run(" in code

    def test_reads_bronze_table(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_silver_layer(sample_spec)
        assert "bronze" in code.lower()

    def test_not_dlt(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_silver_layer(sample_spec)
        assert "@dlt" not in code


class TestGenerateGoldLayer:
    def test_returns_string(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_gold_layer(sample_spec)
        assert isinstance(code, str)

    def test_has_run_function(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_gold_layer(sample_spec)
        assert "def run(" in code

    def test_reads_silver_table(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_gold_layer(sample_spec)
        assert "silver" in code.lower()

    def test_not_dlt(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_gold_layer(sample_spec)
        assert "@dlt" not in code


class TestGeneratePipelineRunner:
    def test_returns_string(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_pipeline_runner(sample_spec)
        assert isinstance(code, str)

    def test_has_run_pipeline_function(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_pipeline_runner(sample_spec)
        assert "def run_pipeline(" in code

    def test_imports_layers(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_pipeline_runner(sample_spec)
        assert "bronze" in code.lower()
        assert "silver" in code.lower()
        assert "gold" in code.lower()

    def test_contains_execution_sequence(self, sample_spec):
        generator = DatabricksSpecGenerator()
        code = generator.generate_pipeline_runner(sample_spec)
        bronze_pos = code.lower().find("bronze")
        silver_pos = code.lower().find("silver", bronze_pos + 1)
        gold_pos = code.lower().find("gold", silver_pos + 1)
        assert bronze_pos < silver_pos < gold_pos


class TestGenerateAll:
    def test_returns_dict(self, sample_spec):
        generator = DatabricksSpecGenerator()
        artifacts = generator.generate_all(sample_spec)

        assert isinstance(artifacts, dict)

    def test_contains_expected_artifacts(self, sample_spec):
        generator = DatabricksSpecGenerator()
        artifacts = generator.generate_all(sample_spec)

        expected_keys = ["dlt_pipeline", "workflow_jobs", "feature_tables"]
        for key in expected_keys:
            assert key in artifacts

    def test_contains_new_artifacts(self, sample_spec):
        generator = DatabricksSpecGenerator()
        artifacts = generator.generate_all(sample_spec)

        for key in ["config", "bronze_layer", "silver_layer", "gold_layer", "pipeline_runner"]:
            assert key in artifacts

    def test_all_artifacts_non_empty(self, sample_spec):
        generator = DatabricksSpecGenerator()
        artifacts = generator.generate_all(sample_spec)

        for key, value in artifacts.items():
            assert value is not None
            if isinstance(value, str):
                assert len(value) > 0
            elif isinstance(value, dict):
                assert len(value) > 0


class TestSaveArtifacts:
    def test_save_all(self, sample_spec, tmp_path):
        generator = DatabricksSpecGenerator(output_dir=str(tmp_path))
        files = generator.save_all(sample_spec)

        assert isinstance(files, list)
        assert len(files) > 0

    def test_creates_files(self, sample_spec, tmp_path):
        generator = DatabricksSpecGenerator(output_dir=str(tmp_path))
        files = generator.save_all(sample_spec)

        for file_path in files:
            assert Path(file_path).exists()

    def test_save_all_creates_config(self, sample_spec, tmp_path):
        generator = DatabricksSpecGenerator(output_dir=str(tmp_path))
        generator.save_all(sample_spec)

        assert (tmp_path / "config.py").exists()

    def test_save_all_creates_pipeline_runner(self, sample_spec, tmp_path):
        generator = DatabricksSpecGenerator(output_dir=str(tmp_path))
        generator.save_all(sample_spec)

        assert (tmp_path / "pipeline_runner.py").exists()

    def test_save_all_creates_layer_dirs(self, sample_spec, tmp_path):
        generator = DatabricksSpecGenerator(output_dir=str(tmp_path))
        generator.save_all(sample_spec)

        for subdir in ["landing", "bronze", "silver", "gold", "training"]:
            assert (tmp_path / subdir).is_dir()

    def test_save_all_bronze_in_subdir(self, sample_spec, tmp_path):
        generator = DatabricksSpecGenerator(output_dir=str(tmp_path))
        generator.save_all(sample_spec)

        assert (tmp_path / "bronze" / "bronze_layer.py").exists()

    def test_save_all_silver_in_subdir(self, sample_spec, tmp_path):
        generator = DatabricksSpecGenerator(output_dir=str(tmp_path))
        generator.save_all(sample_spec)

        assert (tmp_path / "silver" / "silver_merge.py").exists()

    def test_save_all_gold_in_subdir(self, sample_spec, tmp_path):
        generator = DatabricksSpecGenerator(output_dir=str(tmp_path))
        generator.save_all(sample_spec)

        assert (tmp_path / "gold" / "gold_features.py").exists()

    def test_save_all_training_in_subdir(self, sample_spec, tmp_path):
        generator = DatabricksSpecGenerator(output_dir=str(tmp_path))
        generator.save_all(sample_spec)

        assert (tmp_path / "training" / "ml_experiment.py").exists()

    def test_save_all_landing_in_subdir(self, sample_spec, tmp_path):
        generator = DatabricksSpecGenerator(output_dir=str(tmp_path))
        generator.save_all(sample_spec)

        assert (tmp_path / "landing" / "lakeflow_connect.json").exists()
