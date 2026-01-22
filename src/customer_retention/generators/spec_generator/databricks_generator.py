from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from .pipeline_spec import PipelineSpec


class DatabricksSpecGenerator:
    def __init__(self,
                 catalog: str = "main",
                 schema: str = "default",
                 output_dir: str = "./databricks_artifacts"):
        self.catalog = catalog
        self.schema = schema
        self.output_dir = Path(output_dir)

    def generate_lakeflow_connect(self, spec: PipelineSpec) -> dict:
        sources = []
        for source in spec.sources:
            sources.append({
                "name": source.name,
                "type": self._infer_connector_type(source.format),
                "path": source.path,
                "options": source.options
            })

        return {
            "ingestion": {
                "name": f"{spec.name}_ingestion",
                "version": spec.version,
                "target_catalog": self.catalog,
                "target_schema": self.schema,
                "sources": sources,
                "target_tables": [
                    {
                        "name": f"{spec.name}_bronze",
                        "format": "delta",
                        "mode": "append"
                    }
                ],
                "schedule": {
                    "quartz_cron_expression": "0 0 * * * ?",
                    "timezone_id": "UTC"
                }
            }
        }

    def _infer_connector_type(self, format: str) -> str:
        format_map = {
            "csv": "file",
            "parquet": "file",
            "json": "file",
            "delta": "delta",
            "jdbc": "jdbc",
            "kafka": "kafka"
        }
        return format_map.get(format.lower(), "file")

    def generate_dlt_pipeline(self, spec: PipelineSpec) -> str:
        lines = [
            "import dlt",
            "from pyspark.sql import functions as F",
            "",
            f"# DLT Pipeline: {spec.name}",
            f"# Version: {spec.version}",
            "",
        ]

        lines.extend(self._generate_bronze_tables(spec))
        lines.append("")
        lines.extend(self._generate_silver_tables(spec))
        lines.append("")
        lines.extend(self._generate_gold_tables(spec))

        return "\n".join(lines)

    def _generate_bronze_tables(self, spec: PipelineSpec) -> List[str]:
        lines = ["# Bronze Layer - Raw Data Ingestion", ""]

        for source in spec.sources:
            table_name = f"{spec.name}_bronze"
            lines.extend([
                f'@dlt.table(',
                f'    name="{table_name}",',
                f'    comment="Raw data from {source.name}"',
                f')',
                f'@dlt.expect_or_drop("valid_record", "1=1")',
                f'def {table_name}():',
                f'    return (',
                f'        spark.read',
                f'        .format("{source.format}")',
                f'        .load("{source.path}")',
                f'    )',
                ""
            ])

        return lines

    def _generate_silver_tables(self, spec: PipelineSpec) -> List[str]:
        lines = ["# Silver Layer - Cleaned and Standardized", ""]

        table_name = f"{spec.name}_silver"
        bronze_table = f"{spec.name}_bronze"

        expectations = []
        for gate in spec.quality_gates:
            if gate.gate_type == "null_percentage":
                expectations.append(
                    f'@dlt.expect_or_warn("{gate.name}", "{gate.column} IS NOT NULL")'
                )

        lines.extend([
            f'@dlt.table(',
            f'    name="{table_name}",',
            f'    comment="Cleaned and standardized data"',
            f')',
        ])
        for exp in expectations:
            lines.append(exp)

        lines.extend([
            f'def {table_name}():',
            f'    df = dlt.read("{bronze_table}")',
            ""
        ])

        if spec.silver_transforms:
            for transform in spec.silver_transforms:
                if transform.transform_type == "standard_scaling":
                    col = transform.input_columns[0]
                    out_col = transform.output_columns[0]
                    lines.append(f'    # {transform.name}')
                    lines.append(f'    df = df.withColumn("{out_col}", F.col("{col}"))')
                elif transform.transform_type == "one_hot_encoding":
                    col = transform.input_columns[0]
                    lines.append(f'    # {transform.name}')
                    lines.append(f'    # Note: One-hot encoding applied in feature engineering')

            lines.append("")

        lines.extend([
            f'    return df',
            ""
        ])

        return lines

    def _generate_gold_tables(self, spec: PipelineSpec) -> List[str]:
        lines = ["# Gold Layer - Feature Engineering", ""]

        table_name = f"{spec.name}_gold"
        silver_table = f"{spec.name}_silver"

        lines.extend([
            f'@dlt.table(',
            f'    name="{table_name}",',
            f'    comment="Feature-engineered data ready for modeling"',
            f')',
            f'def {table_name}():',
            f'    df = dlt.read("{silver_table}")',
            ""
        ])

        if spec.feature_definitions:
            for feature in spec.feature_definitions:
                lines.append(f'    # Feature: {feature.name}')
                if feature.computation == "days_since_today":
                    col = feature.source_columns[0]
                    lines.append(
                        f'    df = df.withColumn("{feature.name}", '
                        f'F.datediff(F.current_date(), F.col("{col}")))'
                    )

            lines.append("")

        lines.extend([
            f'    return df',
            ""
        ])

        return lines

    def generate_workflow_jobs(self, spec: PipelineSpec) -> dict:
        tasks = [
            {
                "task_key": "run_dlt_pipeline",
                "pipeline_task": {
                    "pipeline_id": f"{{{{pipelines.{spec.name}_dlt.id}}}}"
                }
            },
            {
                "task_key": "train_model",
                "depends_on": [{"task_key": "run_dlt_pipeline"}],
                "notebook_task": {
                    "notebook_path": f"/Repos/{spec.name}/notebooks/train_model",
                    "base_parameters": {
                        "catalog": self.catalog,
                        "schema": self.schema,
                        "table": f"{spec.name}_gold"
                    }
                }
            },
            {
                "task_key": "validate_model",
                "depends_on": [{"task_key": "train_model"}],
                "notebook_task": {
                    "notebook_path": f"/Repos/{spec.name}/notebooks/validate_model"
                }
            }
        ]

        return {
            "name": f"{spec.name}_workflow",
            "tasks": tasks,
            "schedule": {
                "quartz_cron_expression": "0 0 0 * * ?",
                "timezone_id": "UTC",
                "pause_status": "UNPAUSED"
            },
            "email_notifications": {
                "on_failure": []
            },
            "max_concurrent_runs": 1,
            "trigger": {
                "periodic": {
                    "interval": 1,
                    "unit": "DAYS"
                }
            }
        }

    def generate_feature_tables(self, spec: PipelineSpec) -> str:
        primary_key = None
        if spec.schema and spec.schema.primary_key:
            primary_key = spec.schema.primary_key

        lines = [
            "from databricks.feature_store import FeatureStoreClient",
            "from databricks.feature_store import FeatureLookup",
            "from pyspark.sql import functions as F",
            "",
            f"# Feature Store Tables for {spec.name}",
            "",
            "fs = FeatureStoreClient()",
            "",
            f'# Create feature table',
            f'feature_table_name = "{self.catalog}.{self.schema}.{spec.name}_features"',
            ""
        ]

        if primary_key:
            lines.extend([
                f'# Define the feature table with primary key: {primary_key}',
                f'fs.create_table(',
                f'    name=feature_table_name,',
                f'    primary_keys=["{primary_key}"],',
                f'    df=spark.table("{self.catalog}.{self.schema}.{spec.name}_gold"),',
                f'    description="Features for {spec.name}"',
                f')',
                ""
            ])
        else:
            lines.extend([
                f'# Write features to table',
                f'df = spark.table("{self.catalog}.{self.schema}.{spec.name}_gold")',
                f'fs.write_table(',
                f'    name=feature_table_name,',
                f'    df=df,',
                f'    mode="overwrite"',
                f')',
                ""
            ])

        if spec.feature_definitions:
            lines.extend([
                "# Feature lookups for training",
                "feature_lookups = ["
            ])
            for feature in spec.feature_definitions:
                lines.append(f'    FeatureLookup(')
                lines.append(f'        table_name=feature_table_name,')
                lines.append(f'        feature_names=["{feature.name}"],')
                if primary_key:
                    lines.append(f'        lookup_key="{primary_key}"')
                lines.append(f'    ),')
            lines.append("]")

        return "\n".join(lines)

    def generate_mlflow_experiment(self, spec: PipelineSpec) -> str:
        target = spec.model_config.target_column if spec.model_config else "target"
        model_type = spec.model_config.model_type if spec.model_config else "gradient_boosting"
        model_name = spec.model_config.name if spec.model_config else "model"

        lines = [
            "import mlflow",
            "import mlflow.spark",
            "from pyspark.ml.classification import GBTClassifier, RandomForestClassifier",
            "from pyspark.ml.evaluation import BinaryClassificationEvaluator",
            "",
            f"# MLflow Experiment: {spec.name}",
            "",
            f'mlflow.set_experiment("/Users/{{username}}/{spec.name}_experiment")',
            "",
            f'with mlflow.start_run(run_name="{model_name}"):',
            f'    # Load training data',
            f'    df = spark.table("{self.catalog}.{self.schema}.{spec.name}_gold")',
            "",
            f'    # Log parameters',
            f'    mlflow.log_param("target_column", "{target}")',
            f'    mlflow.log_param("model_type", "{model_type}")',
            ""
        ]

        if spec.model_config and spec.model_config.hyperparameters:
            lines.append("    # Log hyperparameters")
            for key, value in spec.model_config.hyperparameters.items():
                lines.append(f'    mlflow.log_param("{key}", {repr(value)})')
            lines.append("")

        lines.extend([
            f'    # Train model',
            f'    model = GBTClassifier(',
            f'        featuresCol="features",',
            f'        labelCol="{target}",',
            f'        maxIter=100',
            f'    )',
            "",
            f'    trained_model = model.fit(df)',
            "",
            f'    # Log model',
            f'    mlflow.spark.log_model(trained_model, "{model_name}")',
            "",
            f'    # Register model in Unity Catalog',
            f'    mlflow.register_model(',
            f'        f"runs/{{mlflow.active_run().info.run_id}}/{model_name}",',
            f'        "{self.catalog}.{self.schema}.{spec.name}_model"',
            f'    )',
        ])

        return "\n".join(lines)

    def generate_unity_catalog_schema(self, spec: PipelineSpec) -> str:
        lines = [
            f"-- Unity Catalog Schema for {spec.name}",
            f"-- Generated from PipelineSpec version {spec.version}",
            "",
            f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{self.schema};",
            "",
        ]

        for layer in ["bronze", "silver", "gold"]:
            table_name = f"{spec.name}_{layer}"
            lines.extend([
                f"CREATE OR REPLACE TABLE {self.catalog}.{self.schema}.{table_name} (",
            ])

            if spec.schema:
                col_defs = []
                for col in spec.schema.columns:
                    spark_type = self._to_spark_type(col.data_type)
                    nullable = "" if col.nullable else " NOT NULL"
                    col_defs.append(f"    {col.name} {spark_type}{nullable}")

                lines.append(",\n".join(col_defs))

            lines.extend([
                ")",
                f"USING DELTA",
                f"COMMENT '{layer.title()} layer table for {spec.name}';",
                ""
            ])

        return "\n".join(lines)

    def _to_spark_type(self, data_type: str) -> str:
        type_map = {
            "string": "STRING",
            "integer": "INT",
            "float": "DOUBLE",
            "timestamp": "TIMESTAMP",
            "date": "DATE",
            "boolean": "BOOLEAN"
        }
        return type_map.get(data_type.lower(), "STRING")

    def generate_all(self, spec: PipelineSpec) -> Dict[str, Any]:
        return {
            "lakeflow_connect": self.generate_lakeflow_connect(spec),
            "dlt_pipeline": self.generate_dlt_pipeline(spec),
            "workflow_jobs": self.generate_workflow_jobs(spec),
            "feature_tables": self.generate_feature_tables(spec),
            "mlflow_experiment": self.generate_mlflow_experiment(spec),
            "unity_catalog_schema": self.generate_unity_catalog_schema(spec)
        }

    def save_all(self, spec: PipelineSpec) -> List[str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        artifacts = self.generate_all(spec)

        lakeflow_path = self.output_dir / f"{spec.name}_lakeflow_connect.json"
        with open(lakeflow_path, "w") as f:
            json.dump(artifacts["lakeflow_connect"], f, indent=2)
        saved_files.append(str(lakeflow_path))

        dlt_path = self.output_dir / f"{spec.name}_dlt_pipeline.py"
        with open(dlt_path, "w") as f:
            f.write(artifacts["dlt_pipeline"])
        saved_files.append(str(dlt_path))

        jobs_path = self.output_dir / f"{spec.name}_workflow_jobs.json"
        with open(jobs_path, "w") as f:
            json.dump(artifacts["workflow_jobs"], f, indent=2)
        saved_files.append(str(jobs_path))

        features_path = self.output_dir / f"{spec.name}_feature_tables.py"
        with open(features_path, "w") as f:
            f.write(artifacts["feature_tables"])
        saved_files.append(str(features_path))

        mlflow_path = self.output_dir / f"{spec.name}_mlflow_experiment.py"
        with open(mlflow_path, "w") as f:
            f.write(artifacts["mlflow_experiment"])
        saved_files.append(str(mlflow_path))

        schema_path = self.output_dir / f"{spec.name}_unity_catalog.sql"
        with open(schema_path, "w") as f:
            f.write(artifacts["unity_catalog_schema"])
        saved_files.append(str(schema_path))

        return saved_files
