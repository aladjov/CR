from typing import List

import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator


class TransformationStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.TRANSFORMATION

    @property
    def title(self) -> str:
        return "04 - Data Transformation"

    @property
    def description(self) -> str:
        return "Apply scaling, encoding, and type transformations with MLflow tracking."

    def _get_transform_recommendations(self) -> dict:
        recommendations = {}
        if not self.findings or not hasattr(self.findings, "columns"):
            return recommendations
        for col_name, col_finding in self.findings.columns.items():
            if hasattr(col_finding, "transformation_recommendations") and col_finding.transformation_recommendations:
                recommendations[col_name] = col_finding.transformation_recommendations
        return recommendations

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        numeric_cols = self.get_numeric_columns()
        categorical_cols = self.get_categorical_columns()
        tracking_uri = self.config.mlflow.tracking_uri
        exp_name = self.config.mlflow.experiment_name
        transform_recs = self._get_transform_recommendations()

        cells = self.header_cells() + [
            self.cb.section("Imports"),
            self.cb.from_imports_cell({
                "customer_retention.stages.transformation": ["NumericTransformer", "CategoricalEncoder"],
                "customer_retention.stages.preprocessing": ["TransformerManager"],
                "customer_retention.integrations.adapters": ["get_mlflow"],
                "pandas": ["pd"],
            }),
            self.cb.section("Setup MLflow Tracking"),
            self.cb.code(f'''mlflow_adapter = get_mlflow(tracking_uri="{tracking_uri}", force_local=True)
mlflow_adapter.start_run("{exp_name}", run_name="04_transformation")
transform_stats = {{}}'''),
            self.cb.section("Load Silver Data"),
            self.cb.code('''df = pd.read_parquet("./experiments/data/silver/customers_cleaned.parquet")
print(f"Loaded shape: {df.shape}")
mlflow_adapter.log_metric("input_rows", df.shape[0])
mlflow_adapter.log_metric("input_columns", df.shape[1])'''),
        ]

        if transform_recs:
            cells.append(self.cb.section("Transformation Recommendations from Exploration"))
            cells.append(self.cb.code(f'''transform_recommendations = {transform_recs}
print(f"Found transformation recommendations for {{len(transform_recommendations)}} columns")'''))

        cells.extend([
            self.cb.section("Initialize Transformer Manager"),
            self.cb.code(f'''numeric_cols = {numeric_cols}
categorical_cols = {categorical_cols}

# TransformerManager ensures consistent transformations between training and scoring
transformer_manager = TransformerManager(scaler_type="standard")'''),
            self.cb.section("Fit and Transform Features"),
            self.cb.code('''# Fit transformers and transform data in one step
# Exclude identifier and target columns from transformation
exclude_cols = ["customer_id", "target"]  # Adjust based on your data
df = transformer_manager.fit_transform(
    df,
    numeric_columns=numeric_cols,
    categorical_columns=categorical_cols,
    exclude_columns=exclude_cols
)

# Log transformation statistics
manifest = transformer_manager.manifest
transform_stats["numeric_cols_scaled"] = len(manifest.numeric_columns)
transform_stats["categorical_cols_encoded"] = len(manifest.categorical_columns)

mlflow_adapter.log_params({
    "scaler_type": manifest.scaler_type,
    "encoder_type": manifest.encoder_type,
    "scaled_columns": str(manifest.numeric_columns)[:250],
    "encoded_columns": str(manifest.categorical_columns)[:250],
})
print(f"Scaled {len(manifest.numeric_columns)} numeric columns")
print(f"Encoded {len(manifest.categorical_columns)} categorical columns")'''),
            self.cb.section("Save Transformers as Artifacts"),
            self.cb.code('''# Save transformers locally and to MLflow
transformer_manager.save("./experiments/data/transformers/transformers.joblib")

# Log to MLflow for scoring pipeline to retrieve
import mlflow
transformer_manager.log_to_mlflow(run_id=mlflow.active_run().info.run_id)
print("Transformers saved locally and logged to MLflow")
print("Scoring pipeline will use these same transformers for consistency")'''),
            self.cb.section("Log Transformation Statistics"),
            self.cb.code('''mlflow_adapter.log_metrics({
    "output_rows": df.shape[0],
    "output_columns": df.shape[1],
    **{k: v for k, v in transform_stats.items() if isinstance(v, (int, float))}
})
print(f"Logged {len(transform_stats)} transformation statistics")'''),
            self.cb.section("Save Transformed Data"),
            self.cb.code('''df.to_parquet("./experiments/data/silver/customers_transformed.parquet", index=False)
mlflow_adapter.end_run()
print(f"Transformed data saved: {df.shape}")'''),
        ])
        return cells

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        catalog = self.config.feature_store.catalog
        schema = self.config.feature_store.schema
        exp_name = self.config.mlflow.experiment_name
        numeric_cols = self.get_numeric_columns()
        categorical_cols = self.get_categorical_columns()
        return self.header_cells() + [
            self.cb.section("Setup MLflow Tracking"),
            self.cb.code(f'''import mlflow

mlflow.set_experiment("/Users/{{spark.conf.get('spark.databricks.notebook.username', 'default')}}/{exp_name}")
mlflow.start_run(run_name="04_transformation")'''),
            self.cb.section("Load Silver Data"),
            self.cb.code(f'''df = spark.table("{catalog}.{schema}.silver_customers")
input_count = df.count()
mlflow.log_metric("input_rows", input_count)'''),
            self.cb.section("Scale Numeric Features"),
            self.cb.code(f'''from pyspark.ml.feature import StandardScaler, VectorAssembler

numeric_cols = {numeric_cols}
if numeric_cols:
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
    df = assembler.transform(df)
    scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    mlflow.log_params({{"scaler_type": "standard", "scaled_columns_count": len(numeric_cols)}})
    print("Numeric features scaled")'''),
            self.cb.section("Encode Categorical Features"),
            self.cb.code(f'''from pyspark.ml.feature import StringIndexer

categorical_cols = {categorical_cols}
for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{{col_name}}_idx")
    df = indexer.fit(df).transform(df)
mlflow.log_params({{"encoder_type": "string_indexer", "encoded_columns_count": len(categorical_cols)}})
print(f"Encoded {{len(categorical_cols)}} categorical columns")'''),
            self.cb.section("Log Statistics"),
            self.cb.code('''output_count = df.count()
mlflow.log_metrics({
    "output_rows": output_count,
    "columns_after_transform": len(df.columns),
})'''),
            self.cb.section("Save Transformed Data"),
            self.cb.code(f'''df.write.format("delta").mode("overwrite").saveAsTable("{catalog}.{schema}.silver_transformed")
mlflow.end_run()
print("Transformed data saved")'''),
        ]
