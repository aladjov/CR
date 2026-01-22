from typing import List
import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator


class CleaningStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.CLEANING

    @property
    def title(self) -> str:
        return "03 - Data Cleaning"

    @property
    def description(self) -> str:
        return "Handle missing values and outliers based on column types with MLflow tracking."

    def _get_cleaning_recommendations(self) -> dict:
        recommendations = {}
        if not self.findings or not hasattr(self.findings, "columns"):
            return recommendations
        for col_name, col_finding in self.findings.columns.items():
            if hasattr(col_finding, "cleaning_recommendations") and col_finding.cleaning_recommendations:
                recommendations[col_name] = col_finding.cleaning_recommendations
        return recommendations

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        numeric_cols = self.get_numeric_columns()
        categorical_cols = self.get_categorical_columns()
        tracking_uri = self.config.mlflow.tracking_uri
        exp_name = self.config.mlflow.experiment_name
        cleaning_recs = self._get_cleaning_recommendations()

        cells = self.header_cells() + [
            self.cb.section("Imports"),
            self.cb.from_imports_cell({
                "customer_retention.stages.cleaning": ["MissingValueHandler", "OutlierHandler"],
                "customer_retention.integrations.adapters": ["get_mlflow"],
                "pandas": ["pd"],
            }),
            self.cb.section("Setup MLflow Tracking"),
            self.cb.code(f'''mlflow_adapter = get_mlflow(tracking_uri="{tracking_uri}", force_local=True)
mlflow_adapter.start_run("{exp_name}", run_name="03_data_cleaning")
cleaning_stats = {{}}'''),
            self.cb.section("Load Bronze Data"),
            self.cb.code('''df = pd.read_parquet("./experiments/data/bronze/customers.parquet")
initial_shape = df.shape
initial_nulls = df.isnull().sum().sum()
print(f"Initial shape: {df.shape}")
print(f"Total missing values: {initial_nulls}")

mlflow_adapter.log_metrics({
    "bronze_rows": initial_shape[0],
    "bronze_columns": initial_shape[1],
    "bronze_total_nulls": initial_nulls,
})'''),
        ]

        if cleaning_recs:
            cells.append(self.cb.section("Apply Cleaning from Exploration Findings"))
            cells.append(self.cb.code(f'''cleaning_recommendations = {cleaning_recs}
print(f"Found cleaning recommendations for {{len(cleaning_recommendations)}} columns")'''))

        cells.extend([
            self.cb.section("Handle Missing Values - Numeric Columns"),
            self.cb.code(f'''numeric_cols = {numeric_cols}
missing_handler = MissingValueHandler(strategy="median")
for col in numeric_cols:
    if col in df.columns and df[col].isnull().any():
        nulls_before = df[col].isnull().sum()
        df[col] = missing_handler.fit_transform(df[col])
        cleaning_stats[f"{{col}}_nulls_imputed"] = nulls_before
        print(f"Imputed {{col}}: {{nulls_before}} missing values")'''),
            self.cb.section("Handle Missing Values - Categorical Columns"),
            self.cb.code(f'''categorical_cols = {categorical_cols}
missing_handler_cat = MissingValueHandler(strategy="mode")
for col in categorical_cols:
    if col in df.columns and df[col].isnull().any():
        nulls_before = df[col].isnull().sum()
        df[col] = missing_handler_cat.fit_transform(df[col])
        cleaning_stats[f"{{col}}_nulls_imputed"] = nulls_before
        print(f"Imputed {{col}}: {{nulls_before}} missing values")'''),
            self.cb.section("Handle Outliers"),
            self.cb.code(f'''outlier_handler = OutlierHandler(method="iqr", treatment="cap")
for col in numeric_cols:
    if col in df.columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
        cleaning_stats[f"{{col}}_outliers_capped"] = outliers
        df[col] = outlier_handler.fit_transform(df[col])
print("Outliers capped using IQR method")'''),
            self.cb.section("Log Cleaning Statistics to MLflow"),
            self.cb.code('''final_nulls = df.isnull().sum().sum()
mlflow_adapter.log_params({
    "numeric_strategy": "median",
    "categorical_strategy": "mode",
    "outlier_method": "iqr",
    "outlier_treatment": "cap",
})
mlflow_adapter.log_metrics({
    "silver_rows": df.shape[0],
    "silver_columns": df.shape[1],
    "silver_total_nulls": final_nulls,
    "nulls_removed": initial_nulls - final_nulls,
    **{k: v for k, v in cleaning_stats.items() if isinstance(v, (int, float))}
})
print(f"Logged {len(cleaning_stats)} cleaning statistics to MLflow")'''),
            self.cb.section("Save to Silver Layer"),
            self.cb.code('''df.to_parquet("./experiments/data/silver/customers_cleaned.parquet", index=False)
mlflow_adapter.end_run()
print(f"Silver layer saved: {df.shape}")'''),
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
mlflow.start_run(run_name="03_data_cleaning")
cleaning_stats = {{}}'''),
            self.cb.section("Load Bronze Data"),
            self.cb.code(f'''df = spark.table("{catalog}.{schema}.bronze_customers")
initial_count = df.count()
print(f"Initial count: {{initial_count}}")
mlflow.log_metric("bronze_rows", initial_count)'''),
            self.cb.section("Handle Missing Values - Numeric Columns"),
            self.cb.code(f'''from pyspark.sql.functions import col, when, lit, sum as spark_sum
from pyspark.ml.feature import Imputer

numeric_cols = {numeric_cols}
imputer = Imputer(inputCols=numeric_cols, outputCols=numeric_cols, strategy="median")
df = imputer.fit(df).transform(df)
mlflow.log_param("numeric_strategy", "median")
print("Numeric columns imputed with median")'''),
            self.cb.section("Handle Missing Values - Categorical Columns"),
            self.cb.code(f'''categorical_cols = {categorical_cols}
for col_name in categorical_cols:
    mode_val = df.groupBy(col_name).count().orderBy("count", ascending=False).first()[0]
    df = df.fillna({{col_name: mode_val}})
mlflow.log_param("categorical_strategy", "mode")
print("Categorical columns imputed with mode")'''),
            self.cb.section("Handle Outliers with IQR"),
            self.cb.code(f'''for col_name in numeric_cols:
    quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
    if len(quantiles) == 2:
        q1, q3 = quantiles
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df.withColumn(col_name, when(col(col_name) < lower, lower)
                          .when(col(col_name) > upper, upper)
                          .otherwise(col(col_name)))
mlflow.log_params({{"outlier_method": "iqr", "outlier_treatment": "cap"}})
print("Outliers capped using IQR")'''),
            self.cb.section("Log Cleaning Statistics"),
            self.cb.code('''final_count = df.count()
mlflow.log_metrics({
    "silver_rows": final_count,
    "rows_preserved_pct": final_count / initial_count * 100,
})
print(f"Final count: {final_count}")'''),
            self.cb.section("Save to Silver Table"),
            self.cb.code(f'''df.write.format("delta").mode("overwrite").saveAsTable("{catalog}.{schema}.silver_customers")
mlflow.end_run()
print("Silver table created")'''),
        ]
