from typing import List
import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator


class FeatureEngineeringStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.FEATURE_ENGINEERING

    @property
    def title(self) -> str:
        return "05 - Feature Engineering"

    @property
    def description(self) -> str:
        return "Create derived features, interactions, and aggregations."

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        return self.header_cells() + [
            self.cb.section("Imports"),
            self.cb.from_imports_cell({
                "customer_retention.stages.features": ["FeatureEngineer", "FeatureEngineerConfig"],
                "customer_retention.stages.features.temporal_features": ["TemporalFeatureGenerator", "ReferenceDateSource"],
                "customer_retention.stages.temporal": ["PointInTimeJoiner", "SnapshotManager"],
                "pathlib": ["Path"],
                "pandas": ["pd"],
                "numpy": ["np"],
            }),
            self.cb.section("Load Latest Training Snapshot"),
            self.cb.code('''snapshot_manager = SnapshotManager(Path("./experiments/data"))
latest_snapshot = snapshot_manager.get_latest_snapshot()
if latest_snapshot:
    df, metadata = snapshot_manager.load_snapshot(latest_snapshot)
    print(f"Loaded snapshot: {latest_snapshot}")
    print(f"Rows: {len(df)}, Features: {len(df.columns)}")
else:
    df = pd.read_parquet("./experiments/data/silver/customers_transformed.parquet")
    print(f"No snapshot found, loaded transformed data: {df.shape}")'''),
            self.cb.section("Point-in-Time Feature Engineering"),
            self.cb.markdown('''**Important**: All temporal features are calculated relative to `feature_timestamp` to prevent data leakage.'''),
            self.cb.code('''if "feature_timestamp" in df.columns:
    temporal_gen = TemporalFeatureGenerator(
        reference_date_source=ReferenceDateSource.FEATURE_TIMESTAMP,
        created_column="signup_date" if "signup_date" in df.columns else None,
        last_order_column="last_activity" if "last_activity" in df.columns else None,
    )
    df = temporal_gen.fit_transform(df)
    print(f"Created temporal features: {temporal_gen.generated_features}")
else:
    print("Warning: No feature_timestamp column found. Using current date (may cause leakage).")
    if "signup_date" in df.columns:
        df["tenure_days"] = (pd.Timestamp.now() - pd.to_datetime(df["signup_date"])).dt.days'''),
            self.cb.section("Validate Point-in-Time Correctness"),
            self.cb.code('''if "feature_timestamp" in df.columns:
    pit_report = PointInTimeJoiner.validate_temporal_integrity(df)
    if pit_report["valid"]:
        print("Point-in-time validation PASSED")
    else:
        print("Point-in-time validation FAILED:")
        for issue in pit_report["issues"]:
            print(f"  - {issue['type']}: {issue['message']}")'''),
            self.cb.section("Create Interaction Features"),
            self.cb.code('''numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
               if c not in ["target", "entity_id"]]
if len(numeric_cols) >= 2:
    for i, col1 in enumerate(numeric_cols[:3]):
        for col2 in numeric_cols[i+1:4]:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    print(f"Created interaction features")'''),
            self.cb.section("Create Ratio Features"),
            self.cb.code('''if "total_spend" in df.columns and "num_transactions" in df.columns:
    df["avg_transaction_value"] = df["total_spend"] / (df["num_transactions"] + 1)
    print("Created avg_transaction_value feature")'''),
            self.cb.section("Save to Gold Layer"),
            self.cb.code('''df.to_parquet("./experiments/data/gold/customers_features.parquet", index=False)
print(f"Gold layer saved: {df.shape}")'''),
        ]

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        catalog = self.config.feature_store.catalog
        schema = self.config.feature_store.schema
        return self.header_cells() + [
            self.cb.section("Load Transformed Data"),
            self.cb.code(f'''df = spark.table("{catalog}.{schema}.silver_transformed")'''),
            self.cb.section("Create Derived Features"),
            self.cb.code('''from pyspark.sql.functions import datediff, current_date, col

if "signup_date" in df.columns:
    df = df.withColumn("tenure_days", datediff(current_date(), col("signup_date")))
    print("Created tenure_days feature")

if "last_activity" in df.columns:
    df = df.withColumn("recency_days", datediff(current_date(), col("last_activity")))
    print("Created recency_days feature")'''),
            self.cb.section("Create Interaction Features"),
            self.cb.code('''numeric_cols = [f.name for f in df.schema.fields if str(f.dataType) in ["IntegerType()", "DoubleType()", "FloatType()"]]
if len(numeric_cols) >= 2:
    df = df.withColumn(f"{numeric_cols[0]}_x_{numeric_cols[1]}", col(numeric_cols[0]) * col(numeric_cols[1]))
    print("Created interaction features")'''),
            self.cb.section("Create Ratio Features"),
            self.cb.code('''if "total_spend" in df.columns and "num_transactions" in df.columns:
    df = df.withColumn("avg_transaction_value", col("total_spend") / (col("num_transactions") + 1))
    print("Created avg_transaction_value feature")'''),
            self.cb.section("Save to Gold Table"),
            self.cb.code(f'''df.write.format("delta").mode("overwrite").saveAsTable("{catalog}.{schema}.gold_customers")
print("Gold table created")'''),
        ]
