from typing import List

import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator


class MonitoringStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.MONITORING

    @property
    def title(self) -> str:
        return "09 - Model Monitoring"

    @property
    def description(self) -> str:
        return "Track model performance, detect drift, and set up alerts."

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        return self.header_cells() + [
            self.cb.section("Imports"),
            self.cb.from_imports_cell({
                "customer_retention.stages.monitoring": ["PerformanceMonitor", "DriftDetector"],
                "customer_retention.analysis.visualization": ["ChartBuilder"],
                "pandas": ["pd"],
                "joblib": ["joblib"],
            }),
            self.cb.section("Load Production Model and Test Data"),
            self.cb.code('''model = joblib.load("./experiments/data/models/best_model.joblib")
df_test = pd.read_parquet("./experiments/data/gold/customers_selected.parquet").sample(n=1000, random_state=42)'''),
            self.cb.section("Generate Predictions"),
            self.cb.code(f'''target_col = "{self.get_target_column()}"
id_cols = {self.get_identifier_columns()}
feature_cols = [c for c in df_test.columns if c not in id_cols + [target_col]]

X_test = df_test[feature_cols]
y_test = df_test[target_col]
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)'''),
            self.cb.section("Calculate Performance Metrics"),
            self.cb.code('''from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

current_metrics = {
    "roc_auc": roc_auc_score(y_test, y_prob),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
}
for name, value in current_metrics.items():
    print(f"{name}: {value:.4f}")'''),
            self.cb.section("Compare to Baseline"),
            self.cb.code('''baseline_metrics = {"roc_auc": 0.75, "precision": 0.60, "recall": 0.70, "f1": 0.65}
monitor = PerformanceMonitor(baseline_metrics)
result = monitor.evaluate(current_metrics)
print(f"Status: {result.status}")
for metric, change in result.changes.items():
    print(f"  {metric}: {change:+.2%}")'''),
            self.cb.section("Detect Feature Drift"),
            self.cb.code('''df_reference = pd.read_parquet("./experiments/data/gold/customers_features.parquet").sample(n=1000, random_state=0)
drift_detector = DriftDetector()
for col in feature_cols[:5]:
    result = drift_detector.detect(df_reference[col], df_test[col])
    if result.has_drift:
        print(f"DRIFT detected in {col}: PSI={result.psi:.4f}")'''),
        ]

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        catalog = self.config.feature_store.catalog
        schema = self.config.feature_store.schema
        model_name = self.config.mlflow.model_name
        target = self.get_target_column()
        return self.header_cells() + [
            self.cb.section("Load Model and Data"),
            self.cb.code(f'''import mlflow

model = mlflow.pyfunc.load_model(f"models:/{catalog}.{schema}.{model_name}@production")
df_test = spark.table("{catalog}.{schema}.gold_selected").sample(0.1)'''),
            self.cb.section("Generate Predictions"),
            self.cb.code(f'''from pyspark.sql.functions import pandas_udf
import pandas as pd

feature_cols = [c for c in df_test.columns if c not in {self.get_identifier_columns()} + ["{target}"]]

@pandas_udf("double")
def predict_udf(*cols):
    df = pd.concat(cols, axis=1)
    df.columns = feature_cols
    return pd.Series(model.predict(df))

df_predictions = df_test.withColumn("prediction", predict_udf(*[df_test[c] for c in feature_cols]))
display(df_predictions.limit(10))'''),
            self.cb.section("Calculate Metrics"),
            self.cb.code(f'''from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="{target}", rawPredictionCol="prediction")
auc = evaluator.evaluate(df_predictions)
print(f"Current AUC: {{auc:.4f}}")'''),
            self.cb.section("Check for Drift"),
            self.cb.code(f'''df_reference = spark.table("{catalog}.{schema}.gold_customers").sample(0.1)

for col in feature_cols[:5]:
    ref_stats = df_reference.select(col).describe().collect()
    cur_stats = df_test.select(col).describe().collect()
    ref_mean = float(ref_stats[1][1]) if ref_stats[1][1] else 0
    cur_mean = float(cur_stats[1][1]) if cur_stats[1][1] else 0
    drift_pct = abs(ref_mean - cur_mean) / (ref_mean + 1e-10) * 100
    if drift_pct > 10:
        print(f"DRIFT in {{col}}: {{drift_pct:.1f}}% mean shift")'''),
        ]
