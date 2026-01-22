from typing import List

import nbformat

from ..base import NotebookStage
from .base_stage import StageGenerator


class ModelTrainingStage(StageGenerator):
    @property
    def stage(self) -> NotebookStage:
        return NotebookStage.MODEL_TRAINING

    @property
    def title(self) -> str:
        return "07 - Model Training"

    @property
    def description(self) -> str:
        return "Train baseline models with MLflow experiment tracking."

    def generate_local_cells(self) -> List[nbformat.NotebookNode]:
        target = self.get_target_column()
        test_size = self.config.test_size
        exp_name = self.config.mlflow.experiment_name
        tracking_uri = self.config.mlflow.tracking_uri
        return self.header_cells() + [
            self.cb.section("Imports"),
            self.cb.from_imports_cell({
                "customer_retention.stages.modeling": ["BaselineTrainer", "ModelEvaluator", "DataSplitter"],
                "customer_retention.integrations.adapters": ["get_mlflow"],
                "customer_retention.analysis.visualization": ["ChartBuilder"],
                "customer_retention.stages.temporal": ["SnapshotManager"],
                "customer_retention.analysis.diagnostics": ["LeakageDetector"],
                "pathlib": ["Path"],
                "pandas": ["pd"],
            }),
            self.cb.section("Load Training Snapshot"),
            self.cb.markdown('''**Important**: We load from a versioned snapshot to ensure reproducibility and prevent data leakage.'''),
            self.cb.code('''snapshot_manager = SnapshotManager(Path("./experiments/data"))
latest_snapshot = snapshot_manager.get_latest_snapshot()

if latest_snapshot:
    df, snapshot_metadata = snapshot_manager.load_snapshot(latest_snapshot)
    print(f"Loaded snapshot: {latest_snapshot}")
    print(f"Snapshot cutoff date: {snapshot_metadata.cutoff_date}")
    print(f"Data hash: {snapshot_metadata.data_hash}")
    print(f"Rows: {snapshot_metadata.row_count}")
else:
    df = pd.read_parquet("./experiments/data/gold/customers_selected.parquet")
    snapshot_metadata = None
    print(f"Warning: No snapshot found, loading from gold layer: {df.shape}")'''),
            self.cb.section("Prepare Train/Test Split"),
            self.cb.code(f'''target_col = "target" if "target" in df.columns else "{target}"
id_cols = ["entity_id"] if "entity_id" in df.columns else {self.get_identifier_columns()}
temporal_cols = ["feature_timestamp", "label_timestamp", "label_available_flag"]
exclude_cols = id_cols + [target_col] + temporal_cols

feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"Using {{len(feature_cols)}} features (excluded: {{exclude_cols}})")

X = df[feature_cols]
y = df[target_col]

splitter = DataSplitter(test_size={test_size}, stratify=True, random_state=42)
X_train, X_test, y_train, y_test = splitter.split(X, y)
print(f"Train: {{len(X_train)}}, Test: {{len(X_test)}}")'''),
            self.cb.section("Run Leakage Detection"),
            self.cb.code('''detector = LeakageDetector()
leakage_result = detector.run_all_checks(X_train, y_train)

if not leakage_result.passed:
    print("WARNING: Leakage detected!")
    for issue in leakage_result.critical_issues:
        print(f"  CRITICAL: {issue.feature} - {issue.recommendation}")
else:
    print("Leakage check PASSED")'''),
            self.cb.section("Setup MLflow Tracking"),
            self.cb.code(f'''mlflow_adapter = get_mlflow(tracking_uri="{tracking_uri}", force_local=True)
experiment_name = "{exp_name}"
print(f"MLflow tracking URI: {tracking_uri}")

snapshot_params = {{}}
if snapshot_metadata:
    snapshot_params = {{
        "snapshot_id": snapshot_metadata.snapshot_id,
        "snapshot_version": snapshot_metadata.version,
        "snapshot_cutoff": str(snapshot_metadata.cutoff_date),
        "snapshot_hash": snapshot_metadata.data_hash,
    }}'''),
            self.cb.section("Train Baseline Models"),
            self.cb.code('''from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

models = {
    "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "random_forest": RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    mlflow_adapter.start_run(experiment_name, run_name=name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_prob)
    results[name] = {"model": model, "metrics": metrics, "y_pred": y_pred, "y_prob": y_prob}

    all_params = {**model.get_params(), **snapshot_params}
    mlflow_adapter.log_params(all_params)
    mlflow_adapter.log_metrics(metrics)
    mlflow_adapter.log_model(model, "model")
    mlflow_adapter.end_run()
    print(f"{name}: AUC={metrics.get('roc_auc', 0):.4f}, F1={metrics.get('f1', 0):.4f}")'''),
            self.cb.section("Compare Models"),
            self.cb.code('''charts = ChartBuilder()
fig = charts.model_comparison_grid(results, y_test)
fig.show()'''),
            self.cb.section("Save Best Model"),
            self.cb.code('''best_model_name = max(results, key=lambda k: results[k]["metrics"].get("roc_auc", 0))
best_model = results[best_model_name]["model"]
import joblib
joblib.dump(best_model, "./experiments/data/models/best_model.joblib")
print(f"Best model: {best_model_name}")'''),
        ]

    def generate_databricks_cells(self) -> List[nbformat.NotebookNode]:
        catalog = self.config.feature_store.catalog
        schema = self.config.feature_store.schema
        target = self.get_target_column()
        exp_name = self.config.mlflow.experiment_name
        model_name = self.config.mlflow.model_name
        return self.header_cells() + [
            self.cb.section("Load Selected Features"),
            self.cb.code(f'''df = spark.table("{catalog}.{schema}.gold_selected")'''),
            self.cb.section("Prepare Features Vector"),
            self.cb.code(f'''from pyspark.ml.feature import VectorAssembler

target_col = "{target}"
feature_cols = [c for c in df.columns if c not in {self.get_identifier_columns()} + [target_col]]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
df_ml = assembler.transform(df).select("features", target_col)
train_df, test_df = df_ml.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {{train_df.count()}}, Test: {{test_df.count()}}")'''),
            self.cb.section("Setup MLflow"),
            self.cb.code(f'''import mlflow
mlflow.set_experiment("/Users/{{spark.conf.get('spark.databricks.notebook.username', 'default')}}/{exp_name}")'''),
            self.cb.section("Train Gradient Boosted Trees"),
            self.cb.code(f'''from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

with mlflow.start_run(run_name="gbt_baseline"):
    gbt = GBTClassifier(featuresCol="features", labelCol="{target}", maxIter=100)
    model = gbt.fit(train_df)

    predictions = model.transform(test_df)
    evaluator = BinaryClassificationEvaluator(labelCol="{target}", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    mlflow.log_param("maxIter", 100)
    mlflow.log_metric("auc_roc", auc)
    mlflow.spark.log_model(model, "model")

    run_id = mlflow.active_run().info.run_id
    print(f"AUC: {{auc:.4f}}, Run ID: {{run_id}}")'''),
            self.cb.section("Register Model"),
            self.cb.code(f'''model_uri = f"runs:/{{run_id}}/model"
mlflow.register_model(model_uri, "{catalog}.{schema}.{model_name}")
print(f"Model registered: {catalog}.{schema}.{model_name}")'''),
        ]
