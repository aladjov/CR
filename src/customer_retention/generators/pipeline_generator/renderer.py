from pathlib import Path
from jinja2 import Environment, FileSystemLoader, BaseLoader
from .models import PipelineConfig, BronzeLayerConfig


class InlineLoader(BaseLoader):
    def __init__(self, templates: dict):
        self._templates = templates

    def get_source(self, environment, template):
        if template in self._templates:
            return self._templates[template], template, lambda: True
        raise Exception(f"Template {template} not found")


TEMPLATES = {
    "config.py.j2": '''from pathlib import Path

PIPELINE_NAME = "{{ config.name }}"
TARGET_COLUMN = "{{ config.target_column }}"
OUTPUT_DIR = Path("{{ config.output_dir }}")

# Iteration tracking
ITERATION_ID = {{ '"%s"' % config.iteration_id if config.iteration_id else 'None' }}
PARENT_ITERATION_ID = {{ '"%s"' % config.parent_iteration_id if config.parent_iteration_id else 'None' }}

# Recommendations hash for experiment tracking
RECOMMENDATIONS_HASH = {{ '"%s"' % config.recommendations_hash if config.recommendations_hash else 'None' }}

# MLflow tracking - centralized at project root
def _find_project_root():
    path = Path(__file__).parent
    for _ in range(10):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
        path = path.parent
    return Path(__file__).parent

PROJECT_ROOT = _find_project_root()
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")

# Feast feature store configuration
FEAST_REPO_PATH = str(OUTPUT_DIR / "feature_repo")
FEAST_FEATURE_VIEW = "{{ config.feast.feature_view_name if config.feast else config.name + '_features' }}"
FEAST_ENTITY_NAME = "{{ config.feast.entity_name if config.feast else 'customer' }}"
FEAST_ENTITY_KEY = "{{ config.feast.entity_key if config.feast else config.sources[0].entity_key }}"
FEAST_TIMESTAMP_COL = "{{ config.feast.timestamp_column if config.feast else 'event_timestamp' }}"
FEAST_TTL_DAYS = {{ config.feast.ttl_days if config.feast else 365 }}

SOURCES = {
{% for source in config.sources %}
    "{{ source.name }}": {
        "path": "{{ source.path }}",
        "format": "{{ source.format }}",
        "entity_key": "{{ source.entity_key }}",
{% if source.time_column %}
        "time_column": "{{ source.time_column }}",
{% endif %}
        "is_event_level": {{ source.is_event_level }},
    },
{% endfor %}
}


def get_bronze_path(source_name: str) -> Path:
    return OUTPUT_DIR / "data" / "bronze" / f"{source_name}.parquet"


def get_silver_path() -> Path:
    return OUTPUT_DIR / "data" / "silver" / "merged.parquet"


def get_gold_path() -> Path:
    return OUTPUT_DIR / "data" / "gold" / "features.parquet"


def get_feast_data_path() -> Path:
    return Path(FEAST_REPO_PATH) / "data" / f"{FEAST_FEATURE_VIEW}.parquet"
''',

    "bronze.py.j2": '''import pandas as pd
from config import SOURCES, get_bronze_path

SOURCE_NAME = "{{ source }}"


def load_{{ source }}():
    source_config = SOURCES[SOURCE_NAME]
    if source_config["format"] == "csv":
        return pd.read_csv(source_config["path"])
    return pd.read_parquet(source_config["path"])


def apply_transformations(df: pd.DataFrame) -> pd.DataFrame:
{% for t in config.transformations %}
{% if t.type.value == "impute_null" %}
    df["{{ t.column }}"] = df["{{ t.column }}"].fillna({{ t.parameters.get("value", 0) }})
{% elif t.type.value == "cap_outlier" %}
    df["{{ t.column }}"] = df["{{ t.column }}"].clip(lower={{ t.parameters.get("lower", 0) }}, upper={{ t.parameters.get("upper", 1000000) }})
{% elif t.type.value == "type_cast" %}
    df["{{ t.column }}"] = df["{{ t.column }}"].astype("{{ t.parameters.get("dtype", "float") }}")
{% endif %}
{% endfor %}
    return df


def run_bronze_{{ source }}():
    df = load_{{ source }}()
    df = apply_transformations(df)
    output_path = get_bronze_path(SOURCE_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


if __name__ == "__main__":
    run_bronze_{{ source }}()
''',

    "silver.py.j2": '''import pandas as pd
from config import SOURCES, get_bronze_path, get_silver_path


def load_bronze_outputs() -> dict:
    return {name: pd.read_parquet(get_bronze_path(name)) for name in SOURCES.keys()}


def merge_sources(bronze_outputs: dict) -> pd.DataFrame:
    base_source = "{{ config.sources[0].name }}"
    merged = bronze_outputs[base_source]
{% for join in config.silver.joins %}
    merged = merged.merge(
        bronze_outputs["{{ join.right_source }}"],
        left_on="{{ join.left_key }}",
        right_on="{{ join.right_key }}",
        how="{{ join.how }}"
    )
{% endfor %}
    return merged


def run_silver_merge():
    bronze_outputs = load_bronze_outputs()
    silver = merge_sources(bronze_outputs)
    output_path = get_silver_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(output_path, index=False)
    return silver


if __name__ == "__main__":
    run_silver_merge()
''',

    "gold.py.j2": '''import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from config import (get_silver_path, get_gold_path, get_feast_data_path,
                    TARGET_COLUMN, RECOMMENDATIONS_HASH, FEAST_REPO_PATH,
                    FEAST_FEATURE_VIEW, FEAST_ENTITY_KEY, FEAST_TIMESTAMP_COL)


def load_silver() -> pd.DataFrame:
    return pd.read_parquet(get_silver_path())


def apply_encodings(df: pd.DataFrame) -> pd.DataFrame:
{% for enc in config.gold.encodings %}
{% if enc.parameters.get("method") == "one_hot" %}
    df = pd.get_dummies(df, columns=["{{ enc.column }}"], prefix="{{ enc.column }}")
{% elif enc.parameters.get("method") == "label" %}
    df["{{ enc.column }}"] = LabelEncoder().fit_transform(df["{{ enc.column }}"].astype(str))
{% endif %}
{% endfor %}
    return df


def apply_scaling(df: pd.DataFrame) -> pd.DataFrame:
{% for scale in config.gold.scalings %}
{% if scale.parameters.get("method") == "standard" %}
    df["{{ scale.column }}"] = StandardScaler().fit_transform(df[["{{ scale.column }}"]])
{% elif scale.parameters.get("method") == "minmax" %}
    df["{{ scale.column }}"] = MinMaxScaler().fit_transform(df[["{{ scale.column }}"]])
{% endif %}
{% endfor %}
    return df


def get_feature_version_tag() -> str:
    if RECOMMENDATIONS_HASH:
        return f"v1.0.0_{RECOMMENDATIONS_HASH}"
    return "v1.0.0"


def add_feast_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Add event_timestamp column required by Feast for point-in-time joins."""
    if FEAST_TIMESTAMP_COL not in df.columns:
        df[FEAST_TIMESTAMP_COL] = datetime.now()
    return df


def materialize_to_feast(df: pd.DataFrame) -> None:
    """Write features to Feast offline store (parquet file for FileSource).

    Excludes original_* columns which contain holdout ground truth values.
    These columns are reserved for scoring validation and must never leak into training.
    """
    feast_path = get_feast_data_path()
    feast_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure entity key and timestamp columns exist
    df_feast = df.copy()
    df_feast = add_feast_timestamp(df_feast)

    # Exclude original_* columns from Feast (holdout ground truth - prevents data leakage)
    original_cols = [c for c in df_feast.columns if c.startswith("original_")]
    if original_cols:
        print(f"  Excluding holdout columns from Feast: {original_cols}")
        df_feast = df_feast.drop(columns=original_cols, errors="ignore")

    # Save to Feast data location
    df_feast.to_parquet(feast_path, index=False)
    print(f"Features materialized to Feast: {feast_path}")
    print(f"  Entity key: {FEAST_ENTITY_KEY}")
    print(f"  Feature view: {FEAST_FEATURE_VIEW}")
    print(f"  Rows: {len(df_feast):,}")


def run_gold_features():
    silver = load_silver()
    gold = apply_encodings(silver)
    gold = apply_scaling(gold)

    # Save to gold parquet (backward compatibility)
    output_path = get_gold_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gold.attrs["recommendations_hash"] = RECOMMENDATIONS_HASH
    gold.attrs["feature_version"] = get_feature_version_tag()
    gold.to_parquet(output_path, index=False)
    print(f"Gold features saved with version: {get_feature_version_tag()}")

    # Materialize to Feast for training/serving consistency
    materialize_to_feast(gold)

    return gold


if __name__ == "__main__":
    run_gold_features()
''',

    "training.py.j2": '''import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
from pathlib import Path
from feast import FeatureStore
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, accuracy_score)
from config import (TARGET_COLUMN, PIPELINE_NAME, RECOMMENDATIONS_HASH, MLFLOW_TRACKING_URI,
                    FEAST_REPO_PATH, FEAST_FEATURE_VIEW, FEAST_ENTITY_KEY, FEAST_TIMESTAMP_COL,
                    get_feast_data_path)


def get_training_data_from_feast() -> pd.DataFrame:
    """Retrieve training data from Feast for training/serving consistency.

    Uses get_historical_features for point-in-time correct feature retrieval.
    This ensures training uses the exact same feature retrieval path as inference.
    """
    feast_path = Path(FEAST_REPO_PATH)

    # Check if Feast repo is initialized
    if not (feast_path / "feature_store.yaml").exists():
        print("Feast repo not initialized, falling back to parquet file")
        return pd.read_parquet(get_feast_data_path())

    try:
        store = FeatureStore(repo_path=str(feast_path))

        # Read the materialized features to get entity keys and timestamps
        features_df = pd.read_parquet(get_feast_data_path())

        # Create entity dataframe for historical feature retrieval
        entity_df = features_df[[FEAST_ENTITY_KEY, FEAST_TIMESTAMP_COL]].copy()

        # Get all feature names (excluding entity key, timestamp, target, and holdout ground truth)
        exclude_cols = {FEAST_ENTITY_KEY, FEAST_TIMESTAMP_COL, TARGET_COLUMN}
        feature_cols = [c for c in features_df.columns
                        if c not in exclude_cols and not c.startswith("original_")]

        # Build feature references
        feature_refs = [f"{FEAST_FEATURE_VIEW}:{col}" for col in feature_cols]

        print(f"Retrieving {len(feature_refs)} features from Feast...")
        print(f"  Feature view: {FEAST_FEATURE_VIEW}")
        print(f"  Entity key: {FEAST_ENTITY_KEY}")

        # Get historical features with point-in-time correctness
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs
        ).to_df()

        # Add target column back
        training_df = training_df.merge(
            features_df[[FEAST_ENTITY_KEY, TARGET_COLUMN]],
            on=FEAST_ENTITY_KEY,
            how="left"
        )

        print(f"  Retrieved {len(training_df):,} rows, {len(training_df.columns)} columns")
        return training_df

    except Exception as e:
        print(f"Feast retrieval failed ({e}), falling back to parquet file")
        return pd.read_parquet(get_feast_data_path())


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for model training.

    Explicitly excludes original_* columns which contain holdout ground truth.
    These columns are reserved for scoring validation and must never be used in training.
    """
    df = df.copy()

    # Drop Feast metadata columns
    drop_cols = [FEAST_ENTITY_KEY, FEAST_TIMESTAMP_COL]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Exclude original_* columns (holdout ground truth - prevents data leakage)
    original_cols = [c for c in df.columns if c.startswith("original_")]
    df = df.drop(columns=original_cols, errors="ignore")

    # Encode categorical columns
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df.select_dtypes(include=["int64", "float64", "int32", "float32"]).fillna(0)


def compute_metrics(y_true, y_proba, y_pred) -> dict:
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def get_feature_importance(model, feature_names) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = abs(model.coef_[0])
    else:
        return None
    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def log_feature_importance(model, feature_names):
    fi = get_feature_importance(model, feature_names)
    if fi is None:
        return
    fi.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")


def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    mlflow.xgboost.autolog(log_datasets=False)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    params = {"objective": "binary:logistic", "eval_metric": ["auc", "logloss"],
              "max_depth": 6, "learning_rate": 0.1, "seed": 42}
    model = xgb.train(params, dtrain, num_boost_round=100,
                      evals=[(dtrain, "train"), (dtest, "eval")], verbose_eval=False)
    return model


def get_model_name_with_hash(base_name: str) -> str:
    if RECOMMENDATIONS_HASH:
        return f"{base_name}_{RECOMMENDATIONS_HASH}"
    return base_name


def run_experiment():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(PIPELINE_NAME)
    print(f"MLflow tracking: {MLFLOW_TRACKING_URI}")

    # Load training data from Feast (ensures training/serving consistency)
    print("\\nLoading training data from Feast...")
    training_data = get_training_data_from_feast()

    y = training_data[TARGET_COLUMN]
    X = prepare_features(training_data.drop(columns=[TARGET_COLUMN]))
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    sklearn_models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    run_name = get_model_name_with_hash("pipeline_run")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"train_samples": len(X_train), "test_samples": len(X_test), "n_features": X.shape[1]})
        mlflow.set_tag("feature_source", "feast")
        mlflow.set_tag("feast_feature_view", FEAST_FEATURE_VIEW)
        if RECOMMENDATIONS_HASH:
            mlflow.set_tag("recommendations_hash", RECOMMENDATIONS_HASH)
        best_model, best_auc = None, 0

        for name, model in sklearn_models.items():
            with mlflow.start_run(run_name=name, nested=True):
                if RECOMMENDATIONS_HASH:
                    mlflow.set_tag("recommendations_hash", RECOMMENDATIONS_HASH)
                mlflow.set_tag("feature_source", "feast")
                model.fit(X_train, y_train)
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                metrics = compute_metrics(y_test, y_proba, y_pred)
                cv = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
                mlflow.log_metrics({**metrics, "cv_mean": cv.mean(), "cv_std": cv.std()})
                log_feature_importance(model, feature_names)
                model_artifact_name = get_model_name_with_hash(f"model_{name}")
                mlflow.sklearn.log_model(model, model_artifact_name)
                print(f"{name}: ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1']:.4f}")
                if metrics["roc_auc"] > best_auc:
                    best_auc, best_model = metrics["roc_auc"], name

        with mlflow.start_run(run_name="xgboost", nested=True):
            if RECOMMENDATIONS_HASH:
                mlflow.set_tag("recommendations_hash", RECOMMENDATIONS_HASH)
            mlflow.set_tag("feature_source", "feast")
            xgb_model = train_xgboost(X_train, y_train, X_test, y_test, feature_names)
            dtest = xgb.DMatrix(X_test, feature_names=feature_names)
            y_proba = xgb_model.predict(dtest)
            y_pred = (y_proba > 0.5).astype(int)
            metrics = compute_metrics(y_test, y_proba, y_pred)
            mlflow.log_metrics(metrics)
            importance = xgb_model.get_score(importance_type="gain")
            fi = pd.DataFrame({"feature": importance.keys(), "importance": importance.values()})
            fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
            fi.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            print(f"xgboost: ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1']:.4f}")
            if metrics["roc_auc"] > best_auc:
                best_auc, best_model = metrics["roc_auc"], "xgboost"

        mlflow.set_tag("best_model", best_model)
        mlflow.log_metric("best_roc_auc", best_auc)
        print(f"Best: {best_model} (ROC-AUC={best_auc:.4f})")


if __name__ == "__main__":
    run_experiment()
''',

    "runner.py.j2": '''from concurrent.futures import ThreadPoolExecutor
from config import PIPELINE_NAME
{% for source in config.sources %}
from bronze.bronze_{{ source.name }} import run_bronze_{{ source.name }}
{% endfor %}
from silver.silver_merge import run_silver_merge
from gold.gold_features import run_gold_features
from training.ml_experiment import run_experiment


def run_pipeline():
    print(f"Starting pipeline: {PIPELINE_NAME}")
    with ThreadPoolExecutor(max_workers={{ config.sources|length }}) as executor:
        bronze_futures = [
{% for source in config.sources %}
            executor.submit(run_bronze_{{ source.name }}),
{% endfor %}
        ]
        for f in bronze_futures:
            f.result()
    print("Bronze complete")
    run_silver_merge()
    print("Silver complete")
    run_gold_features()
    print("Gold complete")
    run_experiment()
    print("Training complete")


if __name__ == "__main__":
    run_pipeline()
''',

    "run_all.py.j2": '''"""{{ config.name }} - Pipeline Runner with MLflow UI"""
import sys
import webbrowser
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent))

from config import PIPELINE_NAME, SOURCES, MLFLOW_TRACKING_URI
{% for source in config.sources %}
from bronze.bronze_{{ source.name }} import run_bronze_{{ source.name }}
{% endfor %}
from silver.silver_merge import run_silver_merge
from gold.gold_features import run_gold_features
from training.ml_experiment import run_experiment


def run_bronze_parallel():
    bronze_funcs = [
{% for source in config.sources %}
        run_bronze_{{ source.name }},
{% endfor %}
    ]
    with ThreadPoolExecutor(max_workers={{ config.sources|length }}) as ex:
        list(ex.map(lambda f: f(), bronze_funcs))


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def start_mlflow_ui():
    port = 5050
    if is_port_in_use(port):
        print(f"\\n⚠ Port {port} is already in use.")
        print(f"  Either mlflow is already running, or kill the old process:")
        print(f"  pkill -f 'mlflow ui'")
        print(f"\\n  Opening browser to existing server...")
        webbrowser.open(f"http://localhost:{port}")
        return None

    print(f"\\nStarting MLflow UI (tracking: {MLFLOW_TRACKING_URI})...")
    process = subprocess.Popen(
        ["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI, "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(2)
    webbrowser.open(f"http://localhost:{port}")
    print(f"MLflow UI running at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    return process


def run_pipeline():
    print(f"Running {PIPELINE_NAME}")
    print("=" * 50)

    print("\\n[1/4] Bronze (parallel)...")
    run_bronze_parallel()
    print("Bronze complete")

    print("\\n[2/4] Silver...")
    run_silver_merge()
    print("Silver complete")

    print("\\n[3/4] Gold...")
    run_gold_features()
    print("Gold complete")

    print("\\n[4/4] Training...")
    run_experiment()
    print("Training complete")

    print("\\n" + "=" * 50)
    print("Pipeline finished!")

    mlflow_process = start_mlflow_ui()
    if mlflow_process:
        try:
            mlflow_process.wait()
        except KeyboardInterrupt:
            mlflow_process.terminate()
            print("\\nMLflow UI stopped")


if __name__ == "__main__":
    run_pipeline()
''',

    "workflow.json.j2": '''{
  "name": "{{ config.name }}_pipeline",
  "tasks": [
{% for source in config.sources %}
    {
      "task_key": "bronze_{{ source.name }}",
      "notebook_task": {
        "notebook_path": "/Workspace/orchestration/{{ config.name }}/bronze/bronze_{{ source.name }}"
      }
    },
{% endfor %}
    {
      "task_key": "silver_merge",
      "depends_on": [
{% for source in config.sources %}
        {"task_key": "bronze_{{ source.name }}"}{{ "," if not loop.last else "" }}
{% endfor %}
      ],
      "notebook_task": {
        "notebook_path": "/Workspace/orchestration/{{ config.name }}/silver/silver_merge"
      }
    },
    {
      "task_key": "gold_features",
      "depends_on": [{"task_key": "silver_merge"}],
      "notebook_task": {
        "notebook_path": "/Workspace/orchestration/{{ config.name }}/gold/gold_features"
      }
    },
    {
      "task_key": "ml_experiment",
      "depends_on": [{"task_key": "gold_features"}],
      "notebook_task": {
        "notebook_path": "/Workspace/orchestration/{{ config.name }}/training/ml_experiment"
      }
    }
  ]
}
''',

    "feature_store.yaml.j2": '''project: {{ config.name }}
registry: data/registry.db
provider: local
online_store:
  type: sqlite
  path: data/online_store.db
offline_store:
  type: file
entity_key_serialization_version: 2
''',

    "features.py.j2": '''"""Feast Feature Definitions for {{ config.name }}

Auto-generated feature view definitions for training/serving consistency.
Feature version: {{ config.recommendations_hash or "unversioned" }}
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Float64, Int64, String

# Entity definition
{{ config.feast.entity_name if config.feast else "customer" }} = Entity(
    name="{{ config.feast.entity_name if config.feast else 'customer' }}",
    join_keys=["{{ config.feast.entity_key if config.feast else config.sources[0].entity_key }}"],
    description="Primary entity for {{ config.name }} pipeline"
)

# File source pointing to materialized features
{{ config.feast.feature_view_name if config.feast else config.name + '_features' }}_source = FileSource(
    path="data/{{ config.feast.feature_view_name if config.feast else config.name + '_features' }}.parquet",
    timestamp_field="{{ config.feast.timestamp_column if config.feast else 'event_timestamp' }}"
)

# Feature view definition
# Note: Features are dynamically determined from the parquet file schema
# This is a placeholder that gets populated when feast apply is run
{{ config.feast.feature_view_name if config.feast else config.name + '_features' }} = FeatureView(
    name="{{ config.feast.feature_view_name if config.feast else config.name + '_features' }}",
    entities=[{{ config.feast.entity_name if config.feast else "customer" }}],
    ttl=timedelta(days={{ config.feast.ttl_days if config.feast else 365 }}),
    source={{ config.feast.feature_view_name if config.feast else config.name + '_features' }}_source,
    tags={
        "pipeline": "{{ config.name }}",
        "recommendations_hash": "{{ config.recommendations_hash or 'none' }}",
        "version": "v1.0.0_{{ config.recommendations_hash or 'unversioned' }}"
    }
)
''',

    "run_scoring.py.j2": '''"""{{ config.name }} - Scoring Pipeline

Generates predictions for holdout records using Feast features and MLflow model.
Compares predictions against original values for validation.
"""
import pandas as pd
import numpy as np
import mlflow
import yaml
from pathlib import Path
from datetime import datetime
from feast import FeatureStore
from config import (PIPELINE_NAME, TARGET_COLUMN, RECOMMENDATIONS_HASH, MLFLOW_TRACKING_URI,
                    FEAST_REPO_PATH, FEAST_FEATURE_VIEW, FEAST_ENTITY_KEY, FEAST_TIMESTAMP_COL,
                    get_feast_data_path)

ORIGINAL_COLUMN = f"original_{TARGET_COLUMN}"
PREDICTIONS_PATH = Path("data/scoring/predictions.parquet")


def load_holdout_manifest() -> dict:
    manifest_path = Path("holdout_manifest.yaml")
    if not manifest_path.exists():
        manifest_path = Path("../explorations/holdout_manifest.yaml")
    with open(manifest_path) as f:
        return yaml.safe_load(f)


def get_scoring_data() -> pd.DataFrame:
    features_df = pd.read_parquet(get_feast_data_path())
    if ORIGINAL_COLUMN not in features_df.columns:
        raise ValueError(f"Column {ORIGINAL_COLUMN} not found - run holdout creation first")
    scoring_mask = features_df[TARGET_COLUMN].isna() & features_df[ORIGINAL_COLUMN].notna()
    return features_df[scoring_mask].copy()


def get_scoring_features_from_feast(scoring_df: pd.DataFrame) -> pd.DataFrame:
    feast_path = Path(FEAST_REPO_PATH)
    if not (feast_path / "feature_store.yaml").exists():
        print("Feast not initialized, using parquet directly")
        return scoring_df
    try:
        store = FeatureStore(repo_path=str(feast_path))
        entity_df = scoring_df[[FEAST_ENTITY_KEY, FEAST_TIMESTAMP_COL]].copy()
        exclude_cols = {FEAST_ENTITY_KEY, FEAST_TIMESTAMP_COL, TARGET_COLUMN, ORIGINAL_COLUMN}
        feature_cols = [c for c in scoring_df.columns if c not in exclude_cols and not c.startswith("original_")]
        feature_refs = [f"{FEAST_FEATURE_VIEW}:{col}" for col in feature_cols]
        result_df = store.get_online_features(
            features=feature_refs,
            entity_rows=[{FEAST_ENTITY_KEY: eid} for eid in scoring_df[FEAST_ENTITY_KEY]]
        ).to_df()
        result_df[ORIGINAL_COLUMN] = scoring_df[ORIGINAL_COLUMN].values
        result_df[FEAST_ENTITY_KEY] = scoring_df[FEAST_ENTITY_KEY].values
        return result_df
    except Exception as e:
        print(f"Feast retrieval failed ({e}), using parquet")
        return scoring_df


def load_best_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(PIPELINE_NAME)
    if not experiment:
        raise ValueError(f"Experiment {PIPELINE_NAME} not found")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              filter_string=f"tags.recommendations_hash = '{RECOMMENDATIONS_HASH}'",
                              order_by=["metrics.best_roc_auc DESC"], max_results=1)
    if not runs:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                                  order_by=["metrics.best_roc_auc DESC"], max_results=1)
    if not runs:
        raise ValueError("No runs found")
    run = runs[0]
    best_model_tag = run.data.tags.get("best_model", "random_forest")
    model_name = f"model_{best_model_tag}"
    if RECOMMENDATIONS_HASH:
        model_name = f"{model_name}_{RECOMMENDATIONS_HASH}"
    model_uri = f"runs:/{run.info.run_id}/{model_name}"
    print(f"Loading model: {model_uri}")
    return mlflow.sklearn.load_model(model_uri), run.info.run_id


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    drop_cols = [FEAST_ENTITY_KEY, FEAST_TIMESTAMP_COL, ORIGINAL_COLUMN]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in df.columns if c.startswith("original_")], errors="ignore")
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df.select_dtypes(include=["int64", "float64", "int32", "float32"]).fillna(0)


def compute_validation_metrics(y_true, y_pred, y_proba) -> dict:
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
    }


def run_scoring():
    print(f"Scoring Pipeline: {PIPELINE_NAME}")
    print("=" * 50)
    scoring_df = get_scoring_data()
    print(f"\\nScoring records: {len(scoring_df):,}")
    features_df = get_scoring_features_from_feast(scoring_df)
    model, run_id = load_best_model()
    X = prepare_features(features_df)
    y_true = features_df[ORIGINAL_COLUMN].values
    print(f"\\nGenerating predictions...")
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_validation_metrics(y_true, y_pred, y_proba)
    print(f"\\nValidation Metrics (vs original values):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    results_df = pd.DataFrame({
        FEAST_ENTITY_KEY: scoring_df[FEAST_ENTITY_KEY].values,
        "prediction": y_pred,
        "probability": y_proba,
        "actual": y_true,
        "correct": (y_pred == y_true).astype(int)
    })
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"\\nPredictions saved: {PREDICTIONS_PATH}")
    print(f"Correct: {results_df['correct'].sum():,}/{len(results_df):,} ({results_df['correct'].mean():.1%})")
    return results_df, metrics


if __name__ == "__main__":
    run_scoring()
''',

    "scoring_dashboard.ipynb.j2": '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# {{ config.name }} - Scoring Dashboard\\n",
    "\\n",
    "Interactive dashboard for exploring scoring results and understanding predictions.\\n",
    "\\n",
    "**Features:**\\n",
    "- Summary metrics and model performance\\n",
    "- Customer-by-customer prediction browser\\n",
    "- SHAP-based feature explanations\\n",
    "- Comparison against ground truth (holdout validation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\\n",
    "from pathlib import Path\\n",
    "sys.path.insert(0, str(Path.cwd().parent))\\n",
    "\\n",
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import mlflow\\n",
    "import shap\\n",
    "import matplotlib.pyplot as plt\\n",
    "from IPython.display import display, HTML\\n",
    "from config import (PIPELINE_NAME, TARGET_COLUMN, MLFLOW_TRACKING_URI,\\n",
    "                    FEAST_ENTITY_KEY, get_feast_data_path)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Scoring Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "PREDICTIONS_PATH = Path(\\"data/scoring/predictions.parquet\\")\\n",
    "ORIGINAL_COLUMN = f\\"original_{TARGET_COLUMN}\\"\\n",
    "\\n",
    "# Load predictions\\n",
    "predictions_df = pd.read_parquet(PREDICTIONS_PATH)\\n",
    "print(f\\"Loaded {len(predictions_df):,} predictions\\")\\n",
    "predictions_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load feature data for explanations\\n",
    "features_df = pd.read_parquet(get_feast_data_path())\\n",
    "scoring_mask = features_df[TARGET_COLUMN].isna() & features_df[ORIGINAL_COLUMN].notna()\\n",
    "scoring_features = features_df[scoring_mask].copy()\\n",
    "print(f\\"Features for {len(scoring_features):,} scoring records\\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Summary Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import (accuracy_score, precision_score, recall_score,\\n",
    "                             f1_score, roc_auc_score, confusion_matrix)\\n",
    "\\n",
    "y_true = predictions_df[\\"actual\\"]\\n",
    "y_pred = predictions_df[\\"prediction\\"]\\n",
    "y_proba = predictions_df[\\"probability\\"]\\n",
    "\\n",
    "metrics = {\\n",
    "    \\"Accuracy\\": accuracy_score(y_true, y_pred),\\n",
    "    \\"Precision\\": precision_score(y_true, y_pred, zero_division=0),\\n",
    "    \\"Recall\\": recall_score(y_true, y_pred, zero_division=0),\\n",
    "    \\"F1 Score\\": f1_score(y_true, y_pred, zero_division=0),\\n",
    "    \\"ROC-AUC\\": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0\\n",
    "}\\n",
    "\\n",
    "print(\\"\\\\n=== Scoring Validation Metrics ===\\")\\n",
    "for name, value in metrics.items():\\n",
    "    print(f\\"  {name}: {value:.4f}\\")\\n",
    "\\n",
    "# Confusion matrix\\n",
    "cm = confusion_matrix(y_true, y_pred)\\n",
    "print(f\\"\\\\nConfusion Matrix:\\")\\n",
    "print(f\\"  TN={cm[0,0]:,}  FP={cm[0,1]:,}\\")\\n",
    "print(f\\"  FN={cm[1,0]:,}  TP={cm[1,1]:,}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize metrics\\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\\n",
    "\\n",
    "# ROC curve (if possible)\\n",
    "from sklearn.metrics import roc_curve\\n",
    "fpr, tpr, _ = roc_curve(y_true, y_proba)\\n",
    "axes[0].plot(fpr, tpr, \\"b-\\", lw=2, label=f\\"ROC (AUC={metrics['ROC-AUC']:.3f})\\")\\n",
    "axes[0].plot([0, 1], [0, 1], \\"k--\\", lw=1)\\n",
    "axes[0].set_xlabel(\\"False Positive Rate\\")\\n",
    "axes[0].set_ylabel(\\"True Positive Rate\\")\\n",
    "axes[0].set_title(\\"ROC Curve\\")\\n",
    "axes[0].legend()\\n",
    "\\n",
    "# Probability distribution\\n",
    "axes[1].hist(y_proba[y_true == 0], bins=30, alpha=0.5, label=\\"Actual=0\\", color=\\"blue\\")\\n",
    "axes[1].hist(y_proba[y_true == 1], bins=30, alpha=0.5, label=\\"Actual=1\\", color=\\"red\\")\\n",
    "axes[1].axvline(x=0.5, color=\\"black\\", linestyle=\\"--\\", label=\\"Threshold\\")\\n",
    "axes[1].set_xlabel(\\"Predicted Probability\\")\\n",
    "axes[1].set_ylabel(\\"Count\\")\\n",
    "axes[1].set_title(\\"Probability Distribution\\")\\n",
    "axes[1].legend()\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Load Model for Explanations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)\\n",
    "client = mlflow.tracking.MlflowClient()\\n",
    "\\n",
    "experiment = client.get_experiment_by_name(PIPELINE_NAME)\\n",
    "runs = client.search_runs(\\n",
    "    experiment_ids=[experiment.experiment_id],\\n",
    "    order_by=[\\"metrics.best_roc_auc DESC\\"],\\n",
    "    max_results=1\\n",
    ")\\n",
    "run = runs[0]\\n",
    "\\n",
    "best_model_tag = run.data.tags.get(\\"best_model\\", \\"random_forest\\")\\n",
    "model_uri = f\\"runs:/{run.info.run_id}/model_{best_model_tag}\\"\\n",
    "print(f\\"Loading model: {model_uri}\\")\\n",
    "model = mlflow.sklearn.load_model(model_uri)\\n",
    "print(f\\"Model type: {type(model).__name__}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare features for SHAP\\n",
    "from sklearn.preprocessing import LabelEncoder\\n",
    "\\n",
    "def prepare_features(df):\\n",
    "    df = df.copy()\\n",
    "    drop_cols = [FEAST_ENTITY_KEY, \\"event_timestamp\\", ORIGINAL_COLUMN, TARGET_COLUMN]\\n",
    "    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors=\\"ignore\\")\\n",
    "    df = df.drop(columns=[c for c in df.columns if c.startswith(\\"original_\\")], errors=\\"ignore\\")\\n",
    "    for col in df.select_dtypes(include=[\\"object\\", \\"category\\"]).columns:\\n",
    "        df[col] = LabelEncoder().fit_transform(df[col].astype(str))\\n",
    "    return df.select_dtypes(include=[\\"int64\\", \\"float64\\", \\"int32\\", \\"float32\\"]).fillna(0)\\n",
    "\\n",
    "X = prepare_features(scoring_features)\\n",
    "feature_names = list(X.columns)\\n",
    "print(f\\"Prepared {len(feature_names)} features for SHAP analysis\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create SHAP explainer\\n",
    "print(\\"Creating SHAP explainer (may take a moment)...\\")\\n",
    "\\n",
    "# Use background sample for efficiency\\n",
    "background_size = min(100, len(X))\\n",
    "background = shap.sample(X, background_size)\\n",
    "\\n",
    "if hasattr(model, \\"predict_proba\\"):\\n",
    "    explainer = shap.Explainer(model.predict_proba, background, feature_names=feature_names)\\n",
    "else:\\n",
    "    explainer = shap.Explainer(model, background, feature_names=feature_names)\\n",
    "\\n",
    "print(\\"Computing SHAP values...\\")\\n",
    "shap_values = explainer(X)\\n",
    "print(f\\"SHAP values computed for {len(shap_values)} records\\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Global Feature Importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use positive class SHAP values if multi-output\\n",
    "if len(shap_values.shape) == 3:\\n",
    "    shap_vals = shap_values[:, :, 1]  # Positive class\\n",
    "else:\\n",
    "    shap_vals = shap_values\\n",
    "\\n",
    "plt.figure(figsize=(10, 8))\\n",
    "shap.summary_plot(shap_vals, X, feature_names=feature_names, show=False, max_display=20)\\n",
    "plt.title(\\"Feature Importance (SHAP Summary)\\")\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Mean absolute SHAP values\\n",
    "mean_shap = np.abs(shap_vals.values).mean(axis=0)\\n",
    "importance_df = pd.DataFrame({\\n",
    "    \\"feature\\": feature_names,\\n",
    "    \\"importance\\": mean_shap\\n",
    "}).sort_values(\\"importance\\", ascending=False)\\n",
    "\\n",
    "print(\\"Top 15 Most Important Features:\\")\\n",
    "display(importance_df.head(15))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Customer Browser"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create combined dataset for browsing\\n",
    "browser_df = predictions_df.merge(\\n",
    "    scoring_features[[FEAST_ENTITY_KEY] + feature_names],\\n",
    "    on=FEAST_ENTITY_KEY,\\n",
    "    how=\\"left\\"\\n",
    ")\\n",
    "\\n",
    "print(f\\"Customer browser ready with {len(browser_df):,} records\\")\\n",
    "print(f\\"\\\\nPrediction Distribution:\\")\\n",
    "print(f\\"  Predicted Positive: {(browser_df['prediction'] == 1).sum():,}\\")\\n",
    "print(f\\"  Predicted Negative: {(browser_df['prediction'] == 0).sum():,}\\")\\n",
    "print(f\\"\\\\nCorrect Predictions: {browser_df['correct'].sum():,}/{len(browser_df):,} ({browser_df['correct'].mean():.1%})\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def show_customer(idx: int):\\n",
    "    \\"\\"\\"Display details and SHAP explanation for a single customer.\\"\\"\\"\\n",
    "    row = browser_df.iloc[idx]\\n",
    "    entity_id = row[FEAST_ENTITY_KEY]\\n",
    "    \\n",
    "    print(f\\"=== Customer {entity_id} ===\\")\\n",
    "    print(f\\"Prediction: {int(row['prediction'])} (probability: {row['probability']:.3f})\\")\\n",
    "    print(f\\"Actual: {int(row['actual'])}\\")\\n",
    "    print(f\\"Correct: {'Yes' if row['correct'] else 'No'}\\")\\n",
    "    print()\\n",
    "    \\n",
    "    # Show top features\\n",
    "    feature_vals = X.iloc[idx]\\n",
    "    if len(shap_values.shape) == 3:\\n",
    "        customer_shap = shap_values[idx, :, 1].values\\n",
    "    else:\\n",
    "        customer_shap = shap_values[idx].values\\n",
    "    \\n",
    "    feature_impact = pd.DataFrame({\\n",
    "        \\"feature\\": feature_names,\\n",
    "        \\"value\\": feature_vals.values,\\n",
    "        \\"shap_impact\\": customer_shap\\n",
    "    }).sort_values(\\"shap_impact\\", key=abs, ascending=False)\\n",
    "    \\n",
    "    print(\\"Top Contributing Features:\\")\\n",
    "    display(feature_impact.head(10))\\n",
    "    \\n",
    "    # Waterfall plot\\n",
    "    plt.figure(figsize=(10, 6))\\n",
    "    if len(shap_values.shape) == 3:\\n",
    "        shap.plots.waterfall(shap_values[idx, :, 1], max_display=10, show=False)\\n",
    "    else:\\n",
    "        shap.plots.waterfall(shap_values[idx], max_display=10, show=False)\\n",
    "    plt.title(f\\"SHAP Explanation for Customer {entity_id}\\")\\n",
    "    plt.tight_layout()\\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Show first few customers\\n",
    "print(\\"Showing first 3 customers:\\\\n\\")\\n",
    "for i in range(min(3, len(browser_df))):\\n",
    "    show_customer(i)\\n",
    "    print(\\"\\\\n\\" + \\"=\\" * 60 + \\"\\\\n\\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Interactive Customer Lookup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Change this to browse different customers\\n",
    "CUSTOMER_INDEX = 0  # Change to explore different customers (0 to N-1)\\n",
    "\\n",
    "show_customer(CUSTOMER_INDEX)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Look up by entity ID\\n",
    "def lookup_customer(entity_id):\\n",
    "    \\"\\"\\"Find and display a customer by their entity ID.\\"\\"\\"\\n",
    "    mask = browser_df[FEAST_ENTITY_KEY] == entity_id\\n",
    "    if not mask.any():\\n",
    "        print(f\\"Customer {entity_id} not found in scoring set\\")\\n",
    "        return\\n",
    "    idx = browser_df[mask].index[0]\\n",
    "    # Find position in X\\n",
    "    x_idx = browser_df.index.get_loc(idx)\\n",
    "    show_customer(x_idx)\\n",
    "\\n",
    "# Example: lookup_customer(12345)\\n",
    "print(\\"Available entity IDs (first 10):\\")\\n",
    "print(browser_df[FEAST_ENTITY_KEY].head(10).tolist())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Error Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyze misclassified customers\\n",
    "incorrect = browser_df[browser_df[\\"correct\\"] == 0]\\n",
    "print(f\\"Misclassified customers: {len(incorrect):,}\\")\\n",
    "\\n",
    "# False positives (predicted 1, actual 0)\\n",
    "fp = incorrect[incorrect[\\"prediction\\"] == 1]\\n",
    "print(f\\"  False Positives: {len(fp):,}\\")\\n",
    "\\n",
    "# False negatives (predicted 0, actual 1)  \\n",
    "fn = incorrect[incorrect[\\"prediction\\"] == 0]\\n",
    "print(f\\"  False Negatives: {len(fn):,}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Show example false positive\\n",
    "if len(fp) > 0:\\n",
    "    print(\\"\\\\n=== Example False Positive ===\\")\\n",
    "    fp_idx = browser_df.index.get_loc(fp.index[0])\\n",
    "    show_customer(fp_idx)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Show example false negative\\n",
    "if len(fn) > 0:\\n",
    "    print(\\"\\\\n=== Example False Negative ===\\")\\n",
    "    fn_idx = browser_df.index.get_loc(fn.index[0])\\n",
    "    show_customer(fn_idx)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Export Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Export detailed results with feature importance\\n",
    "output_dir = Path(\\"data/scoring\\")\\n",
    "\\n",
    "# Save global feature importance\\n",
    "importance_df.to_csv(output_dir / \\"feature_importance.csv\\", index=False)\\n",
    "print(f\\"Feature importance saved to {output_dir / 'feature_importance.csv'}\\")\\n",
    "\\n",
    "# Save detailed predictions with SHAP values for top features\\n",
    "top_features = importance_df.head(10)[\\"feature\\"].tolist()\\n",
    "detailed_df = predictions_df.copy()\\n",
    "\\n",
    "for feat in top_features:\\n",
    "    feat_idx = feature_names.index(feat)\\n",
    "    if len(shap_values.shape) == 3:\\n",
    "        detailed_df[f\\"shap_{feat}\\"] = shap_values[:, feat_idx, 1].values\\n",
    "    else:\\n",
    "        detailed_df[f\\"shap_{feat}\\"] = shap_values[:, feat_idx].values\\n",
    "\\n",
    "detailed_df.to_parquet(output_dir / \\"predictions_with_shap.parquet\\", index=False)\\n",
    "print(f\\"Detailed predictions with SHAP saved to {output_dir / 'predictions_with_shap.parquet'}\\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
}


class CodeRenderer:
    def __init__(self):
        self._env = Environment(loader=InlineLoader(TEMPLATES))

    def render_config(self, config: PipelineConfig) -> str:
        return self._env.get_template("config.py.j2").render(config=config)

    def render_bronze(self, source_name: str, bronze_config: BronzeLayerConfig) -> str:
        return self._env.get_template("bronze.py.j2").render(source=source_name, config=bronze_config)

    def render_silver(self, config: PipelineConfig) -> str:
        return self._env.get_template("silver.py.j2").render(config=config)

    def render_gold(self, config: PipelineConfig) -> str:
        return self._env.get_template("gold.py.j2").render(config=config)

    def render_training(self, config: PipelineConfig) -> str:
        return self._env.get_template("training.py.j2").render(config=config)

    def render_runner(self, config: PipelineConfig) -> str:
        return self._env.get_template("runner.py.j2").render(config=config)

    def render_workflow(self, config: PipelineConfig) -> str:
        return self._env.get_template("workflow.json.j2").render(config=config)

    def render_run_all(self, config: PipelineConfig) -> str:
        return self._env.get_template("run_all.py.j2").render(config=config)

    def render_feast_config(self, config: PipelineConfig) -> str:
        return self._env.get_template("feature_store.yaml.j2").render(config=config)

    def render_feast_features(self, config: PipelineConfig) -> str:
        return self._env.get_template("features.py.j2").render(config=config)

    def render_scoring(self, config: PipelineConfig) -> str:
        return self._env.get_template("run_scoring.py.j2").render(config=config)

    def render_dashboard(self, config: PipelineConfig) -> str:
        return self._env.get_template("scoring_dashboard.ipynb.j2").render(config=config)
