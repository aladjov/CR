from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Tuple

import pandas as pd

from customer_retention.core.compat.detection import get_spark_session
from customer_retention.stages.scoring.config import ScoringConfig
from customer_retention.transforms import ArtifactStore, TransformExecutor

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]
    MlflowClient = None  # type: ignore[assignment,misc]

try:
    from feast import FeatureStore
except ImportError:
    FeatureStore = None  # type: ignore[assignment,misc]


class ScoringDataLoader:
    def __init__(self, config: ScoringConfig):
        self.config = config

    def load_gold_features(self) -> pd.DataFrame:
        if self.config.is_databricks:
            return self._load_gold_from_spark()
        return self._load_gold_from_parquet()

    def load_scoring_features(self, scoring_df: pd.DataFrame) -> pd.DataFrame:
        if self.config.is_databricks or not self.config.feast_repo_path:
            return scoring_df
        return self._try_feast_features(scoring_df)

    def load_model(self) -> Tuple[Any, str]:
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.config.pipeline_name)
        if not experiment:
            raise ValueError(f"Experiment '{self.config.pipeline_name}' not found")
        parent_run = self._find_best_parent_run(client, experiment.experiment_id)
        best_model_tag = parent_run.data.tags.get("best_model", "random_forest")
        model_name = f"model_{best_model_tag}"
        if self.config.recommendations_hash:
            model_name = f"{model_name}_{self.config.recommendations_hash}"
        model_run = self._find_model_run(client, experiment.experiment_id, parent_run, best_model_tag)
        model_uri = f"runs:/{model_run.info.run_id}/{model_name}"
        loader_module = mlflow.xgboost if best_model_tag == "xgboost" else mlflow.sklearn
        return loader_module.load_model(model_uri), model_uri

    def load_transforms(self) -> Tuple[list, list]:
        gold_module = self._load_gold_module()
        return gold_module.ENCODINGS, gold_module.SCALINGS

    def prepare_features(
        self, df: pd.DataFrame, transforms: list,
        executor: TransformExecutor, artifact_store: ArtifactStore,
    ) -> pd.DataFrame:
        df = df.copy()
        drop_cols = [self.config.entity_key, self.config.timestamp_column,
                     self.config.original_column, self.config.target_column]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        df = df.drop(columns=[c for c in df.columns if c.startswith("original_")], errors="ignore")
        df = executor.apply_all(df, transforms, fit_mode=False, artifact_store=artifact_store)
        return df.select_dtypes(include=["int64", "float64", "int32", "float32"]).fillna(0)

    def _load_gold_from_spark(self) -> pd.DataFrame:
        spark = get_spark_session()
        if not spark:
            raise RuntimeError("Spark session unavailable on Databricks")
        table_name = f"{self.config.catalog}.{self.config.schema}.gold_features"
        return spark.table(table_name).toPandas()

    def _load_gold_from_parquet(self) -> pd.DataFrame:
        gold_path = self.config.production_dir / "data" / "gold" / self.config.pipeline_name / "features.parquet"
        if not gold_path.exists():
            raise FileNotFoundError(f"Gold features not found at {gold_path}")
        return pd.read_parquet(gold_path)

    def _try_feast_features(self, scoring_df: pd.DataFrame) -> pd.DataFrame:
        feast_path = Path(self.config.feast_repo_path)
        if not (feast_path / "feature_store.yaml").exists():
            return scoring_df
        try:
            store = FeatureStore(repo_path=str(feast_path))
            exclude_cols = {self.config.entity_key, self.config.timestamp_column,
                            self.config.target_column, self.config.original_column}
            feature_cols = [
                c for c in scoring_df.columns
                if c not in exclude_cols and not c.startswith("original_")
            ]
            feature_refs = [f"{self.config.feast_feature_view}:{col}" for col in feature_cols]
            result_df = store.get_online_features(
                features=feature_refs,
                entity_rows=[{self.config.entity_key: eid} for eid in scoring_df[self.config.entity_key]],
            ).to_df()
            result_df[self.config.original_column] = scoring_df[self.config.original_column].values
            result_df[self.config.entity_key] = scoring_df[self.config.entity_key].values
            return result_df
        except Exception:
            return scoring_df

    def _find_best_parent_run(self, client, experiment_id: str):
        if self.config.recommendations_hash:
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f"tags.recommendations_hash = '{self.config.recommendations_hash}'",
                order_by=["metrics.best_roc_auc DESC"],
                max_results=1,
            )
            if runs:
                return runs[0]
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["metrics.best_roc_auc DESC"],
            max_results=1,
        )
        if not runs:
            raise ValueError(f"No runs found in experiment '{self.config.pipeline_name}'")
        return runs[0]

    def _find_model_run(self, client, experiment_id: str, parent_run, model_tag: str):
        child_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
        )
        return next((c for c in child_runs if c.info.run_name == model_tag), parent_run)

    def _load_gold_module(self):
        if self.config.is_databricks:
            raise FileNotFoundError("Gold module not available on Databricks; transforms stored in MLflow artifacts")
        pipeline_dir = self.config.production_dir.parent
        gold_dir = None
        for candidate in [
            self.config.production_dir / "gold",
            pipeline_dir / "gold",
        ]:
            if (candidate / "gold_features.py").exists():
                gold_dir = candidate
                break
        if not gold_dir:
            search_root = self.config.production_dir.parent
            for gf in search_root.rglob("gold_features.py"):
                gold_dir = gf.parent
                break
        if not gold_dir:
            raise FileNotFoundError(f"gold_features.py not found near {self.config.production_dir}")
        spec = importlib.util.spec_from_file_location("_gold_features_gen", str(gold_dir / "gold_features.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules["_gold_features_gen"] = module
        spec.loader.exec_module(module)
        return module
