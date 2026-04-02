from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

from customer_retention.core.compat.detection import get_spark_session
from customer_retention.core.config.column_config import select_model_ready_columns
from customer_retention.integrations.adapters.factory import get_delta
from customer_retention.stages.scoring.config import ScoringConfig, _load_module_from_path
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

try:
    from pyspark.ml.feature import VectorAssembler as _VectorAssembler
    from pyspark.ml.functions import vector_to_array as _vector_to_array
except ImportError:  # pragma: no cover
    _VectorAssembler = None  # type: ignore[assignment,misc]
    _vector_to_array = None  # type: ignore[assignment]


class ScoringDataLoader:
    def __init__(self, config: ScoringConfig):
        self.config = config

    def load_gold_features(self) -> Any:
        if self.config.is_databricks:
            return self._load_gold_from_spark()
        return self._load_gold_from_delta()

    def load_gold_features_distributed(self) -> Any:
        if self.config.is_databricks:
            return self._load_gold_from_spark_distributed()
        return self._load_gold_from_delta()

    def load_scoring_features(self, scoring_df: Any) -> Any:
        if self.config.is_databricks or not self.config.feast_repo_path:
            return scoring_df
        feast_path = Path(self.config.feast_repo_path)
        if not (feast_path / "feature_store.yaml").exists():
            return scoring_df
        return self._load_feast_features(scoring_df)

    def _resolve_experiment(self, client):
        experiment = client.get_experiment_by_name(self.config.pipeline_name)
        if not experiment:
            experiment = client.get_experiment_by_name(f"training_{self.config.composite_name}")
        return experiment

    def load_model(self, model_tag: str | None = None) -> Tuple[Any, str]:
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        client = MlflowClient()
        experiment = self._resolve_experiment(client)
        if not experiment:
            raise ValueError(f"Experiment '{self.config.pipeline_name}' not found")
        parent_run = self._find_best_parent_run(client, experiment.experiment_id)
        tag = model_tag or parent_run.data.tags.get("best_model", "random_forest")
        model_name = self._model_artifact_name(tag)
        model_run = self._find_model_run(client, experiment.experiment_id, parent_run, tag)
        model_uri = f"runs:/{model_run.info.run_id}/{model_name}"
        if self.config.is_databricks:
            return mlflow.spark.load_model(model_uri), model_uri
        loader_module = mlflow.xgboost if tag == "xgboost" else mlflow.sklearn
        try:
            return loader_module.load_model(model_uri), model_uri
        except Exception:
            logged_models = client.search_logged_models(experiment_ids=[experiment.experiment_id])
            for lm in logged_models:
                if lm.name == model_name and lm.source_run_id == model_run.info.run_id:
                    return loader_module.load_model(lm.model_uri), lm.model_uri
            raise

    def list_trained_model_tags(self) -> List[str]:
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        client = MlflowClient()
        experiment = self._resolve_experiment(client)
        if not experiment:
            return []
        parent_run = self._find_best_parent_run(client, experiment.experiment_id)
        child_runs = self._find_child_runs(
            client, experiment.experiment_id, parent_run.info.run_id,
        )
        if child_runs:
            return [c.info.run_name for c in child_runs]
        return [parent_run.data.tags.get("best_model", "random_forest")]

    def _model_artifact_name(self, model_tag: str) -> str:
        base = f"model_{model_tag}"
        if self.config.is_databricks:
            return base
        if self.config.recommendations_hash:
            return f"{base}_{self.config.recommendations_hash}"
        return base

    def load_artifact_store(self) -> ArtifactStore | None:
        if self.config.is_databricks:
            return None
        return ArtifactStore.from_manifest(self.config.artifacts_path / "manifest.yaml")

    def load_transforms(self) -> Tuple[list, list]:
        if self.config.is_databricks:
            return [], []
        gold_module = self._load_gold_module()
        return gold_module.ENCODINGS, gold_module.SCALINGS

    def prepare_features(
        self,
        df: Any,
        transforms: list,
        executor: TransformExecutor,
        artifact_store: ArtifactStore | None,
    ) -> Any:
        df = df.copy()
        drop_cols = [
            self.config.entity_key,
            self.config.timestamp_column,
            self.config.original_column,
            self.config.target_column,
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        df = df.drop(columns=[c for c in df.columns if c.startswith("original_")], errors="ignore")
        df = executor.apply_all(df, transforms, fit_mode=False, artifact_store=artifact_store)
        df = select_model_ready_columns(df)
        from customer_retention.core.compat import _is_spark_pandas
        if _is_spark_pandas(df):
            import pandas.api.types as ptypes
            dtypes = df.dtypes
            keep = [c for c, dt in dtypes.items()
                    if ptypes.is_numeric_dtype(dt) or ptypes.is_bool_dtype(dt)]
            return df[keep].fillna(0)
        return df.select_dtypes(include=["number", "bool"]).fillna(0)

    def predict_spark_ml(self, model: Any, X: Any, feature_names: list[str]) -> Any:
        from pyspark.sql import functions as F  # noqa: N812

        from customer_retention.core.compat import normalize_timestamps, pandas_dtype_to_spark_schema

        spark = get_spark_session()
        normalized = normalize_timestamps(X[feature_names])
        schema = pandas_dtype_to_spark_schema(normalized)
        spark_df = spark.createDataFrame(normalized, schema=schema)
        assembler = _VectorAssembler(inputCols=feature_names, outputCol="features", handleInvalid="keep")
        assembled = assembler.transform(spark_df)
        predictions = model.transform(assembled)
        return predictions.select(
            _vector_to_array(F.col("probability")).getItem(1).alias("prob")
        ).toPandas()["prob"].to_numpy()

    def load_training_feature_names(self) -> list[str]:
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        client = MlflowClient()
        experiment = self._resolve_experiment(client)
        if not experiment:
            raise ValueError(f"Experiment '{self.config.pipeline_name}' not found in MLflow")
        parent_run = self._find_best_parent_run(client, experiment.experiment_id)
        import json as _json
        local_path = Path(client.download_artifacts(parent_run.info.run_id, "features.json"))
        if local_path.is_dir():
            children = list(local_path.iterdir())
            json_files = [c for c in children if c.suffix == ".json"]
            local_path = json_files[0] if json_files else children[0]
        with open(local_path) as f:
            data = _json.load(f)
        features = data.get("feature_columns")
        if not features:
            raise ValueError(f"features.json in run {parent_run.info.run_id} has no 'feature_columns'")
        return features

    def _load_gold_from_spark(self) -> Any:
        spark = get_spark_session()
        if not spark:
            raise RuntimeError("Spark session unavailable on Databricks")
        cn = self.config.composite_name
        table_name = f"{self.config.catalog}.{self.config.schema}.gold_features_{cn}"
        return spark.table(table_name).toPandas()

    def _load_gold_from_spark_distributed(self) -> Any:
        spark = get_spark_session()
        if not spark:
            raise RuntimeError("Spark session unavailable on Databricks")
        cn = self.config.composite_name
        table_name = f"{self.config.catalog}.{self.config.schema}.gold_features_{cn}"
        spark_df = spark.table(table_name)
        if hasattr(spark_df, "pandas_api"):
            return spark_df.pandas_api()
        return spark_df.to_pandas_on_spark()

    def _load_gold_from_delta(self) -> Any:
        cn = self.config.composite_name
        gold_path = self.config.production_dir / "data" / "gold" / f"gold_features_{cn}"
        storage = get_delta(force_local=True)
        if not storage.exists(str(gold_path)):
            raise FileNotFoundError(f"Gold features not found at {gold_path}")
        return storage.read(str(gold_path))

    def _load_feast_features(self, scoring_df: Any) -> Any:
        feast_path = Path(self.config.feast_repo_path)
        store = FeatureStore(repo_path=str(feast_path))
        exclude_cols = {
            self.config.entity_key,
            self.config.timestamp_column,
            self.config.target_column,
            self.config.original_column,
        }
        feature_cols = [c for c in scoring_df.columns if c not in exclude_cols and not c.startswith("original_")]
        feature_refs = [f"{self.config.feast_feature_view}:{col}" for col in feature_cols]
        result_df = store.get_online_features(
            features=feature_refs,
            entity_rows=[{self.config.entity_key: eid} for eid in scoring_df[self.config.entity_key]],
        ).to_df()
        result_df[self.config.original_column] = scoring_df[self.config.original_column].to_numpy()
        result_df[self.config.entity_key] = scoring_df[self.config.entity_key].to_numpy()
        return result_df

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

    def _find_child_runs(self, client, experiment_id: str, parent_run_id: str):
        """Search for child runs, checking all experiments if needed.

        Nested runs with the sqlite backend sometimes land in the Default
        experiment instead of the parent's experiment.
        """
        child_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        )
        if child_runs:
            return child_runs
        all_ids = [
            e.experiment_id
            for e in client.search_experiments()
            if e.experiment_id != experiment_id
        ]
        if all_ids:
            child_runs = client.search_runs(
                experiment_ids=all_ids,
                filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            )
        return child_runs

    def _find_model_run(self, client, experiment_id: str, parent_run, model_tag: str):
        child_runs = self._find_child_runs(client, experiment_id, parent_run.info.run_id)
        return next((c for c in child_runs if c.info.run_name == model_tag), parent_run)

    def _load_gold_module(self):
        if self.config.is_databricks:
            raise FileNotFoundError("Gold module not available on Databricks; transforms stored in MLflow artifacts")
        config_path = self.config.pipeline_dir / "config.py"
        if not config_path.exists():
            raise FileNotFoundError(f"config.py not found at {config_path}")
        cn = self.config.composite_name
        gold_path = self.config.pipeline_dir / "gold" / f"gold_features_{cn}.py"
        if not gold_path.exists():
            gold_path = self.config.pipeline_dir / "gold" / "gold_features.py"
        if not gold_path.exists():
            raise FileNotFoundError(f"gold_features.py not found in {self.config.pipeline_dir / 'gold'}")
        _load_module_from_path("config", config_path)
        return _load_module_from_path("_gold_features_gen", gold_path)
