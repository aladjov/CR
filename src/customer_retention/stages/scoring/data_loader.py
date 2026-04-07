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


_NATIVE_MODEL_FLAVORS: Tuple[str, ...] = ("spark", "xgboost", "sklearn")


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

    def load_model(self, model_tag: str | None = None) -> Tuple[Any, str]:
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        if self.config.is_databricks and self.config.registered_model_name and model_tag is None:
            alias_uri = f"models:/{self.config.registered_model_name}@production"
            alias_flavor = self._detect_flavor_from_uri(alias_uri)
            if alias_flavor in _NATIVE_MODEL_FLAVORS:
                return self._load_by_flavor(alias_flavor, alias_uri), alias_uri
            nested = self._first_native_logged_model(alias_flavor)
            return self._load_by_flavor(nested["flavor"], nested["model_uri"]), nested["model_uri"]
        entry = self._select_logged_model(model_tag)
        return self._load_by_flavor(entry["flavor"], entry["model_uri"]), entry["model_uri"]

    @staticmethod
    def _load_by_flavor(flavor: str, uri: str) -> Any:
        if flavor == "spark":
            return mlflow.spark.load_model(uri)
        if flavor == "xgboost":
            return mlflow.xgboost.load_model(uri)
        return mlflow.sklearn.load_model(uri)

    @staticmethod
    def _detect_flavor_from_uri(uri: str) -> str:
        from mlflow.models import Model as _MlflowModel
        flavors = _MlflowModel.load(uri).flavors or {}
        for name in _NATIVE_MODEL_FLAVORS:
            if name in flavors:
                return name
        return "wrapped"

    def _first_native_logged_model(self, alias_flavor: str) -> dict:
        if not self.config.logged_models:
            raise ValueError(
                f"Registered model '{self.config.registered_model_name}@production' has no native "
                f"MLflow flavor (detected '{alias_flavor}', likely Feature-Engineering-wrapped) and "
                f"ScoringConfig.logged_models is empty so there is no nested fallback. "
                f"Re-run the generated training pipeline (NB10 output) so it persists nested model "
                f"URIs into training_metadata.json."
            )
        nested = self._select_logged_model(None)
        print(
            f"[SCORING] Registered model '{self.config.registered_model_name}@production' has "
            f"flavor='{alias_flavor}' (not a natively-loadable spark/xgboost/sklearn model — "
            f"likely Feature-Engineering-wrapped). Falling back to nested training-run model: "
            f"{nested['model_uri']} (flavor={nested['flavor']})."
        )
        return nested

    def list_trained_model_tags(self) -> List[str]:
        return [self._entry_display_name(e) for e in self.config.logged_models]

    def _select_logged_model(self, model_tag: str | None) -> dict:
        if not self.config.logged_models:
            raise ValueError(
                "ScoringConfig.logged_models is empty. "
                "Re-run the generated training pipeline (NB10 output) so it persists model URIs "
                "into training_metadata.json."
            )
        if model_tag:
            for entry in self.config.logged_models:
                if self._entry_matches(entry, model_tag):
                    return entry
            available = [self._entry_display_name(e) for e in self.config.logged_models]
            raise ValueError(f"Model tag '{model_tag}' not found in logged_models. Available: {available}")
        target_name = self.config.best_model_name
        if target_name:
            for entry in self.config.logged_models:
                if self._entry_matches(entry, target_name):
                    return entry
        return self.config.logged_models[0]

    @staticmethod
    def _entry_display_name(entry: dict) -> str:
        return entry.get("display_name") or entry.get("artifact_path", "")

    @classmethod
    def _entry_matches(cls, entry: dict, name: str) -> bool:
        candidates = {
            entry.get("display_name"),
            entry.get("artifact_path"),
            entry.get("artifact_path", "").removeprefix("model_") or None,
        }
        candidates.discard(None)
        return name in candidates

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
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df = df.drop(columns=[c for c in df.columns if c.startswith("original_")])
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

        from customer_retention.core.compat import _is_spark_pandas, as_spark_df

        if _is_spark_pandas(X):
            spark_df = as_spark_df(X[feature_names])
        else:
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
        if not self.config.mlflow_run_id:
            raise ValueError(
                "ScoringConfig.mlflow_run_id is empty. "
                "Re-run the generated training pipeline so it persists the parent run id."
            )
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        client = MlflowClient()
        import json as _json
        local_path = Path(client.download_artifacts(self.config.mlflow_run_id, "features.json"))
        if local_path.is_dir():
            children = list(local_path.iterdir())
            json_files = [c for c in children if c.suffix == ".json"]
            local_path = json_files[0] if json_files else children[0]
        with open(local_path) as f:
            data = _json.load(f)
        features = data.get("feature_columns")
        if not features:
            raise ValueError(f"features.json in run {self.config.mlflow_run_id} has no 'feature_columns'")
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
