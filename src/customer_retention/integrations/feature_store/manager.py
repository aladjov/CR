"""Unified feature store manager for leakage-safe feature management.

This module provides a unified interface for feature store operations
that works with both Feast (local) and Databricks (production) backends.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from customer_retention.core.compat import native_pd, safe_drop_duplicates
from customer_retention.stages.temporal import PointInTimeRegistry, SnapshotManager

from .registry import FeatureRegistry


class FeatureStoreBackend(ABC):
    """Abstract base class for feature store backends."""

    @abstractmethod
    def create_feature_table(
        self,
        name: str,
        entity_key: str,
        timestamp_column: str,
        schema: dict[str, str],
        cutoff_date: Optional[datetime] = None,
    ) -> str:
        pass

    @abstractmethod
    def write_features(
        self,
        table_name: str,
        df: Any,
        mode: str = "merge",
        cutoff_date: Optional[datetime] = None,
    ) -> None:
        pass

    @abstractmethod
    def get_historical_features(
        self,
        entity_df: Any,
        feature_refs: list[str],
        timestamp_column: str = "event_timestamp",
    ) -> Any:
        """Get point-in-time correct historical features."""
        pass

    @abstractmethod
    def get_online_features(
        self,
        entity_keys: dict[str, list[Any]],
        feature_refs: list[str],
    ) -> dict[str, Any]:
        """Get features for online serving."""
        pass

    @abstractmethod
    def list_tables(self) -> list[str]:
        """List all feature tables."""
        pass


class FeastBackend(FeatureStoreBackend):

    def __init__(self, repo_path: str = "./feature_store/feature_repo", data_root: Optional[str] = None):
        self.repo_path = Path(repo_path)
        self.repo_path.mkdir(parents=True, exist_ok=True)
        self.data_root = Path(data_root) if data_root else self.repo_path / "data"
        self._store = None
        self._tables: dict[str, dict] = {}
        self._load_table_metadata()
        self.storage = _get_storage()

    @property
    def store(self):
        """Lazy-load Feast store."""
        if self._store is None:
            try:
                from feast import FeatureStore
                self._store = FeatureStore(repo_path=str(self.repo_path))
            except ImportError:
                raise ImportError("Feast is required. Install with: pip install feast")
        return self._store

    def create_feature_table(
        self,
        name: str,
        entity_key: str,
        timestamp_column: str,
        schema: dict[str, str],
        cutoff_date: Optional[datetime] = None,
    ) -> str:
        self._tables[name] = {
            "entity_key": entity_key,
            "timestamp_column": timestamp_column,
            "schema": schema,
            "cutoff_date": cutoff_date.isoformat() if cutoff_date else None,
            "created_at": datetime.now().isoformat(),
        }
        self._save_table_metadata()
        return name

    def _load_table_metadata(self) -> None:
        metadata_path = self.repo_path / "feature_tables_metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r") as f:
                self._tables = json.load(f)

    def _save_table_metadata(self) -> None:
        metadata_path = self.repo_path / "feature_tables_metadata.json"
        with metadata_path.open("w") as f:
            json.dump(self._tables, f, indent=2)

    def _compute_feature_hash(self, df: Any, cutoff_date: Optional[datetime] = None) -> str:
        df_stable = df.reset_index(drop=True).copy()
        for col in df_stable.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns:
            df_stable[col] = df_stable[col].astype(str)
        df_stable = df_stable[sorted(df_stable.columns)]
        data_bytes = native_pd.util.hash_pandas_object(df_stable).to_numpy().tobytes()
        if cutoff_date:
            data_bytes += cutoff_date.isoformat().encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()[:16]

    def get_table_cutoff_date(self, name: str) -> Optional[datetime]:
        if name not in self._tables:
            return None
        cutoff_str = self._tables[name].get("cutoff_date")
        return datetime.fromisoformat(cutoff_str) if cutoff_str else None

    def validate_cutoff_consistency(self, proposed_cutoff: datetime) -> tuple[bool, str]:
        existing_cutoffs = {
            name: self.get_table_cutoff_date(name)
            for name in self._tables
            if self.get_table_cutoff_date(name) is not None
        }
        if not existing_cutoffs:
            return True, "First feature table - cutoff date will be set as reference"

        reference_date = next(iter(existing_cutoffs.values())).date()
        if proposed_cutoff.date() != reference_date:
            return False, (
                f"Cutoff mismatch. Existing tables use {reference_date}. "
                f"Proposed: {proposed_cutoff.date()}. All feature tables must use same cutoff."
            )
        return True, f"Cutoff date matches reference: {reference_date}"

    def write_features(
        self,
        table_name: str,
        df: Any,
        mode: str = "merge",
        cutoff_date: Optional[datetime] = None,
    ) -> None:
        delta_path = self.data_root / table_name
        parquet_path = self.data_root / f"{table_name}.parquet"
        delta_path.parent.mkdir(parents=True, exist_ok=True)

        if self.storage:
            if mode == "merge" and self.storage.exists(str(delta_path)):
                entity_key = self._tables.get(table_name, {}).get("entity_key", "entity_id")
                condition = f"source.{entity_key} = target.{entity_key}"
                self.storage.merge(df, str(delta_path), condition=condition)
                df = self.storage.read(str(delta_path))
            else:
                self.storage.write(df, str(delta_path))
        else:
            if mode == "merge" and parquet_path.exists():
                existing = native_pd.read_parquet(str(parquet_path))
                if table_name in self._tables:
                    entity_key = self._tables[table_name]["entity_key"]
                    df = safe_drop_duplicates(native_pd.concat([existing, df]), subset=[entity_key], keep="last")
            df.to_parquet(str(parquet_path), index=False)

        effective_cutoff = cutoff_date or (
            datetime.fromisoformat(self._tables[table_name]["cutoff_date"])
            if table_name in self._tables and self._tables[table_name].get("cutoff_date")
            else None
        )

        if table_name in self._tables:
            self._tables[table_name]["data_hash"] = self._compute_feature_hash(df, effective_cutoff)
            self._tables[table_name]["row_count"] = len(df)
            self._tables[table_name]["updated_at"] = datetime.now().isoformat()
            self._save_table_metadata()

    def get_historical_features(
        self,
        entity_df: Any,
        feature_refs: list[str],
        timestamp_column: str = "event_timestamp",
    ) -> Any:
        """Get point-in-time correct historical features using Feast."""
        try:
            return self.store.get_historical_features(
                entity_df=entity_df,
                features=feature_refs,
            ).to_df()
        except Exception:
            # Fallback: manual PIT join from parquet files
            return self._manual_pit_join(entity_df, feature_refs, timestamp_column)

    def _manual_pit_join(
        self,
        entity_df: Any,
        feature_refs: list[str],
        timestamp_column: str,
    ) -> Any:
        """Manual point-in-time join when Feast is not configured."""
        result = entity_df.copy()

        for ref in feature_refs:
            parts = ref.split(":")
            if len(parts) != 2:
                continue

            table_name, feature_name = parts
            feature_df = self._read_table_data(table_name)
            if feature_df is None:
                continue
            if feature_name not in feature_df.columns:
                continue

            # Get entity key from table metadata
            entity_key = self._tables.get(table_name, {}).get("entity_key", "entity_id")
            ts_col = self._tables.get(table_name, {}).get("timestamp_column", "feature_timestamp")

            if ts_col in feature_df.columns and timestamp_column in entity_df.columns:
                # Point-in-time join
                merged = result.merge(
                    feature_df[[entity_key, ts_col, feature_name]],
                    on=entity_key,
                    how="left",
                )
                # Keep only features from before the entity timestamp
                valid = merged[merged[ts_col] <= merged[timestamp_column]]
                # Take latest valid feature per entity
                valid = valid.sort_values(ts_col).groupby(entity_key).last().reset_index()
                result = result.merge(
                    valid[[entity_key, feature_name]],
                    on=entity_key,
                    how="left",
                )
            else:
                # Simple join without PIT
                result = result.merge(
                    feature_df[[entity_key, feature_name]],
                    on=entity_key,
                    how="left",
                )

        return result

    def get_online_features(
        self,
        entity_keys: dict[str, list[Any]],
        feature_refs: list[str],
    ) -> dict[str, Any]:
        """Get features for online serving."""
        try:
            entity_rows = [
                {k: v[i] for k, v in entity_keys.items()}
                for i in range(len(next(iter(entity_keys.values()))))
            ]
            return self.store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows,
            ).to_dict()
        except Exception:
            # Fallback: read latest from parquet
            entity_df = native_pd.DataFrame(entity_keys)
            result = self.get_historical_features(
                entity_df, feature_refs, "event_timestamp"
            )
            return result.to_dict("list")

    def list_tables(self) -> list[str]:
        """List all feature tables."""
        if not self.data_root.exists():
            return []
        tables = [p.stem for p in self.data_root.glob("*.parquet")]
        if self.storage:
            for subdir in self.data_root.iterdir():
                if subdir.is_dir() and self.storage.exists(str(subdir)) and subdir.name not in tables:
                    tables.append(subdir.name)
        return tables

    def _read_table_data(self, table_name: str) -> Optional[Any]:
        delta_path = self.data_root / table_name
        parquet_path = self.data_root / f"{table_name}.parquet"
        if self.storage and self.storage.exists(str(delta_path)):
            return self.storage.read(str(delta_path))
        if parquet_path.exists():
            return native_pd.read_parquet(str(parquet_path))
        return None


def _to_spark_df(spark: "Any", df: Any) -> "Any":
    from customer_retention.core.compat import normalize_timestamps, pandas_dtype_to_spark_schema
    normalized = normalize_timestamps(df)
    schema = pandas_dtype_to_spark_schema(normalized)
    return spark.createDataFrame(normalized, schema=schema)


class DatabricksBackend(FeatureStoreBackend):
    """Databricks Feature Engineering backend for production."""

    def __init__(self, catalog: str = "main", schema: str = "features"):
        self.catalog = catalog
        self.schema = schema
        self._client = None

    @property
    def client(self):
        """Lazy-load Databricks Feature Engineering client."""
        if self._client is None:
            try:
                from databricks.feature_engineering import FeatureEngineeringClient
                self._client = FeatureEngineeringClient()
            except ImportError:
                raise ImportError(
                    "Databricks Feature Engineering is required. "
                    "Run on a Databricks cluster."
                )
        return self._client

    def _full_table_name(self, name: str) -> str:
        """Get fully qualified table name."""
        return f"{self.catalog}.{self.schema}.{name}"

    def create_feature_table(
        self,
        name: str,
        entity_key: str,
        timestamp_column: str,
        schema: dict[str, str],
        cutoff_date: Optional[datetime] = None,
    ) -> str:
        from pyspark.sql import SparkSession
        from pyspark.sql.types import FloatType, IntegerType, StringType, StructField, StructType, TimestampType

        spark = SparkSession.builder.getOrCreate()

        type_mapping = {
            "string": StringType(),
            "float64": FloatType(),
            "float": FloatType(),
            "int64": IntegerType(),
            "int": IntegerType(),
            "datetime": TimestampType(),
        }

        fields = [StructField(col_name, type_mapping.get(dtype, StringType()), True) for col_name, dtype in schema.items()]
        spark_schema = StructType(fields)

        empty_df = spark.createDataFrame([], spark_schema)
        full_name = self._full_table_name(name)

        self.client.create_table(
            name=full_name,
            primary_keys=[entity_key],
            timestamp_keys=[timestamp_column] if timestamp_column else None,
            df=empty_df,
            description=f"Point-in-time cutoff: {cutoff_date.isoformat() if cutoff_date else 'N/A'}",
        )

        return full_name

    def write_features(
        self,
        table_name: str,
        df: Any,
        mode: str = "merge",
        cutoff_date: Optional[datetime] = None,
    ) -> None:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        spark_df = _to_spark_df(spark, df)

        full_name = self._full_table_name(table_name)
        self.client.write_table(name=full_name, df=spark_df, mode=mode)

    def get_historical_features(
        self,
        entity_df: Any,
        feature_refs: list[str],
        timestamp_column: str = "event_timestamp",
    ) -> Any:
        """Get point-in-time correct historical features."""
        from databricks.feature_engineering import FeatureLookup
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        entity_spark = _to_spark_df(spark, entity_df)

        lookups = []
        for ref in feature_refs:
            parts = ref.split(":")
            if len(parts) == 2:
                table_name, feature_name = parts
                full_name = self._full_table_name(table_name)
                lookups.append(
                    FeatureLookup(
                        table_name=full_name,
                        feature_names=[feature_name],
                        lookup_key=list(entity_df.columns[:1]),
                        timestamp_lookup_key=timestamp_column,
                    )
                )

        training_set = self.client.create_training_set(
            df=entity_spark,
            feature_lookups=lookups,
            label=None,
        )

        return training_set.load_df().toPandas()

    def get_online_features(
        self,
        entity_keys: dict[str, list[Any]],
        feature_refs: list[str],
    ) -> dict[str, Any]:
        """Get features for online serving via Model Serving."""
        from databricks.feature_engineering import FeatureLookup
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        entity_df = native_pd.DataFrame(entity_keys)
        entity_spark = _to_spark_df(spark, entity_df)

        lookups = []
        for ref in feature_refs:
            parts = ref.split(":")
            if len(parts) == 2:
                table_name, _ = parts
                full_name = self._full_table_name(table_name)
                lookups.append(
                    FeatureLookup(
                        table_name=full_name,
                        lookup_key=list(entity_keys.keys()),
                    )
                )

        result = self.client.score_batch(df=entity_spark, feature_lookups=lookups)
        return result.toPandas().to_dict("list")

    def list_tables(self) -> list[str]:
        """List all feature tables in the schema."""
        tables = self.client.list_tables()
        prefix = f"{self.catalog}.{self.schema}."
        return [
            t.name.replace(prefix, "")
            for t in tables
            if t.name.startswith(prefix)
        ]


class FeatureStoreManager:
    """Unified manager for feature store operations.

    This class provides a high-level interface for feature store operations
    that works seamlessly with both local (Feast) and production (Databricks)
    backends, while ensuring point-in-time correctness.

    Example:
        >>> manager = FeatureStoreManager.create(backend="feast")
        >>> manager.publish_features(df, registry, "customer_features")
        >>> training_df = manager.get_training_features(
        ...     entity_df, registry, ["tenure_months", "total_spend_30d"]
        ... )
    """

    def __init__(self, backend: FeatureStoreBackend, output_path: Optional[Path] = None):
        self.backend = backend
        self.output_path = Path(output_path) if output_path else Path("./output")
        self.snapshot_manager = SnapshotManager(self.output_path)
        self.pit_registry = PointInTimeRegistry(self.output_path)

    @classmethod
    def create(
        cls,
        backend: str = "feast",
        repo_path: str = "./feature_store/feature_repo",
        catalog: str = "main",
        schema: str = "features",
        output_path: Optional[str] = None,
        data_root: Optional[str] = None,
    ) -> "FeatureStoreManager":
        """Factory method to create a manager with the appropriate backend.

        Args:
            backend: Backend type ("feast" or "databricks")
            repo_path: Path to Feast repo (for feast backend)
            catalog: Unity Catalog name (for databricks backend)
            schema: Schema name (for databricks backend)
            output_path: Path for output files
            data_root: Override data directory for feast backend (e.g. gold path)

        Returns:
            Configured FeatureStoreManager

        Raises:
            ValueError: If unknown backend specified
        """
        if backend == "feast":
            store_backend = FeastBackend(repo_path=repo_path, data_root=data_root)
        elif backend == "databricks":
            store_backend = DatabricksBackend(catalog=catalog, schema=schema)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'feast' or 'databricks'.")

        return cls(
            backend=store_backend,
            output_path=Path(output_path) if output_path else None,
        )

    def publish_features(
        self,
        df: Any,
        registry: FeatureRegistry,
        table_name: str,
        entity_key: str = "entity_id",
        timestamp_column: str = "feature_timestamp",
        mode: str = "merge",
        cutoff_date: Optional[datetime] = None,
    ) -> str:
        effective_cutoff = cutoff_date or self.pit_registry.get_reference_cutoff() or datetime.now()

        is_valid, message = self.pit_registry.validate_cutoff(effective_cutoff)
        if not is_valid:
            raise ValueError(f"Point-in-time consistency error: {message}")

        if isinstance(self.backend, FeastBackend):
            backend_valid, backend_msg = self.backend.validate_cutoff_consistency(effective_cutoff)
            if not backend_valid:
                raise ValueError(f"Feature store cutoff mismatch: {backend_msg}")

        missing_features = [f for f in registry.list_features() if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features in DataFrame: {missing_features}")

        schema = {entity_key: "string", timestamp_column: "datetime"}
        for feature_name in registry.list_features():
            if feature_name in df.columns:
                feature = registry.get(feature_name)
                schema[feature_name] = feature.data_type if feature else "float64"

        self.backend.create_feature_table(
            name=table_name,
            entity_key=entity_key,
            timestamp_column=timestamp_column,
            schema=schema,
            cutoff_date=effective_cutoff,
        )

        columns_to_write = [entity_key, timestamp_column] + [f for f in registry.list_features() if f in df.columns]
        self.backend.write_features(table_name, df[columns_to_write], mode=mode, cutoff_date=effective_cutoff)

        return table_name

    def get_training_features(
        self,
        entity_df: Any,
        registry: FeatureRegistry,
        feature_names: Optional[list[str]] = None,
        table_name: str = "customer_features",
        timestamp_column: str = "event_timestamp",
    ) -> Any:
        """Get point-in-time correct features for training.

        Args:
            entity_df: DataFrame with entity keys and timestamps
            registry: Feature registry
            feature_names: Specific features to retrieve (all if None)
            table_name: Feature table name
            timestamp_column: Timestamp column in entity_df

        Returns:
            DataFrame with entity keys, timestamps, and features
        """
        feature_refs = registry.get_feature_refs(
            table_name,
            feature_names or registry.list_features(),
        )

        return self.backend.get_historical_features(
            entity_df=entity_df,
            feature_refs=feature_refs,
            timestamp_column=timestamp_column,
        )

    def get_inference_features(
        self,
        entity_df: Any,
        registry: FeatureRegistry,
        feature_names: Optional[list[str]] = None,
        table_name: str = "customer_features",
        timestamp_column: str = "event_timestamp",
    ) -> Any:
        """Get point-in-time correct features for batch inference.

        This is the recommended method for batch inference as it ensures
        features are retrieved as they existed at the specified inference
        timestamp, preventing future data leakage.

        Args:
            entity_df: DataFrame with entity keys and inference timestamps
                      Must have entity_id column and a timestamp column
            registry: Feature registry
            feature_names: Specific features to retrieve (all if None)
            table_name: Feature table name
            timestamp_column: Name of the timestamp column in entity_df

        Returns:
            DataFrame with entity keys, timestamps, and features

        Example:
            >>> # Create entity DataFrame with inference timestamp
            >>> entity_df = pd.DataFrame({
            ...     "entity_id": ["cust_1", "cust_2"],
            ...     "event_timestamp": [datetime.now(), datetime.now()]
            ... })
            >>> # Get features as of the inference timestamp
            >>> features_df = manager.get_inference_features(
            ...     entity_df, registry, timestamp_column="event_timestamp"
            ... )
        """
        feature_refs = registry.get_feature_refs(
            table_name,
            feature_names or registry.list_features(),
        )

        return self.backend.get_historical_features(
            entity_df=entity_df,
            feature_refs=feature_refs,
            timestamp_column=timestamp_column,
        )

    def get_online_features(
        self,
        entity_keys: dict[str, list[Any]],
        registry: FeatureRegistry,
        feature_names: Optional[list[str]] = None,
        table_name: str = "customer_features",
    ) -> dict[str, Any]:
        """Get latest features for online/real-time inference.

        This returns the latest feature values without point-in-time
        correctness. Use for real-time serving where you want the
        most recent features.

        For batch inference with PIT correctness, use get_inference_features().

        Args:
            entity_keys: Dictionary of entity key column to values
            registry: Feature registry
            feature_names: Specific features to retrieve (all if None)
            table_name: Feature table name

        Returns:
            Dictionary of feature values
        """
        feature_refs = registry.get_feature_refs(
            table_name,
            feature_names or registry.list_features(),
        )

        return self.backend.get_online_features(
            entity_keys=entity_keys,
            feature_refs=feature_refs,
        )

    def create_training_set_from_snapshot(
        self,
        snapshot_id: str,
        registry: FeatureRegistry,
        target_column: str = "target",
    ) -> tuple[Any, Any]:
        """Create a training set from a snapshot.

        This loads a versioned snapshot and prepares it for training,
        ensuring only the registered features are used.

        Args:
            snapshot_id: ID of the snapshot to load
            registry: Feature registry
            target_column: Name of the target column

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df, metadata = self.snapshot_manager.load_snapshot(snapshot_id)

        # Get feature columns that exist in both registry and snapshot
        feature_columns = [
            f for f in registry.list_features()
            if f in df.columns
        ]

        X = df[feature_columns]
        y = df[target_column] if target_column in df.columns else None

        return X, y

    def list_tables(self) -> list[str]:
        """List all feature tables.

        Returns:
            List of table names
        """
        return self.backend.list_tables()


def get_feature_store_manager(
    backend: Optional[str] = None,
    **kwargs,
) -> FeatureStoreManager:
    """Get a feature store manager, auto-detecting environment.

    Args:
        backend: Explicit backend ("feast" or "databricks"), or None for auto-detect
        **kwargs: Additional arguments for the manager

    Returns:
        Configured FeatureStoreManager
    """
    if backend is None:
        # Auto-detect environment
        try:
            from customer_retention.core.compat.detection import is_databricks
            if is_databricks():
                backend = "databricks"
            else:
                backend = "feast"
        except ImportError:
            backend = "feast"

    return FeatureStoreManager.create(backend=backend, **kwargs)


def _get_storage():
    try:
        from customer_retention.integrations.adapters.factory import get_delta
        return get_delta()
    except ImportError:
        return None
