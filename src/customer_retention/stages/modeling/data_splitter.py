import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sklearn.model_selection import GroupShuffleSplit, train_test_split

from customer_retention.core.compat import (
    DataFrame,
    Series,
    _is_spark_pandas,
    as_spark_df,
    native_pd,
    temporal_quantile,
    to_pandas,
)
from customer_retention.core.config.column_config import select_model_ready_columns
from customer_retention.parity import ApplyOpKind, apply_op

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.findings import FeatureAvailabilityMetadata


class SplitStrategy(Enum):
    RANDOM_STRATIFIED = "random_stratified"
    TEMPORAL = "temporal"
    GROUP = "group"
    CUSTOM = "custom"


@dataclass
class SplitConfig:
    test_size: float = 0.11
    validation_size: float = 0.10
    stratify: bool = True
    random_state: int = 42
    temporal_column: Optional[str] = None
    group_column: Optional[str] = None


@dataclass
class SplitResult:
    X_train: DataFrame
    X_test: DataFrame
    y_train: Series
    y_test: Series
    X_val: Optional[DataFrame] = None
    y_val: Optional[Series] = None
    split_info: Dict[str, Any] = field(default_factory=dict)
    train_metadata: Dict[str, Series] = field(default_factory=dict)


@dataclass
class SplitWarning:
    column: str
    issue: str
    severity: str
    recommendation: str

    def to_dict(self) -> Dict[str, str]:
        return {"column": self.column, "issue": self.issue, "severity": self.severity, "recommendation": self.recommendation}


class DataSplitter:
    def __init__(self, target_column: str, strategy: SplitStrategy = SplitStrategy.RANDOM_STRATIFIED, test_size: float = 0.11, validation_size: float = 0.10, stratify: bool = True, random_state: int = 42, temporal_column: Optional[str] = None, group_column: Optional[str] = None, exclude_columns: Optional[List[str]] = None, include_validation: bool = False, purge_gap_days: Optional[int] = None):
        self.target_column = target_column
        self.strategy = strategy
        self.test_size = test_size
        self.validation_size = validation_size
        self.stratify = stratify
        self.random_state = random_state
        self.temporal_column = temporal_column
        self.group_column = group_column
        self.exclude_columns = exclude_columns or []
        self.include_validation = include_validation
        self.purge_gap_days = purge_gap_days

    @apply_op(kind=ApplyOpKind.TRAINING_SPLIT)
    def split(self, df: DataFrame, feature_availability: Optional["FeatureAvailabilityMetadata"] = None) -> SplitResult:
        if _is_spark_pandas(df):
            return self._split_spark(as_spark_df(df), feature_availability)

        self._validate_minority_samples(df)
        availability_warnings = self.validate_feature_availability(df, feature_availability)

        if self.strategy == SplitStrategy.TEMPORAL:
            if not self.group_column:
                raise ValueError(
                    "Temporal split requires group_column (entity column) to prevent "
                    "entity leakage. Use group_column='entity_id' or switch to "
                    "SplitStrategy.RANDOM_STRATIFIED."
                )
            result = self._temporal_split(df)
        elif self.strategy == SplitStrategy.GROUP:
            result = self._group_split(to_pandas(df))
        else:
            result = self._stratified_split(to_pandas(df))

        if availability_warnings:
            result.split_info["availability_warnings"] = [w.to_dict() for w in availability_warnings]
        return result

    def _split_spark(self, spark_df, feature_availability: Optional["FeatureAvailabilityMetadata"] = None) -> SplitResult:
        from pyspark.sql import functions as F  # noqa: N812

        min_count = spark_df.groupBy(self.target_column).count().agg(F.min("count")).head()[0] or 0
        if min_count * self.test_size < 50:
            warnings.warn(
                f"Insufficient minority samples: expected ~{min_count * self.test_size:.0f} in test set. "
                "Consider using a smaller test_size or collecting more data.", UserWarning,
            )

        availability_warnings = self.validate_feature_availability(spark_df, feature_availability)

        if self.strategy == SplitStrategy.TEMPORAL:
            if not self.group_column:
                raise ValueError(
                    "Temporal split requires group_column (entity column) to prevent "
                    "entity leakage. Use group_column='entity_id' or switch to "
                    "SplitStrategy.RANDOM_STRATIFIED."
                )
            result = self._distributed_temporal_split(spark_df)
        elif self.strategy == SplitStrategy.GROUP:
            result = self._group_split(spark_df.toPandas())
        else:
            result = self._stratified_split(spark_df.toPandas())

        if availability_warnings:
            result.split_info["availability_warnings"] = [w.to_dict() for w in availability_warnings]
        return result

    def validate_feature_availability(self, df: DataFrame, availability: Optional["FeatureAvailabilityMetadata"]) -> List[SplitWarning]:
        if availability is None:
            return []
        if self.strategy != SplitStrategy.TEMPORAL:
            return []
        warnings_list: List[SplitWarning] = []
        for col in availability.new_tracking:
            if col in df.columns:
                feat_info = availability.features.get(col)
                first_date = feat_info.first_valid_date if feat_info else "unknown"
                warnings_list.append(SplitWarning(
                    column=col, issue="new_tracking", severity="warning",
                    recommendation=f"Feature '{col}' only available from {first_date}. Training data before this date will have missing values.",
                ))
        for col in availability.retired_tracking:
            if col in df.columns:
                feat_info = availability.features.get(col)
                last_date = feat_info.last_valid_date if feat_info else "unknown"
                warnings_list.append(SplitWarning(
                    column=col, issue="retired", severity="warning",
                    recommendation=f"Feature '{col}' retired at {last_date}. Test data after this date will have missing values.",
                ))
        for col in availability.partial_window:
            if col in df.columns:
                feat_info = availability.features.get(col)
                first_date = feat_info.first_valid_date if feat_info else "unknown"
                last_date = feat_info.last_valid_date if feat_info else "unknown"
                warnings_list.append(SplitWarning(
                    column=col, issue="partial_window", severity="warning",
                    recommendation=f"Feature '{col}' only available {first_date} to {last_date}. Both train and test may have gaps.",
                ))
        return warnings_list

    @staticmethod
    def _safe_stratify_col(y: Series) -> Optional[Series]:
        counts = y.value_counts()
        if (counts < 2).any():
            return None
        return y

    def _stratified_split(self, df: DataFrame) -> SplitResult:
        X, y = self._prepare_features_target(df)
        stratify_col = self._safe_stratify_col(y) if self.stratify else None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify_col)

        X_val, y_val = None, None
        if self.include_validation:
            val_ratio = self.validation_size / (1 - self.test_size)
            stratify_train = self._safe_stratify_col(y_train) if self.stratify else None
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=self.random_state, stratify=stratify_train)

        return SplitResult(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            X_val=X_val, y_val=y_val,
            split_info=self._build_split_info(X_train, X_test, X_val)
        )

    def _temporal_split(self, df: DataFrame) -> SplitResult:
        if _is_spark_pandas(df):
            return self._distributed_temporal_split(df)

        col = self.temporal_column
        group_col = self.group_column
        cutoff_date = temporal_quantile(df[col], 1 - self.test_size)
        if native_pd.isna(cutoff_date):
            raise ValueError(
                f"Temporal column '{col}' has no valid dates — cannot compute split cutoff. "
                f"Check that the column is not all-null and that rows remain after target filtering."
            )

        # Entity-grouped temporal split: hold out entire entities so no entity
        # appears in both train and test.  Entity selection uses GroupShuffleSplit
        # which is robust to panel data (all entities spanning the full range).
        # Temporal purge then removes late rows from train entities.
        groups = df[group_col]
        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_idx, test_idx = next(gss.split(df, groups=groups))

        test_df = df.iloc[test_idx]
        train_df = df.iloc[train_idx]

        purge_gap_rows = 0
        if self.purge_gap_days and self.purge_gap_days > 0:
            purge_start = cutoff_date - timedelta(days=self.purge_gap_days)
            pre_purge = len(train_df)
            train_df = train_df[train_df[col] < purge_start]
            purge_gap_rows = pre_purge - len(train_df)

        X_val, y_val = None, None
        if self.include_validation:
            val_frac = self.validation_size / (1 - self.test_size)
            train_groups = train_df[group_col]
            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=self.random_state)
            t_idx, v_idx = next(gss_val.split(train_df, groups=train_groups))
            X_val, y_val = self._prepare_features_target(train_df.iloc[v_idx])
            train_df = train_df.iloc[t_idx]

        X_train, y_train = self._prepare_features_target(train_df)
        X_test, y_test = self._prepare_features_target(test_df)

        train_entities = groups.iloc[train_idx].nunique()
        test_entities = groups.iloc[test_idx].nunique()
        split_info = self._build_split_info(X_train, X_test, X_val)
        split_info["cutoff_date"] = str(cutoff_date)
        split_info["train_entities"] = int(train_entities)
        split_info["test_entities"] = int(test_entities)
        if self.purge_gap_days and self.purge_gap_days > 0:
            split_info["purge_gap_days"] = self.purge_gap_days
            split_info["purge_gap_rows"] = purge_gap_rows

        return SplitResult(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            X_val=X_val, y_val=y_val,
            split_info=split_info,
        )

    def _distributed_temporal_split(self, df: DataFrame) -> SplitResult:
        from pyspark.sql import functions as F  # noqa: N812

        from customer_retention.core.compat.spark_backend import _as_pandas_api

        col = self.temporal_column
        group_col = self.group_column
        spark_df = as_spark_df(df)
        idx_cols = [c for c in spark_df.columns if c.startswith("__index_level_")]
        if idx_cols:
            spark_df = spark_df.drop(*idx_cols)

        # Temporal cutoff (one Spark job) — used for purge, not entity selection
        epoch = F.unix_timestamp(F.col(col).cast("timestamp"))
        agg_row = spark_df.agg(
            F.percentile_approx(epoch, float(1 - self.test_size)).alias("cutoff"),
            F.count("*").alias("total"),
        ).head()
        cutoff_date = native_pd.Timestamp(agg_row["cutoff"], unit="s")
        total_rows = int(agg_row["total"])
        if native_pd.isna(cutoff_date):
            raise ValueError(
                f"Temporal column '{col}' has no valid dates — cannot compute split cutoff. "
                f"Check that the column is not all-null and that rows remain after target filtering."
            )

        # Entity-grouped split — fully distributed via deterministic hash.
        # hash(entity_id + seed) assigns each entity to a bucket 0..99;
        # entities in buckets < test_pct go to test, rest to train.
        # No .collect() — works with 400K+ entities without driver OOM.
        test_pct = int(self.test_size * 100)
        seed = F.lit(self.random_state)
        bucket = F.abs(F.hash(F.col(group_col), seed)) % F.lit(100)
        is_test_entity = bucket < F.lit(test_pct)

        test_spark = spark_df.filter(is_test_entity)
        train_spark = spark_df.filter(~is_test_entity)

        # Temporal purge on training side
        if self.purge_gap_days and self.purge_gap_days > 0:
            purge_start = cutoff_date - timedelta(days=self.purge_gap_days)
            train_spark = train_spark.filter(F.col(col) < F.lit(purge_start))

        X_val, y_val = None, None
        if self.include_validation:
            # Sub-split train entities: hash to a second bucket for validation
            val_pct = int(self.validation_size * 100 / (1 - self.test_size))
            val_bucket = F.abs(F.hash(F.col(group_col), F.lit(self.random_state + 1))) % F.lit(100)
            val_spark = train_spark.filter(val_bucket < F.lit(val_pct))
            train_spark = train_spark.filter(val_bucket >= F.lit(val_pct))
            val_spark = val_spark.localCheckpoint(eager=True)
            val_ps = _as_pandas_api(val_spark)
            X_val, y_val = self._select_features_target_from_ps(val_ps, val_spark.schema)

        train_spark = train_spark.localCheckpoint(eager=True)
        test_spark = test_spark.localCheckpoint(eager=True)

        train_ps = _as_pandas_api(train_spark)
        test_ps = _as_pandas_api(test_spark)

        X_train, y_train = self._select_features_target_from_ps(train_ps, train_spark.schema)
        X_test, y_test = self._select_features_target_from_ps(test_ps, test_spark.schema)

        train_meta: Dict[str, Series] = {}
        all_fields = {f.name for f in train_spark.schema.fields}
        meta_cols = [c for c in self.exclude_columns if c in all_fields]
        if meta_cols:
            train_meta = {c: train_ps[c] for c in meta_cols}

        split_info = self._build_split_info(X_train, X_test, X_val)
        split_info["cutoff_date"] = str(cutoff_date)
        if self.purge_gap_days and self.purge_gap_days > 0:
            split_info["purge_gap_days"] = self.purge_gap_days
            kept = split_info["train_size"] + split_info["test_size"]
            if X_val is not None:
                kept += split_info.get("validation_size", 0)
            split_info["purge_gap_rows"] = total_rows - kept

        return SplitResult(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            X_val=X_val, y_val=y_val,
            split_info=split_info,
            train_metadata=train_meta,
        )

    def _select_features_target_from_ps(self, ps_df: DataFrame, schema: Any) -> tuple[DataFrame, Series]:
        """Select feature columns and target from a pyspark.pandas DataFrame.

        Uses the Spark schema to identify and exclude non-numeric types.
        No checkpoint or re-read — operates on already-materialized data.
        """
        from pyspark.sql.types import DateType, DayTimeIntervalType, TimestampNTZType, TimestampType

        exclude = {self.target_column} | set(self.exclude_columns)
        skip_types = (TimestampType, TimestampNTZType, DateType, DayTimeIntervalType)
        feature_cols = [
            f.name for f in schema.fields
            if f.name not in exclude and not isinstance(f.dataType, skip_types)
        ]
        return ps_df[feature_cols], ps_df[self.target_column]

    def _spark_select_features_target(self, spark_df: Any, extract_metadata: bool = False) -> tuple[DataFrame, Series, Dict[str, Series]]:
        from pyspark.sql.types import DateType, DayTimeIntervalType, TimestampNTZType, TimestampType

        from customer_retention.core.compat.spark_backend import _as_pandas_api

        exclude = {self.target_column} | set(self.exclude_columns)
        skip_types = (TimestampType, TimestampNTZType, DateType, DayTimeIntervalType)
        all_fields = {f.name for f in spark_df.schema.fields}
        feature_cols = [
            f.name for f in spark_df.schema.fields
            if f.name not in exclude and not isinstance(f.dataType, skip_types)
        ]

        features = _as_pandas_api(spark_df.select(feature_cols).localCheckpoint(eager=True))
        target = _as_pandas_api(spark_df.select(self.target_column))[self.target_column]
        metadata: Dict[str, Series] = {}
        if extract_metadata:
            meta_cols = [c for c in self.exclude_columns if c in all_fields]
            if meta_cols:
                meta_ps = _as_pandas_api(spark_df.select(meta_cols))
                metadata = {c: meta_ps[c] for c in meta_cols}
        return features, target, metadata

    def _group_split(self, df: DataFrame) -> SplitResult:
        X, y = self._prepare_features_target(df)
        groups = df[self.group_column]

        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_val, y_val = None, None
        if self.include_validation:
            val_ratio = self.validation_size / (1 - self.test_size)
            train_groups = groups.iloc[train_idx]
            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=self.random_state)
            train_idx2, val_idx2 = next(gss_val.split(X_train, y_train, train_groups))
            X_val, y_val = X_train.iloc[val_idx2], y_train.iloc[val_idx2]
            X_train, y_train = X_train.iloc[train_idx2], y_train.iloc[train_idx2]

        return SplitResult(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            X_val=X_val, y_val=y_val,
            split_info=self._build_split_info(X_train, X_test, X_val)
        )

    def _prepare_features_target(self, df: DataFrame) -> tuple[DataFrame, Series]:
        exclude = [self.target_column] + self.exclude_columns
        feature_cols = [c for c in df.columns if c not in exclude]
        return select_model_ready_columns(df[feature_cols]), df[self.target_column]

    def _validate_minority_samples(self, df: DataFrame):
        class_counts = df[self.target_column].value_counts()
        minority_count = class_counts.min()
        expected_minority_test = minority_count * self.test_size

        if expected_minority_test < 50:
            warnings.warn(
                f"Insufficient minority samples: expected ~{expected_minority_test:.0f} in test set. "
                "Consider using a smaller test_size or collecting more data.",
                UserWarning
            )

    def _build_split_info(self, X_train, X_test, X_val) -> Dict[str, Any]:
        info = {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "strategy": self.strategy.value,
            "random_state": self.random_state,
        }
        if X_val is not None:
            info["validation_size"] = len(X_val)
        return info
