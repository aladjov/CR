from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from customer_retention.core.compat import (
    DataFrame,
    Series,
    _is_spark_pandas,
    as_spark_df,
    bulk_label_encode,
    bulk_median_impute,
    collect_for_sklearn,
    concat,
    safe_sample,
    spark_cast_float32,
    spark_checkpoint,
)
from customer_retention.core.compat.timing import TimingEntry, log_timing, start_collecting, stop_collecting

from .cross_validator import _CV_DATE_COL, _CV_ENTITY_COL
from .data_splitter import DataSplitter, SplitStrategy
from .feature_scaler import FeatureScaler, ScalerType

ProgressCallback = Optional["Callable[[int, int, str, float], None]"]


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def print_preparation_progress(step: int, total: int, label: str, elapsed: float) -> None:
    print(f"  [{step}/{total}] {label}: {elapsed:.1f}s")


class PreparationProgressTracker:
    """Progress callback with wall-time tracking and ETA."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    def __call__(self, step: int, total: int, label: str, elapsed: float) -> None:
        wall = time.monotonic() - self._start
        remaining = (wall / step) * (total - step) if step > 0 else 0
        eta = f", ETA ~{_fmt_duration(remaining)}" if step < total else ""
        print(f"  [{step}/{total}] {label}: {elapsed:.1f}s (wall: {_fmt_duration(wall)}{eta})")


@dataclass
class TrainingPreparationResult:
    X_train: Any
    X_test: Any
    X_train_scaled: Any
    X_test_scaled: Any
    y_train: Any
    y_test: Any
    y_test_np: np.ndarray
    feature_names: list[str]
    train_entities: Any
    train_dates: Any
    split_info: dict[str, Any]
    class_distribution: dict[int, int]
    zero_variance_dropped: list[str]
    null_counts: dict[str, int] = field(default_factory=dict)
    timing_entries: list[TimingEntry] = field(default_factory=list)


class TrainingPreparator:
    def __init__(
        self,
        target_column: str,
        feature_columns: list[str],
        purge_gap_days: int = 104,
        test_size: float = 0.2,
        scaler_type: ScalerType = ScalerType.STANDARD,
        max_rows: Optional[int] = None,
        use_float32: bool = True,
        on_progress: ProgressCallback = None,
    ):
        self._target = target_column
        self._feature_columns = list(feature_columns)
        self._purge_gap_days = purge_gap_days
        self._test_size = test_size
        self._scaler_type = scaler_type
        self._max_rows = max_rows
        self._use_float32 = use_float32
        self._on_progress = on_progress

    _TOTAL_STEPS = 10

    def _report(self, step: int, label: str, elapsed: float) -> None:
        if self._on_progress:
            self._on_progress(step, self._TOTAL_STEPS, label, elapsed)

    def _log_sub(self, msg: str) -> None:
        if self._on_progress:
            print(f"      → {msg}")

    def prepare(self, df: DataFrame) -> TrainingPreparationResult:
        start_collecting()
        _s = 0

        _s += 1
        with log_timing("classify_columns") as _t:
            feature_cols, obj_cols = self._classify_columns(df, self._feature_columns)
        self._report(_s, _t.label, _t.elapsed)

        _s += 1
        with log_timing("drop_missing_target") as _t:
            df, _nan_count = self._drop_missing_target(df)
        self._report(_s, _t.label, _t.elapsed)

        _s += 1
        with log_timing("encode_object_columns") as _t:
            if obj_cols:
                df = bulk_label_encode(df, obj_cols)
        self._report(_s, _t.label, _t.elapsed)

        _s += 1
        with log_timing("checkpoint") as _t:
            df = spark_checkpoint(df)
        self._report(_s, _t.label, _t.elapsed)

        _s += 1
        with log_timing("median_impute") as _t:
            df = bulk_median_impute(df, columns=feature_cols)
            df = spark_checkpoint(df)
        self._report(_s, _t.label, _t.elapsed)

        _s += 1
        with log_timing("sample_entities") as _t:
            df = self._sample_entities(df)
        self._report(_s, _t.label, _t.elapsed)

        _s += 1
        with log_timing("temporal_split") as _t:
            split_result = self._temporal_split(df, feature_cols)
        self._report(_s, _t.label, _t.elapsed)

        X_train, X_test = split_result.X_train, split_result.X_test
        y_train, y_test = split_result.y_train, split_result.y_test
        train_entities, train_dates = self._extract_train_metadata(split_result, df)

        _s += 1
        with log_timing("fillna_and_drop_zero_variance") as _t:
            X_train, X_test, zero_var, null_counts_map = self._fillna_and_drop_zero_variance(X_train, X_test)
        self._report(_s, _t.label, _t.elapsed)

        feature_cols = [c for c in X_train.columns if c not in {_CV_ENTITY_COL, _CV_DATE_COL}]

        _s += 1
        distributed = _is_spark_pandas(X_train)
        self._log_sub(
            f"finalize: {'distributed' if distributed else 'local'}, "
            f"type={type(X_train).__module__}.{type(X_train).__name__}, "
            f"shape={X_train.shape}"
        )
        with log_timing("scale_features") as _t:
            if distributed:
                result = self._finalize_distributed(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    train_entities,
                    train_dates,
                    feature_cols,
                )
            else:
                result = self._finalize_local(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    train_entities,
                    train_dates,
                    feature_cols,
                )
        self._report(_s, _t.label, _t.elapsed)

        _s += 1
        with log_timing("class_distribution") as _t:
            class_dist = self._class_distribution(result.y_train)
        self._report(_s, _t.label, _t.elapsed)

        timing_entries = stop_collecting()

        return TrainingPreparationResult(
            X_train=result.X_train,
            X_test=result.X_test,
            X_train_scaled=result.X_train_scaled,
            X_test_scaled=result.X_test_scaled,
            y_train=result.y_train,
            y_test=result.y_test,
            y_test_np=result.y_test_np,
            feature_names=result.feature_names,
            train_entities=result.train_entities,
            train_dates=result.train_dates,
            split_info=split_result.split_info,
            class_distribution=class_dist,
            zero_variance_dropped=zero_var,
            null_counts=null_counts_map,
            timing_entries=timing_entries,
        )

    def _classify_columns(self, df: DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
        """Classify feature columns by type. Returns (non-datetime features, string features).

        On distributed DataFrames uses Spark schema (metadata-only, no plan analysis).
        """
        if _is_spark_pandas(df):
            return self._classify_via_schema(as_spark_df(df), feature_cols)
        dtypes = df.dtypes
        non_dt = [c for c in feature_cols if not str(dtypes.get(c, "")).startswith(("datetime", "timedelta"))]
        string_cols = [c for c in non_dt if str(dtypes.get(c, "")).startswith("object")]
        return non_dt, string_cols

    @staticmethod
    def _classify_via_schema(spark_df: Any, feature_cols: list[str]) -> tuple[list[str], list[str]]:
        from pyspark.sql.types import DateType, DayTimeIntervalType, StringType, TimestampNTZType, TimestampType

        type_map = {f.name: f.dataType for f in spark_df.schema.fields}
        skip = (TimestampType, TimestampNTZType, DateType, DayTimeIntervalType)
        non_dt = [c for c in feature_cols if c in type_map and not isinstance(type_map[c], skip)]
        string_cols = [c for c in non_dt if isinstance(type_map.get(c), StringType)]
        return non_dt, string_cols

    def _drop_missing_target(self, df: DataFrame) -> tuple[DataFrame, int]:
        if _is_spark_pandas(df):
            return self._drop_missing_target_spark(df)
        mask = df[self._target].isna()
        nan_count = int(mask.sum())
        if nan_count == len(df):
            raise ValueError(f"Cannot proceed: all target values are NaN in column '{self._target}'")
        if nan_count > 0:
            df = df[~mask]
        return df, nan_count

    def _drop_missing_target_spark(self, df: DataFrame) -> tuple[DataFrame, int]:
        """Single Spark agg job instead of separate sum() + len()."""
        from pyspark.sql import functions as F  # noqa: N812

        from customer_retention.core.compat.spark_backend import _as_pandas_api

        spark_df = as_spark_df(df)
        row = spark_df.agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col(self._target).isNull(), 1).otherwise(0)).alias("nulls"),
        ).head()
        nan_count, total = int(row["nulls"] or 0), int(row["total"])
        if nan_count == total:
            raise ValueError(f"Cannot proceed: all target values are NaN in column '{self._target}'")
        if nan_count > 0:
            spark_df = spark_df.filter(F.col(self._target).isNotNull())
            return _as_pandas_api(spark_df), nan_count
        return df, 0

    def _sample_entities(self, df: DataFrame) -> DataFrame:
        if self._max_rows is None:
            return df
        if _is_spark_pandas(df):
            return self._sample_entities_spark(df)
        if len(df) <= self._max_rows:
            return df
        n_entities = df["entity_id"].nunique()
        rows_per_entity = len(df) / max(1, n_entities)
        target_entities = max(100, int(self._max_rows / rows_per_entity))
        entity_df = df[["entity_id"]].drop_duplicates()
        sampled_entity_df = safe_sample(entity_df, n=target_entities)
        return df.merge(sampled_entity_df, on="entity_id", how="inner")

    def _sample_entities_spark(self, df: DataFrame) -> DataFrame:
        """Single agg job for row count + entity count instead of separate len() + nunique()."""
        from pyspark.sql import functions as F  # noqa: N812

        spark_df = as_spark_df(df)
        row = spark_df.agg(
            F.count("*").alias("total"),
            F.countDistinct(F.col("entity_id")).alias("n_entities"),
        ).head()
        total, n_entities = int(row["total"]), int(row["n_entities"])
        if total <= self._max_rows:
            return df
        rows_per_entity = total / max(1, n_entities)
        target_entities = max(100, int(self._max_rows / rows_per_entity))
        entity_df = df[["entity_id"]].drop_duplicates()
        sampled_entity_df = safe_sample(entity_df, n=target_entities)
        return df.merge(sampled_entity_df, on="entity_id", how="inner")

    def _temporal_split(self, df: DataFrame, feature_cols: list[str]) -> Any:
        exclude = ["as_of_date", "entity_id"]
        split_df = df[feature_cols + [self._target, "as_of_date", "entity_id"]]
        splitter = DataSplitter(
            target_column=self._target,
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date",
            group_column="entity_id",
            test_size=self._test_size,
            purge_gap_days=self._purge_gap_days,
            exclude_columns=exclude,
        )
        return splitter.split(split_df)

    def _extract_train_metadata(self, split_result: Any, df: DataFrame) -> tuple[Any, Any]:
        if split_result.train_metadata:
            return split_result.train_metadata["entity_id"], split_result.train_metadata["as_of_date"]
        train_idx = split_result.X_train.index
        return df.loc[train_idx, "entity_id"], df.loc[train_idx, "as_of_date"]

    def _fillna_and_drop_zero_variance(
        self,
        X_train: DataFrame,
        X_test: DataFrame,
    ) -> tuple[DataFrame, DataFrame, list[str], dict[str, int]]:
        if _is_spark_pandas(X_train):
            return self._spark_fillna_and_drop(X_train, X_test)
        combined_nulls, zero_var = self._pandas_nulls_and_zero_var(X_train, X_test)
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        if zero_var:
            X_train = X_train.drop(columns=zero_var)
            X_test = X_test.drop(columns=zero_var)
            for c in zero_var:
                combined_nulls.pop(c, None)
        return X_train, X_test, zero_var, combined_nulls

    def _spark_fillna_and_drop(
        self,
        X_train: DataFrame,
        X_test: DataFrame,
    ) -> tuple[DataFrame, DataFrame, list[str], dict[str, int]]:
        from customer_retention.core.compat.spark_backend import _as_pandas_api

        t0 = time.monotonic()
        train_spark = as_spark_df(X_train)
        test_spark = as_spark_df(X_test)
        self._log_sub(f"to spark: {time.monotonic() - t0:.1f}s")

        t1 = time.monotonic()
        combined_nulls, zero_var = self._spark_nulls_and_zero_var(X_train, X_test)
        self._log_sub(f"nulls+zerovar: {time.monotonic() - t1:.1f}s")

        t2 = time.monotonic()
        zero_set = set(zero_var)
        keep_cols = [c for c in train_spark.columns if c not in zero_set]
        train_spark = train_spark.select(keep_cols).na.fill(0.0)
        test_spark = test_spark.select(keep_cols).na.fill(0.0)
        train_spark = train_spark.localCheckpoint(eager=True)
        test_spark = test_spark.localCheckpoint(eager=True)
        self._log_sub(f"fillna+drop+checkpoint: {time.monotonic() - t2:.1f}s")

        for c in zero_var:
            combined_nulls.pop(c, None)
        return _as_pandas_api(train_spark), _as_pandas_api(test_spark), zero_var, combined_nulls

    _AGG_BATCH = 500

    def _spark_nulls_and_zero_var(
        self,
        X_train: DataFrame,
        X_test: DataFrame,
    ) -> tuple[dict[str, int], list[str]]:
        import pyspark.sql.functions as F  # noqa: N812
        from pyspark.sql.types import NumericType

        train_spark = as_spark_df(X_train)
        test_spark = as_spark_df(X_test)
        all_cols = list(train_spark.columns)
        num_set = {f.name for f in train_spark.schema.fields if isinstance(f.dataType, NumericType)}

        safe = {c: f"__a{i}__" for i, c in enumerate(all_cols)}
        train_work = train_spark.toDF(*[safe[c] for c in all_cols])
        test_work = test_spark.toDF(*[safe[c] for c in all_cols])

        train_nulls: dict[str, int] = {}
        train_std: dict[str, float | None] = {}
        test_nulls: dict[str, int] = {}

        t0 = time.monotonic()
        for start in range(0, len(all_cols), self._AGG_BATCH):
            batch = all_cols[start : start + self._AGG_BATCH]
            sb = [safe[c] for c in batch]
            exprs = []
            for c, s in zip(batch, sb):
                exprs.append(F.sum(F.when(F.col(s).isNull(), F.lit(1)).otherwise(F.lit(0))).alias(f"{s}__n"))
                if c in num_set:
                    exprs.append(F.stddev(F.coalesce(F.col(s).cast("double"), F.lit(0))).alias(f"{s}__s"))
            row = train_work.agg(*exprs).head()
            for c, s in zip(batch, sb):
                train_nulls[c] = int(row[f"{s}__n"] or 0)
                if c in num_set:
                    train_std[c] = row[f"{s}__s"]
        self._log_sub(f"agg train ({len(all_cols)} cols): {time.monotonic() - t0:.1f}s")

        t1 = time.monotonic()
        for start in range(0, len(all_cols), self._AGG_BATCH):
            batch = all_cols[start : start + self._AGG_BATCH]
            sb = [safe[c] for c in batch]
            exprs = [F.sum(F.when(F.col(s).isNull(), F.lit(1)).otherwise(F.lit(0))).alias(f"{s}__n") for s in sb]
            row = test_work.agg(*exprs).head()
            for c, s in zip(batch, sb):
                test_nulls[c] = int(row[f"{s}__n"] or 0)
        self._log_sub(f"agg test ({len(all_cols)} cols): {time.monotonic() - t1:.1f}s")

        combined_nulls = {c: train_nulls[c] + test_nulls[c] for c in all_cols}
        zero_var = [c for c in all_cols if c in num_set and (train_std.get(c) is None or train_std[c] == 0)]
        return combined_nulls, zero_var

    @staticmethod
    def _pandas_nulls_and_zero_var(
        X_train: DataFrame,
        X_test: DataFrame,
    ) -> tuple[dict[str, int], list[str]]:
        train_nulls = X_train.isnull().sum()
        test_nulls = X_test.isnull().sum()
        combined_nulls = {c: int(train_nulls[c]) + int(test_nulls[c]) for c in X_train.columns}
        filled = X_train.fillna(0)
        num = filled.select_dtypes(include="number")
        if num.empty:
            return combined_nulls, []
        stds = num.std()
        zero_var = stds[(stds == 0) | stds.isna()].index.tolist()
        return combined_nulls, zero_var

    def _finalize_distributed(
        self,
        X_train: DataFrame,
        X_test: DataFrame,
        y_train: Series,
        y_test: Series,
        train_entities: Series,
        train_dates: Series,
        feature_cols: list[str],
    ) -> TrainingPreparationResult:
        from customer_retention.core.compat.spark_backend import _as_pandas_api

        from .spark_feature_scaler import SparkFeatureScaler

        t0 = time.monotonic()
        train_bundle = concat(
            [
                X_train,
                y_train.rename("__y__"),
                train_entities.rename(_CV_ENTITY_COL),
                train_dates.rename(_CV_DATE_COL),
            ],
            axis=1,
        )
        train_bundle = spark_checkpoint(train_bundle)
        self._log_sub(f"checkpoint train ({len(train_bundle.columns)} cols): {time.monotonic() - t0:.1f}s")

        t1 = time.monotonic()
        test_bundle = concat([X_test, y_test.rename("__y__")], axis=1)
        test_bundle = spark_checkpoint(test_bundle)
        self._log_sub(f"checkpoint test: {time.monotonic() - t1:.1f}s")

        t2 = time.monotonic()
        train_spark = as_spark_df(train_bundle)
        test_spark = as_spark_df(test_bundle)
        scaler = SparkFeatureScaler(scaler_type=self._scaler_type)
        scaler._feature_names = feature_cols
        scaler._params = scaler._compute_params_spark(train_spark)
        self._log_sub(f"scaler params ({len(feature_cols)} features): {time.monotonic() - t2:.1f}s")

        t3 = time.monotonic()
        train_scaled_spark = scaler._apply_spark(train_spark)
        test_scaled_spark = scaler._apply_spark(test_spark)

        non_y_cols = [c for c in train_spark.columns if c != "__y__"]
        test_non_y = [c for c in test_spark.columns if c != "__y__"]
        X_train = _as_pandas_api(train_spark.select(non_y_cols))
        X_train_scaled = _as_pandas_api(train_scaled_spark.select(non_y_cols))
        X_test = _as_pandas_api(test_spark.select(test_non_y))
        X_test_scaled = _as_pandas_api(test_scaled_spark.select(test_non_y))
        if self._use_float32:
            X_train = spark_cast_float32(X_train)
            X_train_scaled = spark_cast_float32(X_train_scaled)
            X_test = spark_cast_float32(X_test)
            X_test_scaled = spark_cast_float32(X_test_scaled)
        y_train = train_bundle["__y__"]
        y_test = test_bundle["__y__"]

        y_test_np = collect_for_sklearn(y_test).to_numpy()
        train_entities = collect_for_sklearn(train_entities)
        train_dates = collect_for_sklearn(train_dates)
        self._log_sub(f"scale + wrap + collect: {time.monotonic() - t3:.1f}s")

        return TrainingPreparationResult(
            X_train=X_train,
            X_test=X_test,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_test_np=y_test_np,
            feature_names=feature_cols,
            train_entities=train_entities,
            train_dates=train_dates,
            split_info={},
            class_distribution={},
            zero_variance_dropped=[],
        )

    def _finalize_local(
        self,
        X_train: DataFrame,
        X_test: DataFrame,
        y_train: Series,
        y_test: Series,
        train_entities: Series,
        train_dates: Series,
        feature_cols: list[str],
    ) -> TrainingPreparationResult:
        X_train = spark_checkpoint(X_train)
        X_test = spark_checkpoint(X_test)

        if self._use_float32:
            X_train = spark_cast_float32(X_train)
            X_test = spark_cast_float32(X_test)

        X_train = collect_for_sklearn(X_train)
        X_test = collect_for_sklearn(X_test)
        y_train = collect_for_sklearn(y_train)
        y_test = collect_for_sklearn(y_test)
        train_entities = collect_for_sklearn(train_entities)
        train_dates = collect_for_sklearn(train_dates)
        gc.collect()

        scaler = FeatureScaler(scaler_type=self._scaler_type)
        scaling_result = scaler.fit_transform(X_train, X_test)
        X_train_scaled = scaling_result.X_train_scaled
        X_test_scaled = scaling_result.X_test_scaled

        if self._use_float32:
            X_train_scaled = X_train_scaled.astype("float32")
            X_test_scaled = X_test_scaled.astype("float32")
        X_train_scaled = X_train_scaled.fillna(0)
        X_test_scaled = X_test_scaled.fillna(0)

        X_train[_CV_ENTITY_COL] = train_entities.to_numpy()
        X_train[_CV_DATE_COL] = train_dates.to_numpy()

        return TrainingPreparationResult(
            X_train=X_train,
            X_test=X_test,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_test_np=y_test.to_numpy(),
            feature_names=feature_cols,
            train_entities=train_entities,
            train_dates=train_dates,
            split_info={},
            class_distribution={},
            zero_variance_dropped=[],
        )

    def _class_distribution(self, y: Series) -> dict[int, int]:
        vc = collect_for_sklearn(y.value_counts())
        return {int(k): int(v) for k, v in vc.items()}


def prepare_for_diagnostics(
    gold: DataFrame,
    *,
    target_column: str,
    feature_names: list[str],
    purge_gap_days: int,
    entity_column: str = "entity_id",
    date_column: str = "as_of_date",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Any:
    """Reproduce NB08's temporal train/test split from a saved gold table.

    NB09 uses the saved models on the exact train/test partition that NB08 trained on.
    Running ``TrainingPreparator.prepare`` again is fragile: its zero-variance drop
    and entity sampling can diverge from NB08's in-memory run and silently remove
    columns that the trained model still expects. Instead this helper:

    * Fails fast if any model feature is missing from gold (so the caller learns
      the gold table is out of date instead of silently corrupting diagnostics).
    * Selects only the final ``feature_names`` + split metadata.
    * Drops rows with a missing target and median-imputes the feature columns,
      mirroring the preparator's pre-split steps.
    * Delegates to ``DataSplitter`` with the same temporal/entity config NB08 used
      so the hash-based entity split and purge gap match exactly.

    Returns a :class:`~customer_retention.stages.modeling.data_splitter.SplitResult`.
    Works on both pandas and pyspark.pandas — heavy operations stay distributed
    and use batched bulk helpers.
    """
    missing = [c for c in feature_names if c not in gold.columns]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise KeyError(
            f"Gold table is missing {len(missing)} feature(s) the trained model expects: "
            f"{preview}{suffix}. Re-run NB08 so the saved gold contains every column listed "
            f"in exploration_diagnostics.json#feature_names."
        )

    selected = gold[list(feature_names) + [target_column, entity_column, date_column]]
    df = _drop_missing_target_rows(selected, target_column)
    df = bulk_median_impute(df, columns=list(feature_names))

    splitter = DataSplitter(
        target_column=target_column,
        strategy=SplitStrategy.TEMPORAL,
        temporal_column=date_column,
        group_column=entity_column,
        test_size=test_size,
        random_state=random_state,
        purge_gap_days=purge_gap_days,
        exclude_columns=[date_column, entity_column],
    )
    return splitter.split(df)


def _drop_missing_target_rows(df: DataFrame, target_column: str) -> DataFrame:
    if _is_spark_pandas(df):
        from pyspark.sql import functions as F  # noqa: N812

        from customer_retention.core.compat.spark_backend import _as_pandas_api

        spark_df = as_spark_df(df).filter(F.col(target_column).isNotNull())
        return _as_pandas_api(spark_df)
    return df[df[target_column].notna()]
