from __future__ import annotations

import logging
import time
from typing import Any

from customer_retention.core.compat import (
    _is_native_spark_df,
    _is_spark_pandas,
    as_spark_df,
    as_tz_naive,
    native_pd,
    normalize_timestamps,
    pandas_dtype_to_spark_schema,
    safe_drop_duplicates,
    to_datetime,
)
from customer_retention.core.compat.detection import get_spark_session
from customer_retention.core.compat.spark_backend import _as_pandas_api
from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.stages.temporal.temporal_merger import (
    DatasetMergeInput,
    MergeReport,
    TemporalMerger,
)

logger = logging.getLogger(__name__)


def _empty_spine_schema(entity_key: str, as_of_column: str):
    from pyspark.sql.types import StringType, StructField, StructType, TimestampNTZType

    return StructType([
        StructField(entity_key, StringType(), True),
        StructField(as_of_column, TimestampNTZType(), True),
    ])


def _to_native_spark(df: Any) -> Any:
    """Convert any DataFrame to a native Spark DataFrame."""
    if _is_native_spark_df(df):
        return df
    if _is_spark_pandas(df):
        return as_spark_df(df)
    # pandas DataFrame
    spark = get_spark_session()
    pdf = normalize_timestamps(df)
    return spark.createDataFrame(pdf, schema=pandas_dtype_to_spark_schema(pdf))


def _spark_rename_columns(sdf: Any, rename_map: dict[str, str]) -> Any:
    """Apply column renames on a native Spark DataFrame."""
    for old_name, new_name in rename_map.items():
        sdf = sdf.withColumnRenamed(old_name, new_name)
    return sdf


def _break_lineage(sdf: Any) -> Any:
    """Truncate Spark execution plan via local checkpoint.

    Prevents driver OOM on wide merges by materialising the DataFrame
    in executor storage and discarding the accumulated join lineage.
    """
    return sdf.localCheckpoint(eager=True)


_SCRATCH_TABLE_PREFIX = "_silver_merge_ckpt_"


def _break_lineage_delta(
    sdf: Any,
    namespace: str,
    scratch_tables: list[str] | None,
) -> Any:
    """Detach Spark logical plan via Delta scratch-table round-trip.

    Used in place of ``_break_lineage`` (``localCheckpoint(eager=True)``)
    when the merged projection is too wide for Catalyst's binding pass —
    at the SPS scale (~1900 columns × N joins) ``localCheckpoint`` trips
    ``[INTERNAL_ERROR_ATTRIBUTE_NOT_FOUND]`` (SQLSTATE XX000) because
    attribute IDs collide during plan canonicalisation.

    Writing to Delta and re-reading is fully distributed (executors
    write/read in parallel, no driver materialisation) and produces a
    fresh logical plan with no accumulated multi-join lineage.

    Parameters
    ----------
    sdf:
        Native Spark DataFrame to materialise.
    namespace:
        ``"catalog.schema"`` where scratch tables are written.
    scratch_tables:
        Mutable list to which the new scratch table's fully-qualified
        name is appended; used by ``SparkTemporalMerger.cleanup_scratch_tables``.
    """
    import uuid

    scratch_name = f"{_SCRATCH_TABLE_PREFIX}{uuid.uuid4().hex[:12]}"
    table = f"{namespace}.{scratch_name}"
    (
        sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table)
    )
    if scratch_tables is not None:
        scratch_tables.append(table)
    spark = get_spark_session()
    return spark.table(table)


class SparkTemporalMerger(TemporalMerger):
    def __init__(self, config=None):
        super().__init__(config)
        self._scratch_tables: list[str] = []

    def cleanup_scratch_tables(self, spark: Any | None = None) -> int:
        """Drop every Delta scratch table created during ``merge_all``.

        Safe to call repeatedly — already-dropped tables are no-ops.
        Returns the number of tables dropped. Always called for the
        instance's own scratch list; callers wanting to sweep
        independently can ``SHOW TABLES … LIKE '_silver_merge_ckpt_*'``.
        """
        if spark is None:
            spark = get_spark_session()
        if spark is None:
            return 0
        dropped = 0
        for table in list(self._scratch_tables):
            try:
                spark.sql(f"DROP TABLE IF EXISTS {table}")
                dropped += 1
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.warning("scratch cleanup: failed to drop %s", table)
        self._scratch_tables.clear()
        return dropped

    def merge_one(
        self,
        merged_sdf: Any,
        ds: DatasetMergeInput,
    ) -> tuple[Any, set[str]]:
        """Merge a single dataset into the accumulated result (native Spark).

        Returns ``(merged_sdf, new_column_names)``.  Does **not** break
        lineage — the caller is responsible for materialisation (e.g. a
        Delta write/read cycle).
        """
        existing_cols = set(merged_sdf.columns)
        right_sdf = _to_native_spark(ds.df)

        if ds.granularity == DatasetGranularity.EVENT_LEVEL:
            merged_sdf = self._spark_join_event(
                merged_sdf, right_sdf, ds, existing_cols,
            )
        elif ds.feature_timestamp_column and ds.feature_timestamp_column in set(right_sdf.columns):
            merged_sdf = self._spark_join_asof(
                merged_sdf, right_sdf, ds, existing_cols,
            )
        else:
            if ds.feature_timestamp_column:
                logger.warning(
                    "Dataset '%s': feature_timestamp_column '%s' not in columns, "
                    "using broadcast join",
                    ds.name, ds.feature_timestamp_column,
                )
            merged_sdf = self._spark_join_broadcast(
                merged_sdf, right_sdf, ds, existing_cols,
            )

        new_cols = set(merged_sdf.columns) - existing_cols
        return merged_sdf, new_cols

    def merge_all(
        self,
        spine: Any,
        datasets: list[DatasetMergeInput],
    ) -> tuple[Any, Any]:
        spark = get_spark_session()
        if spark is None:
            return super().merge_all(spine, datasets)

        entity_key = self.config.entity_key
        as_of_column = self.config.as_of_column
        checkpoint_every = max(1, self.config.checkpoint_every)

        merged_sdf = _to_native_spark(spine)

        t_spine = time.monotonic()
        report = MergeReport(
            spine_rows=merged_sdf.count(),
            spine_entities=merged_sdf.select(entity_key).distinct().count(),
            spine_dates=merged_sdf.select(as_of_column).distinct().count(),
        )
        report.spine_stats_seconds = time.monotonic() - t_spine

        t_loop = time.monotonic()
        n_datasets = len(datasets)
        for i, ds in enumerate(datasets):
            t_ds = time.monotonic()
            merged_sdf, new_cols = self.merge_one(merged_sdf, ds)
            ds_seconds = time.monotonic() - t_ds

            is_last = (i + 1) == n_datasets
            on_boundary = (i + 1) % checkpoint_every == 0
            if is_last or on_boundary:
                t_ckpt = time.monotonic()
                if self.config.scratch_namespace:
                    merged_sdf = _break_lineage_delta(
                        merged_sdf,
                        self.config.scratch_namespace,
                        self._scratch_tables,
                    )
                else:
                    merged_sdf = _break_lineage(merged_sdf)
                report.checkpoint_seconds += time.monotonic() - t_ckpt
                report.checkpoint_count += 1

            report.datasets_merged.append(ds.name)
            report.columns_per_dataset[ds.name] = len(new_cols)
            report.seconds_per_dataset[ds.name] = ds_seconds
            logger.info(
                "Merged %d/%d '%s': +%d cols → %d total (%.1fs)",
                i + 1, n_datasets, ds.name, len(new_cols),
                len(merged_sdf.columns), ds_seconds,
            )

        report.total_columns = len(merged_sdf.columns)
        report.renamed_columns = {
            col: col
            for col in merged_sdf.columns
            if self.config.conflict_separator in col
        }

        if self.config.validate_temporal:
            from customer_retention.stages.temporal.point_in_time_join import (
                PointInTimeJoiner,
            )
            t_validate = time.monotonic()
            result_psdf = _as_pandas_api(merged_sdf)
            report.temporal_integrity = (
                PointInTimeJoiner.validate_temporal_integrity(result_psdf)
            )
            report.validation_seconds = time.monotonic() - t_validate
            report.merge_total_seconds = time.monotonic() - t_loop
            return result_psdf, report

        report.merge_total_seconds = time.monotonic() - t_loop
        return _as_pandas_api(merged_sdf), report

    def _spark_join_event(
        self,
        left_sdf: Any,
        right_sdf: Any,
        ds: DatasetMergeInput,
        existing_cols: set[str],
    ) -> Any:
        """Equi-join on (entity_key, as_of_date) for event-level datasets."""
        entity_key = self.config.entity_key
        as_of_column = self.config.as_of_column
        join_cols = [entity_key, as_of_column]

        if as_of_column not in right_sdf.columns:
            right_sdf = right_sdf.dropDuplicates([entity_key])
            return self._spark_join_broadcast(
                left_sdf, right_sdf, ds, existing_cols,
            )

        right_sdf = right_sdf.dropDuplicates(join_cols)

        right_feature_cols = set(right_sdf.columns) - set(join_cols)
        rename_map = self._resolve_conflicts(
            existing_cols, right_feature_cols, set(join_cols),
            ds.name, self.config.conflict_separator,
        )
        if rename_map:
            right_sdf = _spark_rename_columns(right_sdf, rename_map)

        return left_sdf.join(right_sdf, on=join_cols, how="left")

    def _spark_join_broadcast(
        self,
        left_sdf: Any,
        right_sdf: Any,
        ds: DatasetMergeInput,
        existing_cols: set[str],
    ) -> Any:
        """Broadcast join on entity_key for entity-level datasets."""
        import pyspark.sql.functions as F  # noqa: N812

        entity_key = self.config.entity_key
        join_keys = {entity_key}

        right_sdf = right_sdf.dropDuplicates([entity_key])

        right_feature_cols = set(right_sdf.columns) - join_keys
        rename_map = self._resolve_conflicts(
            existing_cols, right_feature_cols, join_keys,
            ds.name, self.config.conflict_separator,
        )
        if rename_map:
            right_sdf = _spark_rename_columns(right_sdf, rename_map)

        return left_sdf.join(F.broadcast(right_sdf), on=entity_key, how="left")

    def _spark_join_asof(
        self,
        left_sdf: Any,
        right_sdf: Any,
        ds: DatasetMergeInput,
        existing_cols: set[str],
    ) -> Any:
        """Window-based point-in-time (as-of) join for entity datasets with timestamps."""
        import pyspark.sql.functions as F  # noqa: N812
        from pyspark.sql import Window

        entity_key = self.config.entity_key
        as_of_column = self.config.as_of_column
        ft_col = ds.feature_timestamp_column

        join_keys = {entity_key, as_of_column}
        right_feature_cols = set(right_sdf.columns) - {entity_key, ft_col}
        rename_map = self._resolve_conflicts(
            existing_cols, right_feature_cols, join_keys,
            ds.name, self.config.conflict_separator,
        )
        if rename_map:
            right_sdf = _spark_rename_columns(right_sdf, rename_map)
            ft_col = rename_map.get(ft_col, ft_col)

        feature_cols = [
            c for c in right_sdf.columns if c not in {entity_key, ft_col}
        ]

        # Use a unique alias for the right timestamp to avoid ambiguity
        right_ts_alias = f"__{ds.name}_ts__"
        right_sdf = right_sdf.withColumnRenamed(ft_col, right_ts_alias)

        # Inner join on entity_key, filter right_ts <= as_of_date
        joined = left_sdf.join(right_sdf, on=entity_key, how="inner")
        joined = joined.filter(F.col(right_ts_alias) <= F.col(as_of_column))

        # Window: pick the most recent right row per (entity_key, as_of_date)
        w = Window.partitionBy(entity_key, as_of_column).orderBy(
            F.col(right_ts_alias).desc()
        )
        joined = joined.withColumn("_rn_", F.row_number().over(w))
        joined = joined.filter(F.col("_rn_") == 1).drop("_rn_", right_ts_alias)

        # Left join back to preserve all spine rows
        best_cols = [entity_key, as_of_column] + feature_cols
        best = joined.select(*[c for c in best_cols if c in joined.columns])

        return left_sdf.join(best, on=[entity_key, as_of_column], how="left")

    def build_spine(
        self, entity_ids: Any, grid_dates: list[str]
    ) -> Any:
        spark = get_spark_session()
        parsed_dates = as_tz_naive(to_datetime(grid_dates))

        # Native Spark DataFrame path — entities stay distributed, no driver collect
        if spark and _is_native_spark_df(entity_ids):
            return self._build_spine_from_spark_df(
                spark, entity_ids, parsed_dates,
            )

        unique_entities = entity_ids.drop_duplicates().reset_index(drop=True)

        if len(unique_entities) == 0 or len(parsed_dates) == 0:
            if spark:
                schema = _empty_spine_schema(
                    self.config.entity_key, self.config.as_of_column
                )
                return _as_pandas_api(spark.createDataFrame([], schema))
            return native_pd.DataFrame(
                columns=[self.config.entity_key, self.config.as_of_column]
            ).astype({self.config.as_of_column: "datetime64[ns]"})

        if not spark:
            idx = native_pd.MultiIndex.from_product(
                [unique_entities, parsed_dates],
                names=[self.config.entity_key, self.config.as_of_column],
            )
            return idx.to_frame(index=False)

        entities_pdf = native_pd.DataFrame(
            {self.config.entity_key: unique_entities.to_numpy()}
        )
        dates_pdf = normalize_timestamps(
            native_pd.DataFrame(
                {self.config.as_of_column: parsed_dates}
            )
        )

        entities_sdf = spark.createDataFrame(entities_pdf)
        dates_sdf = spark.createDataFrame(
            dates_pdf, schema=pandas_dtype_to_spark_schema(dates_pdf)
        )

        spine_sdf = entities_sdf.crossJoin(dates_sdf)
        return _as_pandas_api(spine_sdf)

    def _build_spine_from_spark_df(
        self, spark: Any, entity_ids_sdf: Any, parsed_dates: Any,
    ) -> Any:
        """Build spine from a native Spark DataFrame — no driver collect."""
        entity_col = entity_ids_sdf.columns[0]
        entities_sdf = entity_ids_sdf.dropDuplicates()
        if entity_col != self.config.entity_key:
            entities_sdf = entities_sdf.withColumnRenamed(
                entity_col, self.config.entity_key,
            )
        if len(parsed_dates) == 0:
            schema = _empty_spine_schema(
                self.config.entity_key, self.config.as_of_column,
            )
            return _as_pandas_api(spark.createDataFrame([], schema))
        dates_pdf = normalize_timestamps(
            native_pd.DataFrame({self.config.as_of_column: parsed_dates})
        )
        dates_sdf = spark.createDataFrame(
            dates_pdf, schema=pandas_dtype_to_spark_schema(dates_pdf),
        )
        return _as_pandas_api(entities_sdf.crossJoin(dates_sdf))

    def _merge_entity_asof(
        self,
        spine: Any,
        dataset: DatasetMergeInput,
        existing_cols: set[str],
    ) -> Any:
        entity_key = self.config.entity_key
        left_time_col = self.config.as_of_column
        ft_col = dataset.feature_timestamp_column
        right_df = dataset.df.copy()

        join_keys = {entity_key, left_time_col}
        right_feature_cols = set(right_df.columns) - {entity_key, ft_col}
        rename_map = self._resolve_conflicts(
            existing_cols, right_feature_cols, join_keys,
            dataset.name, self.config.conflict_separator,
        )
        if rename_map:
            right_df = right_df.rename(columns=rename_map)
            ft_col = rename_map.get(ft_col, ft_col)

        right_time_col = ft_col
        feature_cols = [
            c for c in right_df.columns if c not in {entity_key, right_time_col}
        ]

        joined = spine.merge(right_df, on=entity_key, how="inner")

        joined[left_time_col] = to_datetime(joined[left_time_col])
        joined[right_time_col] = to_datetime(joined[right_time_col])

        joined = joined[joined[right_time_col] <= joined[left_time_col]]

        joined = joined.sort_values(
            [entity_key, left_time_col, right_time_col],
            ascending=[True, True, False],
        )

        joined = safe_drop_duplicates(
            joined, subset=[entity_key, left_time_col], keep="first"
        )

        keep_cols = [entity_key, left_time_col] + feature_cols
        best = joined[[c for c in keep_cols if c in joined.columns]]

        result = spine.merge(best, on=[entity_key, left_time_col], how="left")
        return result
