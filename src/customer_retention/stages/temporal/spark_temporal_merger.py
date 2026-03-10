from __future__ import annotations

from typing import Any

from customer_retention.core.compat import (
    _is_native_spark_df,
    as_tz_naive,
    native_pd,
    normalize_timestamps,
    pandas_dtype_to_spark_schema,
    safe_drop_duplicates,
    to_datetime,
)
from customer_retention.core.compat.detection import get_spark_session
from customer_retention.core.compat.spark_backend import _as_pandas_api
from customer_retention.stages.temporal.temporal_merger import (
    DatasetMergeInput,
    TemporalMerger,
)


def _empty_spine_schema(entity_key: str, as_of_column: str):
    from pyspark.sql.types import StringType, StructField, StructType, TimestampNTZType

    return StructType([
        StructField(entity_key, StringType(), True),
        StructField(as_of_column, TimestampNTZType(), True),
    ])


class SparkTemporalMerger(TemporalMerger):
    def merge_all(
        self,
        spine: Any,
        datasets: list[DatasetMergeInput],
    ) -> tuple[Any, Any]:
        converted = []
        for ds in datasets:
            df = ds.df
            if _is_native_spark_df(df):
                df = _as_pandas_api(df)
            converted.append(DatasetMergeInput(
                name=ds.name,
                df=df,
                granularity=ds.granularity,
                feature_timestamp_column=ds.feature_timestamp_column,
            ))
        return super().merge_all(spine, converted)

    def build_spine(
        self, entity_ids: Any, grid_dates: list[str]
    ) -> Any:
        unique_entities = entity_ids.drop_duplicates().reset_index(drop=True)
        parsed_dates = as_tz_naive(to_datetime(grid_dates))
        spark = get_spark_session()

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
