"""SparkTemporalFeatureEngineer — distributed temporal feature engineering.

When input is a native Spark DataFrame, all heavy computations (lagged windows,
lifecycle, recency, regularity) run as bulk Spark SQL operations.  Only aggregated
results (one row per entity) are collected to native pandas for the final merge.
Velocity, acceleration, and cohort operate on already-aggregated lag features.

When input is native pandas, delegates to TemporalFeatureEngineer (parent).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from customer_retention.core.compat import (
    _is_native_spark_df,
    _is_spark_pandas,
    as_spark_df,
)

from .temporal_feature_engineer import (
    FeatureGroup,
    FeatureGroupResult,
    ReferenceMode,
    TemporalFeatureEngineer,
    TemporalFeatureResult,
)

_SPARK_AGG = {"sum": "sum", "mean": "avg", "max": "max", "min": "min", "count": "count"}


def _to_spark(obj: Any) -> Any:
    if _is_native_spark_df(obj):
        return obj
    if _is_spark_pandas(obj):
        return as_spark_df(obj)
    return None


def _epoch(col_expr):
    import pyspark.sql.functions as F  # noqa: N812
    return F.unix_timestamp(col_expr.cast("timestamp")).cast("double")


def _disabled_group(group: FeatureGroup) -> FeatureGroupResult:
    return FeatureGroupResult(
        group=group, features=[],
        rationale=TemporalFeatureEngineer.RATIONALES[group], enabled=False)


def _lagged_windows_spark(spark_df, entity_col, time_col, value_cols, ref_spark, config):
    import pyspark.sql.functions as F  # noqa: N812

    df = spark_df.join(ref_spark, on=entity_col)
    df = df.withColumn("_days_before_ref",
        (_epoch(F.col("reference_date")) - _epoch(F.col(time_col))) / F.lit(86400.0))

    agg_exprs: list = []
    feature_names: list[str] = []

    for lag in range(config.num_lags):
        start, end = lag * config.lag_window_days, (lag + 1) * config.lag_window_days
        mask = (F.col("_days_before_ref") >= start) & (F.col("_days_before_ref") < end)
        for col in value_cols:
            masked = F.when(mask, F.col(col))
            for agg in config.lag_aggregations:
                name = f"lag{lag}_{col}_{agg}"
                feature_names.append(name)
                spark_agg_name = _SPARK_AGG.get(agg)
                if spark_agg_name is None:
                    raise ValueError(f"Unsupported aggregation: {agg!r}")
                spark_fn = getattr(F, spark_agg_name)
                if agg == "count":
                    agg_exprs.append(
                        F.coalesce(spark_fn(masked), F.lit(0)).cast("int").alias(name))
                else:
                    agg_exprs.append(spark_fn(masked).alias(name))

    agged = df.groupBy(entity_col).agg(*agg_exprs)
    result = ref_spark.select(entity_col).join(agged, on=entity_col, how="left")

    for lag in range(config.num_lags):
        for col in value_cols:
            if "count" in config.lag_aggregations:
                cname = f"lag{lag}_{col}_count"
                result = result.withColumn(
                    cname, F.coalesce(F.col(cname), F.lit(0)).cast("int"))

    return result, FeatureGroupResult(
        group=FeatureGroup.LAGGED_WINDOWS, features=feature_names,
        rationale=TemporalFeatureEngineer.RATIONALES[FeatureGroup.LAGGED_WINDOWS])


def _lifecycle_spark(spark_df, entity_col, time_col, value_cols, ref_spark, config):
    import pyspark.sql.functions as F  # noqa: N812

    history = spark_df.groupBy(entity_col).agg(
        F.min(time_col).alias("_first"), F.max(time_col).alias("_last"),
    ).withColumn("_history_secs", _epoch(F.col("_last")) - _epoch(F.col("_first")))

    eligible = history.filter(
        F.col("_history_secs").isNotNull()
        & (F.col("_history_secs") >= config.min_history_days * 86400))

    df = spark_df.join(eligible, on=entity_col)
    first_ep = _epoch(F.col("_first"))
    time_ep = _epoch(F.col(time_col))
    split1 = first_ep + F.col("_history_secs") * F.lit(config.lifecycle_splits[0])
    split2 = first_ep + F.col("_history_secs") * F.lit(
        config.lifecycle_splits[0] + config.lifecycle_splits[1])

    df = df.withColumn("_phase",
        F.when(time_ep < split1, "beginning")
         .when(time_ep < split2, "middle")
         .otherwise("end"))

    feature_names: list[str] = []
    result = ref_spark.select(entity_col)

    for col in value_cols:
        pivoted = (df.groupBy(entity_col)
                   .pivot("_phase", ["beginning", "middle", "end"])
                   .agg(F.sum(col)))
        for phase in ["beginning", "middle", "end"]:
            feat = f"{col}_{phase}"
            feature_names.append(feat)
            pivoted = pivoted.withColumnRenamed(phase, feat)

        result = result.join(pivoted, on=entity_col, how="left")

        trend = f"{col}_trend_ratio"
        feature_names.append(trend)
        beg = F.col(f"{col}_beginning")
        result = result.withColumn(trend,
            F.when(beg.isNotNull() & (beg > 0), F.col(f"{col}_end") / beg))

    return result, FeatureGroupResult(
        group=FeatureGroup.LIFECYCLE, features=feature_names,
        rationale=TemporalFeatureEngineer.RATIONALES[FeatureGroup.LIFECYCLE])


def _recency_spark(spark_df, entity_col, time_col, ref_spark):
    import pyspark.sql.functions as F  # noqa: N812

    stats = spark_df.groupBy(entity_col).agg(
        F.min(time_col).alias("_first"), F.max(time_col).alias("_last"))

    result = ref_spark.join(stats, on=entity_col, how="left")

    ref_ep = _epoch(F.col("reference_date"))
    days_since_last = (ref_ep - _epoch(F.col("_last"))) / F.lit(86400.0)
    days_since_first = (ref_ep - _epoch(F.col("_first"))) / F.lit(86400.0)
    active_span = (_epoch(F.col("_last")) - _epoch(F.col("_first"))) / F.lit(86400.0)

    result = (result
        .withColumn("days_since_last_event", days_since_last)
        .withColumn("days_since_first_event", days_since_first)
        .withColumn("active_span_days", active_span)
        .withColumn("recency_ratio",
            F.when(F.col("active_span_days") > 0,
                F.col("days_since_last_event") /
                (F.col("active_span_days") + F.col("days_since_last_event"))
            ).otherwise(F.lit(0.0)))
        .withColumn("recency_ratio",
            F.greatest(F.lit(0.0), F.least(F.lit(1.0), F.col("recency_ratio"))))
        .drop("_first", "_last", "reference_date"))

    return result, FeatureGroupResult(
        group=FeatureGroup.RECENCY,
        features=["days_since_last_event", "days_since_first_event",
                   "active_span_days", "recency_ratio"],
        rationale=TemporalFeatureEngineer.RATIONALES[FeatureGroup.RECENCY])


def _regularity_spark(spark_df, entity_col, time_col, ref_spark):
    import pyspark.sql.functions as F  # noqa: N812
    from pyspark.sql.window import Window

    w = Window.partitionBy(entity_col).orderBy(time_col)
    prev_ep = _epoch(F.lag(time_col).over(w))
    cur_ep = _epoch(F.col(time_col))

    gaps_df = (spark_df
        .withColumn("_gap_days", (cur_ep - prev_ep) / F.lit(86400.0))
        .filter(F.col("_gap_days").isNotNull()))

    gap_stats = gaps_df.groupBy(entity_col).agg(
        F.avg("_gap_days").alias("inter_event_gap_mean"),
        F.stddev("_gap_days").alias("inter_event_gap_std"),
        F.max("_gap_days").alias("inter_event_gap_max"))

    event_stats = spark_df.groupBy(entity_col).agg(
        _epoch(F.min(time_col)).alias("_first_ep"),
        _epoch(F.max(time_col)).alias("_last_ep"),
        F.count(time_col).alias("_cnt"),
    ).withColumn("_total_days",
        (F.col("_last_ep") - F.col("_first_ep")) / F.lit(86400.0))
    event_stats = event_stats.withColumn("event_frequency",
        F.when(F.col("_total_days") > 0,
            F.col("_cnt") / F.col("_total_days") * F.lit(30.0))
         .otherwise(F.col("_cnt").cast("double")))

    result = ref_spark.select(entity_col)
    result = result.join(gap_stats, on=entity_col, how="left")
    result = result.join(
        event_stats.select(entity_col, "event_frequency"),
        on=entity_col, how="left")

    gap_mean = F.col("inter_event_gap_mean")
    gap_std = F.coalesce(F.col("inter_event_gap_std"), F.lit(0.0))
    result = result.withColumn("regularity_score",
        F.when(gap_mean.isNotNull() & (gap_mean > 0),
            F.greatest(F.lit(0.0), F.lit(1.0) - gap_std / gap_mean))
         .when(gap_mean.isNotNull() & (gap_mean == 0), F.lit(1.0)))

    return result, FeatureGroupResult(
        group=FeatureGroup.REGULARITY,
        features=["event_frequency", "inter_event_gap_mean",
                   "inter_event_gap_std", "inter_event_gap_max",
                   "regularity_score"],
        rationale=TemporalFeatureEngineer.RATIONALES[FeatureGroup.REGULARITY])


class SparkTemporalFeatureEngineer(TemporalFeatureEngineer):

    def compute(
        self,
        events_df: Any,
        entity_col: str,
        time_col: str,
        value_cols: List[str],
        reference_dates: Optional[Any] = None,
        reference_col: Optional[str] = None,
    ) -> TemporalFeatureResult:
        spark_df = _to_spark(events_df)
        if spark_df is None:
            return super().compute(
                events_df, entity_col, time_col, value_cols,
                reference_dates, reference_col)

        import pyspark.sql.functions as F  # noqa: N812

        spark_df = spark_df.withColumn(time_col, F.to_timestamp(F.col(time_col)))
        for vc in value_cols:
            if vc in spark_df.columns:
                spark_df = spark_df.withColumn(vc, F.col(vc).cast("double"))

        ref_spark = self._resolve_ref_dates_spark(
            spark_df, entity_col, time_col, reference_dates, reference_col)

        lag_spark, lag_group = _lagged_windows_spark(
            spark_df, entity_col, time_col, value_cols, ref_spark, self.config)
        lag_pd = lag_spark.toPandas()

        all_features: list = [lag_pd]
        feature_groups: list[FeatureGroupResult] = [lag_group]

        self._append_velocity(all_features, feature_groups, lag_pd, value_cols, entity_col)
        self._append_distributed_groups(
            all_features, feature_groups,
            spark_df, entity_col, time_col, value_cols, ref_spark)
        self._append_cohort(all_features, feature_groups, lag_pd, value_cols, entity_col)

        result_df = all_features[0]
        for df in all_features[1:]:
            result_df = result_df.merge(df, on=entity_col, how="left")

        return TemporalFeatureResult(
            features_df=result_df, feature_groups=feature_groups,
            config=self.config, entity_col=entity_col, value_cols=value_cols)

    def _append_velocity(self, all_features, feature_groups, lag_pd, value_cols, entity_col):
        if self.config.compute_velocity:
            feat, grp = self._compute_velocity(lag_pd, value_cols)
            all_features.append(feat)
            feature_groups.append(grp)
        else:
            feature_groups.append(_disabled_group(FeatureGroup.VELOCITY))

        if self.config.compute_acceleration and self.config.compute_velocity:
            feat, grp = self._compute_acceleration(
                all_features[1] if len(all_features) > 1 else lag_pd,
                lag_pd, value_cols, entity_col)
            all_features.append(feat)
            feature_groups.append(grp)
        else:
            feature_groups.append(_disabled_group(FeatureGroup.ACCELERATION))

    def _append_distributed_groups(
        self, all_features, feature_groups,
        spark_df, entity_col, time_col, value_cols, ref_spark,
    ):
        if self.config.compute_lifecycle:
            spark_result, grp = _lifecycle_spark(
                spark_df, entity_col, time_col, value_cols, ref_spark, self.config)
            all_features.append(spark_result.toPandas())
            feature_groups.append(grp)
        else:
            feature_groups.append(_disabled_group(FeatureGroup.LIFECYCLE))

        if self.config.compute_recency:
            spark_result, grp = _recency_spark(
                spark_df, entity_col, time_col, ref_spark)
            all_features.append(spark_result.toPandas())
            feature_groups.append(grp)
        else:
            feature_groups.append(_disabled_group(FeatureGroup.RECENCY))

        if self.config.compute_regularity:
            spark_result, grp = _regularity_spark(
                spark_df, entity_col, time_col, ref_spark)
            all_features.append(spark_result.toPandas())
            feature_groups.append(grp)
        else:
            feature_groups.append(_disabled_group(FeatureGroup.REGULARITY))

    def _append_cohort(self, all_features, feature_groups, lag_pd, value_cols, entity_col):
        if self.config.compute_cohort:
            feat, grp = self._compute_cohort_comparison(lag_pd, value_cols, entity_col)
            all_features.append(feat)
            feature_groups.append(grp)
        else:
            feature_groups.append(_disabled_group(FeatureGroup.COHORT_COMPARISON))

    def _resolve_ref_dates_spark(self, spark_df, entity_col, time_col, reference_dates, reference_col):
        import pyspark.sql.functions as F  # noqa: N812

        if self.config.reference_mode == ReferenceMode.GLOBAL_DATE:
            ref_date = self.config.global_reference_date or datetime.now()
            return spark_df.select(entity_col).distinct().withColumn(
                "reference_date", F.lit(ref_date).cast("timestamp"))

        if reference_dates is not None and reference_col is not None:
            ref_spark = _to_spark(reference_dates)
            if ref_spark is None:
                ref_spark = spark_df.sparkSession.createDataFrame(reference_dates)
            return ref_spark.select(
                F.col(entity_col),
                F.to_timestamp(F.col(reference_col)).alias("reference_date"))

        return spark_df.groupBy(entity_col).agg(
            F.max(time_col).alias("reference_date"))
