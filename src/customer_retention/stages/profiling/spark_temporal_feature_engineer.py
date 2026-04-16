"""SparkTemporalFeatureEngineer — fully distributed temporal feature engineering.

All 7 feature groups run as native Spark SQL operations. Results are joined in
Spark and returned as pyspark.pandas — nothing is collected to the driver.

When input is native pandas, delegates to TemporalFeatureEngineer (parent).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from customer_retention.core.compat import (
    _is_native_spark_df,
    _is_spark_pandas,
    as_pandas_api,
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


def _velocity_acceleration_spark(lag_spark, entity_col, value_cols, config):
    import pyspark.sql.functions as F  # noqa: N812

    velocity_names: list[str] = []
    accel_names: list[str] = []
    result = lag_spark
    cols = set(lag_spark.columns)

    for col in value_cols:
        lag0, lag1, lag2 = f"lag0_{col}_sum", f"lag1_{col}_sum", f"lag2_{col}_sum"

        if config.compute_velocity and lag0 in cols and lag1 in cols:
            v_name = f"{col}_velocity"
            result = result.withColumn(v_name,
                (F.col(lag0) - F.col(lag1)) / F.lit(float(config.lag_window_days)))
            velocity_names.append(v_name)

            vp_name = f"{col}_velocity_pct"
            safe_lag1 = F.when(F.col(lag1) == 0, F.lit(None).cast("double")).otherwise(F.col(lag1))
            result = result.withColumn(vp_name, (F.col(lag0) - safe_lag1) / safe_lag1)
            velocity_names.append(vp_name)

        if config.compute_acceleration and config.compute_velocity:
            if lag1 in cols and lag2 in cols:
                v01 = (F.col(lag0) - F.col(lag1)) / F.lit(float(config.lag_window_days))
                v12 = (F.col(lag1) - F.col(lag2)) / F.lit(float(config.lag_window_days))
                a_name = f"{col}_acceleration"
                result = result.withColumn(a_name, v01 - v12)
                accel_names.append(a_name)

            v_name = f"{col}_velocity"
            if v_name in velocity_names and lag0 in cols:
                m_name = f"{col}_momentum"
                result = result.withColumn(m_name, F.col(lag0) * F.col(v_name))
                accel_names.append(m_name)

    select_cols = [entity_col] + velocity_names + accel_names
    velocity_group = FeatureGroupResult(
        group=FeatureGroup.VELOCITY, features=velocity_names,
        rationale=TemporalFeatureEngineer.RATIONALES[FeatureGroup.VELOCITY],
        enabled=config.compute_velocity)
    accel_group = FeatureGroupResult(
        group=FeatureGroup.ACCELERATION, features=accel_names,
        rationale=TemporalFeatureEngineer.RATIONALES[FeatureGroup.ACCELERATION],
        enabled=config.compute_acceleration and config.compute_velocity)

    return result.select(*select_cols), velocity_group, accel_group


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
            ).otherwise(F.lit(None).cast("double")))
        .withColumn("recency_ratio",
            F.when(F.col("recency_ratio").isNotNull(),
                F.greatest(F.lit(0.0), F.least(F.lit(1.0), F.col("recency_ratio")))))
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


def _cohort_spark(lag_spark, entity_col, value_cols):
    import pyspark.sql.functions as F  # noqa: N812
    from pyspark.sql.window import Window

    w_global = Window.partitionBy()
    result = lag_spark
    feature_names: list[str] = []
    cols = set(lag_spark.columns)

    for col in value_cols:
        lag0 = f"lag0_{col}_sum"
        if lag0 not in cols:
            continue

        cohort_mean = F.avg(F.col(lag0)).over(w_global)
        cohort_std = F.stddev(F.col(lag0)).over(w_global)

        vs_mean = f"{col}_vs_cohort_mean"
        result = result.withColumn(vs_mean, F.col(lag0) - cohort_mean)
        feature_names.append(vs_mean)

        vs_pct = f"{col}_vs_cohort_pct"
        result = result.withColumn(vs_pct,
            F.when(cohort_mean != 0, F.col(lag0) / cohort_mean))
        feature_names.append(vs_pct)

        z_name = f"{col}_cohort_zscore"
        result = result.withColumn(z_name,
            F.when(cohort_std.isNotNull() & (cohort_std > 0),
                (F.col(lag0) - cohort_mean) / cohort_std))
        feature_names.append(z_name)

    return result.select(entity_col, *feature_names), FeatureGroupResult(
        group=FeatureGroup.COHORT_COMPARISON, features=feature_names,
        rationale=TemporalFeatureEngineer.RATIONALES[FeatureGroup.COHORT_COMPARISON])


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

        spark_df = spark_df.withColumn(time_col, F.expr(f"try_to_timestamp(`{time_col}`)"))
        for vc in value_cols:
            if vc in spark_df.columns:
                spark_df = spark_df.withColumn(vc, F.col(vc).cast("double"))

        ref_spark = self._resolve_ref_dates_spark(
            spark_df, entity_col, time_col, reference_dates, reference_col)

        lag_spark, lag_group = _lagged_windows_spark(
            spark_df, entity_col, time_col, value_cols, ref_spark, self.config)

        spark_parts: list = [lag_spark]
        feature_groups: list[FeatureGroupResult] = [lag_group]

        self._append_velocity_acceleration(
            spark_parts, feature_groups, lag_spark, entity_col, value_cols)
        self._append_distributed_groups(
            spark_parts, feature_groups,
            spark_df, entity_col, time_col, value_cols, ref_spark)
        self._append_cohort(
            spark_parts, feature_groups, lag_spark, entity_col, value_cols)

        merged = spark_parts[0]
        for part in spark_parts[1:]:
            merged = merged.join(part, on=entity_col, how="left")

        return TemporalFeatureResult(
            features_df=as_pandas_api(merged), feature_groups=feature_groups,
            config=self.config, entity_col=entity_col, value_cols=value_cols)

    def _append_velocity_acceleration(
        self, spark_parts, feature_groups, lag_spark, entity_col, value_cols,
    ):
        if self.config.compute_velocity:
            va_spark, v_grp, a_grp = _velocity_acceleration_spark(
                lag_spark, entity_col, value_cols, self.config)
            spark_parts.append(va_spark)
            feature_groups.append(v_grp)
            feature_groups.append(a_grp)
        else:
            feature_groups.append(_disabled_group(FeatureGroup.VELOCITY))
            feature_groups.append(_disabled_group(FeatureGroup.ACCELERATION))

    def _append_distributed_groups(
        self, spark_parts, feature_groups,
        spark_df, entity_col, time_col, value_cols, ref_spark,
    ):
        if self.config.compute_lifecycle:
            spark_result, grp = _lifecycle_spark(
                spark_df, entity_col, time_col, value_cols, ref_spark, self.config)
            spark_parts.append(spark_result)
            feature_groups.append(grp)
        else:
            feature_groups.append(_disabled_group(FeatureGroup.LIFECYCLE))

        if self.config.compute_recency:
            spark_result, grp = _recency_spark(
                spark_df, entity_col, time_col, ref_spark)
            spark_parts.append(spark_result)
            feature_groups.append(grp)
        else:
            feature_groups.append(_disabled_group(FeatureGroup.RECENCY))

        if self.config.compute_regularity:
            spark_result, grp = _regularity_spark(
                spark_df, entity_col, time_col, ref_spark)
            spark_parts.append(spark_result)
            feature_groups.append(grp)
        else:
            feature_groups.append(_disabled_group(FeatureGroup.REGULARITY))

    def _append_cohort(self, spark_parts, feature_groups, lag_spark, entity_col, value_cols):
        if self.config.compute_cohort:
            spark_result, grp = _cohort_spark(lag_spark, entity_col, value_cols)
            spark_parts.append(spark_result)
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
                F.expr(f"try_to_timestamp(`{reference_col}`)").alias("reference_date"))

        return spark_df.groupBy(entity_col).agg(
            F.max(time_col).alias("reference_date"))
