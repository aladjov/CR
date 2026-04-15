from __future__ import annotations

import math
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Iterable, Optional

from customer_retention.core.compat import (
    _is_spark_pandas,
    as_spark_df,
    concat,
    head_as_list,
    is_numeric_dtype,
    native_pd,
    pd,
    qcut,
    safe_isin,
    safe_query,
    safe_sample,
    safe_to_datetime,
)

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.project_context import IntentConfig

_SEGMENT_ID_COL = "_segment_entity_id"


def apply_temporal_lookback(df: Any, time_col: str, intent: IntentConfig) -> Any:
    if intent.lookback_periods is None:
        return df
    from customer_retention.analysis.auto_explorer.snapshot_grid import CADENCE_DAYS

    ts = safe_to_datetime(df[time_col], errors="coerce")
    upper = ts.max()
    if native_pd.isna(upper):
        return df
    cap = native_pd.Timestamp(intent.history_upper_limit) if intent.history_upper_limit else None
    if cap is not None and cap < upper:
        upper = cap
    lookback_days = intent.lookback_periods * CADENCE_DAYS[intent.cadence_interval]
    lower = upper - timedelta(days=lookback_days)
    mask = ts >= lower
    if cap is not None:
        mask = mask & (ts <= upper)
    return df[mask]


class SegmentEntitySelection:
    """Entity IDs that pass the segment filters.

    On native pandas the IDs live in a set. On pyspark.pandas they stay as a
    single-column Spark DataFrame so the driver never has to hold the full list
    in memory — intersections and membership tests use Spark semi/inner joins.
    """

    __slots__ = ("_data", "_count")

    def __init__(self, data: Any):
        self._data: Any = data
        self._count: Optional[int] = None

    @classmethod
    def from_set(cls, ids: Iterable) -> "SegmentEntitySelection":
        return cls(set(ids))

    @classmethod
    def from_spark_df(cls, spark_df: Any, entity_col: str) -> "SegmentEntitySelection":
        renamed = spark_df.withColumnRenamed(entity_col, _SEGMENT_ID_COL) \
            if entity_col != _SEGMENT_ID_COL else spark_df
        # Cache the passing-entity DataFrame so `.count()` materializes it once
        # and the downstream semi-join in `filter_frame` reuses the cached
        # block — otherwise Spark rescans the filter subquery on every action.
        return cls(renamed.select(_SEGMENT_ID_COL).cache())

    @property
    def is_distributed(self) -> bool:
        return not isinstance(self._data, (set, frozenset))

    def __len__(self) -> int:
        if self._count is None:
            self._count = int(self._data.count()) if self.is_distributed else len(self._data)
        return self._count

    def __contains__(self, value: Any) -> bool:
        if self.is_distributed:
            raise TypeError(
                "SegmentEntitySelection is distributed — use .filter_frame() "
                "instead of 'in' checks (collecting would OOM the driver).",
            )
        return value in self._data

    def __iter__(self):
        if self.is_distributed:
            raise TypeError(
                "SegmentEntitySelection is distributed — use .filter_frame() "
                "instead of iterating (collecting would OOM the driver).",
            )
        return iter(self._data)

    def __eq__(self, other: Any) -> bool:
        if self.is_distributed:
            return NotImplemented
        if isinstance(other, SegmentEntitySelection):
            if other.is_distributed:
                return NotImplemented
            return set(self._data) == set(other._data)
        if isinstance(other, (set, frozenset)):
            return set(self._data) == set(other)
        return NotImplemented

    def __hash__(self):  # unhashable — sets are mutable
        raise TypeError("SegmentEntitySelection is unhashable")

    def intersect(self, other: "SegmentEntitySelection") -> "SegmentEntitySelection":
        if not self.is_distributed and not other.is_distributed:
            return SegmentEntitySelection(set(self._data) & set(other._data))
        joined = self._as_spark_df().join(other._as_spark_df(), on=_SEGMENT_ID_COL, how="inner")
        return SegmentEntitySelection(joined.cache())

    def filter_frame(self, df: Any, entity_col: str) -> Any:
        """Return df restricted to rows whose entity_col value is in this selection."""
        if not self.is_distributed:
            return safe_isin(df, entity_col, self._data)
        from customer_retention.core.compat.spark_backend import _as_pandas_api
        spark_df = as_spark_df(df)
        semi_keys = self._data if _SEGMENT_ID_COL == entity_col \
            else self._data.withColumnRenamed(_SEGMENT_ID_COL, entity_col)
        return _as_pandas_api(spark_df.join(semi_keys, on=entity_col, how="left_semi"))

    def _as_spark_df(self) -> Any:
        if self.is_distributed:
            return self._data
        from customer_retention.core.compat.detection import get_spark_session
        spark = get_spark_session()
        if spark is None:
            raise RuntimeError("No active Spark session available for distributed intersection")
        return spark.createDataFrame([(v,) for v in self._data], [_SEGMENT_ID_COL])


def _spark_passing_entities(df: Any, query_expr: str, entity_col: str) -> Any:
    """Distributed: one-column Spark DataFrame of entities where every row passes."""
    from pyspark.sql import functions as F  # noqa: N812

    from customer_retention.core.compat import _spark_safe_query_expr

    spark_df = as_spark_df(df)
    spark_expr = _spark_safe_query_expr(query_expr)
    return (
        spark_df
        .withColumn("_m", F.when(F.expr(spark_expr), F.lit(1)).otherwise(F.lit(0)))
        .groupBy(entity_col)
        .agg(F.count("*").alias("_total"), F.sum("_m").alias("_matching"))
        .filter(F.col("_total") == F.col("_matching"))
        .select(entity_col)
    )


def _pandas_passing_entities(df: Any, query_expr: str, entity_col: str) -> set:
    """Pandas path: inner join avoids fillna type issues."""
    pre_counts = df.groupby(entity_col).size().rename("_pre").reset_index()
    post_counts = (
        safe_query(df, query_expr).groupby(entity_col).size().rename("_post").reset_index()
    )
    merged = pre_counts.merge(post_counts, on=entity_col, how="inner")
    passing = merged[merged["_pre"] == merged["_post"]]
    return set(passing[entity_col].to_numpy())


def _expose_frames_as_views(frames: dict[str, Any]) -> None:
    # Filter expressions may contain SQL subqueries that reference sibling
    # datasets by name (e.g. `ACCOUNT_ID in (select ACCOUNT_ID from contract
    # where CONTRACT_START_DATE is not null)`). Spark's SQL parser resolves
    # unqualified identifiers through the session catalog — so we register
    # every Spark-backed frame as a session-scoped temp view keyed by its
    # dataset name. The views live on the lazy plans returned from here; the
    # Spark session drops them on termination, no cleanup required.
    for name, frame in frames.items():
        if not _is_spark_pandas(frame):
            continue
        as_spark_df(frame).createOrReplaceTempView(name)


def resolve_segment_entity_ids(
    frames: dict[str, pd.DataFrame],
    filters: Optional[dict[str, str]],
    entity_columns: dict[str, str],
) -> Optional[SegmentEntitySelection]:
    if not filters:
        return None
    _expose_frames_as_views(frames)
    selections: list[SegmentEntitySelection] = []
    for dataset_name, query_expr in filters.items():
        if dataset_name not in frames:
            continue
        df = frames[dataset_name]
        entity_col = entity_columns[dataset_name]
        if _is_spark_pandas(df):
            selections.append(SegmentEntitySelection.from_spark_df(
                _spark_passing_entities(df, query_expr, entity_col), entity_col,
            ))
        else:
            selections.append(SegmentEntitySelection.from_set(
                _pandas_passing_entities(df, query_expr, entity_col),
            ))
    if not selections:
        return None
    result = selections[0]
    for s in selections[1:]:
        result = result.intersect(s)
    return result


def apply_sample_filters(
    df: pd.DataFrame,
    dataset_name: str,
    filters: Optional[dict[str, str]],
) -> pd.DataFrame:
    if not filters or dataset_name not in filters:
        return df
    return safe_query(df, filters[dataset_name])


def estimate_sampling_accuracy(
    total_entities: int,
    target_rate: float,
    sample_sizes: list[int],
    n_cohorts: int = 1,
) -> list[dict]:
    results = []
    p = max(0.0, min(1.0, target_rate))
    for n in sample_sizes:
        n = min(n, total_entities)
        if n <= 0:
            continue
        ci = 1.96 * math.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        corr_err = 1 / math.sqrt(n) if n > 0 else 0.0
        per_cohort = n / max(1, n_cohorts)
        minority_expected = n * min(p, 1 - p)
        results.append({
            "sample_size": n,
            "pct_of_total": n / total_entities if total_entities > 0 else 1.0,
            "churn_rate_ci": ci,
            "correlation_error": corr_err,
            "entities_per_cohort": per_cohort,
            "cohort_ok": per_cohort >= 30,
            "minority_expected": minority_expected,
        })
    return results


def _compute_group_budget(
    group_counts: dict[str, int],
    n_remaining: int,
    total_remaining: int,
) -> dict[str, int]:
    budget: dict[str, int] = {}
    budget_left = n_remaining
    groups = list(group_counts.items())
    for i, (key, count) in enumerate(groups):
        if i == len(groups) - 1:
            n_take = budget_left
        else:
            n_take = max(1, round(n_remaining * count / total_remaining))
            n_take = min(n_take, budget_left)
        n_take = min(n_take, count)
        if n_take > 0:
            budget[key] = n_take
            budget_left -= n_take
    return budget


def _spark_windowed_sample(
    remaining_df: Any,
    entity_col: str,
    group_budget: dict[str, int],
    n_remaining: int,
    random_state: int,
) -> list:
    from pyspark.sql import Window, functions

    spark_df = as_spark_df(remaining_df)

    map_args: list = []
    for k, v in group_budget.items():
        map_args.extend([functions.lit(k), functions.lit(v)])
    budget_map = functions.create_map(*map_args)

    w = Window.partitionBy("_strat_key").orderBy(functions.rand(seed=random_state))

    result = (
        spark_df
        .withColumn("_budget", budget_map[functions.col("_strat_key")])
        .withColumn("_row_num", functions.row_number().over(w))
        .filter(functions.col("_row_num") <= functions.col("_budget"))
        .select(entity_col)
        .limit(n_remaining)
    )

    return [row[0] for row in result.collect()]


def _pandas_group_sample(
    remaining_df: Any,
    entity_col: str,
    group_budget: dict[str, int],
    n_remaining: int,
    random_state: int,
) -> list:
    sampled_parts = []
    for key, n_take in group_budget.items():
        if n_take <= 0:
            continue
        group_df = remaining_df[remaining_df["_strat_key"] == key]
        sampled_parts.append(safe_sample(group_df, n_take, random_state=random_state))

    if sampled_parts:
        sampled_df = concat(sampled_parts)
        return head_as_list(sampled_df[entity_col], n_remaining)
    return []


def stratified_entity_sample(
    entity_df: pd.DataFrame,
    n_entities: int,
    entity_col: str,
    target_col: Optional[str] = None,
    time_col: Optional[str] = None,
    extra_strat_cols: Optional[list[str]] = None,
    min_rare_count: int = 5,
    random_state: int = 42,
) -> list:
    if n_entities <= 0:
        return []

    deduped = entity_df.drop_duplicates(subset=[entity_col])
    total = len(deduped)
    if n_entities >= total:
        return head_as_list(deduped[entity_col], total)

    strat_parts = []

    if target_col and target_col in deduped.columns:
        strat_parts.append(deduped[target_col].astype(str))

    if time_col and time_col in deduped.columns:
        ts = safe_to_datetime(deduped[time_col], errors="coerce")
        cohort_key = ts.dt.year.astype(str) + "-Q" + ts.dt.quarter.astype(str)
        cohort_key = cohort_key.fillna("unknown")
        strat_parts.append(cohort_key)

    for col in (extra_strat_cols or []):
        if col not in deduped.columns:
            continue
        series = deduped[col]
        if is_numeric_dtype(series):
            try:
                binned = qcut(series, q=4, labels=False, duplicates="drop").astype(str)
            except (ValueError, TypeError):
                binned = series.astype(str)
            strat_parts.append(binned)
        else:
            strat_parts.append(series.astype(str))

    if strat_parts:
        strat_key = strat_parts[0]
        for part in strat_parts[1:]:
            strat_key = strat_key + "|" + part
        deduped = deduped.copy()
        deduped["_strat_key"] = strat_key
    else:
        deduped = deduped.copy()
        deduped["_strat_key"] = "all"

    rare_ids = []
    if target_col and target_col in deduped.columns:
        class_counts = deduped[target_col].value_counts().to_dict()
        for cls_val, cnt in class_counts.items():
            if cnt <= min_rare_count:
                rare_mask = deduped[target_col] == cls_val
                rare_ids.extend(head_as_list(deduped.loc[rare_mask, entity_col], cnt))

    remaining_df = safe_isin(deduped, entity_col, rare_ids, negate=True) if rare_ids else deduped
    n_remaining = n_entities - len(rare_ids)

    if n_remaining <= 0:
        return rare_ids[:n_entities]

    group_counts = remaining_df["_strat_key"].value_counts().to_dict()
    total_remaining = sum(group_counts.values())

    group_budget = _compute_group_budget(group_counts, n_remaining, total_remaining)

    if _is_spark_pandas(remaining_df):
        sampled_ids = _spark_windowed_sample(
            remaining_df, entity_col, group_budget, n_remaining, random_state,
        )
    else:
        sampled_ids = _pandas_group_sample(
            remaining_df, entity_col, group_budget, n_remaining, random_state,
        )

    return rare_ids + sampled_ids


def stratified_holdout_split(
    entity_df: pd.DataFrame,
    entity_ids: list,
    holdout_fraction: float,
    entity_col: str,
    target_col: Optional[str] = None,
    time_col: Optional[str] = None,
    extra_strat_cols: Optional[list[str]] = None,
    random_state: int = 42,
) -> tuple[list, list]:
    """Split pre-sampled entity IDs into train/holdout preserving strata proportions.

    Returns (train_ids, holdout_ids).
    """
    if holdout_fraction <= 0.0:
        return list(entity_ids), []
    if holdout_fraction >= 1.0:
        return [], list(entity_ids)

    id_set = set(entity_ids)
    deduped = entity_df.drop_duplicates(subset=[entity_col])
    deduped = deduped[deduped[entity_col].isin(id_set)].copy()

    n_holdout = max(1, int(len(id_set) * holdout_fraction))

    # Build strata keys — same logic as stratified_entity_sample
    strat_parts: list = []

    if target_col and target_col in deduped.columns:
        strat_parts.append(deduped[target_col].astype(str))

    if time_col and time_col in deduped.columns:
        ts = safe_to_datetime(deduped[time_col], errors="coerce")
        cohort_key = ts.dt.year.astype(str) + "-Q" + ts.dt.quarter.astype(str)
        cohort_key = cohort_key.fillna("unknown")
        strat_parts.append(cohort_key)

    for col in (extra_strat_cols or []):
        if col not in deduped.columns:
            continue
        series = deduped[col]
        if is_numeric_dtype(series):
            try:
                binned = qcut(series, q=4, labels=False, duplicates="drop").astype(str)
            except (ValueError, TypeError):
                binned = series.astype(str)
            strat_parts.append(binned)
        else:
            strat_parts.append(series.astype(str))

    if strat_parts:
        strat_key = strat_parts[0]
        for part in strat_parts[1:]:
            strat_key = strat_key + "|" + part
        deduped["_strat_key"] = strat_key
    else:
        deduped["_strat_key"] = "all"

    # Proportional holdout per stratum
    group_counts = deduped["_strat_key"].value_counts().to_dict()
    total = sum(group_counts.values())

    holdout_budget = _compute_group_budget(group_counts, n_holdout, total)

    # Sample holdout from each stratum using a shifted random state
    holdout_ids_list: list = []
    for key, n_take in holdout_budget.items():
        if n_take <= 0:
            continue
        group_df = deduped[deduped["_strat_key"] == key]
        sampled = safe_sample(group_df, n_take, random_state=random_state + 1)
        holdout_ids_list.extend(head_as_list(sampled[entity_col], n_take))

    holdout_set = set(holdout_ids_list)
    train_ids = [eid for eid in entity_ids if eid not in holdout_set]

    return train_ids, holdout_ids_list
