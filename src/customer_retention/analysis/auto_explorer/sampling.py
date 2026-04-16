from __future__ import annotations

import json
import math
from dataclasses import dataclass
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
    safe_len,
    safe_query,
    safe_sample,
    safe_to_datetime,
    spark_persist,
)

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.project_context import IntentConfig

_SEGMENT_ID_COL = "_segment_entity_id"
_STRAT_KEY_COL = "_strat_key"
_CANDIDATE_SAMPLE_SIZES: tuple[int, ...] = (500, 1000, 2000, 5000, 10000)

_LOOKBACK_MIN_INPUT_ROWS = 1000
_LOOKBACK_MIN_RETAINED_ROWS = 100
_LOOKBACK_MIN_RETENTION_RATIO = 0.001


def apply_temporal_lookback(df: Any, time_col: str, intent: IntentConfig) -> Any:
    if intent.lookback_periods is None:
        return df
    from customer_retention.analysis.auto_explorer.snapshot_grid import CADENCE_DAYS

    ts = safe_to_datetime(df[time_col], errors="coerce")
    upper = ts.max()
    if native_pd.isna(upper):
        return df
    cap = native_pd.Timestamp(intent.history_upper_limit) if intent.history_upper_limit else None
    upper_capped = cap if cap is not None and cap < upper else upper
    lookback_days = intent.lookback_periods * CADENCE_DAYS[intent.cadence_interval]
    lower = upper_capped - timedelta(days=lookback_days)
    mask = ts >= lower
    if cap is not None:
        mask = mask & (ts <= upper_capped)
    filtered = df[mask]
    _assert_lookback_retention(df, filtered, time_col, intent, lookback_days, lower, upper_capped)
    return filtered


def _assert_lookback_retention(
    pre_df: Any,
    post_df: Any,
    time_col: str,
    intent: IntentConfig,
    lookback_days: int,
    lower: Any,
    upper: Any,
) -> None:
    """Fail fast when the lookback filter destroys almost all data.

    Triggered only on non-trivial inputs (>= 1000 rows) that collapse to
    < 100 rows AND retain < 0.1%. Interval-type datasets (contracts,
    subscriptions) that skip the lookback via raw_time_column_role don't
    reach this code path — this guard catches misconfiguration where a
    lookback is applied to data whose timestamps predate the window.
    """
    pre_count = safe_len(pre_df)
    if pre_count < _LOOKBACK_MIN_INPUT_ROWS:
        return
    post_count = safe_len(post_df)
    if post_count >= _LOOKBACK_MIN_RETAINED_ROWS:
        return
    if pre_count > 0 and post_count / pre_count >= _LOOKBACK_MIN_RETENTION_RATIO:
        return
    lower_str = lower.isoformat() if hasattr(lower, "isoformat") else str(lower)
    upper_str = upper.isoformat() if hasattr(upper, "isoformat") else str(upper)
    raise ValueError(
        f"Temporal lookback filter on '{time_col}' retained only {post_count:,} "
        f"of {pre_count:,} rows. Window: [{lower_str}, {upper_str}] "
        f"({lookback_days} days = {intent.lookback_periods} x "
        f"{intent.cadence_interval.value}). "
        f"Resolve by one of: "
        f"(1) for interval-type datasets (contracts/subscriptions whose "
        f"events span START->END), set "
        f"raw_time_column_role=RawTimeColumnRole.INTERVAL_START_TIME in NB00 "
        f"semantics_overrides to skip the lookback; "
        f"(2) increase intent.lookback_periods (or set it to None) so the "
        f"window covers more history; "
        f"(3) verify that '{time_col}' is populated — a mostly-NULL column "
        f"will silently collapse under the filter."
    )


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

    def __hash__(self):
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
    for name, frame in frames.items():
        if not _is_spark_pandas(frame):
            continue
        as_spark_df(frame).createOrReplaceTempView(name)


@dataclass
class SegmentFilterStats:
    """Per-filter input/output counts captured during segment resolution."""

    dataset: str
    filter_expr: str
    input_entities: int
    output_entities: int

    @property
    def pass_rate(self) -> float:
        return self.output_entities / self.input_entities if self.input_entities else 0.0

    def is_suspicious(self, threshold: float = 0.05) -> bool:
        """True when the filter removed more than ``1 - threshold`` of entities.

        A very low pass rate does not necessarily mean the filter is wrong —
        e.g. SPS's "accounts with at least one contract" legitimately keeps
        only customers from a prospect-heavy account table. But visibility
        at sample time lets the user confirm the intent.
        """
        return 0.0 < self.pass_rate < threshold


def resolve_segment_entity_ids(
    frames: dict[str, pd.DataFrame],
    filters: Optional[dict[str, str]],
    entity_columns: dict[str, str],
    *,
    diagnostics: Optional[list] = None,
) -> Optional[SegmentEntitySelection]:
    """Intersect per-dataset filters into a single entity-ID selection.

    When ``diagnostics`` is a list, a :class:`SegmentFilterStats` is appended
    for each applied filter recording the input + output entity counts. Keeps
    fail-fast visibility at sample time without changing the return type.
    """
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
            selection = SegmentEntitySelection.from_spark_df(
                _spark_passing_entities(df, query_expr, entity_col), entity_col,
            )
        else:
            selection = SegmentEntitySelection.from_set(
                _pandas_passing_entities(df, query_expr, entity_col),
            )
        selections.append(selection)
        if diagnostics is not None:
            input_count = int(safe_len(df[entity_col].drop_duplicates())) if entity_col in df.columns else 0
            diagnostics.append(SegmentFilterStats(
                dataset=dataset_name, filter_expr=query_expr,
                input_entities=input_count, output_entities=len(selection),
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


def _binned_series(series: Any) -> Any:
    if not is_numeric_dtype(series):
        return series.astype(str)
    try:
        return qcut(series, q=4, labels=False, duplicates="drop").astype(str)
    except (ValueError, TypeError):
        return series.astype(str)


def _build_strat_key_column(
    df: Any,
    target_col: Optional[str],
    time_col: Optional[str],
    extra_strat_cols: Optional[list[str]],
) -> Any:
    """Compose a single "|"-joined string stratification key from target, quarter, and extras.

    Returns None when no stratification dimension is requested.
    """
    parts: list = []
    if target_col and target_col in df.columns:
        parts.append(df[target_col].astype(str))
    if time_col and time_col in df.columns:
        ts = safe_to_datetime(df[time_col], errors="coerce")
        cohort = ts.dt.year.astype(str) + "-Q" + ts.dt.quarter.astype(str)
        parts.append(cohort.fillna("unknown"))
    for col in (extra_strat_cols or []):
        if col in df.columns:
            parts.append(_binned_series(df[col]))
    if not parts:
        return None
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key


def _attach_strat_key(df: Any, strat_key: Any) -> Any:
    stratified = df.copy()
    stratified[_STRAT_KEY_COL] = strat_key if strat_key is not None else "all"
    return stratified


def _spark_windowed_take(
    df: Any,
    entity_col: str,
    group_budget: dict[str, int],
    total_take: int,
    random_state: int,
) -> list:
    """Draw per-stratum samples in a SINGLE distributed pass via window row_number."""
    from pyspark.sql import Window, functions

    spark_df = as_spark_df(df)
    map_args: list = []
    for k, v in group_budget.items():
        map_args.extend([functions.lit(k), functions.lit(v)])
    budget_map = functions.create_map(*map_args)
    w = Window.partitionBy(_STRAT_KEY_COL).orderBy(functions.rand(seed=random_state))
    result = (
        spark_df
        .withColumn("_budget", budget_map[functions.col(_STRAT_KEY_COL)])
        .withColumn("_row_num", functions.row_number().over(w))
        .filter(functions.col("_row_num") <= functions.col("_budget"))
        .select(entity_col)
        .limit(total_take)
    )
    return [row[0] for row in result.collect()]


def _pandas_windowed_take(
    df: Any,
    entity_col: str,
    group_budget: dict[str, int],
    total_take: int,
    random_state: int,
) -> list:
    parts = []
    for key, n_take in group_budget.items():
        if n_take <= 0:
            continue
        group_df = df[df[_STRAT_KEY_COL] == key]
        parts.append(safe_sample(group_df, n_take, random_state=random_state))
    if not parts:
        return []
    combined = concat(parts)
    return head_as_list(combined[entity_col], total_take)


def _stratified_ids_by_budget(
    df: Any,
    entity_col: str,
    group_budget: dict[str, int],
    total_take: int,
    random_state: int,
) -> list:
    if _is_spark_pandas(df):
        return _spark_windowed_take(df, entity_col, group_budget, total_take, random_state)
    return _pandas_windowed_take(df, entity_col, group_budget, total_take, random_state)


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
    total = safe_len(deduped)
    if n_entities >= total:
        return head_as_list(deduped[entity_col], total)

    strat_key = _build_strat_key_column(deduped, target_col, time_col, extra_strat_cols)
    deduped = _attach_strat_key(deduped, strat_key)

    rare_ids: list = []
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

    group_counts = remaining_df[_STRAT_KEY_COL].value_counts().to_dict()
    total_remaining = sum(group_counts.values())
    group_budget = _compute_group_budget(group_counts, n_remaining, total_remaining)

    sampled_ids = _stratified_ids_by_budget(
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
    """Split pre-sampled entity IDs into (train, holdout), preserving strata proportions.

    Distributed path uses a single windowed Spark pass; pandas path loops per stratum.
    """
    if holdout_fraction <= 0.0:
        return list(entity_ids), []
    if holdout_fraction >= 1.0:
        return [], list(entity_ids)

    id_set = set(entity_ids)
    deduped = entity_df.drop_duplicates(subset=[entity_col])
    deduped = deduped[deduped[entity_col].isin(id_set)]
    n_holdout = max(1, int(len(id_set) * holdout_fraction))

    strat_key = _build_strat_key_column(deduped, target_col, time_col, extra_strat_cols)
    deduped = _attach_strat_key(deduped, strat_key)

    group_counts = deduped[_STRAT_KEY_COL].value_counts().to_dict()
    total = sum(group_counts.values())
    holdout_budget = _compute_group_budget(group_counts, n_holdout, total)

    holdout_ids_list = _stratified_ids_by_budget(
        deduped, entity_col, holdout_budget, n_holdout, random_state + 1,
    )
    holdout_set = set(holdout_ids_list)
    train_ids = [eid for eid in entity_ids if eid not in holdout_set]
    return train_ids, holdout_ids_list


@dataclass(frozen=True)
class PopulationSummary:
    total_entities: int
    target_rate: float
    n_cohorts: int
    time_column: Optional[str]


def summarize_population(
    target_df: Any,
    entity_col: Optional[str],
    target_col: Optional[str],
    time_col: Optional[str],
) -> PopulationSummary:
    """Entity-level population stats: total, target rate, cohort count (1 pass each)."""
    if target_df is None or not entity_col or entity_col not in target_df.columns:
        return PopulationSummary(0, 0.5, 1, None)
    entity_level = target_df.drop_duplicates(subset=[entity_col])
    total = safe_len(entity_level)
    rate = 0.5
    if target_col and target_col in entity_level.columns:
        rate = float(entity_level[target_col].mean())
    n_cohorts = 1
    resolved_time = None
    if time_col and time_col in entity_level.columns:
        dates = safe_to_datetime(entity_level[time_col], errors="coerce").dropna()
        if safe_len(dates) > 0:
            n_cohorts = max(1, int((dates.dt.year * 4 + dates.dt.quarter).nunique()))
            resolved_time = time_col
    return PopulationSummary(total, rate, n_cohorts, resolved_time)


def candidate_sample_sizes(total_entities: int) -> list[int]:
    return sorted({s for s in (*_CANDIDATE_SAMPLE_SIZES, total_entities) if 0 < s <= total_entities})


def _sample_column_set(
    target_df: Any,
    entity_col: str,
    target_col: Optional[str],
    time_col: Optional[str],
    strat_cols: Optional[list[str]],
) -> list[str]:
    cols = [entity_col]
    for c in (target_col, time_col):
        if c and c in target_df.columns and c not in cols:
            cols.append(c)
    for c in (strat_cols or []):
        if c in target_df.columns and c not in cols:
            cols.append(c)
    return cols


def prepare_sample_frame(
    target_df: Any,
    entity_col: str,
    target_col: Optional[str],
    time_col: Optional[str],
    strat_cols: Optional[list[str]],
    segment_ids: Optional[SegmentEntitySelection],
) -> Any:
    """Project to sampling columns, apply segment filter, persist for reuse across phases."""
    cols = _sample_column_set(target_df, entity_col, target_col, time_col, strat_cols)
    frame = target_df[cols]
    if segment_ids is not None:
        frame = segment_ids.filter_frame(frame, entity_col)
    return spark_persist(frame)


def _format_accuracy_row(e: dict) -> str:
    return (
        f"| {e['sample_size']:,} | {e['pct_of_total']:.0%} "
        f"| +/-{e['churn_rate_ci']:.3f} "
        f"| +/-{e['correlation_error']:.3f} "
        f"| {e['minority_expected']:,.0f} "
        f"| {'yes' if e['cohort_ok'] else 'no'} |"
    )


def render_population_markdown(
    summary: PopulationSummary,
    effective_total: int,
    was_filtered: bool,
    unfiltered_entities: int,
    estimates: list[dict],
) -> str:
    table = (
        "| Sample Size | % of Total | Churn Rate 95% CI | Correlation Error | Minority Class | Cohort Coverage |\n"
        "|---:|---:|---:|---:|---:|:---|\n"
        + "\n".join(_format_accuracy_row(e) for e in estimates)
    )
    pop_label = f"{effective_total:,} entities"
    if was_filtered:
        pop_label += f" (filtered from {unfiltered_entities:,})"
    return (
        f"**Population:** {pop_label}, target rate {summary.target_rate:.1%}, "
        f"{summary.n_cohorts} cohorts\n\n{table}"
    )


def render_sample_result_markdown(
    sampled_count: int,
    effective_total: int,
    train_count: int,
    holdout_count: int,
    holdout_fraction: float,
    has_time_col: bool,
    strat_cols: Optional[list[str]],
) -> str:
    strat_desc = "target"
    if has_time_col:
        strat_desc += " + cohort"
    if strat_cols:
        strat_desc += " + " + ", ".join(strat_cols)
    pct = sampled_count / effective_total if effective_total else 0.0
    return (
        f"**Sampled:** {sampled_count:,} / {effective_total:,} entities "
        f"({pct:.0%}), stratified by {strat_desc}\n\n"
        f"- **Train:** {train_count:,} entities\n"
        f"- **Holdout:** {holdout_count:,} entities ({holdout_fraction:.0%})"
    )


def render_segment_filter_markdown(
    filters: dict[str, str],
    diagnostics: Optional[list] = None,
    *,
    suspicious_pass_rate_threshold: float = 0.05,
) -> str:
    """Render the applied segment filters.

    When ``diagnostics`` is a list of :class:`SegmentFilterStats`, each filter
    line includes its input/output entity counts, pass rate, and a ⚠️ flag
    when the rate falls below ``suspicious_pass_rate_threshold``. Without
    diagnostics the legacy expression-only rendering is preserved.
    """
    if not diagnostics:
        lines = [f"  - **{k}**: `{v}`" for k, v in filters.items()]
        return "**Segment filters:**\n" + "\n".join(lines)

    lines = []
    for stat in diagnostics:
        flag = " ⚠️" if stat.is_suspicious(suspicious_pass_rate_threshold) else ""
        lines.append(
            f"  - **{stat.dataset}**{flag}: `{stat.filter_expr}` → "
            f"{stat.output_entities:,} / {stat.input_entities:,} entities "
            f"({stat.pass_rate:.1%} pass rate)"
        )
    return "**Segment filters:**\n" + "\n".join(lines)


def _assert_sampled_ids_have_labels(
    *, entity_df: Any, entity_col: str, target_col: str,
    ids: list, pool_label: str, max_examples: int = 10,
) -> None:
    """Fail fast if any sampled ID will produce a NULL target downstream.

    Guards against two failure modes that both manifest as NULL ``target_col``
    rows after the spine merge:

    1. **Missing ID** — sampled ID has no row in ``entity_df`` at all. Happens
       when the sampler reads a broader source than the one that will be
       materialized into the primary entity dataset.
    2. **Null target** — sampled ID has a row but its target value is NULL.
       Happens when the user-code label derivation runs after sampling but
       fails to cover every sampled entity.

    Either case creates orphan panel rows that ``drop_missing_target`` silently
    removes at training — wasting 30-60% of compute and making the training
    set size opaque. Failing at the sampler surfaces the misconfiguration
    where the user can still act on it.

    ``target_col`` missing from ``entity_df`` short-circuits the check —
    caller's responsibility to ensure the column is present when validation
    is intended.
    """
    if not ids or target_col not in entity_df.columns:
        return
    id_set = set(ids)
    matched = entity_df[entity_df[entity_col].isin(id_set)]
    matched_ids = set(head_as_list(matched[entity_col].unique(), len(id_set)))
    missing = [i for i in ids if i not in matched_ids]
    null_mask = matched[target_col].isna()
    null_ids: list = []
    if bool(null_mask.any()):
        null_ids = head_as_list(matched.loc[null_mask, entity_col].unique(), len(id_set))

    if not missing and not null_ids:
        return

    raise ValueError(
        f"Sampler {pool_label} pool: {len(missing) + len(null_ids)} of {len(ids)} IDs "
        f"will produce NULL {target_col!r} values when the pipeline materializes silver_merged.\n"
        f"  {len(missing)} absent from entity dataset, {len(null_ids)} with null target.\n"
        f"  First absent IDs:     {missing[:max_examples]}\n"
        f"  First null-target IDs: {null_ids[:max_examples]}\n"
        f"Fix upstream filters so every sampled entity has a resolvable label before "
        f"save_sample_ids() is called, or pre-filter the sampler input to entities that "
        f"will be present in the primary entity dataset."
    )


def save_sample_ids(
    namespace: Any, train_ids: list, holdout_ids: list,
    *,
    entity_df: Optional[Any] = None,
    entity_col: Optional[str] = None,
    target_col: Optional[str] = None,
) -> None:
    """Persist sampled train + holdout entity IDs.

    When ``entity_df`` + ``entity_col`` + ``target_col`` are supplied, every
    sampled ID is validated against the entity dataset via
    :func:`_assert_sampled_ids_have_labels` BEFORE either file is written. A
    validation failure raises ``ValueError`` and leaves both files unwritten,
    so callers never see a partial state.
    """
    if entity_df is not None and entity_col and target_col:
        _assert_sampled_ids_have_labels(
            entity_df=entity_df, entity_col=entity_col, target_col=target_col,
            ids=train_ids, pool_label="training",
        )
        _assert_sampled_ids_have_labels(
            entity_df=entity_df, entity_col=entity_col, target_col=target_col,
            ids=holdout_ids, pool_label="holdout",
        )
    train_path = namespace.sample_entity_ids_path
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(json.dumps(train_ids, default=str))
    if holdout_ids:
        namespace.holdout_entity_ids_path.write_text(json.dumps(holdout_ids, default=str))
