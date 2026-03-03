from __future__ import annotations

import math
from typing import Any, Optional

from customer_retention.core.compat import (
    _is_spark_pandas,
    as_spark_df,
    concat,
    head_as_list,
    pd,
    safe_sample,
    safe_to_datetime,
)


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
        if pd.api.types.is_numeric_dtype(series):
            try:
                binned = pd.qcut(series, q=4, labels=False, duplicates="drop").astype(str)
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

    remaining_df = deduped[~deduped[entity_col].isin(rare_ids)] if rare_ids else deduped
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
