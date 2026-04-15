"""Parity tests: pre-optimization stratified sample/holdout vs the refactored code.

The *_original_* functions in this file are copies of the sampling.py
implementation from commit 4e75304 (before the shared-strat-key refactor and
Spark-windowed holdout split).  They are deliberately inlined so the test
survives future framework refactors and keeps proving the optimized path
produces the same behavior.

- Pandas inputs: exact ID parity is expected (same algorithm on both paths).
- Spark inputs: the optimized holdout uses a single windowed pass, which uses
  a different RNG than the per-stratum ``safe_sample`` loop — only structural
  invariants (stratum sizes, disjointness, subset) can match exactly.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.sampling import (
    _compute_group_budget,
    stratified_entity_sample,
    stratified_holdout_split,
)
from customer_retention.core.compat import (
    _is_spark_pandas,
    concat,
    head_as_list,
    is_numeric_dtype,
    qcut,
    safe_isin,
    safe_sample,
    safe_to_datetime,
)


def _original_stratified_entity_sample(
    entity_df: Any,
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
        sampled_ids = _original_spark_windowed_sample(
            remaining_df, entity_col, group_budget, n_remaining, random_state,
        )
    else:
        sampled_ids = _original_pandas_group_sample(
            remaining_df, entity_col, group_budget, n_remaining, random_state,
        )
    return rare_ids + sampled_ids


def _original_spark_windowed_sample(
    remaining_df, entity_col, group_budget, n_remaining, random_state,
):
    from pyspark.sql import Window, functions

    from customer_retention.core.compat import as_spark_df
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


def _original_pandas_group_sample(remaining_df, entity_col, group_budget, n_remaining, random_state):
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


def _original_stratified_holdout_split(
    entity_df: Any,
    entity_ids: list,
    holdout_fraction: float,
    entity_col: str,
    target_col: Optional[str] = None,
    time_col: Optional[str] = None,
    extra_strat_cols: Optional[list[str]] = None,
    random_state: int = 42,
) -> tuple[list, list]:
    if holdout_fraction <= 0.0:
        return list(entity_ids), []
    if holdout_fraction >= 1.0:
        return [], list(entity_ids)

    id_set = set(entity_ids)
    deduped = entity_df.drop_duplicates(subset=[entity_col])
    deduped = deduped[deduped[entity_col].isin(id_set)].copy()
    n_holdout = max(1, int(len(id_set) * holdout_fraction))

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

    group_counts = deduped["_strat_key"].value_counts().to_dict()
    total = sum(group_counts.values())
    holdout_budget = _compute_group_budget(group_counts, n_holdout, total)

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


def _make_entity_df(n: int = 300) -> pd.DataFrame:
    return pd.DataFrame({
        "entity_id": list(range(n)),
        "target": [1] * (n // 5) + [0] * (n - n // 5),
        "ts": pd.date_range("2020-01-01", periods=n, freq="D"),
        "region": (["east", "west", "north", "south"] * ((n // 4) + 1))[:n],
    })


class TestPandasParityStratifiedEntitySample:
    """On pandas inputs, optimized == original (same algorithm, same seed)."""

    def test_target_only(self):
        df = _make_entity_df(200)
        old = _original_stratified_entity_sample(df, 50, "entity_id", "target", random_state=42)
        new = stratified_entity_sample(df, 50, "entity_id", "target", random_state=42)
        assert new == old

    def test_target_and_time(self):
        df = _make_entity_df(300)
        old = _original_stratified_entity_sample(
            df, 80, "entity_id", "target", time_col="ts", random_state=7,
        )
        new = stratified_entity_sample(
            df, 80, "entity_id", "target", time_col="ts", random_state=7,
        )
        assert new == old

    def test_target_time_and_extras(self):
        df = _make_entity_df(300)
        old = _original_stratified_entity_sample(
            df, 100, "entity_id", "target",
            time_col="ts", extra_strat_cols=["region"], random_state=42,
        )
        new = stratified_entity_sample(
            df, 100, "entity_id", "target",
            time_col="ts", extra_strat_cols=["region"], random_state=42,
        )
        assert new == old

    def test_rare_class_preserved_same_order(self):
        df = pd.DataFrame({
            "entity_id": list(range(100)),
            "target": [1] * 3 + [0] * 97,
        })
        old = _original_stratified_entity_sample(df, 20, "entity_id", "target", min_rare_count=5)
        new = stratified_entity_sample(df, 20, "entity_id", "target", min_rare_count=5)
        assert new == old

    def test_no_target_column(self):
        df = _make_entity_df(200)
        old = _original_stratified_entity_sample(df, 40, "entity_id", random_state=3)
        new = stratified_entity_sample(df, 40, "entity_id", random_state=3)
        assert new == old

    def test_n_exceeds_total_returns_full(self):
        df = _make_entity_df(30)
        old = _original_stratified_entity_sample(df, 999, "entity_id", "target")
        new = stratified_entity_sample(df, 999, "entity_id", "target")
        assert new == old


class TestPandasParityStratifiedHoldoutSplit:
    """On pandas inputs, optimized == original."""

    def test_target_only(self):
        df = _make_entity_df(200)
        ids = list(range(200))
        old_t, old_h = _original_stratified_holdout_split(
            df, ids, holdout_fraction=0.2, entity_col="entity_id",
            target_col="target", random_state=42,
        )
        new_t, new_h = stratified_holdout_split(
            df, ids, holdout_fraction=0.2, entity_col="entity_id",
            target_col="target", random_state=42,
        )
        assert new_t == old_t
        assert new_h == old_h

    def test_target_and_time(self):
        df = _make_entity_df(300)
        ids = list(range(300))
        old_t, old_h = _original_stratified_holdout_split(
            df, ids, holdout_fraction=0.15, entity_col="entity_id",
            target_col="target", time_col="ts", random_state=11,
        )
        new_t, new_h = stratified_holdout_split(
            df, ids, holdout_fraction=0.15, entity_col="entity_id",
            target_col="target", time_col="ts", random_state=11,
        )
        assert new_t == old_t
        assert new_h == old_h

    def test_target_time_extras(self):
        df = _make_entity_df(300)
        ids = list(range(300))
        old_t, old_h = _original_stratified_holdout_split(
            df, ids, holdout_fraction=0.25, entity_col="entity_id",
            target_col="target", time_col="ts", extra_strat_cols=["region"],
            random_state=42,
        )
        new_t, new_h = stratified_holdout_split(
            df, ids, holdout_fraction=0.25, entity_col="entity_id",
            target_col="target", time_col="ts", extra_strat_cols=["region"],
            random_state=42,
        )
        assert new_t == old_t
        assert new_h == old_h

    def test_edge_zero_fraction(self):
        df = _make_entity_df(50)
        ids = list(range(50))
        old_t, old_h = _original_stratified_holdout_split(
            df, ids, 0.0, "entity_id", target_col="target",
        )
        new_t, new_h = stratified_holdout_split(
            df, ids, 0.0, "entity_id", target_col="target",
        )
        assert new_t == old_t and new_h == old_h

    def test_edge_full_fraction(self):
        df = _make_entity_df(50)
        ids = list(range(50))
        old_t, old_h = _original_stratified_holdout_split(
            df, ids, 1.0, "entity_id", target_col="target",
        )
        new_t, new_h = stratified_holdout_split(
            df, ids, 1.0, "entity_id", target_col="target",
        )
        assert new_t == old_t and new_h == old_h


@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("sampling_parity")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )


def _as_psdf(spark, pdf):
    from customer_retention.core.compat.spark_backend import _as_pandas_api
    return _as_pandas_api(spark.createDataFrame(pdf))


@pytest.mark.spark
class TestSparkParityStratifiedEntitySample:
    """Spark path already used windowed sampling in the original; parity must be exact."""

    def test_target_and_time(self, spark):
        pdf = _make_entity_df(300)
        psdf = _as_psdf(spark, pdf)
        old = _original_stratified_entity_sample(
            psdf, 80, "entity_id", "target", time_col="ts", random_state=7,
        )
        new = stratified_entity_sample(
            _as_psdf(spark, pdf), 80, "entity_id", "target", time_col="ts", random_state=7,
        )
        assert sorted(old) == sorted(new)

    def test_target_time_and_extras(self, spark):
        pdf = _make_entity_df(300)
        old = _original_stratified_entity_sample(
            _as_psdf(spark, pdf), 100, "entity_id", "target",
            time_col="ts", extra_strat_cols=["region"], random_state=42,
        )
        new = stratified_entity_sample(
            _as_psdf(spark, pdf), 100, "entity_id", "target",
            time_col="ts", extra_strat_cols=["region"], random_state=42,
        )
        assert sorted(old) == sorted(new)


@pytest.mark.spark
class TestSparkParityStratifiedHoldoutSplit:
    """The optimized holdout replaces per-stratum loops with a single windowed pass.

    Per-stratum ``safe_sample`` has a different RNG than ``Window+rand``, so
    exact-ID parity is not possible.  Structural invariants (stratum sizes,
    disjointness, subset) must match exactly.
    """

    @staticmethod
    def _stratum_counts(ids, pdf, strat_cols):
        sub = pdf[pdf["entity_id"].isin(ids)].copy()
        sub["_k"] = ""
        for c in strat_cols:
            sub["_k"] = sub["_k"] + "|" + sub[c].astype(str)
        return sub["_k"].value_counts().to_dict()

    @staticmethod
    def _basic_invariants(train, holdout, all_ids):
        assert set(train).isdisjoint(set(holdout))
        assert set(train + holdout) == set(all_ids)
        assert len(set(holdout)) == len(holdout)
        assert len(set(train)) == len(train)

    def test_stratum_sizes_match_on_target_only(self, spark):
        pdf = _make_entity_df(300)
        all_ids = list(range(300))
        # Budget is derived from group counts — both paths see the same budget.
        old_train, old_holdout = _original_stratified_holdout_split(
            _as_psdf(spark, pdf), all_ids, 0.2, "entity_id", "target", random_state=42,
        )
        new_train, new_holdout = stratified_holdout_split(
            _as_psdf(spark, pdf), all_ids, 0.2, "entity_id", "target", random_state=42,
        )
        self._basic_invariants(old_train, old_holdout, all_ids)
        self._basic_invariants(new_train, new_holdout, all_ids)
        assert len(new_holdout) == len(old_holdout)
        old_counts = self._stratum_counts(old_holdout, pdf, ["target"])
        new_counts = self._stratum_counts(new_holdout, pdf, ["target"])
        assert old_counts == new_counts

    def test_stratum_sizes_match_on_target_and_time(self, spark):
        pdf = _make_entity_df(300)
        all_ids = list(range(300))
        old_train, old_holdout = _original_stratified_holdout_split(
            _as_psdf(spark, pdf), all_ids, 0.15, "entity_id",
            target_col="target", time_col="ts", random_state=11,
        )
        new_train, new_holdout = stratified_holdout_split(
            _as_psdf(spark, pdf), all_ids, 0.15, "entity_id",
            target_col="target", time_col="ts", random_state=11,
        )
        self._basic_invariants(old_train, old_holdout, all_ids)
        self._basic_invariants(new_train, new_holdout, all_ids)
        assert len(new_holdout) == len(old_holdout)
        # Cohort key: year+quarter combined with target
        pdf2 = pdf.copy()
        pdf2["_cohort"] = (
            pd.to_datetime(pdf2["ts"]).dt.year.astype(str)
            + "Q" + pd.to_datetime(pdf2["ts"]).dt.quarter.astype(str)
        )
        old_counts = self._stratum_counts(old_holdout, pdf2, ["target", "_cohort"])
        new_counts = self._stratum_counts(new_holdout, pdf2, ["target", "_cohort"])
        assert old_counts == new_counts

    def test_new_path_issues_single_spark_job_per_holdout(self, spark, monkeypatch):
        """The optimization: single windowed pass replaces ~N-stratum loop."""
        pdf = _make_entity_df(300)
        all_ids = list(range(300))

        from pyspark.sql import DataFrame

        call_counter = {"n": 0}
        real_collect = DataFrame.collect

        def counting_collect(self):
            call_counter["n"] += 1
            return real_collect(self)

        monkeypatch.setattr(DataFrame, "collect", counting_collect)
        stratified_holdout_split(
            _as_psdf(spark, pdf), all_ids, 0.2, "entity_id",
            target_col="target", time_col="ts", random_state=42,
        )
        # Window+row_number emits exactly one collect() for holdout IDs.
        # Value-counts/count/etc use head/take, not collect.  The original
        # looped with N strata — typically >= 4 collects for this fixture.
        assert call_counter["n"] == 1, f"Expected 1 collect, got {call_counter['n']}"
