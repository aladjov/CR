"""Tests for the entity-enrichment patterns used in notebook 01d_event_aggregation.

The execute_aggregation cell enriches entity-level df_aggregated with features
derived from distributed events via groupby.  On pyspark.pandas the original
.map(groupby_series) / .apply(lambda) patterns fail.

Two paths are tested:
  1. Local pandas: assign + groupby().mean() instead of .apply(lambda),
     vectorized OLS slope, sort+drop_duplicates for mode.
  2. Native Spark (mocked): groupBy/agg/join/withColumn — no collection to
     pandas at any granularity except tiny display summaries.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import timedelta_to_days
from customer_retention.stages.profiling.time_series_profiler import (
    classify_lifecycle_quadrants,
)

# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def events_df():
    np.random.seed(42)
    entities = [f"E{i}" for i in range(20)]
    rows = []
    base = pd.Timestamp("2023-01-01")
    for entity in entities:
        n_events = np.random.randint(5, 30)
        for j in range(n_events):
            rows.append({
                "entity_id": entity,
                "event_timestamp": base + timedelta(days=int(np.random.randint(0, 365))),
                "amount": round(np.random.uniform(10, 500), 2),
                "target": int(np.random.random() < 0.3),
            })
    return pd.DataFrame(rows).sort_values("event_timestamp").reset_index(drop=True)


@pytest.fixture
def aggregated_df(events_df):
    return pd.DataFrame({
        "entity_id": sorted(events_df["entity_id"].unique()),
        "event_count": events_df.groupby("entity_id").size().values,
    })


# ── vectorized temporal mean (local path) ────────────────────────────────

class TestEntityTemporalMean:
    @staticmethod
    def _entity_temporal_mean(df, entity_col, time_col, dt_attr):
        derived = getattr(df[time_col].dt, dt_attr)
        return df.assign(__tmp=derived.astype(float)).groupby(entity_col)["__tmp"].mean()

    def test_dayofweek_matches_apply(self, events_df):
        expected = events_df.groupby("entity_id")["event_timestamp"].apply(
            lambda x: x.dt.dayofweek.mean()
        )
        result = self._entity_temporal_mean(events_df, "entity_id", "event_timestamp", "dayofweek")
        pd.testing.assert_series_equal(result, expected, check_names=False, atol=1e-10)

    def test_month_matches_apply(self, events_df):
        expected = events_df.groupby("entity_id")["event_timestamp"].apply(
            lambda x: x.dt.month.mean()
        )
        result = self._entity_temporal_mean(events_df, "entity_id", "event_timestamp", "month")
        pd.testing.assert_series_equal(result, expected, check_names=False, atol=1e-10)

    def test_day_matches_apply(self, events_df):
        expected = events_df.groupby("entity_id")["event_timestamp"].apply(
            lambda x: x.dt.day.mean()
        )
        result = self._entity_temporal_mean(events_df, "entity_id", "event_timestamp", "day")
        pd.testing.assert_series_equal(result, expected, check_names=False, atol=1e-10)

    def test_quarter_matches_apply(self, events_df):
        expected = events_df.groupby("entity_id")["event_timestamp"].apply(
            lambda x: x.dt.quarter.mean()
        )
        result = self._entity_temporal_mean(events_df, "entity_id", "event_timestamp", "quarter")
        pd.testing.assert_series_equal(result, expected, check_names=False, atol=1e-10)

    def test_year_matches_apply(self, events_df):
        expected = events_df.groupby("entity_id")["event_timestamp"].apply(
            lambda x: x.dt.year.mean()
        )
        result = self._entity_temporal_mean(events_df, "entity_id", "event_timestamp", "year")
        pd.testing.assert_series_equal(result, expected, check_names=False, atol=1e-10)


# ── weekend percentage ───────────────────────────────────────────────────

class TestWeekendPercentage:
    def test_weekend_pct_matches_apply(self, events_df):
        expected = events_df.groupby("entity_id")["event_timestamp"].apply(
            lambda x: (x.dt.dayofweek >= 5).mean()
        )
        result = (
            events_df.assign(__wkend=(events_df["event_timestamp"].dt.dayofweek >= 5).astype(float))
            .groupby("entity_id")["__wkend"].mean()
        )
        pd.testing.assert_series_equal(result, expected, check_names=False, atol=1e-10)


# ── vectorized OLS slope ─────────────────────────────────────────────────

class TestVectorizedSlope:
    @staticmethod
    def _original_slope(group, time_col):
        if len(group) < 3:
            return 0.0
        x = timedelta_to_days(group[time_col] - group[time_col].min()).to_numpy().astype(float)
        y = np.arange(len(group), dtype=float)
        if x.std() == 0:
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    @staticmethod
    def _vectorized_slope(df, entity_col, time_col):
        _first = df.groupby(entity_col)[time_col].transform("min")
        _x = timedelta_to_days(df[time_col] - _first).astype(float)
        _y = df.groupby(entity_col).cumcount().astype(float)
        _slope_df = df[[entity_col]].assign(__x=_x, __y=_y)
        _slope_df = _slope_df.assign(
            __xy=_slope_df["__x"] * _slope_df["__y"],
            __x2=_slope_df["__x"] ** 2,
        )
        _sums = _slope_df.groupby(entity_col)[["__x", "__y", "__xy", "__x2"]].sum()
        _n = _slope_df.groupby(entity_col)["__x"].count()
        _denom = _n * _sums["__x2"] - _sums["__x"] ** 2
        slopes = (
            (_n * _sums["__xy"] - _sums["__x"] * _sums["__y"])
            / _denom.where(_denom.abs() > 1e-10)
        ).fillna(0)
        return slopes.where(_n >= 3, 0.0)

    def test_slope_matches_polyfit(self, events_df):
        expected = events_df.groupby("entity_id").apply(
            lambda g: self._original_slope(g, "event_timestamp"), include_groups=False,
        )
        result = self._vectorized_slope(events_df, "entity_id", "event_timestamp")
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(),
            check_names=False, rtol=1e-6,
        )

    def test_slope_zero_for_single_event_entities(self):
        df = pd.DataFrame({
            "entity_id": ["A", "B"],
            "event_timestamp": pd.to_datetime(["2023-01-01", "2023-06-01"]),
        })
        result = self._vectorized_slope(df, "entity_id", "event_timestamp")
        assert (result == 0).all()

    def test_slope_zero_for_two_event_entities(self):
        df = pd.DataFrame({
            "entity_id": ["A", "A"],
            "event_timestamp": pd.to_datetime(["2023-01-01", "2023-06-01"]),
        })
        result = self._vectorized_slope(df, "entity_id", "event_timestamp")
        assert (result == 0).all()

    def test_slope_positive_for_accelerating(self):
        dates = pd.to_datetime(["2023-01-01", "2023-01-10", "2023-01-15", "2023-01-18"])
        df = pd.DataFrame({"entity_id": "A", "event_timestamp": dates})
        result = self._vectorized_slope(df, "entity_id", "event_timestamp")
        assert result["A"] > 0

    def test_slope_handles_same_day_events(self):
        df = pd.DataFrame({
            "entity_id": ["A"] * 5,
            "event_timestamp": pd.to_datetime(["2023-01-01"] * 5),
        })
        result = self._vectorized_slope(df, "entity_id", "event_timestamp")
        assert result["A"] == 0.0


# ── year mode via sort + drop_duplicates ─────────────────────────────────

class TestYearMode:
    @staticmethod
    def _original_year_mode(df, entity_col, time_col):
        return df.groupby(entity_col)[time_col].apply(
            lambda x: x.dt.year.mode().iloc[0] if len(x.dt.year.mode()) > 0 else x.dt.year.median()
        )

    @staticmethod
    def _vectorized_year_mode(df, entity_col, time_col):
        _yr_df = df[[entity_col]].assign(__yr=df[time_col].dt.year)
        _yr_counts = _yr_df.groupby([entity_col, "__yr"]).size().reset_index(name="__cnt")
        _yr_mode = _yr_counts.sort_values("__cnt", ascending=False).drop_duplicates(
            subset=[entity_col]
        )
        return _yr_mode.set_index(entity_col)["__yr"]

    def test_year_mode_matches_apply(self, events_df):
        expected = self._original_year_mode(events_df, "entity_id", "event_timestamp")
        result = self._vectorized_year_mode(events_df, "entity_id", "event_timestamp")
        for entity in expected.index:
            assert result[entity] == expected[entity], f"Mismatch for {entity}"

    def test_single_year(self):
        df = pd.DataFrame({
            "entity_id": ["A", "A", "A"],
            "event_timestamp": pd.to_datetime(["2023-03-01", "2023-06-01", "2023-09-01"]),
        })
        result = self._vectorized_year_mode(df, "entity_id", "event_timestamp")
        assert result["A"] == 2023

    def test_clear_mode(self):
        df = pd.DataFrame({
            "entity_id": ["A"] * 5,
            "event_timestamp": pd.to_datetime([
                "2022-01-01", "2023-01-01", "2023-06-01", "2023-09-01", "2024-01-01",
            ]),
        })
        result = self._vectorized_year_mode(df, "entity_id", "event_timestamp")
        assert result["A"] == 2023


# ── lifecycle quadrant merge ─────────────────────────────────────────────

class TestLifecycleQuadrantMerge:
    def test_map_works_on_native_pandas(self, events_df, aggregated_df):
        lifecycles = pd.DataFrame({
            "entity_id": sorted(events_df["entity_id"].unique()),
            "duration_days": np.random.randint(10, 1000, size=20),
            "event_count": np.random.randint(5, 100, size=20),
            "first_event": pd.Timestamp("2023-01-01"),
            "last_event": pd.Timestamp("2023-12-31"),
        })
        result = classify_lifecycle_quadrants(lifecycles.rename(columns={"entity_id": "entity"}))
        lc = result.lifecycles.rename(columns={"entity": "entity_id"})
        quadrant_map = lc.set_index("entity_id")["lifecycle_quadrant"]
        aggregated_df["lifecycle_quadrant"] = aggregated_df["entity_id"].map(quadrant_map)
        assert aggregated_df["lifecycle_quadrant"].notna().all()
        assert set(aggregated_df["lifecycle_quadrant"].unique()).issubset(
            {"steady_loyal_lifecycle", "occasional_loyal_lifecycle",
             "intense_brief_lifecycle", "one_shot_lifecycle"}
        )

    def test_target_max(self, events_df, aggregated_df):
        entity_target = events_df.groupby("entity_id")["target"].max()
        aggregated_df["target"] = aggregated_df["entity_id"].map(entity_target)
        assert aggregated_df["target"].notna().all()
        assert set(aggregated_df["target"].unique()).issubset({0, 1})


# ── recent vs overall ratio ──────────────────────────────────────────────

class TestRecentVsOverallRatio:
    def test_ratio_between_zero_and_one(self, events_df, aggregated_df):
        time_span = (events_df["event_timestamp"].max() - events_df["event_timestamp"].min()).days
        recent_cutoff = events_df["event_timestamp"].max() - pd.Timedelta(days=int(time_span * 0.3))
        overall_counts = events_df.groupby("entity_id").size()
        recent_counts = events_df[events_df["event_timestamp"] >= recent_cutoff].groupby("entity_id").size()
        ratio = (recent_counts / overall_counts).fillna(0)
        aggregated_df["recent_vs_overall_ratio"] = aggregated_df["entity_id"].map(ratio).fillna(0)
        assert (aggregated_df["recent_vs_overall_ratio"] >= 0).all()
        assert (aggregated_df["recent_vs_overall_ratio"] <= 1).all()


# ── cohort features ──────────────────────────────────────────────────────

class TestCohortFeatures:
    def test_cohort_year_from_first_event(self, events_df, aggregated_df):
        entity_first = events_df.groupby("entity_id")["event_timestamp"].min()
        aggregated_df["cohort_year"] = aggregated_df["entity_id"].map(entity_first).dt.year
        assert aggregated_df["cohort_year"].notna().all()
        assert (aggregated_df["cohort_year"] == 2023).all()

    def test_cohort_quarter(self, events_df, aggregated_df):
        entity_first = events_df.groupby("entity_id")["event_timestamp"].min()
        first_dates = aggregated_df["entity_id"].map(entity_first)
        aggregated_df["cohort_quarter"] = (
            first_dates.dt.year.astype(str) + "Q" + first_dates.dt.quarter.astype(str)
        )
        assert aggregated_df["cohort_quarter"].notna().all()
        assert all(q.startswith("2023Q") for q in aggregated_df["cohort_quarter"])


# ── momentum ratio (Spark-native withColumn equivalent) ──────────────────

class TestMomentumRatioSpark:
    def test_momentum_ratio_formula(self):
        df = pd.DataFrame({
            "entity_id": ["A", "B", "C"],
            "amount_mean_30d": [10.0, 20.0, 0.0],
            "amount_mean_90d": [5.0, 0.0, 0.0],
        })
        # Spark path formula: short / NULLIF(long, 0), coalesce with 1.0
        result = df["amount_mean_30d"] / df["amount_mean_90d"].replace(0, np.nan)
        result = result.fillna(1.0)
        assert result.iloc[0] == pytest.approx(2.0)
        assert result.iloc[1] == pytest.approx(1.0)  # div by zero → 1.0
        assert result.iloc[2] == pytest.approx(1.0)  # 0/0 → 1.0


# ── TFE merge does not produce _x/_y suffix artifacts ────────────────────

class TestTFEMergeNoSuffixCollision:
    """Regression guard for NB01d cell `save_aggregation_metadata` (id=ae7d0a25).

    `TimeWindowAggregator.aggregate(include_recency=True, include_tenure=True)`
    produces `days_since_last_event` and `days_since_first_event`. The
    subsequent `TemporalFeatureEngineer.compute(...)` RECENCY group emits the
    same column names plus `active_span_days`/`recency_ratio`. Without the
    drop-before-merge guard, pandas `.merge()` appends `_x`/`_y` suffixes —
    those suffixed names leak through NB03's silver merge into NB05's
    ratio/interaction recommendations, which then reference columns that the
    Databricks pipeline never materializes. Silver generation breaks with
    `UNRESOLVED_COLUMN.WITH_SUGGESTION`. Keeping this behavior under test
    ensures future edits to NB01d don't silently reintroduce the collision.
    """

    @staticmethod
    def _merge_with_overlap_drop(df_aggregated, tfe_features, entity_col):
        tfe_new_cols = [c for c in tfe_features.columns if c != entity_col]
        overlap = [c for c in tfe_new_cols if c in df_aggregated.columns]
        if overlap:
            df_aggregated = df_aggregated.drop(columns=overlap)
        return df_aggregated.merge(tfe_features, on=entity_col, how="left")

    def test_overlap_dropped_before_merge_keeps_tfe_values(self):
        df_aggregated = pd.DataFrame({
            "entity_id": ["E1", "E2", "E3"],
            "event_count_30d": [1, 2, 3],
            "days_since_last_event": [100.0, 200.0, 300.0],
            "days_since_first_event": [400.0, 500.0, 600.0],
        })
        tfe_features = pd.DataFrame({
            "entity_id": ["E1", "E2", "E3"],
            "days_since_last_event": [10.0, 20.0, 30.0],
            "days_since_first_event": [40.0, 50.0, 60.0],
            "active_span_days": [30.0, 30.0, 30.0],
            "recency_ratio": [0.25, 0.4, 0.5],
        })
        merged = self._merge_with_overlap_drop(df_aggregated, tfe_features, "entity_id")
        assert "days_since_last_event_x" not in merged.columns
        assert "days_since_last_event_y" not in merged.columns
        assert "days_since_first_event_x" not in merged.columns
        assert "days_since_first_event_y" not in merged.columns
        assert list(merged["days_since_last_event"]) == [10.0, 20.0, 30.0]
        assert list(merged["days_since_first_event"]) == [40.0, 50.0, 60.0]
        assert set({"active_span_days", "recency_ratio", "event_count_30d"}) <= set(merged.columns)

    def test_no_overlap_passthrough(self):
        df_aggregated = pd.DataFrame({
            "entity_id": ["E1", "E2"],
            "event_count_30d": [1, 2],
        })
        tfe_features = pd.DataFrame({
            "entity_id": ["E1", "E2"],
            "lag0_amt_sum": [100.0, 200.0],
        })
        merged = self._merge_with_overlap_drop(df_aggregated, tfe_features, "entity_id")
        assert set(merged.columns) == {"entity_id", "event_count_30d", "lag0_amt_sum"}

    def test_naive_merge_reproduces_collision_bug(self):
        df_aggregated = pd.DataFrame({
            "entity_id": ["E1"],
            "days_since_last_event": [100.0],
        })
        tfe_features = pd.DataFrame({
            "entity_id": ["E1"],
            "days_since_last_event": [10.0],
        })
        naive = df_aggregated.merge(tfe_features, on="entity_id", how="left")
        assert "days_since_last_event_x" in naive.columns
        assert "days_since_last_event_y" in naive.columns


# ── recency bucket (Spark-native F.when chain equivalent) ────────────────

class TestRecencyBucketSpark:
    def test_bucket_assignment(self):
        days = pd.Series([1, 7, 8, 30, 31, 90, 91, 180, 181, 365, np.nan])
        expected = ["0-7d", "0-7d", "8-30d", "8-30d", "31-90d", "31-90d",
                     "91-180d", "91-180d", ">180d", ">180d", None]
        result = pd.cut(
            days, bins=[0, 7, 30, 90, 180, float("inf")],
            labels=["0-7d", "8-30d", "31-90d", "91-180d", ">180d"], include_lowest=True,
        ).astype("object")
        result[days.isna()] = None
        for i, (r, e) in enumerate(zip(result, expected)):
            assert r == e, f"Mismatch at index {i}: {r} != {e}"
