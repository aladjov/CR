import pandas as pd

from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.stages.temporal.spark_temporal_merger import SparkTemporalMerger
from customer_retention.stages.temporal.temporal_merger import (
    DatasetMergeInput,
    MergeConfig,
)


def _spine(entities, dates, entity_key="entity_id"):
    return pd.MultiIndex.from_product(
        [entities, pd.to_datetime(dates)],
        names=[entity_key, "as_of_date"],
    ).to_frame(index=False)


def _event_snapshot(entity_key, as_of_col, data):
    df = pd.DataFrame(data)
    df[as_of_col] = pd.to_datetime(df[as_of_col])
    return df


def _entity_df(entity_key, data):
    return pd.DataFrame(data)


def _merge_input(name, df, granularity, feature_ts=None):
    return DatasetMergeInput(
        name=name,
        df=df,
        granularity=granularity,
        feature_timestamp_column=feature_ts,
    )


class TestSparkBuildSpine:
    def test_cross_product(self):
        merger = SparkTemporalMerger()
        entities = pd.Series(["A", "B", "C"])
        dates = ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
        spine = merger.build_spine(entities, dates)
        assert len(spine) == 12
        assert set(spine["entity_id"]) == {"A", "B", "C"}
        assert spine["as_of_date"].nunique() == 4

    def test_deduplicates_entities(self):
        merger = SparkTemporalMerger()
        entities = pd.Series(["A", "A", "B", "B", "B"])
        dates = ["2024-01-01", "2024-02-01"]
        spine = merger.build_spine(entities, dates)
        assert len(spine) == 4

    def test_empty_grid_returns_empty(self):
        merger = SparkTemporalMerger()
        entities = pd.Series(["A", "B"])
        spine = merger.build_spine(entities, [])
        assert len(spine) == 0
        assert "entity_id" in spine.columns
        assert "as_of_date" in spine.columns

    def test_empty_entities_returns_empty(self):
        merger = SparkTemporalMerger()
        entities = pd.Series([], dtype=str)
        dates = ["2024-01-01"]
        spine = merger.build_spine(entities, dates)
        assert len(spine) == 0

    def test_custom_column_names(self):
        cfg = MergeConfig(entity_key="customer_id", as_of_column="observation_date")
        merger = SparkTemporalMerger(config=cfg)
        entities = pd.Series(["X", "Y"])
        dates = ["2024-06-01"]
        spine = merger.build_spine(entities, dates)
        assert "customer_id" in spine.columns
        assert "observation_date" in spine.columns
        assert len(spine) == 2


class TestSparkAsofJoin:
    def test_backward_join_picks_most_recent(self):
        merger = SparkTemporalMerger()
        spine = _spine(["A"], ["2024-03-01", "2024-06-01"])
        entity_df = pd.DataFrame({
            "entity_id": ["A", "A"],
            "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "score": [10, 20],
        })
        ds = _merge_input(
            "scores", entity_df, DatasetGranularity.ENTITY_LEVEL,
            feature_ts="feature_timestamp",
        )
        result = merger._merge_entity_asof(spine, ds, set(spine.columns))
        row_mar = result[result["as_of_date"] == pd.Timestamp("2024-03-01")]
        row_jun = result[result["as_of_date"] == pd.Timestamp("2024-06-01")]
        assert row_mar["score"].iloc[0] == 10
        assert row_jun["score"].iloc[0] == 20

    def test_no_future_data_included(self):
        merger = SparkTemporalMerger()
        spine = _spine(["A"], ["2024-03-01"])
        entity_df = pd.DataFrame({
            "entity_id": ["A"],
            "feature_timestamp": pd.to_datetime(["2024-06-01"]),
            "score": [99],
        })
        ds = _merge_input(
            "scores", entity_df, DatasetGranularity.ENTITY_LEVEL,
            feature_ts="feature_timestamp",
        )
        result = merger._merge_entity_asof(spine, ds, set(spine.columns))
        assert result["score"].isna().all()

    def test_no_matching_timestamp_produces_nan(self):
        merger = SparkTemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01"])
        entity_df = pd.DataFrame({
            "entity_id": ["B"],
            "feature_timestamp": pd.to_datetime(["2023-06-01"]),
            "score": [42],
        })
        ds = _merge_input(
            "scores", entity_df, DatasetGranularity.ENTITY_LEVEL,
            feature_ts="feature_timestamp",
        )
        result = merger._merge_entity_asof(spine, ds, set(spine.columns))
        assert result[result["entity_id"] == "A"]["score"].isna().all()
        assert result[result["entity_id"] == "B"]["score"].iloc[0] == 42

    def test_spine_row_count_preserved(self):
        merger = SparkTemporalMerger()
        spine = _spine(["A", "B", "C"], ["2024-01-01", "2024-02-01"])
        entity_df = pd.DataFrame({
            "entity_id": ["A"],
            "feature_timestamp": pd.to_datetime(["2023-06-01"]),
            "score": [10],
        })
        ds = _merge_input(
            "scores", entity_df, DatasetGranularity.ENTITY_LEVEL,
            feature_ts="feature_timestamp",
        )
        result = merger._merge_entity_asof(spine, ds, set(spine.columns))
        assert len(result) == len(spine)

    def test_column_conflict_resolved(self):
        merger = SparkTemporalMerger()
        spine = _spine(["A"], ["2024-03-01"])
        spine["score"] = 999
        entity_df = pd.DataFrame({
            "entity_id": ["A"],
            "feature_timestamp": pd.to_datetime(["2024-01-01"]),
            "score": [42],
        })
        ds = _merge_input(
            "scores", entity_df, DatasetGranularity.ENTITY_LEVEL,
            feature_ts="feature_timestamp",
        )
        result = merger._merge_entity_asof(spine, ds, set(spine.columns))
        assert "scores__score" in result.columns


class TestSparkMergeAll:
    def test_mixed_event_entity_asof(self):
        merger = SparkTemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01", "2024-02-01"])

        event_df = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A", "A", "B", "B"],
            "as_of_date": ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"],
            "amount": [10, 20, 30, 40],
        })

        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "tier": ["gold", "silver"],
        })

        asof_df = pd.DataFrame({
            "entity_id": ["A", "A", "B"],
            "feature_timestamp": pd.to_datetime(["2023-06-01", "2024-01-15", "2023-12-01"]),
            "credit_score": [700, 720, 650],
        })

        datasets = [
            _merge_input("txns", event_df, DatasetGranularity.EVENT_LEVEL),
            _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL),
            _merge_input(
                "credit", asof_df, DatasetGranularity.ENTITY_LEVEL,
                feature_ts="feature_timestamp",
            ),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 4
        assert "amount" in result.columns
        assert "tier" in result.columns
        assert "credit_score" in result.columns
        assert set(report.datasets_merged) == {"txns", "customers", "credit"}

    def test_report_populated_correctly(self):
        merger = SparkTemporalMerger()
        spine = _spine(["A", "B", "C"], ["2024-01-01", "2024-02-01"])
        event_df = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A"],
            "as_of_date": ["2024-01-01"],
            "x": [1],
            "y": [2],
        })
        datasets = [_merge_input("events", event_df, DatasetGranularity.EVENT_LEVEL)]
        _, report = merger.merge_all(spine, datasets)
        assert report.spine_rows == 6
        assert report.spine_entities == 3
        assert report.spine_dates == 2
        assert report.columns_per_dataset == {"events": 2}
        assert report.total_columns >= 4
