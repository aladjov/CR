from unittest.mock import MagicMock

import pandas as pd
import pytest

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


class TestSparkBuildSpineWithMockedSpark:
    @pytest.fixture()
    def mock_spark(self, monkeypatch):
        spark = MagicMock()
        created_sdfs = []

        class _FakeField:
            def __init__(self, name):
                self.name = name

        class _FakeSchema:
            def __init__(self, fields):
                self.fields = fields

        def _fake_empty_spine_schema(entity_key, as_of_column):
            return _FakeSchema([_FakeField(entity_key), _FakeField(as_of_column)])

        def _create_df(pdf_or_data, schema=None):
            sdf = MagicMock()
            if isinstance(pdf_or_data, list) and len(pdf_or_data) == 0 and schema is not None:
                col_names = [f.name for f in schema.fields]
                sdf._pdf = pd.DataFrame(columns=col_names)
            else:
                sdf._pdf = pdf_or_data.copy()

            def _cross_join(other):
                cross_result = MagicMock()
                cross_pdf = sdf._pdf.merge(other._pdf, how="cross")
                cross_result.pandas_api.return_value = cross_pdf
                cross_result._pdf = cross_pdf
                return cross_result

            sdf.crossJoin = _cross_join
            sdf.pandas_api.return_value = sdf._pdf
            created_sdfs.append(sdf)
            return sdf

        spark.createDataFrame.side_effect = _create_df
        monkeypatch.setattr(
            "customer_retention.stages.temporal.spark_temporal_merger.get_spark_session",
            lambda: spark,
        )
        monkeypatch.setattr(
            "customer_retention.stages.temporal.spark_temporal_merger._as_pandas_api",
            lambda sdf: sdf.pandas_api(),
        )
        monkeypatch.setattr(
            "customer_retention.stages.temporal.spark_temporal_merger._empty_spine_schema",
            _fake_empty_spine_schema,
        )
        return spark

    def test_native_pandas_entities_use_spark_crossjoin(self, mock_spark):
        merger = SparkTemporalMerger()
        entities = pd.Series(["A", "B"])
        spine = merger.build_spine(entities, ["2024-01-01", "2024-02-01"])
        assert mock_spark.createDataFrame.called
        assert len(spine) == 4

    def test_spine_has_correct_columns(self, mock_spark):
        merger = SparkTemporalMerger()
        spine = merger.build_spine(pd.Series(["X"]), ["2024-06-01"])
        assert "entity_id" in spine.columns
        assert "as_of_date" in spine.columns

    def test_empty_entities_returns_empty(self, mock_spark):
        merger = SparkTemporalMerger()
        spine = merger.build_spine(pd.Series([], dtype=str), ["2024-01-01"])
        assert len(spine) == 0

    def test_empty_dates_returns_empty(self, mock_spark):
        merger = SparkTemporalMerger()
        spine = merger.build_spine(pd.Series(["A"]), [])
        assert len(spine) == 0

    def test_empty_entities_returns_spark_dataframe(self, mock_spark):
        merger = SparkTemporalMerger()
        spine = merger.build_spine(pd.Series([], dtype=str), ["2024-01-01"])
        assert mock_spark.createDataFrame.called
        assert len(spine) == 0
        assert "entity_id" in spine.columns
        assert "as_of_date" in spine.columns

    def test_empty_dates_returns_spark_dataframe(self, mock_spark):
        merger = SparkTemporalMerger()
        spine = merger.build_spine(pd.Series(["A"]), [])
        assert mock_spark.createDataFrame.called
        assert len(spine) == 0
        assert "entity_id" in spine.columns
        assert "as_of_date" in spine.columns

    def test_merge_all_empty_spine_with_datasets(self, mock_spark):
        merger = SparkTemporalMerger()
        spine = merger.build_spine(pd.Series([], dtype=str), ["2024-01-01"])
        entity_df = pd.DataFrame({
            "entity_id": ["A", "B"],
            "tier": ["gold", "silver"],
        })
        datasets = [
            _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 0
        assert report.spine_rows == 0


class TestMergeAllConvertsSparkDataFrames:
    def test_entity_broadcast_with_spark_df(self, monkeypatch):
        merger = SparkTemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01"])
        entity_pdf = pd.DataFrame({
            "entity_id": ["A", "B"],
            "tier": ["gold", "silver"],
        })
        spark_df = MagicMock()
        spark_df.rdd = MagicMock()

        monkeypatch.setattr(
            "customer_retention.stages.temporal.spark_temporal_merger._as_pandas_api",
            lambda sdf: entity_pdf,
        )
        datasets = [
            _merge_input("customers", spark_df, DatasetGranularity.ENTITY_LEVEL),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 2
        assert "tier" in result.columns

    def test_event_snapshot_with_spark_df(self, monkeypatch):
        merger = SparkTemporalMerger()
        spine = _spine(["A"], ["2024-01-01", "2024-02-01"])
        event_pdf = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A", "A"],
            "as_of_date": ["2024-01-01", "2024-02-01"],
            "amount": [10, 20],
        })
        spark_df = MagicMock()
        spark_df.rdd = MagicMock()

        monkeypatch.setattr(
            "customer_retention.stages.temporal.spark_temporal_merger._as_pandas_api",
            lambda sdf: event_pdf,
        )
        datasets = [
            _merge_input("events", spark_df, DatasetGranularity.EVENT_LEVEL),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 2
        assert "amount" in result.columns

    def test_original_input_not_mutated(self, monkeypatch):
        merger = SparkTemporalMerger()
        spine = _spine(["A"], ["2024-01-01"])
        entity_pdf = pd.DataFrame({"entity_id": ["A"], "val": [1]})
        spark_df = MagicMock()
        spark_df.rdd = MagicMock()

        monkeypatch.setattr(
            "customer_retention.stages.temporal.spark_temporal_merger._as_pandas_api",
            lambda sdf: entity_pdf,
        )
        ds = _merge_input("src", spark_df, DatasetGranularity.ENTITY_LEVEL)
        merger.merge_all(spine, [ds])
        assert ds.df is spark_df
