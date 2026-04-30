import logging
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


# ---------------------------------------------------------------------------
# FakeSparkDF: a thin pandas wrapper that simulates the native Spark DF API
# ---------------------------------------------------------------------------

class _FakeBroadcast:
    """Wrapper returned by F.broadcast() — just passes through the DF."""
    def __init__(self, sdf):
        self._fake_spark_df = sdf

    # forward attribute access so join() can read .columns etc.
    def __getattr__(self, name):
        return getattr(self._fake_spark_df, name)


class _FakeCol:
    """Simulates pyspark.sql.Column for filter / orderBy expressions."""
    def __init__(self, name):
        self._name = name

    def __le__(self, other):
        if isinstance(other, _FakeCol):
            return _FakeCondition(
                lambda df: df[self._name] <= df[other._name]
            )
        return _FakeCondition(lambda df: df[self._name] <= other)

    def __eq__(self, other):
        if isinstance(other, _FakeCol):
            return _FakeCondition(
                lambda df: df[self._name] == df[other._name]
            )
        return _FakeCondition(lambda df: df[self._name] == other)

    def desc(self):
        return _FakeSortKey(self._name, ascending=False)


class _FakeCondition:
    def __init__(self, fn):
        self._fn = fn

    def _evaluate(self, pdf):
        return self._fn(pdf)


class _FakeSortKey:
    def __init__(self, name, ascending):
        self.name = name
        self.ascending = ascending


class _FakeWindowSpec:
    def __init__(self, partition_cols=None, order_keys=None):
        self.partition_cols = partition_cols or []
        self.order_keys = order_keys or []

    def orderBy(self, *keys):  # noqa: N802
        return _FakeWindowSpec(self.partition_cols, list(keys))


class _FakeWindow:
    @staticmethod
    def partitionBy(*cols):  # noqa: N802
        return _FakeWindowSpec(list(cols))


class _FakeRowNumberExpr:
    """Evaluates ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ... DESC)."""
    def __init__(self, window_spec):
        self._ws = window_spec

    def _evaluate(self, pdf):
        part = [c if isinstance(c, str) else c._name for c in self._ws.partition_cols]
        sort_col = self._ws.order_keys[0].name
        ascending = self._ws.order_keys[0].ascending
        pdf = pdf.sort_values(sort_col, ascending=ascending)
        return pdf.groupby(part).cumcount() + 1


class _FakeRowNumber:
    def over(self, ws):
        return _FakeRowNumberExpr(ws)


class FakeSparkDF:
    """Wraps a pandas DataFrame with the subset of Spark DataFrame API
    used by SparkTemporalMerger's native join methods."""

    _is_fake_spark_df = True

    def __init__(self, pdf):
        self._pdf = pdf.copy()

    @property
    def columns(self):
        return list(self._pdf.columns)

    def count(self):
        return len(self._pdf)

    def distinct(self):
        return FakeSparkDF(self._pdf.drop_duplicates())

    def select(self, *cols):
        names = [c if isinstance(c, str) else c._name for c in cols]
        names = [n for n in names if n in self._pdf.columns]
        return FakeSparkDF(self._pdf[names])

    def drop(self, *cols):
        return FakeSparkDF(self._pdf.drop(columns=list(cols), errors="ignore"))

    def dropDuplicates(self, cols=None):  # noqa: N802
        return FakeSparkDF(self._pdf.drop_duplicates(subset=cols, keep="last"))

    def withColumnRenamed(self, old, new):  # noqa: N802
        return FakeSparkDF(self._pdf.rename(columns={old: new}))

    def withColumn(self, name, expr):  # noqa: N802
        pdf = self._pdf.copy()
        pdf[name] = expr._evaluate(pdf)
        return FakeSparkDF(pdf)

    def filter(self, cond):
        mask = cond._evaluate(self._pdf)
        return FakeSparkDF(self._pdf[mask].reset_index(drop=True))

    def join(self, other, on, how="inner"):
        if isinstance(other, _FakeBroadcast):
            other = other._fake_spark_df
        if isinstance(on, str):
            on = [on]
        result = self._pdf.merge(other._pdf, on=on, how=how)
        return FakeSparkDF(result)

    def crossJoin(self, other):  # noqa: N802
        cross_pdf = self._pdf.merge(other._pdf, how="cross")
        return FakeSparkDF(cross_pdf)

    def localCheckpoint(self, eager=True):  # noqa: N802
        return FakeSparkDF(self._pdf.copy())

    def pandas_api(self):
        return self._pdf


# Fake pyspark.sql.functions and Window replacements
class _FakeFunctions:
    @staticmethod
    def col(name):
        return _FakeCol(name)

    @staticmethod
    def broadcast(sdf):
        return _FakeBroadcast(sdf)

    @staticmethod
    def row_number():
        return _FakeRowNumber()


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
    """Tests merge_all via the pandas fallback (no Spark session)."""

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

    @pytest.mark.spark
    def test_native_pandas_entities_use_spark_crossjoin(self, mock_spark):
        merger = SparkTemporalMerger()
        entities = pd.Series(["A", "B"])
        spine = merger.build_spine(entities, ["2024-01-01", "2024-02-01"])
        assert mock_spark.createDataFrame.called
        assert len(spine) == 4

    @pytest.mark.spark
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


# ---------------------------------------------------------------------------
# Tests for build_spine with native Spark DataFrame input
# ---------------------------------------------------------------------------

def _patch_build_spine_spark(monkeypatch):
    """Patch so build_spine recognises FakeSparkDF as native Spark."""
    mock_spark = MagicMock()

    def _create_df(pdf_or_data, schema=None):
        if isinstance(pdf_or_data, list) and len(pdf_or_data) == 0:
            col_names = [f.name for f in schema.fields] if schema else []
            return FakeSparkDF(pd.DataFrame(columns=col_names))
        return FakeSparkDF(pdf_or_data.copy())

    mock_spark.createDataFrame.side_effect = _create_df
    monkeypatch.setattr(
        "customer_retention.stages.temporal.spark_temporal_merger.get_spark_session",
        lambda: mock_spark,
    )
    monkeypatch.setattr(
        "customer_retention.stages.temporal.spark_temporal_merger._as_pandas_api",
        lambda sdf: sdf.pandas_api(),
    )
    monkeypatch.setattr(
        "customer_retention.stages.temporal.spark_temporal_merger._is_native_spark_df",
        lambda df: isinstance(df, FakeSparkDF),
    )
    return mock_spark


@pytest.mark.spark
class TestSparkBuildSpineWithNativeDF:
    """Tests build_spine when entity_ids is already a native Spark DataFrame."""

    def test_native_spark_df_produces_cross_product(self, monkeypatch):
        _patch_build_spine_spark(monkeypatch)
        merger = SparkTemporalMerger()
        entity_sdf = FakeSparkDF(pd.DataFrame({"entity_id": ["A", "B", "C"]}))
        spine = merger.build_spine(entity_sdf, ["2024-01-01", "2024-02-01"])
        assert len(spine) == 6
        assert set(spine["entity_id"]) == {"A", "B", "C"}
        assert "as_of_date" in spine.columns

    def test_native_spark_df_deduplicates_entities(self, monkeypatch):
        _patch_build_spine_spark(monkeypatch)
        merger = SparkTemporalMerger()
        entity_sdf = FakeSparkDF(pd.DataFrame({"entity_id": ["A", "A", "B"]}))
        spine = merger.build_spine(entity_sdf, ["2024-01-01"])
        assert len(spine) == 2

    def test_native_spark_df_renames_entity_column(self, monkeypatch):
        _patch_build_spine_spark(monkeypatch)
        merger = SparkTemporalMerger()
        entity_sdf = FakeSparkDF(pd.DataFrame({"customer_id": ["X", "Y"]}))
        spine = merger.build_spine(entity_sdf, ["2024-06-01"])
        assert "entity_id" in spine.columns
        assert len(spine) == 2

    def test_native_spark_df_empty_dates(self, monkeypatch):
        mock_spark = _patch_build_spine_spark(monkeypatch)

        class _FakeField:
            def __init__(self, name):
                self.name = name

        class _FakeSchema:
            def __init__(self, fields):
                self.fields = fields

        monkeypatch.setattr(
            "customer_retention.stages.temporal.spark_temporal_merger._empty_spine_schema",
            lambda ek, ac: _FakeSchema([_FakeField(ek), _FakeField(ac)]),
        )
        merger = SparkTemporalMerger()
        entity_sdf = FakeSparkDF(pd.DataFrame({"entity_id": ["A"]}))
        spine = merger.build_spine(entity_sdf, [])
        assert len(spine) == 0

    def test_no_create_data_frame_called_for_entities(self, monkeypatch):
        """Entity IDs should stay distributed — no createDataFrame for entities."""
        mock_spark = _patch_build_spine_spark(monkeypatch)
        merger = SparkTemporalMerger()
        entity_sdf = FakeSparkDF(pd.DataFrame({"entity_id": ["A", "B"]}))
        merger.build_spine(entity_sdf, ["2024-01-01"])
        # createDataFrame should only be called for dates, not entities
        assert mock_spark.createDataFrame.call_count == 1
        call_args = mock_spark.createDataFrame.call_args
        assert "as_of_date" in call_args[0][0].columns


# ---------------------------------------------------------------------------
# Tests for lineage breaking in merge_all
# ---------------------------------------------------------------------------

class TestMergeAllLineageBreaking:
    """Verify that merge_all checkpoints after each dataset join."""

    def test_checkpoint_called_per_dataset(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        checkpoint_calls = []

        import customer_retention.stages.temporal.spark_temporal_merger as mod

        def _tracking_break(sdf):
            checkpoint_calls.append(len(sdf.columns))
            return sdf.localCheckpoint(eager=True)

        monkeypatch.setattr(mod, "_break_lineage", _tracking_break)

        merger = SparkTemporalMerger(
            config=MergeConfig(validate_temporal=False, checkpoint_every=1)
        )
        spine = _spine(["A", "B"], ["2024-01-01"])
        event_df = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A", "B"],
            "as_of_date": ["2024-01-01", "2024-01-01"],
            "amount": [10, 20],
        })
        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "tier": ["gold", "silver"],
        })
        datasets = [
            _merge_input("txns", event_df, DatasetGranularity.EVENT_LEVEL),
            _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(checkpoint_calls) == 2
        assert "amount" in result.columns
        assert "tier" in result.columns

    def test_checkpoint_preserves_data_correctness(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(
            config=MergeConfig(validate_temporal=False, checkpoint_every=1)
        )
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
        datasets = [
            _merge_input("txns", event_df, DatasetGranularity.EVENT_LEVEL),
            _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 4
        row_a_jan = result[
            (result["entity_id"] == "A")
            & (result["as_of_date"] == pd.Timestamp("2024-01-01"))
        ]
        assert row_a_jan["amount"].iloc[0] == 10
        assert row_a_jan["tier"].iloc[0] == "gold"

    def test_timing_logged(self, monkeypatch, caplog):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine = _spine(["A"], ["2024-01-01"])
        entity_df = _entity_df("entity_id", {"entity_id": ["A"], "tier": ["gold"]})
        datasets = [
            _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL),
        ]
        with caplog.at_level(logging.INFO, logger="customer_retention.stages.temporal.spark_temporal_merger"):
            merger.merge_all(spine, datasets)
        assert any("Merged 1/1" in msg and "customers" in msg for msg in caplog.messages)


class TestMergeAllCheckpointEvery:
    """checkpoint_every controls _break_lineage frequency without changing output.

    Default is 4 — for 8 sources that means 2 checkpoints (boundary + last)
    instead of 8, halving executor-disk I/O for the SPS-shape merge while
    keeping bounded lineage growth between checkpoints.
    """

    def _build_eight_dataset_inputs(self):
        spine = _spine(["A", "B", "C"], ["2024-01-01", "2024-02-01"])
        datasets = []
        for i in range(8):
            df = _event_snapshot("entity_id", "as_of_date", {
                "entity_id": ["A", "A", "B", "B", "C", "C"],
                "as_of_date": [
                    "2024-01-01", "2024-02-01",
                    "2024-01-01", "2024-02-01",
                    "2024-01-01", "2024-02-01",
                ],
                f"f_{i}": [i, 10 + i, 20 + i, 30 + i, 40 + i, 50 + i],
            })
            datasets.append(
                _merge_input(f"src_{i}", df, DatasetGranularity.EVENT_LEVEL)
            )
        return spine, datasets

    def _count_checkpoints_for(self, monkeypatch, checkpoint_every, spine, datasets):
        _patch_native_spark(monkeypatch)
        import customer_retention.stages.temporal.spark_temporal_merger as mod

        calls = []

        def _tracking_break(sdf):
            calls.append(len(sdf.columns))
            return sdf.localCheckpoint(eager=True)

        monkeypatch.setattr(mod, "_break_lineage", _tracking_break)
        merger = SparkTemporalMerger(
            config=MergeConfig(
                validate_temporal=False, checkpoint_every=checkpoint_every,
            )
        )
        result, report = merger.merge_all(spine, datasets)
        return result, report, calls

    def test_default_checkpoint_every_is_four(self):
        cfg = MergeConfig()
        assert cfg.checkpoint_every == 4

    def test_eight_sources_with_default_runs_two_checkpoints(self, monkeypatch):
        spine, datasets = self._build_eight_dataset_inputs()
        result, _, calls = self._count_checkpoints_for(
            monkeypatch, 4, spine, datasets,
        )
        # Boundary at i=3 (after 4th merge) + last at i=7 = 2 calls
        assert len(calls) == 2
        for i in range(8):
            assert f"f_{i}" in result.columns

    def test_checkpoint_every_one_matches_old_behavior(self, monkeypatch):
        spine, datasets = self._build_eight_dataset_inputs()
        _, _, calls = self._count_checkpoints_for(
            monkeypatch, 1, spine, datasets,
        )
        assert len(calls) == 8

    def test_checkpoint_every_eight_runs_one_checkpoint(self, monkeypatch):
        spine, datasets = self._build_eight_dataset_inputs()
        _, _, calls = self._count_checkpoints_for(
            monkeypatch, 8, spine, datasets,
        )
        # Only the final-merge checkpoint fires
        assert len(calls) == 1

    def test_checkpoint_every_zero_clamps_to_one(self, monkeypatch):
        spine, datasets = self._build_eight_dataset_inputs()
        # checkpoint_every=0 would divide-by-zero in modulo; max(1, ...) clamps it
        _, _, calls = self._count_checkpoints_for(
            monkeypatch, 0, spine, datasets,
        )
        assert len(calls) == 8

    def test_parity_output_identical_across_intervals(self, monkeypatch):
        """Same inputs + different checkpoint_every -> identical output frames.

        This is the regression guard: the optimization must not change values,
        rows, or columns. Compares old behavior (every=1) against new default
        (every=4) and the most aggressive setting (every=N).
        """
        spine, datasets = self._build_eight_dataset_inputs()

        result_one, report_one, _ = self._count_checkpoints_for(
            monkeypatch, 1, spine, datasets,
        )
        result_four, report_four, _ = self._count_checkpoints_for(
            monkeypatch, 4, spine, datasets,
        )
        result_eight, report_eight, _ = self._count_checkpoints_for(
            monkeypatch, 8, spine, datasets,
        )

        # Same row count
        assert len(result_one) == len(result_four) == len(result_eight)
        # Same column set
        assert set(result_one.columns) == set(result_four.columns)
        assert set(result_one.columns) == set(result_eight.columns)
        # Same values: align on (entity_id, as_of_date) and compare each feature
        sort_keys = ["entity_id", "as_of_date"]
        cols = sorted(result_one.columns)
        ro = result_one[cols].sort_values(sort_keys).reset_index(drop=True)
        rf = result_four[cols].sort_values(sort_keys).reset_index(drop=True)
        re_ = result_eight[cols].sort_values(sort_keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(ro, rf, check_dtype=False)
        pd.testing.assert_frame_equal(ro, re_, check_dtype=False)
        # Report fields independent of checkpoint cadence
        assert report_one.spine_rows == report_four.spine_rows == report_eight.spine_rows
        assert report_one.total_columns == report_four.total_columns == report_eight.total_columns
        assert report_one.datasets_merged == report_four.datasets_merged == report_eight.datasets_merged
        assert report_one.columns_per_dataset == report_four.columns_per_dataset == report_eight.columns_per_dataset

    def test_report_populates_per_source_seconds(self, monkeypatch):
        """seconds_per_dataset must contain a non-negative entry per merged source."""
        spine, datasets = self._build_eight_dataset_inputs()
        _, report, _ = self._count_checkpoints_for(monkeypatch, 4, spine, datasets)
        assert set(report.seconds_per_dataset.keys()) == {f"src_{i}" for i in range(8)}
        for name, secs in report.seconds_per_dataset.items():
            assert isinstance(secs, float)
            assert secs >= 0.0, f"{name}: {secs}"

    def test_report_populates_summary_timers(self, monkeypatch):
        """spine_stats_seconds, checkpoint_seconds, merge_total_seconds populated.

        validation_seconds is 0 unless validate_temporal=True.
        """
        spine, datasets = self._build_eight_dataset_inputs()
        _, report, _ = self._count_checkpoints_for(monkeypatch, 4, spine, datasets)
        assert report.spine_stats_seconds >= 0.0
        assert report.checkpoint_seconds >= 0.0
        assert report.merge_total_seconds >= 0.0
        assert report.checkpoint_count == 2
        # validate_temporal=False in _count_checkpoints_for, so no validation timing
        assert report.validation_seconds == 0.0

    def test_seconds_per_dataset_default_factory_is_independent(self):
        """Two MergeReport instances must not share a mutable default."""
        from customer_retention.stages.temporal.temporal_merger import MergeReport
        a = MergeReport()
        b = MergeReport()
        a.seconds_per_dataset["x"] = 1.0
        assert b.seconds_per_dataset == {}

    def test_parity_with_mixed_granularities(self, monkeypatch):
        """Parity must hold across event-level, entity-level, and asof datasets,
        not just same-shape repeat sources."""
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
            "feature_timestamp": pd.to_datetime(
                ["2023-06-01", "2024-01-15", "2023-12-01"]
            ),
            "credit_score": [700, 720, 650],
        })

        def _build_datasets():
            return [
                _merge_input("txns", event_df.copy(), DatasetGranularity.EVENT_LEVEL),
                _merge_input(
                    "customers", entity_df.copy(), DatasetGranularity.ENTITY_LEVEL,
                ),
                _merge_input(
                    "credit", asof_df.copy(), DatasetGranularity.ENTITY_LEVEL,
                    feature_ts="feature_timestamp",
                ),
            ]

        result_one, _, _ = self._count_checkpoints_for(
            monkeypatch, 1, spine.copy(), _build_datasets(),
        )
        result_four, _, _ = self._count_checkpoints_for(
            monkeypatch, 4, spine.copy(), _build_datasets(),
        )

        sort_keys = ["entity_id", "as_of_date"]
        cols = sorted(result_one.columns)
        ro = result_one[cols].sort_values(sort_keys).reset_index(drop=True)
        rf = result_four[cols].sort_values(sort_keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(ro, rf, check_dtype=False)


# ---------------------------------------------------------------------------
# Tests for the native Spark merge_all path
# ---------------------------------------------------------------------------

def _patch_native_spark(monkeypatch):
    """Patch the module so merge_all uses FakeSparkDF instead of real Spark."""
    fake_spark = MagicMock()  # just needs to be truthy

    monkeypatch.setattr(
        "customer_retention.stages.temporal.spark_temporal_merger.get_spark_session",
        lambda: fake_spark,
    )

    def _fake_to_native(df):
        if isinstance(df, FakeSparkDF):
            return df
        return FakeSparkDF(df)

    monkeypatch.setattr(
        "customer_retention.stages.temporal.spark_temporal_merger._to_native_spark",
        _fake_to_native,
    )
    monkeypatch.setattr(
        "customer_retention.stages.temporal.spark_temporal_merger._as_pandas_api",
        lambda sdf: sdf.pandas_api(),
    )

    # Patch pyspark imports used inside the join methods
    import types
    fake_F = types.ModuleType("pyspark.sql.functions")
    fake_F.col = _FakeFunctions.col
    fake_F.broadcast = _FakeFunctions.broadcast
    fake_F.row_number = _FakeFunctions.row_number

    monkeypatch.setattr(
        "customer_retention.stages.temporal.spark_temporal_merger.SparkTemporalMerger._spark_join_broadcast",
        _patched_broadcast_join,
    )
    monkeypatch.setattr(
        "customer_retention.stages.temporal.spark_temporal_merger.SparkTemporalMerger._spark_join_asof",
        _patched_asof_join,
    )
    monkeypatch.setattr(
        "customer_retention.stages.temporal.spark_temporal_merger.SparkTemporalMerger._spark_join_event",
        _patched_event_join,
    )

    return fake_spark


def _patched_event_join(self, left_sdf, right_sdf, ds, existing_cols):
    """FakeSparkDF-compatible version of _spark_join_event."""
    entity_key = self.config.entity_key
    as_of_column = self.config.as_of_column
    join_cols = [entity_key, as_of_column]

    if as_of_column not in right_sdf.columns:
        right_sdf = right_sdf.dropDuplicates([entity_key])
        return _patched_broadcast_join(self, left_sdf, right_sdf, ds, existing_cols)

    right_sdf = right_sdf.dropDuplicates(join_cols)

    right_feature_cols = set(right_sdf.columns) - set(join_cols)
    rename_map = self._resolve_conflicts(
        existing_cols, right_feature_cols, set(join_cols),
        ds.name, self.config.conflict_separator,
    )
    for old_name, new_name in rename_map.items():
        right_sdf = right_sdf.withColumnRenamed(old_name, new_name)

    return left_sdf.join(right_sdf, on=join_cols, how="left")


def _patched_broadcast_join(self, left_sdf, right_sdf, ds, existing_cols):
    """FakeSparkDF-compatible version of _spark_join_broadcast (no F.broadcast)."""
    entity_key = self.config.entity_key
    join_keys = {entity_key}

    right_sdf = right_sdf.dropDuplicates([entity_key])

    right_feature_cols = set(right_sdf.columns) - join_keys
    rename_map = self._resolve_conflicts(
        existing_cols, right_feature_cols, join_keys,
        ds.name, self.config.conflict_separator,
    )
    for old_name, new_name in rename_map.items():
        right_sdf = right_sdf.withColumnRenamed(old_name, new_name)

    return left_sdf.join(right_sdf, on=entity_key, how="left")


def _patched_asof_join(self, left_sdf, right_sdf, ds, existing_cols):
    """FakeSparkDF-compatible version of _spark_join_asof using FakeSparkDF ops."""
    F = _FakeFunctions
    entity_key = self.config.entity_key
    as_of_column = self.config.as_of_column
    ft_col = ds.feature_timestamp_column

    join_keys = {entity_key, as_of_column}
    right_feature_cols = set(right_sdf.columns) - {entity_key, ft_col}
    rename_map = self._resolve_conflicts(
        existing_cols, right_feature_cols, join_keys,
        ds.name, self.config.conflict_separator,
    )
    for old_name, new_name in rename_map.items():
        right_sdf = right_sdf.withColumnRenamed(old_name, new_name)
        if old_name == ft_col:
            ft_col = new_name

    feature_cols = [
        c for c in right_sdf.columns if c not in {entity_key, ft_col}
    ]

    right_ts_alias = f"__{ds.name}_ts__"
    right_sdf = right_sdf.withColumnRenamed(ft_col, right_ts_alias)

    joined = left_sdf.join(right_sdf, on=entity_key, how="inner")
    joined = joined.filter(F.col(right_ts_alias).__le__(F.col(as_of_column)))

    w = _FakeWindow.partitionBy(entity_key, as_of_column).orderBy(
        F.col(right_ts_alias).desc()
    )
    joined = joined.withColumn("_rn_", _FakeRowNumber().over(w))
    joined = joined.filter(_FakeCondition(lambda df: df["_rn_"] == 1))
    joined = joined.drop("_rn_", right_ts_alias)

    best_cols = [entity_key, as_of_column] + feature_cols
    best = joined.select(*[c for c in best_cols if c in joined.columns])

    return left_sdf.join(best, on=[entity_key, as_of_column], how="left")


class TestMergeOne:
    """Tests for SparkTemporalMerger.merge_one() — single-dataset join."""

    def test_event_join_adds_columns(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine_pdf = _spine(["A", "B"], ["2024-01-01", "2024-02-01"])
        spine_sdf = FakeSparkDF(spine_pdf)

        event_df = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A", "A", "B", "B"],
            "as_of_date": ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"],
            "amount": [10, 20, 30, 40],
        })
        ds = _merge_input("txns", event_df, DatasetGranularity.EVENT_LEVEL)
        result_sdf, new_cols = merger.merge_one(spine_sdf, ds)
        assert "amount" in new_cols
        assert result_sdf.count() == 4

    def test_broadcast_join_adds_columns(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine_sdf = FakeSparkDF(_spine(["A", "B"], ["2024-01-01"]))

        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "tier": ["gold", "silver"],
        })
        ds = _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL)
        result_sdf, new_cols = merger.merge_one(spine_sdf, ds)
        assert "tier" in new_cols
        result_pdf = result_sdf.pandas_api()
        assert result_pdf[result_pdf["entity_id"] == "A"]["tier"].iloc[0] == "gold"

    def test_asof_join_adds_columns(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine_sdf = FakeSparkDF(_spine(["A"], ["2024-03-01", "2024-06-01"]))

        asof_df = pd.DataFrame({
            "entity_id": ["A", "A"],
            "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "score": [10, 20],
        })
        ds = _merge_input(
            "credit", asof_df, DatasetGranularity.ENTITY_LEVEL,
            feature_ts="feature_timestamp",
        )
        result_sdf, new_cols = merger.merge_one(spine_sdf, ds)
        assert "score" in new_cols

    def test_does_not_break_lineage(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        checkpoint_calls = []

        import customer_retention.stages.temporal.spark_temporal_merger as mod
        original_break = mod._break_lineage

        def _tracking_break(sdf):
            checkpoint_calls.append(1)
            return original_break(sdf)

        monkeypatch.setattr(mod, "_break_lineage", _tracking_break)

        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine_sdf = FakeSparkDF(_spine(["A"], ["2024-01-01"]))
        entity_df = _entity_df("entity_id", {"entity_id": ["A"], "tier": ["gold"]})
        ds = _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL)
        merger.merge_one(spine_sdf, ds)
        assert len(checkpoint_calls) == 0, "merge_one must not call _break_lineage"

    def test_column_conflicts_resolved(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine_pdf = _spine(["A", "B"], ["2024-01-01"])
        spine_pdf["score"] = [1, 2]
        spine_sdf = FakeSparkDF(spine_pdf)

        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "score": [10, 20],
        })
        ds = _merge_input("src", entity_df, DatasetGranularity.ENTITY_LEVEL)
        result_sdf, new_cols = merger.merge_one(spine_sdf, ds)
        assert "src__score" in new_cols
        assert "src__score" in result_sdf.columns

    def test_spine_rows_preserved(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine_sdf = FakeSparkDF(_spine(["A", "B", "C"], ["2024-01-01", "2024-02-01"]))

        entity_df = _entity_df("entity_id", {
            "entity_id": ["A"],
            "tier": ["gold"],
        })
        ds = _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL)
        result_sdf, _ = merger.merge_one(spine_sdf, ds)
        assert result_sdf.count() == 6


class TestNativeSparkMergeAll:
    """Tests the native Spark join path via FakeSparkDF mocks."""

    def test_event_join(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine = _spine(["A", "B"], ["2024-01-01", "2024-02-01"])

        event_df = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A", "A", "B", "B"],
            "as_of_date": ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"],
            "amount": [10, 20, 30, 40],
        })

        datasets = [
            _merge_input("txns", event_df, DatasetGranularity.EVENT_LEVEL),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 4
        assert "amount" in result.columns
        assert report.spine_rows == 4
        assert report.datasets_merged == ["txns"]

    def test_broadcast_join(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine = _spine(["A", "B"], ["2024-01-01"])

        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "tier": ["gold", "silver"],
        })

        datasets = [
            _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 2
        assert "tier" in result.columns
        assert result[result["entity_id"] == "A"]["tier"].iloc[0] == "gold"
        assert result[result["entity_id"] == "B"]["tier"].iloc[0] == "silver"
        assert report.columns_per_dataset == {"customers": 1}

    def test_asof_join(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine = _spine(["A"], ["2024-03-01", "2024-06-01"])

        asof_df = pd.DataFrame({
            "entity_id": ["A", "A"],
            "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "score": [10, 20],
        })

        datasets = [
            _merge_input(
                "credit", asof_df, DatasetGranularity.ENTITY_LEVEL,
                feature_ts="feature_timestamp",
            ),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 2
        assert "score" in result.columns
        row_mar = result[result["as_of_date"] == pd.Timestamp("2024-03-01")]
        row_jun = result[result["as_of_date"] == pd.Timestamp("2024-06-01")]
        assert row_mar["score"].iloc[0] == 10
        assert row_jun["score"].iloc[0] == 20

    def test_asof_join_no_future_data(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine = _spine(["A"], ["2024-03-01"])

        asof_df = pd.DataFrame({
            "entity_id": ["A"],
            "feature_timestamp": pd.to_datetime(["2024-06-01"]),
            "score": [99],
        })

        datasets = [
            _merge_input(
                "scores", asof_df, DatasetGranularity.ENTITY_LEVEL,
                feature_ts="feature_timestamp",
            ),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 1
        assert result["score"].isna().all()

    def test_asof_join_preserves_spine_rows(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine = _spine(["A", "B", "C"], ["2024-01-01", "2024-02-01"])

        asof_df = pd.DataFrame({
            "entity_id": ["A"],
            "feature_timestamp": pd.to_datetime(["2023-06-01"]),
            "score": [10],
        })

        datasets = [
            _merge_input(
                "scores", asof_df, DatasetGranularity.ENTITY_LEVEL,
                feature_ts="feature_timestamp",
            ),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 6  # all spine rows preserved

    def test_mixed_join_types(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
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
            "feature_timestamp": pd.to_datetime([
                "2023-06-01", "2024-01-15", "2023-12-01",
            ]),
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

    def test_report_populated(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine = _spine(["A", "B", "C"], ["2024-01-01", "2024-02-01"])

        event_df = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A"],
            "as_of_date": ["2024-01-01"],
            "x": [1],
            "y": [2],
        })

        datasets = [
            _merge_input("events", event_df, DatasetGranularity.EVENT_LEVEL),
        ]
        _, report = merger.merge_all(spine, datasets)
        assert report.spine_rows == 6
        assert report.spine_entities == 3
        assert report.spine_dates == 2
        assert report.columns_per_dataset == {"events": 2}
        assert report.total_columns >= 4

    def test_column_conflict_resolved(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine = _spine(["A", "B"], ["2024-01-01"])
        spine["score"] = [1, 2]  # spine already has "score"

        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "score": [10, 20],
        })

        datasets = [
            _merge_input("src", entity_df, DatasetGranularity.ENTITY_LEVEL),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert "src__score" in result.columns
        assert len(report.renamed_columns) > 0

    def test_empty_spine(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine = pd.DataFrame(columns=["entity_id", "as_of_date"])
        spine["as_of_date"] = pd.to_datetime(spine["as_of_date"])

        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "tier": ["gold", "silver"],
        })

        datasets = [
            _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 0
        assert report.spine_rows == 0

    def test_event_without_as_of_date_falls_back_to_broadcast(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine = _spine(["A", "B"], ["2024-01-01"])

        # Event DF without as_of_date column → should fall back to broadcast
        event_df = pd.DataFrame({
            "entity_id": ["A", "B"],
            "metric": [100, 200],
        })

        datasets = [
            _merge_input("events", event_df, DatasetGranularity.EVENT_LEVEL),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 2
        assert "metric" in result.columns

    def test_original_input_not_mutated(self, monkeypatch):
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine = _spine(["A"], ["2024-01-01"])
        entity_pdf = pd.DataFrame({"entity_id": ["A"], "val": [1]})
        original_cols = list(entity_pdf.columns)
        ds = _merge_input("src", entity_pdf, DatasetGranularity.ENTITY_LEVEL)
        merger.merge_all(spine, [ds])
        assert list(ds.df.columns) == original_cols

    def test_asof_fallback_to_broadcast_when_timestamp_missing(self, monkeypatch):
        """When feature_timestamp_column is set but missing from DataFrame, fall back to broadcast."""
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine = _spine(["A", "B"], ["2024-01-01"])

        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "dow_cos": [0.5, 0.7],
            "dow_sin": [0.3, 0.4],
        })
        datasets = [
            _merge_input(
                "emails", entity_df, DatasetGranularity.ENTITY_LEVEL,
                feature_ts="nonexistent_timestamp",
            ),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 2
        assert "dow_cos" in result.columns
        assert report.datasets_merged == ["emails"]

    def test_asof_fallback_logs_warning_when_timestamp_missing(self, monkeypatch, caplog):
        """When falling back from asof to broadcast, a warning is logged."""
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine = _spine(["A"], ["2024-01-01"])

        entity_df = _entity_df("entity_id", {"entity_id": ["A"], "score": [10]})
        datasets = [
            _merge_input(
                "credit", entity_df, DatasetGranularity.ENTITY_LEVEL,
                feature_ts="missing_ts",
            ),
        ]
        with caplog.at_level(logging.WARNING, logger="customer_retention.stages.temporal.spark_temporal_merger"):
            merger.merge_all(spine, datasets)
        assert any("missing_ts" in msg and "credit" in msg for msg in caplog.messages)

    def test_merge_one_asof_fallback_when_timestamp_missing(self, monkeypatch):
        """merge_one also falls back to broadcast when timestamp column is missing."""
        _patch_native_spark(monkeypatch)
        merger = SparkTemporalMerger(config=MergeConfig(validate_temporal=False))
        spine_sdf = FakeSparkDF(_spine(["A", "B"], ["2024-01-01"]))

        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "tier": ["gold", "silver"],
        })
        ds = _merge_input(
            "customers", entity_df, DatasetGranularity.ENTITY_LEVEL,
            feature_ts="no_such_column",
        )
        result_sdf, new_cols = merger.merge_one(spine_sdf, ds)
        assert "tier" in new_cols
        assert result_sdf.count() == 2

    def test_fallback_to_parent_without_spark(self, monkeypatch):
        """When get_spark_session returns None, merge_all falls back to parent."""
        monkeypatch.setattr(
            "customer_retention.stages.temporal.spark_temporal_merger.get_spark_session",
            lambda: None,
        )
        merger = SparkTemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01"])
        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "tier": ["gold", "silver"],
        })
        datasets = [
            _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 2
        assert "tier" in result.columns
