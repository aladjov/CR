import pandas as pd

from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.stages.temporal.temporal_merger import (
    DatasetMergeInput,
    MergeConfig,
    TemporalMerger,
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


class TestMergeConfigDefaults:
    def test_default_values(self):
        cfg = MergeConfig()
        assert cfg.entity_key == "entity_id"
        assert cfg.as_of_column == "as_of_date"
        assert cfg.tolerance_days is None
        assert cfg.conflict_separator == "__"
        assert cfg.validate_temporal is True

    def test_custom_entity_key(self):
        cfg = MergeConfig(entity_key="customer_id")
        assert cfg.entity_key == "customer_id"

    def test_custom_tolerance(self):
        cfg = MergeConfig(tolerance_days=30)
        assert cfg.tolerance_days == 30


class TestBuildSpine:
    def test_cross_product(self):
        merger = TemporalMerger()
        entities = pd.Series(["A", "B", "C"])
        dates = ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
        spine = merger.build_spine(entities, dates)
        assert len(spine) == 12
        assert set(spine["entity_id"]) == {"A", "B", "C"}
        assert spine["as_of_date"].nunique() == 4

    def test_deduplicates_entities(self):
        merger = TemporalMerger()
        entities = pd.Series(["A", "A", "B", "B", "B"])
        dates = ["2024-01-01", "2024-02-01"]
        spine = merger.build_spine(entities, dates)
        assert len(spine) == 4

    def test_empty_grid_returns_empty(self):
        merger = TemporalMerger()
        entities = pd.Series(["A", "B"])
        spine = merger.build_spine(entities, [])
        assert len(spine) == 0
        assert "entity_id" in spine.columns
        assert "as_of_date" in spine.columns

    def test_empty_entities_returns_empty(self):
        merger = TemporalMerger()
        entities = pd.Series([], dtype=str)
        dates = ["2024-01-01"]
        spine = merger.build_spine(entities, dates)
        assert len(spine) == 0

    def test_as_of_date_is_datetime64(self):
        merger = TemporalMerger()
        entities = pd.Series(["A"])
        dates = ["2024-01-01"]
        spine = merger.build_spine(entities, dates)
        assert pd.api.types.is_datetime64_any_dtype(spine["as_of_date"])

    def test_custom_column_names(self):
        cfg = MergeConfig(entity_key="customer_id", as_of_column="observation_date")
        merger = TemporalMerger(config=cfg)
        entities = pd.Series(["X", "Y"])
        dates = ["2024-06-01"]
        spine = merger.build_spine(entities, dates)
        assert "customer_id" in spine.columns
        assert "observation_date" in spine.columns
        assert len(spine) == 2


class TestResolveColumnConflicts:
    def test_no_conflicts_returns_empty(self):
        left = {"entity_id", "as_of_date", "col_a"}
        right = {"entity_id", "as_of_date", "col_b"}
        rename = TemporalMerger._resolve_conflicts(
            left, right, {"entity_id", "as_of_date"}, "orders", "__"
        )
        assert rename == {}

    def test_single_conflict_gets_prefixed(self):
        left = {"entity_id", "as_of_date", "amount"}
        right = {"entity_id", "as_of_date", "amount"}
        rename = TemporalMerger._resolve_conflicts(
            left, right, {"entity_id", "as_of_date"}, "orders", "__"
        )
        assert rename == {"amount": "orders__amount"}

    def test_join_keys_never_renamed(self):
        left = {"entity_id", "as_of_date", "x"}
        right = {"entity_id", "as_of_date", "x"}
        rename = TemporalMerger._resolve_conflicts(
            left, right, {"entity_id", "as_of_date"}, "ds", "__"
        )
        assert "entity_id" not in rename
        assert "as_of_date" not in rename

    def test_multiple_conflicts(self):
        left = {"entity_id", "a", "b", "c"}
        right = {"entity_id", "a", "b", "d"}
        rename = TemporalMerger._resolve_conflicts(
            left, right, {"entity_id"}, "src", "__"
        )
        assert rename == {"a": "src__a", "b": "src__b"}

    def test_custom_separator(self):
        left = {"entity_id", "val"}
        right = {"entity_id", "val"}
        rename = TemporalMerger._resolve_conflicts(
            left, right, {"entity_id"}, "ds", "."
        )
        assert rename == {"val": "ds.val"}


class TestMergeEventSnapshot:
    def test_exact_join_matches(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01", "2024-02-01"])
        event_df = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A", "A", "B", "B"],
            "as_of_date": ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"],
            "metric": [10, 20, 30, 40],
        })
        ds = _merge_input("events", event_df, DatasetGranularity.EVENT_LEVEL)
        result = merger._merge_event_snapshot(spine, ds, set(spine.columns))
        assert "metric" in result.columns
        assert len(result) == 4

    def test_missing_combos_are_nan(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01", "2024-02-01"])
        event_df = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A"],
            "as_of_date": ["2024-01-01"],
            "metric": [99],
        })
        ds = _merge_input("events", event_df, DatasetGranularity.EVENT_LEVEL)
        result = merger._merge_event_snapshot(spine, ds, set(spine.columns))
        assert len(result) == 4
        assert result["metric"].isna().sum() == 3

    def test_column_conflict_prefixed(self):
        merger = TemporalMerger()
        spine = _spine(["A"], ["2024-01-01"])
        spine["metric"] = 999
        event_df = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A"],
            "as_of_date": ["2024-01-01"],
            "metric": [10],
        })
        ds = _merge_input("events", event_df, DatasetGranularity.EVENT_LEVEL)
        result = merger._merge_event_snapshot(spine, ds, set(spine.columns))
        assert "events__metric" in result.columns

    def test_spine_row_count_preserved(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B", "C"], ["2024-01-01", "2024-02-01", "2024-03-01"])
        event_df = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A"],
            "as_of_date": ["2024-01-01"],
            "metric": [1],
        })
        ds = _merge_input("events", event_df, DatasetGranularity.EVENT_LEVEL)
        result = merger._merge_event_snapshot(spine, ds, set(spine.columns))
        assert len(result) == len(spine)


class TestMergeEventSnapshotFallbackToBroadcast:
    def test_missing_as_of_column_broadcasts(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01", "2024-02-01"])
        aggregated_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "event_count": [5, 3],
        })
        ds = _merge_input("events", aggregated_df, DatasetGranularity.EVENT_LEVEL)
        result = merger._merge_event_snapshot(spine, ds, set(spine.columns))
        assert len(result) == 4
        assert "event_count" in result.columns
        a_rows = result[result["entity_id"] == "A"]
        assert (a_rows["event_count"] == 5).all()

    def test_missing_as_of_preserves_spine_rows(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B", "C"], ["2024-01-01", "2024-02-01", "2024-03-01"])
        aggregated_df = _entity_df("entity_id", {
            "entity_id": ["A"],
            "metric": [99],
        })
        ds = _merge_input("events", aggregated_df, DatasetGranularity.EVENT_LEVEL)
        result = merger._merge_event_snapshot(spine, ds, set(spine.columns))
        assert len(result) == 9
        assert result[result["entity_id"] == "B"]["metric"].isna().all()

    def test_missing_as_of_resolves_column_conflicts(self):
        merger = TemporalMerger()
        spine = _spine(["A"], ["2024-01-01"])
        spine["metric"] = 999
        aggregated_df = _entity_df("entity_id", {
            "entity_id": ["A"],
            "metric": [42],
        })
        ds = _merge_input("events", aggregated_df, DatasetGranularity.EVENT_LEVEL)
        result = merger._merge_event_snapshot(spine, ds, set(spine.columns))
        assert "events__metric" in result.columns


class TestMergeEntityBroadcast:
    def test_entity_features_on_every_date(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01", "2024-02-01"])
        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "segment": ["gold", "silver"],
        })
        ds = _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL)
        result = merger._merge_entity_broadcast(spine, ds, set(spine.columns))
        assert len(result) == 4
        a_rows = result[result["entity_id"] == "A"]
        assert all(a_rows["segment"] == "gold")

    def test_missing_entity_produces_nan(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01"])
        entity_df = _entity_df("entity_id", {
            "entity_id": ["A"],
            "segment": ["gold"],
        })
        ds = _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL)
        result = merger._merge_entity_broadcast(spine, ds, set(spine.columns))
        assert len(result) == 2
        assert result[result["entity_id"] == "B"]["segment"].isna().all()

    def test_extra_entity_not_in_spine_ignored(self):
        merger = TemporalMerger()
        spine = _spine(["A"], ["2024-01-01"])
        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "Z"],
            "segment": ["gold", "platinum"],
        })
        ds = _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL)
        result = merger._merge_entity_broadcast(spine, ds, set(spine.columns))
        assert len(result) == 1
        assert "Z" not in result["entity_id"].values

    def test_spine_row_count_preserved(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B", "C"], ["2024-01-01", "2024-02-01"])
        entity_df = _entity_df("entity_id", {
            "entity_id": ["A"],
            "segment": ["gold"],
        })
        ds = _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL)
        result = merger._merge_entity_broadcast(spine, ds, set(spine.columns))
        assert len(result) == 6


class TestAsofJoinDatetimeNormalization:
    def test_tz_aware_right_merges_with_naive_spine(self):
        merger = TemporalMerger()
        spine = _spine(["A"], ["2024-03-01", "2024-06-01"])
        entity_df = pd.DataFrame({
            "entity_id": ["A", "A"],
            "feature_timestamp": pd.to_datetime(
                ["2024-01-01", "2024-04-01"]
            ).tz_localize("UTC"),
            "score": [10, 20],
        })
        ds = _merge_input(
            "scores", entity_df, DatasetGranularity.ENTITY_LEVEL,
            feature_ts="feature_timestamp",
        )
        result = merger._merge_entity_asof(spine, ds, set(spine.columns))
        assert len(result) == 2
        row_mar = result[result["as_of_date"] == pd.Timestamp("2024-03-01")]
        assert row_mar["score"].iloc[0] == 10

    def test_microsecond_resolution_right_merges_with_ns_spine(self):
        merger = TemporalMerger()
        spine = _spine(["A"], ["2024-03-01"])
        entity_df = pd.DataFrame({
            "entity_id": ["A"],
            "feature_timestamp": pd.to_datetime(["2024-01-01"]).astype(
                "datetime64[us]"
            ),
            "score": [42],
        })
        ds = _merge_input(
            "scores", entity_df, DatasetGranularity.ENTITY_LEVEL,
            feature_ts="feature_timestamp",
        )
        result = merger._merge_entity_asof(spine, ds, set(spine.columns))
        assert result["score"].iloc[0] == 42

    def test_tz_aware_us_resolution_matches_error_scenario(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B"], ["2024-03-01", "2024-06-01"])
        entity_df = pd.DataFrame({
            "entity_id": ["A", "B"],
            "contract_start": pd.to_datetime(
                ["2023-06-15", "2024-01-20"]
            ).tz_localize("UTC").as_unit("us"),
            "tier": ["gold", "silver"],
        })
        ds = _merge_input(
            "profiles", entity_df, DatasetGranularity.ENTITY_LEVEL,
            feature_ts="contract_start",
        )
        result = merger._merge_entity_asof(spine, ds, set(spine.columns))
        assert len(result) == 4
        a_rows = result[result["entity_id"] == "A"]
        assert (a_rows["tier"] == "gold").all()


class TestMergeEntityAsof:
    def test_backward_join_picks_most_recent(self):
        merger = TemporalMerger()
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
        merger = TemporalMerger()
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
        merger = TemporalMerger()
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


class TestApplyTolerance:
    def test_within_tolerance_kept(self):
        df = pd.DataFrame({
            "entity_id": ["A", "A"],
            "as_of_date": pd.to_datetime(["2024-03-01", "2024-06-01"]),
            "feature_timestamp": pd.to_datetime(["2024-02-15", "2024-05-20"]),
            "score": [10, 20],
        })
        result, null_count = TemporalMerger._apply_tolerance(
            df, "feature_timestamp", "as_of_date", 30,
            feature_cols=["score"],
        )
        assert null_count == 0
        assert result["score"].notna().all()

    def test_beyond_tolerance_nulled(self):
        df = pd.DataFrame({
            "entity_id": ["A"],
            "as_of_date": pd.to_datetime(["2024-06-01"]),
            "feature_timestamp": pd.to_datetime(["2024-01-01"]),
            "score": [10],
        })
        result, null_count = TemporalMerger._apply_tolerance(
            df, "feature_timestamp", "as_of_date", 30,
            feature_cols=["score"],
        )
        assert null_count == 1
        assert result["score"].isna().all()

    def test_none_tolerance_skips(self):
        df = pd.DataFrame({
            "entity_id": ["A"],
            "as_of_date": pd.to_datetime(["2024-06-01"]),
            "feature_timestamp": pd.to_datetime(["2020-01-01"]),
            "score": [10],
        })
        result, null_count = TemporalMerger._apply_tolerance(
            df, "feature_timestamp", "as_of_date", None,
            feature_cols=["score"],
        )
        assert null_count == 0
        assert result["score"].iloc[0] == 10

    def test_returns_null_count(self):
        df = pd.DataFrame({
            "entity_id": ["A", "B", "C"],
            "as_of_date": pd.to_datetime(["2024-06-01"] * 3),
            "feature_timestamp": pd.to_datetime([
                "2024-05-30", "2024-01-01", "2024-02-01"
            ]),
            "score": [1, 2, 3],
        })
        _, null_count = TemporalMerger._apply_tolerance(
            df, "feature_timestamp", "as_of_date", 30,
            feature_cols=["score"],
        )
        assert null_count == 2


class TestMergeAll:
    def test_single_event_dataset(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01", "2024-02-01"])
        event_df = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A", "A", "B", "B"],
            "as_of_date": ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"],
            "amount": [10, 20, 30, 40],
        })
        datasets = [_merge_input("txns", event_df, DatasetGranularity.EVENT_LEVEL)]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 4
        assert "amount" in result.columns
        assert report.datasets_merged == ["txns"]

    def test_single_entity_dataset(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01", "2024-02-01"])
        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "segment": ["gold", "silver"],
        })
        datasets = [_merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL)]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 4
        assert "segment" in result.columns
        assert report.datasets_merged == ["customers"]

    def test_mixed_event_and_entity(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01", "2024-02-01"])
        event_df = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A", "B"],
            "as_of_date": ["2024-01-01", "2024-02-01"],
            "amount": [100, 200],
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
        assert "amount" in result.columns
        assert "tier" in result.columns
        assert set(report.datasets_merged) == {"txns", "customers"}

    def test_report_populated_correctly(self):
        merger = TemporalMerger()
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

    def test_temporal_validation_runs(self):
        merger = TemporalMerger(config=MergeConfig(validate_temporal=True))
        spine = _spine(["A"], ["2024-01-01"])
        event_df = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A"],
            "as_of_date": ["2024-01-01"],
            "val": [1],
        })
        datasets = [_merge_input("ds", event_df, DatasetGranularity.EVENT_LEVEL)]
        _, report = merger.merge_all(spine, datasets)
        assert "valid" in report.temporal_integrity

    def test_column_conflicts_across_two_event_datasets(self):
        merger = TemporalMerger()
        spine = _spine(["A"], ["2024-01-01"])
        ev1 = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A"],
            "as_of_date": ["2024-01-01"],
            "amount": [10],
        })
        ev2 = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A"],
            "as_of_date": ["2024-01-01"],
            "amount": [20],
        })
        datasets = [
            _merge_input("txns", ev1, DatasetGranularity.EVENT_LEVEL),
            _merge_input("emails", ev2, DatasetGranularity.EVENT_LEVEL),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert "amount" in result.columns
        assert "emails__amount" in result.columns
        assert len(report.renamed_columns) > 0


class TestMergeAllAggregatedEventFallback:
    def test_aggregated_event_without_as_of_broadcasts(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01", "2024-02-01"])
        aggregated_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "event_count": [5, 3],
            "total_amount": [100.0, 60.0],
        })
        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "B"],
            "segment": ["gold", "silver"],
        })
        datasets = [
            _merge_input("events", aggregated_df, DatasetGranularity.EVENT_LEVEL),
            _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 4
        assert "event_count" in result.columns
        assert "segment" in result.columns
        assert set(report.datasets_merged) == {"events", "customers"}


class TestEventSnapshotDeduplicatesJoinKeys:
    def test_duplicate_entity_as_of_pairs_deduplicated(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01", "2024-02-01"])
        event_df = _event_snapshot("entity_id", "as_of_date", {
            "entity_id": ["A", "A", "A", "B"],
            "as_of_date": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-01-01"],
            "metric": [10, 20, 30, 40],
        })
        ds = _merge_input("events", event_df, DatasetGranularity.EVENT_LEVEL)
        result = merger._merge_event_snapshot(spine, ds, set(spine.columns))
        assert len(result) == len(spine)

    def test_raw_event_data_without_as_of_deduplicated(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01", "2024-02-01"])
        raw_events = _entity_df("entity_id", {
            "entity_id": ["A", "A", "A", "B", "B"],
            "ticket_type": ["bug", "feature", "bug", "support", "billing"],
        })
        ds = _merge_input("tickets", raw_events, DatasetGranularity.EVENT_LEVEL)
        result = merger._merge_event_snapshot(spine, ds, set(spine.columns))
        assert len(result) == len(spine)

    def test_entity_broadcast_deduplicates_entity_key(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01", "2024-02-01"])
        dup_entity = _entity_df("entity_id", {
            "entity_id": ["A", "A", "B"],
            "segment": ["gold", "platinum", "silver"],
        })
        ds = _merge_input("customers", dup_entity, DatasetGranularity.ENTITY_LEVEL)
        result = merger._merge_entity_broadcast(spine, ds, set(spine.columns))
        assert len(result) == len(spine)

    def test_merge_all_preserves_spine_with_raw_event_data(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B", "C"], ["2024-01-01", "2024-02-01", "2024-03-01"])
        raw_events = _entity_df("entity_id", {
            "entity_id": ["A", "A", "A", "B", "B", "C"],
            "amount": [10, 20, 30, 40, 50, 60],
        })
        entity_df = _entity_df("entity_id", {
            "entity_id": ["A", "B", "C"],
            "tier": ["gold", "silver", "bronze"],
        })
        datasets = [
            _merge_input("tickets", raw_events, DatasetGranularity.EVENT_LEVEL),
            _merge_input("customers", entity_df, DatasetGranularity.ENTITY_LEVEL),
        ]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == len(spine)
        assert report.spine_rows == len(spine)


class TestMergeAllEdgeCases:
    def test_empty_datasets_returns_spine_unchanged(self):
        merger = TemporalMerger()
        spine = _spine(["A", "B"], ["2024-01-01"])
        result, report = merger.merge_all(spine, [])
        assert len(result) == 2
        assert list(result.columns) == ["entity_id", "as_of_date"]
        assert report.datasets_merged == []

    def test_single_entity_single_date(self):
        merger = TemporalMerger()
        spine = _spine(["A"], ["2024-01-01"])
        entity_df = _entity_df("entity_id", {
            "entity_id": ["A"],
            "val": [42],
        })
        datasets = [_merge_input("ds", entity_df, DatasetGranularity.ENTITY_LEVEL)]
        result, report = merger.merge_all(spine, datasets)
        assert len(result) == 1
        assert result["val"].iloc[0] == 42
        assert report.spine_rows == 1
        assert report.spine_entities == 1
        assert report.spine_dates == 1
