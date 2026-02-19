import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.active_dataset_store import (
    load_active_dataset,
    load_active_dataset_distributed,
    load_gold_features,
    load_gold_features_distributed,
    load_merge_dataset,
    load_merge_dataset_distributed,
    load_silver_merged,
    load_silver_merged_distributed,
    save_active_dataset,
    save_aggregated_dataset,
    save_gold_features,
)
from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.integrations.adapters.factory import get_delta


@pytest.fixture()
def namespace(tmp_path):
    ns = RunNamespace(root=tmp_path, run_id="proj-test1234")
    ns.setup()
    return ns


@pytest.fixture()
def sample_df():
    return pd.DataFrame({"customer_id": [1, 2, 3], "revenue": [100.0, 200.0, 300.0]})


class TestSaveAndLoadRoundTrip:
    def test_save_and_load_round_trip(self, namespace, sample_df):
        save_active_dataset(namespace, "customers", sample_df)
        loaded = load_active_dataset(namespace, "customers")
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_returns_dlt_path(self, namespace, sample_df):
        path = save_active_dataset(namespace, "customers", sample_df)
        assert path == namespace.landing_table_dir("customers")


class TestLoadMissing:
    def test_load_missing_raises_file_not_found(self, namespace):
        with pytest.raises(FileNotFoundError):
            load_active_dataset(namespace, "nonexistent")


class TestOverwrite:
    def test_overwrite_replaces_data(self, namespace, sample_df):
        save_active_dataset(namespace, "customers", sample_df)
        new_df = pd.DataFrame({"customer_id": [10], "score": [0.5]})
        save_active_dataset(namespace, "customers", new_df)
        loaded = load_active_dataset(namespace, "customers")
        pd.testing.assert_frame_equal(loaded, new_df)


class TestLandingTableDirPathConvention:
    def test_landing_table_dir_path_convention(self, namespace, sample_df):
        save_active_dataset(namespace, "orders", sample_df)
        expected_dir = namespace.landing_table_dir("orders")
        assert expected_dir.is_dir()


class TestLoadMergeDataset:
    def test_entity_level_returns_active_dataset(self, namespace, sample_df):
        save_active_dataset(namespace, "customers", sample_df)
        result = load_merge_dataset(namespace, "customers", DatasetGranularity.ENTITY_LEVEL)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_event_level_prefers_aggregated_delta(self, namespace, sample_df):
        save_active_dataset(namespace, "events", sample_df)
        agg_df = pd.DataFrame({"customer_id": [1, 2], "as_of_date": ["2024-01-01", "2024-02-01"], "count": [10, 20]})
        save_aggregated_dataset(namespace, "events", agg_df)
        result = load_merge_dataset(namespace, "events", DatasetGranularity.EVENT_LEVEL)
        pd.testing.assert_frame_equal(result, agg_df)

    def test_event_level_falls_back_when_no_aggregated(self, namespace, sample_df):
        save_active_dataset(namespace, "events", sample_df)
        result = load_merge_dataset(namespace, "events", DatasetGranularity.EVENT_LEVEL)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_unknown_granularity_returns_active_dataset(self, namespace, sample_df):
        save_active_dataset(namespace, "misc", sample_df)
        result = load_merge_dataset(namespace, "misc", DatasetGranularity.UNKNOWN)
        pd.testing.assert_frame_equal(result, sample_df)


class TestLoadSilverMerged:
    def test_returns_silver_merged_when_exists(self, namespace):
        merged_df = pd.DataFrame({"customer_id": [1, 2], "churned": [0, 1], "feature_a": [10, 20]})
        delta = get_delta(force_local=True)
        delta.write(merged_df, str(namespace.silver_merged_path), mode="overwrite")
        result = load_silver_merged(namespace, "events", DatasetGranularity.EVENT_LEVEL)
        pd.testing.assert_frame_equal(result, merged_df)

    def test_falls_back_to_bronze_when_no_silver(self, namespace, sample_df):
        agg_df = pd.DataFrame({"customer_id": [1, 2], "count": [10, 20]})
        save_active_dataset(namespace, "events", sample_df)
        save_aggregated_dataset(namespace, "events", agg_df)
        result = load_silver_merged(namespace, "events", DatasetGranularity.EVENT_LEVEL)
        pd.testing.assert_frame_equal(result, agg_df)

    def test_falls_back_to_landing_when_nothing(self, namespace, sample_df):
        save_active_dataset(namespace, "events", sample_df)
        result = load_silver_merged(namespace, "events", DatasetGranularity.EVENT_LEVEL)
        pd.testing.assert_frame_equal(result, sample_df)


class TestGoldFeatures:
    def test_save_and_load_gold_round_trip(self, namespace, sample_df):
        save_gold_features(namespace, "cust_emai__abc1234", sample_df)
        loaded = load_gold_features(namespace, "cust_emai__abc1234")
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_save_gold_returns_path(self, namespace, sample_df):
        path = save_gold_features(namespace, "cust_emai__abc1234", sample_df)
        assert path == namespace.gold_table_dir("cust_emai__abc1234")

    def test_load_gold_missing_raises(self, namespace):
        with pytest.raises(FileNotFoundError):
            load_gold_features(namespace, "nonexistent__0000000")


class TestLoadMergeDatasetDistributed:
    def test_reads_without_conversion(self, namespace, sample_df):
        save_active_dataset(namespace, "customers", sample_df)
        result = load_merge_dataset_distributed(
            namespace, "customers", DatasetGranularity.ENTITY_LEVEL,
        )
        pd.testing.assert_frame_equal(result, sample_df)

    def test_event_level_prefers_aggregated(self, namespace, sample_df):
        save_active_dataset(namespace, "events", sample_df)
        agg_df = pd.DataFrame({
            "customer_id": [1, 2],
            "as_of_date": ["2024-01-01", "2024-02-01"],
            "count": [10, 20],
        })
        save_aggregated_dataset(namespace, "events", agg_df)
        result = load_merge_dataset_distributed(
            namespace, "events", DatasetGranularity.EVENT_LEVEL,
        )
        pd.testing.assert_frame_equal(result, agg_df)

    def test_event_level_falls_back_to_landing(self, namespace, sample_df):
        save_active_dataset(namespace, "events", sample_df)
        result = load_merge_dataset_distributed(
            namespace, "events", DatasetGranularity.EVENT_LEVEL,
        )
        pd.testing.assert_frame_equal(result, sample_df)

    def test_missing_raises_file_not_found(self, namespace):
        with pytest.raises(FileNotFoundError):
            load_merge_dataset_distributed(
                namespace, "nonexistent", DatasetGranularity.ENTITY_LEVEL,
            )


class TestLoadActiveDatasetDistributed:
    def test_reads_without_conversion(self, namespace, sample_df):
        save_active_dataset(namespace, "customers", sample_df)
        result = load_active_dataset_distributed(namespace, "customers")
        pd.testing.assert_frame_equal(result, sample_df)

    def test_missing_raises_file_not_found(self, namespace):
        with pytest.raises(FileNotFoundError):
            load_active_dataset_distributed(namespace, "nonexistent")


class TestLoadSilverMergedDistributed:
    def test_returns_silver_when_exists(self, namespace):
        merged_df = pd.DataFrame({
            "customer_id": [1, 2],
            "churned": [0, 1],
            "feature_a": [10, 20],
        })
        delta = get_delta(force_local=True)
        delta.write(merged_df, str(namespace.silver_merged_path), mode="overwrite")
        result = load_silver_merged_distributed(
            namespace, "events", DatasetGranularity.EVENT_LEVEL,
        )
        pd.testing.assert_frame_equal(result, merged_df)

    def test_falls_back_to_bronze_when_no_silver(self, namespace, sample_df):
        agg_df = pd.DataFrame({"customer_id": [1, 2], "count": [10, 20]})
        save_active_dataset(namespace, "events", sample_df)
        save_aggregated_dataset(namespace, "events", agg_df)
        result = load_silver_merged_distributed(
            namespace, "events", DatasetGranularity.EVENT_LEVEL,
        )
        pd.testing.assert_frame_equal(result, agg_df)

    def test_falls_back_to_landing_when_nothing(self, namespace, sample_df):
        save_active_dataset(namespace, "events", sample_df)
        result = load_silver_merged_distributed(
            namespace, "events", DatasetGranularity.EVENT_LEVEL,
        )
        pd.testing.assert_frame_equal(result, sample_df)


class TestLoadGoldFeaturesDistributed:
    def test_round_trip(self, namespace, sample_df):
        save_gold_features(namespace, "cust_emai__abc1234", sample_df)
        result = load_gold_features_distributed(namespace, "cust_emai__abc1234")
        pd.testing.assert_frame_equal(result, sample_df)

    def test_missing_raises(self, namespace):
        with pytest.raises(FileNotFoundError):
            load_gold_features_distributed(namespace, "nonexistent__0000000")
