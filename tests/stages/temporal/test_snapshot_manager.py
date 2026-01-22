import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from customer_retention.stages.temporal.snapshot_manager import SnapshotManager


class TestSnapshotManager:
    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def manager(self, temp_dir):
        return SnapshotManager(temp_dir)

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "entity_id": ["A", "B", "C", "D"],
            "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
            "label_timestamp": pd.to_datetime(["2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01"]),
            "label_available_flag": [True, True, True, False],
            "target": [1, 0, 1, 0],
            "feature_1": [100, 200, 300, 400],
            "feature_2": [10, 20, 30, 40]
        })


class TestCreateSnapshot(TestSnapshotManager):
    def test_create_snapshot_filters_by_availability(self, manager, sample_df):
        cutoff = datetime(2024, 7, 1)
        metadata = manager.create_snapshot(sample_df, cutoff, "target")

        assert metadata.row_count == 3  # Only label_available_flag == True

    def test_create_snapshot_filters_by_cutoff(self, manager, sample_df):
        cutoff = datetime(2024, 2, 15)  # Between row B and C feature_timestamps
        metadata = manager.create_snapshot(sample_df, cutoff, "target")

        assert metadata.row_count == 2  # Only rows with feature_timestamp <= cutoff AND available

    def test_create_snapshot_generates_version(self, manager, sample_df):
        cutoff = datetime(2024, 7, 1)
        metadata1 = manager.create_snapshot(sample_df, cutoff, "target")
        metadata2 = manager.create_snapshot(sample_df, cutoff, "target")

        assert metadata1.version == 1
        assert metadata2.version == 2
        assert metadata1.snapshot_id == "training_v1"
        assert metadata2.snapshot_id == "training_v2"

    def test_create_snapshot_computes_hash(self, manager, sample_df):
        cutoff = datetime(2024, 7, 1)
        metadata = manager.create_snapshot(sample_df, cutoff, "target")

        assert metadata.data_hash is not None
        assert len(metadata.data_hash) == 16

    def test_create_snapshot_identifies_features(self, manager, sample_df):
        cutoff = datetime(2024, 7, 1)
        metadata = manager.create_snapshot(sample_df, cutoff, "target")

        assert "feature_1" in metadata.feature_columns
        assert "feature_2" in metadata.feature_columns
        assert "entity_id" in metadata.feature_columns
        assert "target" not in metadata.feature_columns
        assert "feature_timestamp" not in metadata.feature_columns


class TestLoadSnapshot(TestSnapshotManager):
    def test_load_snapshot_returns_data_and_metadata(self, manager, sample_df):
        cutoff = datetime(2024, 7, 1)
        created = manager.create_snapshot(sample_df, cutoff, "target")

        df, metadata = manager.load_snapshot(created.snapshot_id)

        assert len(df) == created.row_count
        assert metadata.snapshot_id == created.snapshot_id

    def test_load_nonexistent_snapshot_raises_error(self, manager):
        with pytest.raises(FileNotFoundError, match="Snapshot not found"):
            manager.load_snapshot("nonexistent_v1")

    def test_load_snapshot_verifies_integrity(self, manager, sample_df, temp_dir):
        cutoff = datetime(2024, 7, 1)
        metadata = manager.create_snapshot(sample_df, cutoff, "target")

        snapshot_path = temp_dir / "snapshots" / f"{metadata.snapshot_id}.parquet"
        tampered_df = pd.DataFrame({"tampered": [1, 2, 3]})
        tampered_df.to_parquet(snapshot_path, index=False)

        with pytest.raises(ValueError, match="integrity check failed"):
            manager.load_snapshot(metadata.snapshot_id)


class TestListSnapshots(TestSnapshotManager):
    def test_list_snapshots_empty(self, manager):
        assert manager.list_snapshots() == []

    def test_list_snapshots_returns_all(self, manager, sample_df):
        cutoff = datetime(2024, 7, 1)
        manager.create_snapshot(sample_df, cutoff, "target")
        manager.create_snapshot(sample_df, cutoff, "target", snapshot_name="validation")

        snapshots = manager.list_snapshots()
        assert len(snapshots) == 2
        assert "training_v1" in snapshots
        assert "validation_v1" in snapshots


class TestGetLatestSnapshot(TestSnapshotManager):
    def test_get_latest_when_none_exist(self, manager):
        assert manager.get_latest_snapshot() is None

    def test_get_latest_returns_highest_version(self, manager, sample_df):
        cutoff = datetime(2024, 7, 1)
        manager.create_snapshot(sample_df, cutoff, "target")
        manager.create_snapshot(sample_df, cutoff, "target")
        manager.create_snapshot(sample_df, cutoff, "target")

        latest = manager.get_latest_snapshot()
        assert latest == "training_v3"


class TestCompareSnapshots(TestSnapshotManager):
    def test_compare_snapshots(self, manager, sample_df):
        cutoff1 = datetime(2024, 5, 1)
        cutoff2 = datetime(2024, 7, 1)

        meta1 = manager.create_snapshot(sample_df, cutoff1, "target")
        meta2 = manager.create_snapshot(sample_df, cutoff2, "target")

        comparison = manager.compare_snapshots(meta1.snapshot_id, meta2.snapshot_id)

        assert comparison["snapshot_1"] == meta1.snapshot_id
        assert comparison["snapshot_2"] == meta2.snapshot_id
        assert "row_diff" in comparison
        assert "column_diff" in comparison
