import pytest
from datetime import datetime
import pandas as pd

from customer_retention.stages.temporal import SnapshotManager


class TestSnapshotCutoffInHash:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "entity_id": ["A", "B", "C"],
            "feature_1": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
            "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
            "label_timestamp": pd.to_datetime(["2024-03-01", "2024-03-15", "2024-04-01"]),
            "label_available_flag": [True, True, True],
        })

    def test_same_data_different_cutoff_produces_different_hash(self, tmp_path, sample_df):
        manager = SnapshotManager(tmp_path)

        cutoff_1 = datetime(2024, 6, 1)
        cutoff_2 = datetime(2024, 7, 1)

        hash_1 = manager._compute_hash(sample_df, cutoff_1)
        hash_2 = manager._compute_hash(sample_df, cutoff_2)

        assert hash_1 != hash_2

    def test_same_data_same_cutoff_produces_same_hash(self, tmp_path, sample_df):
        manager = SnapshotManager(tmp_path)

        cutoff = datetime(2024, 6, 1)

        hash_1 = manager._compute_hash(sample_df, cutoff)
        hash_2 = manager._compute_hash(sample_df, cutoff)

        assert hash_1 == hash_2

    def test_different_data_same_cutoff_produces_different_hash(self, tmp_path):
        manager = SnapshotManager(tmp_path)

        df_1 = pd.DataFrame({"a": [1, 2, 3]})
        df_2 = pd.DataFrame({"a": [1, 2, 4]})
        cutoff = datetime(2024, 6, 1)

        hash_1 = manager._compute_hash(df_1, cutoff)
        hash_2 = manager._compute_hash(df_2, cutoff)

        assert hash_1 != hash_2

    def test_hash_without_cutoff_differs_from_hash_with_cutoff(self, tmp_path, sample_df):
        manager = SnapshotManager(tmp_path)

        cutoff = datetime(2024, 6, 1)

        hash_without = manager._compute_hash(sample_df, None)
        hash_with = manager._compute_hash(sample_df, cutoff)

        assert hash_without != hash_with

    def test_snapshot_integrity_includes_cutoff(self, tmp_path, sample_df):
        manager = SnapshotManager(tmp_path)
        cutoff = datetime(2024, 6, 1)

        metadata = manager.create_snapshot(sample_df, cutoff, "target", "test")

        assert metadata.data_hash is not None
        assert metadata.cutoff_date == cutoff

    def test_load_snapshot_verifies_cutoff_in_hash(self, tmp_path, sample_df):
        manager = SnapshotManager(tmp_path)
        cutoff = datetime(2024, 6, 1)

        metadata = manager.create_snapshot(sample_df, cutoff, "target", "test")

        df_loaded, meta_loaded = manager.load_snapshot(metadata.snapshot_id)

        assert meta_loaded.cutoff_date == cutoff
        assert len(df_loaded) > 0

    def test_modified_cutoff_in_metadata_fails_integrity_check(self, tmp_path, sample_df):
        manager = SnapshotManager(tmp_path)
        cutoff = datetime(2024, 6, 1)

        metadata = manager.create_snapshot(sample_df, cutoff, "target", "test")

        metadata_path = tmp_path / "snapshots" / f"{metadata.snapshot_id}_metadata.json"
        import json
        with open(metadata_path) as f:
            meta_dict = json.load(f)

        meta_dict["cutoff_date"] = datetime(2024, 7, 1).isoformat()
        with open(metadata_path, "w") as f:
            json.dump(meta_dict, f)

        with pytest.raises(ValueError, match="integrity check failed"):
            manager.load_snapshot(metadata.snapshot_id)
