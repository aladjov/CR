from datetime import datetime

from customer_retention.stages.temporal import (
    DatasetSnapshot,
    PointInTimeRegistry,
)


class TestDatasetSnapshot:
    def test_to_dict_and_from_dict_roundtrip(self):
        snapshot = DatasetSnapshot(
            dataset_name="retail_churn",
            snapshot_id="snap_20240601_abc123",
            cutoff_date=datetime(2024, 6, 1, 12, 0, 0),
            source_path="/data/retail.csv",
            row_count=10000,
        )
        data = snapshot.to_dict()
        restored = DatasetSnapshot.from_dict(data)

        assert restored.dataset_name == snapshot.dataset_name
        assert restored.snapshot_id == snapshot.snapshot_id
        assert restored.cutoff_date == snapshot.cutoff_date
        assert restored.source_path == snapshot.source_path
        assert restored.row_count == snapshot.row_count


class TestPointInTimeRegistryInit:
    def test_creates_empty_registry(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        assert len(registry.snapshots) == 0
        assert registry.get_reference_cutoff() is None

    def test_loads_existing_registry(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        registry.register_snapshot("dataset1", "snap1", datetime(2024, 6, 1), "/data/1.csv", 1000)
        registry._save()

        new_registry = PointInTimeRegistry(tmp_path)
        assert len(new_registry.snapshots) == 1
        assert new_registry.get_snapshot("dataset1") is not None


class TestPointInTimeRegistryRegister:
    def test_register_first_snapshot(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        cutoff = datetime(2024, 6, 1)
        snapshot = registry.register_snapshot("retail", "snap_001", cutoff, "/data/retail.csv", 5000)

        assert snapshot.dataset_name == "retail"
        assert snapshot.cutoff_date == cutoff
        assert registry.get_reference_cutoff() == cutoff

    def test_register_multiple_snapshots_same_cutoff(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        cutoff = datetime(2024, 6, 1)

        registry.register_snapshot("retail", "snap_001", cutoff, "/data/retail.csv", 5000)
        registry.register_snapshot("bank", "snap_002", cutoff, "/data/bank.csv", 3000)
        registry.register_snapshot("telecom", "snap_003", cutoff, "/data/telecom.csv", 8000)

        assert len(registry.snapshots) == 3
        report = registry.check_consistency()
        assert report.is_consistent

    def test_register_overwrites_existing_dataset(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        cutoff = datetime(2024, 6, 1)

        registry.register_snapshot("retail", "snap_001", cutoff, "/data/retail.csv", 5000)
        registry.register_snapshot("retail", "snap_002", cutoff, "/data/retail_v2.csv", 6000)

        assert len(registry.snapshots) == 1
        assert registry.get_snapshot("retail").snapshot_id == "snap_002"
        assert registry.get_snapshot("retail").row_count == 6000


class TestPointInTimeRegistryConsistency:
    def test_empty_registry_is_consistent(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        report = registry.check_consistency()

        assert report.is_consistent
        assert report.reference_cutoff is None
        assert len(report.inconsistent_datasets) == 0

    def test_single_dataset_is_consistent(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        cutoff = datetime(2024, 6, 1)
        registry.register_snapshot("retail", "snap_001", cutoff, "/data/retail.csv", 5000)

        report = registry.check_consistency()
        assert report.is_consistent
        assert report.reference_cutoff == cutoff

    def test_detects_inconsistent_cutoff_dates(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)

        registry.register_snapshot("retail", "snap_001", datetime(2024, 6, 1), "/data/retail.csv", 5000)
        registry.register_snapshot("bank", "snap_002", datetime(2024, 6, 15), "/data/bank.csv", 3000)

        report = registry.check_consistency()
        assert not report.is_consistent
        assert "bank" in report.inconsistent_datasets
        assert "Re-run exploration" in report.message

    def test_same_date_different_time_is_consistent(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)

        registry.register_snapshot("retail", "snap_001", datetime(2024, 6, 1, 10, 0, 0), "/data/retail.csv", 5000)
        registry.register_snapshot("bank", "snap_002", datetime(2024, 6, 1, 14, 30, 0), "/data/bank.csv", 3000)

        report = registry.check_consistency()
        assert report.is_consistent


class TestPointInTimeRegistryValidation:
    def test_validate_first_cutoff_always_valid(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        is_valid, message = registry.validate_cutoff(datetime(2024, 6, 1))

        assert is_valid
        assert "First dataset" in message

    def test_validate_matching_cutoff(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        cutoff = datetime(2024, 6, 1)
        registry.register_snapshot("retail", "snap_001", cutoff, "/data/retail.csv", 5000)

        is_valid, message = registry.validate_cutoff(datetime(2024, 6, 1, 15, 0, 0))
        assert is_valid
        assert "matches reference" in message

    def test_validate_mismatched_cutoff(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        registry.register_snapshot("retail", "snap_001", datetime(2024, 6, 1), "/data/retail.csv", 5000)

        is_valid, message = registry.validate_cutoff(datetime(2024, 7, 1))
        assert not is_valid
        assert "mismatch" in message
        assert "re-exploration" in message


class TestPointInTimeRegistryOutOfSync:
    def test_get_out_of_sync_datasets(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)

        registry.register_snapshot("retail", "snap_001", datetime(2024, 6, 1), "/data/retail.csv", 5000)
        registry.register_snapshot("bank", "snap_002", datetime(2024, 6, 15), "/data/bank.csv", 3000)
        registry.register_snapshot("telecom", "snap_003", datetime(2024, 6, 1), "/data/telecom.csv", 8000)

        out_of_sync = registry.get_out_of_sync_datasets(datetime(2024, 6, 1))
        assert len(out_of_sync) == 1
        assert "bank" in out_of_sync

    def test_no_out_of_sync_when_all_match(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        cutoff = datetime(2024, 6, 1)

        registry.register_snapshot("retail", "snap_001", cutoff, "/data/retail.csv", 5000)
        registry.register_snapshot("bank", "snap_002", cutoff, "/data/bank.csv", 3000)

        out_of_sync = registry.get_out_of_sync_datasets(cutoff)
        assert len(out_of_sync) == 0


class TestPointInTimeRegistryClear:
    def test_clear_removes_all_snapshots(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        registry.register_snapshot("retail", "snap_001", datetime(2024, 6, 1), "/data/retail.csv", 5000)
        registry.register_snapshot("bank", "snap_002", datetime(2024, 6, 1), "/data/bank.csv", 3000)

        registry.clear_registry()

        assert len(registry.snapshots) == 0
        assert not registry.registry_path.exists()

    def test_clear_then_register_works(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        registry.register_snapshot("retail", "snap_001", datetime(2024, 6, 1), "/data/retail.csv", 5000)
        registry.clear_registry()
        registry.register_snapshot("bank", "snap_002", datetime(2024, 7, 1), "/data/bank.csv", 3000)

        assert len(registry.snapshots) == 1
        assert registry.get_reference_cutoff().date() == datetime(2024, 7, 1).date()


class TestPointInTimeRegistryUpdateAll:
    def test_update_cutoff_for_all_datasets(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        old_cutoff = datetime(2024, 6, 1)
        new_cutoff = datetime(2024, 7, 1)

        registry.register_snapshot("retail", "snap_001", old_cutoff, "/data/retail.csv", 5000)
        registry.register_snapshot("bank", "snap_002", old_cutoff, "/data/bank.csv", 3000)

        affected = registry.update_cutoff_for_all(new_cutoff)

        assert len(affected) == 2
        assert "retail" in affected
        assert "bank" in affected
        assert registry.get_snapshot("retail").cutoff_date == new_cutoff
        assert registry.get_snapshot("bank").cutoff_date == new_cutoff


class TestPointInTimeRegistryPersistence:
    def test_persists_across_instances(self, tmp_path):
        cutoff = datetime(2024, 6, 1)

        registry1 = PointInTimeRegistry(tmp_path)
        registry1.register_snapshot("retail", "snap_001", cutoff, "/data/retail.csv", 5000)

        registry2 = PointInTimeRegistry(tmp_path)
        assert len(registry2.snapshots) == 1
        assert registry2.get_snapshot("retail").cutoff_date == cutoff

    def test_registry_file_created(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        registry.register_snapshot("retail", "snap_001", datetime(2024, 6, 1), "/data/retail.csv", 5000)

        assert registry.registry_path.exists()
        assert registry.registry_path.name == "point_in_time_registry.json"
