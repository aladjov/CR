from datetime import datetime

import pytest

from customer_retention.stages.temporal import PointInTimeRegistry
from customer_retention.stages.temporal.snapshot_manager import (
    compute_composite_dataset_name,
    require_consistent_cutoffs,
)


class TestRequireConsistentCutoffsValidation:
    def test_raises_when_dataset_not_registered(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        registry.register_snapshot("customers", "snap_001", datetime(2024, 6, 1), "/data/customers.csv", 5000)

        with pytest.raises(ValueError, match="not registered.*transactions"):
            require_consistent_cutoffs(registry, ["customers", "transactions"])

    def test_raises_when_no_datasets_registered(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)

        with pytest.raises(ValueError, match="not registered"):
            require_consistent_cutoffs(registry, ["customers"])

    def test_raises_on_empty_dataset_names(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)

        with pytest.raises(ValueError, match="at least one"):
            require_consistent_cutoffs(registry, [])


class TestRequireConsistentCutoffsEnforcement:
    def test_raises_on_inconsistent_cutoff_dates(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        registry.register_snapshot("customers", "snap_001", datetime(2024, 6, 1), "/data/customers.csv", 5000)
        registry.register_snapshot("transactions", "snap_002", datetime(2024, 7, 15), "/data/transactions.csv", 50000)

        with pytest.raises(ValueError, match="Inconsistent cutoff"):
            require_consistent_cutoffs(registry, ["customers", "transactions"])

    def test_inconsistent_error_lists_out_of_sync_datasets(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        registry.register_snapshot("customers", "snap_001", datetime(2024, 6, 1), "/data/customers.csv", 5000)
        registry.register_snapshot("transactions", "snap_002", datetime(2024, 7, 1), "/data/transactions.csv", 50000)

        with pytest.raises(ValueError, match="transactions"):
            require_consistent_cutoffs(registry, ["customers", "transactions"])

    def test_same_date_different_time_is_consistent(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        registry.register_snapshot("customers", "snap_001", datetime(2024, 6, 1, 10, 0), "/data/customers.csv", 5000)
        registry.register_snapshot("transactions", "snap_002", datetime(2024, 6, 1, 14, 30), "/data/transactions.csv", 50000)

        result = require_consistent_cutoffs(registry, ["customers", "transactions"])
        assert result.startswith("combined_")


class TestRequireConsistentCutoffsResult:
    def test_returns_composite_name_when_consistent(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        cutoff = datetime(2024, 6, 1)
        registry.register_snapshot("customers", "snap_001", cutoff, "/data/customers.csv", 5000)
        registry.register_snapshot("transactions", "snap_002", cutoff, "/data/transactions.csv", 50000)

        result = require_consistent_cutoffs(registry, ["customers", "transactions"])
        expected = compute_composite_dataset_name(["customers", "transactions"])
        assert result == expected

    def test_single_dataset_returns_composite_name(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        registry.register_snapshot("customers", "snap_001", datetime(2024, 6, 1), "/data/customers.csv", 5000)

        result = require_consistent_cutoffs(registry, ["customers"])
        expected = compute_composite_dataset_name(["customers"])
        assert result == expected

    def test_checks_only_specified_datasets(self, tmp_path):
        registry = PointInTimeRegistry(tmp_path)
        cutoff = datetime(2024, 6, 1)
        registry.register_snapshot("customers", "snap_001", cutoff, "/data/customers.csv", 5000)
        registry.register_snapshot("transactions", "snap_002", cutoff, "/data/transactions.csv", 50000)
        registry.register_snapshot("unrelated", "snap_003", datetime(2024, 9, 1), "/data/unrelated.csv", 1000)

        result = require_consistent_cutoffs(registry, ["customers", "transactions"])
        assert result.startswith("combined_")
