from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from customer_retention.analysis.auto_explorer.dataset_context import (
    DatasetContext,
    DatasetContextScanner,
    DatasetEntry,
    mount_target_column,
)

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"
TINY_PROFILES = FIXTURES / "3set_tiny_customer_profiles.csv"
TINY_TRANSACTIONS = FIXTURES / "3set_tiny_edi_transactions.csv"
TINY_TICKETS = FIXTURES / "3set_tiny_support_tickets.csv"


class TestDatasetContextDataModel:
    def test_create_dataset_entry_with_target(self):
        entry = DatasetEntry(
            path="data/profiles.csv", has_target=True,
            target_column="churned", entity_column="customer_id",
        )
        assert entry.has_target is True
        assert entry.target_column == "churned"
        assert entry.entity_column == "customer_id"

    def test_create_dataset_entry_without_target(self):
        entry = DatasetEntry(
            path="data/transactions.csv", has_target=False,
            join_key="customer_id", join_to="profiles",
            relationship="many_to_one",
        )
        assert entry.has_target is False
        assert entry.target_column is None
        assert entry.join_key == "customer_id"
        assert entry.join_to == "profiles"
        assert entry.relationship == "many_to_one"

    def test_create_dataset_context(self):
        ctx = DatasetContext(
            target_dataset="profiles",
            target_column="churned",
            entity_column="customer_id",
            datasets={
                "profiles": DatasetEntry(path="p.csv", has_target=True, target_column="churned", entity_column="customer_id"),
                "transactions": DatasetEntry(path="t.csv", has_target=False, join_key="customer_id", join_to="profiles"),
            },
        )
        assert ctx.target_dataset == "profiles"
        assert len(ctx.datasets) == 2
        assert ctx.datasets["profiles"].has_target is True
        assert ctx.datasets["transactions"].has_target is False


class TestDatasetContextSerialization:
    def _make_context(self):
        return DatasetContext(
            target_dataset="profiles",
            target_column="churned",
            entity_column="customer_id",
            datasets={
                "profiles": DatasetEntry(path="p.csv", has_target=True, target_column="churned", entity_column="customer_id"),
                "transactions": DatasetEntry(path="t.csv", has_target=False, join_key="customer_id", join_to="profiles", relationship="many_to_one"),
            },
        )

    def test_save_to_yaml(self, tmp_path):
        ctx = self._make_context()
        out = tmp_path / "dataset_context.yaml"
        ctx.save(out)
        assert out.exists()
        data = yaml.safe_load(out.read_text())
        assert data["target_dataset"] == "profiles"
        assert data["target_column"] == "churned"
        assert "profiles" in data["datasets"]
        assert "transactions" in data["datasets"]

    def test_load_from_yaml(self, tmp_path):
        ctx = self._make_context()
        out = tmp_path / "dataset_context.yaml"
        ctx.save(out)
        loaded = DatasetContext.load(out)
        assert loaded.target_dataset == "profiles"
        assert loaded.target_column == "churned"
        assert loaded.entity_column == "customer_id"
        assert loaded.datasets["profiles"].has_target is True
        assert loaded.datasets["transactions"].join_key == "customer_id"

    def test_round_trip_with_three_datasets(self, tmp_path):
        ctx = DatasetContext(
            target_dataset="profiles",
            target_column="churned",
            entity_column="customer_id",
            datasets={
                "profiles": DatasetEntry(path="p.csv", has_target=True, target_column="churned", entity_column="customer_id"),
                "transactions": DatasetEntry(path="t.csv", has_target=False, join_key="customer_id", join_to="profiles", relationship="many_to_one"),
                "tickets": DatasetEntry(path="k.csv", has_target=False, join_key="customer_id", join_to="profiles", relationship="many_to_one"),
            },
        )
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = DatasetContext.load(out)
        assert len(loaded.datasets) == 3
        for name in ("profiles", "transactions", "tickets"):
            assert loaded.datasets[name].path == ctx.datasets[name].path
            assert loaded.datasets[name].has_target == ctx.datasets[name].has_target
            assert loaded.datasets[name].join_key == ctx.datasets[name].join_key

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DatasetContext.load(tmp_path / "nope.yaml")


class TestDatasetContextScanner:
    def _scan_tiny(self):
        return DatasetContextScanner().scan([TINY_PROFILES, TINY_TRANSACTIONS, TINY_TICKETS])

    def test_scan_returns_dataset_context(self):
        ctx = self._scan_tiny()
        assert isinstance(ctx, DatasetContext)

    def test_scan_detects_target_dataset(self):
        ctx = self._scan_tiny()
        assert ctx.target_dataset == "3set_tiny_customer_profiles"

    def test_scan_detects_target_column(self):
        ctx = self._scan_tiny()
        assert ctx.target_column == "churned"

    def test_scan_detects_entity_column(self):
        ctx = self._scan_tiny()
        assert ctx.entity_column == "customer_id"

    def test_scan_marks_target_dataset_entry(self):
        ctx = self._scan_tiny()
        assert ctx.datasets["3set_tiny_customer_profiles"].has_target is True

    def test_scan_marks_non_target_datasets(self):
        ctx = self._scan_tiny()
        assert ctx.datasets["3set_tiny_edi_transactions"].has_target is False
        assert ctx.datasets["3set_tiny_support_tickets"].has_target is False

    def test_scan_detects_join_keys(self):
        ctx = self._scan_tiny()
        assert ctx.datasets["3set_tiny_edi_transactions"].join_key == "customer_id"
        assert ctx.datasets["3set_tiny_support_tickets"].join_key == "customer_id"

    def test_scan_detects_relationship_type(self):
        ctx = self._scan_tiny()
        tx_rel = ctx.datasets["3set_tiny_edi_transactions"].relationship
        assert tx_rel in ("one_to_one", "many_to_one")

    def test_scan_single_dataset_with_target(self):
        ctx = DatasetContextScanner().scan([TINY_PROFILES])
        assert ctx.target_dataset == "3set_tiny_customer_profiles"
        assert ctx.target_column == "churned"

    def test_scan_single_dataset_no_target(self):
        ctx = DatasetContextScanner().scan([TINY_TRANSACTIONS])
        assert ctx.target_dataset is None

    def test_scan_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            DatasetContextScanner().scan([Path("/nonexistent/file.csv")])


class TestMountTargetColumn:
    def _scan_context(self):
        return DatasetContextScanner().scan([TINY_PROFILES, TINY_TRANSACTIONS, TINY_TICKETS])

    def test_mount_adds_target_column(self):
        import pandas as pd
        ctx = self._scan_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        result, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert "churned" in result.columns

    def test_mount_returns_dataframe_and_info(self):
        import pandas as pd
        ctx = self._scan_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        result, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert hasattr(result, "columns")
        assert isinstance(info, dict)

    def test_mount_info_contains_source(self):
        import pandas as pd
        ctx = self._scan_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        _, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert "target_mounted_from" in info

    def test_mount_info_contains_key(self):
        import pandas as pd
        ctx = self._scan_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        _, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert "target_mount_key" in info

    def test_mount_preserves_original_columns(self):
        import pandas as pd
        ctx = self._scan_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        original_cols = list(df.columns)
        result, _ = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        for col in original_cols:
            assert col in result.columns

    def test_mount_preserves_row_count(self):
        import pandas as pd
        ctx = self._scan_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        result, _ = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert len(result) == len(df)

    def test_mount_target_values_correct(self):
        import pandas as pd
        ctx = self._scan_context()
        df = pd.read_csv(TINY_TRANSACTIONS)
        result, _ = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        c1_row = result[result["customer_id"] == "C00001"].iloc[0]
        c2_row = result[result["customer_id"] == "C00002"].iloc[0]
        assert c1_row["churned"] == 0
        assert c2_row["churned"] == 1

    def test_mount_no_context_returns_none(self):
        import pandas as pd
        ctx = DatasetContext()
        df = pd.read_csv(TINY_TRANSACTIONS)
        result, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert info is None

    def test_mount_dataset_already_has_target(self):
        import pandas as pd
        ctx = self._scan_context()
        df = pd.read_csv(TINY_PROFILES)
        result, info = mount_target_column(df, ctx, "3set_tiny_customer_profiles")
        assert info is None

    def test_mount_no_join_key_returns_none(self):
        import pandas as pd
        ctx = self._scan_context()
        ctx.datasets["3set_tiny_edi_transactions"].join_key = None
        df = pd.read_csv(TINY_TRANSACTIONS)
        result, info = mount_target_column(df, ctx, "3set_tiny_edi_transactions")
        assert info is None


class TestDatasetContextCutoffDate:
    def test_cutoff_date_default_none(self):
        ctx = DatasetContext()
        assert ctx.cutoff_date is None

    def test_cutoff_date_saved_to_yaml(self, tmp_path):
        ctx = DatasetContext(cutoff_date=datetime(2024, 6, 1))
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        data = yaml.safe_load(out.read_text())
        assert "cutoff_date" in data
        assert data["cutoff_date"] == "2024-06-01T00:00:00"

    def test_cutoff_date_loaded_from_yaml(self, tmp_path):
        ctx = DatasetContext(cutoff_date=datetime(2024, 6, 1))
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = DatasetContext.load(out)
        assert loaded.cutoff_date == datetime(2024, 6, 1)

    def test_cutoff_date_round_trip(self, tmp_path):
        dt = datetime(2024, 3, 15, 10, 30, 0)
        ctx = DatasetContext(cutoff_date=dt)
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = DatasetContext.load(out)
        assert loaded.cutoff_date == dt

    def test_cutoff_date_none_omitted_from_yaml(self, tmp_path):
        ctx = DatasetContext()
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        data = yaml.safe_load(out.read_text())
        assert "cutoff_date" not in data

    def test_cutoff_date_tz_aware_stored_as_naive(self, tmp_path):
        tz_dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        ctx = DatasetContext(cutoff_date=tz_dt)
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = DatasetContext.load(out)
        assert loaded.cutoff_date == datetime(2024, 6, 1)
        assert loaded.cutoff_date.tzinfo is None


class TestDatasetContextForSingleDataset:
    def test_creates_single_entry_context(self):
        ctx = DatasetContext.for_single_dataset(
            path="data/profiles.csv", dataset_name="profiles",
        )
        assert len(ctx.datasets) == 1
        assert "profiles" in ctx.datasets

    def test_target_dataset_set(self):
        ctx = DatasetContext.for_single_dataset(
            path="data/profiles.csv", dataset_name="profiles",
            target_column="churned",
        )
        assert ctx.target_dataset == "profiles"
        assert ctx.target_column == "churned"

    def test_entity_column_set(self):
        ctx = DatasetContext.for_single_dataset(
            path="data/profiles.csv", dataset_name="profiles",
            entity_column="customer_id",
        )
        assert ctx.entity_column == "customer_id"

    def test_has_target_true_when_target_given(self):
        ctx = DatasetContext.for_single_dataset(
            path="data/profiles.csv", dataset_name="profiles",
            target_column="churned",
        )
        assert ctx.datasets["profiles"].has_target is True

    def test_has_target_false_when_no_target(self):
        ctx = DatasetContext.for_single_dataset(
            path="data/profiles.csv", dataset_name="profiles",
        )
        assert ctx.datasets["profiles"].has_target is False

    def test_cutoff_date_stored(self):
        dt = datetime(2024, 6, 1)
        ctx = DatasetContext.for_single_dataset(
            path="data/profiles.csv", dataset_name="profiles",
            cutoff_date=dt,
        )
        assert ctx.cutoff_date == dt


class TestDatasetContextJoinDatasetName:
    def test_default_none(self):
        ctx = DatasetContext()
        assert ctx.join_dataset_name is None

    def test_saved_to_yaml(self, tmp_path):
        ctx = DatasetContext(join_dataset_name="edi_ticketing")
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        data = yaml.safe_load(out.read_text())
        assert data["join_dataset_name"] == "edi_ticketing"

    def test_loaded_from_yaml(self, tmp_path):
        ctx = DatasetContext(join_dataset_name="edi_ticketing")
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = DatasetContext.load(out)
        assert loaded.join_dataset_name == "edi_ticketing"

    def test_round_trip(self, tmp_path):
        ctx = DatasetContext(
            target_dataset="profiles",
            target_column="churned",
            join_dataset_name="retail_churn",
            datasets={
                "profiles": DatasetEntry(path="p.csv", has_target=True, target_column="churned"),
            },
        )
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = DatasetContext.load(out)
        assert loaded.join_dataset_name == "retail_churn"
        assert loaded.target_dataset == "profiles"

    def test_none_omitted_from_yaml(self, tmp_path):
        ctx = DatasetContext()
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        data = yaml.safe_load(out.read_text())
        assert "join_dataset_name" not in data

    def test_for_single_dataset_accepts_join_name(self):
        ctx = DatasetContext.for_single_dataset(
            path="data/profiles.csv", dataset_name="profiles",
            join_dataset_name="retail_churn",
        )
        assert ctx.join_dataset_name == "retail_churn"

    def test_backward_compat_load_without_field(self, tmp_path):
        out = tmp_path / "ctx.yaml"
        out.write_text(yaml.dump({
            "target_dataset": "profiles",
            "target_column": "churned",
            "entity_column": "customer_id",
            "datasets": {"profiles": {"path": "p.csv", "has_target": True}},
        }))
        loaded = DatasetContext.load(out)
        assert loaded.join_dataset_name is None
