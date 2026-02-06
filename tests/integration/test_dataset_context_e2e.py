from pathlib import Path

import pandas as pd

from customer_retention.analysis.auto_explorer.dataset_context import (
    DatasetContext,
    DatasetContextScanner,
    mount_target_column,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
TINY_PROFILES = FIXTURES / "3set_tiny_customer_profiles.csv"
TINY_TRANSACTIONS = FIXTURES / "3set_tiny_edi_transactions.csv"
TINY_TICKETS = FIXTURES / "3set_tiny_support_tickets.csv"


def _scan_all():
    return DatasetContextScanner().scan([TINY_PROFILES, TINY_TRANSACTIONS, TINY_TICKETS])


class TestScanToMountEndToEnd:
    def test_scan_three_datasets_and_save_context(self, tmp_path):
        ctx = _scan_all()
        out = tmp_path / "dataset_context.yaml"
        ctx.save(out)
        assert out.exists()
        loaded = DatasetContext.load(out)
        assert len(loaded.datasets) == 3

    def test_full_flow_scan_save_load_mount_transactions(self, tmp_path):
        ctx = _scan_all()
        out = tmp_path / "dataset_context.yaml"
        ctx.save(out)
        loaded = DatasetContext.load(out)
        df = pd.read_csv(TINY_TRANSACTIONS)
        result, info = mount_target_column(df, loaded, "3set_tiny_edi_transactions")
        assert "churned" in result.columns
        c1 = result[result["customer_id"] == "C00001"].iloc[0]
        c2 = result[result["customer_id"] == "C00002"].iloc[0]
        assert c1["churned"] == 0
        assert c2["churned"] == 1

    def test_full_flow_scan_save_load_mount_tickets(self, tmp_path):
        ctx = _scan_all()
        out = tmp_path / "dataset_context.yaml"
        ctx.save(out)
        loaded = DatasetContext.load(out)
        df = pd.read_csv(TINY_TICKETS)
        result, info = mount_target_column(df, loaded, "3set_tiny_support_tickets")
        assert "churned" in result.columns
        c1 = result[result["customer_id"] == "C00001"].iloc[0]
        c2 = result[result["customer_id"] == "C00002"].iloc[0]
        assert c1["churned"] == 0
        assert c2["churned"] == 1

    def test_mount_preserves_event_rows(self):
        ctx = _scan_all()
        df_tx = pd.read_csv(TINY_TRANSACTIONS)
        result_tx, _ = mount_target_column(df_tx, ctx, "3set_tiny_edi_transactions")
        assert len(result_tx) == 2

        df_tk = pd.read_csv(TINY_TICKETS)
        result_tk, _ = mount_target_column(df_tk, ctx, "3set_tiny_support_tickets")
        assert len(result_tk) == 2

    def test_target_dataset_not_mounted_onto_itself(self):
        ctx = _scan_all()
        df = pd.read_csv(TINY_PROFILES)
        result, info = mount_target_column(df, ctx, "3set_tiny_customer_profiles")
        assert info is None

    def test_context_yaml_survives_round_trip_with_real_data(self, tmp_path):
        ctx = _scan_all()
        out = tmp_path / "ctx.yaml"
        ctx.save(out)
        loaded = DatasetContext.load(out)
        assert loaded.target_dataset == ctx.target_dataset
        assert loaded.target_column == ctx.target_column
        assert loaded.entity_column == ctx.entity_column
        for name in ctx.datasets:
            orig = ctx.datasets[name]
            rnd = loaded.datasets[name]
            assert rnd.has_target == orig.has_target
            assert rnd.join_key == orig.join_key
            assert rnd.join_to == orig.join_to
            assert rnd.relationship == orig.relationship


class TestScanDetectsRealDataRelationships:
    def test_transactions_join_key_is_customer_id(self):
        ctx = _scan_all()
        assert ctx.datasets["3set_tiny_edi_transactions"].join_key == "customer_id"

    def test_tickets_join_key_is_customer_id(self):
        ctx = _scan_all()
        assert ctx.datasets["3set_tiny_support_tickets"].join_key == "customer_id"

    def test_profiles_identified_as_target_holder(self):
        ctx = _scan_all()
        assert ctx.target_dataset == "3set_tiny_customer_profiles"

    def test_target_column_is_churned(self):
        ctx = _scan_all()
        assert ctx.target_column == "churned"

    def test_entity_column_is_customer_id(self):
        ctx = _scan_all()
        assert ctx.entity_column == "customer_id"

    def test_transactions_relationship_type(self):
        ctx = _scan_all()
        rel = ctx.datasets["3set_tiny_edi_transactions"].relationship
        assert rel in ("one_to_one", "many_to_one")

    def test_profiles_entry_has_target_true(self):
        ctx = _scan_all()
        assert ctx.datasets["3set_tiny_customer_profiles"].has_target is True

    def test_transactions_entry_has_target_false(self):
        ctx = _scan_all()
        assert ctx.datasets["3set_tiny_edi_transactions"].has_target is False

    def test_tickets_entry_has_target_false(self):
        ctx = _scan_all()
        assert ctx.datasets["3set_tiny_support_tickets"].has_target is False
