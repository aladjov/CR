"""Tests for time series fixture generator - TDD approach."""
import pandas as pd
import pytest
from pathlib import Path

from tests.fixtures.generators.time_series_generator import (
    TimeSeriesGenerator,
    TransactionGenerator,
    EmailGenerator,
)


FIXTURES_DIR = Path(__file__).parent.parent
RETAIL_FIXTURE = FIXTURES_DIR / "customer_retention_retail.csv"


@pytest.fixture
def retail_df():
    """Load the base retail fixture."""
    return pd.read_csv(RETAIL_FIXTURE)


@pytest.fixture
def generator(retail_df):
    """Create generator seeded with retail data."""
    return TimeSeriesGenerator(retail_df, seed=42)


class TestTimeSeriesGenerator:
    """Tests for the main generator class."""

    def test_loads_customer_ids(self, generator, retail_df):
        """Generator should extract customer IDs from retail data."""
        assert len(generator.customer_ids) == len(retail_df)
        assert set(generator.customer_ids) == set(retail_df["custid"])

    def test_reproducible_with_seed(self, retail_df):
        """Same seed should produce identical results."""
        gen1 = TimeSeriesGenerator(retail_df, seed=42)
        gen2 = TimeSeriesGenerator(retail_df, seed=42)

        txn1 = gen1.generate_transactions(sample_size=100)
        txn2 = gen2.generate_transactions(sample_size=100)

        pd.testing.assert_frame_equal(txn1, txn2)


class TestTransactionGenerator:
    """Tests for transaction time series generation."""

    def test_transaction_schema(self, generator):
        """Transactions should have expected columns."""
        txn = generator.generate_transactions(sample_size=100)

        expected_cols = [
            "transaction_id",
            "customer_id",
            "transaction_date",
            "amount",
            "product_category",
            "discount_applied",
        ]
        assert list(txn.columns) == expected_cols

    def test_transaction_customer_ids_match_retail(self, generator, retail_df):
        """All transaction customer_ids should exist in retail data."""
        txn = generator.generate_transactions(sample_size=1000)
        assert txn["customer_id"].isin(retail_df["custid"]).all()

    def test_transaction_dates_within_customer_range(self, generator, retail_df):
        """Transaction dates should be between firstorder and lastorder."""
        txn = generator.generate_transactions(sample_size=1000)
        txn["transaction_date"] = pd.to_datetime(txn["transaction_date"])

        retail_copy = retail_df.copy()
        retail_copy["firstorder"] = pd.to_datetime(retail_copy["firstorder"], errors="coerce")
        retail_copy["lastorder"] = pd.to_datetime(retail_copy["lastorder"], errors="coerce")
        retail_dates = retail_copy.drop_duplicates("custid").set_index("custid")

        checked = 0
        for cust_id in txn["customer_id"].unique():
            if checked >= 50:
                break
            cust_txns = txn[txn["customer_id"] == cust_id]["transaction_date"]
            if cust_id in retail_dates.index:
                first = retail_dates.loc[cust_id, "firstorder"]
                last = retail_dates.loc[cust_id, "lastorder"]
                if pd.notna(first) and pd.notna(last):
                    assert cust_txns.min() >= first
                    assert cust_txns.max() <= last
                    checked += 1

    def test_transaction_amounts_realistic(self, generator):
        """Transaction amounts should be reasonable (positive, typical retail range)."""
        txn = generator.generate_transactions(sample_size=1000)
        assert (txn["amount"] > 0).all()
        assert txn["amount"].median() < 500  # Reasonable retail range

    def test_aggregated_avgorder_similar_to_retail(self, generator, retail_df):
        """Aggregated mean(amount) should approximate retail avgorder."""
        txn = generator.generate_transactions()

        agg = txn.groupby("customer_id")["amount"].mean().reset_index()
        agg.columns = ["custid", "calc_avgorder"]

        merged = retail_df.merge(agg, on="custid", how="inner")

        # Allow 20% tolerance on median difference
        retail_median = merged["avgorder"].median()
        calc_median = merged["calc_avgorder"].median()
        assert abs(calc_median - retail_median) / retail_median < 0.20

    def test_unique_transaction_ids(self, generator):
        """Each transaction should have unique ID."""
        txn = generator.generate_transactions(sample_size=1000)
        assert txn["transaction_id"].nunique() == len(txn)


class TestEmailGenerator:
    """Tests for email time series generation."""

    def test_email_schema(self, generator):
        """Emails should have expected columns."""
        emails = generator.generate_emails(sample_size=100)

        expected_cols = [
            "email_id",
            "customer_id",
            "sent_date",
            "campaign_type",
            "opened",
            "clicked",
        ]
        assert list(emails.columns) == expected_cols

    def test_email_customer_ids_match_retail(self, generator, retail_df):
        """All email customer_ids should exist in retail data."""
        emails = generator.generate_emails(sample_size=1000)
        assert emails["customer_id"].isin(retail_df["custid"]).all()

    def test_email_opened_clicked_binary(self, generator):
        """Opened and clicked should be 0 or 1."""
        emails = generator.generate_emails(sample_size=1000)
        assert emails["opened"].isin([0, 1]).all()
        assert emails["clicked"].isin([0, 1]).all()

    def test_clicked_implies_opened(self, generator):
        """If clicked=1, then opened should be 1."""
        emails = generator.generate_emails(sample_size=1000)
        clicked_rows = emails[emails["clicked"] == 1]
        assert (clicked_rows["opened"] == 1).all()

    def test_aggregated_esent_similar_to_retail(self, generator, retail_df):
        """Aggregated email count should approximate retail esent."""
        emails = generator.generate_emails()

        agg = emails.groupby("customer_id").size().reset_index(name="calc_esent")
        agg.columns = ["custid", "calc_esent"]

        merged = retail_df[retail_df["esent"] > 0].merge(agg, on="custid", how="inner")

        if len(merged) > 0:
            retail_median = merged["esent"].median()
            calc_median = merged["calc_esent"].median()
            # Allow 30% tolerance
            assert abs(calc_median - retail_median) / retail_median < 0.30

    def test_aggregated_openrate_similar_to_retail(self, generator, retail_df):
        """Aggregated open rate should approximate retail eopenrate."""
        emails = generator.generate_emails()

        agg = emails.groupby("customer_id").agg(
            calc_openrate=("opened", lambda x: x.mean() * 100)
        ).reset_index()
        agg.columns = ["custid", "calc_openrate"]

        merged = retail_df[retail_df["esent"] > 0].merge(agg, on="custid", how="inner")

        if len(merged) > 0:
            retail_median = merged["eopenrate"].median()
            calc_median = merged["calc_openrate"].median()
            # Allow 30% tolerance
            if retail_median > 0:
                assert abs(calc_median - retail_median) / retail_median < 0.30

    def test_unique_email_ids(self, generator):
        """Each email should have unique ID."""
        emails = generator.generate_emails(sample_size=1000)
        assert emails["email_id"].nunique() == len(emails)


class TestFullGeneration:
    """Integration tests for full fixture generation."""

    def test_generate_and_save_transactions(self, generator, tmp_path):
        """Should generate and save transaction CSV."""
        output = tmp_path / "transactions.csv"
        txn = generator.generate_transactions(sample_size=100)  # 100 customers
        txn.to_csv(output, index=False)

        loaded = pd.read_csv(output)
        assert len(loaded) > 0
        assert len(loaded) == len(txn)
        assert list(loaded.columns) == list(txn.columns)
        # Verify sample_size controls customer count, not row count
        assert loaded["customer_id"].nunique() <= 100

    def test_generate_and_save_emails(self, generator, tmp_path):
        """Should generate and save email CSV."""
        output = tmp_path / "emails.csv"
        emails = generator.generate_emails(sample_size=100)  # 100 customers
        emails.to_csv(output, index=False)

        loaded = pd.read_csv(output)
        assert len(loaded) > 0
        assert len(loaded) == len(emails)
        assert list(loaded.columns) == list(emails.columns)
        # Verify sample_size controls customer count, not row count
        assert loaded["customer_id"].nunique() <= 100
