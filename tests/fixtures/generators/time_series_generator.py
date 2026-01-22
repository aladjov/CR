"""
Time series fixture generator for customer retention testing.

Generates transaction and email event data that aggregates to metrics
similar to the retail fixture (avgorder, ordfreq, esent, eopenrate, eclickrate).
"""
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TransactionGenerator:
    """Generates transaction events for customers."""

    rng: np.random.Generator
    retail_df: pd.DataFrame

    def generate(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Generate transaction events based on retail customer data."""
        retail = self._prepare_retail_data()
        if sample_size:
            retail = retail.sample(n=min(sample_size, len(retail)), random_state=42)

        transactions = []
        for _, row in retail.iterrows():
            cust_txns = self._generate_customer_transactions(row)
            transactions.extend(cust_txns)

        if not transactions:
            return self._empty_transactions_df()

        df = pd.DataFrame(transactions)
        df["transaction_id"] = [f"TXN{i:08d}" for i in range(len(df))]
        return df[["transaction_id", "customer_id", "transaction_date",
                   "amount", "product_category", "discount_applied"]]

    def _prepare_retail_data(self) -> pd.DataFrame:
        """Parse dates and filter to customers with orders."""
        df = self.retail_df.copy()
        df["firstorder"] = pd.to_datetime(df["firstorder"], errors="coerce")
        df["lastorder"] = pd.to_datetime(df["lastorder"], errors="coerce")
        return df[df["firstorder"].notna() & df["lastorder"].notna()]

    def _generate_customer_transactions(self, row: pd.Series) -> list:
        """Generate transactions for a single customer."""
        cust_id = row["custid"]
        first_order = row["firstorder"]
        last_order = row["lastorder"]
        avg_order = row["avgorder"]
        ord_freq = row["ordfreq"]

        date_range_days = (last_order - first_order).days
        if date_range_days <= 0:
            num_orders = 1
        else:
            num_orders = max(1, int(ord_freq * date_range_days / 30) + 1)

        num_orders = min(num_orders, 50)  # Cap to avoid huge datasets

        transactions = []
        categories = ["Electronics", "Clothing", "Home", "Food", "Beauty"]

        for _ in range(num_orders):
            if date_range_days > 0:
                days_offset = self.rng.integers(0, date_range_days + 1)
            else:
                days_offset = 0

            txn_date = first_order + timedelta(days=int(days_offset))
            amount = max(1.0, self.rng.normal(avg_order, avg_order * 0.3))
            category = self.rng.choice(categories)
            discount = int(self.rng.random() < 0.2)

            transactions.append({
                "customer_id": cust_id,
                "transaction_date": txn_date.strftime("%Y-%m-%d"),
                "amount": round(amount, 2),
                "product_category": category,
                "discount_applied": discount,
            })

        return transactions

    def _empty_transactions_df(self) -> pd.DataFrame:
        """Return empty DataFrame with correct schema."""
        return pd.DataFrame(columns=[
            "transaction_id", "customer_id", "transaction_date",
            "amount", "product_category", "discount_applied"
        ])


@dataclass
class EmailGenerator:
    """Generates email events for customers."""

    rng: np.random.Generator
    retail_df: pd.DataFrame

    def generate(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Generate email events based on retail customer data."""
        retail = self._prepare_retail_data()
        if sample_size:
            retail = retail.sample(n=min(sample_size, len(retail)), random_state=42)

        emails = []
        for _, row in retail.iterrows():
            cust_emails = self._generate_customer_emails(row)
            emails.extend(cust_emails)

        if not emails:
            return self._empty_emails_df()

        df = pd.DataFrame(emails)
        df["email_id"] = [f"EML{i:08d}" for i in range(len(df))]
        return df[["email_id", "customer_id", "sent_date",
                   "campaign_type", "opened", "clicked"]]

    def _prepare_retail_data(self) -> pd.DataFrame:
        """Parse dates and filter to customers with email data."""
        df = self.retail_df.copy()
        df["created"] = pd.to_datetime(df["created"], errors="coerce")
        df["lastorder"] = pd.to_datetime(df["lastorder"], errors="coerce")
        return df[df["esent"] > 0]

    def _generate_customer_emails(self, row: pd.Series) -> list:
        """Generate emails for a single customer."""
        cust_id = row["custid"]
        esent = int(row["esent"])
        open_rate = row["eopenrate"] / 100.0
        click_rate = row["eclickrate"] / 100.0

        created = row["created"]
        last_order = row["lastorder"]
        if pd.isna(created) or pd.isna(last_order):
            start_date = pd.Timestamp("2010-01-01")
            end_date = pd.Timestamp("2014-01-31")
        else:
            start_date = created
            end_date = last_order + timedelta(days=30)

        date_range_days = max(1, (end_date - start_date).days)

        emails = []
        campaign_types = ["Newsletter", "Promotion", "Welcome", "Reminder", "Survey"]

        for _ in range(esent):
            days_offset = self.rng.integers(0, date_range_days + 1)
            sent_date = start_date + timedelta(days=int(days_offset))
            campaign = self.rng.choice(campaign_types)

            opened = int(self.rng.random() < open_rate)
            clicked = int(opened and self.rng.random() < (click_rate / max(open_rate, 0.01)))

            emails.append({
                "customer_id": cust_id,
                "sent_date": sent_date.strftime("%Y-%m-%d"),
                "campaign_type": campaign,
                "opened": opened,
                "clicked": clicked,
            })

        return emails

    def _empty_emails_df(self) -> pd.DataFrame:
        """Return empty DataFrame with correct schema."""
        return pd.DataFrame(columns=[
            "email_id", "customer_id", "sent_date",
            "campaign_type", "opened", "clicked"
        ])


class TimeSeriesGenerator:
    """Main generator combining transaction and email generation."""

    def __init__(self, retail_df: pd.DataFrame, seed: int = 42):
        self.retail_df = retail_df
        self.rng = np.random.default_rng(seed)
        self.customer_ids = retail_df["custid"].tolist()

        self._txn_gen = TransactionGenerator(rng=self.rng, retail_df=retail_df)
        self._email_gen = EmailGenerator(rng=self.rng, retail_df=retail_df)

    def generate_transactions(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Generate transaction time series data."""
        return self._txn_gen.generate(sample_size)

    def generate_emails(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Generate email time series data."""
        return self._email_gen.generate(sample_size)

    def generate_all(self, output_dir: str) -> dict:
        """Generate all fixtures and save to output directory."""
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        txn = self.generate_transactions()
        txn.to_csv(out / "customer_transactions.csv", index=False)

        emails = self.generate_emails()
        emails.to_csv(out / "customer_emails.csv", index=False)

        return {
            "transactions": len(txn),
            "emails": len(emails),
        }


if __name__ == "__main__":
    # Generate fixtures when run directly
    from pathlib import Path

    fixtures_dir = Path(__file__).parent.parent
    retail_df = pd.read_csv(fixtures_dir / "customer_retention_retail.csv")

    generator = TimeSeriesGenerator(retail_df)
    stats = generator.generate_all(str(fixtures_dir))

    print(f"Generated {stats['transactions']:,} transactions")
    print(f"Generated {stats['emails']:,} emails")
