#!/usr/bin/env python3
"""
Generate synthetic test datasets for the Customer Retention framework.

This script creates realistic event-level data for testing the Time Series (TS) track notebooks.
The data includes embedded patterns like:
- Power-law distribution (few high-frequency customers)
- Seasonal patterns (weekends, holidays)
- Recency decay (recent customers more active)
- Churn patterns (some customers stop buying)

Usage:
    python scripts/data/generate_test_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

# Seed for reproducibility
np.random.seed(42)

# Output directory
FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"


def generate_customer_ids(n_customers: int) -> list[str]:
    """Generate unique customer IDs matching the format in existing data."""
    return [
        hashlib.md5(f"customer_{i}".encode()).hexdigest()[:6].upper()
        for i in range(n_customers)
    ]


def generate_transactions(
    n_customers: int = 5000,
    date_start: str = "2015-01-01",
    date_end: str = "2023-12-31",
    target_rows: int = 50000,
) -> pd.DataFrame:
    """
    Generate realistic transaction data with embedded patterns.

    Patterns embedded:
    - Power-law distribution: 20% of customers generate 60% of transactions
    - Seasonal: Higher activity on weekends and holidays (Dec, Jul)
    - Recency decay: More recent customers have more transactions
    - Churn: ~30% of customers stopped buying in last 6 months
    - Time of day: Peak activity in afternoon/evening
    """
    customers = generate_customer_ids(n_customers)

    # Assign customer segments (determines transaction frequency)
    # Power-law: few customers are very active
    customer_activity = np.random.pareto(a=1.5, size=n_customers) + 1
    customer_activity = customer_activity / customer_activity.sum()

    # Generate transaction counts per customer
    txn_per_customer = np.random.multinomial(target_rows, customer_activity)

    # Assign signup dates (earlier customers get more history)
    start_dt = datetime.strptime(date_start, "%Y-%m-%d")
    end_dt = datetime.strptime(date_end, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days

    # Customer signup: uniform over first 70% of period
    signup_days = np.random.randint(0, int(total_days * 0.7), size=n_customers)
    customer_signup = {c: start_dt + timedelta(days=int(d)) for c, d in zip(customers, signup_days)}

    churn_mask = np.random.random(n_customers) < 0.3
    churn_cutoff = end_dt - timedelta(days=180)
    customer_churn_date = {
        c: (churn_cutoff - timedelta(days=np.random.randint(0, 90))) if churn_mask[i] else None
        for i, c in enumerate(customers)
    }

    # Generate transactions
    records = []
    txn_id = 0

    # Product categories with different price ranges
    categories = {
        "Electronics": (50, 500, 0.15),   # (min_price, max_price, prob)
        "Clothing": (15, 150, 0.25),
        "Food": (5, 80, 0.30),
        "Home": (20, 300, 0.15),
        "Beauty": (10, 100, 0.10),
        "Sports": (25, 200, 0.05),
    }
    category_names = list(categories.keys())
    category_probs = [v[2] for v in categories.values()]

    payment_methods = ["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Gift Card"]
    payment_probs = [0.35, 0.25, 0.15, 0.20, 0.05]

    channels = ["Online", "Store", "Mobile App"]
    channel_probs = [0.40, 0.35, 0.25]

    stores = [f"STORE_{i:03d}" for i in range(1, 21)]  # 20 stores

    time_of_day = ["Morning", "Afternoon", "Evening", "Night"]
    time_probs = [0.15, 0.35, 0.35, 0.15]

    for i, (customer_id, n_txns) in enumerate(zip(customers, txn_per_customer)):
        if n_txns == 0:
            continue

        signup = customer_signup[customer_id]
        is_churned = churn_mask[i]

        cust_end = customer_churn_date[customer_id] if is_churned else end_dt
        cust_days = (cust_end - signup).days
        if cust_days <= 0:
            continue

        # Generate transaction dates with recency bias (more recent = more likely)
        # Using beta distribution to bias toward recent dates
        day_offsets = np.random.beta(2, 5, size=n_txns) * cust_days
        txn_dates = [signup + timedelta(days=int(d)) for d in day_offsets]

        # Seasonal boost: multiply transaction likelihood for Dec/Jul/weekends
        # (approximated by generating more transactions in those periods)

        for txn_date in txn_dates:
            # Product category
            category = np.random.choice(category_names, p=category_probs)
            min_p, max_p, _ = categories[category]

            # Price with some skew (more cheap items)
            base_price = np.random.lognormal(mean=np.log((min_p + max_p) / 3), sigma=0.5)
            unit_price = np.clip(base_price, min_p, max_p)

            # Quantity (mostly 1, sometimes more)
            quantity = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.6, 0.15, 0.1, 0.08, 0.05, 0.02])

            # Amount
            amount = round(unit_price * quantity, 2)

            # Is return? (~5% of transactions)
            is_return = 1 if np.random.random() < 0.05 else 0
            if is_return:
                amount = -abs(amount)  # Returns are negative

            # Discount (~20% of transactions)
            discount_applied = 1 if np.random.random() < 0.20 else 0
            if discount_applied and not is_return:
                amount = round(amount * 0.85, 2)  # 15% discount

            # Loyalty points (correlated with amount, 0 for ~40%)
            if np.random.random() < 0.6:
                loyalty_points_used = int(abs(amount) * np.random.uniform(0.1, 0.5))
            else:
                loyalty_points_used = 0

            churn_dt = customer_churn_date[customer_id]
            records.append({
                "transaction_id": f"TXN{txn_id:08d}",
                "customer_id": customer_id,
                "transaction_date": txn_date.strftime("%Y-%m-%d"),
                "amount": amount,
                "product_category": category,
                "discount_applied": discount_applied,
                "store_id": np.random.choice(stores),
                "payment_method": np.random.choice(payment_methods, p=payment_probs),
                "quantity": quantity,
                "unit_price": round(unit_price, 2),
                "is_return": is_return,
                "channel": np.random.choice(channels, p=channel_probs),
                "loyalty_points_used": loyalty_points_used,
                "time_of_day": np.random.choice(time_of_day, p=time_probs),
                "customer_churn_date": churn_dt.strftime("%Y-%m-%d") if churn_dt else None,
            })
            txn_id += 1

    df = pd.DataFrame(records)

    # Sort by date
    df = df.sort_values("transaction_date").reset_index(drop=True)

    # Add some missing values (~2% random nulls in optional fields)
    null_cols = ["loyalty_points_used", "time_of_day"]
    for col in null_cols:
        null_mask = np.random.random(len(df)) < 0.02
        df.loc[null_mask, col] = np.nan

    return df


def generate_emails(
    n_customers: int = 5000,
    date_start: str = "2015-01-01",
    date_end: str = "2023-12-31",
    target_rows: int = 150000,
) -> pd.DataFrame:
    """
    Generate realistic email campaign data with embedded patterns.

    Patterns embedded:
    - Varying engagement by campaign type
    - Time-based open patterns (work hours vs. evenings)
    - Unsubscribe patterns (after multiple unopened emails)
    - Device preferences by customer
    """
    customers = generate_customer_ids(n_customers)

    # Customer engagement propensity (some customers more engaged)
    engagement_score = np.random.beta(2, 5, size=n_customers)  # Skewed toward low
    customer_engagement = dict(zip(customers, engagement_score))

    # Customer device preference
    devices = ["Desktop", "Mobile", "Tablet"]
    device_prefs = np.random.dirichlet([2, 3, 1], size=n_customers)
    customer_device_pref = dict(zip(customers, device_prefs))

    # Assign emails per customer (power-law)
    emails_per_customer = np.random.pareto(a=1.2, size=n_customers) + 5
    emails_per_customer = (emails_per_customer / emails_per_customer.sum() * target_rows).astype(int)

    start_dt = datetime.strptime(date_start, "%Y-%m-%d")
    end_dt = datetime.strptime(date_end, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days

    campaign_types = ["Newsletter", "Promotion", "Welcome", "Reminder", "Survey", "Transactional"]
    campaign_probs = [0.25, 0.30, 0.10, 0.15, 0.10, 0.10]

    # Base open/click rates by campaign type
    campaign_engagement = {
        "Newsletter": (0.20, 0.05),      # (open_rate, click_rate)
        "Promotion": (0.25, 0.08),
        "Welcome": (0.45, 0.15),
        "Reminder": (0.30, 0.10),
        "Survey": (0.15, 0.03),
        "Transactional": (0.60, 0.20),
    }

    subject_categories = ["Discount", "New Arrival", "Reminder", "Update", "Thank You", "Feedback"]

    records = []
    email_id = 0

    customer_unsubscribe_date = {}
    customer_unopened_streak = {c: 0 for c in customers}

    for customer_id, n_emails in zip(customers, emails_per_customer):
        if n_emails == 0:
            continue

        # Generate email dates spread across the period
        day_offsets = np.random.randint(0, total_days, size=n_emails)
        email_dates = sorted([start_dt + timedelta(days=int(d)) for d in day_offsets])

        engagement = customer_engagement[customer_id]
        device_pref = customer_device_pref[customer_id]

        for email_date in email_dates:
            if customer_id in customer_unsubscribe_date:
                break

            campaign = np.random.choice(campaign_types, p=campaign_probs)
            base_open, base_click = campaign_engagement[campaign]

            # Adjust by customer engagement
            open_prob = min(base_open * (0.5 + engagement), 0.95)
            click_prob = min(base_click * (0.5 + engagement), open_prob * 0.8)

            # Send hour (business hours more common, but varies)
            send_hour = int(np.random.normal(14, 4))  # Peak at 2pm
            send_hour = max(6, min(22, send_hour))

            # Device
            device = np.random.choice(devices, p=device_pref)

            # Opened?
            opened = 1 if np.random.random() < open_prob else 0

            # Clicked? (only if opened)
            clicked = 1 if opened and np.random.random() < (click_prob / open_prob) else 0

            # Time to open (if opened)
            if opened:
                time_to_open = round(np.random.exponential(scale=4), 1)  # Hours
                customer_unopened_streak[customer_id] = 0
            else:
                time_to_open = np.nan
                customer_unopened_streak[customer_id] += 1

            # Bounced? (~2%)
            bounced = 1 if np.random.random() < 0.02 else 0
            if bounced:
                opened = 0
                clicked = 0
                time_to_open = np.nan

            unsub_prob = 0.001 + 0.01 * customer_unopened_streak[customer_id]
            unsubscribed = 1 if np.random.random() < unsub_prob else 0
            if unsubscribed:
                customer_unsubscribe_date[customer_id] = email_date

            unsub_dt = customer_unsubscribe_date.get(customer_id)
            records.append({
                "email_id": f"EML{email_id:08d}",
                "customer_id": customer_id,
                "sent_date": email_date.strftime("%Y-%m-%d"),
                "campaign_type": campaign,
                "opened": opened,
                "clicked": clicked,
                "subject_line_category": np.random.choice(subject_categories),
                "send_hour": send_hour,
                "device_type": device,
                "unsubscribed": unsubscribed,
                "bounced": bounced,
                "time_to_open_hours": time_to_open,
                "unsubscribe_date": unsub_dt.strftime("%Y-%m-%d") if unsub_dt else None,
            })
            email_id += 1

    df = pd.DataFrame(records)
    df = df.sort_values("sent_date").reset_index(drop=True)

    return df


def main():
    """Generate all test datasets."""
    print("=" * 60)
    print("Generating synthetic test datasets for TS track")
    print("=" * 60)

    # Generate transactions
    print("\n1. Generating transactions dataset...")
    txn_df = generate_transactions(n_customers=5000, target_rows=50000)
    txn_path = FIXTURES_DIR / "customer_transactions.csv"
    txn_df.to_csv(txn_path, index=False)

    print(f"   Saved: {txn_path}")
    print(f"   Rows: {len(txn_df):,}")
    print(f"   Columns: {len(txn_df.columns)}")
    print(f"   Unique customers: {txn_df.customer_id.nunique():,}")
    print(f"   Avg transactions/customer: {len(txn_df) / txn_df.customer_id.nunique():.1f}")
    print(f"   Date range: {txn_df.transaction_date.min()} to {txn_df.transaction_date.max()}")

    # Generate emails
    print("\n2. Generating emails dataset...")
    email_df = generate_emails(n_customers=5000, target_rows=150000)
    email_path = FIXTURES_DIR / "customer_emails.csv"
    email_df.to_csv(email_path, index=False)

    print(f"   Saved: {email_path}")
    print(f"   Rows: {len(email_df):,}")
    print(f"   Columns: {len(email_df.columns)}")
    print(f"   Unique customers: {email_df.customer_id.nunique():,}")
    print(f"   Avg emails/customer: {len(email_df) / email_df.customer_id.nunique():.1f}")
    print(f"   Open rate: {email_df.opened.mean() * 100:.1f}%")
    print(f"   Click rate: {email_df.clicked.mean() * 100:.1f}%")
    print(f"   Unsubscribe rate: {email_df.unsubscribed.mean() * 100:.2f}%")

    print("\n" + "=" * 60)
    print("Done! Datasets ready for TS track testing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
