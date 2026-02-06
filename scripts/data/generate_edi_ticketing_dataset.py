#!/usr/bin/env python3
"""
Generate 3 CSV datasets for churn modeling (configurable years of history, default 4):

1) customer_profiles.csv
   - customer_id, company_name, phone, address, contract_start, contract_end, churn_end (if churned else null),
     churned (label lives here)

2) edi_transactions.csv
   - EDI transaction log for customers during their 1-year contract window
   - 1..1000 tx/customer, skewed/non-normal, varied types, success/failed, timestamps

3) support_tickets.csv
   - Ticketing log (Zendesk/Zoho-like) for customers with transactions
   - tickets tied to failed transactions + general tickets
   - response times ~1 hour .. 2 days; resolution up to 30 days

Default output directory:
    FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"

No new dependencies beyond the standard data stack (numpy/pandas).
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"


# ----------------------------
# Helpers
# ----------------------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def utc_dt(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def print_progress(current: int, total: int, label: str = "", width: int = 40) -> None:
    frac = current / total if total else 1.0
    filled = int(width * frac)
    bar = "=" * filled + ">" * (1 if filled < width else 0) + "." * (width - filled - 1)
    pct = frac * 100
    print(f"\r  {label} [{bar}] {pct:5.1f}%", end="", flush=True)
    if current >= total:
        print()


def iso_z(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return utc_dt(dt).isoformat().replace("+00:00", "Z")


def sample_contract_start(rng: np.random.Generator, anchor_end: datetime, n_years: int = 4) -> datetime:
    """
    Sample contract start uniformly-ish across an n_years window ending at anchor_end.
    Slightly biased toward more recent starts (more realistic).
    """
    span = 365 * n_years
    u = rng.beta(1.4, 1.0)  # bias toward recent (larger u)
    days_back = int((1.0 - u) * span)
    start = anchor_end - timedelta(days=days_back)

    # add day jitter and random time-of-day
    start -= timedelta(days=int(rng.integers(0, 14)))
    start = start.replace(hour=int(rng.integers(0, 24)), minute=int(rng.integers(0, 60)), second=0, microsecond=0)
    return utc_dt(start)


def mixture_response_seconds(rng: np.random.Generator) -> int:
    """
    First response time:
      - mostly 1h..6h
      - some 6h..2d
      - rare 2d..5d
    """
    u = rng.random()
    if u < 0.70:
        sec = int(rng.lognormal(mean=math.log(2.5 * 3600), sigma=0.6))
        return int(clamp(sec, 1 * 3600, 6 * 3600))
    if u < 0.95:
        sec = int(rng.lognormal(mean=math.log(14 * 3600), sigma=0.7))
        return int(clamp(sec, 6 * 3600, 2 * 24 * 3600))
    sec = int(rng.lognormal(mean=math.log(3 * 24 * 3600), sigma=0.7))
    return int(clamp(sec, 2 * 24 * 3600, 5 * 24 * 3600))


def mixture_resolution_seconds(rng: np.random.Generator) -> int:
    """
    Resolution time:
      - many resolved within 4h..2d
      - some take 2d..7d
      - fewer take 7d..30d
    """
    u = rng.random()
    if u < 0.65:
        sec = int(rng.lognormal(mean=math.log(18 * 3600), sigma=0.9))
        return int(clamp(sec, 4 * 3600, 2 * 24 * 3600))
    if u < 0.90:
        sec = int(rng.lognormal(mean=math.log(3.5 * 24 * 3600), sigma=0.8))
        return int(clamp(sec, 2 * 24 * 3600, 7 * 24 * 3600))
    sec = int(rng.lognormal(mean=math.log(12 * 24 * 3600), sigma=0.8))
    return int(clamp(sec, 7 * 24 * 3600, 30 * 24 * 3600))


def company_name(rng: np.random.Generator) -> str:
    """
    Generate slightly goofy "bad Chinese business name" vibes + normal-ish words.
    """
    goofy = [
        "Lucky", "Golden", "Dragon", "Panda", "GreatWall", "Jade", "Fortune", "Prosper", "Harmony", "RedSun",
        "MoonNoodle", "Shiny", "Super", "Mega", "Ultra", "Happy", "Bamboo", "Tiger", "Phoenix", "SilkRoad",
        "RiceCloud", "Dumpling", "TeaTech", "WokLogic", "Ninja", "Koi", "Ginseng", "HotPot", "Bao", "Wonton",
    ]
    biz = [
        "Logistics", "Trading", "Holdings", "Solutions", "Systems", "Networks", "Integration", "Supply",
        "Commerce", "Data", "Cloud", "Industries", "Group", "Partners", "Enterprises", "Services", "Dynamics",
    ]
    extra = [
        "International", "Global", "Unlimited", "Prime", "Express", "Works", "Factory", "Digital", "Modern",
        "Union", "Collective", "Grand", "Century", "Future", "Superior", "Value",
    ]

    w1 = rng.choice(goofy)
    w2 = rng.choice(goofy + biz)  # sometimes double-goofy
    w3 = rng.choice(biz)
    if rng.random() < 0.55:
        mid = rng.choice(extra)
        base = f"{w1} {w2} {mid} {w3}"
    else:
        base = f"{w1} {w2} {w3}"

    # Add suffix number: year-ish or small integer
    if rng.random() < 0.70:
        if rng.random() < 0.50:
            base += f" {int(rng.integers(1, 9))}"
        else:
            base += f" {int(rng.integers(2012, 2027))}"
    return base


def fake_phone(rng: np.random.Generator) -> str:
    # US-like, obviously fake
    area = int(rng.integers(200, 999))
    exch = int(rng.integers(200, 999))
    line = int(rng.integers(1000, 9999))
    return f"+1-{area:03d}-{exch:03d}-{line:04d}"


def fake_address(rng: np.random.Generator) -> str:
    streets = ["Market St", "Broadway", "Main St", "Industrial Rd", "Canal Ave", "Maple Dr", "Sunset Blvd", "River Pkwy"]
    cities = ["New York", "Jersey City", "Chicago", "Dallas", "Atlanta", "Seattle", "San Jose", "Phoenix", "Miami"]
    states = ["NY", "NJ", "IL", "TX", "GA", "WA", "CA", "AZ", "FL"]
    n = int(rng.integers(10, 9999))
    suite = ""
    if rng.random() < 0.55:
        suite = f", Suite {int(rng.integers(100, 999))}"
    zipc = int(rng.integers(10000, 99999))
    return f"{n} {rng.choice(streets)}{suite}, {rng.choice(cities)}, {rng.choice(states)} {zipc:05d}"


# ----------------------------
# Generation
# ----------------------------
def generate(
    output_dir: Path,
    n_customers: int = 1_000,
    churn_rate: float = 0.10,
    seed: int = 42,
    anchor_end: datetime | None = None,
    n_years: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Stable default anchor_end
    if anchor_end is None:
        anchor_end = utc_dt(datetime(2026, 2, 5, 12, 0, 0))
    else:
        anchor_end = utc_dt(anchor_end)

    customer_ids = np.array([f"C{idx:05d}" for idx in range(1, n_customers + 1)], dtype=object)

    # Customer-level tiers (influence behavior)
    activity_tier = rng.choice(["low", "mid", "high"], size=n_customers, p=[0.45, 0.40, 0.15])
    risk_tier = rng.choice(["low", "med", "high"], size=n_customers, p=[0.55, 0.35, 0.10])

    # Contract start over 5-year window; contract is 1 year fixed
    contract_start = np.array([sample_contract_start(rng, anchor_end, n_years) for _ in range(n_customers)], dtype=object)
    contract_end = np.array([utc_dt(s + timedelta(days=365)) for s in contract_start], dtype=object)

    # Churn label: ~60% non-renewal (churn_end == contract_end),
    # ~40% early termination (churn_end before contract_end).
    # Only customers whose churn_end <= anchor_end can be marked as churned.
    churned = np.zeros(n_customers, dtype=int)
    churn_end = np.array([None] * n_customers, dtype=object)
    churn_candidates = rng.choice(n_customers, size=int(round(n_customers * churn_rate * 1.5)), replace=False)
    actual_churned = 0
    target_churned = int(round(n_customers * churn_rate))
    for i in churn_candidates:
        if actual_churned >= target_churned:
            break
        if rng.random() < 0.60:
            candidate_date = contract_end[i]
        else:
            contract_days = max(30, (contract_end[i] - contract_start[i]).days)
            early_frac = rng.beta(2.5, 1.5)
            early_days = int(30 + early_frac * (contract_days - 30))
            candidate_date = utc_dt(contract_start[i] + timedelta(days=early_days))
        if candidate_date <= anchor_end:
            churned[i] = 1
            churn_end[i] = candidate_date
            actual_churned += 1

    # Company profile fields
    names = []
    phones = []
    addresses = []
    # Ensure uniqueness-ish of names by re-rolling collisions
    seen = set()
    for _ in range(n_customers):
        nm = company_name(rng)
        tries = 0
        while nm in seen and tries < 5:
            nm = company_name(rng)
            tries += 1
        seen.add(nm)
        names.append(nm)
        phones.append(fake_phone(rng))
        addresses.append(fake_address(rng))

    df_customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "company_name": names,
            "phone": phones,
            "address": addresses,
            "contract_start": [iso_z(d) for d in contract_start],
            "contract_end": [iso_z(d) for d in contract_end],
            "churn_end": [iso_z(d) if d is not None else "" for d in churn_end],
            "churned": churned.astype(int),
            "activity_tier": activity_tier,
            "risk_tier": risk_tier,
        }
    )

    # ----------------------------
    # Transactions
    # ----------------------------
    # Skewed tx counts 1..1000
    pareto = rng.pareto(a=1.25, size=n_customers)
    base = {"low": 10, "mid": 40, "high": 160}
    tx_counts = np.empty(n_customers, dtype=int)
    for i in range(n_customers):
        m = base[str(activity_tier[i])]
        n = int(1 + m * (1 + pareto[i]))
        tx_counts[i] = int(clamp(n, 1, 1000))

    tx_types = np.array(["850", "810", "855", "856", "820", "997", "214", "940", "945"], dtype=object)
    directions = np.array(["inbound", "outbound"], dtype=object)
    channels = np.array(["sftp", "as2", "api", "van"], dtype=object)
    currencies = np.array(["USD", "EUR", "GBP"], dtype=object)
    partners = np.array([f"P{p:04d}" for p in range(1, 801)], dtype=object)

    risk_fail_base = {"low": 0.015, "med": 0.035, "high": 0.070}
    tier_amount_scale = {"low": 1.0, "mid": 1.7, "high": 3.0}

    tx_rows: list[dict] = []
    tx_id_counter = 1

    print("Generating transactions...")
    for i in range(n_customers):
        if i % 200 == 0 or i == n_customers - 1:
            print_progress(i + 1, n_customers, "Transactions")
        cid = customer_ids[i]
        start = contract_start[i]
        end = contract_end[i]
        is_churn = int(churned[i])

        active_days = max(1, (end - start).days)
        n_tx = int(tx_counts[i])

        # Burstiness mix
        u = rng.random(n_tx)
        clustered = rng.random(n_tx) < 0.30
        offsets = np.empty(n_tx, dtype=float)
        offsets[~clustered] = u[~clustered] * active_days

        k = int(rng.integers(2, 7))
        anchors = rng.integers(0, active_days, size=k)
        jitter = rng.normal(loc=0.0, scale=max(1.0, active_days / 50.0), size=n_tx)
        offsets[clustered] = anchors[rng.integers(0, k, size=clustered.sum())] + jitter[clustered]
        offsets = np.clip(offsets, 0, active_days)

        # Timestamps
        event_times = []
        for d in np.sort(offsets):
            dt = start + timedelta(days=float(d), seconds=int(rng.integers(0, 24 * 3600)))
            event_times.append(utc_dt(dt))

        # Amounts (lognormal + spikes)
        mean = math.log(120.0 * tier_amount_scale[str(activity_tier[i])])
        sigma = 1.05 if str(activity_tier[i]) != "high" else 1.15
        amounts = rng.lognormal(mean=mean, sigma=sigma, size=n_tx)
        spikes = rng.random(n_tx) < 0.02
        amounts[spikes] *= rng.uniform(5, 25, size=spikes.sum())
        amounts = np.round(np.clip(amounts, 5.0, 250_000.0), 2)

        # Other fields
        ttypes = rng.choice(tx_types, size=n_tx, p=[0.14, 0.18, 0.08, 0.16, 0.07, 0.09, 0.06, 0.11, 0.11])
        dirs = rng.choice(directions, size=n_tx, p=[0.52, 0.48])
        chs = rng.choice(channels, size=n_tx, p=[0.35, 0.20, 0.25, 0.20])
        curr = rng.choice(currencies, size=n_tx, p=[0.86, 0.10, 0.04])
        partner = rng.choice(partners, size=n_tx)

        file_kb = rng.lognormal(mean=math.log(180), sigma=0.9, size=n_tx)
        file_kb = np.round(np.clip(file_kb, 5, 250_000)).astype(int)

        proc_ms = (file_kb * rng.lognormal(mean=math.log(2.8), sigma=0.6, size=n_tx)).astype(int)
        proc_ms = np.clip(proc_ms, 30, 900_000)

        base_fail = risk_fail_base[str(risk_tier[i])]
        status = []
        err_code = []

        # If churned: increase instability near end-of-contract (last ~60 days)
        for j in range(n_tx):
            dt = event_times[j]
            file_penalty = 0.0000009 * file_kb[j]
            p_fail = base_fail + file_penalty

            if is_churn:
                days_to_end = (end - dt).total_seconds() / (24 * 3600)
                # ramp during last 60 days
                ramp = 0.0
                if days_to_end <= 60:
                    x = (60 - days_to_end) / 12.0
                    ramp = 0.10 * sigmoid(x)  # up to +10%
                p_fail += ramp

            p_fail = clamp(p_fail, 0.002, 0.35)

            failed = rng.random() < p_fail
            if failed:
                status.append("failed")
                err_code.append(
                    rng.choice(
                        ["VALIDATION", "MAPPING", "PARTNER_ACK_TIMEOUT", "DUPLICATE", "SCHEMA_MISMATCH", "TRANSPORT"],
                        p=[0.26, 0.18, 0.22, 0.10, 0.14, 0.10],
                    )
                )
            else:
                status.append("success")
                err_code.append("")

        for j in range(n_tx):
            tx_id = f"T{tx_id_counter:09d}"
            tx_id_counter += 1
            tx_rows.append(
                {
                    "transaction_id": tx_id,
                    "customer_id": cid,
                    "event_timestamp": iso_z(event_times[j]),
                    "edi_transaction_type": str(ttypes[j]),
                    "direction": str(dirs[j]),
                    "channel": str(chs[j]),
                    "partner_id": str(partner[j]),
                    "status": status[j],
                    "error_code": err_code[j],
                    "amount": float(amounts[j]),
                    "currency": str(curr[j]),
                    "file_size_kb": int(file_kb[j]),
                    "processing_time_ms": int(proc_ms[j]),
                }
            )

    df_tx = pd.DataFrame(tx_rows)
    df_tx.sort_values(["customer_id", "event_timestamp"], inplace=True)

    # ----------------------------
    # Tickets
    # ----------------------------
    ticket_rows: list[dict] = []
    ticket_id_counter = 1

    # Failed transactions by customer (for linking)
    failed_by_customer: dict[str, list[tuple[str, str]]] = {}
    failed_df = df_tx.loc[df_tx["status"] == "failed", ["customer_id", "transaction_id", "event_timestamp"]]
    for cust_id, tx_id, ts in failed_df.itertuples(index=False):
        failed_by_customer.setdefault(cust_id, []).append((tx_id, ts))

    categories = [
        ("EDI Failure", ["Validation error", "Partner ACK missing", "Mapping issue", "Transport issue", "Duplicate"]),
        ("Billing", ["Invoice question", "Payment failure", "Refund request", "Pricing change"]),
        ("Account", ["User access", "SSO/SAML", "Password reset", "Permissions"]),
        ("How-to", ["Setup guidance", "Mapping question", "Reporting question", "API usage"]),
        ("Performance", ["Slow processing", "Queue backlog", "Timeouts"]),
    ]
    priorities = ["low", "normal", "high", "urgent"]
    channels_t = ["email", "web", "phone", "chat"]
    sla_hours = {"low": 24, "normal": 8, "high": 4, "urgent": 1}

    def make_ticket(
        cid: str,
        created_dt: datetime,
        category: str,
        subcategory: str,
        priority: str,
        related_tx: str,
        cstart: datetime,
        cend: datetime,
        is_churn: int,
        act_tier: str,
        r_tier: str,
    ) -> None:
        nonlocal ticket_id_counter
        tid = f"Z{ticket_id_counter:09d}"
        ticket_id_counter += 1

        first_resp_sec = mixture_response_seconds(rng)
        resolve_sec = mixture_resolution_seconds(rng)

        first_resp = utc_dt(created_dt + timedelta(seconds=first_resp_sec))
        resolved = utc_dt(created_dt + timedelta(seconds=resolve_sec))

        # Slightly more open tickets for churned
        p_open = 0.03 + (0.05 if is_churn else 0.0)
        status = "open" if (rng.random() < p_open) else "closed"

        if status == "open":
            resolved_str = ""
            close_hours = ""
        else:
            resolved_str = iso_z(resolved)
            close_hours = round((resolved - utc_dt(created_dt)).total_seconds() / 3600.0, 2)

        target = sla_hours[priority]
        fr_hours = (first_resp - utc_dt(created_dt)).total_seconds() / 3600.0
        sla_breached = int(fr_hours > target)

        # Satisfaction if closed
        satisfaction = ""
        if status == "closed" and rng.random() < 0.65:
            base = 4.3
            if is_churn:
                base -= 0.8
            if priority in ("urgent", "high"):
                base -= 0.3
            if category == "EDI Failure":
                base -= 0.4
            score = int(clamp(round(rng.normal(loc=base, scale=0.9)), 1, 5))
            satisfaction = str(score)

        ticket_rows.append(
            {
                "ticket_id": tid,
                "customer_id": cid,
                "created_at": iso_z(created_dt),
                "first_response_at": iso_z(first_resp),
                "resolved_at": resolved_str,
                "status": status,
                "category": category,
                "subcategory": subcategory,
                "priority": priority,
                "channel": str(rng.choice(channels_t, p=[0.45, 0.25, 0.15, 0.15])),
                "related_transaction_id": related_tx,
                "first_response_hours": round(fr_hours, 2),
                "time_to_close_hours": close_hours,
                "sla_first_response_hours": target,
                "sla_breached": sla_breached,
                "satisfaction": satisfaction,
                # keep tiers here for convenience (label still in customer_profiles)
                "activity_tier": act_tier,
                "risk_tier": r_tier,
            }
        )

    print("Generating tickets...")
    for i in range(n_customers):
        if i % 200 == 0 or i == n_customers - 1:
            print_progress(i + 1, n_customers, "Tickets      ")
        cid = customer_ids[i]
        cstart = contract_start[i]
        cend = contract_end[i]
        is_churn = int(churned[i])
        act_t = str(activity_tier[i])
        r_t = str(risk_tier[i])

        failed_list = failed_by_customer.get(cid, [])
        n_failed = len(failed_list)

        # More tickets for higher risk and churned, correlated with failures
        risk_mult = {"low": 0.7, "med": 1.0, "high": 1.4}[r_t]
        churn_mult = 1.4 if is_churn else 1.0

        lam_general = 0.6 * risk_mult * churn_mult  # ~0..2+
        lam_fail = 0.08 * n_failed * risk_mult * churn_mult

        n_general = int(min(int(rng.poisson(lam=lam_general)), 12))
        n_fail_tickets = int(min(int(rng.poisson(lam=lam_fail)), 40))

        # Link to failed transactions
        fail_samples: list[tuple[str, str]] = []
        if n_fail_tickets > 0 and n_failed > 0:
            idx = rng.choice(n_failed, size=min(n_fail_tickets, n_failed), replace=False)
            fail_samples = [failed_list[k] for k in idx]
        extra_unlinked_fail = max(0, n_fail_tickets - len(fail_samples))

        # Linked failure tickets (created shortly after failure)
        for (txid, ts) in fail_samples:
            tx_dt = utc_dt(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            delay_sec = int(rng.lognormal(mean=math.log(2 * 3600), sigma=1.0))
            delay_sec = int(clamp(delay_sec, 10 * 60, 2 * 24 * 3600))
            created = tx_dt + timedelta(seconds=delay_sec)

            # Keep within contract
            if created < cstart:
                created = cstart + timedelta(hours=int(rng.integers(1, 24)))
            if created > cend:
                created = cend - timedelta(hours=int(rng.integers(1, 48)))

            category, subs = categories[0]
            sub = str(rng.choice(subs))
            pr = str(rng.choice(priorities, p=[0.10, 0.55, 0.25, 0.10]))

            make_ticket(cid, utc_dt(created), category, sub, pr, txid, cstart, cend, is_churn, act_t, r_t)

        # Unlinked failure tickets
        for _ in range(extra_unlinked_fail):
            span_days = max(1, (cend - cstart).days)
            bias = rng.beta(2.0, 1.2)  # skew later
            created = cstart + timedelta(days=float(bias * span_days), seconds=int(rng.integers(0, 24 * 3600)))

            category, subs = categories[0]
            sub = str(rng.choice(subs))
            pr = str(rng.choice(priorities, p=[0.12, 0.58, 0.22, 0.08]))
            make_ticket(cid, utc_dt(created), category, sub, pr, "", cstart, cend, is_churn, act_t, r_t)

        # General tickets
        for _ in range(n_general):
            span_days = max(1, (cend - cstart).days)
            u = rng.random()
            d = (0.35 * u + 0.65 * (u**0.6)) * span_days
            created = cstart + timedelta(days=float(d), seconds=int(rng.integers(0, 24 * 3600)))

            cat, subs = categories[int(rng.integers(1, len(categories)))]
            sub = str(rng.choice(subs))
            pr = (
                str(rng.choice(priorities, p=[0.18, 0.50, 0.24, 0.08]))
                if is_churn
                else str(rng.choice(priorities, p=[0.26, 0.55, 0.16, 0.03]))
            )
            make_ticket(cid, utc_dt(created), cat, sub, pr, "", cstart, cend, is_churn, act_t, r_t)

    df_tickets = pd.DataFrame(ticket_rows)
    df_tickets.sort_values(["customer_id", "created_at"], inplace=True)

    # ----------------------------
    # Write CSVs
    # ----------------------------
    customer_path = output_dir / "3set_customer_profiles.csv"
    tx_path = output_dir / "3set_edi_transactions.csv"
    tickets_path = output_dir / "3set_support_tickets.csv"

    df_customers.to_csv(customer_path, index=False)
    df_tx.to_csv(tx_path, index=False)
    df_tickets.to_csv(tickets_path, index=False)

    print(f"Wrote: {customer_path} ({len(df_customers):,} rows)")
    print(f"Wrote: {tx_path} ({len(df_tx):,} rows)")
    print(f"Wrote: {tickets_path} ({len(df_tickets):,} rows)")
    print(f"Churned customers: {int(df_customers['churned'].sum()):,} / {n_customers:,}")

    return df_customers, df_tx, df_tickets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write CSVs (default: tests/fixtures relative to repo root from this script)",
    )
    ap.add_argument("--n-customers", type=int, default=1_000)
    ap.add_argument("--churn-rate", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-years", type=int, default=4, help="Years of history to generate (default: 4)")
    ap.add_argument("--anchor-end", default="2026-02-05T12:00:00Z", help="ISO8601 anchor for history window")
    args = ap.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else FIXTURES_DIR
    anchor_end = utc_dt(datetime.fromisoformat(args.anchor_end.replace("Z", "+00:00")))

    generate(
        output_dir=output_dir,
        n_customers=args.n_customers,
        churn_rate=args.churn_rate,
        seed=args.seed,
        anchor_end=anchor_end,
        n_years=args.n_years,
    )


if __name__ == "__main__":
    main()
