"""Synthetic contract fixture for lifecycle enrichment tests.

Five accounts cover every code path that ``enrich_lifecycle_dataset`` needs to
handle. The fixture is shaped like the SPS contract table (uppercase column
names, real column semantics) so the tests double as documentation of the
expected SPS behavior.
"""
from __future__ import annotations

import pytest

from customer_retention.core.compat import native_pd

# Synthetic but realistic timestamps. The exact dates are arbitrary; the
# *relative* ordering is what matters for the tests.
T0 = native_pd.Timestamp("2024-01-01")
DAY = native_pd.Timedelta(days=1)


def _row(
    account_id: str,
    contract_id: str,
    status: str,
    start: native_pd.Timestamp,
    btd: native_pd.Timestamp | None,
    contract_end: native_pd.Timestamp | None,
    document_plan: str = "PRO",
    contract_term: int = 12,
    take_rate: float = 0.15,
) -> dict:
    return {
        "ACCOUNT_ID": account_id,
        "CONTRACT_ID": contract_id,
        "CONTRACT_STATUS": status,
        "CONTRACT_START_DATE": start,
        "BILLING_TERMINATION_DATE": btd,
        "CONTRACT_END_DATE": contract_end,
        "DOCUMENT_PLAN": document_plan,
        "CONTRACT_TERM": contract_term,
        "TAKE_RATE": take_rate,
    }


@pytest.fixture
def synthetic_contract_records() -> list[dict]:
    rows: list[dict] = []

    # ---------------------------------------------------------------- Account A
    # 10 active contracts (NULL BTD). Should produce 10 STARTs, 0 TERMINATEs.
    for i in range(10):
        rows.append(
            _row(
                account_id="A",
                contract_id=f"A-{i:02d}",
                status="Active",
                start=T0 + i * 30 * DAY,
                btd=None,
                contract_end=T0 + (i + 12) * 30 * DAY,  # far future evergreen
            )
        )

    # ---------------------------------------------------------------- Account B
    # 10 contracts: 7 cancelled with real BTD + 3 active. 17 events expected.
    for i in range(7):
        start = T0 + i * 60 * DAY
        rows.append(
            _row(
                account_id="B",
                contract_id=f"B-{i:02d}",
                status="Cancelled",
                start=start,
                btd=start + 30 * DAY,  # cancelled 30 days after start
                contract_end=start + 90 * DAY,
            )
        )
    for i in range(3):
        rows.append(
            _row(
                account_id="B",
                contract_id=f"B-act-{i:02d}",
                status="Active",
                start=T0 + (10 + i) * 60 * DAY,
                btd=None,
                contract_end=None,
            )
        )

    # ---------------------------------------------------------------- Account C
    # Active contract whose CONTRACT_END_DATE is in the past (the framework
    # must NOT promote this to a TERMINATE because the status is Active).
    rows.append(
        _row(
            account_id="C",
            contract_id="C-00",
            status="Active",
            start=T0,
            btd=None,
            contract_end=T0 + 5 * DAY,  # well in the past relative to "now"
        )
    )

    # ---------------------------------------------------------------- Account D
    # Cancelled with NULL BTD but populated CONTRACT_END_DATE — exercises the
    # multi-element coalesce fallback when valid_to_columns has 2 entries.
    rows.append(
        _row(
            account_id="D",
            contract_id="D-00",
            status="Cancelled",
            start=T0,
            btd=None,
            contract_end=T0 + 60 * DAY,
        )
    )

    # ---------------------------------------------------------------- Account E
    # Cancelled with BTD < CONTRACT_START_DATE — the inverted-row corruption.
    rows.append(
        _row(
            account_id="E",
            contract_id="E-00",
            status="Cancelled",
            start=T0 + 30 * DAY,
            btd=T0,  # earlier than start
            contract_end=T0 + 60 * DAY,
        )
    )

    return rows


@pytest.fixture
def contract_pdf(synthetic_contract_records: list[dict]) -> native_pd.DataFrame:
    """Native pandas DataFrame view of the synthetic fixture."""
    return native_pd.DataFrame(synthetic_contract_records)


# ---------------------------------------------------------------------------
# Spark-pandas variant — only available when pyspark is importable. Tests that
# need both engines parametrize over ``df_factory`` to keep the test bodies
# identical for the two backends.
# ---------------------------------------------------------------------------


def _make_pandas(records: list[dict]) -> native_pd.DataFrame:
    return native_pd.DataFrame(records)


def _make_spark_pandas(records: list[dict]):
    pytest.importorskip("pyspark")
    import pyspark.pandas as ps
    from pyspark.sql import SparkSession

    # In environments where only Databricks Connect is configured (e.g. CI on
    # the laptop), creating a local Spark session raises. Skip cleanly so
    # the test runs locally on the pandas path only.
    try:
        SparkSession.builder.master("local[1]").getOrCreate()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"local Spark session unavailable: {exc}")

    return ps.DataFrame(records)


@pytest.fixture(params=["pandas", "spark_pandas"], ids=["pandas", "spark_pandas"])
def df_factory(request):
    if request.param == "pandas":
        return _make_pandas
    return _make_spark_pandas
