"""Synthetic case_history fixture for SCD reconstruction tests.

Cases A-G cover every code path that ``reconstruct_scd_history_at_grid``
needs to handle. The fixture is shaped like the SPS case_history table
(uppercase column names, real column semantics) so the tests double as
documentation of the expected SPS behavior.
"""
from __future__ import annotations

import pytest

from customer_retention.core.compat import native_pd

T0 = native_pd.Timestamp("2024-01-01")
DAY = native_pd.Timedelta(days=1)


def _h(
    case_id: str,
    field: str,
    new_value,
    created_date: native_pd.Timestamp,
    *,
    case_history_id: str = "",
    old_value=None,
) -> dict:
    return {
        "CASE_HISTORY_ID": case_history_id or f"{case_id}-{field}-{int((created_date - T0).days)}",
        "CASE_ID": case_id,
        "FIELD": field,
        "OLD_VALUE": old_value,
        "NEW_VALUE": new_value,
        "CREATED_DATE": created_date,
    }


@pytest.fixture
def synthetic_history_records() -> list[dict]:
    rows: list[dict] = []

    # Case A — 5 status changes over 6 months
    rows += [
        _h("A", "Status", "Open", T0 + 1 * DAY, old_value=None),
        _h("A", "Status", "In Progress", T0 + 30 * DAY, old_value="Open"),
        _h("A", "Status", "On Hold", T0 + 60 * DAY, old_value="In Progress"),
        _h("A", "Status", "In Progress", T0 + 90 * DAY, old_value="On Hold"),
        _h("A", "Status", "Closed", T0 + 150 * DAY, old_value="In Progress"),
    ]

    # Case B — interleaved Status + Priority changes
    rows += [
        _h("B", "Status", "Open", T0 + 5 * DAY, old_value=None),
        _h("B", "Priority", "Low", T0 + 10 * DAY, old_value=None),
        _h("B", "Status", "In Progress", T0 + 40 * DAY, old_value="Open"),
        _h("B", "Priority", "High", T0 + 50 * DAY, old_value="Low"),
        _h("B", "Status", "Closed", T0 + 100 * DAY, old_value="In Progress"),
    ]

    # Case C — zero history rows (parent fallback path).  Note: NO entry here.

    # Case D — collision: same field on the same parent at the exact same
    # CREATED_DATE.  Larger CASE_HISTORY_ID is "later" and must win.
    rows += [
        _h(
            "D",
            "Status",
            "First",
            T0 + 5 * DAY,
            case_history_id="D-status-aaa",
            old_value=None,
        ),
        _h(
            "D",
            "Status",
            "Second",
            T0 + 5 * DAY,
            case_history_id="D-status-zzz",
            old_value="First",
        ),
    ]

    # Case E — change to a field that's NOT tracked
    rows += [
        _h("E", "CustomField_X", "anything", T0 + 5 * DAY),
    ]

    # Case F — change BEFORE the first grid date (cold-start)
    rows += [
        _h("F", "Status", "PreGrid", T0 - 30 * DAY, old_value=None),
    ]

    return rows


@pytest.fixture
def synthetic_parent_records() -> list[dict]:
    return [
        {"CASE_ID": "A", "CREATED_DATE": T0, "CASE_STATUS": "Open", "PRIORITY": "Med"},
        {"CASE_ID": "B", "CREATED_DATE": T0, "CASE_STATUS": "Closed", "PRIORITY": "Med"},
        {"CASE_ID": "C", "CREATED_DATE": T0, "CASE_STATUS": "Pending", "PRIORITY": "Low"},
        {"CASE_ID": "D", "CREATED_DATE": T0, "CASE_STATUS": "Closed", "PRIORITY": "Med"},
        {"CASE_ID": "E", "CREATED_DATE": T0, "CASE_STATUS": "Open", "PRIORITY": "Low"},
        {"CASE_ID": "F", "CREATED_DATE": T0, "CASE_STATUS": "Closed", "PRIORITY": "Med"},
    ]


@pytest.fixture
def grid_dates() -> list[native_pd.Timestamp]:
    return [T0, T0 + 30 * DAY, T0 + 60 * DAY, T0 + 90 * DAY, T0 + 120 * DAY]


def _make_pandas(records: list[dict]) -> native_pd.DataFrame:
    return native_pd.DataFrame(records)


def _make_spark_pandas(records: list[dict]):
    pytest.importorskip("pyspark")
    import pyspark.pandas as ps
    from pyspark.sql import SparkSession

    try:
        SparkSession.builder.master("local[1]").getOrCreate()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"local Spark session unavailable: {exc}")

    return ps.DataFrame(records)


@pytest.fixture(params=["pandas", "spark_pandas"], ids=["pandas", "spark_pandas"])
def df_factory(request):
    if request.param == "pandas":
        return _make_pandas
    return _make_spark_pandas
