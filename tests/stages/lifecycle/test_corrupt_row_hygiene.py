"""Cycle 8 — corrupt-row hygiene for lifecycle enrichment.

The pre-cycle-8 implementation only classifies rows with
``valid_from > effective_valid_to`` as "corrupt". Two additional classes
silently escape the ``on_corrupt_row="skip"`` filter and surface as NULL
``event_timestamp`` values downstream:

1. ``valid_from IS NULL`` — START emitted with NULL event_timestamp.
2. Terminal status with ``effective_valid_to IS NULL`` — START emitted,
   TERMINATE dropped by the ``isNotNull`` filter. The row is a semantic
   contradiction: status says the lifespan ended but no termination date
   exists.

This module locks the cycle-8 invariant: under ``on_corrupt_row="skip"``,
the enriched output contains **zero NULL event_timestamp rows and zero
terminal-status rows with a NULL effective_valid_to**, across pandas and
pyspark.pandas backends.
"""
from __future__ import annotations

import pytest

from customer_retention.core.compat import native_pd
from customer_retention.stages.lifecycle import (
    LifecycleEnrichmentConfig,
    enrich_lifecycle_dataset,
)


def _base_config(**overrides) -> LifecycleEnrichmentConfig:
    base = dict(
        enriched_view_name="enriched_contract",
        parent_entity_key="ACCOUNT_ID",
        sub_entity_key="CONTRACT_ID",
        valid_from_column="CONTRACT_START_DATE",
        valid_to_columns=("BILLING_TERMINATION_DATE",),
        status_column="CONTRACT_STATUS",
        terminal_status_values=("Cancelled", "Terminated", "Expired"),
        drop_columns=("BILLING_TERMINATION_DATE", "CONTRACT_STATUS"),
    )
    base.update(overrides)
    return LifecycleEnrichmentConfig(**base)


def _to_native(df) -> native_pd.DataFrame:
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return df


def _hygiene_oracle(records, *, config: LifecycleEnrichmentConfig) -> int:
    """First-principles recomputation of the cycle-8 invariant.

    Counts how many input records are corrupt under the extended policy. A
    corrupt record yields zero emitted events under ``skip``; a clean
    record yields one START (always) plus one TERMINATE (if the record is
    terminally-statused with a non-null valid-to).
    """
    terminal = set(config.terminal_status_values)
    n_corrupt = 0
    for r in records:
        vf = r.get(config.valid_from_column)
        vts = [r.get(c) for c in config.valid_to_columns]
        effective = next((v for v in vts if v is not None), None)
        is_terminal = r.get(config.status_column) in terminal
        if vf is None:
            n_corrupt += 1
            continue
        if is_terminal and effective is None:
            n_corrupt += 1
            continue
        if effective is not None and vf > effective:
            n_corrupt += 1
    return n_corrupt


@pytest.fixture
def four_row_fixture() -> list[dict]:
    """PLAN C8 fixture: NULL-start, terminal-NULL-terminate, inverted, valid."""
    T0 = native_pd.Timestamp("2024-01-01")
    DAY = native_pd.Timedelta(days=1)
    return [
        {
            "ACCOUNT_ID": "A",
            "CONTRACT_ID": "A-null-start",
            "CONTRACT_STATUS": "Active",
            "CONTRACT_START_DATE": None,
            "BILLING_TERMINATION_DATE": None,
        },
        {
            "ACCOUNT_ID": "B",
            "CONTRACT_ID": "B-cancelled-no-term",
            "CONTRACT_STATUS": "Cancelled",
            "CONTRACT_START_DATE": T0,
            "BILLING_TERMINATION_DATE": None,
        },
        {
            "ACCOUNT_ID": "C",
            "CONTRACT_ID": "C-inverted",
            "CONTRACT_STATUS": "Cancelled",
            "CONTRACT_START_DATE": T0 + 30 * DAY,
            "BILLING_TERMINATION_DATE": T0,
        },
        {
            "ACCOUNT_ID": "D",
            "CONTRACT_ID": "D-valid",
            "CONTRACT_STATUS": "Cancelled",
            "CONTRACT_START_DATE": T0,
            "BILLING_TERMINATION_DATE": T0 + 30 * DAY,
        },
    ]


class TestHygieneOracle:
    """Sanity-check the first-principles oracle before using it in tests."""

    def test_oracle_counts_all_three_corruption_classes(self, four_row_fixture):
        n = _hygiene_oracle(four_row_fixture, config=_base_config())
        assert n == 3

    def test_oracle_clean_fixture_reports_zero(self):
        T0 = native_pd.Timestamp("2024-01-01")
        DAY = native_pd.Timedelta(days=1)
        rows = [{
            "ACCOUNT_ID": "X",
            "CONTRACT_ID": "X-1",
            "CONTRACT_STATUS": "Active",
            "CONTRACT_START_DATE": T0,
            "BILLING_TERMINATION_DATE": None,
        }]
        assert _hygiene_oracle(rows, config=_base_config()) == 0


class TestSkipPolicyCoversAllCorruptionClasses:
    """Gate checks — all three corruption classes are filtered under skip."""

    def test_skip_drops_null_start_row(self, df_factory):
        T0 = native_pd.Timestamp("2024-01-01")
        rows = [{
            "ACCOUNT_ID": "A",
            "CONTRACT_ID": "A-null",
            "CONTRACT_STATUS": "Active",
            "CONTRACT_START_DATE": None,
            "BILLING_TERMINATION_DATE": None,
        }, {
            "ACCOUNT_ID": "B",
            "CONTRACT_ID": "B-ok",
            "CONTRACT_STATUS": "Active",
            "CONTRACT_START_DATE": T0,
            "BILLING_TERMINATION_DATE": None,
        }]
        df = df_factory(rows)
        out = _to_native(enrich_lifecycle_dataset(df, config=_base_config(on_corrupt_row="skip")))
        assert len(out) == 1
        assert out["event_timestamp"].notna().all()
        assert (out["CONTRACT_ID"] == "B-ok").all()

    def test_skip_drops_terminal_null_valid_to_row(self, df_factory):
        T0 = native_pd.Timestamp("2024-01-01")
        DAY = native_pd.Timedelta(days=1)
        rows = [{
            "ACCOUNT_ID": "A",
            "CONTRACT_ID": "A-term-null",
            "CONTRACT_STATUS": "Cancelled",
            "CONTRACT_START_DATE": T0,
            "BILLING_TERMINATION_DATE": None,
        }, {
            "ACCOUNT_ID": "B",
            "CONTRACT_ID": "B-ok",
            "CONTRACT_STATUS": "Cancelled",
            "CONTRACT_START_DATE": T0,
            "BILLING_TERMINATION_DATE": T0 + 30 * DAY,
        }]
        df = df_factory(rows)
        out = _to_native(enrich_lifecycle_dataset(df, config=_base_config(on_corrupt_row="skip")))
        assert len(out) == 2
        assert out["event_timestamp"].notna().all()
        assert set(out["CONTRACT_ID"]) == {"B-ok"}

    def test_skip_still_drops_inverted_row(self, df_factory):
        T0 = native_pd.Timestamp("2024-01-01")
        DAY = native_pd.Timedelta(days=1)
        rows = [{
            "ACCOUNT_ID": "A",
            "CONTRACT_ID": "A-inv",
            "CONTRACT_STATUS": "Cancelled",
            "CONTRACT_START_DATE": T0 + 30 * DAY,
            "BILLING_TERMINATION_DATE": T0,
        }]
        df = df_factory(rows)
        out = _to_native(enrich_lifecycle_dataset(df, config=_base_config(on_corrupt_row="skip")))
        assert len(out) == 0

    def test_skip_four_class_fixture_leaves_only_valid_row(
        self, df_factory, four_row_fixture
    ):
        cfg = _base_config(on_corrupt_row="skip")
        df = df_factory(four_row_fixture)
        out = _to_native(enrich_lifecycle_dataset(df, config=cfg))
        assert set(out["CONTRACT_ID"]) == {"D-valid"}
        assert len(out) == 2
        assert out["event_timestamp"].notna().all()
        assert _hygiene_oracle(four_row_fixture, config=cfg) == 3


class TestRaisePolicyReportsAllCorruptionClasses:
    def test_raise_reports_null_start_as_corrupt(self, df_factory):
        rows = [{
            "ACCOUNT_ID": "A",
            "CONTRACT_ID": "A-null",
            "CONTRACT_STATUS": "Active",
            "CONTRACT_START_DATE": None,
            "BILLING_TERMINATION_DATE": None,
        }]
        df = df_factory(rows)
        with pytest.raises(ValueError, match="corrupt lifecycle"):
            enrich_lifecycle_dataset(df, config=_base_config(on_corrupt_row="raise"))

    def test_raise_reports_terminal_null_valid_to_as_corrupt(self, df_factory):
        T0 = native_pd.Timestamp("2024-01-01")
        rows = [{
            "ACCOUNT_ID": "A",
            "CONTRACT_ID": "A-term-null",
            "CONTRACT_STATUS": "Cancelled",
            "CONTRACT_START_DATE": T0,
            "BILLING_TERMINATION_DATE": None,
        }]
        df = df_factory(rows)
        with pytest.raises(ValueError, match="corrupt lifecycle"):
            enrich_lifecycle_dataset(df, config=_base_config(on_corrupt_row="raise"))

    def test_raise_clean_fixture_does_not_raise(self, df_factory):
        T0 = native_pd.Timestamp("2024-01-01")
        DAY = native_pd.Timedelta(days=1)
        rows = [{
            "ACCOUNT_ID": "X",
            "CONTRACT_ID": "X-1",
            "CONTRACT_STATUS": "Cancelled",
            "CONTRACT_START_DATE": T0,
            "BILLING_TERMINATION_DATE": T0 + 30 * DAY,
        }]
        df = df_factory(rows)
        out = _to_native(enrich_lifecycle_dataset(df, config=_base_config(on_corrupt_row="raise")))
        assert len(out) == 2


class TestWarnPolicyDropsNonInvertedCorruption:
    """warn keeps inverted rows (clamp to valid_from) but drops NULL-start
    and terminal-NULL-terminate rows — they have no sensible clamp target."""

    def test_warn_drops_null_start_and_keeps_clamped_inverted(self, df_factory):
        T0 = native_pd.Timestamp("2024-01-01")
        DAY = native_pd.Timedelta(days=1)
        rows = [{
            "ACCOUNT_ID": "A",
            "CONTRACT_ID": "A-null",
            "CONTRACT_STATUS": "Active",
            "CONTRACT_START_DATE": None,
            "BILLING_TERMINATION_DATE": None,
        }, {
            "ACCOUNT_ID": "B",
            "CONTRACT_ID": "B-inv",
            "CONTRACT_STATUS": "Cancelled",
            "CONTRACT_START_DATE": T0 + 30 * DAY,
            "BILLING_TERMINATION_DATE": T0,
        }]
        df = df_factory(rows)
        out = _to_native(enrich_lifecycle_dataset(df, config=_base_config(on_corrupt_row="warn")))
        assert (out["CONTRACT_ID"] != "A-null").all()
        assert out["event_timestamp"].notna().all()
        b_rows = out[out["CONTRACT_ID"] == "B-inv"]
        assert len(b_rows) == 2
        start_ts = native_pd.Timestamp(b_rows[b_rows["event_type"] == "start"]["event_timestamp"].iloc[0])
        term_ts = native_pd.Timestamp(b_rows[b_rows["event_type"] == "terminate"]["event_timestamp"].iloc[0])
        assert term_ts == start_ts


class TestRegressionContractEvidence:
    """Reproduces the shape of the 2/23,290 NULL event_timestamp escape
    observed on the engagement's run artifacts (diagnosis §7B). Synthetic
    only — no client-identifying values."""

    def test_contract_landing_has_no_null_event_timestamp_after_skip(self, df_factory):
        T0 = native_pd.Timestamp("2024-01-01")
        DAY = native_pd.Timedelta(days=1)
        valid_rows = [
            {
                "ACCOUNT_ID": f"A-{i}",
                "CONTRACT_ID": f"A-{i}-1",
                "CONTRACT_STATUS": "Active" if i % 3 else "Cancelled",
                "CONTRACT_START_DATE": T0 + i * DAY,
                "BILLING_TERMINATION_DATE": (T0 + (i + 30) * DAY) if i % 3 == 0 else None,
            }
            for i in range(30)
        ]
        escape_rows = [
            {
                "ACCOUNT_ID": "ESC-1",
                "CONTRACT_ID": "ESC-1-null-start",
                "CONTRACT_STATUS": "Active",
                "CONTRACT_START_DATE": None,
                "BILLING_TERMINATION_DATE": None,
            },
            {
                "ACCOUNT_ID": "ESC-2",
                "CONTRACT_ID": "ESC-2-null-start",
                "CONTRACT_STATUS": "Cancelled",
                "CONTRACT_START_DATE": None,
                "BILLING_TERMINATION_DATE": T0 + 5 * DAY,
            },
        ]
        df = df_factory(valid_rows + escape_rows)
        out = _to_native(
            enrich_lifecycle_dataset(df, config=_base_config(on_corrupt_row="skip"))
        )
        assert out["event_timestamp"].notna().all()
        assert not out["CONTRACT_ID"].str.startswith("ESC-").any()
