"""Tests for the VALUE_COLUMN retention guard (Cycle 7 — G9).

A dataset's ``VALUE_COLUMN`` is the raw numeric column bronze rolls up into
``{sum, mean, max, count}_{window}`` aggregates. It is sometimes separately
prefix-excluded in gold (zero-on-termination leak path), but the *raw* column
must reach landing so bronze has something to aggregate.

Cycle 7 (run ``spschurn-e4ad6e1b``) lost ``NET_PRICE`` from
``landing/subscription`` despite the documented override. The engagement has
multiple drop surfaces:

1. ``LifecycleEnrichmentConfig.drop_columns`` in NB00 user-code (runs at
   ``global_temp.sps_enriched_<dataset>`` time, *before* NB01).
2. NB01 ``DROP_COLUMNS[<dataset>]`` — the "normal" drop list.
3. NB01 ``AUTO_DROP_TEXT_COLUMNS`` — numeric-safe; not a VALUE_COLUMN risk.

The guard cross-checks a declared list of value columns against any subset of
these surfaces and fails loud with every offending (dataset, column, surface)
triple in the message so the operator can go fix the right cell.

Tests skip gracefully pre-fix via ``ImportError`` so this file lands before
the framework module does.
"""
from __future__ import annotations

import pytest

pytest.importorskip(
    "customer_retention.stages.lifecycle.validation",
    reason="value-column retention guard not yet implemented (Cycle 7 pre-fix)",
)

from customer_retention.stages.lifecycle.config import LifecycleEnrichmentConfig
from customer_retention.stages.lifecycle.validation import (
    ValueColumnDropError,
    assert_value_columns_retained,
)


def _oracle_conflicts(
    value_columns, drop_surfaces: dict[str, list[str]]
) -> list[tuple[str, str]]:
    """Pure first-principles: every (column, surface) pair where column appears."""
    wanted = set(value_columns)
    return sorted(
        (col, surface)
        for surface, dropped in drop_surfaces.items()
        for col in dropped
        if col in wanted
    )


def _minimal_lifecycle_kwargs(**overrides):
    base = dict(
        enriched_view_name="sps_enriched_subscription",
        parent_entity_key="ACCOUNT_ID",
        sub_entity_key="SUBSCRIPTION_ID",
        valid_from_column="SUBSCRIPTION_START_DATE",
        valid_to_columns=("TERMINATED_DATE",),
    )
    base.update(overrides)
    return base


class TestAssertValueColumnsRetained:
    def test_no_drops_no_raise(self):
        assert_value_columns_retained(
            dataset="subscription",
            value_columns=("NET_PRICE",),
            drop_surfaces={"NB01_DROP_COLUMNS": [], "enrich_drop_columns": []},
        )

    def test_value_column_not_in_any_surface_passes(self):
        assert_value_columns_retained(
            dataset="subscription",
            value_columns=("NET_PRICE",),
            drop_surfaces={
                "NB01_DROP_COLUMNS": ["IS_ACTIVE", "SUBSCRIPTION_STATUS"],
                "enrich_drop_columns": ["TERMINATED_DATE"],
            },
        )

    def test_value_column_in_nb01_drop_raises(self):
        with pytest.raises(ValueColumnDropError) as exc:
            assert_value_columns_retained(
                dataset="subscription",
                value_columns=("NET_PRICE",),
                drop_surfaces={"NB01_DROP_COLUMNS": ["NET_PRICE", "IS_ACTIVE"]},
            )
        msg = str(exc.value)
        assert "subscription" in msg
        assert "NET_PRICE" in msg
        assert "NB01_DROP_COLUMNS" in msg

    def test_value_column_in_lifecycle_drop_raises(self):
        with pytest.raises(ValueColumnDropError) as exc:
            assert_value_columns_retained(
                dataset="subscription",
                value_columns=("NET_PRICE",),
                drop_surfaces={
                    "NB01_DROP_COLUMNS": ["IS_ACTIVE"],
                    "enrich_drop_columns": ["TERMINATED_DATE", "NET_PRICE"],
                },
            )
        assert "enrich_drop_columns" in str(exc.value)

    def test_multi_surface_multi_column_lists_every_conflict(self):
        with pytest.raises(ValueColumnDropError) as exc:
            assert_value_columns_retained(
                dataset="subscription",
                value_columns=("NET_PRICE", "QUANTITY"),
                drop_surfaces={
                    "NB01_DROP_COLUMNS": ["NET_PRICE"],
                    "enrich_drop_columns": ["QUANTITY"],
                },
            )
        msg = str(exc.value)
        assert "NET_PRICE" in msg and "QUANTITY" in msg
        assert "NB01_DROP_COLUMNS" in msg and "enrich_drop_columns" in msg

    def test_empty_value_columns_is_noop(self):
        assert_value_columns_retained(
            dataset="case",
            value_columns=(),
            drop_surfaces={"NB01_DROP_COLUMNS": ["NET_PRICE"]},
        )

    def test_case_sensitive_by_design(self):
        assert_value_columns_retained(
            dataset="subscription",
            value_columns=("NET_PRICE",),
            drop_surfaces={"NB01_DROP_COLUMNS": ["net_price"]},
        )

    def test_conflicts_match_oracle(self):
        surfaces = {
            "NB01_DROP_COLUMNS": ["NET_PRICE", "FOO"],
            "enrich_drop_columns": ["QUANTITY", "BAR"],
        }
        wanted = ("NET_PRICE", "QUANTITY", "UNRELATED")
        expected = _oracle_conflicts(wanted, surfaces)
        with pytest.raises(ValueColumnDropError) as exc:
            assert_value_columns_retained(
                dataset="subscription", value_columns=wanted, drop_surfaces=surfaces,
            )
        for col, surface in expected:
            assert col in str(exc.value)
            assert surface in str(exc.value)


class TestLifecycleEnrichmentConfigProtectedColumns:
    def test_minimal_no_protected_is_noop(self):
        cfg = LifecycleEnrichmentConfig(**_minimal_lifecycle_kwargs())
        assert cfg.protected_columns == ()

    def test_protected_column_not_in_drop_accepted(self):
        cfg = LifecycleEnrichmentConfig(
            **_minimal_lifecycle_kwargs(
                drop_columns=("TERMINATED_DATE",),
                protected_columns=("NET_PRICE",),
            )
        )
        assert cfg.protected_columns == ("NET_PRICE",)
        assert cfg.drop_columns == ("TERMINATED_DATE",)

    def test_protected_column_in_drop_rejected(self):
        with pytest.raises(ValueError, match="protected_columns"):
            LifecycleEnrichmentConfig(
                **_minimal_lifecycle_kwargs(
                    drop_columns=("NET_PRICE", "TERMINATED_DATE"),
                    protected_columns=("NET_PRICE",),
                )
            )

    def test_protected_column_round_trip(self):
        cfg = LifecycleEnrichmentConfig(
            **_minimal_lifecycle_kwargs(
                drop_columns=("TERMINATED_DATE",),
                protected_columns=("NET_PRICE", "QUANTITY"),
            )
        )
        rebuilt = LifecycleEnrichmentConfig.from_dict(cfg.to_dict())
        assert rebuilt == cfg
        assert isinstance(rebuilt.protected_columns, tuple)


class TestRegressionG9:
    """Reproduce the exact C7 failure shape with synthetic inputs."""

    def test_subscription_net_price_dropped_by_enrich_raises(self):
        """Mirror of run ``spschurn-e4ad6e1b`` §4: NET_PRICE in the
        enrichment drop surface → the guard surfaces the exact (dataset,
        column, surface) triple the operator needs to fix."""
        with pytest.raises(ValueColumnDropError) as exc:
            assert_value_columns_retained(
                dataset="subscription",
                value_columns=("NET_PRICE",),
                drop_surfaces={
                    "enrich_drop_columns": [
                        "TERMINATED_DATE",
                        "SUBSCRIPTION_STATUS",
                        "IS_ACTIVE",
                        "NET_PRICE",
                    ],
                    "NB01_DROP_COLUMNS": [
                        "ETL_CREATED_TIMESTAMP",
                        "SUBSCRIPTION_ID",
                    ],
                },
            )
        msg = str(exc.value)
        assert "subscription" in msg
        assert "NET_PRICE" in msg
        assert "enrich_drop_columns" in msg

    def test_subscription_net_price_dropped_by_nb01_raises(self):
        with pytest.raises(ValueColumnDropError) as exc:
            assert_value_columns_retained(
                dataset="subscription",
                value_columns=("NET_PRICE",),
                drop_surfaces={
                    "enrich_drop_columns": ["TERMINATED_DATE"],
                    "NB01_DROP_COLUMNS": [
                        "ETL_CREATED_TIMESTAMP",
                        "SUBSCRIPTION_ID",
                        "NET_PRICE",
                    ],
                },
            )
        assert "NB01_DROP_COLUMNS" in str(exc.value)
        assert "NET_PRICE" in str(exc.value)

    def test_clean_subscription_passes(self):
        """Post-fix: the documented overrides doc layout has no NET_PRICE
        in any drop surface. Guard stays silent."""
        assert_value_columns_retained(
            dataset="subscription",
            value_columns=("NET_PRICE",),
            drop_surfaces={
                "enrich_drop_columns": [
                    "TERMINATED_DATE", "SUBSCRIPTION_STATUS", "IS_ACTIVE",
                    "MONTHLY_NET_TAKE_RATE", "SUBSCRIPTION_END_DATE",
                    "TOTAL_DOCUMENTS", "OVERAGE_RATE", "AD_SPEND",
                ],
                "NB01_DROP_COLUMNS": [
                    "ETL_CREATED_TIMESTAMP", "ETL_UPDATED_TIMESTAMP",
                    "SUBSCRIPTION_ID", "IS_ACTIVE", "SUBSCRIPTION_STATUS",
                    "MONTHLY_NET_TAKE_RATE", "SUBSCRIPTION_END_DATE",
                    "TOTAL_DOCUMENTS", "OVERAGE_RATE", "AD_SPEND",
                ],
            },
        )
