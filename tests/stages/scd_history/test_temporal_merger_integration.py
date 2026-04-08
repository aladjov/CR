"""Integration test: reconstructed SCD history → TemporalMerger.

Validates that the wide ``(parent_record, as_of_date, *fields)`` view produced
by ``reconstruct_scd_history_at_grid`` is consumable by the existing
``TemporalMerger`` via the equi-join branch (Case B at
``temporal_merger.py:130-141``) — i.e., no merger changes are needed.
"""
from __future__ import annotations

from customer_retention.core.compat import native_pd
from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.stages.scd_history import (
    SCDHistoryReconstructionConfig,
    reconstruct_scd_history_at_grid,
)
from customer_retention.stages.temporal.temporal_merger import (
    DatasetMergeInput,
    MergeConfig,
    TemporalMerger,
)


def _build_history() -> native_pd.DataFrame:
    return native_pd.DataFrame(
        [
            {
                "CASE_HISTORY_ID": "h-1",
                "CASE_ID": "A",
                "FIELD": "Status",
                "OLD_VALUE": None,
                "NEW_VALUE": "Open",
                "CREATED_DATE": native_pd.Timestamp("2024-01-05"),
            },
            {
                "CASE_HISTORY_ID": "h-2",
                "CASE_ID": "A",
                "FIELD": "Status",
                "OLD_VALUE": "Open",
                "NEW_VALUE": "Closed",
                "CREATED_DATE": native_pd.Timestamp("2024-02-15"),
            },
            {
                "CASE_HISTORY_ID": "h-3",
                "CASE_ID": "B",
                "FIELD": "Status",
                "OLD_VALUE": None,
                "NEW_VALUE": "InProgress",
                "CREATED_DATE": native_pd.Timestamp("2024-01-10"),
            },
        ]
    )


def _build_parent() -> native_pd.DataFrame:
    return native_pd.DataFrame(
        [
            {"CASE_ID": "A", "CASE_STATUS": "Open"},
            {"CASE_ID": "B", "CASE_STATUS": "InProgress"},
        ]
    )


def _build_parent_with_static_attrs() -> native_pd.DataFrame:
    """Parent with both reconstructable (CASE_STATUS) and static-only
    (CASE_TYPE, OWNER_EMPLOYEE_ID) attribute columns — exactly the shape the
    SPS section 0.13.5 cell joins state into."""
    return native_pd.DataFrame(
        [
            {
                "CASE_ID": "A",
                "ACCOUNT_ID": "acc-1",
                "CASE_STATUS": "Open",
                "CASE_TYPE": "Support",
                "OWNER_EMPLOYEE_ID": "emp-100",
            },
            {
                "CASE_ID": "B",
                "ACCOUNT_ID": "acc-2",
                "CASE_STATUS": "InProgress",
                "CASE_TYPE": "Billing",
                "OWNER_EMPLOYEE_ID": "emp-200",
            },
        ]
    )


def _build_config() -> SCDHistoryReconstructionConfig:
    return SCDHistoryReconstructionConfig(
        enriched_view_name="reconstructed_case_history",
        parent_record_key="CASE_ID",
        field_column="FIELD",
        new_value_column="NEW_VALUE",
        old_value_column="OLD_VALUE",
        change_timestamp_column="CREATED_DATE",
        unique_row_id_column="CASE_HISTORY_ID",
        tracked_fields=("Status",),
        parent_table_dataset_name="case",
        parent_creation_timestamp_column="CREATED_DATE",
        parent_value_columns=(("Status", "CASE_STATUS"),),
    )


class TestReconstructedViewMergesViaEquiJoin:
    def test_event_level_equi_join_attaches_field_state(self):
        grid_dates = [
            native_pd.Timestamp("2024-01-01"),
            native_pd.Timestamp("2024-02-01"),
            native_pd.Timestamp("2024-03-01"),
        ]

        reconstructed = reconstruct_scd_history_at_grid(
            _build_history(), grid_dates, _build_config(), _build_parent()
        )
        # The reconstructor returns CASE_ID; merger expects entity_id by default.
        reconstructed = reconstructed.rename(columns={"CASE_ID": "entity_id"})

        merger = TemporalMerger(MergeConfig(entity_key="entity_id"))
        spine = merger.build_spine(["A", "B"], grid_dates)

        merged, _report = merger.merge_all(
            spine,
            [
                DatasetMergeInput(
                    name="case_history",
                    df=reconstructed,
                    granularity=DatasetGranularity.EVENT_LEVEL,
                    feature_timestamp_column=None,
                )
            ],
        )

        assert len(merged) == len(spine)
        assert "Status" in merged.columns
        # raw change-log columns are NOT present
        for forbidden in ("FIELD", "OLD_VALUE", "NEW_VALUE", "CASE_HISTORY_ID"):
            assert forbidden not in merged.columns

        a_jan = merged[
            (merged["entity_id"] == "A")
            & (merged["as_of_date"] == native_pd.Timestamp("2024-01-01"))
        ]
        assert a_jan["Status"].iloc[0] == "Open"

        a_feb = merged[
            (merged["entity_id"] == "A")
            & (merged["as_of_date"] == native_pd.Timestamp("2024-02-01"))
        ]
        assert a_feb["Status"].iloc[0] == "Open"

        a_mar = merged[
            (merged["entity_id"] == "A")
            & (merged["as_of_date"] == native_pd.Timestamp("2024-03-01"))
        ]
        assert a_mar["Status"].iloc[0] == "Closed"


class TestParentJoinShape:
    """Verify that joining the reconstructed state with the static parent
    table produces a per-(parent, anchor) view with N rows per parent
    (one per grid anchor), each carrying both static and reconstructed
    columns. This mirrors the SPS section 0.13.5 cell flow exactly."""

    def test_each_parent_row_becomes_n_rows_one_per_anchor(self):
        grid_dates = [
            native_pd.Timestamp("2024-01-01"),
            native_pd.Timestamp("2024-02-01"),
            native_pd.Timestamp("2024-03-01"),
        ]
        parent = _build_parent_with_static_attrs()
        history = _build_history()
        config = _build_config()

        state = reconstruct_scd_history_at_grid(history, grid_dates, config, parent)

        # Drop overlapping reconstructable column from parent before join (the
        # SPS cell does this so the reconstructed Status is the source of
        # truth at each anchor; otherwise the join produces duplicate columns).
        parent_static = parent.drop(columns=["CASE_STATUS"])
        augmented = parent_static.merge(
            state, on=config.parent_record_key, how="inner"
        )

        # Each parent row → N rows (one per grid anchor)
        assert len(augmented) == len(parent) * len(grid_dates)
        assert (augmented.groupby("CASE_ID").size() == len(grid_dates)).all()

        # Static parent columns are present and broadcast across anchors
        for col in ("ACCOUNT_ID", "CASE_TYPE", "OWNER_EMPLOYEE_ID"):
            assert col in augmented.columns
            for case_id in ("A", "B"):
                values = augmented[augmented["CASE_ID"] == case_id][col].unique()
                assert len(values) == 1, f"{col} should be constant per case"

        # Reconstructed state column is present and varies across anchors
        assert "Status" in augmented.columns

        # Spot-check Case A: Status changes from Open (parent fallback for
        # 2024-01-01 anchor) → Open (post first change at 2024-01-05) → Closed
        # (post second change at 2024-02-15)
        a_rows = augmented[augmented["CASE_ID"] == "A"].sort_values("as_of_date")
        statuses = list(a_rows["Status"])
        assert statuses == ["Open", "Open", "Closed"]

    def test_augmented_view_drops_raw_changelog_columns(self):
        grid_dates = [
            native_pd.Timestamp("2024-01-01"),
            native_pd.Timestamp("2024-02-01"),
        ]
        parent = _build_parent_with_static_attrs()
        history = _build_history()
        config = _build_config()

        state = reconstruct_scd_history_at_grid(history, grid_dates, config, parent)
        augmented = parent.drop(columns=["CASE_STATUS"]).merge(
            state, on=config.parent_record_key, how="inner"
        )

        # Raw change-log columns must NOT leak into the augmented view
        for forbidden in ("FIELD", "OLD_VALUE", "NEW_VALUE", "CASE_HISTORY_ID"):
            assert forbidden not in augmented.columns

    def test_inner_join_drops_history_orphans(self):
        """Cases that exist in history but not in the (filtered) parent are
        dropped — exactly the ~28% case_history orphan drop documented in
        the SPS sanity report."""
        grid_dates = [native_pd.Timestamp("2024-02-01")]
        # History has Case Z but parent does not
        history = native_pd.concat(
            [
                _build_history(),
                native_pd.DataFrame(
                    [
                        {
                            "CASE_HISTORY_ID": "h-orphan",
                            "CASE_ID": "Z",
                            "FIELD": "Status",
                            "OLD_VALUE": None,
                            "NEW_VALUE": "Pending",
                            "CREATED_DATE": native_pd.Timestamp("2024-01-15"),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        parent = _build_parent_with_static_attrs()  # No Case Z
        config = _build_config()

        state = reconstruct_scd_history_at_grid(history, grid_dates, config, parent)
        augmented = parent.drop(columns=["CASE_STATUS"]).merge(
            state, on=config.parent_record_key, how="inner"
        )

        # Case Z is dropped — only A and B survive
        assert set(augmented["CASE_ID"]) == {"A", "B"}
        assert "Z" not in set(augmented["CASE_ID"])

    def test_augmented_view_round_trips_through_temporal_merger(self):
        """End-to-end: augmented per-(parent, anchor) view → TemporalMerger
        equi-join branch → silver-shaped output. This is the contract the
        SPS cell relies on."""
        grid_dates = [
            native_pd.Timestamp("2024-01-01"),
            native_pd.Timestamp("2024-02-01"),
            native_pd.Timestamp("2024-03-01"),
        ]
        parent = _build_parent_with_static_attrs()
        history = _build_history()
        config = _build_config()

        state = reconstruct_scd_history_at_grid(history, grid_dates, config, parent)
        augmented = parent.drop(columns=["CASE_STATUS"]).merge(
            state, on=config.parent_record_key, how="inner"
        )
        # Frameworks expects entity_id; map ACCOUNT_ID -> entity_id
        augmented = augmented.rename(columns={"ACCOUNT_ID": "entity_id"})

        merger = TemporalMerger(MergeConfig(entity_key="entity_id"))
        spine = merger.build_spine(["acc-1", "acc-2"], grid_dates)

        merged, report = merger.merge_all(
            spine,
            [
                DatasetMergeInput(
                    name="case",
                    df=augmented,
                    granularity=DatasetGranularity.EVENT_LEVEL,
                    feature_timestamp_column=None,
                )
            ],
        )

        # The merger took the equi-join branch (one row per spine row)
        assert len(merged) == len(spine)
        # Both static and reconstructed columns survived the merge
        for col in ("CASE_TYPE", "OWNER_EMPLOYEE_ID", "Status"):
            assert col in merged.columns
        assert "case" in report.datasets_merged
