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
