"""SCD history reconstruction.

Reconstructs per-(parent_record, grid_date) field state from a Salesforce-style
change-log table via a backward-asof window. The output is a wide
``(parent_record_key, as_of_date, *tracked_fields)`` view that the framework's
existing temporal merger handles via the standard equi-join branch.

Quick Start::

    from customer_retention.stages.scd_history import (
        SCDHistoryReconstructionConfig, reconstruct_scd_history_at_grid,
    )

    config = SCDHistoryReconstructionConfig(
        enriched_view_name="sps_reconstructed_case_history",
        parent_record_key="CASE_ID",
        field_column="FIELD",
        new_value_column="NEW_VALUE",
        old_value_column="OLD_VALUE",
        change_timestamp_column="CREATED_DATE",
        unique_row_id_column="CASE_HISTORY_ID",
        tracked_fields=("Status", "Priority"),
        parent_table_dataset_name="case",
        parent_creation_timestamp_column="CREATED_DATE",
        parent_value_columns=(("Status", "CASE_STATUS"), ("Priority", "PRIORITY")),
    )
    reconstructed = reconstruct_scd_history_at_grid(
        history_df=case_history_df,
        grid_dates=snapshot_grid.grid_dates,
        config=config,
        parent_df=case_df,
    )
"""

from .augment import augment_parent_with_scd_state
from .config import SCDHistoryReconstructionConfig
from .reconstruct import reconstruct_scd_history_at_grid

__all__ = [
    "SCDHistoryReconstructionConfig",
    "augment_parent_with_scd_state",
    "reconstruct_scd_history_at_grid",
]
