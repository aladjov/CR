"""SCD history reconstruction.

Reconstructs per-(parent_record, grid_date) field state from a slowly-changing-
dimension change-log table via a backward-asof window. Works with any change-
log shape (Salesforce-style ``FIELD/NEW_VALUE`` rows, audit-trigger tables,
CDC streams). The output is a wide
``(parent_record_key, as_of_date, *tracked_fields)`` view that the framework's
existing temporal merger handles via the standard equi-join branch.

Quick Start::

    from customer_retention.stages.scd_history import (
        SCDHistoryReconstructionConfig, reconstruct_scd_history_at_grid,
    )

    config = SCDHistoryReconstructionConfig(
        enriched_view_name="reconstructed_parent_history",
        parent_record_key="parent_id",
        field_column="field_name",
        new_value_column="new_value",
        old_value_column="old_value",
        change_timestamp_column="changed_at",
        unique_row_id_column="change_id",
        tracked_fields=("status", "priority"),
        parent_table_dataset_name="parent",
        parent_creation_timestamp_column="created_at",
        parent_value_columns=(("status", "current_status"), ("priority", "current_priority")),
    )
    reconstructed = reconstruct_scd_history_at_grid(
        history_df=change_log_df,
        grid_dates=snapshot_grid.grid_dates,
        config=config,
        parent_df=parent_df,
    )
"""

from .augment import (
    augment_and_persist_parent_dataset,
    augment_parent_with_scd_state,
)
from .config import SCDHistoryReconstructionConfig
from .reconstruct import reconstruct_scd_history_at_grid

__all__ = [
    "SCDHistoryReconstructionConfig",
    "augment_and_persist_parent_dataset",
    "augment_parent_with_scd_state",
    "reconstruct_scd_history_at_grid",
]
