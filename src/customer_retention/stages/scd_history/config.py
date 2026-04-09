"""Configuration for SCD history reconstruction.

A ``SCDHistoryReconstructionConfig`` describes how to reconstruct per-grid-date
field state from a slowly-changing-dimension change-log table (Salesforce
field history, audit-trigger tables, CDC streams — any shape with one row per
``(parent, field, new_value, change_timestamp)``). For each tracked field
``F`` and parent record ``R`` the reconstructor produces, at every grid anchor
``G``, the new-value of the most recent change with
``change_timestamp <= G`` (backward-asof semantics), with optional fallback
to a parent-table column when the record has no recorded changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Literal, Optional, Tuple

OnUnknownField = Literal["raise", "skip", "warn"]
_VALID_UNKNOWN_FIELD_POLICIES: Tuple[str, ...] = ("raise", "skip", "warn")


@dataclass(frozen=True)
class SCDHistoryReconstructionConfig:
    enriched_view_name: str
    parent_record_key: str
    field_column: str
    new_value_column: str
    change_timestamp_column: str
    tracked_fields: Tuple[str, ...]
    old_value_column: Optional[str] = None
    parent_table_dataset_name: Optional[str] = None
    parent_creation_timestamp_column: Optional[str] = None
    parent_value_columns: Tuple[Tuple[str, str], ...] = ()
    on_unknown_field: OnUnknownField = "raise"
    output_record_key: Optional[str] = None
    unique_row_id_column: Optional[str] = None

    def __post_init__(self) -> None:
        for attr in (
            "enriched_view_name",
            "parent_record_key",
            "field_column",
            "new_value_column",
            "change_timestamp_column",
        ):
            if not getattr(self, attr):
                raise ValueError(f"{attr} must not be empty")
        if not self.tracked_fields:
            raise ValueError("tracked_fields must not be empty")
        if self.on_unknown_field not in _VALID_UNKNOWN_FIELD_POLICIES:
            raise ValueError(
                f"on_unknown_field must be one of {_VALID_UNKNOWN_FIELD_POLICIES}, "
                f"got {self.on_unknown_field!r}"
            )
        if self.parent_value_columns:
            if not self.parent_table_dataset_name:
                raise ValueError(
                    "parent_value_columns requires parent_table_dataset_name to be set"
                )
            tracked_set = set(self.tracked_fields)
            for field_name, _parent_col in self.parent_value_columns:
                if field_name not in tracked_set:
                    raise ValueError(
                        f"parent_value_columns references {field_name!r} which is "
                        f"not in tracked_fields"
                    )
        if self.unique_row_id_column is not None:
            if not self.unique_row_id_column or not self.unique_row_id_column.strip():
                raise ValueError(
                    "unique_row_id_column must not be empty or whitespace-only"
                )

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        raw["tracked_fields"] = list(raw["tracked_fields"])
        raw["parent_value_columns"] = [list(pair) for pair in raw["parent_value_columns"]]
        return raw

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SCDHistoryReconstructionConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise TypeError(
                f"unknown SCDHistoryReconstructionConfig field(s): {sorted(unknown)}"
            )
        kwargs: Dict[str, Any] = dict(data)
        if "tracked_fields" in kwargs and kwargs["tracked_fields"] is not None:
            kwargs["tracked_fields"] = tuple(kwargs["tracked_fields"])
        if "parent_value_columns" in kwargs and kwargs["parent_value_columns"] is not None:
            kwargs["parent_value_columns"] = tuple(
                tuple(pair) for pair in kwargs["parent_value_columns"]
            )
        return cls(**kwargs)
