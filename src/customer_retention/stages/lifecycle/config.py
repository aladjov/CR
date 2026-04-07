"""Configuration for lifecycle enrichment.

A `LifecycleEnrichmentConfig` describes how to convert an entity-with-lifespan
table (one row per service unit, with a start timestamp and an optional
termination timestamp) into a doubled event stream — one START event per row
plus one TERMINATE event per terminated row. The framework's existing
event-level pipeline then handles the rest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Literal, Optional, Tuple

OnCorruptRow = Literal["raise", "skip", "warn"]
_VALID_CORRUPT_POLICIES: Tuple[str, ...] = ("raise", "skip", "warn")


@dataclass(frozen=True)
class LifecycleEnrichmentConfig:
    enriched_view_name: str
    parent_entity_key: str
    sub_entity_key: str
    valid_from_column: str
    valid_to_columns: Tuple[str, ...]
    status_column: Optional[str] = None
    terminal_status_values: Tuple[str, ...] = ()
    drop_columns: Tuple[str, ...] = ()
    on_corrupt_row: OnCorruptRow = "raise"
    event_type_column: str = "event_type"
    event_timestamp_column: str = "event_timestamp"

    def __post_init__(self) -> None:
        if not self.enriched_view_name:
            raise ValueError("enriched_view_name must not be empty")
        if not self.parent_entity_key:
            raise ValueError("parent_entity_key must not be empty")
        if not self.sub_entity_key:
            raise ValueError("sub_entity_key must not be empty")
        if not self.valid_from_column:
            raise ValueError("valid_from_column must not be empty")
        if not self.valid_to_columns:
            raise ValueError("valid_to_columns must not be empty")
        if self.terminal_status_values and not self.status_column:
            raise ValueError(
                "terminal_status_values requires status_column to be set"
            )
        if self.status_column and not self.terminal_status_values:
            raise ValueError(
                "status_column requires terminal_status_values to be set"
            )
        if self.on_corrupt_row not in _VALID_CORRUPT_POLICIES:
            raise ValueError(
                f"on_corrupt_row must be one of {_VALID_CORRUPT_POLICIES}, "
                f"got {self.on_corrupt_row!r}"
            )
        if self.valid_from_column in self.drop_columns:
            raise ValueError(
                f"valid_from_column {self.valid_from_column!r} must not appear "
                f"in drop_columns — the doubled stream needs it for "
                f"event_timestamp on START rows"
            )

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        # Tuples → lists for JSON-friendly round-trip.
        for key in ("valid_to_columns", "terminal_status_values", "drop_columns"):
            raw[key] = list(raw[key])
        return raw

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LifecycleEnrichmentConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise TypeError(
                f"unknown LifecycleEnrichmentConfig field(s): {sorted(unknown)}"
            )
        kwargs: Dict[str, Any] = dict(data)
        for key in ("valid_to_columns", "terminal_status_values", "drop_columns"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)
