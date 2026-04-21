"""Fail-fast guards for lifecycle-stage column retention.

A dataset's ``VALUE_COLUMN`` (NET_PRICE on subscription, USD_BOOKINGS_* on
opportunity, etc.) is the raw numeric that bronze rolls up into windowed
aggregates. It must survive every drop surface upstream of the bronze
aggregator or the entire ``{VALUE}_{sum,mean,max,count}_{window}`` family
silently collapses to zero.

Cycle 7 (engagement ``engagement_e4ad6e1b``) lost subscription's
``NET_PRICE`` at landing; bronze then emitted 0 value-family aggregates. The
drop surfaces user-code cells have available today are not systematically
cross-checked against the declared VALUE_COLUMN list, so a column can slip
into any one of them unnoticed.

This module provides one function: ``assert_value_columns_retained`` — a
symmetric, surface-agnostic guard that fails loud with every offending
(column, surface) pair so the operator goes straight to the right cell.
"""
from __future__ import annotations

from typing import Iterable, Mapping


class ValueColumnDropError(ValueError):
    """Raised when a declared VALUE_COLUMN appears in any drop surface."""


def assert_value_columns_retained(
    *,
    dataset: str,
    value_columns: Iterable[str],
    drop_surfaces: Mapping[str, Iterable[str]],
) -> None:
    wanted = {c for c in value_columns if c}
    if not wanted:
        return
    conflicts = sorted(
        (col, surface)
        for surface, dropped in drop_surfaces.items()
        for col in dropped
        if col in wanted
    )
    if not conflicts:
        return
    lines = [f"  - {col!r} dropped by {surface!r}" for col, surface in conflicts]
    raise ValueColumnDropError(
        f"VALUE_COLUMN retention violation on dataset {dataset!r}:\n"
        + "\n".join(lines)
        + "\nRemove the column from the offending surface or stop declaring it "
        "as a VALUE_COLUMN."
    )
