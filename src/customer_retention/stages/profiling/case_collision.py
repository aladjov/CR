"""Case-collision detector for landing schemas (PA-3).

Spark folds bare-identifier column names case-insensitively, so a frame
with both ``opportunity_id`` and ``OPPORTUNITY_ID`` (a Snowflake
quoted-identifier source carrying both forms) writes to the same
underlying column at ``saveAsTable`` time and fails with
``[COLUMN_ALREADY_EXISTS]``. ``df.drop("opportunity_id")`` drops both
because the resolution itself is case-insensitive — only a case-EXACT
``df.select(*[c for c in df.columns if c not in {…}])`` keeps the
desired variant.

This module produces the deterministic side of the SPS engagement's
``LANDING_DROP_COLUMNS_OVERRIDES`` workaround: detect the collision at
NB02 source-integrity time, emit a `landing.drop_columns`
recommendation pre-populated with the redundant variant(s), and let
``RecommendationRegistry.add_landing_drop_columns`` carry it through
to NB10 codegen via the existing case-EXACT landing template.

The detector is intentionally narrow: it does NOT attempt to merge
values across case-variants, infer which side the operator wants to
keep, or rename the survivor. It returns a structured list of
collisions and lets the caller (NB02 cell or registry-init helper)
decide.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class CaseCollision:
    """One case-fold collision in a column list.

    Attributes:
        kept: The variant retained — first occurrence in the input order.
        dropped: Every other variant whose ``casefold()`` matches ``kept``.
    """

    kept: str
    dropped: List[str]


def detect_case_collisions(columns: Iterable[str]) -> List[CaseCollision]:
    """Return every case-fold collision in ``columns``.

    Iterates the column list once, preserving input order. The first
    occurrence of each casefolded form is the ``kept`` variant; any
    subsequent column that folds to the same form is appended to the
    matching collision's ``dropped`` list. Columns with a unique
    casefold yield no entry in the output.

    The casefold key uses :py:meth:`str.casefold` rather than
    :py:meth:`str.lower` to handle locale-dependent edge cases (e.g.
    German ß) that Spark's SQL-parser case-insensitive resolution also
    treats as collisions.

    Args:
        columns: Column names from a landing-time DataFrame, ideally in
            ``df.columns`` order so ``kept`` is deterministic relative
            to the upstream schema.

    Returns:
        List of :class:`CaseCollision`, one per casefold key with two or
        more variants. Empty when every column is uniquely cased.
    """
    by_key: dict[str, List[str]] = {}
    order: List[str] = []
    for col in columns:
        if col is None:
            continue
        key = col.casefold()
        bucket = by_key.get(key)
        if bucket is None:
            by_key[key] = [col]
            order.append(key)
        else:
            bucket.append(col)
    out: List[CaseCollision] = []
    for key in order:
        bucket = by_key[key]
        if len(bucket) < 2:
            continue
        out.append(CaseCollision(kept=bucket[0], dropped=list(bucket[1:])))
    return out


def register_case_collision_drops(
    registry,
    *,
    dataset: str,
    columns: Iterable[str],
    source_notebook: str = "02_source_integrity",
) -> List[CaseCollision]:
    """Detect collisions in ``columns`` and register drop_columns recs.

    Convenience wrapper: detects collisions via
    :func:`detect_case_collisions`, then for each collision calls
    ``registry.add_landing_drop_columns(dataset, dropped, rationale,
    source_notebook)`` so the redundant variants are dropped at landing
    time. Returns the detected collisions so the caller can render an
    audit-trail (NB02's recommendations table).

    Idempotent: ``add_landing_drop_columns`` itself dedupes per
    ``(dataset, column)``, so re-running this helper on an already-
    populated registry is a no-op.
    """
    collisions = detect_case_collisions(columns)
    if not collisions:
        return collisions
    for c in collisions:
        rationale = (
            f"Case-fold collision on {c.kept!r}: upstream carries "
            f"{[c.kept, *c.dropped]!r}. Spark folds bare identifiers "
            f"case-insensitively, so saveAsTable fails with "
            f"[COLUMN_ALREADY_EXISTS]. Dropping {c.dropped!r} keeps "
            f"the first-seen variant {c.kept!r}."
        )
        registry.add_landing_drop_columns(
            dataset=dataset,
            columns=c.dropped,
            rationale=rationale,
            source_notebook=source_notebook,
        )
    return collisions
