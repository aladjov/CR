"""Backfill ``eligibility_policy.eligibility_rules_prose`` for existing rows.

c02's policy-row builder calls ``_render_predicate_prose_safely`` once at
build time, so prose populates only for rows written in that derivation
run. When the prerequisite sidecars (column_descriptions, feature_meta,
feature_population_stats) became available *after* derivation already
wrote the policy table, every existing row stays ``NULL`` — Cycle 013
P4 surfaced exactly that. Re-running derivation fixes it for fresh rows
but doesn't touch the existing ones.

This module provides a one-shot helper that:
  1. Discovers the interpretation sidecars via the same discovery bundle
     derivation uses (file-tracked — no env vars).
  2. Reads every active row whose prose column is ``NULL`` / empty.
  3. Renders prose for each via ``compile_predicate_prose`` and merges
     the result back into the Delta table keyed on the natural primary
     key (``eligibility_policy_id`` × ``version``).

Callers:
  - ``causal_notebooks/c02_archetype_derivation.ipynb`` — invoke after
    the derivation cell on every run so existing rows pick up prose
    when sidecars finally materialize.
  - Any one-off cleanup notebook — pass an explicit ``namespace`` to
    target a specific run.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from customer_retention.stages.causal.interpretation.predicate_prose import (
    compile_predicate_prose,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import SparkSession

    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    from customer_retention.stages.causal.interpretation.discovery import (
        InterpretationSidecars,
    )


@dataclass
class ProseBackfillResult:
    """Summary returned by :func:`backfill_eligibility_prose`."""

    candidates: int = 0
    rendered: int = 0
    updated: int = 0
    warnings: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []

    def summary(self) -> str:
        return (
            f"eligibility_rules_prose backfill: candidates={self.candidates} "
            f"rendered={self.rendered} updated={self.updated}"
        )


def backfill_eligibility_prose(
    spark: "SparkSession",
    table_fqn: str,
    *,
    namespace: Optional["RunNamespace"] = None,
    experiments_root: Optional[Path] = None,
) -> ProseBackfillResult:
    """Render prose for every active row missing ``eligibility_rules_prose``.

    Returns a :class:`ProseBackfillResult` summarising work done. A row
    counts as updated when its rendered prose was non-empty AND the
    Delta MERGE target it. Rows with malformed JSON predicates are
    skipped silently (counted as candidates but not rendered).
    """
    from customer_retention.stages.causal.interpretation.discovery import (
        discover_interpretation_sidecars,
    )

    bundle = discover_interpretation_sidecars(
        namespace=namespace, experiments_root=experiments_root,
    )
    bundle.emit_warnings(logger_=logger)

    if bundle.namespace is None:
        return ProseBackfillResult(warnings=list(bundle.warnings))

    candidates = _read_null_prose_rows(spark, table_fqn)
    if not candidates:
        logger.info("backfill_eligibility_prose: no NULL prose rows found in %s", table_fqn)
        return ProseBackfillResult(warnings=list(bundle.warnings))

    rendered: List[Tuple[str, str, str]] = []
    for policy_id, version, predicate_json in candidates:
        prose = _safe_render(predicate_json, bundle)
        if prose:
            rendered.append((policy_id, version, prose))

    if not rendered:
        logger.warning(
            "backfill_eligibility_prose: %d candidates but compile_predicate_prose "
            "returned empty for every row — sidecar inputs may be incomplete.",
            len(candidates),
        )
        return ProseBackfillResult(
            candidates=len(candidates),
            warnings=list(bundle.warnings),
        )

    updated = _merge_prose_back(spark, table_fqn, rendered)
    logger.info(
        "backfill_eligibility_prose: rendered=%d updated=%d candidates=%d table=%s",
        len(rendered), updated, len(candidates), table_fqn,
    )
    return ProseBackfillResult(
        candidates=len(candidates),
        rendered=len(rendered),
        updated=updated,
        warnings=list(bundle.warnings),
    )


def _read_null_prose_rows(
    spark: "SparkSession", table_fqn: str,
) -> List[Tuple[str, str, str]]:
    """Return ``(policy_id, version, predicate_json_str)`` for rows missing prose.

    Filters to ``status = 'active'`` so we don't backfill superseded /
    archived rows. The ``collect()`` is intentional: candidates are
    bounded by ``|active eligibility_policy rows|`` which is small
    (≤ |archetypes| × |playbooks|).
    """
    df = spark.sql(
        f"""
        SELECT eligibility_policy_id, version, eligibility_rules
        FROM {table_fqn}
        WHERE status = 'active'
          AND (eligibility_rules_prose IS NULL OR eligibility_rules_prose = '')
          AND eligibility_rules IS NOT NULL
        """
    )
    out: List[Tuple[str, str, str]] = []
    for row in df.collect():
        out.append((row["eligibility_policy_id"], row["version"], row["eligibility_rules"]))
    return out


def _safe_render(
    predicate_json_str: str, bundle: "InterpretationSidecars",
) -> Optional[str]:
    """Render one row, returning None on parse / render failure.

    Catches only the **expected** failure modes per the no-defensive-code rule
    in ``docs/Coding_Practices.md``: malformed JSON (``TypeError`` /
    ``ValueError`` from ``json.loads``) and structural mismatches between the
    persisted predicate and the current ``compile_predicate_prose`` schema
    (``KeyError`` / ``AttributeError``). Any other exception type propagates
    so the operator sees the real root cause instead of silent NULL fills.
    """
    try:
        predicate = json.loads(predicate_json_str)
    except (TypeError, ValueError):
        return None
    try:
        prose = compile_predicate_prose(
            predicate,
            feature_meta=bundle.feature_meta,
            population_stats=bundle.population_stats,
            column_descriptions=bundle.column_descriptions,
        )
    except (KeyError, AttributeError, TypeError):
        return None
    return prose or None


def _merge_prose_back(
    spark: "SparkSession",
    table_fqn: str,
    rows: List[Tuple[str, str, str]],
) -> int:
    """Delta MERGE the rendered rows on ``(eligibility_policy_id, version)``."""
    from delta.tables import DeltaTable

    source = spark.createDataFrame(
        rows, schema="eligibility_policy_id STRING, version STRING, prose STRING",
    )
    (
        DeltaTable.forName(spark, table_fqn).alias("t")
        .merge(
            source.alias("s"),
            "t.eligibility_policy_id = s.eligibility_policy_id "
            "AND t.version = s.version",
        )
        .whenMatchedUpdate(set={"eligibility_rules_prose": "s.prose"})
        .execute()
    )
    return len(rows)


__all__ = [
    "ProseBackfillResult",
    "backfill_eligibility_prose",
]
