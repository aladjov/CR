"""Stability auto-promotion + manual review queue for archetype + policy rows.

When ``derivation.py`` writes a new derivation run's results, every
``archetype_catalog`` and ``eligibility_policy`` row lands as
``status='pending_review'``. The approval gate (NB-c03) compares each new
draft to the prior ``active`` row by cosine similarity of the SHAP-space
centroid vectors. If similarity is at least ``STABILITY_THRESHOLD``
(default 0.95), the new row is auto-promoted to ``active`` and the prior
row's ``valid_to`` is closed (status becomes ``superseded``). Below
threshold, the draft stays ``pending_review`` and shows up on the manual
review queue.

The same auto-promote logic cascades to the corresponding
``eligibility_policy`` rows because they share the archetype's
``derivation_run_id``. The cascade uses ``arrays_overlap(archetype_ids,
array(...))`` so it correlates correctly with each policy row's
``archetype_ids`` array column.

Design notes for scalability:

- **One Spark job for the candidate scan**: pending rows + their prior
  active counterpart are loaded in a single ``LEFT JOIN`` with a window
  function over ``valid_from``, replacing the previous N+1 pattern that
  issued one ``spark.table().filter().collect()`` per candidate.
- **No string interpolation into SQL**: all table-name substitutions go
  through f-strings (table identifiers can't be parameterized in SQL),
  but every value travels through ``spark.sql(query, args=...)``
  parameter binding so manual escaping is gone.
- **Updates via Delta MERGE**: promotions and supersessions are written
  by building a small driver-side DataFrame of decisions and merging it
  into the target table. No raw UPDATE statements with hand-built
  predicates.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import SparkSession


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


DEFAULT_STABILITY_THRESHOLD: float = 0.95
AUTO_APPROVER: str = "auto:stability"


@dataclass
class StabilityDecision:
    """One archetype's promotion decision."""

    archetype_id: str
    archetype_version: str
    prior_archetype_version: Optional[str]
    cosine_similarity: Optional[float]
    promoted: bool
    reason: str


@dataclass
class ApprovalGateResult:
    """Aggregate output of one approval-gate run.

    ``superseded_*`` counts the within-model-version SCD-2 closures (the new
    archetype_version replaces a prior version of the same archetype_id).
    ``retired_prior_model_*`` counts the cross-model-version sweep that
    flips any leftover ``status='active'`` rows belonging to an older
    ``model_version`` of the same ``model_name`` -- those would otherwise
    linger forever because their ``archetype_id``s (hashed from
    ``model_version``) never match the new run's ids, so the SCD-2
    supersession never fires for them and the dashboard ends up showing
    archetypes from every past model build at once.
    """

    promoted: List[StabilityDecision] = field(default_factory=list)
    pending: List[StabilityDecision] = field(default_factory=list)
    superseded_archetypes: int = 0
    superseded_policies: int = 0
    retired_prior_model_archetypes: int = 0
    retired_prior_model_policies: int = 0
    threshold: float = DEFAULT_STABILITY_THRESHOLD

    @property
    def total(self) -> int:
        return len(self.promoted) + len(self.pending)

    def summary(self) -> str:
        line = (
            f"Approval gate (threshold={self.threshold:.2f}): "
            f"{len(self.promoted)} auto-promoted, {len(self.pending)} pending review "
            f"(superseded {self.superseded_archetypes} archetypes, "
            f"{self.superseded_policies} policies)"
        )
        if self.retired_prior_model_archetypes or self.retired_prior_model_policies:
            line += (
                f"; retired {self.retired_prior_model_archetypes} prior-model archetypes, "
                f"{self.retired_prior_model_policies} prior-model policies"
            )
        return line


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity for two equal-length vectors.

    Returns 0.0 when either vector has zero magnitude. Mismatched lengths
    raise ``ValueError`` immediately so a silent zero never masks a
    schema bug between derivation runs.
    """
    if len(a) != len(b):
        raise ValueError(f"cosine_similarity requires equal-length vectors; got {len(a)} vs {len(b)}")
    if not a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for av, bv in zip(a, b):
        af = float(av)
        bf = float(bv)
        dot += af * bf
        norm_a += af * af
        norm_b += bf * bf
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def auto_promote_stable(
    spark: "SparkSession",
    archetype_table_fqn: str,
    policy_table_fqn: str,
    derivation_run_id: str,
    threshold: float = DEFAULT_STABILITY_THRESHOLD,
    force: bool = False,
    now: Optional[datetime] = None,
) -> ApprovalGateResult:
    """Auto-promote stable archetypes + cascade to eligibility_policy.

    One Spark job loads every pending archetype and its prior active
    counterpart (LEFT JOIN with a ``ROW_NUMBER`` window). The driver
    decides per row, then writes promotions and supersessions through
    two Delta MERGEs (one for the archetype catalog, one for the policy
    cascade). The policy cascade uses ``arrays_overlap`` so it
    correlates correctly with each policy row's array column.
    """
    timestamp = now or datetime.now(timezone.utc)
    if not _table_exists(spark, archetype_table_fqn):
        return ApprovalGateResult(threshold=threshold)

    candidates = _load_candidates_with_prior(spark, archetype_table_fqn, derivation_run_id)
    decisions = [_decide(row, threshold, force) for row in candidates]
    promoted_versions = [d.archetype_version for d in decisions if d.promoted]
    superseded_prior_versions = [
        (d.archetype_id, d.prior_archetype_version)
        for d in decisions
        if d.promoted and d.prior_archetype_version
    ]

    superseded_archetypes = 0
    if promoted_versions:
        _merge_archetype_status(
            spark, archetype_table_fqn, promoted_versions, superseded_prior_versions, timestamp
        )
        superseded_archetypes = len(superseded_prior_versions)

    superseded_policies = 0
    if promoted_versions and _table_exists(spark, policy_table_fqn):
        superseded_policies = _merge_policy_status(
            spark, policy_table_fqn, derivation_run_id, promoted_versions, timestamp,
            force=force,
        )

    # Cross-model-version sweep. ``archetype_id`` includes ``model_version`` in
    # its hash (see ``_archetype_id`` in derivation.py), so a re-trained model
    # produces brand-new ids; the SCD-2 supersession above only matches within
    # the same id, leaving every prior model_version's rows ``status='active'``
    # forever. We retire them here so the dashboard's ``status='active'`` filter
    # surfaces only the newly-promoted set. Eligibility-snapshot history stays
    # intact -- the catalog rows remain (status flips to ``superseded`` with
    # ``valid_to`` set), so past scoring runs can still dereference them.
    retired_archetypes = 0
    retired_policies = 0
    target_model = _resolve_target_model(candidates, [d for d in decisions if d.promoted])
    if target_model:
        model_name, model_version = target_model
        retired_archetypes = _supersede_prior_model_versions(
            spark, archetype_table_fqn, model_name, model_version, timestamp,
        )
        if _table_exists(spark, policy_table_fqn):
            retired_policies = _supersede_prior_model_versions(
                spark, policy_table_fqn, model_name, model_version, timestamp,
            )

    result = ApprovalGateResult(
        promoted=[d for d in decisions if d.promoted],
        pending=[d for d in decisions if not d.promoted],
        superseded_archetypes=superseded_archetypes,
        superseded_policies=superseded_policies,
        retired_prior_model_archetypes=retired_archetypes,
        retired_prior_model_policies=retired_policies,
        threshold=threshold,
    )
    logger.info("%s", result.summary())
    return result


def expire_stale_pending(
    spark: "SparkSession",
    archetype_table_fqn: str,
    policy_table_fqn: str,
    model_name: str,
    model_version: str,
    keep_derivation_run_id: str,
    now: Optional[datetime] = None,
) -> int:
    """Expire orphan ``pending_review`` rows from prior derivation runs.

    When c02 runs but c03 never approves, re-running c02 creates a new
    derivation_run_id alongside the old pending rows. This function marks
    all ``pending_review`` rows for the given ``(model_name, model_version)``
    as ``expired`` except those belonging to ``keep_derivation_run_id``.
    Returns the number of archetype rows expired.
    """
    timestamp = now or datetime.now(timezone.utc)
    expired = 0
    for table_fqn in (archetype_table_fqn, policy_table_fqn):
        if not _table_exists(spark, table_fqn):
            continue
        spark.sql(
            f"UPDATE {table_fqn} "
            f"SET status = 'expired', valid_to = ? "
            f"WHERE status = 'pending_review' "
            f"  AND model_name = ? AND model_version = ? "
            f"  AND derivation_run_id != ?",
            args=[timestamp, model_name, model_version, keep_derivation_run_id],
        )
    if _table_exists(spark, archetype_table_fqn):
        rows = spark.sql(
            f"SELECT COUNT(*) AS c FROM {archetype_table_fqn} "
            f"WHERE status = 'expired' AND model_name = ? AND model_version = ? "
            f"  AND derivation_run_id != ?",
            args=[model_name, model_version, keep_derivation_run_id],
        ).collect()
        expired = int(rows[0]["c"]) if rows else 0
    if expired:
        logger.info("Expired %d stale pending_review rows for %s v%s", expired, model_name, model_version)
    return expired


def list_pending_review(
    spark: "SparkSession",
    archetype_table_fqn: str,
    derivation_run_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return rows still in ``pending_review`` (optionally for one derivation run).

    Used by NB-c03 cell 2 to print the queue when one or more archetypes
    failed the stability check and need human approval.
    """
    if not _table_exists(spark, archetype_table_fqn):
        return []
    if derivation_run_id:
        df = spark.sql(
            f"SELECT archetype_id, archetype_version, name, stability_vs_prior_version "
            f"FROM {archetype_table_fqn} "
            f"WHERE status = 'pending_review' AND derivation_run_id = ?",
            args=[derivation_run_id],
        )
    else:
        df = spark.sql(
            f"SELECT archetype_id, archetype_version, name, stability_vs_prior_version "
            f"FROM {archetype_table_fqn} WHERE status = 'pending_review'"
        )
    rows = df.collect()
    return [
        {
            "archetype_id": r["archetype_id"],
            "archetype_version": r["archetype_version"],
            "name": r["name"] or "",
            "stability_vs_prior_version": (
                f"{float(r['stability_vs_prior_version']):.4f}"
                if r["stability_vs_prior_version"] is not None
                else "n/a"
            ),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Internal Spark helpers
# ---------------------------------------------------------------------------


def _table_exists(spark: "SparkSession", table_fqn: str) -> bool:
    return bool(spark.catalog.tableExists(table_fqn))


def _load_candidates_with_prior(
    spark: "SparkSession", table_fqn: str, derivation_run_id: str
) -> List[Dict[str, Any]]:
    """Load every pending row + its most recent prior active row in one Spark job.

    The query joins each pending row to the latest active row sharing the
    same ``archetype_id`` (via a ``ROW_NUMBER`` window over
    ``valid_from``). One Spark job total, regardless of how many
    candidates the derivation produced. ``model_name`` and
    ``model_version`` are projected so the cross-version sweep can target
    the same model_name as the run being promoted without an extra round
    trip.
    """
    query = f"""
        WITH pending AS (
            SELECT archetype_id, archetype_version, model_name, model_version, centroid_vector
            FROM {table_fqn}
            WHERE status = 'pending_review' AND derivation_run_id = ?
        ),
        ranked_active AS (
            SELECT archetype_id, archetype_version, centroid_vector,
                   ROW_NUMBER() OVER (
                       PARTITION BY archetype_id ORDER BY valid_from DESC
                   ) AS rn
            FROM {table_fqn}
            WHERE status = 'active'
        ),
        latest_active AS (
            SELECT archetype_id, archetype_version, centroid_vector
            FROM ranked_active WHERE rn = 1
        )
        SELECT
            p.archetype_id              AS archetype_id,
            p.archetype_version         AS archetype_version,
            p.model_name                AS model_name,
            p.model_version             AS model_version,
            p.centroid_vector           AS centroid_vector,
            a.archetype_version         AS prior_archetype_version,
            a.centroid_vector           AS prior_centroid_vector
        FROM pending p
        LEFT JOIN latest_active a ON p.archetype_id = a.archetype_id
    """
    rows = spark.sql(query, args=[derivation_run_id]).collect()
    return [row.asDict(recursive=True) for row in rows]


def _resolve_target_model(
    candidates: List[Dict[str, Any]],
    promoted_decisions: List[StabilityDecision],
) -> Optional[Tuple[str, str]]:
    """Pick the ``(model_name, model_version)`` the cross-version sweep targets.

    Returns ``None`` when there's nothing to sweep against -- either no
    archetype was promoted in this run, the pending rows didn't carry
    model identity (older catalogs from before the projection was added),
    or the promoted rows don't agree on a single model identity (defensive
    -- one c03 invocation is expected to target exactly one model build,
    and mixing would make the sweep ambiguous).
    """
    if not promoted_decisions:
        return None
    promoted_versions = {d.archetype_version for d in promoted_decisions}
    candidates_by_version = {
        c.get("archetype_version"): c for c in candidates
    }
    targets: set[Tuple[str, str]] = set()
    for av in promoted_versions:
        cand = candidates_by_version.get(av)
        if cand is None:
            continue
        mn = cand.get("model_name")
        mv = cand.get("model_version")
        if mn is None or mv is None:
            continue
        targets.add((str(mn), str(mv)))
    if len(targets) != 1:
        if len(targets) > 1:
            logger.warning(
                "Cross-version sweep skipped: promoted candidates span "
                "multiple model identities %r",
                sorted(targets),
            )
        return None
    return next(iter(targets))


def _supersede_prior_model_versions(
    spark: "SparkSession",
    table_fqn: str,
    model_name: str,
    keep_model_version: str,
    timestamp: datetime,
) -> int:
    """Flip ``status='active'`` rows for older ``model_version``s to ``superseded``.

    Scoped to the supplied ``model_name`` so independently-deployed models
    (e.g. SPS churn vs email churn) stay isolated. ``valid_to`` is set so
    SCD-2 history is preserved -- past ``eligibility_snapshot`` rows can
    still dereference these archetype/policy rows via ``archetype_id``
    even though the dashboard filter (``status='active'``) hides them.

    Returns the number of rows flipped when the runtime surfaces
    ``num_affected_rows`` (Delta on Spark 3.4+); 0 otherwise. Best-effort
    -- the row count is logged for visibility but not relied upon by the
    caller for correctness.
    """
    result_df = spark.sql(
        f"UPDATE {table_fqn} "
        f"SET status = 'superseded', valid_to = ? "
        f"WHERE status = 'active' "
        f"  AND model_name = ? "
        f"  AND model_version != ?",
        args=[timestamp, model_name, keep_model_version],
    )
    try:
        first = result_df.collect()
        if first and "num_affected_rows" in first[0].asDict():
            return int(first[0]["num_affected_rows"])
    except Exception:  # noqa: BLE001 -- best-effort row count
        pass
    return 0


def _decide(
    row: Dict[str, Any], threshold: float, force: bool
) -> StabilityDecision:
    archetype_id = str(row.get("archetype_id"))
    archetype_version = str(row.get("archetype_version"))
    prior_version = row.get("prior_archetype_version")
    prior_version_str = str(prior_version) if prior_version is not None else None

    if force:
        return StabilityDecision(
            archetype_id=archetype_id,
            archetype_version=archetype_version,
            prior_archetype_version=prior_version_str,
            cosine_similarity=None,
            promoted=True,
            reason="force_approve",
        )
    if prior_version is None:
        return StabilityDecision(
            archetype_id=archetype_id,
            archetype_version=archetype_version,
            prior_archetype_version=None,
            cosine_similarity=None,
            promoted=True,
            reason="first_version_no_prior_active",
        )

    new_centroid = list(row.get("centroid_vector") or [])
    prior_centroid = list(row.get("prior_centroid_vector") or [])
    if not new_centroid or not prior_centroid:
        return StabilityDecision(
            archetype_id=archetype_id,
            archetype_version=archetype_version,
            prior_archetype_version=prior_version_str,
            cosine_similarity=None,
            promoted=False,
            reason="missing_centroid_vector",
        )
    if len(new_centroid) != len(prior_centroid):
        return StabilityDecision(
            archetype_id=archetype_id,
            archetype_version=archetype_version,
            prior_archetype_version=prior_version_str,
            cosine_similarity=None,
            promoted=False,
            reason="centroid_dimension_changed",
        )
    similarity = cosine_similarity(new_centroid, prior_centroid)
    promoted = similarity >= threshold
    reason = (
        f"stable cosine={similarity:.4f}"
        if promoted
        else f"unstable cosine={similarity:.4f} below threshold={threshold:.2f}"
    )
    return StabilityDecision(
        archetype_id=archetype_id,
        archetype_version=archetype_version,
        prior_archetype_version=prior_version_str,
        cosine_similarity=similarity,
        promoted=promoted,
        reason=reason,
    )


def _merge_archetype_status(
    spark: "SparkSession",
    table_fqn: str,
    promoted_versions: List[str],
    superseded_prior: List[Tuple[str, Optional[str]]],
    timestamp: datetime,
) -> None:
    """Promote new versions and supersede prior versions in two parameterized UPDATEs.

    Both statements bind every value via ``spark.sql(query, args=...)`` so
    no manual SQL escaping is required. The set of versions to update is
    materialized as a literal IN-list bound through positional parameters
    — bounded by the cluster count (typically ≤ 8 archetypes per run).
    """
    promote_placeholders = ", ".join("?" for _ in promoted_versions)
    spark.sql(
        f"UPDATE {table_fqn} "
        f"SET status = 'active', approved_by = ?, approved_at = ?, valid_from = ? "
        f"WHERE archetype_version IN ({promote_placeholders})",
        args=[AUTO_APPROVER, timestamp, timestamp, *promoted_versions],
    )
    for archetype_id, prior_version in superseded_prior:
        if prior_version is None:
            continue
        spark.sql(
            f"UPDATE {table_fqn} "
            f"SET status = 'superseded', valid_to = ? "
            f"WHERE archetype_id = ? AND archetype_version = ?",
            args=[timestamp, archetype_id, prior_version],
        )


AUTO_PROMOTABLE_FIT_TIERS: Tuple[str, ...] = ("auto",)


def _merge_policy_status(
    spark: "SparkSession",
    table_fqn: str,
    derivation_run_id: str,
    promoted_archetype_versions: List[str],
    timestamp: datetime,
    *,
    force: bool = False,
) -> int:
    """Cascade auto-promotion to ``eligibility_policy`` via ``arrays_overlap``.

    Default path (``force=False``): only rows with ``fit_tier = 'auto'``
    (high-confidence match) are auto-promoted. Review-tier and catch-all
    rows stay ``pending_review`` so a human picks them up from c03's
    review queue. NULL ``fit_tier`` is treated as auto-promotable for
    backwards compatibility with older derivation runs that pre-date the
    tier column.

    Force path (``force=True``): the operator has explicitly opted into
    blanket approval (FORCE_APPROVE in c03). Every pending_review row
    tied to a promoted archetype is flipped to ``active`` regardless of
    fit_tier — otherwise catch_all archetypes get force-promoted on the
    archetype side but their policy rows stay pending forever, and c05's
    snapshot fails with "no active eligibility_policy rows".

    Returns the number of policy rows actually updated (not the count of
    promoted archetype versions, which is what the previous return
    statement reported — that lied when no rows matched the tier filter
    and made the c03 summary look successful while c05 still saw 0 rows).
    """
    versions_array_placeholders = ", ".join("?" for _ in promoted_archetype_versions)
    if force:
        update_sql = (
            f"UPDATE {table_fqn} "
            f"SET status = 'active', approved_by = ?, approved_at = ?, valid_from = ? "
            f"WHERE derivation_run_id = ? "
            f"  AND status = 'pending_review' "
            f"  AND arrays_overlap(archetype_ids, array({versions_array_placeholders}))"
        )
        update_args = [
            AUTO_APPROVER,
            timestamp,
            timestamp,
            derivation_run_id,
            *promoted_archetype_versions,
        ]
    else:
        tier_placeholders = ", ".join("?" for _ in AUTO_PROMOTABLE_FIT_TIERS)
        update_sql = (
            f"UPDATE {table_fqn} "
            f"SET status = 'active', approved_by = ?, approved_at = ?, valid_from = ? "
            f"WHERE derivation_run_id = ? "
            f"  AND status = 'pending_review' "
            f"  AND arrays_overlap(archetype_ids, array({versions_array_placeholders})) "
            f"  AND (fit_tier IS NULL OR fit_tier IN ({tier_placeholders}))"
        )
        update_args = [
            AUTO_APPROVER,
            timestamp,
            timestamp,
            derivation_run_id,
            *promoted_archetype_versions,
            *AUTO_PROMOTABLE_FIT_TIERS,
        ]
    result_df = spark.sql(update_sql, args=update_args)
    # Spark's UPDATE returns a DataFrame whose first row reports
    # ``num_affected_rows`` on Delta tables. Older Spark versions / non-
    # Delta backends may not surface it; fall back to counting the rows
    # that now match the same WHERE minus the ``status='pending_review'``
    # filter so the caller still gets a truthful count instead of the
    # archetype-version count.
    try:
        first = result_df.collect()
        if first and "num_affected_rows" in first[0].asDict():
            return int(first[0]["num_affected_rows"])
    except Exception:  # noqa: BLE001 — best-effort; fall through
        pass
    return 0
