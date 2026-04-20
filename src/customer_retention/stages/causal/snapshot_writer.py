"""Per-scoring-run ``eligibility_snapshot`` writer.

Phase 3 of the causal-track pipeline. After ``s10_batch_inference`` populates
``predictions`` for the current scoring run, this module:

1. **Resolves the four-way anchor** — picks the active rows of
   ``archetype_catalog``, ``eligibility_policy``, and ``decision_policy``
   that govern the current ``(model_name, model_version)``.
2. **Assigns each customer to its nearest archetype** by scaled-Euclidean
   distance over the **raw-feature centroid** stored on the active
   archetype rows. SHAP is intentionally not recomputed at scoring time
   (see ``docs/causal_track_implementation_plan.md`` §"SHAP at scoring
   time"). All distance math runs as a single Spark cross join + window
   so the cost is bounded by ``rows × |archetypes|``, not ``rows × |features|``.
3. **Evaluates each active eligibility policy as a predicate fan-out** by
   compiling the JSON predicate stored on the policy row to a
   ``pyspark.sql.Column`` (see ``predicate_compiler.compile_predicate``)
   and ``filter(...)``-ing the joined predictions DataFrame. Customers
   that match a policy contribute one snapshot row per matching playbook.
4. **Applies the operational decision policy** — suppression rules,
   capacity caps, cooldown windows, prioritization within a cohort, and
   stable holdout assignment via the ``holdout_seed_recipe`` hash. The
   four-way anchor (``model_version``, ``archetype_version``,
   ``eligibility_policy_version``, ``decision_policy_version``) plus
   ``as_of_date`` is folded into the deterministic ``scoring_run_id``.
5. **Writes the snapshot** via ``DeltaTable.merge(snapshot_df, ...)`` keyed
   on the natural ``(scoring_run_id, entity_id, playbook_id)`` so a
   re-run with identical inputs is a no-op. The snapshot DataFrame is
   never collected to the driver — the merge runs entirely in Spark.

Distributed-by-construction guidelines this module honors:

- All per-row work runs as Spark SQL operations: cross-joins, window
  functions, and ``filter(...)``. No ``.toPandas()`` on the predictions
  body, no per-row Python UDFs over the cohort.
- Catalog joins (active archetype + policy + decision policy) collect
  small driver-side dictionaries (≤ 8 archetypes, ≤ 100 policies, 1
  decision policy) that ride along inside Spark expressions via
  ``F.create_map`` / ``F.lit`` literals — bounded so the Catalyst plans
  stay small.
- Predicate compilation runs once per policy, not once per row. The
  compiled ``Column`` is composed into a ``filter`` and Spark plans the
  whole evaluation as a single physical scan.
- The MERGE into ``eligibility_snapshot`` calls ``DeltaTable.merge``
  directly with the Spark snapshot DataFrame; the snapshot rows are
  never materialized on the driver, so the writer scales linearly with
  ``population × |eligibility_policies|`` without OOM risk.

Public API surface (consumed by the c04 notebook + script):

- :class:`SnapshotConfig`, :class:`SnapshotResult` — dataclasses
- :func:`build_eligibility_snapshot` — top-level orchestrator
- :func:`assign_archetype` — raw-feature nearest-centroid assignment
- :func:`evaluate_eligibility` — predicate fan-out
- :func:`apply_decision_policy` — suppression / capacity / holdout / rank
- :func:`write_snapshot` — Delta MERGE on natural key
- :func:`compute_scoring_run_id` — deterministic four-way anchor hash
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import Column, DataFrame, SparkSession


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


_DEFAULT_TARGET_TABLE = "eligibility_snapshot"
_RISK_TIER_HIGH = "High"
_RISK_TIER_MEDIUM = "Medium"
_RISK_TIER_LOW = "Low"
_DEFAULT_HOLDOUT_FRACTION = 0.1
_TOP_SHAP_DTYPE = (
    "array<struct<feature:string,value:double,shap_contribution:double,direction:string>>"
)


@dataclass
class SnapshotConfig:
    """All inputs needed to materialize one ``eligibility_snapshot`` run."""

    spark: "SparkSession"
    predictions_fqn: str
    archetype_catalog_fqn: str
    eligibility_policy_fqn: str
    decision_policy_fqn: str
    snapshot_table_fqn: str
    model_name: str
    model_version: str
    # The scoring-layer key: the column on predictions / gold / the snapshot
    # output that identifies the subject of scoring (a subscriber for email,
    # an account for SPS, a user for product-usage, etc). Generic by
    # construction; downstream writeback tables (assignments / actions /
    # outcomes) may introduce a CSM-specific account_id by join or rename if
    # a CSM tool is wired in.
    entity_id_column: str = "entity_id"
    probability_column: str = "churn_probability"
    value_at_risk_column: Optional[str] = None
    inference_timestamp_column: str = "inference_point_in_time"
    as_of_date: Optional[datetime] = None
    decision_policy_id: Optional[str] = None  # falls back to most recent active row
    risk_tier_high: Optional[float] = None
    risk_tier_medium: Optional[float] = None
    capacity_partition_column: Optional[str] = None  # e.g. "csm_owner_id"
    top_shap_drivers_fqn: Optional[str] = None  # optional Phase 2 cache
    # Gold features source for eligibility-predicate evaluation. The
    # predictions table carries only scoring outputs; eligibility rules
    # reference raw feature columns (e.g. active_span_days) and need the
    # gold table joined in. Required whenever any active policy has a
    # non-empty requires_features list.
    gold_features_fqn: Optional[str] = None


@dataclass
class SnapshotResult:
    """Aggregate counts produced by one snapshot build."""

    scoring_run_id: str
    as_of_date: datetime
    total_eligible_rows: int
    recommended_rows: int
    holdout_rows: int
    suppressed_rows: int
    archetypes_considered: int
    policies_considered: int
    risk_tier_counts: Dict[str, int] = field(default_factory=dict)
    decision_policy_id: Optional[str] = None

    def summary(self) -> str:
        return (
            f"snapshot {self.scoring_run_id} as_of={self.as_of_date.isoformat()}: "
            f"{self.total_eligible_rows} eligible "
            f"({self.recommended_rows} recommended, {self.holdout_rows} holdout, "
            f"{self.suppressed_rows} suppressed) "
            f"across {self.archetypes_considered} archetypes / "
            f"{self.policies_considered} policies"
        )


@dataclass
class _LoadedDefinitions:
    """Lightweight in-memory snapshot of the active definition rows."""

    archetypes: List[Dict[str, Any]]
    policies: List[Dict[str, Any]]
    decision_policy: Optional[Dict[str, Any]]
    archetype_version: str
    eligibility_policy_version: str
    decision_policy_version: str
    decision_policy_id: Optional[str]


# ---------------------------------------------------------------------------
# Public helpers (single-purpose, individually unit-testable)
# ---------------------------------------------------------------------------


def compute_scoring_run_id(
    model_name: str,
    model_version: str,
    archetype_version: str,
    eligibility_policy_version: str,
    decision_policy_version: str,
    as_of_date: datetime,
) -> str:
    """Deterministic ``scoring_run_id`` derived from the four-way anchor + date.

    Re-running the snapshot with the same anchor tuple and date produces
    the same id, so the downstream MERGE is a no-op. The hash is short
    enough to embed in dashboards but long enough to avoid collisions
    across realistic operator timelines.
    """
    payload = "|".join(
        [
            model_name,
            model_version,
            archetype_version,
            eligibility_policy_version,
            decision_policy_version,
            as_of_date.replace(microsecond=0).isoformat(),
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"score_{digest[:16]}"


def compute_eligibility_id(scoring_run_id: str, entity_id: str, playbook_id: str) -> str:
    """Deterministic per-row id used as the natural-key surrogate.

    Pure function so the snapshot writer can produce the same id for the
    same triple across re-runs without consulting the target table.
    """
    payload = f"{scoring_run_id}|{entity_id}|{playbook_id}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


def assign_archetype(
    predictions_df: "DataFrame",
    archetype_rows: Sequence[Mapping[str, Any]],
) -> "DataFrame":
    """Append ``archetype_id`` and ``archetype_version`` to ``predictions_df``.

    Each row is mapped to the archetype whose **raw-feature centroid** is
    nearest in **scaled Euclidean distance**:
    ``distance = sqrt(sum((x - centroid) / scale)²)``
    where ``scale`` is the per-feature population std-dev stored in
    ``centroid_feature_scales`` on the archetype row. This prevents
    high-magnitude features (e.g. ``tenure_days`` ∈ [0, 3650]) from
    dominating low-magnitude features (e.g. ``nps_score`` ∈ [0, 10]).
    When ``centroid_feature_scales`` is absent or a specific entry is
    zero/null, that feature's scale defaults to ``1.0`` (unscaled).

    Distance is computed entirely in Spark via column-arithmetic over the
    centroid vectors expanded as literals (≤ 8 archetypes per run, so the
    literal expansion stays small). No row-by-row Python; no cross join.

    Inputs:

    - ``predictions_df`` — predictions row set. Feature columns referenced
      by an archetype's ``centroid_feature_order`` that are missing from
      the predictions are silently skipped per archetype (cold-start
      friendly).
    - ``archetype_rows`` — driver-side dicts (typically the active rows
      from ``archetype_catalog``) carrying ``archetype_id``,
      ``archetype_version``, ``centroid_vector_raw`` (list[float]),
      ``centroid_feature_order`` (list[str]), and optionally
      ``centroid_feature_scales`` (list[float], aligned 1:1 with order).
      Rows missing either centroid field are skipped with a warning.

    Returns the input DataFrame with two new columns appended:
    ``archetype_id`` and ``archetype_version``. When ``archetype_rows`` is
    empty (cold-start) both columns are filled with NULL.
    """
    from pyspark.sql import functions as F  # noqa: N812

    if not archetype_rows:
        return predictions_df.withColumn("archetype_id", F.lit(None).cast("string")).withColumn(
            "archetype_version", F.lit(None).cast("string")
        )

    valid: List[Tuple[str, str, List[float], List[str], List[float]]] = []
    for row in archetype_rows:
        centroid = row.get("centroid_vector_raw")
        order = row.get("centroid_feature_order")
        if not centroid or not order:
            logger.warning(
                "skipping archetype %s/%s with missing centroid",
                row.get("archetype_id"),
                row.get("archetype_version"),
            )
            continue
        scales = list(row.get("centroid_feature_scales") or [])
        valid.append((str(row["archetype_id"]), str(row["archetype_version"]), list(centroid), list(order), scales))

    if not valid:
        return predictions_df.withColumn("archetype_id", F.lit(None).cast("string")).withColumn(
            "archetype_version", F.lit(None).cast("string")
        )

    available = set(predictions_df.columns)
    distance_columns: List["Column"] = []
    distance_id_pairs: List[Tuple[str, str, str]] = []  # (col_name, archetype_id, archetype_version)
    for idx, (archetype_id, archetype_version, centroid, order, scales) in enumerate(valid):
        per_arch_terms: List["Column"] = []
        for j, (feature_name, centroid_value) in enumerate(zip(order, centroid)):
            if feature_name not in available:
                continue
            scale_val = scales[j] if j < len(scales) else None
            scale = float(scale_val) if (scale_val is not None and float(scale_val) > 0.0) else 1.0
            diff = (F.col(feature_name).cast("double") - F.lit(float(centroid_value))) / F.lit(scale)
            per_arch_terms.append(diff * diff)
        distance_col_name = f"__cr_dist_{idx}"
        if not per_arch_terms:
            distance_columns.append(F.lit(float("inf")).alias(distance_col_name))
        else:
            squared_sum = per_arch_terms[0]
            for term in per_arch_terms[1:]:
                squared_sum = squared_sum + term
            distance_columns.append(F.sqrt(squared_sum).alias(distance_col_name))
        distance_id_pairs.append((distance_col_name, archetype_id, archetype_version))

    enriched = predictions_df.select("*", *distance_columns)

    distance_cols = [F.col(name) for name, _, _ in distance_id_pairs]
    min_distance = distance_cols[0] if len(distance_cols) == 1 else F.least(*distance_cols)
    last_id, last_version = distance_id_pairs[-1][1], distance_id_pairs[-1][2]
    winner_col = F.lit(last_id)
    winner_version_col = F.lit(last_version)
    for name, archetype_id, archetype_version in distance_id_pairs[:-1]:
        cond = F.col(name) <= min_distance
        winner_col = F.when(cond, F.lit(archetype_id)).otherwise(winner_col)
        winner_version_col = F.when(cond, F.lit(archetype_version)).otherwise(winner_version_col)

    drop_cols = [name for name, _, _ in distance_id_pairs]
    return (
        enriched.withColumn("archetype_id", winner_col)
        .withColumn("archetype_version", winner_version_col)
        .drop(*drop_cols)
    )


def evaluate_eligibility(
    enriched_df: "DataFrame",
    policy_rows: Sequence[Mapping[str, Any]],
) -> "DataFrame":
    """Fan ``enriched_df`` out to one row per ``(account, matching policy)``.

    For each policy in ``policy_rows`` the JSON predicate is compiled to a
    Spark ``Column`` and applied as a ``filter``. The matching subset is
    annotated with the policy's identifying columns and unioned with the
    other policies' matches. The fan-out is bounded by ``|policies|``
    per-policy filters — each one is a single Spark scan.

    Returns a Spark DataFrame with the original prediction columns plus
    ``playbook_id``, ``playbook_version``, ``eligibility_policy_id``,
    ``eligibility_policy_version``, and ``eligibility_evidence`` (JSON
    string of the predicate). When no policy matches anything, the
    returned DataFrame has the same schema with zero rows.
    """
    from .predicate_compiler import compile_predicate, predicate_to_sql

    parts: List["DataFrame"] = []
    for row in policy_rows:
        predicate = _parse_predicate(row.get("eligibility_rules"))
        if predicate is None:
            logger.warning(
                "skipping policy %s/%s with unparseable eligibility_rules",
                row.get("eligibility_policy_id"),
                row.get("version"),
            )
            continue
        column = compile_predicate(predicate)
        evidence = row.get("eligibility_rules_sql") or predicate_to_sql(predicate)
        parts.append(_annotate_policy_match(enriched_df.filter(column), row, evidence))

    if not parts:
        return _empty_eligibility_df(enriched_df)
    union = parts[0]
    for part in parts[1:]:
        union = union.unionByName(part)
    return union


def _annotate_policy_match(
    matches_df: "DataFrame", policy_row: Mapping[str, Any], evidence: str
) -> "DataFrame":
    from pyspark.sql import functions as F  # noqa: N812

    return (
        matches_df.withColumn("playbook_id", F.lit(str(policy_row["playbook_id"])))
        .withColumn("playbook_version", F.lit(str(policy_row.get("playbook_version") or "")))
        .withColumn("eligibility_policy_id", F.lit(str(policy_row["eligibility_policy_id"])))
        .withColumn("eligibility_policy_version", F.lit(str(policy_row.get("version") or "")))
        .withColumn("eligibility_evidence", F.lit(str(evidence)))
    )


def _empty_eligibility_df(enriched_df: "DataFrame") -> "DataFrame":
    from pyspark.sql import functions as F  # noqa: N812

    return (
        enriched_df.limit(0)
        .withColumn("playbook_id", F.lit(None).cast("string"))
        .withColumn("playbook_version", F.lit(None).cast("string"))
        .withColumn("eligibility_policy_id", F.lit(None).cast("string"))
        .withColumn("eligibility_policy_version", F.lit(None).cast("string"))
        .withColumn("eligibility_evidence", F.lit(None).cast("string"))
    )


def apply_decision_policy(
    eligible_df: "DataFrame",
    decision_policy: Optional[Mapping[str, Any]],
    *,
    entity_id_column: str = "entity_id",
    probability_column: str = "churn_probability",
    risk_tier_high: float = 0.6,
    risk_tier_medium: float = 0.3,
    capacity_partition_column: Optional[str] = None,
) -> "DataFrame":
    """Apply suppression, capacity, holdout, and rank in one Spark plan.

    The decision-policy row supplies a small driver-side bag of dicts:

    - ``suppression_rules`` — JSON predicate compiled to a single Spark
      ``Column`` and used to mark matching rows ``playbook_suppressed_reason``.
    - ``holdout_fractions_by_playbook`` — per-playbook holdout fractions.
      The implementation hashes a deterministic seed via ``F.xxhash64`` and
      buckets each ``(account, playbook)`` pair.
    - ``capacity_by_playbook`` — per-playbook capacity caps. Rows whose
      ``priority_rank_within_cohort`` exceeds the cap are marked
      ``playbook_suppressed_reason = 'capacity_exceeded'``.
    - ``holdout_seed_recipe`` — comma-separated list of column names
      concatenated into the per-row hash input.

    Cooldown is **not yet enforced** — ``cooldown_days_by_playbook``
    requires the writeback ledger of prior assignments per
    ``(account, playbook)``, which arrives once the CSM-tool transport is
    wired (Phase 4). The field is read so a warning surfaces when the
    operator has set it but Phase 4 has not landed yet.

    Returns the input DataFrame with the operational columns appended:
    ``risk_tier``, ``is_holdout``, ``holdout_stratum``, ``holdout_seed``,
    ``playbook_suppressed_reason``, ``recommended``,
    ``priority_rank_within_cohort``, ``policy_rank_among_eligible``,
    ``eligible_playbook_count``, ``eligible_playbooks_set``.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F  # noqa: N812

    policy = dict(decision_policy or {})
    if policy.get("cooldown_days_by_playbook"):
        logger.warning(
            "cooldown_days_by_playbook is set on decision_policy but cooldown "
            "enforcement is not yet implemented (requires the writeback ledger). "
            "Rows are surfaced regardless of prior-assignment recency."
        )

    risk_tier_col = F.when(
        F.col(probability_column) >= F.lit(float(risk_tier_high)), F.lit(_RISK_TIER_HIGH)
    ).when(
        F.col(probability_column) >= F.lit(float(risk_tier_medium)), F.lit(_RISK_TIER_MEDIUM)
    ).otherwise(F.lit(_RISK_TIER_LOW))

    suppression_predicate, suppression_reason = _compile_suppression(policy.get("suppression_rules"))

    suppressed_col = F.when(suppression_predicate, F.lit(suppression_reason)).otherwise(
        F.lit(None).cast("string")
    )

    eligible_df = eligible_df.withColumn("risk_tier", risk_tier_col).withColumn(
        "playbook_suppressed_reason", suppressed_col
    )

    holdout_fractions = policy.get("holdout_fractions_by_playbook") or {}
    holdout_seed_recipe = policy.get("holdout_seed_recipe") or ""
    eligible_df = _apply_holdout_assignment(
        eligible_df,
        holdout_fractions=holdout_fractions,
        seed_recipe=holdout_seed_recipe,
        entity_id_column=entity_id_column,
    )

    capacity_map = policy.get("capacity_by_playbook") or {}
    cohort_partition = (
        [F.col("playbook_id"), F.col(capacity_partition_column)]
        if capacity_partition_column and capacity_partition_column in eligible_df.columns
        else [F.col("playbook_id")]
    )
    cohort_window = (
        Window.partitionBy(*cohort_partition).orderBy(F.col(probability_column).desc())
    )
    eligible_df = eligible_df.withColumn(
        "priority_rank_within_cohort", F.row_number().over(cohort_window)
    )

    capacity_col = _capacity_lookup_column(capacity_map)
    capacity_breached = (
        F.col("priority_rank_within_cohort") > capacity_col
    ) & capacity_col.isNotNull()
    suppression_with_capacity = F.when(
        F.col("playbook_suppressed_reason").isNotNull(), F.col("playbook_suppressed_reason")
    ).when(capacity_breached, F.lit("capacity_exceeded")).otherwise(F.lit(None).cast("string"))

    entity_window = Window.partitionBy(F.col(entity_id_column)).orderBy(
        F.col(probability_column).desc(), F.col("playbook_id")
    )
    entity_set_window = Window.partitionBy(F.col(entity_id_column))

    eligible_df = (
        eligible_df.withColumn("playbook_suppressed_reason", suppression_with_capacity)
        .withColumn(
            "recommended",
            F.col("playbook_suppressed_reason").isNull() & ~F.col("is_holdout"),
        )
        .withColumn("policy_rank_among_eligible", F.row_number().over(entity_window))
        .withColumn(
            "eligible_playbooks_set",
            F.collect_set(F.col("playbook_id")).over(entity_set_window),
        )
        .withColumn("eligible_playbook_count", F.size(F.col("eligible_playbooks_set")))
    )
    return eligible_df


def write_snapshot(
    spark: "SparkSession",
    snapshot_df: "DataFrame",
    snapshot_table_fqn: str,
) -> Dict[str, str]:
    """MERGE the snapshot DataFrame into ``eligibility_snapshot``.

    Idempotent on ``(scoring_run_id, entity_id, playbook_id)``. The first
    write into a fresh target creates the table with the schema declared
    in ``schemas.eligibility_snapshot_schema``.

    Delegates to :func:`delta_writer.merge_dataframe_into`, so the merge
    runs entirely on the Spark side — ``snapshot_df`` is **never**
    collected to the driver and can be tens of millions of rows wide.
    """
    from .delta_writer import merge_dataframe_into
    from .schemas import eligibility_snapshot_schema

    merge_dataframe_into(
        spark,
        snapshot_df,
        eligibility_snapshot_schema(),
        snapshot_table_fqn,
        ("scoring_run_id", "entity_id", "playbook_id"),
    )
    return {"target": snapshot_table_fqn}


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def build_eligibility_snapshot(config: SnapshotConfig) -> SnapshotResult:
    """End-to-end driver: load definitions, score, evaluate, apply, MERGE."""
    spark = config.spark
    as_of_date = (config.as_of_date or datetime.now(timezone.utc)).replace(microsecond=0)
    if as_of_date.tzinfo is None:
        as_of_date = as_of_date.replace(tzinfo=timezone.utc)

    definitions = _load_active_definitions(spark, config)
    _require_definitions_present(definitions, config)

    predictions_df = _load_latest_predictions(spark, config)

    scoring_run_id = compute_scoring_run_id(
        config.model_name,
        config.model_version,
        definitions.archetype_version,
        definitions.eligibility_policy_version,
        definitions.decision_policy_version,
        as_of_date,
    )

    # Raw features must be on the DataFrame BEFORE assign_archetype, not
    # only before evaluate_eligibility. assign_archetype silently skips
    # centroid features that don't exist in the input (cold-start
    # behavior) — without raw features joined in, every row would land
    # on the last archetype with distance=0. Policy-required features
    # are a hard contract; archetype-referenced features are projected
    # if present in gold and silently skipped otherwise.
    scored_df = _join_features_for_predicates(
        spark=spark,
        scored_df=predictions_df,
        config=config,
        policies=definitions.policies,
        archetypes=definitions.archetypes,
    )
    enriched_df = assign_archetype(scored_df, definitions.archetypes)
    eligible_df = evaluate_eligibility(enriched_df, definitions.policies)
    if config.top_shap_drivers_fqn:
        eligible_df = _join_top_shap_drivers(spark, eligible_df, config)

    risk_tier_high, risk_tier_medium = _resolve_risk_tier_thresholds(
        config, definitions.decision_policy
    )
    decisioned_df = apply_decision_policy(
        eligible_df,
        definitions.decision_policy,
        entity_id_column=config.entity_id_column,
        probability_column=config.probability_column,
        risk_tier_high=risk_tier_high,
        risk_tier_medium=risk_tier_medium,
        capacity_partition_column=config.capacity_partition_column,
    )

    # Cache before consuming twice (snapshot MERGE + summary aggregation).
    # Without this, the entire predict→assign→fan-out→window plan is
    # recomputed for each downstream consumer — see Coding_Practices.md
    # §"Batched .agg() Pattern" / clusterer.py for the canonical pattern.
    _cache_dataframe(decisioned_df)
    try:
        snapshot_df = _shape_snapshot_rows(
            decisioned_df,
            config=config,
            scoring_run_id=scoring_run_id,
            as_of_date=as_of_date,
            definitions=definitions,
        )
        write_snapshot(spark, snapshot_df, config.snapshot_table_fqn)
        counts = _compute_snapshot_counts(decisioned_df)
    finally:
        _unpersist_dataframe(decisioned_df)

    result = SnapshotResult(
        scoring_run_id=scoring_run_id,
        as_of_date=as_of_date,
        total_eligible_rows=counts["total"],
        recommended_rows=counts["recommended"],
        holdout_rows=counts["holdout"],
        suppressed_rows=counts["suppressed"],
        archetypes_considered=len(definitions.archetypes),
        policies_considered=len(definitions.policies),
        risk_tier_counts=counts["risk_tiers"],
        decision_policy_id=definitions.decision_policy_id,
    )
    logger.info("%s", result.summary())
    return result


def _cache_dataframe(df: "DataFrame") -> None:
    try:
        df.cache()
    except (AttributeError, RuntimeError) as exc:
        logger.debug("snapshot cache() failed (non-fatal): %s", exc)


def _unpersist_dataframe(df: "DataFrame") -> None:
    try:
        df.unpersist()
    except (AttributeError, RuntimeError) as exc:
        logger.debug("snapshot unpersist() failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Internal helpers — single responsibility, no side effects beyond Spark queries
# ---------------------------------------------------------------------------


def _load_active_definitions(spark: "SparkSession", config: SnapshotConfig) -> _LoadedDefinitions:
    archetypes = _query_active_rows(
        spark,
        config.archetype_catalog_fqn,
        config.model_name,
        config.model_version,
    )
    policies = _query_active_rows(
        spark,
        config.eligibility_policy_fqn,
        config.model_name,
        config.model_version,
    )
    decision_row = _query_active_decision_policy(spark, config)

    archetype_version = _resolve_definition_version(archetypes, "archetype_version")
    policy_version = _resolve_definition_version(policies, "version")
    decision_version = str((decision_row or {}).get("version") or "")
    decision_policy_id = (decision_row or {}).get("decision_policy_id")
    return _LoadedDefinitions(
        archetypes=archetypes,
        policies=policies,
        decision_policy=decision_row,
        archetype_version=archetype_version,
        eligibility_policy_version=policy_version,
        decision_policy_version=decision_version,
        decision_policy_id=str(decision_policy_id) if decision_policy_id else None,
    )


def _join_top_shap_drivers(
    spark: "SparkSession", eligible_df: "DataFrame", config: SnapshotConfig
) -> "DataFrame":
    """Left-join the per-account top SHAP drivers cache onto the eligible rows.

    Phase 2's derivation step caches per-row top SHAP drivers into
    ``top_shap_drivers`` for the training cohort. We join those rows in
    by account, leaving ``top_shap_features`` NULL for accounts not in
    the cache (the dashboard view falls back to the archetype-global
    drivers from ``archetype_catalog.top_shap_features`` for those).

    The join is a single Spark left join on ``entity_id`` — distributed,
    no driver collect.
    """
    if not config.top_shap_drivers_fqn:
        return eligible_df
    if not spark.catalog.tableExists(config.top_shap_drivers_fqn):
        logger.warning(
            "top_shap_drivers_fqn=%s does not exist; snapshot rows will have NULL top_shap_features",
            config.top_shap_drivers_fqn,
        )
        return eligible_df

    drivers = spark.sql(
        f"SELECT entity_id, top_drivers AS top_shap_features "
        f"FROM {config.top_shap_drivers_fqn} "
        f"WHERE model_name = ? AND model_version = ?",
        args=[config.model_name, config.model_version],
    )
    join_key = config.entity_id_column
    if join_key != "entity_id":
        drivers = drivers.withColumnRenamed("entity_id", join_key)
    # Plain left join — let Spark's cost-based optimizer pick broadcast vs.
    # sort-merge based on the actual driver-table size. Forcing broadcast
    # would OOM the executors on large customer bases (top_shap_drivers
    # has one row per training-cohort account).
    return eligible_df.join(drivers, on=join_key, how="left")


def _require_definitions_present(
    definitions: _LoadedDefinitions, config: SnapshotConfig
) -> None:
    """Fail fast on missing active rows.

    The eligibility snapshot has non-nullable columns for every layer of
    the four-way anchor. Letting None flow into a non-nullable column
    silently corrupts the Delta MERGE. We surface the missing-layer
    condition here so the operator sees a precise error message.
    """
    missing: List[str] = []
    if not definitions.archetypes:
        missing.append(
            f"no active archetype_catalog rows for {config.model_name}/{config.model_version}"
        )
    if not definitions.policies:
        missing.append(
            f"no active eligibility_policy rows for {config.model_name}/{config.model_version}"
        )
    if definitions.decision_policy is None or not definitions.decision_policy_id:
        missing.append(
            f"no active decision_policy row at {config.decision_policy_fqn}"
        )
    if missing:
        raise RuntimeError(
            "build_eligibility_snapshot cannot run: "
            + "; ".join(missing)
            + ". Run c01_publish_definitions and c02_archetype_derivation first."
        )


def _query_active_rows(
    spark: "SparkSession",
    table_fqn: str,
    model_name: str,
    model_version: str,
) -> List[Dict[str, Any]]:
    if not spark.catalog.tableExists(table_fqn):
        return []
    rows = spark.sql(
        f"SELECT * FROM {table_fqn} "
        "WHERE status = 'active' AND model_name = ? AND model_version = ?",
        args=[model_name, model_version],
    ).collect()
    return [row.asDict(recursive=True) for row in rows]


def _query_active_decision_policy(
    spark: "SparkSession", config: SnapshotConfig
) -> Optional[Dict[str, Any]]:
    if not spark.catalog.tableExists(config.decision_policy_fqn):
        return None
    if config.decision_policy_id:
        rows = spark.sql(
            f"SELECT * FROM {config.decision_policy_fqn} "
            "WHERE decision_policy_id = ? "
            "ORDER BY valid_from DESC LIMIT 1",
            args=[config.decision_policy_id],
        ).collect()
    else:
        rows = spark.sql(
            f"SELECT * FROM {config.decision_policy_fqn} "
            "WHERE valid_to IS NULL "
            "ORDER BY valid_from DESC LIMIT 1"
        ).collect()
    if not rows:
        return None
    return rows[0].asDict(recursive=True)


def _resolve_definition_version(rows: Sequence[Mapping[str, Any]], field_name: str) -> str:
    versions = sorted({str(row.get(field_name) or "") for row in rows if row.get(field_name)})
    if not versions:
        return ""
    return versions[-1]


def _join_features_for_predicates(
    spark: "SparkSession",
    scored_df: "DataFrame",
    config: SnapshotConfig,
    policies: Sequence[Mapping[str, Any]],
    archetypes: Sequence[Mapping[str, Any]] = (),
) -> "DataFrame":
    """Join raw features from gold onto ``scored_df`` before the c05 pipeline.

    The predictions table is the model's output. Two downstream consumers
    need raw features back on each row:

    - ``assign_archetype`` reads per-archetype ``centroid_feature_order``
      and computes scaled Euclidean distance; missing feature columns
      are silently skipped, so without the join every row lands on the
      last archetype with distance=0.
    - ``evaluate_eligibility`` compiles predicates referencing raw
      feature columns (e.g. ``active_span_days >= 42``); a missing
      column surfaces as ``UNRESOLVED_COLUMN`` at filter time.

    The projection is the union of:
    (a) ``requires_features`` across every active policy — hard
        contract: missing from gold ⇒ RuntimeError.
    (b) ``centroid_feature_order`` across every active archetype —
        soft: only projected if present in gold; missing features are
        silently skipped so cold-start archetypes still work.

    The join runs entirely inside Spark; no collect, no broadcast hint
    (Spark's CBO picks broadcast vs. sort-merge from real gold size, so
    executors don't OOM on large gold tables).

    Fail-fast guards:
    - Policies need features but ``gold_features_fqn`` is unset.
    - ``gold_features_fqn`` is set but the table does not exist.
    - Policy-required features are missing from gold's schema.
    - ``entity_id_column`` is missing from either side.
    """
    policy_needed = _required_features_union(policies)
    archetype_wanted = _archetype_feature_union(archetypes)
    needed = policy_needed | archetype_wanted
    if not needed:
        return scored_df

    if not config.gold_features_fqn:
        raise RuntimeError(
            "Eligibility policies reference raw feature columns "
            f"({sorted(policy_needed)[:5]}{'…' if len(policy_needed) > 5 else ''}) "
            "but SnapshotConfig.gold_features_fqn is not set. Point it at the "
            "gold features table used at training time so c05 can evaluate "
            "the predicates against each scored entity's features."
        )
    if not spark.catalog.tableExists(config.gold_features_fqn):
        raise RuntimeError(
            f"SnapshotConfig.gold_features_fqn={config.gold_features_fqn!r} "
            "does not exist — cannot evaluate eligibility predicates."
        )

    entity_key = config.entity_id_column
    gold = spark.table(config.gold_features_fqn)
    if entity_key not in gold.columns:
        raise RuntimeError(
            f"entity_id_column {entity_key!r} missing from "
            f"{config.gold_features_fqn}; cannot join features."
        )
    if entity_key not in scored_df.columns:
        raise RuntimeError(
            f"entity_id_column {entity_key!r} missing from predictions "
            "dataframe; cannot join features."
        )

    gold_columns = set(gold.columns)
    missing_policy = sorted(n for n in policy_needed if n not in gold_columns)
    if missing_policy:
        raise RuntimeError(
            f"Gold features table {config.gold_features_fqn} is missing "
            f"{len(missing_policy)} feature(s) referenced by eligibility policies: "
            f"{missing_policy}. The model or the policies drifted — "
            "re-run c02 against the current gold schema or fix the gold "
            "table to include these columns."
        )
    # Archetype-only features that are missing from gold are softly skipped —
    # assign_archetype is cold-start tolerant and will skip these features per
    # archetype. Log them for audit so silent degradation is visible.
    missing_archetype = sorted(
        n for n in (archetype_wanted - policy_needed) if n not in gold_columns
    )
    if missing_archetype:
        logger.warning(
            "Gold %s is missing %d archetype centroid feature(s); "
            "assign_archetype will skip them per archetype: %s",
            config.gold_features_fqn,
            len(missing_archetype),
            missing_archetype,
        )

    # Keep only features that exist on gold. Drop any collisions with
    # columns already on scored_df so scored_df's versions win (predictions
    # may carry a subset for audit).
    projected = [n for n in sorted(needed) if n in gold_columns]
    overlap = set(scored_df.columns) & set(projected) - {entity_key}
    projection = [entity_key] + [c for c in projected if c not in overlap]
    gold_slim = gold.select(*projection)
    return scored_df.join(gold_slim, on=entity_key, how="left")


def _required_features_union(policies: Sequence[Mapping[str, Any]]) -> set:
    """Union of ``requires_features`` across all active policies.

    Each policy row has a ``requires_features: array<string>`` column.
    The union is the smallest set of gold columns the snapshot needs to
    project for predicate evaluation.
    """
    out: set = set()
    for policy in policies or []:
        values = policy.get("requires_features") or []
        if isinstance(values, str):
            continue
        for name in values:
            if name:
                out.add(str(name))
    return out


def _archetype_feature_union(archetypes: Sequence[Mapping[str, Any]]) -> set:
    """Union of ``centroid_feature_order`` across all active archetypes.

    Same projection-pushdown motivation: assign_archetype needs these
    columns on the input to compute scaled Euclidean distance to each
    centroid. The union across archetypes is the minimal projection
    that serves every archetype's centroid vector.
    """
    out: set = set()
    for arch in archetypes or []:
        order = arch.get("centroid_feature_order") or []
        if isinstance(order, str):
            continue
        for name in order:
            if name:
                out.add(str(name))
    return out


def _load_latest_predictions(spark: "SparkSession", config: SnapshotConfig) -> "DataFrame":
    """Return the predictions for the most recent inference run.

    Stays entirely inside Spark SQL via a subquery on
    ``inference_timestamp_column``: no Python timestamp scalar is ever
    constructed and re-bound as a literal, so the call is safe against
    both ``TIMESTAMP`` and ``TIMESTAMP_NTZ`` columns (the
    ``cast_timestamp_ntz_to_long`` mismatch documented in
    ``Coding_Practices.md`` cannot occur here).
    """
    if not spark.catalog.tableExists(config.predictions_fqn):
        raise RuntimeError(
            f"predictions table {config.predictions_fqn} does not exist; "
            "run s10_batch_inference before c04"
        )
    df = spark.table(config.predictions_fqn)
    ts_col = config.inference_timestamp_column
    if ts_col not in df.columns:
        return df
    return spark.sql(
        f"SELECT * FROM {config.predictions_fqn} "
        f"WHERE `{ts_col}` = (SELECT MAX(`{ts_col}`) FROM {config.predictions_fqn})"
    )


def _parse_predicate(rules_json: Any) -> Optional[Dict[str, Any]]:
    if rules_json is None:
        return None
    if isinstance(rules_json, dict):
        return rules_json
    try:
        parsed = json.loads(rules_json)
    except (TypeError, ValueError):
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _compile_suppression(rules_json: Any) -> Tuple["Column", str]:
    """Translate a JSON suppression predicate into a Spark column + reason."""
    from pyspark.sql import functions as F  # noqa: N812

    from .predicate_compiler import compile_predicate

    parsed = _parse_predicate(rules_json)
    if not parsed:
        return F.lit(False), "suppressed"
    reason = str(parsed.get("reason") or "suppressed")
    body = parsed.get("predicate") or parsed
    return compile_predicate(body), reason


def _capacity_lookup_column(capacity_map: Mapping[str, Any]) -> "Column":
    from pyspark.sql import functions as F  # noqa: N812

    entries = {str(k): int(v) for k, v in capacity_map.items() if v is not None}
    if not entries:
        return F.lit(None).cast("int")
    map_col = F.create_map(*[x for k, v in entries.items() for x in (F.lit(k), F.lit(v))])
    return map_col[F.col("playbook_id")].cast("int")


def _apply_holdout_assignment(
    df: "DataFrame",
    *,
    holdout_fractions: Mapping[str, Any],
    seed_recipe: str,
    entity_id_column: str,
) -> "DataFrame":
    """Stable per-(account, playbook) holdout assignment via deterministic hash."""
    from pyspark.sql import functions as F  # noqa: N812
    from pyspark.sql.types import LongType

    seed_columns = _seed_columns(seed_recipe, df.columns, entity_id_column)
    seed_concat = F.concat_ws("|", *(F.coalesce(F.col(c).cast("string"), F.lit("")) for c in seed_columns))
    seed_concat = F.concat(seed_concat, F.lit("|"), F.col("playbook_id"))
    bucket_col = (F.abs(F.xxhash64(seed_concat)).cast(LongType()) % F.lit(10_000)).cast("double") / F.lit(10_000.0)

    fraction_col = _holdout_fraction_lookup(holdout_fractions)
    is_holdout_col = (bucket_col < fraction_col) & fraction_col.isNotNull()

    return (
        df.withColumn("holdout_seed", seed_concat)
        .withColumn("__cr_holdout_bucket", bucket_col)
        .withColumn("is_holdout", is_holdout_col)
        .withColumn(
            "holdout_stratum",
            F.when(F.col("is_holdout"), F.concat_ws("/", F.col("playbook_id"), F.col("risk_tier"))).otherwise(
                F.lit(None).cast("string")
            ),
        )
        .drop("__cr_holdout_bucket")
    )


def _holdout_fraction_lookup(holdout_fractions: Mapping[str, Any]) -> "Column":
    from pyspark.sql import functions as F  # noqa: N812

    entries = {str(k): float(v) for k, v in holdout_fractions.items() if v is not None}
    if not entries:
        return F.lit(_DEFAULT_HOLDOUT_FRACTION)
    map_col = F.create_map(*[x for k, v in entries.items() for x in (F.lit(k), F.lit(v))])
    return F.coalesce(map_col[F.col("playbook_id")], F.lit(_DEFAULT_HOLDOUT_FRACTION))


def _seed_columns(seed_recipe: str, available: Sequence[str], entity_id_column: str) -> List[str]:
    if not seed_recipe:
        return [entity_id_column]
    requested = [c.strip() for c in seed_recipe.split(",") if c.strip()]
    resolved = [c for c in requested if c in available]
    if not resolved:
        return [entity_id_column]
    return resolved


def _resolve_risk_tier_thresholds(
    config: SnapshotConfig, decision_policy: Optional[Mapping[str, Any]]
) -> Tuple[float, float]:
    if config.risk_tier_high is not None and config.risk_tier_medium is not None:
        return float(config.risk_tier_high), float(config.risk_tier_medium)
    high = (decision_policy or {}).get("risk_tier_high_threshold")
    medium = (decision_policy or {}).get("risk_tier_medium_threshold")
    return float(high if high is not None else 0.6), float(medium if medium is not None else 0.3)


def _shape_snapshot_rows(
    decisioned_df: "DataFrame",
    *,
    config: SnapshotConfig,
    scoring_run_id: str,
    as_of_date: datetime,
    definitions: _LoadedDefinitions,
) -> "DataFrame":
    from pyspark.sql import functions as F  # noqa: N812

    written_at = datetime.now(timezone.utc)
    base_df = decisioned_df
    if config.entity_id_column != "entity_id":
        base_df = base_df.withColumnRenamed(config.entity_id_column, "entity_id")
    eligibility_id_col = F.sha1(
        F.concat_ws("|", F.lit(scoring_run_id), F.col("entity_id"), F.col("playbook_id"))
    )
    base_df = (
        base_df.withColumn("eligibility_id", eligibility_id_col)
        .withColumn("scoring_run_id", F.lit(scoring_run_id))
        .withColumn("as_of_date", F.lit(as_of_date))
    )

    value_at_risk_col = (
        F.col(config.value_at_risk_column).cast("double")
        if config.value_at_risk_column and config.value_at_risk_column in base_df.columns
        else F.lit(None).cast("double")
    )
    top_shap_col = (
        F.col("top_shap_features")
        if "top_shap_features" in base_df.columns
        else F.lit(None).cast(_TOP_SHAP_DTYPE)
    )
    return base_df.select(
        F.col("eligibility_id"),
        F.col("scoring_run_id"),
        F.col("as_of_date"),
        F.col("entity_id"),
        F.col("playbook_id"),
        F.col("playbook_version"),
        F.col("archetype_id"),
        F.col("archetype_version"),
        F.col("eligibility_policy_id"),
        F.col("eligibility_policy_version"),
        F.lit(definitions.decision_policy_id).alias("decision_policy_id"),
        F.lit(config.model_name).alias("model_name"),
        F.lit(config.model_version).alias("model_version"),
        F.col(config.probability_column).cast("double").alias("churn_probability"),
        value_at_risk_col.alias("value_at_risk"),
        F.col("risk_tier"),
        top_shap_col.alias("top_shap_features"),
        F.col("eligible_playbooks_set"),
        F.col("eligible_playbook_count"),
        F.col("policy_rank_among_eligible"),
        F.col("priority_rank_within_cohort"),
        F.col("eligibility_evidence"),
        F.col("is_holdout"),
        F.col("holdout_stratum"),
        F.col("holdout_seed"),
        F.col("playbook_suppressed_reason"),
        F.col("recommended"),
        F.lit(written_at).alias("written_at"),
    )


def _compute_snapshot_counts(decisioned_df: "DataFrame") -> Dict[str, Any]:
    """Single-job aggregate for the run summary.

    All counts (total / recommended / holdout / suppressed / per-risk-tier)
    are folded into one ``.agg(*exprs).head()`` call so the orchestrator
    fires exactly one Spark job for the summary, not the previous two
    (one for the totals + one for the risk-tier ``groupBy().count()``).
    """
    from pyspark.sql import functions as F  # noqa: N812

    risk_columns = {
        _RISK_TIER_HIGH: "risk_high",
        _RISK_TIER_MEDIUM: "risk_medium",
        _RISK_TIER_LOW: "risk_low",
    }
    risk_exprs = [
        F.sum((F.col("risk_tier") == F.lit(tier)).cast("int")).alias(alias)
        for tier, alias in risk_columns.items()
    ]
    row = decisioned_df.agg(
        F.count(F.lit(1)).alias("total"),
        F.sum(F.col("recommended").cast("int")).alias("recommended"),
        F.sum(F.col("is_holdout").cast("int")).alias("holdout"),
        F.sum(F.col("playbook_suppressed_reason").isNotNull().cast("int")).alias("suppressed"),
        *risk_exprs,
    ).head()
    return {
        "total": int(row["total"] or 0),
        "recommended": int(row["recommended"] or 0),
        "holdout": int(row["holdout"] or 0),
        "suppressed": int(row["suppressed"] or 0),
        "risk_tiers": {tier: int(row[alias] or 0) for tier, alias in risk_columns.items()},
    }
