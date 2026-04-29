"""Top-level orchestrator for the archetype-derivation pipeline.

Wires together the Phase 2 modules:

    shap_attribution.load_attribution_from_model_uri  (training artifact)
    shap_runner.compute_shap_distributed              (linear attribution)
    clusterer.cluster_kmeans
    clusterer.cluster_centroids_raw   (raw-feature centroids for runtime)
    clusterer.cluster_target_means     (mean churn probability per cluster)
    clusterer.cluster_size_stats
    rule_extractor.extract_eligibility_rules
    playbook_mapper.map_archetypes_to_playbooks
    llm_namer.build_llm_namer
    delta_writer.merge_into            (archetype_catalog, eligibility_policy)

NB12 cell 3 calls a single function on this module:
``derive_archetypes_and_policies(config)``. Everything else is private.

The orchestrator stays distributed throughout the heavy steps. The only
points where data lands on the driver are the small attribution dict
(loaded once from the model's MLflow run), the per-cluster aggregations
bounded by ``k``, and the bounded surrogate-tree training set
(max ``MAX_SAMPLE_ROWS`` rows). The full training cohort is never
collected.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from customer_retention.stages.modeling.shap_attribution import (
    ShapAttribution,
    load_attribution_from_model_uri,
)

from .approval_gate import cosine_similarity
from .clusterer import (
    CLUSTER_COL,
    DEFAULT_FEATURE_CAP,
    DEFAULT_K_CAP,
    DEFAULT_K_RANGE,
    cluster_centroids_raw,
    cluster_kmeans,
    cluster_shap_centroids,
    cluster_size_stats,
    cluster_target_means,
)
from .llm_namer import LLMNamer, build_llm_namer
from .playbook_mapper import (
    ArchetypeMapping,
    ArchetypeSummary,
    map_archetypes_to_playbooks,
)
from .rule_extractor import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_PURITY,
    ExtractedRule,
    extract_eligibility_rules,
)
from .schemas import archetype_catalog_schema, eligibility_policy_schema
from .shap_runner import compute_shap_distributed

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame, SparkSession


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


DEFAULT_FIT_AUTO_THRESHOLD: float = 0.5
DEFAULT_FIT_REVIEW_THRESHOLD: float = 0.2
FIT_TIER_AUTO: str = "auto"
FIT_TIER_REVIEW: str = "review"
FIT_TIER_MANUAL: str = "manual"
FIT_TIER_CATCH_ALL: str = "catch_all"


@dataclass
class FitThresholds:
    """Three-tier cutoffs for archetype-to-playbook fit scores.

    Every (archetype, playbook) pair produced by the namer is classified:

    - ``fit_score >= auto`` → tier ``auto``. Candidate for auto-promotion
      in c03 approval gate.
    - ``review <= fit_score < auto`` → tier ``review``. Written as
      ``pending_review`` for manual approval; never auto-promoted.
    - ``fit_score < review`` → tier ``manual``. Not written as a regular
      policy row. If the archetype has no ``auto``/``review`` matches
      AND a catch-all playbook is configured, one catch-all row is
      emitted for that archetype (tier ``catch_all``).
    """

    auto: float = DEFAULT_FIT_AUTO_THRESHOLD
    review: float = DEFAULT_FIT_REVIEW_THRESHOLD

    def classify(self, fit_score: float) -> str:
        if fit_score >= self.auto:
            return FIT_TIER_AUTO
        if fit_score >= self.review:
            return FIT_TIER_REVIEW
        return FIT_TIER_MANUAL


@dataclass
class DerivationConfig:
    """Inputs for one derivation run.

    The orchestrator does not own catalog/schema resolution — the caller
    (NB12 cell 3) passes the fully-qualified table names. Defaults match
    the names defined in ``schemas.py``.
    """

    spark: "SparkSession"
    training_df: "DataFrame"
    raw_feature_df: "DataFrame"
    feature_columns: Sequence[str]
    model_uri: str
    target_column: str
    join_key: str = "entity_id"
    archetype_catalog_fqn: str = ""
    eligibility_policy_fqn: str = ""
    playbooks: Sequence[Dict[str, Any]] = field(default_factory=list)
    gold_feature_names: Sequence[str] = field(default_factory=list)
    model_name: str = ""
    model_version: str = ""
    derivation_run_id: Optional[str] = None
    k_range: Tuple[int, int] = DEFAULT_K_RANGE
    k_cap: int = DEFAULT_K_CAP
    feature_cap: int = DEFAULT_FEATURE_CAP
    max_depth: int = DEFAULT_MAX_DEPTH
    min_purity: float = DEFAULT_MIN_PURITY
    llm_endpoint_name: Optional[str] = None
    llm_namer: Optional[LLMNamer] = None
    enrichment_builder: Optional[Any] = None
    # Optional explicit RunNamespace used for prose rendering sidecar lookup.
    # When None, `_render_predicate_prose_safely` falls back to env-based
    # discovery (flaky on clusters where ``get_experiments_dir`` doesn't
    # resolve to the active run mount). Passing it explicitly is strongly
    # recommended on Databricks.
    namespace: Optional[Any] = None
    write: bool = True
    attribution: Optional[ShapAttribution] = None
    fit_thresholds: FitThresholds = field(default_factory=FitThresholds)
    default_playbook_id: Optional[str] = None


@dataclass
class DerivationResult:
    """Output of one derivation run, suitable for the NB12 summary cell."""

    derivation_run_id: str
    best_k: int
    best_silhouette: float
    cluster_sizes: List[Tuple[int, int]]
    cluster_target_means: List[Tuple[int, float]]
    archetype_rows: List[Dict[str, Any]]
    eligibility_policy_rows: List[Dict[str, Any]]
    attribution: ShapAttribution
    extracted_rules: List[ExtractedRule]
    mappings: List[ArchetypeMapping]
    llm_model_id: Optional[str]

    def summary(self) -> str:
        tier_counts = self.fit_tier_counts()
        tier_str = (
            f"auto={tier_counts.get(FIT_TIER_AUTO, 0)} "
            f"review={tier_counts.get(FIT_TIER_REVIEW, 0)} "
            f"catch_all={tier_counts.get(FIT_TIER_CATCH_ALL, 0)}"
        )
        coverage = self.coverage_report()
        uncovered = [a for a, tiers in coverage.items() if not tiers]
        uncovered_str = (
            f"; archetypes without any matched playbook: {sorted(uncovered)}"
            if uncovered
            else ""
        )
        return (
            f"Derivation {self.derivation_run_id}: best_k={self.best_k} "
            f"silhouette={self.best_silhouette:.4f}; "
            f"{len(self.archetype_rows)} archetype rows + "
            f"{len(self.eligibility_policy_rows)} policy rows ({tier_str}); "
            f"llm={self.llm_model_id or 'prose_overlap'}"
            f"{uncovered_str}"
        )

    def fit_tier_counts(self) -> Dict[str, int]:
        """Count policy rows per fit tier."""
        counts: Dict[str, int] = {}
        for row in self.eligibility_policy_rows:
            tier = str(row.get("fit_tier") or FIT_TIER_MANUAL)
            counts[tier] = counts.get(tier, 0) + 1
        return counts

    def coverage_report(self) -> Dict[str, List[str]]:
        """Map archetype_id → list of tiers present in its policy rows.

        Empty list means the archetype has no eligibility_policy rows at
        all — not covered even by the catch-all fallback. The c02 summary
        flags this; c05's snapshot will fail if any active archetype has
        zero active policy rows.
        """
        by_archetype: Dict[str, List[str]] = {
            str(row.get("archetype_id")): []
            for row in self.archetype_rows
        }
        # archetype_id lookup via archetype_version (policy rows carry version in archetype_ids[0])
        version_to_id = {
            str(row.get("archetype_version")): str(row.get("archetype_id"))
            for row in self.archetype_rows
        }
        for policy in self.eligibility_policy_rows:
            ids = policy.get("archetype_ids") or []
            for aver in ids:
                arch_id = version_to_id.get(str(aver))
                if arch_id is not None:
                    by_archetype.setdefault(arch_id, []).append(
                        str(policy.get("fit_tier") or FIT_TIER_MANUAL)
                    )
        return by_archetype


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def derive_archetypes_and_policies(config: DerivationConfig) -> DerivationResult:
    """Run the full SHAP → clustering → rules → mapping → write pipeline.

    Returns a ``DerivationResult`` that the c02 derivation cell prints and
    persists in its session metadata. When ``config.write`` is ``False``,
    the rows are built and returned but no Delta writes happen — useful
    for unit tests.
    """
    derivation_run_id = config.derivation_run_id or _make_run_id(config)
    logger.info("Starting causal derivation run_id=%s", derivation_run_id)

    attribution = config.attribution or load_attribution_from_model_uri(config.model_uri)
    logger.info(
        "Attribution loaded: %d features, sample_size=%d",
        len(attribution.feature_columns),
        attribution.sample_size,
    )

    shap_result = compute_shap_distributed(
        spark_df=config.training_df,
        attribution=attribution,
        join_key=config.join_key,
    )
    if shap_result.shap_df is None:
        raise RuntimeError("compute_shap_distributed returned an empty result")

    clustering = cluster_kmeans(
        shap_df=shap_result.shap_df,
        feature_columns=shap_result.shap_columns,
        k_range=config.k_range,
        k_cap=config.k_cap,
        feature_cap=config.feature_cap,
    )
    if clustering.labelled_df is None:
        raise RuntimeError("cluster_kmeans returned an empty labelled DataFrame")
    selected_shap_columns = clustering.feature_order
    selected_raw_features = [
        c[len("shap_") :] if c.startswith("shap_") else c for c in selected_shap_columns
    ]

    raw_with_clusters = _join_cluster_labels(
        clustering.labelled_df, config.raw_feature_df, config.join_key
    )

    raw_centroids, raw_feature_order = cluster_centroids_raw(
        raw_with_clusters,
        feature_columns=selected_raw_features,
        cluster_col=CLUSTER_COL,
    )
    feature_scales = _compute_feature_scales(raw_with_clusters, raw_feature_order)
    shap_centroids, _ = cluster_shap_centroids(
        clustering.labelled_df,
        shap_columns=selected_shap_columns,
        cluster_col=CLUSTER_COL,
    )
    sizes = cluster_size_stats(clustering.labelled_df, cluster_col=CLUSTER_COL)
    mean_targets = _join_target_means(
        clustering.labelled_df, raw_with_clusters, config.target_column
    )

    surrogate_inputs = _collect_surrogate_inputs(
        raw_with_clusters, selected_raw_features, CLUSTER_COL
    )
    extracted_rules = extract_eligibility_rules(
        raw_features=surrogate_inputs.features,
        feature_names=raw_feature_order,
        cluster_labels=surrogate_inputs.labels,
        max_depth=config.max_depth,
        min_purity=config.min_purity,
    )

    summaries = _build_archetype_summaries(
        sizes=sizes,
        mean_targets=mean_targets,
        shap_centroids=shap_centroids,
        shap_feature_order=selected_shap_columns,
        raw_centroids=raw_centroids,
        raw_feature_order=raw_feature_order,
    )

    namer = config.llm_namer or build_llm_namer(config.llm_endpoint_name)
    mappings = map_archetypes_to_playbooks(
        archetypes=summaries,
        playbooks=config.playbooks,
        gold_feature_names=config.gold_feature_names,
        llm_namer=namer,
        enrichment_builder=config.enrichment_builder,
    )

    timestamp = datetime.now(timezone.utc)
    archetype_rows = _build_archetype_rows(
        config=config,
        derivation_run_id=derivation_run_id,
        timestamp=timestamp,
        summaries=summaries,
        clustering_silhouette=clustering.best_silhouette,
        shap_centroids=shap_centroids,
        raw_centroids=raw_centroids,
        raw_feature_order=raw_feature_order,
        feature_scales=feature_scales,
        extracted_rules=extracted_rules,
        mappings=mappings,
        sizes=sizes,
        mean_targets=mean_targets,
    )
    policy_rows = _build_policy_rows(
        config=config,
        derivation_run_id=derivation_run_id,
        timestamp=timestamp,
        archetype_rows=archetype_rows,
        extracted_rules=extracted_rules,
        mappings=mappings,
    )

    _validate_policy_coverage(archetype_rows, policy_rows, mappings, config)

    if config.write:
        _write_rows(config, archetype_rows, policy_rows)

    return DerivationResult(
        derivation_run_id=derivation_run_id,
        best_k=clustering.best_k,
        best_silhouette=clustering.best_silhouette,
        cluster_sizes=sizes,
        cluster_target_means=mean_targets,
        archetype_rows=archetype_rows,
        eligibility_policy_rows=policy_rows,
        attribution=attribution,
        extracted_rules=extracted_rules,
        mappings=mappings,
        llm_model_id=namer.model_id,
    )


# ---------------------------------------------------------------------------
# Internals — Spark joins / surrogate inputs
# ---------------------------------------------------------------------------


@dataclass
class _SurrogateInputs:
    features: Any
    labels: Any


def _join_cluster_labels(
    labelled_shap_df: "DataFrame", raw_df: "DataFrame", join_key: str
) -> "DataFrame":
    """Attach per-entity cluster labels to the raw feature frame.

    Both sides are collapsed to one row per ``join_key`` before the inner
    join. If gold is snapshot-grained (multiple rows per ``entity_id``),
    the naïve join fans out to ``|raw| × |labelled-per-key|`` rows per
    entity and biases every downstream per-cluster aggregate
    (centroids, sizes, mean churn) toward entities with more snapshots.
    Collapsing first keeps each entity at one row and preserves the
    contract that archetypes are a property of the *entity*, not the
    snapshot.

    Caveat for multi-snapshot callers: ``dropDuplicates([join_key])`` keeps
    an arbitrary row per entity (Spark picks whichever survives the hash
    shuffle). The archetype label is stable across snapshots for a given
    entity so clustering is unaffected, but the *raw-feature centroid* is
    computed from the kept snapshot's features and will therefore drift
    between runs if gold has multiple snapshots per account. Callers that
    need reproducible centroids in that regime must pre-dedupe with an
    explicit ordering (e.g. ``Window.partitionBy(join_key).orderBy(
    F.col(timestamp_col).desc())`` + ``row_number() == 1``) before calling
    ``derive_archetypes_and_policies``."""
    cluster_lookup = labelled_shap_df.select(join_key, CLUSTER_COL).dropDuplicates([join_key])
    raw_unique = raw_df.dropDuplicates([join_key])
    return raw_unique.join(cluster_lookup, on=join_key, how="inner")


def _join_target_means(
    labelled_shap_df: "DataFrame",
    raw_with_clusters: "DataFrame",
    target_column: str,
) -> List[Tuple[int, float]]:
    if target_column in raw_with_clusters.columns:
        return cluster_target_means(raw_with_clusters, target_column)
    if target_column in labelled_shap_df.columns:
        return cluster_target_means(labelled_shap_df, target_column)
    return []


def _collect_surrogate_inputs(
    raw_with_clusters: "DataFrame",
    feature_columns: Sequence[str],
    cluster_col: str,
) -> _SurrogateInputs:
    """Collect a bounded subsample for the driver-side surrogate-tree fit.

    Caps the collect at ``rule_extractor.MAX_SAMPLE_ROWS`` so memory is
    bounded regardless of cohort size. Sampling is done in Spark via
    ``sample(fraction)`` so the work happens distributed; only the small
    sampled rows land on the driver.
    """
    import numpy as np

    from .rule_extractor import MAX_SAMPLE_ROWS

    select_cols = list(feature_columns) + [cluster_col]
    selected = raw_with_clusters.select(*select_cols)
    fraction = _bounded_fraction(selected, MAX_SAMPLE_ROWS)
    sampled = selected.sample(withReplacement=False, fraction=fraction, seed=42) if fraction < 1.0 else selected
    rows = sampled.limit(MAX_SAMPLE_ROWS).collect()
    if not rows:
        raise RuntimeError("Surrogate tree input is empty — no rows after sampling")
    matrix = np.array(
        [[_row_value(row, c) for c in feature_columns] for row in rows], dtype=float
    )
    labels = np.array([int(row[cluster_col]) for row in rows], dtype=int)
    return _SurrogateInputs(features=matrix, labels=labels)


def _bounded_fraction(spark_df: "DataFrame", cap: int, row_count: Optional[int] = None) -> float:
    total = row_count if row_count is not None else spark_df.count()
    if total <= 0 or total <= cap:
        return 1.0
    return min(1.0, (cap * 1.5) / total)


def _row_value(row: Any, column: str) -> float:
    """Coerce a driver-side row cell to a finite ``float`` for the numpy
    surrogate-tree matrix. NULL, NaN, and unparseable strings all collapse to
    ``0.0`` — sklearn ≤ 1.2 raises on NaN, and newer sklearn learns a "null
    pattern" as a split predicate (wrong semantics for eligibility rules)."""
    value = row[column]
    if value is None:
        return 0.0
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(coerced) else coerced


# ---------------------------------------------------------------------------
# Internals — summaries + row builders
# ---------------------------------------------------------------------------


_TOP_DRIVER_COUNT: int = 10


_SCALE_BATCH: int = 200


def _compute_feature_scales(
    raw_with_clusters: "DataFrame",
    feature_order: Sequence[str],
) -> List[float]:
    """Return per-feature population std-dev aligned with ``feature_order``.

    Batches aggregations in groups of ``_SCALE_BATCH`` columns per
    ``.agg()`` call so Catalyst plans stay O(200²) instead of O(N²) —
    matches Coding_Practices.md's bulk-agg pattern for per-column stats
    at ≥100 columns. Features absent from the DataFrame, or with zero /
    null / NaN std-dev (constant columns), fall back to ``1.0`` so the
    runtime's scaled-distance formula degrades to unscaled rather than
    dividing by zero.

    The returned list is aligned 1:1 with ``feature_order`` and is stored
    as ``centroid_feature_scales`` on every archetype row so the runtime
    snapshot writer can apply scaled Euclidean distance:
    ``scaled_diff = (x - centroid) / scale``.
    """
    from pyspark.sql import functions as F  # noqa: N812

    available = set(raw_with_clusters.columns)
    indexed = [(i, c) for i, c in enumerate(feature_order) if c in available]
    if not indexed:
        return [1.0] * len(feature_order)

    scales = [1.0] * len(feature_order)
    for start in range(0, len(indexed), _SCALE_BATCH):
        batch = indexed[start : start + _SCALE_BATCH]
        exprs = [
            F.stddev_pop(F.col(c)).alias(f"__scale_{i}") for i, (_, c) in enumerate(batch)
        ]
        row = raw_with_clusters.agg(*exprs).head()
        if row is None:
            continue
        for i, (feature_idx, _c) in enumerate(batch):
            v = row[f"__scale_{i}"]
            scales[feature_idx] = float(v) if (v is not None and float(v) > 0.0) else 1.0
    return scales


def _build_archetype_summaries(
    sizes: List[Tuple[int, int]],
    mean_targets: List[Tuple[int, float]],
    shap_centroids: List[List[float]],
    shap_feature_order: Sequence[str],
    raw_centroids: List[List[float]],
    raw_feature_order: Sequence[str],
) -> List[ArchetypeSummary]:
    target_lookup = {cluster_id: mean for cluster_id, mean in mean_targets}
    raw_lookup = {idx: vec for idx, vec in enumerate(raw_centroids)}
    summaries: List[ArchetypeSummary] = []
    for idx, (cluster_id, size) in enumerate(sizes):
        shap_vec = shap_centroids[idx] if idx < len(shap_centroids) else []
        raw_vec = raw_lookup.get(idx, [])
        positives, negatives = _split_drivers(
            shap_vec=shap_vec,
            shap_feature_order=shap_feature_order,
            raw_vec=raw_vec,
            raw_feature_order=raw_feature_order,
        )
        summaries.append(
            ArchetypeSummary(
                cluster_index=int(cluster_id),
                cluster_size=int(size),
                cluster_mean_churn_probability=float(target_lookup.get(cluster_id, 0.0)),
                top_positive_drivers=positives,
                top_negative_drivers=negatives,
            )
        )
    return summaries


def _split_drivers(
    shap_vec: Sequence[float],
    shap_feature_order: Sequence[str],
    raw_vec: Sequence[float],
    raw_feature_order: Sequence[str],
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    raw_lookup = {name: raw_vec[i] if i < len(raw_vec) else 0.0 for i, name in enumerate(raw_feature_order)}
    drivers: List[Dict[str, Any]] = []
    for idx, name in enumerate(shap_feature_order):
        if idx >= len(shap_vec):
            continue
        feature_name = name[len("shap_") :] if name.startswith("shap_") else name
        drivers.append(
            {
                "feature": feature_name,
                "mean_shap": float(shap_vec[idx]),
                "mean_value": float(raw_lookup.get(feature_name, 0.0)),
            }
        )
    drivers.sort(key=lambda d: -abs(d["mean_shap"]))
    positives = [d for d in drivers if d["mean_shap"] > 0][:_TOP_DRIVER_COUNT]
    negatives = [d for d in drivers if d["mean_shap"] < 0][:_TOP_DRIVER_COUNT]
    return positives, negatives


def _build_archetype_rows(
    config: DerivationConfig,
    derivation_run_id: str,
    timestamp: datetime,
    summaries: List[ArchetypeSummary],
    clustering_silhouette: float,
    shap_centroids: List[List[float]],
    raw_centroids: List[List[float]],
    raw_feature_order: List[str],
    feature_scales: List[float],
    extracted_rules: List[ExtractedRule],
    mappings: List[ArchetypeMapping],
    sizes: List[Tuple[int, int]],
    mean_targets: List[Tuple[int, float]],
) -> List[Dict[str, Any]]:
    rules_by_index = {rule.cluster_index: rule for rule in extracted_rules}
    mapping_by_index = {m.cluster_index: m for m in mappings}
    target_lookup = dict(mean_targets)
    rows: List[Dict[str, Any]] = []
    for idx, summary in enumerate(summaries):
        cluster_index = summary.cluster_index
        rule = rules_by_index.get(cluster_index)
        mapping = mapping_by_index.get(cluster_index)
        archetype_id = _archetype_id(config.model_name, config.model_version, cluster_index)
        archetype_version = _archetype_version(derivation_run_id, cluster_index)
        feature_thresholds = _struct_feature_thresholds(rule)
        shap_vector = shap_centroids[idx] if idx < len(shap_centroids) else []
        raw_vector = raw_centroids[idx] if idx < len(raw_centroids) else []
        rows.append(
            {
                "archetype_id": archetype_id,
                "archetype_version": archetype_version,
                "model_name": config.model_name,
                "model_version": config.model_version,
                "derivation_run_id": derivation_run_id,
                "derivation_method": "kmeans+shap",
                "derivation_params": json.dumps(
                    {
                        "k_range": list(config.k_range),
                        "k_cap": config.k_cap,
                        "silhouette": clustering_silhouette,
                    }
                ),
                "cluster_raw_id": str(cluster_index),
                "cluster_size": int(sizes[idx][1]) if idx < len(sizes) else int(summary.cluster_size),
                "cluster_mean_churn_probability": float(target_lookup.get(cluster_index, summary.cluster_mean_churn_probability)),
                "top_shap_features": _top_shap_struct(summary),
                "centroid_vector": shap_vector,
                "centroid_vector_raw": raw_vector,
                "centroid_feature_order": list(raw_feature_order),
                "centroid_feature_scales": list(feature_scales),
                "feature_thresholds": feature_thresholds,
                "name": mapping.archetype_name if mapping else f"Archetype {cluster_index}",
                "description": mapping.archetype_description if mapping else "",
                "rationale": mapping.rationale if mapping else "",
                "llm_model_id": mapping.llm_model_id if mapping else None,
                "stability_vs_prior_version": None,
                "status": "pending_review",
                "approved_by": None,
                "approved_at": None,
                "valid_from": None,
                "valid_to": None,
            }
        )
    return rows


def _build_policy_rows(
    config: DerivationConfig,
    derivation_run_id: str,
    timestamp: datetime,
    archetype_rows: List[Dict[str, Any]],
    extracted_rules: List[ExtractedRule],
    mappings: List[ArchetypeMapping],
) -> List[Dict[str, Any]]:
    """Apply the three-tier fit classifier and emit eligibility_policy rows.

    Behavior per archetype:

    - Every ``auto``-tier decision → one policy row (status=pending_review;
      approval gate promotes on fit_score >= auto threshold).
    - Every ``review``-tier decision → one policy row (status=pending_review;
      never auto-promoted — human picks from the queue).
    - ``manual``-tier decisions are NOT written as regular rows.
    - If an archetype has zero ``auto``/``review`` rows AND
      ``config.default_playbook_id`` is set, one catch-all row is emitted
      so every archetype has at least one eligible playbook. The
      catch-all row carries ``fit_tier = catch_all`` and its rationale
      flags that the archetype has no proper match.
    """
    rules_by_index = {rule.cluster_index: rule for rule in extracted_rules}
    archetype_by_cluster = {
        int(row["cluster_raw_id"]): row for row in archetype_rows
    }
    playbook_lookup = {str(p["playbook_id"]): p for p in config.playbooks}
    thresholds = config.fit_thresholds
    default_pb_id = (config.default_playbook_id or "").strip() or None
    rows: List[Dict[str, Any]] = []
    for mapping in mappings:
        archetype_row = archetype_by_cluster.get(mapping.cluster_index)
        if not archetype_row:
            continue
        rule = rules_by_index.get(mapping.cluster_index)
        if rule is None:
            continue
        emitted_for_archetype = 0
        for decision in mapping.fit_decisions:
            playbook = playbook_lookup.get(decision.playbook_id)
            if not playbook:
                continue
            tier = thresholds.classify(float(decision.fit_score))
            if tier == FIT_TIER_MANUAL:
                continue
            rows.append(
                _policy_row(
                    config=config,
                    derivation_run_id=derivation_run_id,
                    archetype_row=archetype_row,
                    playbook=playbook,
                    rule=rule,
                    mapping=mapping,
                    playbook_id=str(decision.playbook_id),
                    fit_score=float(decision.fit_score),
                    fit_tier=tier,
                    rationale=f"[{tier}] {decision.rationale}",
                )
            )
            emitted_for_archetype += 1
        if emitted_for_archetype == 0 and default_pb_id:
            default_pb = playbook_lookup.get(default_pb_id)
            if default_pb is not None:
                rows.append(
                    _policy_row(
                        config=config,
                        derivation_run_id=derivation_run_id,
                        archetype_row=archetype_row,
                        playbook=default_pb,
                        rule=rule,
                        mapping=mapping,
                        playbook_id=default_pb_id,
                        fit_score=0.0,
                        fit_tier=FIT_TIER_CATCH_ALL,
                        rationale=(
                            f"[{FIT_TIER_CATCH_ALL}] no playbook met fit thresholds "
                            f"(auto>={thresholds.auto:.2f}, review>={thresholds.review:.2f}) "
                            f"for archetype {archetype_row.get('archetype_id')}; "
                            f"routing to default playbook '{default_pb_id}' so the archetype "
                            "is served, but this needs manual review."
                        ),
                    )
                )
    return rows


def _policy_row(
    config: DerivationConfig,
    derivation_run_id: str,
    archetype_row: Dict[str, Any],
    playbook: Dict[str, Any],
    rule: "ExtractedRule",
    mapping: ArchetypeMapping,
    playbook_id: str,
    fit_score: float,
    fit_tier: str,
    rationale: str,
) -> Dict[str, Any]:
    policy_id = _policy_id(
        model_version=config.model_version,
        playbook_id=playbook_id,
        archetype_version=archetype_row["archetype_version"],
    )
    return {
        "eligibility_policy_id": policy_id,
        "version": archetype_row["archetype_version"],
        "playbook_id": playbook_id,
        "playbook_version": str(playbook.get("version", "1.0.0")),
        "model_name": config.model_name,
        "model_version": config.model_version,
        "archetype_ids": [archetype_row["archetype_version"]],
        "derivation_run_id": derivation_run_id,
        "derivation_method": mapping.derivation_method,
        "eligibility_rules": json.dumps(rule.predicate_json),
        "eligibility_rules_sql": rule.predicate_sql,
        "eligibility_rules_prose": _render_predicate_prose_safely(
            rule.predicate_json, namespace=config.namespace,
        ),
        "requires_features": list(rule.used_features),
        "expected_uplift_pct": _safe_float(playbook.get("expected_uplift_pct_default")),
        "fit_score": fit_score,
        "fit_tier": fit_tier,
        "rationale": rationale,
        "llm_model_id": mapping.llm_model_id,
        "status": "pending_review",
        "approved_by": None,
        "approved_at": None,
        "valid_from": None,
        "valid_to": None,
    }


def _render_predicate_prose_safely(
    predicate_json: Dict[str, Any],
    *,
    namespace: Optional[Any] = None,
) -> Optional[str]:
    """Best-effort prose render — any failure degrades to None (dashboard falls back to SQL).

    Uses ``discover_interpretation_sidecars`` so silent failures (missing
    namespace, empty sidecars) become loud WARNING-level log lines instead
    of NULL columns in ``eligibility_policy.eligibility_rules_prose``.
    Cycle 013 D3 surfaced exactly that mode — the consumer was returning
    None without telling anyone why.
    """
    try:
        from customer_retention.stages.causal.interpretation import (
            compile_predicate_prose,
            discover_interpretation_sidecars,
        )

        bundle = discover_interpretation_sidecars(namespace=namespace)
        bundle.emit_warnings(logger_=logger)
        if bundle.namespace is None:
            return None
        return compile_predicate_prose(
            predicate_json,
            feature_meta=bundle.feature_meta,
            population_stats=bundle.population_stats,
            column_descriptions=bundle.column_descriptions,
        )
    except Exception:  # noqa: BLE001 — best-effort; None is acceptable
        return None


def _validate_policy_coverage(
    archetype_rows: List[Dict[str, Any]],
    policy_rows: List[Dict[str, Any]],
    mappings: Sequence[ArchetypeMapping],
    config: DerivationConfig,
) -> None:
    """Fail fast when archetypes were produced but no policy rows were mapped.

    Two distinct silent-failure modes land here:

    A. ``mapped_any == False`` — every archetype's ``fit_decisions`` list is
       empty. This usually means the namer (LLM or prose-overlap fallback)
       returned no decisions, often because no playbooks were loaded.
    B. ``mapped_any == True`` — the namer DID emit decisions, but
       ``_build_policy_rows`` classified all of them as ``manual``
       (fit_score < ``FitThresholds.review``) and dropped them. Without a
       ``default_playbook_id`` no catch-all rows are emitted either, so the
       eligibility_policy table ends up empty.

    Surface the gap here so operators see the root cause at derivation
    time instead of three steps later in c05's snapshot writer.
    """
    if not archetype_rows or policy_rows:
        return
    mapped_any = any(mapping.fit_decisions for mapping in mappings)
    playbook_count = len(config.playbooks)
    feature_count = len(config.gold_feature_names)
    if not mapped_any:
        reason = (
            "no playbook mapped to any archetype — every archetype "
            "produced an empty fit_decisions list. Likely cause: "
            f"{playbook_count} playbook(s) loaded but the namer returned "
            "no decisions (LLM endpoint unreachable + no prose-overlap "
            "fallback configured, or no playbooks at all). Verify that "
            "c01_publish_definitions wrote rows to playbook_catalog and "
            "that LLM_ENDPOINT_NAME is reachable, or set it to '' to use "
            "the deterministic prose_overlap matcher."
        )
    else:
        thresholds = config.fit_thresholds
        reason = (
            "every mapping produced fit_decisions but every fit_score "
            f"fell below FIT_REVIEW_THRESHOLD ({thresholds.review:.2f}), "
            "so _build_policy_rows classified them all as 'manual' and "
            "dropped them. With DEFAULT_PLAYBOOK_ID unset no catch-all "
            "rows are emitted either. Two unblocks (either is fine):\n"
            "  (1) set DEFAULT_PLAYBOOK_ID to one of your playbook IDs in "
            "      the c02 config cell — every archetype then gets a "
            "      catch_all row pointing at that playbook for manual "
            "      review in c03.\n"
            "  (2) lower FIT_REVIEW_THRESHOLD (e.g. 0.05) so weak prose "
            "      overlaps land as 'review' rows instead of being "
            "      dropped as 'manual'.\n"
            "Long-term: tighten playbook prose so 'description' / "
            "'when_applicable' references the business concepts named "
            "by your top SHAP drivers — the prose_overlap matcher scores "
            "by token overlap with the archetype's driver feature names."
        )
    raise RuntimeError(
        f"Derivation produced {len(archetype_rows)} archetype rows but 0 "
        f"eligibility_policy rows ({playbook_count} playbooks, "
        f"{feature_count} gold features). {reason}"
    )


def _write_rows(
    config: DerivationConfig,
    archetype_rows: List[Dict[str, Any]],
    policy_rows: List[Dict[str, Any]],
) -> None:
    from .delta_writer import merge_into

    if not config.archetype_catalog_fqn:
        raise ValueError("DerivationConfig.archetype_catalog_fqn is required when write=True")
    if not config.eligibility_policy_fqn:
        raise ValueError("DerivationConfig.eligibility_policy_fqn is required when write=True")

    merge_into(
        spark=config.spark,
        rows=archetype_rows,
        schema=archetype_catalog_schema(),
        table_fqn=config.archetype_catalog_fqn,
        merge_keys=["archetype_id", "archetype_version"],
    )
    merge_into(
        spark=config.spark,
        rows=policy_rows,
        schema=eligibility_policy_schema(),
        table_fqn=config.eligibility_policy_fqn,
        merge_keys=["eligibility_policy_id", "version"],
    )


# ---------------------------------------------------------------------------
# Internals — small pure helpers
# ---------------------------------------------------------------------------


def _archetype_id(model_name: str, model_version: str, cluster_index: int) -> str:
    payload = f"{model_name}|{model_version}|{cluster_index}".encode()
    digest = hashlib.sha1(payload).hexdigest()[:8]
    return f"arch_{cluster_index}_{digest}"


def _archetype_version(derivation_run_id: str, cluster_index: int) -> str:
    payload = f"{derivation_run_id}|{cluster_index}".encode()
    digest = hashlib.sha1(payload).hexdigest()[:8]
    return f"v_{digest}"


def _policy_id(model_version: str, playbook_id: str, archetype_version: str) -> str:
    payload = f"{model_version}|{playbook_id}|{archetype_version}".encode()
    digest = hashlib.sha1(payload).hexdigest()[:12]
    return f"pol_{digest}"


def _make_run_id(config: DerivationConfig) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = hashlib.sha1(f"{config.model_name}|{config.model_version}".encode()).hexdigest()[:6]
    return f"deriv_{timestamp}_{suffix}"


def _struct_feature_thresholds(rule: Optional[ExtractedRule]) -> Dict[str, Dict[str, float]]:
    if rule is None:
        return {}
    return {name: dict(values) for name, values in rule.feature_thresholds.items()}


def _top_shap_struct(summary: ArchetypeSummary) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in summary.top_positive_drivers:
        out.append(
            {
                "feature": d["feature"],
                "mean_shap": float(d.get("mean_shap", 0.0)),
                "mean_value": float(d.get("mean_value", 0.0)),
                "direction": "positive",
            }
        )
    for d in summary.top_negative_drivers:
        out.append(
            {
                "feature": d["feature"],
                "mean_shap": float(d.get("mean_shap", 0.0)),
                "mean_value": float(d.get("mean_value", 0.0)),
                "direction": "negative",
            }
        )
    return out


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stability_score(new_centroid: Sequence[float], prior_centroid: Sequence[float]) -> float:
    """Convenience re-export so tests for the orchestrator don't need to import approval_gate."""
    return cosine_similarity(new_centroid, prior_centroid)
