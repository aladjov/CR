"""Distributed KMeans clustering of SHAP-space points.

The Phase 2 plan calls for ``pyspark.ml.clustering.KMeans`` with k-means‖
initialization and a silhouette sweep over ``k ∈ {4, ..., 8}``. KMeans is
the only fully-distributed Spark-native clustering option (HDBSCAN has no
Spark port and would force a driver-side collect of every SHAP vector,
which OOMs at our scale).

The clusterer runs on the SHAP-space DataFrame produced by ``shap_runner``
— one Spark row per training-cohort row, with the SHAP value of every
feature as a column. Before the silhouette sweep, ``select_top_shap_features``
ranks features by mean absolute SHAP and keeps only the top
``DEFAULT_FEATURE_CAP`` (50). This is mandatory on Spark Connect: KMeans
fits serialize the trained model, and on >100 columns the serialized model
can exceed the 1 GB ``MODEL_SIZE_OVERFLOW_EXCEPTION`` limit (same failure
mode documented in ``Coding_Practices.md`` for ``Imputer.fit``). Reducing
to ~50 features also makes archetypes human-interpretable instead of
sprawling across hundreds of dimensions.

``cluster_kmeans`` assembles the selected SHAP columns into a single
vector, runs the silhouette sweep, picks the best ``k``, and returns the
fitted model plus per-row cluster labels and per-cluster centroid
summaries. The cached ``assembled`` DataFrame is unpersisted in a
``finally`` block so the cache never leaks across runs.

A second function, ``cluster_centroids_raw``, computes per-cluster mean
vectors over the *raw* feature columns (joined back to the same rows by
``entity_id``). These raw centroids are what the snapshot writer uses at
scoring time — runtime archetype assignment is nearest-neighbor on raw
features, not SHAP values, because SHAP isn't available for new rows.

All Spark operations are batched. Per-cluster summaries use a single
``groupBy(cluster_col).agg(...)`` job over the SHAP-vector columns, not a
loop of one job per cluster. The silhouette sweep fits each candidate ``k``
in series but each fit is one Spark job, not N.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


DEFAULT_K_RANGE: Tuple[int, int] = (4, 12)
DEFAULT_K_CAP: int = 8
DEFAULT_SEED: int = 42
DEFAULT_MAX_ITER: int = 30
DEFAULT_FEATURE_CAP: int = 50
SHAP_VECTOR_COL: str = "__shap_features__"
CLUSTER_COL: str = "cluster_id"


@dataclass
class ClusterCandidate:
    """One ``k`` from the silhouette sweep."""

    k: int
    silhouette: float


@dataclass
class ClusteringResult:
    """Output of one silhouette-swept KMeans run.

    ``labelled_df`` is a Spark DataFrame with the original columns plus a
    ``cluster_id`` IntegerType column. ``centroids`` is the per-cluster
    SHAP-space mean vector — equivalent to ``model.clusterCenters()`` but
    surfaced as a list of lists so the orchestrator can serialize it
    without touching pyspark types.
    """

    best_k: int
    best_silhouette: float
    candidates: List[ClusterCandidate] = field(default_factory=list)
    centroids: List[List[float]] = field(default_factory=list)
    feature_order: List[str] = field(default_factory=list)
    labelled_df: Optional["DataFrame"] = None
    model: Any = None

    @property
    def cluster_count(self) -> int:
        return self.best_k


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cluster_kmeans(
    shap_df: "DataFrame",
    feature_columns: Sequence[str],
    k_range: Tuple[int, int] = DEFAULT_K_RANGE,
    k_cap: int = DEFAULT_K_CAP,
    seed: int = DEFAULT_SEED,
    max_iter: int = DEFAULT_MAX_ITER,
    feature_cap: int = DEFAULT_FEATURE_CAP,
) -> ClusteringResult:
    """Run silhouette-swept KMeans on a SHAP-space Spark DataFrame.

    When ``feature_columns`` exceeds ``feature_cap`` (default 50), the
    feature set is reduced via ``select_top_shap_features`` to the top
    cap by mean absolute SHAP value. This is mandatory on Spark Connect:
    KMeans serializes the trained model and >100 columns can exceed the
    1 GB ``MODEL_SIZE_OVERFLOW_EXCEPTION`` limit. The selection happens
    in one batched Spark job (``F.mean(F.abs(col))`` over all columns).

    The cached intermediate is released in a ``finally`` block so the
    Spark cache is never leaked across derivation runs.

    Returns a ``ClusteringResult`` with the best ``k``, the per-row
    cluster labels (as a Spark column on a derived DataFrame), and the
    SHAP-space centroids in the same column order as the *selected*
    feature subset (exposed as ``feature_order``).
    """
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import ClusteringEvaluator
    from pyspark.ml.feature import VectorAssembler

    if not feature_columns:
        raise ValueError("cluster_kmeans requires at least one feature column")

    feature_order = select_top_shap_features(shap_df, feature_columns, feature_cap)
    assembled = (
        VectorAssembler(
            inputCols=feature_order,
            outputCol=SHAP_VECTOR_COL,
            handleInvalid="keep",
        )
        .transform(shap_df)
    )
    assembled.cache()

    try:
        k_low, k_high = k_range
        k_low = max(2, int(k_low))
        k_high = min(int(k_high), int(k_cap))
        if k_high < k_low:
            k_high = k_low

        candidates: List[ClusterCandidate] = []
        best_model = None
        best_labelled = None
        best_score = float("-inf")
        best_k = k_low
        evaluator = ClusteringEvaluator(featuresCol=SHAP_VECTOR_COL, predictionCol=CLUSTER_COL)
        for k in range(k_low, k_high + 1):
            model = (
                KMeans(
                    featuresCol=SHAP_VECTOR_COL,
                    predictionCol=CLUSTER_COL,
                    k=k,
                    seed=seed,
                    maxIter=max_iter,
                )
                .fit(assembled)
            )
            labelled = model.transform(assembled)
            score = float(evaluator.evaluate(labelled))
            candidates.append(ClusterCandidate(k=k, silhouette=score))
            logger.info("KMeans k=%d silhouette=%.4f", k, score)
            if score > best_score:
                best_score = score
                best_k = k
                best_model = model
                best_labelled = labelled

        if best_model is None or best_labelled is None:
            raise RuntimeError("KMeans silhouette sweep produced no candidates — check k_range")

        labelled_df = best_labelled.drop(SHAP_VECTOR_COL)
        centroids = [[float(v) for v in c] for c in best_model.clusterCenters()]
        return ClusteringResult(
            best_k=best_k,
            best_silhouette=best_score,
            candidates=candidates,
            centroids=centroids,
            feature_order=feature_order,
            labelled_df=labelled_df,
            model=best_model,
        )
    finally:
        try:
            assembled.unpersist()
        except (AttributeError, RuntimeError) as exc:
            # AttributeError: unit-test mocks may not implement unpersist().
            # RuntimeError: Spark connection already closed by the time the
            # finally block runs (rare during interpreter shutdown).
            logger.debug("cluster_kmeans unpersist failed (non-fatal): %s", exc)


_BULK_AGG_BATCH: int = 200


def select_top_shap_features(
    shap_df: "DataFrame",
    feature_columns: Sequence[str],
    feature_cap: int = DEFAULT_FEATURE_CAP,
) -> List[str]:
    """Pick the top-N SHAP columns by mean absolute SHAP magnitude.

    Batches columns in groups of ``_BULK_AGG_BATCH`` so Catalyst plans stay
    O(200²) regardless of feature count (Coding_Practices.md bulk-agg
    pattern). Each column is wrapped in ``_nan_safe`` so stray NaN values
    don't poison the magnitude estimate. Below the cap the input list is
    returned unchanged so callers can pass any column count safely.
    """
    from pyspark.sql import functions as F  # noqa: N812

    feature_list = list(feature_columns)
    if len(feature_list) <= feature_cap:
        return feature_list
    magnitudes: List[Tuple[str, float]] = []
    for start in range(0, len(feature_list), _BULK_AGG_BATCH):
        batch = feature_list[start : start + _BULK_AGG_BATCH]
        agg_exprs = [
            F.mean(F.abs(_nan_safe(c))).alias(f"__mag_{i}") for i, c in enumerate(batch)
        ]
        row = shap_df.agg(*agg_exprs).head()
        if row is None:
            magnitudes.extend((c, 0.0) for c in batch)
            continue
        magnitudes.extend(
            (c, _safe_double(row[f"__mag_{i}"])) for i, c in enumerate(batch)
        )
    magnitudes.sort(key=lambda pair: -pair[1])
    selected = [name for name, _ in magnitudes[:feature_cap]]
    logger.info(
        "select_top_shap_features: kept %d of %d columns (cap=%d)",
        len(selected),
        len(feature_list),
        feature_cap,
    )
    return selected


def cluster_centroids_raw(
    raw_features_df: "DataFrame",
    feature_columns: Sequence[str],
    cluster_col: str = CLUSTER_COL,
) -> Tuple[List[List[float]], List[str]]:
    """Compute per-cluster mean vectors over the raw feature columns.

    The orchestrator joins the labelled SHAP DataFrame back onto the raw
    feature DataFrame on ``entity_id`` (or whatever the join key is) so
    the same row is represented in both spaces, then calls this helper.

    Single Spark job: one ``groupBy(cluster_col).agg(F.mean(...))`` over
    all raw feature columns. Each column is wrapped in ``_nan_safe(col)``
    so NaN rows are treated as NULL (and therefore skipped by ``F.mean``);
    Spark's native ``F.mean`` skips NULL but NOT NaN — a single NaN would
    otherwise poison the whole cluster mean for that feature and land NaN
    in the Delta ``archetype_catalog``.
    """
    from pyspark.sql import functions as F  # noqa: N812

    if not feature_columns:
        raise ValueError("cluster_centroids_raw requires at least one feature column")
    feature_order = list(feature_columns)
    agg_exprs = [F.mean(_nan_safe(c)).alias(f"__mean_{i}") for i, c in enumerate(feature_order)]
    rows = (
        raw_features_df.groupBy(cluster_col)
        .agg(*agg_exprs)
        .orderBy(cluster_col)
        .collect()
    )
    centroids: List[List[float]] = []
    for row in rows:
        centroids.append([_safe_double(row[f"__mean_{i}"]) for i in range(len(feature_order))])
    return centroids, feature_order


def cluster_size_stats(
    labelled_df: "DataFrame", cluster_col: str = CLUSTER_COL
) -> List[Tuple[int, int]]:
    """Return ``[(cluster_id, size), ...]`` from a single ``groupBy().count()``."""
    rows = labelled_df.groupBy(cluster_col).count().orderBy(cluster_col).collect()
    return [(int(row[cluster_col]), int(row["count"])) for row in rows]


def cluster_target_means(
    labelled_df: "DataFrame",
    target_col: str,
    cluster_col: str = CLUSTER_COL,
) -> List[Tuple[int, float]]:
    """Return ``[(cluster_id, mean_target), ...]`` for cluster mean churn probability."""
    from pyspark.sql import functions as F  # noqa: N812

    rows = (
        labelled_df.groupBy(cluster_col)
        .agg(F.mean(_nan_safe(target_col)).alias("__mean_target"))
        .orderBy(cluster_col)
        .collect()
    )
    return [(int(row[cluster_col]), _safe_double(row["__mean_target"])) for row in rows]


def cluster_shap_centroids(
    labelled_df: "DataFrame",
    shap_columns: Sequence[str],
    cluster_col: str = CLUSTER_COL,
) -> Tuple[List[List[float]], List[str]]:
    """Per-cluster mean SHAP-value vectors via a single batched ``groupBy().agg()``."""
    from pyspark.sql import functions as F  # noqa: N812

    feature_order = list(shap_columns)
    agg_exprs = [F.mean(_nan_safe(c)).alias(f"__shap_mean_{i}") for i, c in enumerate(feature_order)]
    rows = (
        labelled_df.groupBy(cluster_col)
        .agg(*agg_exprs)
        .orderBy(cluster_col)
        .collect()
    )
    centroids = [[_safe_double(row[f"__shap_mean_{i}"]) for i in range(len(feature_order))] for row in rows]
    return centroids, feature_order


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nan_safe(col_name: str) -> Any:
    """Cast a Spark column to ``double`` and coerce NaN to NULL.

    ``F.mean`` / ``F.stddev_pop`` skip NULL but not NaN, so without this
    wrapper a single NaN value poisons the entire aggregation result.
    Wrapping with ``F.nanvl(col, NULL)`` turns NaN into NULL so the
    aggregator treats it as missing — matching the semantic we want for
    per-cluster centroids and means."""
    from pyspark.sql import functions as F  # noqa: N812

    return F.nanvl(F.col(col_name).cast("double"), F.lit(None).cast("double"))


def _safe_double(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return 0.0
    import math

    return 0.0 if math.isnan(coerced) else coerced
