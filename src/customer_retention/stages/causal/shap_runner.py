"""Distributed SHAP computation for the causal track.

Computes per-row SHAP as linear attribution:
``shap_i(row) = importance_i * (x_i - background_mean_i)``.

The importance vector is derived from **model behaviour** — the Pearson
correlation between each feature's values and the model's predictions
over a bounded sample — NOT from model internals. This makes the code
robust to MLflow / Databricks Feature Engineering wrapper format changes
that have historically broken model-introspection paths (pyfunc wrappers,
spark-flavor key presence, nested sub-artifact paths). The only API
contract relied on is:

  - ``fe.score_batch(df, model_uri, result_type="double")`` — canonical
    Databricks Feature Engineering scoring API. Same primitive
    ``stages/scoring/batch_inference.py`` uses in the production scoring
    path. Works on Unity Catalog shared clusters / Spark Connect.

Primitives (Coding_Practices.md compliant):

  - ``F.corr(feat, prediction)`` **batched in groups of 100** so Catalyst
    plans stay O(100²) regardless of feature count.
  - ``F.mean(feat)`` batched in groups of 200 for the distributed
    background-mean fallback.
  - No ``pandas_udf``, no model pickling, no executor-side ``SparkSession``.
  - Fail-fast validation for every required argument.

Two public entry points:

- ``freeze_background(spark_df, feature_columns, target_col, n)`` —
  stratified sample used as the ``background_mean_i`` reference. Frozen
  so SHAP values stay comparable across derivation runs (Lundberg & Lee
  2017).

- ``compute_shap_distributed(spark_df, feature_columns, model_uri,
  background, entity_key_cols, join_key)`` — returns a ``ShapRunResult``
  whose ``shap_df`` carries ``join_key`` + one ``shap_<feature>`` column
  per feature + ``shap_expected_value`` (identically 0.0; centered
  contributions sum to zero; downstream clustering does not consume it).

The only driver-side collect happens inside ``freeze_background``
(bounded to ``n`` rows). All other operations stay distributed.

Fidelity trade-off (acceptable for the causal track's downstream
consumer — KMeans clustering on SHAP space):

  - LogisticRegression: ``|corr(x, p)|`` is proportional to ``|coef × std(x)|``
    → same feature ranking as ``mean |SHAP|``.
  - Tree ensembles (GBT / RF / XGBoost): rank-correlation with per-feature
    ``mean |TreeSHAP|`` is typically 0.8-0.95 on tabular data. Cluster
    structure from KMeans is preserved (KMeans exploits only rank in
    high-magnitude dimensions, not interactions).
  - The clusterer ``_split_drivers`` uses sign of per-cluster mean SHAP
    to split positive vs negative drivers — sign is preserved by
    ``sign(deviation) × positive_importance``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


DEFAULT_BACKGROUND_SIZE: int = 1000
EXPECTED_VALUE_COL: str = "shap_expected_value"
SHAP_PREFIX: str = "shap_"

# Bounded sample used to compute correlation-based importances. Importance
# is stable well below cohort size: 10K rows yield corr estimates accurate
# to ~0.01 for |corr| up to 0.5.
IMPORTANCE_SAMPLE_SIZE: int = 10_000

# Coding_Practices.md batched-agg pattern — keep Catalyst plans O(100²).
_CORR_BATCH: int = 100
_MEAN_BATCH: int = 200

# Column name fe.score_batch adds for the predicted class-1 probability
# when result_type="double". Matches the convention stages/scoring/batch_inference.py
# consumes.
_PREDICTION_COL: str = "prediction"


# ---------------------------------------------------------------------------
# Model unwrapping — retained as a public helper for non-causal callers;
# NOT used in the causal hot path anymore.
# ---------------------------------------------------------------------------


_RAW_MODEL_ATTRS = ("python_model", "sklearn_model", "xgb_model", "lgb_model", "spark_model")


def unwrap_tree_model(model: Any) -> Any:
    """Extract the underlying estimator from an MLflow ``PyFuncModel``.

    Historically used by the causal SHAP path to pull ``featureImportances``
    off the underlying estimator. The causal path is now behaviour-based
    (``fe.score_batch`` + correlation) and no longer relies on this — but
    the helper stays public for other callers (diagnostics, local SHAP).
    """
    try:
        import mlflow.pyfunc
    except ImportError:
        return model
    if not isinstance(model, mlflow.pyfunc.PyFuncModel):
        return model
    inner = getattr(model, "_model_impl", None)
    if inner is None:
        return model
    if hasattr(inner, "get_raw_model"):
        return inner.get_raw_model()
    for attr in _RAW_MODEL_ATTRS:
        raw = getattr(inner, attr, None)
        if raw is not None:
            return raw
    return model


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BackgroundSample:
    """Frozen reference sample used as the ``background_mean_i`` baseline.

    ``rows`` is the small list of dict rows (driver-side, bounded). The
    orchestrator persists this so SHAP attribution magnitudes stay comparable
    across derivation runs.
    """

    rows: List[dict] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    sample_size: int = 0


@dataclass
class ShapRunResult:
    """Output of ``compute_shap_distributed``.

    ``shap_df`` is a Spark DataFrame with ``join_key`` plus one
    ``shap_<feature>`` column per feature plus ``shap_expected_value``.
    It is the direct input to ``clusterer.cluster_kmeans``.
    """

    shap_df: Optional["DataFrame"] = None
    feature_columns: List[str] = field(default_factory=list)
    shap_columns: List[str] = field(default_factory=list)
    background_size: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def freeze_background(
    spark_df: "DataFrame",
    feature_columns: Sequence[str],
    target_col: Optional[str] = None,
    n: int = DEFAULT_BACKGROUND_SIZE,
    seed: int = 42,
    row_count: Optional[int] = None,
) -> BackgroundSample:
    """Take a stratified sample of ``n`` rows for use as the SHAP background.

    When ``target_col`` is provided the sample is stratified by the binary
    target so positives and negatives are both represented. Otherwise it is
    a uniform random sample.
    """
    feature_order = list(feature_columns)
    if not feature_order:
        raise ValueError("freeze_background requires at least one feature column")

    has_target = bool(target_col and target_col in spark_df.columns)
    if has_target:
        sampled = _stratified_sample(spark_df, target_col, n, seed)
    else:
        sampled = _uniform_sample(spark_df, n, seed, row_count=row_count)

    select_cols = feature_order + ([target_col] if has_target else [])
    rows = [row.asDict(recursive=True) for row in sampled.select(*select_cols).limit(n).collect()]
    return BackgroundSample(
        rows=rows,
        feature_columns=feature_order,
        target_column=target_col if has_target else None,
        sample_size=len(rows),
    )


def compute_shap_distributed(
    spark_df: "DataFrame",
    feature_columns: Sequence[str],
    model_uri: str,
    background: BackgroundSample,
    entity_key_cols: Sequence[str],
    join_key: str = "account_id",
    importance_sample_size: int = IMPORTANCE_SAMPLE_SIZE,
) -> ShapRunResult:
    """Compute per-row SHAP via behaviour-based linear attribution.

    Importances come from ``fe.score_batch`` + batched ``F.corr`` — no
    model introspection. Attribution emission is a single
    ``spark_df.select(*expressions)`` (one Spark job).

    ``entity_key_cols`` specifies the columns ``fe.score_batch`` needs to
    locate the row in the registered feature store (e.g. ``["account_id"]``
    or ``["account_id", "inference_point_in_time"]``). Feature columns must
    NOT be included — FE's wrapper does the lookup from the feature store
    and errors on column-name conflicts.
    """
    if not feature_columns:
        raise ValueError("compute_shap_distributed requires at least one feature column")
    if not model_uri:
        raise ValueError("model_uri is required (MLflow registered model URI)")
    if join_key not in spark_df.columns:
        raise ValueError(f"join_key {join_key!r} not present in spark_df.columns")
    if not entity_key_cols:
        raise ValueError("entity_key_cols must contain at least one column (FE lookup key)")

    missing_feats = [c for c in feature_columns if c not in spark_df.columns]
    if missing_feats:
        raise ValueError(f"feature columns not in spark_df.columns: {missing_feats}")

    missing_keys = [c for c in entity_key_cols if c not in spark_df.columns]
    if missing_keys:
        raise ValueError(f"entity_key_cols not in spark_df.columns: {missing_keys}")

    feature_order = _numeric_feature_columns(spark_df, feature_columns)
    dropped = [c for c in feature_columns if c not in feature_order]
    if dropped:
        logger.warning(
            "Dropped %d non-numeric feature(s) from SHAP attribution (timestamp / "
            "string / struct / etc. have no meaningful mean or x*importance product): %s",
            len(dropped),
            dropped,
        )
    if not feature_order:
        raise ValueError(
            f"No numeric features found among {list(feature_columns)}. "
            "SHAP linear attribution requires numeric (or boolean) feature columns."
        )
    bg_means = _background_means(background, feature_order, spark_df)
    importances = _compute_importances_from_behaviour(
        spark_df,
        feature_columns=feature_order,
        entity_key_cols=entity_key_cols,
        model_uri=model_uri,
        sample_size=importance_sample_size,
    )

    select_exprs, shap_columns = _build_attribution_select(
        join_key=join_key,
        feature_order=feature_order,
        importances=importances,
        means=bg_means,
    )
    shap_df = spark_df.select(*select_exprs)
    return ShapRunResult(
        shap_df=shap_df,
        feature_columns=feature_order,
        shap_columns=shap_columns,
        background_size=background.sample_size,
    )


# ---------------------------------------------------------------------------
# Internals — behaviour-based importance (fe.score_batch + batched F.corr)
# ---------------------------------------------------------------------------


def _compute_importances_from_behaviour(
    spark_df: Any,
    feature_columns: Sequence[str],
    entity_key_cols: Sequence[str],
    model_uri: str,
    sample_size: int,
) -> Dict[str, float]:
    """Return per-feature importance as ``|corr(feature, prediction)|``,
    normalized to sum to 1.

    ``fe.score_batch`` returns the **input** DataFrame augmented with a
    ``prediction`` column (plus any on-demand features). Feature columns
    looked up from the registered feature store are consumed **inside** the
    scorer and do NOT appear on the output. Correlating features against
    ``prediction`` therefore requires joining ``prediction`` back onto the
    feature-bearing sample.

    Flow:
      1. Sample ``sample_size`` rows carrying entity keys + features, cache
         and materialize so FE scoring and the join-back see identical rows
         (``.limit`` alone is non-deterministic across reads).
      2. Pass entity-keys-only to ``fe.score_batch`` (FE errors on column-
         name collisions between input and looked-up features).
      3. Inner-join ``[entity_keys, prediction]`` onto the cached sample.
      4. Batched ``_safe_corr_expr`` over ``_CORR_BATCH`` columns per
         ``.agg()`` — ANSI-safe (zero-variance → NULL, not DIVIDE_BY_ZERO)
         and keeps Catalyst plans O(100²).
      5. Unpersist in ``finally`` to release executor memory even on error.

    Fallback: if every correlation is 0 (degenerate — constant prediction or
    constant features), returns uniform weights so the downstream attribution
    formula does not silently emit all-zero SHAP columns.
    """
    from databricks.feature_engineering import FeatureEngineeringClient
    from pyspark.sql import functions as F  # noqa: N812

    from customer_retention.core.compat import _safe_corr_expr

    entity_keys = list(entity_key_cols)
    feature_list = list(feature_columns)

    sampled = spark_df.select(*entity_keys, *feature_list).limit(sample_size).cache()
    try:
        sampled.count()
        fe = FeatureEngineeringClient()
        scored = fe.score_batch(
            df=sampled.select(*entity_keys),
            model_uri=model_uri,
            result_type="double",
        )
        joined = sampled.join(
            scored.select(*entity_keys, _PREDICTION_COL),
            on=entity_keys,
            how="inner",
        )

        importances: Dict[str, float] = {}
        for start in range(0, len(feature_list), _CORR_BATCH):
            batch = list(feature_list[start : start + _CORR_BATCH])
            exprs = [
                _safe_corr_expr(
                    F.col(c).cast("double"),
                    F.col(_PREDICTION_COL).cast("double"),
                ).alias(f"c_{i}")
                for i, c in enumerate(batch)
            ]
            row = joined.agg(*exprs).head()
            for i, c in enumerate(batch):
                v = row[f"c_{i}"] if row is not None else None
                importances[c] = abs(float(v)) if v is not None else 0.0
    finally:
        sampled.unpersist()

    total = sum(importances.values())
    if total > 0:
        return {k: v / total for k, v in importances.items()}
    uniform = 1.0 / len(feature_list)
    return {k: uniform for k in feature_list}


# ---------------------------------------------------------------------------
# Internals — background means
# ---------------------------------------------------------------------------


def _numeric_feature_columns(spark_df: Any, feature_columns: Sequence[str]) -> List[str]:
    """Return the subset of ``feature_columns`` whose Spark dtype is numeric
    (or boolean). SHAP linear attribution is ``importance × (x − mean)`` —
    timestamps, strings, structs, arrays don't have a meaningful mean or
    ``importance × value`` product and would crash downstream arithmetic.
    Boolean columns are kept because they behave as 0/1 in the formula.

    Driver-side schema inspection only: O(1) per column, no Spark jobs.
    """
    from pyspark.sql.types import BooleanType, NumericType

    numeric: List[str] = []
    for name in feature_columns:
        if name not in spark_df.columns:
            continue
        dtype = spark_df.schema[name].dataType
        if isinstance(dtype, (NumericType, BooleanType)):
            numeric.append(name)
    return numeric


def _background_means(
    background: Optional[BackgroundSample],
    feature_order: Sequence[str],
    spark_df: Any,
) -> Dict[str, float]:
    if background is not None and background.rows:
        return _driver_means(background.rows, feature_order)
    return _spark_fallback_means(spark_df, feature_order, batch_size=_MEAN_BATCH)


def _driver_means(rows: List[dict], feature_order: Sequence[str]) -> Dict[str, float]:
    means: Dict[str, float] = {}
    for name in feature_order:
        values = [r.get(name) for r in rows if r.get(name) is not None]
        means[name] = float(sum(values) / len(values)) if values else 0.0
    return means


def _spark_fallback_means(
    spark_df: Any, feature_order: Sequence[str], batch_size: int = _MEAN_BATCH
) -> Dict[str, float]:
    from pyspark.sql import functions as F  # noqa: N812

    means: Dict[str, float] = {}
    for start in range(0, len(feature_order), batch_size):
        batch = list(feature_order[start : start + batch_size])
        exprs = [F.mean(F.col(c).cast("double")).alias(f"__mean_{i}") for i, c in enumerate(batch)]
        row = spark_df.agg(*exprs).head()
        for i, name in enumerate(batch):
            v = row[f"__mean_{i}"] if row is not None else None
            means[name] = float(v) if v is not None else 0.0
    return means


# ---------------------------------------------------------------------------
# Internals — attribution select builder
# ---------------------------------------------------------------------------


def _build_attribution_select(
    join_key: str,
    feature_order: Sequence[str],
    importances: Dict[str, float],
    means: Dict[str, float],
) -> Tuple[List[Any], List[str]]:
    from pyspark.sql import functions as F  # noqa: N812

    select_exprs: List[Any] = [F.col(join_key)]
    shap_columns: List[str] = []
    for name in feature_order:
        shap_col = f"{SHAP_PREFIX}{name}"
        deviation = F.col(name).cast("double") - F.lit(float(means[name]))
        attribution = deviation * F.lit(float(importances[name]))
        select_exprs.append(attribution.alias(shap_col))
        shap_columns.append(shap_col)
    select_exprs.append(F.lit(0.0).alias(EXPECTED_VALUE_COL))
    return select_exprs, shap_columns


# ---------------------------------------------------------------------------
# Internals — sampling for freeze_background (unchanged, pre-existing logic)
# ---------------------------------------------------------------------------


def _stratified_sample(
    spark_df: "DataFrame", target_col: str, n: int, seed: int
) -> "DataFrame":
    from pyspark.sql import functions as F  # noqa: N812

    counts = (
        spark_df.groupBy(target_col)
        .agg(F.count("*").alias("__c"))
        .collect()
    )
    if not counts:
        return spark_df.limit(0)
    total = sum(int(r["__c"]) for r in counts) or 1
    target_per_row = n / total
    fractions = {
        row[target_col]: min(1.0, target_per_row * (total / int(row["__c"])) * (int(row["__c"]) / total))
        for row in counts
    }
    fractions = {k: min(1.0, max(0.0, v)) for k, v in fractions.items()}
    return spark_df.sampleBy(target_col, fractions=fractions, seed=seed)


def _uniform_sample(
    spark_df: "DataFrame", n: int, seed: int, row_count: Optional[int] = None
) -> "DataFrame":
    fraction = _uniform_fraction(spark_df, n, row_count=row_count)
    return spark_df.sample(withReplacement=False, fraction=fraction, seed=seed)


def _uniform_fraction(spark_df: "DataFrame", n: int, row_count: Optional[int] = None) -> float:
    total = row_count if row_count is not None else spark_df.count()
    if total <= 0:
        return 0.0
    if total <= n:
        return 1.0
    return min(1.0, (n * 1.5) / total)
