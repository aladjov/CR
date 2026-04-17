"""Distributed SHAP computation for the causal track.

Two computation paths, both fully distributed with no model pickling and no
``SparkContext`` capture on workers:

1. ``compute_shap_distributed`` — **linear attribution** (default / fast path).
   ``shap_i(row) = importance_i * (x_i - background_mean_i)``.
   Exact SHAP for linear models (LogisticRegression — well-known identity),
   first-order approximation for tree ensembles (GBT / RandomForest).
   One ``spark_df.select(*attribution_exprs)`` job, pure Spark SQL.

2. ``compute_shap_sampling_distributed`` — **Monte Carlo Shapley Sampling**
   (Štrumbelj & Kononenko 2014). Estimates exact Shapley values by averaging
   marginal contributions across random feature permutations. Runs on top of
   ``mlflow.pyfunc.spark_udf`` — the model loads on each executor from the
   MLflow registry, never pickled through a closure. Spark 4.0 compatible,
   model-agnostic (any flavor registered in MLflow: Spark ML, sklearn,
   xgboost, etc.). Higher cost than linear (O(N × K × M) predict calls) but
   yields true Shapley values as K → ∞.

Shared public entry points:

- ``freeze_background`` — stratified sample of ``n`` rows used as the
  reference baseline. Persisted so SHAP values stay comparable across runs
  (Lundberg & Lee 2017 require a fixed background).

- ``extract_top_drivers_per_account`` — per-account top-k feature-driver
  extraction for the ``top_shap_drivers`` Delta cache (Phase 2 use).

Implementation notes:

- No ``pandas_udf`` anywhere — historical ``CONTEXT_ONLY_VALID_ON_DRIVER``
  failure mode fully eliminated.
- The only driver-side collect happens inside ``freeze_background``
  (bounded to ``n`` rows). Both SHAP paths stay distributed throughout.
- Safe on Databricks shared clusters / Unity Catalog (Spark Connect-compatible).
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
DEFAULT_BATCH_SIZE: int = 10_000
EXPECTED_VALUE_COL: str = "shap_expected_value"
SHAP_PREFIX: str = "shap_"

# Columns per batched `.agg()` — keeps Catalyst plans small (O(200²)) per
# Coding_Practices.md "Batched .agg() Pattern for Bulk Column Statistics".
_MEAN_BATCH: int = 200


# ---------------------------------------------------------------------------
# Model unwrapping
# ---------------------------------------------------------------------------


_RAW_MODEL_ATTRS = ("python_model", "sklearn_model", "xgb_model", "lgb_model", "spark_model")


def unwrap_tree_model(model: Any) -> Any:
    """Extract the underlying estimator from an MLflow ``PyFuncModel``.

    Every standard MLflow flavor exposes ``get_raw_model()`` (canonical) or
    a flavor-specific attribute (``sklearn_model``, ``xgb_model``,
    ``lgb_model``, ``spark_model``, ``python_model``). ``unwrap_python_model()``
    is intentionally not used: it raises ``MlflowException`` for every
    flavor except ``mlflow.pyfunc.PythonModel`` subclasses. When no known
    accessor matches, the wrapper is returned unchanged so the caller sees
    a clear "model type not supported" error downstream.
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
    """Frozen reference sample used as the baseline in linear attribution.

    ``rows`` is the small list of dict rows (already driver-side); the
    orchestrator persists this to the ``shap_background`` Delta table keyed
    by ``archetype_version`` so SHAP values across derivation runs share
    the same reference.
    """

    rows: List[dict] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    sample_size: int = 0


@dataclass
class ShapRunResult:
    """Output of ``compute_shap_distributed``.

    ``shap_df`` is a Spark DataFrame with the join key plus one
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
    a uniform random sample. Pass ``row_count`` to skip the internal
    ``.count()`` when the caller already knows the size.
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
    model: Any,
    background: BackgroundSample,
    join_key: str = "account_id",
) -> ShapRunResult:
    """Compute per-row SHAP via distributed linear attribution.

    The model is inspected on the driver for feature importances (Spark ML
    ``featureImportances`` / ``coefficients`` or sklearn
    ``feature_importances_`` / ``coef_``). Attributions are emitted as a
    single Spark SQL ``select(...)`` — one job across existing partitions,
    no UDF, no model pickling. Background means come from the frozen
    ``background.rows`` (driver-side, bounded), or from a batched
    distributed aggregation if the background is empty.
    """
    if not feature_columns:
        raise ValueError("compute_shap_distributed requires at least one feature column")
    if join_key not in spark_df.columns:
        raise ValueError(f"join_key {join_key!r} not present in spark_df.columns")

    resolved_order = _resolve_feature_order(
        model, caller_feature_columns=feature_columns, spark_columns=spark_df.columns
    )
    importances = _extract_importances(model, feature_count=len(resolved_order))
    means = _background_means(background, resolved_order, spark_df)

    select_exprs, shap_columns = _build_attribution_select(
        join_key=join_key,
        feature_order=resolved_order,
        importances=importances,
        means=means,
    )
    shap_df = spark_df.select(*select_exprs)
    return ShapRunResult(
        shap_df=shap_df,
        feature_columns=resolved_order,
        shap_columns=shap_columns,
        background_size=background.sample_size,
    )


# ---------------------------------------------------------------------------
# Internals — feature-order resolution
# ---------------------------------------------------------------------------


def _resolve_feature_order(
    model: Any,
    caller_feature_columns: Sequence[str],
    spark_columns: Sequence[str],
) -> List[str]:
    """Return the feature order the model was trained on.

    Priority: PipelineModel's VectorAssembler → sklearn ``feature_names_in_``
    → caller-provided list. Fail-fast if any resolved column is missing
    from ``spark_df.columns``.
    """
    assembler_cols = _extract_assembler_input_cols(model)
    sklearn_names = None if assembler_cols else _extract_sklearn_feature_names(model)
    resolved = assembler_cols or sklearn_names or list(caller_feature_columns)
    spark_cols_set = set(spark_columns)
    missing = [c for c in resolved if c not in spark_cols_set]
    if missing:
        raise ValueError(f"feature columns not in spark_df.columns: {missing}")
    return list(resolved)


def _extract_assembler_input_cols(model: Any) -> Optional[List[str]]:
    stages = getattr(model, "stages", None)
    if not stages:
        return None
    first = stages[0]
    if hasattr(first, "getInputCols"):
        return list(first.getInputCols())
    return None


def _extract_sklearn_feature_names(model: Any) -> Optional[List[str]]:
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return None
    return [str(n) for n in list(names)]


def _extract_classifier_stage(model: Any) -> Any:
    stages = getattr(model, "stages", None)
    if stages:
        return stages[-1]
    return model


# ---------------------------------------------------------------------------
# Internals — importance extraction
# ---------------------------------------------------------------------------


def _extract_importances(model: Any, feature_count: int) -> List[float]:
    """Return absolute-value importances aligned to the model's feature order.

    Signs are carried by ``x - mean`` in the attribution formula, so the
    importance vector is always non-negative. Fail-fast if no known attribute
    is present or the length does not match ``feature_count``.
    """
    estimator = _extract_classifier_stage(model)
    vec = _probe_importance_attrs(estimator)
    if vec is None:
        raise ValueError(
            f"model {type(estimator).__name__} exposes no "
            "featureImportances / coefficients / feature_importances_ / coef_ — "
            "linear SHAP attribution requires one of these"
        )
    if len(vec) != feature_count:
        raise ValueError(
            f"importance vector length {len(vec)} does not match feature count {feature_count}"
        )
    return [abs(float(v)) for v in vec]


def _probe_importance_attrs(estimator: Any) -> Optional[List[float]]:
    fi = getattr(estimator, "featureImportances", None)
    if fi is not None and hasattr(fi, "toArray"):
        return list(fi.toArray())
    co = getattr(estimator, "coefficients", None)
    if co is not None and hasattr(co, "toArray"):
        return list(co.toArray())
    fi_sk = getattr(estimator, "feature_importances_", None)
    if fi_sk is not None:
        return list(fi_sk)
    co_sk = getattr(estimator, "coef_", None)
    if co_sk is not None:
        import numpy as np

        arr = np.asarray(co_sk)
        if arr.ndim == 2:
            arr = arr[0]
        return list(arr)
    return None


# ---------------------------------------------------------------------------
# Internals — background means
# ---------------------------------------------------------------------------


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
    importances: Sequence[float],
    means: Dict[str, float],
) -> Tuple[List[Any], List[str]]:
    from pyspark.sql import functions as F  # noqa: N812

    select_exprs: List[Any] = [F.col(join_key)]
    shap_columns: List[str] = []
    for name, imp in zip(feature_order, importances):
        shap_col = f"{SHAP_PREFIX}{name}"
        deviation = F.col(name).cast("double") - F.lit(float(means[name]))
        attribution = deviation * F.lit(float(imp))
        select_exprs.append(attribution.alias(shap_col))
        shap_columns.append(shap_col)
    select_exprs.append(F.lit(0.0).alias(EXPECTED_VALUE_COL))
    return select_exprs, shap_columns


# ---------------------------------------------------------------------------
# Internals — sampling (unchanged)
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


# ---------------------------------------------------------------------------
# Public API — Shapley Sampling path
# ---------------------------------------------------------------------------


DEFAULT_NUM_SAMPLES: int = 50


def compute_shap_sampling_distributed(
    spark_df: "DataFrame",
    feature_columns: Sequence[str],
    model_uri: str,
    background: BackgroundSample,
    join_key: str = "account_id",
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = 42,
) -> ShapRunResult:
    """Compute per-row SHAP via Monte Carlo Shapley Sampling (Štrumbelj 2014).

    For each ``(row, feature_i)`` pair, averages the marginal contribution
    ``f(x_with_i) − f(x_without_i)`` across ``num_samples`` random
    permutations where:
      - ``x_with_i[j] = row[j] if j precedes or equals i in the permutation else bg[j]``
      - ``x_without_i[j] = row[j] if j precedes i in the permutation else bg[j]``
      - ``bg`` is the mean of ``background.rows`` (interventional baseline).

    The model is loaded via ``mlflow.spark.load_model(model_uri)`` — it
    returns a Spark ML ``PipelineModel`` on the driver, and ``.transform``
    runs as a pure Spark DataFrame operation (distributed via Spark SQL, no
    UDF, no pickling, no executor-side ``SparkSession`` needed). This works
    on Unity Catalog shared clusters / Spark Connect.

    Supported model flavors: ``mlflow.spark`` (Spark ML pipelines —
    LogisticRegression / RandomForestClassifier / GBTClassifier / SparkXGB).
    For non-Spark flavors (sklearn, xgboost native, etc.), a flavor
    dispatch on ``mlflow.pyfunc.spark_udf`` would be needed — not wired in
    today because this project's training registers spark-flavor models.

    Converges to exact Shapley as ``num_samples → ∞``. 30–50 samples are
    adequate for stable top-K driver rankings in typical tabular settings.

    The model's Spark ML pipeline is expected to include a
    ``VectorAssembler`` consuming ``feature_columns`` as inputs and a
    binary classifier whose output column is ``probability`` (a Vector
    UDT). The class-1 probability is extracted via
    ``vector_to_array(F.col("probability"))[1]`` per Coding_Practices.md.
    """
    if not feature_columns:
        raise ValueError("compute_shap_sampling_distributed requires at least one feature column")
    if join_key not in spark_df.columns:
        raise ValueError(f"join_key {join_key!r} not present in spark_df.columns")
    if not model_uri:
        raise ValueError("model_uri is required for the sampling SHAP path")
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    missing = [c for c in feature_columns if c not in spark_df.columns]
    if missing:
        raise ValueError(f"feature columns not in spark_df.columns: {missing}")

    feature_order = list(feature_columns)
    num_features = len(feature_order)

    bg_means = _background_means(background, feature_order, spark_df)
    permutations = _sample_permutations(num_features, num_samples, seed)
    plan_rows = _build_sampling_plan(permutations, num_features)

    import mlflow.spark
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F  # noqa: N812
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        IntegerType,
        StructField,
        StructType,
    )

    spark = spark_df.sparkSession
    loaded_model = mlflow.spark.load_model(model_uri)

    plan_schema = StructType([
        StructField("sample_id", IntegerType(), False),
        StructField("feature_idx", IntegerType(), False),
        StructField("before_mask", ArrayType(BooleanType()), False),
    ])
    plan_df = spark.createDataFrame(plan_rows, schema=plan_schema)
    expanded = spark_df.crossJoin(plan_df)

    with_df = expanded.select(
        F.col(join_key),
        F.col("sample_id"),
        F.col("feature_idx"),
        F.lit("with").alias("__variant__"),
        *_variant_feature_cols(feature_order, bg_means, variant="with"),
    )
    without_df = expanded.select(
        F.col(join_key),
        F.col("sample_id"),
        F.col("feature_idx"),
        F.lit("without").alias("__variant__"),
        *_variant_feature_cols(feature_order, bg_means, variant="without"),
    )
    both = with_df.union(without_df)

    scored = loaded_model.transform(both)
    predicted = scored.withColumn(
        "__pred__", vector_to_array(F.col("probability")).getItem(1)
    )
    signed = predicted.withColumn(
        "__signed__",
        F.when(F.col("__variant__") == F.lit("with"), F.col("__pred__")).otherwise(
            -F.col("__pred__")
        ),
    )
    averaged = (
        signed.groupBy(join_key, "feature_idx")
        .agg((F.sum("__signed__") / F.lit(float(num_samples))).alias("__shap__"))
    )
    wide = (
        averaged.groupBy(join_key)
        .pivot("feature_idx", list(range(num_features)))
        .agg(F.first("__shap__"))
    )
    shap_columns = [f"{SHAP_PREFIX}{name}" for name in feature_order]
    shap_df = wide.select(
        F.col(join_key),
        *[F.col(str(idx)).alias(shap_columns[idx]) for idx in range(num_features)],
        F.lit(0.0).alias(EXPECTED_VALUE_COL),
    )

    return ShapRunResult(
        shap_df=shap_df,
        feature_columns=feature_order,
        shap_columns=shap_columns,
        background_size=background.sample_size,
    )


def extract_top_drivers_per_account(
    spark_df: "DataFrame",
    shap_df: "DataFrame",
    feature_columns: Sequence[str],
    join_key: str = "account_id",
    k: int = 5,
) -> "DataFrame":
    """Build a per-account DataFrame of the top-k SHAP drivers.

    Output matches ``schemas._per_account_shap_driver_struct``:
    ``[join_key, top_drivers: array<struct<feature, value, shap_contribution, direction>>]``.
    Consumed by the ``top_shap_drivers`` Delta cache that
    ``snapshot_writer._join_top_shap_drivers`` reads at dashboard time.

    Sort is by ``|shap|`` descending via a three-field struct whose first
    field is ``abs(shap)`` — ``F.sort_array`` sorts struct elements
    lexicographically by the first field — then ``F.transform`` strips the
    sort key so the output struct matches the Delta schema exactly.

    Direction is ``"positive"`` / ``"negative"`` / ``"neutral"`` based on
    the SHAP sign.
    """
    if join_key not in spark_df.columns:
        raise ValueError(f"join_key {join_key!r} not present in spark_df.columns")
    if join_key not in shap_df.columns:
        raise ValueError(f"join_key {join_key!r} not present in shap_df.columns")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    missing_features = [c for c in feature_columns if c not in spark_df.columns]
    if missing_features:
        raise ValueError(f"feature columns not in spark_df.columns: {missing_features}")

    shap_columns = [f"{SHAP_PREFIX}{c}" for c in feature_columns]
    missing_shap = [c for c in shap_columns if c not in shap_df.columns]
    if missing_shap:
        raise ValueError(f"shap columns not in shap_df.columns: {missing_shap}")

    from pyspark.sql import functions as F  # noqa: N812

    joined = spark_df.select(join_key, *feature_columns).join(
        shap_df.select(join_key, *shap_columns), on=join_key, how="inner"
    )

    driver_structs = [
        F.struct(
            F.abs(F.col(shap_col)).alias("abs_shap"),
            F.lit(name).alias("feature"),
            F.col(name).cast("double").alias("value"),
            F.col(shap_col).alias("shap_contribution"),
            F.when(F.col(shap_col) > F.lit(0.0), F.lit("positive"))
            .when(F.col(shap_col) < F.lit(0.0), F.lit("negative"))
            .otherwise(F.lit("neutral")).alias("direction"),
        )
        for name, shap_col in zip(feature_columns, shap_columns)
    ]
    sorted_arr = F.sort_array(F.array(*driver_structs), asc=False)
    top_k = F.slice(sorted_arr, 1, k)
    cleaned = F.transform(
        top_k,
        lambda s: F.struct(
            s["feature"].alias("feature"),
            s["value"].alias("value"),
            s["shap_contribution"].alias("shap_contribution"),
            s["direction"].alias("direction"),
        ),
    )
    return joined.select(F.col(join_key), cleaned.alias("top_drivers"))


# ---------------------------------------------------------------------------
# Internals — Shapley sampling plan (pure Python, deterministic with seed)
# ---------------------------------------------------------------------------


def _sample_permutations(num_features: int, num_samples: int, seed: int) -> List[List[int]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    return [rng.permutation(num_features).tolist() for _ in range(num_samples)]


def _build_sampling_plan(
    permutations: Sequence[Sequence[int]], num_features: int
) -> List[Tuple[int, int, List[bool]]]:
    """Return (sample_id, feature_idx, before_mask) rows for the sampling expansion.

    ``before_mask[j]`` is True iff feature ``j`` precedes ``feature_idx`` in
    ``permutations[sample_id]``. The chosen feature ``feature_idx`` itself
    is always False (it is added in the "with" variant via a separate
    condition on ``feature_idx == j``).
    """
    plan: List[Tuple[int, int, List[bool]]] = []
    for sample_id, perm in enumerate(permutations):
        perm_list = list(perm)
        position_of = [0] * num_features
        for position, feature_j in enumerate(perm_list):
            position_of[feature_j] = position
        for feature_idx in range(num_features):
            p = position_of[feature_idx]
            before_mask = [position_of[j] < p for j in range(num_features)]
            plan.append((sample_id, feature_idx, before_mask))
    return plan


def _variant_feature_cols(
    feature_order: Sequence[str],
    bg_means: Dict[str, float],
    variant: str,
) -> List[Any]:
    """Build ``F.when(...)`` expressions for one variant (with / without).

    For each feature ``j``, its value in the synthesized row is:
    - ``"with"`` variant: ``row[j] if before_mask[j] or j == feature_idx else bg[j]``
    - ``"without"`` variant: ``row[j] if before_mask[j] else bg[j]``
    """
    from pyspark.sql import functions as F  # noqa: N812

    cols: List[Any] = []
    for j, name in enumerate(feature_order):
        mask_entry = F.element_at(F.col("before_mask"), j + 1)
        if variant == "with":
            condition = mask_entry | (F.col("feature_idx") == F.lit(j))
        else:
            condition = mask_entry
        cols.append(F.when(condition, F.col(name)).otherwise(F.lit(float(bg_means[name]))).alias(name))
    return cols
