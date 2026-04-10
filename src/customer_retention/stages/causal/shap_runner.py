"""Distributed SHAP computation for the causal track.

Computes SHAP values across the entire training cohort by capturing the
loaded model and a frozen background sample in a ``pandas_udf`` closure,
then running one ``shap.TreeExplainer`` per Spark partition. The model,
feature order, and background frame ride along inside the closure (Spark
Connect serializes the closed-over variables automatically — no explicit
``sparkContext.broadcast``, which is banned on Databricks shared clusters /
Unity Catalog). This matches the Databricks "scaling SHAP" pattern: the
model is loaded once on the driver and each executor instantiates one
explainer per partition, in parallel — no driver-side collect of the full
population.

Two public entry points:

- ``freeze_background(spark_df, target_col, n)`` — stratified sample of
  ``n`` rows that is reused as the SHAP reference dataset. Frozen once
  per derivation run and persisted to ``shap_background`` so SHAP value
  comparisons across runs stay valid (Lundberg & Lee 2017). The frozen
  rows are passed straight into ``compute_shap_distributed`` and become
  the ``data=`` argument of ``shap.TreeExplainer`` on every executor.

- ``compute_shap_distributed(spark_df, model, feature_columns,
  background, batch_size)`` — returns a Spark DataFrame with the original
  columns plus one ``shap_<feature>`` column per feature, plus an
  ``expected_value`` column for the model's base value.

Stays distributed throughout. The only collect happens inside
``freeze_background``, which collects exactly ``n`` rows (default 1000) so
the size is bounded regardless of cohort size. ``compute_shap_distributed``
takes an optional ``row_count`` argument to short-circuit a redundant
``.count()`` when the caller already knows the size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

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


# ---------------------------------------------------------------------------
# Model unwrapping
# ---------------------------------------------------------------------------


def unwrap_tree_model(model: Any) -> Any:
    """Extract the tree-native estimator from an MLflow PyFuncModel.

    ``shap.TreeExplainer`` requires the raw sklearn / xgboost / lightgbm
    estimator, not the ``mlflow.pyfunc.PyFuncModel`` wrapper (which only
    exposes ``predict()``). This function probes the known internal
    attribute chains. If the model is already a raw estimator it is
    returned as-is.
    """
    try:
        import mlflow.pyfunc
        if isinstance(model, mlflow.pyfunc.PyFuncModel):
            inner = getattr(model, "_model_impl", None)
            if inner is not None:
                # sklearn / xgboost / lightgbm flavors store the raw model
                # on the ``python_model`` attribute of the implementation.
                py_model = getattr(inner, "python_model", None)
                if py_model is not None:
                    return py_model
            # mlflow >= 2.10 stores it under .unwrap_python_model()
            if hasattr(model, "unwrap_python_model"):
                return model.unwrap_python_model()
    except (ImportError, AttributeError):
        pass
    return model


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BackgroundSample:
    """Frozen SHAP background sample.

    ``rows`` is the small list of dict rows; ``feature_columns`` lists the
    columns in stable order. The orchestrator persists this to the
    ``shap_background`` Delta table keyed by ``archetype_version`` so SHAP
    values across derivation runs share the same reference.
    """

    rows: List[dict] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    sample_size: int = 0


@dataclass
class ShapRunResult:
    """Output of ``compute_shap_distributed``.

    ``shap_df`` is a Spark DataFrame with one ``shap_<feature>`` column
    per feature plus the original join key (``account_id``). It is the
    direct input to the clusterer's ``cluster_kmeans``.
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
    target so positives and negatives are both represented. Otherwise it
    is a uniform random sample. The result is small enough to collect to
    the driver and broadcast to executors.

    Pass ``row_count`` to skip the internal ``.count()`` call when the
    caller already knows the size — saves one full Spark scan per
    derivation run.
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
    batch_size: int = DEFAULT_BATCH_SIZE,
    row_count: Optional[int] = None,
) -> ShapRunResult:
    """Compute per-row SHAP values via a partition-wise ``pandas_udf``.

    The model, feature order, and frozen background frame are captured in
    the UDF closure so Spark Connect serializes them automatically — no
    ``sparkContext.broadcast`` (banned on shared clusters). Each executor
    instantiates one ``shap.TreeExplainer(model, data=background_df)`` per
    partition (cached via a module-level ``_PARTITION_STATE`` dict so a
    single executor reuses the explainer across batches). For each batch
    the UDF returns the SHAP values as a struct of doubles, which is
    exploded into per-feature columns by the driver-side ``df.select(...)``.

    Pass ``row_count`` to skip the internal ``.count()`` used for partition
    sizing — saves one full Spark scan per derivation run.

    Returns a ``ShapRunResult`` whose ``shap_df`` carries the join key plus
    one ``shap_<feature>`` column per feature plus ``shap_expected_value``.
    """
    from pyspark.sql import functions as F  # noqa: N812
    from pyspark.sql.types import ArrayType, DoubleType, StructField, StructType

    feature_order = list(feature_columns)
    if not feature_order:
        raise ValueError("compute_shap_distributed requires at least one feature column")
    if join_key not in spark_df.columns:
        raise ValueError(f"join_key {join_key!r} not present in spark_df.columns")

    return_schema = StructType(
        [
            StructField("shap_values", ArrayType(DoubleType()), True),
            StructField("expected_value", DoubleType(), True),
        ]
    )

    explainer_udf = _build_shap_udf(
        return_schema=return_schema,
        model_wrapper=_PicklableModelWrapper(model, background),
        feature_order=feature_order,
    )

    rows_per_partition = max(batch_size, 1)
    repartitioned = spark_df.select(join_key, *feature_order)
    if rows_per_partition < DEFAULT_BATCH_SIZE * 8:
        repartitioned = repartitioned.repartition(
            _partition_count(spark_df, rows_per_partition, row_count=row_count)
        )

    intermediate = repartitioned.withColumn(
        "__shap_struct__",
        explainer_udf(F.struct(*[F.col(c) for c in feature_order])),
    )

    select_exprs = [F.col(join_key)]
    shap_columns: List[str] = []
    for idx, name in enumerate(feature_order):
        col_name = f"{SHAP_PREFIX}{name}"
        select_exprs.append(F.col("__shap_struct__.shap_values").getItem(idx).alias(col_name))
        shap_columns.append(col_name)
    select_exprs.append(F.col("__shap_struct__.expected_value").alias(EXPECTED_VALUE_COL))

    shap_df = intermediate.select(*select_exprs)
    return ShapRunResult(
        shap_df=shap_df,
        feature_columns=feature_order,
        shap_columns=shap_columns,
        background_size=background.sample_size,
    )


# ---------------------------------------------------------------------------
# Internals — sampling
# ---------------------------------------------------------------------------


def _stratified_sample(
    spark_df: "DataFrame", target_col: str, n: int, seed: int
) -> "DataFrame":
    """Per-stratum sample using ``DataFrame.sampleBy`` (Spark-native)."""
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
    fractions = {row[target_col]: min(1.0, target_per_row * (total / int(row["__c"]) ) * (int(row["__c"]) / total))
                 for row in counts}
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


def _partition_count(
    spark_df: "DataFrame", rows_per_partition: int, row_count: Optional[int] = None
) -> int:
    total = row_count if row_count is not None else spark_df.count()
    if total <= 0:
        return 1
    return max(1, int((total + rows_per_partition - 1) // rows_per_partition))


# ---------------------------------------------------------------------------
# Internals — pandas_udf body
# ---------------------------------------------------------------------------


class _PicklableModelWrapper:
    """Picklable holder for the model + background that the UDF closure captures.

    Spark Connect serializes UDF closures through pickle on the driver and
    unpickles them on each executor. Most sklearn / xgboost / lightgbm
    models are picklable; this wrapper bundles the model with the frozen
    SHAP background rows so each executor builds its TreeExplainer with the
    same reference dataset (Lundberg & Lee 2017 — SHAP values are only
    comparable across runs when the background is fixed).

    The background frame is rebuilt from ``rows`` and ``feature_columns``
    inside ``explainer()`` so we don't ship a pyspark.pandas object to the
    executors — only plain Python lists / dicts.
    """

    def __init__(self, model: Any, background: Optional[BackgroundSample] = None) -> None:
        self.model = model
        self.background_rows: List[dict] = list(background.rows) if background else []
        self.background_feature_order: List[str] = (
            list(background.feature_columns) if background else []
        )

    def explainer(self) -> Any:
        import pandas as _pd
        import shap

        if self.background_rows and self.background_feature_order:
            background_df = _pd.DataFrame(
                self.background_rows, columns=self.background_feature_order
            )
            return shap.TreeExplainer(self.model, data=background_df)
        return shap.TreeExplainer(self.model)


def _build_shap_udf(
    return_schema: Any,
    model_wrapper: "_PicklableModelWrapper",
    feature_order: List[str],
) -> Any:
    """Construct the ``pandas_udf`` that runs SHAP per partition.

    The UDF takes a struct of feature columns and returns a struct of
    ``(shap_values, expected_value)`` per row. ``model_wrapper`` and
    ``feature_order`` are captured in the closure so Spark Connect
    serializes them with the UDF — no explicit ``sparkContext.broadcast``,
    which is banned on shared clusters. The explainer is built once per
    partition and cached in a module-level ``_PARTITION_STATE`` dict; calls
    inside the same partition reuse it.
    """
    from pyspark.sql.functions import pandas_udf

    captured_features = list(feature_order)

    @pandas_udf(return_schema)
    def shap_udf(features_struct):
        import pandas as _pd

        explainer = _PARTITION_STATE.get("explainer")
        if explainer is None:
            explainer = model_wrapper.explainer()
            _PARTITION_STATE["explainer"] = explainer
        frame = _pd.DataFrame(list(features_struct), columns=captured_features)
        shap_values = explainer.shap_values(frame)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        expected = explainer.expected_value
        if hasattr(expected, "__len__"):
            expected = float(expected[1] if len(expected) > 1 else expected[0])
        else:
            expected = float(expected)
        return _pd.DataFrame(
            {
                "shap_values": [list(map(float, row)) for row in shap_values],
                "expected_value": [expected] * len(shap_values),
            }
        )

    return shap_udf


_PARTITION_STATE: dict = {}
