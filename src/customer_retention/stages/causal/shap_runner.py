"""Linear SHAP attribution over raw feature rows using a pre-computed
``ShapAttribution`` artifact.

``shap_i(row) = importance_i * (x_i - background_mean_i)``

Both ``importance_i`` and ``background_mean_i`` come from the training run
via ``stages.modeling.shap_attribution`` — frozen with the model artifact,
loaded once per derivation, and reused for every row. There is no
``fe.score_batch`` round-trip in this hot path: the attribution is a pure
function of ``(frozen model, training cohort)`` and recomputing it per
consumer wastes compute and is non-deterministic (``.limit(N)`` varies).

Emission is a single ``spark_df.select(*expressions)`` — one Spark job
over the caller's DataFrame, all parallelism used, no shuffle. The only
driver-side materialization is the small attribution dict already loaded
from the MLflow artifact.

Fidelity trade-off (acceptable for the causal track — KMeans over SHAP
space):

  - LogisticRegression: ``|corr(x, p)|`` is proportional to ``|coef × std(x)|``
    → same feature ranking as ``mean |SHAP|``.
  - Tree ensembles: rank-correlation with per-feature ``mean |TreeSHAP|``
    is ~0.8-0.95 on tabular data; cluster structure preserved.
  - ``_split_drivers`` uses sign of per-cluster mean SHAP to split
    positive vs negative drivers — preserved by
    ``sign(deviation) × positive_importance``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from customer_retention.stages.modeling.shap_attribution import ShapAttribution

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame


EXPECTED_VALUE_COL: str = "shap_expected_value"
SHAP_PREFIX: str = "shap_"


# ---------------------------------------------------------------------------
# Model unwrapping — retained as a public helper for diagnostics / local SHAP
# callers. NOT used in the causal hot path (attribution replaces model
# introspection).
# ---------------------------------------------------------------------------


_RAW_MODEL_ATTRS = ("python_model", "sklearn_model", "xgb_model", "lgb_model", "spark_model")


def unwrap_tree_model(model: Any) -> Any:
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
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class ShapRunResult:
    """Output of ``compute_shap_distributed``.

    ``shap_df`` carries ``join_key`` + one ``shap_<feature>`` column per
    attribution feature + ``shap_expected_value`` (identically 0.0 under
    the centered linear formulation).
    """

    shap_df: Optional["DataFrame"] = None
    feature_columns: List[str] = field(default_factory=list)
    shap_columns: List[str] = field(default_factory=list)
    background_size: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_shap_distributed(
    spark_df: "DataFrame",
    attribution: ShapAttribution,
    join_key: str = "account_id",
) -> ShapRunResult:
    """Emit per-row SHAP from a pre-computed attribution artifact.

    One ``spark_df.select(*expressions)`` — attribution dicts are broadcast
    as ``F.lit`` constants in the plan; Spark runs the multiplication and
    subtraction per row in parallel across all executor cores, no shuffle.

    Fail-fast validation — per Coding_Practices.md, no silent fallback:

      - ``attribution.feature_columns`` must be non-empty (the training
        template always produces at least one).
      - ``join_key`` must exist in ``spark_df``.
      - Every feature in ``attribution.feature_columns`` must exist in
        ``spark_df``. If the scoring-time schema has drifted from the
        training artifact, upstream (gold / feature store) is out of sync
        and refusing to emit SHAP is safer than fabricating it.
    """
    if not attribution.feature_columns:
        raise ValueError("attribution.feature_columns is empty")
    if join_key not in spark_df.columns:
        raise ValueError(f"join_key {join_key!r} not present in spark_df.columns")

    missing = [c for c in attribution.feature_columns if c not in spark_df.columns]
    if missing:
        raise ValueError(
            f"attribution features not present in spark_df.columns (training/scoring "
            f"schema drift): {missing[:10]} (+{max(len(missing) - 10, 0)} more)"
        )

    feature_order = list(attribution.feature_columns)
    select_exprs, shap_columns = _build_attribution_select(
        join_key=join_key,
        feature_order=feature_order,
        importances=attribution.importances,
        means=attribution.background_means,
    )
    shap_df = spark_df.select(*select_exprs)
    return ShapRunResult(
        shap_df=shap_df,
        feature_columns=feature_order,
        shap_columns=shap_columns,
        background_size=attribution.sample_size,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_attribution_select(
    join_key: str,
    feature_order: Sequence[str],
    importances: Dict[str, float],
    means: Dict[str, float],
) -> Tuple[List[Any], List[str]]:
    """Emit ``shap_<feature> = coalesce(nanvl(x - mean, 0), 0) * importance``.

    ``nanvl`` replaces NaN with 0; outer ``coalesce`` replaces NULL with 0.
    Lundberg & Lee linear-SHAP semantics: when ``x`` is unknown we assume it
    sits at the background mean (deviation 0 → contribution 0), which is
    safer than dropping the row (would shift the clustering cohort) or
    poisoning the SHAP vector with NaN (Spark-ML KMeans rejects NaN /
    Infinity in feature vectors)."""
    from pyspark.sql import functions as F  # noqa: N812

    select_exprs: List[Any] = [F.col(join_key)]
    shap_columns: List[str] = []
    for name in feature_order:
        shap_col = f"{SHAP_PREFIX}{name}"
        raw_deviation = F.col(name).cast("double") - F.lit(float(means.get(name, 0.0)))
        safe_deviation = F.coalesce(F.nanvl(raw_deviation, F.lit(0.0)), F.lit(0.0))
        attribution = safe_deviation * F.lit(float(importances.get(name, 0.0)))
        select_exprs.append(attribution.alias(shap_col))
        shap_columns.append(shap_col)
    select_exprs.append(F.lit(0.0).alias(EXPECTED_VALUE_COL))
    return select_exprs, shap_columns
