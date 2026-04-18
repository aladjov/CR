"""Training-time SHAP attribution artifact — importance vector + background
means persisted alongside the model so downstream SHAP consumers skip the
``fe.score_batch`` round-trip entirely.

The attribution is a pure function of ``(frozen model, training cohort)``.
Once training finishes it is invariant; recomputing per-consumer
(c02 derivation, drift reports, ad-hoc SHAP) wastes compute and is
non-deterministic because ``.limit(sample_size)`` varies between reads.

Persisted shape (MLflow artifact ``shap_attribution.json``):

- ``importances[feature]`` — ``|corr(feature, prediction)|`` normalized to
  sum to 1. Linear-attribution ranking proxy: identical to ``|coef × std|``
  for LogisticRegression, rank-correlated with ``mean |TreeSHAP|`` for
  tree ensembles (see Lundberg & Lee 2017).
- ``background_means[feature]`` — ``E[feature]`` over the same cached
  sample, used as the baseline in ``shap_i = importance_i * (x_i - mean_i)``.
- ``feature_columns`` — ordered list, canonical feature set for the
  logged model.
- ``sample_size`` — materialized row count for audit.

Both ``importances`` and ``background_means`` are computed via batched
``.agg()`` over a single cached, materialized sample of the training frame.
No driver collect of data rows; only small scalar aggregates via ``.head()``.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


DEFAULT_SAMPLE_SIZE: int = 10_000
ARTIFACT_FILENAME: str = "shap_attribution.json"
_PREDICTION_COL: str = "prediction"
_CORR_BATCH: int = 100
_MEAN_BATCH: int = 200


@dataclass
class ShapAttribution:
    """Frozen attribution payload. JSON round-trip via ``to_dict`` /
    ``from_dict`` so MLflow artifacts remain human-diffable."""

    importances: Dict[str, float] = field(default_factory=dict)
    background_means: Dict[str, float] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    sample_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "importances": dict(self.importances),
            "background_means": dict(self.background_means),
            "feature_columns": list(self.feature_columns),
            "sample_size": int(self.sample_size),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShapAttribution":
        return cls(
            importances=dict(data.get("importances", {})),
            background_means=dict(data.get("background_means", {})),
            feature_columns=list(data.get("feature_columns", [])),
            sample_size=int(data.get("sample_size", 0)),
        )


def compute_shap_attribution(
    pipeline_model: Any,
    df: Any,
    feature_columns: List[str],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> ShapAttribution:
    """Compute importances + background means on a single cached sample.

    Steps — one ``cache()`` + materialize, then two batched ``.agg()`` loops:

      1. Fill nulls on feature columns (matches training-time preprocessing
         so the assembler's ``handleInvalid="error"`` does not trip).
      2. ``pipeline_model.transform(sample).select(feature_cols, prediction)``
         — runs the logged pipeline (VectorAssembler + estimator) once per
         batch via the Spark plan; caching post-transform ensures the model
         runs once total.
      3. Batched ``_safe_corr_expr(feat, prediction)`` in groups of
         ``_CORR_BATCH`` (ANSI-safe — zero variance → NULL, not
         DIVIDE_BY_ZERO).
      4. Batched ``F.mean(feat)`` in groups of ``_MEAN_BATCH``.
      5. ``unpersist()`` in ``finally`` so the cached sample never leaks.

    All heavy work stays on executors; the driver only collects scalar
    aggregate rows via ``.head()``.
    """
    from pyspark.sql import functions as F  # noqa: N812

    from customer_retention.core.compat import _safe_corr_expr

    feature_list = list(feature_columns)
    if not feature_list:
        raise ValueError("compute_shap_attribution requires at least one feature column")

    filled = df.fillna(0, subset=feature_list)
    scored = (
        pipeline_model.transform(filled.limit(sample_size))
        .select(*feature_list, _PREDICTION_COL)
        .cache()
    )
    try:
        materialized = scored.count()

        importances: Dict[str, float] = {}
        for start in range(0, len(feature_list), _CORR_BATCH):
            batch = feature_list[start : start + _CORR_BATCH]
            exprs = [
                _safe_corr_expr(
                    F.col(c).cast("double"),
                    F.col(_PREDICTION_COL).cast("double"),
                ).alias(f"c_{i}")
                for i, c in enumerate(batch)
            ]
            row = scored.agg(*exprs).head()
            for i, c in enumerate(batch):
                v = row[f"c_{i}"] if row is not None else None
                importances[c] = abs(float(v)) if v is not None else 0.0

        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}
        else:
            uniform = 1.0 / len(feature_list)
            importances = {k: uniform for k in feature_list}

        means: Dict[str, float] = {}
        for start in range(0, len(feature_list), _MEAN_BATCH):
            batch = feature_list[start : start + _MEAN_BATCH]
            exprs = [
                F.mean(F.col(c).cast("double")).alias(f"m_{i}")
                for i, c in enumerate(batch)
            ]
            row = scored.agg(*exprs).head()
            for i, c in enumerate(batch):
                v = row[f"m_{i}"] if row is not None else None
                means[c] = float(v) if v is not None else 0.0
    finally:
        scored.unpersist()

    return ShapAttribution(
        importances=importances,
        background_means=means,
        feature_columns=feature_list,
        sample_size=int(materialized),
    )


def log_attribution(
    attribution: ShapAttribution,
    artifact_path: str = ARTIFACT_FILENAME,
) -> None:
    """Attach the attribution to the currently active MLflow run so it is
    version-bound to the logged model. Downstream consumers load via the
    run_id resolved from the registered model alias."""
    import mlflow

    mlflow.log_dict(attribution.to_dict(), artifact_path)


def resolve_run_id_for_model_uri(model_uri: str) -> str:
    """Return the MLflow ``run_id`` behind ``models:/<name>@<alias>`` or
    ``models:/<name>/<version>``. Fails fast on malformed URIs — no silent
    fallback — per Coding_Practices.md."""
    from mlflow.tracking import MlflowClient

    if not model_uri.startswith("models:/"):
        raise ValueError(f"model_uri must start with 'models:/': {model_uri!r}")
    spec = model_uri[len("models:/") :]
    client = MlflowClient()
    if "@" in spec:
        name, alias = spec.split("@", 1)
        version = client.get_model_version_by_alias(name, alias)
    elif "/" in spec:
        name, ver = spec.rsplit("/", 1)
        version = client.get_model_version(name, ver)
    else:
        raise ValueError(f"Unsupported model_uri form: {model_uri!r}")
    return version.run_id


def load_attribution_from_run(
    run_id: str,
    artifact_path: str = ARTIFACT_FILENAME,
) -> ShapAttribution:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    with tempfile.TemporaryDirectory() as tmp:
        local = client.download_artifacts(run_id, artifact_path, tmp)
        data = json.loads(Path(local).read_text())
    return ShapAttribution.from_dict(data)


def load_attribution_from_model_uri(
    model_uri: str,
    artifact_path: str = ARTIFACT_FILENAME,
) -> ShapAttribution:
    return load_attribution_from_run(
        resolve_run_id_for_model_uri(model_uri), artifact_path
    )
