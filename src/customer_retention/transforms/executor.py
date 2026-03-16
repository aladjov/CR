"""TransformExecutor — single dispatch table for all transformation types.

Maps :class:`TransformationStep` types to the appropriate function in
:mod:`ops` (stateless) or class in :mod:`fitted` (stateful).
"""

from __future__ import annotations

from customer_retention.core.compat import DataFrame, _is_spark_pandas, spark_checkpoint
from customer_retention.generators.pipeline_generator.models import (
    PipelineTransformationType,
    TransformationStep,
)

_CHECKPOINT_INTERVAL = 10


def _invalidate_psseries(df: DataFrame) -> None:
    """Clear stale ``_psseries`` cache on pyspark.pandas DataFrames.

    pyspark.pandas ``__setitem__`` calls ``_update_internal_frame`` which
    updates ``_internal`` (column count changes) but does NOT reset the
    ``_psseries`` dict.  The next column read hits an assertion comparing
    ``len(column_labels)`` vs ``len(_psseries)``.  Clearing the cache
    forces a correct rebuild on next access.
    """
    if hasattr(df, '_psseries'):
        df._psseries = None


from . import ops
from .artifact_store import ArtifactStore
from .fitted import FittedEncoder, FittedPowerTransform, FittedScaler


class TransformExecutor:
    """Applies :class:`TransformationStep` objects to DataFrames.

    Works with both pandas and pyspark.pandas DataFrames via the compat
    layer.  Used by generated pipelines and exploration code alike.
    """

    def apply(
        self,
        df: DataFrame,
        step: TransformationStep,
        *,
        fit_mode: bool = False,
        artifact_store: ArtifactStore | None = None,
    ) -> DataFrame:
        handler = self._DISPATCH.get(step.type)
        if handler is None:
            raise ValueError(f"Unknown transformation type: {step.type}")
        return handler(self, df, step, fit_mode=fit_mode, artifact_store=artifact_store)

    def apply_all(
        self,
        df: DataFrame,
        steps: list[TransformationStep],
        *,
        fit_mode: bool = False,
        artifact_store: ArtifactStore | None = None,
    ) -> DataFrame:
        distributed = _is_spark_pandas(df)
        if distributed:
            self._precompute_quantiles(df, steps)
        for i, step in enumerate(steps):
            df = self.apply(df, step, fit_mode=fit_mode, artifact_store=artifact_store)
            if distributed:
                _invalidate_psseries(df)
                if (i + 1) % _CHECKPOINT_INTERVAL == 0 and i + 1 < len(steps):
                    df = spark_checkpoint(df)
        return df

    def _precompute_quantiles(self, df: DataFrame, steps: list[TransformationStep]) -> None:
        cap_steps = [
            s for s in steps
            if s.type == PipelineTransformationType.CAP_THEN_LOG and s.column in df.columns
        ]
        if not cap_steps:
            return
        quantiles = {s.column: df[s.column].quantile(0.99) for s in cap_steps}
        for step in cap_steps:
            step.parameters["_precomputed_q99"] = float(quantiles[step.column])

    def _apply_fitted(self, fitted, df, step, *, fit_mode=False, artifact_store=None):
        if fit_mode:
            return fitted.fit_transform(df, step.column, artifact_store)
        return fitted.transform(df, step.column, artifact_store)

    def _handle_impute_null(self, df, step, **kw):
        return ops.apply_impute_null(df, step.column, value=step.parameters.get("value", 0))

    def _handle_cap_outlier(self, df, step, **kw):
        return ops.apply_cap_outlier(
            df,
            step.column,
            lower=step.parameters.get("lower", 0),
            upper=step.parameters.get("upper", 1_000_000),
        )

    def _handle_type_cast(self, df, step, **kw):
        return ops.apply_type_cast(df, step.column, dtype=step.parameters.get("dtype", "float"))

    def _handle_drop_column(self, df, step, **kw):
        return ops.apply_drop_column(df, step.column)

    def _handle_winsorize(self, df, step, **kw):
        return ops.apply_winsorize(
            df,
            step.column,
            lower_bound=step.parameters.get("lower_bound", 0),
            upper_bound=step.parameters.get("upper_bound", 1_000_000),
        )

    def _handle_segment_aware_cap(self, df, step, **kw):
        return ops.apply_segment_aware_cap(
            df, step.column, n_segments=step.parameters.get("n_segments", 2)
        )

    def _handle_log_transform(self, df, step, **kw):
        return ops.apply_log_transform(df, step.column)

    def _handle_sqrt_transform(self, df, step, **kw):
        return ops.apply_sqrt_transform(df, step.column)

    def _handle_zero_inflation(self, df, step, **kw):
        return ops.apply_zero_inflation_handling(df, step.column)

    def _handle_cap_then_log(self, df, step, **kw):
        return ops.apply_cap_then_log(
            df, step.column, _precomputed_q99=step.parameters.get("_precomputed_q99"),
        )

    def _handle_encode(self, df, step, *, fit_mode=False, artifact_store=None, **kw):
        method = step.parameters.get("method", "one_hot")
        if method == "one_hot":
            return ops.apply_one_hot_encode(df, step.column)
        return self._apply_fitted(
            FittedEncoder(), df, step, fit_mode=fit_mode, artifact_store=artifact_store
        )

    def _handle_scale(self, df, step, *, fit_mode=False, artifact_store=None, **kw):
        method = step.parameters.get("method", "standard")
        return self._apply_fitted(
            FittedScaler(method), df, step, fit_mode=fit_mode, artifact_store=artifact_store
        )

    def _handle_yeo_johnson(self, df, step, *, fit_mode=False, artifact_store=None, **kw):
        return self._apply_fitted(
            FittedPowerTransform(), df, step, fit_mode=fit_mode, artifact_store=artifact_store
        )

    def _handle_feature_select(self, df, step, **kw):
        return ops.apply_feature_select(df, step.column)

    def _handle_derived_column(self, df, step, **kw):
        method = step.parameters.get("method") or step.parameters.get("action")
        if method == "ratio":
            return ops.apply_derived_ratio(
                df,
                step.column,
                numerator=step.parameters.get("numerator", ""),
                denominator=step.parameters.get("denominator", ""),
            )
        if method == "interaction":
            return ops.apply_derived_interaction(
                df,
                step.column,
                col_a=step.parameters.get("col_a", ""),
                col_b=step.parameters.get("col_b", ""),
            )
        if method == "composite":
            return ops.apply_derived_composite(df, step.column, columns=step.parameters.get("columns", []))
        return df

    _DISPATCH = {
        PipelineTransformationType.IMPUTE_NULL: _handle_impute_null,
        PipelineTransformationType.CAP_OUTLIER: _handle_cap_outlier,
        PipelineTransformationType.TYPE_CAST: _handle_type_cast,
        PipelineTransformationType.DROP_COLUMN: _handle_drop_column,
        PipelineTransformationType.WINSORIZE: _handle_winsorize,
        PipelineTransformationType.SEGMENT_AWARE_CAP: _handle_segment_aware_cap,
        PipelineTransformationType.LOG_TRANSFORM: _handle_log_transform,
        PipelineTransformationType.SQRT_TRANSFORM: _handle_sqrt_transform,
        PipelineTransformationType.YEO_JOHNSON: _handle_yeo_johnson,
        PipelineTransformationType.ZERO_INFLATION_HANDLING: _handle_zero_inflation,
        PipelineTransformationType.CAP_THEN_LOG: _handle_cap_then_log,
        PipelineTransformationType.ENCODE: _handle_encode,
        PipelineTransformationType.SCALE: _handle_scale,
        PipelineTransformationType.FEATURE_SELECT: _handle_feature_select,
        PipelineTransformationType.DERIVED_COLUMN: _handle_derived_column,
    }
