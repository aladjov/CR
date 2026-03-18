"""TransformExecutor — single dispatch table for all transformation types.

Maps :class:`TransformationStep` types to the appropriate function in
:mod:`ops` (stateless) or class in :mod:`fitted` (stateful).
"""

from __future__ import annotations

from customer_retention.core.compat import DataFrame, _is_spark_pandas, as_spark_df
from customer_retention.core.compat.spark_backend import _as_pandas_api
from customer_retention.generators.pipeline_generator.models import (
    PipelineTransformationType,
    TransformationStep,
)

_CHECKPOINT_INTERVAL = 10

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
            return self._apply_all_distributed(df, steps, fit_mode=fit_mode, artifact_store=artifact_store)
        for step in steps:
            df = self.apply(df, step, fit_mode=fit_mode, artifact_store=artifact_store)
        return df

    def _apply_all_distributed(self, df: DataFrame, steps: list[TransformationStep], *, fit_mode: bool = False, artifact_store: ArtifactStore | None = None) -> DataFrame:
        spark_df = as_spark_df(df)
        self._precompute_quantiles_distributed(spark_df, steps)
        if fit_mode and artifact_store:
            self._prefit_distributed(spark_df, steps, artifact_store)
        roundtrip_count = 0
        for i, step in enumerate(steps):
            handler = _SPARK_DISPATCH.get(step.type)
            result = handler(spark_df, step) if handler is not None else None
            if result is not None:
                spark_df = result
            else:
                spark_result = self._apply_fitted_spark(spark_df, step, fit_mode, artifact_store)
                if spark_result is not None:
                    spark_df = spark_result
                else:
                    ps_df = _as_pandas_api(spark_df)
                    spark_df = as_spark_df(self.apply(ps_df, step, fit_mode=fit_mode, artifact_store=artifact_store))
                    roundtrip_count += 1
            if roundtrip_count > 0 and roundtrip_count % _CHECKPOINT_INTERVAL == 0 and i + 1 < len(steps):
                spark_df = spark_df.localCheckpoint(eager=True)
        return _as_pandas_api(spark_df)

    def _precompute_quantiles(self, df: DataFrame, steps: list[TransformationStep]) -> None:
        cap_steps = [
            s for s in steps
            if s.type == PipelineTransformationType.CAP_THEN_LOG and s.column in df.columns
        ]
        if not cap_steps:
            return
        if _is_spark_pandas(df):
            self._precompute_quantiles_distributed(as_spark_df(df), cap_steps)
        else:
            for s in cap_steps:
                s.parameters["_precomputed_q99"] = float(df[s.column].quantile(0.99))

    def _precompute_quantiles_distributed(self, spark_df, steps: list[TransformationStep]) -> None:
        spark_cols = set(spark_df.columns)
        cap_steps = [
            s for s in steps
            if s.type == PipelineTransformationType.CAP_THEN_LOG and s.column in spark_cols
        ]
        if not cap_steps:
            return
        cols = [s.column for s in cap_steps]
        quantile_lists = spark_df.approxQuantile(cols, [0.99], 0.01)
        for step, qs in zip(cap_steps, quantile_lists):
            step.parameters["_precomputed_q99"] = float(qs[0]) if qs else None

    def _prefit_distributed(self, spark_df, steps: list[TransformationStep], artifact_store: ArtifactStore) -> None:
        from pyspark.sql import functions as F  # noqa: N812
        spark_cols = set(spark_df.columns)
        self._batch_fit_scalers(spark_df, steps, artifact_store, spark_cols, F)
        self._batch_fit_power_transforms(spark_df, steps, artifact_store, spark_cols)

    def _batch_fit_scalers(self, spark_df, steps, artifact_store, spark_cols, F) -> None:
        scale_steps = [s for s in steps if s.type == PipelineTransformationType.SCALE and s.column in spark_cols]
        if not scale_steps:
            return
        standard = [s for s in scale_steps if s.parameters.get("method", "standard") == "standard"]
        minmax = [s for s in scale_steps if s.parameters.get("method") == "minmax"]
        if standard:
            cols = [s.column for s in standard]
            exprs = []
            for c in cols:
                exprs.extend([F.mean(c).alias(f"{c}__mean"), F.stddev_pop(c).alias(f"{c}__std"), F.count(c).alias(f"{c}__cnt")])
            row = spark_df.agg(*exprs).collect()[0]
            for s in standard:
                c = s.column
                mean = float(row[f"{c}__mean"] or 0)
                std = float(row[f"{c}__std"] or 0)
                count = int(row[f"{c}__cnt"] or 0)
                scaler = FittedScaler("standard")
                scaler.fit_from_precomputed(mean, std, count)
                artifact_store.register("scaler", c, scaler._scaler)
                scale = std if std > 0 else 1.0
                s.parameters["_spark_fitted"] = {"kind": "standard", "mean": mean, "scale": scale}
        if minmax:
            cols = [s.column for s in minmax]
            exprs = []
            for c in cols:
                exprs.extend([F.min(c).alias(f"{c}__min"), F.max(c).alias(f"{c}__max"), F.count(c).alias(f"{c}__cnt")])
            row = spark_df.agg(*exprs).collect()[0]
            for s in minmax:
                c = s.column
                col_min = float(row[f"{c}__min"] or 0)
                col_max = float(row[f"{c}__max"] or 0)
                count = int(row[f"{c}__cnt"] or 0)
                scaler = FittedScaler("minmax")
                scaler.fit_from_precomputed(0, 0, count, col_min, col_max)
                artifact_store.register("scaler", c, scaler._scaler)
                rng = col_max - col_min
                scale = 1.0 / rng if rng > 0 else 1.0
                offset = -col_min / rng if rng > 0 else -col_min
                s.parameters["_spark_fitted"] = {"kind": "minmax", "scale": scale, "offset": offset}

    def _batch_fit_power_transforms(self, spark_df, steps, artifact_store, spark_cols) -> None:
        yj_steps = [s for s in steps if s.type == PipelineTransformationType.YEO_JOHNSON and s.column in spark_cols]
        if not yj_steps:
            return
        columns = [s.column for s in yj_steps]
        n = spark_df.count()
        max_sample = FittedPowerTransform._MAX_FIT_SAMPLE
        if n > max_sample:
            from pyspark.sql import functions as F  # noqa: N812
            frac = min(1.0, max_sample / max(1, n))
            sample_pdf = spark_df.select([F.coalesce(F.col(c), F.lit(0.0)).alias(c) for c in columns]).sample(False, frac, seed=42).limit(max_sample).toPandas()
        else:
            from pyspark.sql import functions as F  # noqa: N812
            sample_pdf = spark_df.select([F.coalesce(F.col(c), F.lit(0.0)).alias(c) for c in columns]).toPandas()
        for s in yj_steps:
            pt = FittedPowerTransform()
            pt.fit_from_local(sample_pdf[s.column])
            artifact_store.register("power_transformer", s.column, pt._pt)
            lmbda = float(pt._pt.lambdas_[0])
            std_mean = float(pt._pt._scaler.mean_[0]) if pt._pt.standardize else None
            std_scale = float(pt._pt._scaler.scale_[0]) if pt._pt.standardize else None
            s.parameters["_spark_fitted"] = {"lmbda": lmbda, "std_mean": std_mean, "std_scale": std_scale, "standardize": pt._pt.standardize}

    @staticmethod
    def _apply_fitted_spark(spark_df, step: TransformationStep, fit_mode: bool, artifact_store: ArtifactStore | None):
        from . import spark_ops
        params = step.parameters.get("_spark_fitted")
        if step.type == PipelineTransformationType.SCALE:
            if params:
                if params["kind"] == "standard":
                    return spark_ops.spark_standard_scale(spark_df, step.column, mean=params["mean"], scale=params["scale"])
                return spark_ops.spark_minmax_scale(spark_df, step.column, scale=params["scale"], offset=params["offset"])
            if not fit_mode and artifact_store and artifact_store.has(f"{step.column}_scaler"):
                return _apply_scale_from_artifact(spark_df, step, artifact_store)
        if step.type == PipelineTransformationType.YEO_JOHNSON:
            if params:
                return spark_ops.spark_yeo_johnson(spark_df, step.column, lmbda=params["lmbda"], std_mean=params.get("std_mean"), std_scale=params.get("std_scale"), standardize=params.get("standardize", True))
            if not fit_mode and artifact_store and artifact_store.has(f"{step.column}_power_transformer"):
                return _apply_yj_from_artifact(spark_df, step, artifact_store)
        return None

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


def _spark_derived(spark_df, step: TransformationStep):
    from . import spark_ops
    method = step.parameters.get("method") or step.parameters.get("action")
    if method == "ratio":
        return spark_ops.spark_derived_ratio(spark_df, step.column, numerator=step.parameters.get("numerator", ""), denominator=step.parameters.get("denominator", ""))
    if method == "interaction":
        return spark_ops.spark_derived_interaction(spark_df, step.column, col_a=step.parameters.get("col_a", ""), col_b=step.parameters.get("col_b", ""))
    if method == "composite":
        return spark_ops.spark_derived_composite(spark_df, step.column, columns=step.parameters.get("columns", []))
    return spark_df


def _make_spark_dispatch():
    from . import spark_ops
    return {
        PipelineTransformationType.IMPUTE_NULL: lambda df, s: spark_ops.spark_impute_null(df, s.column, value=s.parameters.get("value", 0)),
        PipelineTransformationType.CAP_OUTLIER: lambda df, s: spark_ops.spark_cap_outlier(df, s.column, lower=s.parameters.get("lower", 0), upper=s.parameters.get("upper", 1_000_000)),
        PipelineTransformationType.TYPE_CAST: lambda df, s: spark_ops.spark_type_cast(df, s.column, dtype=s.parameters.get("dtype", "float")),
        PipelineTransformationType.DROP_COLUMN: lambda df, s: spark_ops.spark_drop_column(df, s.column),
        PipelineTransformationType.WINSORIZE: lambda df, s: spark_ops.spark_winsorize(df, s.column, lower_bound=s.parameters.get("lower_bound", 0), upper_bound=s.parameters.get("upper_bound", 1_000_000)),
        PipelineTransformationType.SEGMENT_AWARE_CAP: lambda df, s: spark_ops.spark_segment_aware_cap(df, s.column, n_segments=s.parameters.get("n_segments", 2)),
        PipelineTransformationType.LOG_TRANSFORM: lambda df, s: spark_ops.spark_log_transform(df, s.column),
        PipelineTransformationType.SQRT_TRANSFORM: lambda df, s: spark_ops.spark_sqrt_transform(df, s.column),
        PipelineTransformationType.ZERO_INFLATION_HANDLING: lambda df, s: spark_ops.spark_zero_inflation(df, s.column),
        PipelineTransformationType.CAP_THEN_LOG: lambda df, s: spark_ops.spark_cap_then_log(df, s.column, q99=s.parameters.get("_precomputed_q99")),
        PipelineTransformationType.ENCODE: lambda df, s: spark_ops.spark_one_hot_encode(df, s.column) if s.parameters.get("method", "one_hot") == "one_hot" else None,
        PipelineTransformationType.FEATURE_SELECT: lambda df, s: spark_ops.spark_feature_select(df, s.column),
        PipelineTransformationType.DERIVED_COLUMN: _spark_derived,
    }


def _apply_scale_from_artifact(spark_df, step, artifact_store):
    from sklearn.preprocessing import StandardScaler

    from . import spark_ops
    scaler = artifact_store.load(f"{step.column}_scaler")
    if isinstance(scaler, StandardScaler):
        return spark_ops.spark_standard_scale(spark_df, step.column, mean=float(scaler.mean_[0]), scale=float(scaler.scale_[0]))
    return spark_ops.spark_minmax_scale(spark_df, step.column, scale=float(scaler.scale_[0]), offset=float(scaler.min_[0]))


def _apply_yj_from_artifact(spark_df, step, artifact_store):
    from . import spark_ops
    pt = artifact_store.load(f"{step.column}_power_transformer")
    lmbda = float(pt.lambdas_[0])
    std_mean = float(pt._scaler.mean_[0]) if pt.standardize else None
    std_scale = float(pt._scaler.scale_[0]) if pt.standardize else None
    return spark_ops.spark_yeo_johnson(spark_df, step.column, lmbda=lmbda, std_mean=std_mean, std_scale=std_scale, standardize=pt.standardize)


try:
    _SPARK_DISPATCH = _make_spark_dispatch()
except ImportError:
    _SPARK_DISPATCH = {}
