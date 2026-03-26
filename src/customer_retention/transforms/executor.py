"""TransformExecutor — single dispatch table for all transformation types.

Maps :class:`TransformationStep` types to the appropriate function in
:mod:`ops` (stateless) or class in :mod:`fitted` (stateful).
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from customer_retention.core.compat import DataFrame, _is_spark_pandas, as_spark_df
from customer_retention.core.compat.spark_backend import _as_pandas_api
from customer_retention.generators.pipeline_generator.models import (
    PipelineTransformationType,
    TransformationStep,
)

StepDoneCallback = Callable[[int, int, TransformationStep, float, float], None]
BatchDoneCallback = Callable[[int, int, int, PipelineTransformationType, float, float], None]


def format_step_progress(step_idx: int, total: int, step: TransformationStep, step_seconds: float, total_seconds: float) -> str:
    done = step_idx + 1
    remaining = (total_seconds / done) * (total - done) if done < total else 0.0
    return f"  [{done}/{total}] {step.type.name} \u2192 {step.column} ({step_seconds:.1f}s, ~{remaining:.0f}s left)"


def print_step_progress(step_idx: int, total: int, step: TransformationStep, step_seconds: float, total_seconds: float) -> None:
    print(format_step_progress(step_idx, total, step, step_seconds, total_seconds))


def format_batch_progress(start_idx: int, end_idx: int, total: int, transform_type: PipelineTransformationType, batch_seconds: float, total_seconds: float) -> str:
    done = end_idx + 1
    count = done - start_idx
    remaining = (total_seconds / done) * (total - done) if done < total else 0.0
    return f"  [{start_idx + 1}-{done}/{total}] {transform_type.name} \u00d7 {count} columns ({batch_seconds:.1f}s, ~{remaining:.0f}s left)"


def print_batch_progress(start_idx: int, end_idx: int, total: int, transform_type: PipelineTransformationType, batch_seconds: float, total_seconds: float) -> None:
    print(format_batch_progress(start_idx, end_idx, total, transform_type, batch_seconds, total_seconds))


_CHECKPOINT_INTERVAL = 10
_PLAN_TRUNCATION_INTERVAL = 30
_BATCH_CHUNK_SIZE = 500

from . import ops
from .artifact_store import ArtifactStore
from .fitted import FittedEncoder, FittedPowerTransform, FittedScaler

# ── Batch grouping ────────────────────────────────────────────────


def _decompose_into_waves(steps: list[TransformationStep]) -> list[list[TransformationStep]]:
    waves: list[list[TransformationStep]] = []
    current: list[TransformationStep] = []
    seen: set[str] = set()
    for step in steps:
        if step.column in seen:
            waves.append(current)
            current = []
            seen = set()
        current.append(step)
        seen.add(step.column)
        if step.type == PipelineTransformationType.ZERO_INFLATION_HANDLING:
            seen.add(f"{step.column}_is_zero")
            seen.add(f"{step.column}_log")
    if current:
        waves.append(current)
    return waves


def _sort_wave_by_type(wave: list[TransformationStep], batchable_types: frozenset) -> list[TransformationStep]:
    type_buckets: dict[PipelineTransformationType, list[TransformationStep]] = {}
    non_batchable: list[TransformationStep] = []
    for step in wave:
        if step.type in batchable_types:
            type_buckets.setdefault(step.type, []).append(step)
        else:
            non_batchable.append(step)
    result: list[TransformationStep] = []
    for typ in type_buckets:
        result.extend(type_buckets[typ])
    result.extend(non_batchable)
    return result


def _chunk_list(items: list, chunk_size: int) -> list[list]:
    if len(items) <= chunk_size:
        return [items]
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _group_batchable_runs(steps: list[TransformationStep], batchable_types: frozenset) -> list[list[TransformationStep]]:
    result: list[list[TransformationStep]] = []
    for wave in _decompose_into_waves(steps):
        sorted_wave = _sort_wave_by_type(wave, batchable_types)
        i = 0
        while i < len(sorted_wave):
            step = sorted_wave[i]
            if step.type in batchable_types:
                batch = [step]
                j = i + 1
                while j < len(sorted_wave) and sorted_wave[j].type == step.type:
                    batch.append(sorted_wave[j])
                    j += 1
                for chunk in _chunk_list(batch, _BATCH_CHUNK_SIZE):
                    result.append(chunk)
                i = j
            else:
                result.append([step])
                i += 1
    return result


def _yj_params_from_steps(steps):
    return [
        (s.column, s.parameters["_fitted_yj"]["lmbda"], s.parameters["_fitted_yj"].get("std_mean"),
         s.parameters["_fitted_yj"].get("std_scale"), s.parameters["_fitted_yj"].get("standardize", True))
        for s in steps if "_fitted_yj" in s.parameters
    ]


_PANDAS_BATCH = {
    PipelineTransformationType.LOG_TRANSFORM: lambda df, steps: ops.apply_batch_log_transform(df, [s.column for s in steps]),
    PipelineTransformationType.SQRT_TRANSFORM: lambda df, steps: ops.apply_batch_sqrt_transform(df, [s.column for s in steps]),
    PipelineTransformationType.ZERO_INFLATION_HANDLING: lambda df, steps: ops.apply_batch_zero_inflation(df, [s.column for s in steps]),
    PipelineTransformationType.CAP_THEN_LOG: lambda df, steps: ops.apply_batch_cap_then_log(df, [(s.column, s.parameters.get("_precomputed_q99")) for s in steps]),
    PipelineTransformationType.YEO_JOHNSON: lambda df, steps: ops.apply_batch_yeo_johnson(df, _yj_params_from_steps(steps)),
}

_PANDAS_BATCHABLE_TYPES = frozenset(_PANDAS_BATCH)


def _preload_yj_params(steps: list, artifact_store) -> None:
    for s in steps:
        if s.type == PipelineTransformationType.YEO_JOHNSON and "_fitted_yj" not in s.parameters:
            aid = f"{s.column}_power_transformer"
            if artifact_store.has(aid):
                pt = artifact_store.load(aid)
                s.parameters["_fitted_yj"] = {
                    "lmbda": float(pt.lambdas_[0]),
                    "std_mean": float(pt._scaler.mean_[0]) if pt.standardize else None,
                    "std_scale": float(pt._scaler.scale_[0]) if pt.standardize else None,
                    "standardize": pt.standardize,
                }


def _resolve_batch_reporter(on_step_done, on_batch_done):
    if on_batch_done is not None:
        return on_batch_done
    if on_step_done is print_step_progress:
        return print_batch_progress
    return None


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
        on_step_done: Optional[StepDoneCallback] = None,
        on_batch_done: Optional[BatchDoneCallback] = None,
    ) -> DataFrame:
        distributed = _is_spark_pandas(df)
        if distributed:
            return self._apply_all_distributed(df, steps, fit_mode=fit_mode, artifact_store=artifact_store, on_step_done=on_step_done, on_batch_done=on_batch_done)
        if not fit_mode and artifact_store:
            _preload_yj_params(steps, artifact_store)
        batch_reporter = _resolve_batch_reporter(on_step_done, on_batch_done)
        batchable = _PANDAS_BATCHABLE_TYPES - {PipelineTransformationType.YEO_JOHNSON} if fit_mode else _PANDAS_BATCHABLE_TYPES
        t0 = time.perf_counter()
        step_idx = 0
        for batch in _group_batchable_runs(steps, batchable):
            batch_handler = _PANDAS_BATCH.get(batch[0].type) if len(batch) > 1 else None
            if batch_handler:
                t_batch = time.perf_counter()
                df = batch_handler(df, batch)
                now = time.perf_counter()
                if batch_reporter:
                    batch_reporter(step_idx, step_idx + len(batch) - 1, len(steps), batch[0].type, now - t_batch, now - t0)
                elif on_step_done:
                    on_step_done(step_idx + len(batch) - 1, len(steps), batch[-1], now - t_batch, now - t0)
                step_idx += len(batch)
            else:
                for step in batch:
                    t_step = time.perf_counter()
                    df = self.apply(df, step, fit_mode=fit_mode, artifact_store=artifact_store)
                    if on_step_done:
                        now = time.perf_counter()
                        on_step_done(step_idx, len(steps), step, now - t_step, now - t0)
                    step_idx += 1
        return df

    def _apply_all_distributed(self, df: DataFrame, steps: list[TransformationStep], *, fit_mode: bool = False, artifact_store: ArtifactStore | None = None, on_step_done: Optional[StepDoneCallback] = None, on_batch_done: Optional[BatchDoneCallback] = None) -> DataFrame:
        spark_df = as_spark_df(df)
        self._precompute_quantiles_distributed(spark_df, steps)
        if fit_mode and artifact_store:
            self._prefit_distributed(spark_df, steps, artifact_store)
        elif not fit_mode and artifact_store:
            _preload_yj_params(steps, artifact_store)
        batch_reporter = _resolve_batch_reporter(on_step_done, on_batch_done)
        roundtrip_count = 0
        since_checkpoint = 0
        t0 = time.perf_counter()
        step_idx = 0
        for batch in _group_batchable_runs(steps, _SPARK_BATCHABLE_TYPES):
            batch_handler = _SPARK_BATCH_DISPATCH.get(batch[0].type) if len(batch) > 1 else None
            if batch_handler:
                t_batch = time.perf_counter()
                spark_df = batch_handler(spark_df, batch)
                since_checkpoint += 1
                now = time.perf_counter()
                if batch_reporter:
                    batch_reporter(step_idx, step_idx + len(batch) - 1, len(steps), batch[0].type, now - t_batch, now - t0)
                elif on_step_done:
                    on_step_done(step_idx + len(batch) - 1, len(steps), batch[-1], now - t_batch, now - t0)
                step_idx += len(batch)
            else:
                for step in batch:
                    t_step = time.perf_counter()
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
                    if on_step_done:
                        now = time.perf_counter()
                        on_step_done(step_idx, len(steps), step, now - t_step, now - t0)
                    since_checkpoint += 1
                    step_idx += 1
            needs_checkpoint = (
                (roundtrip_count > 0 and roundtrip_count % _CHECKPOINT_INTERVAL == 0)
                or since_checkpoint >= _PLAN_TRUNCATION_INTERVAL
            )
            if needs_checkpoint and step_idx < len(steps):
                spark_df = spark_df.localCheckpoint(eager=True)
                since_checkpoint = 0
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
        fitted = []
        for s in yj_steps:
            pt = FittedPowerTransform()
            pt.fit_from_local(sample_pdf[s.column])
            fitted.append(pt._pt)
        for s, pt_obj in zip(yj_steps, fitted):
            artifact_store.register("power_transformer", s.column, pt_obj)
            lmbda = float(pt_obj.lambdas_[0])
            std_mean = float(pt_obj._scaler.mean_[0]) if pt_obj.standardize else None
            std_scale = float(pt_obj._scaler.scale_[0]) if pt_obj.standardize else None
            s.parameters["_fitted_yj"] = {"lmbda": lmbda, "std_mean": std_mean, "std_scale": std_scale, "standardize": pt_obj.standardize}

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
            yj_params = step.parameters.get("_fitted_yj") or params
            if yj_params:
                return spark_ops.spark_yeo_johnson(spark_df, step.column, lmbda=yj_params["lmbda"], std_mean=yj_params.get("std_mean"), std_scale=yj_params.get("std_scale"), standardize=yj_params.get("standardize", True))
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


def _make_spark_batch_dispatch():
    from . import spark_ops
    return {
        PipelineTransformationType.LOG_TRANSFORM: lambda df, steps: spark_ops.spark_batch_log_transform(df, [s.column for s in steps]),
        PipelineTransformationType.SQRT_TRANSFORM: lambda df, steps: spark_ops.spark_batch_sqrt_transform(df, [s.column for s in steps]),
        PipelineTransformationType.ZERO_INFLATION_HANDLING: lambda df, steps: spark_ops.spark_batch_zero_inflation(df, [s.column for s in steps]),
        PipelineTransformationType.CAP_THEN_LOG: lambda df, steps: spark_ops.spark_batch_cap_then_log(df, [(s.column, s.parameters.get("_precomputed_q99")) for s in steps]),
        PipelineTransformationType.YEO_JOHNSON: lambda df, steps: spark_ops.spark_batch_yeo_johnson(df, _yj_params_from_steps(steps)),
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
    _SPARK_BATCH_DISPATCH = _make_spark_batch_dispatch()
    _SPARK_BATCHABLE_TYPES = frozenset(_SPARK_BATCH_DISPATCH)
except ImportError:
    _SPARK_DISPATCH = {}
    _SPARK_BATCH_DISPATCH = {}
    _SPARK_BATCHABLE_TYPES = frozenset()
