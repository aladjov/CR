from typing import Any, Dict, Tuple

import numpy as np

from customer_retention.core.compat import DataFrame, _is_spark_pandas, as_spark_df

from .feature_scaler import FeatureScaler, ScalerType, ScalingResult


class SparkFeatureScaler(FeatureScaler):
    def __init__(
        self,
        scaler_type: ScalerType = ScalerType.ROBUST,
        fit_on_train_only: bool = True,
        save_scaler: bool = True,
    ):
        super().__init__(
            scaler_type=scaler_type,
            fit_on_train_only=fit_on_train_only,
            save_scaler=save_scaler,
        )
        self._params: Dict[str, Dict[str, float]] = {}

    def fit_transform(
        self,
        X_train: DataFrame,
        X_test: DataFrame,
    ) -> ScalingResult:
        self._feature_names = list(X_train.columns)

        if self.scaler_type == ScalerType.NONE:
            return ScalingResult(
                scaler=None,
                X_train_scaled=X_train,
                X_test_scaled=X_test,
                scaling_params={},
            )

        if _is_spark_pandas(X_train):
            return self._fit_transform_spark(X_train, X_test)

        self._params = self._compute_params(X_train)
        X_train_scaled = self._apply(X_train)
        X_test_scaled = self._apply(X_test)

        scaling_params = self._extract_params()

        return ScalingResult(
            scaler=self._params if self.save_scaler else None,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            scaling_params=scaling_params,
        )

    def _fit_transform_spark(
        self,
        X_train: DataFrame,
        X_test: DataFrame,
    ) -> ScalingResult:
        from customer_retention.core.compat.spark_backend import _as_pandas_api

        train_spark = as_spark_df(X_train)
        test_spark = as_spark_df(X_test)

        self._params = self._compute_params_spark(train_spark)

        X_train_scaled = _as_pandas_api(self._apply_spark(train_spark))
        X_test_scaled = _as_pandas_api(self._apply_spark(test_spark))

        scaling_params = self._extract_params()

        return ScalingResult(
            scaler=self._params if self.save_scaler else None,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            scaling_params=scaling_params,
        )

    def transform(self, X: DataFrame) -> DataFrame:
        if not self._params:
            return X
        if _is_spark_pandas(X):
            from customer_retention.core.compat.spark_backend import _as_pandas_api

            return _as_pandas_api(self._apply_spark(as_spark_df(X)))
        return self._apply(X)

    _BATCH = 500

    def _compute_params_spark(self, spark_df: Any) -> Dict[str, Dict[str, float]]:
        """Compute scaling parameters via batched .agg() — no row amplification."""
        from pyspark.sql import functions as F  # noqa: N812

        cols = self._feature_names
        if not cols:
            return {}
        safe = {c: f"__s{i}__" for i, c in enumerate(cols)}
        work = spark_df.toDF(*[safe.get(c, c) for c in spark_df.columns])

        params: Dict[str, Dict[str, float]] = {}

        for start in range(0, len(cols), self._BATCH):
            batch = cols[start : start + self._BATCH]
            sb = [safe[c] for c in batch]
            if self.scaler_type == ScalerType.STANDARD:
                exprs = []
                for s in sb:
                    exprs.extend(
                        [
                            F.mean(F.col(s).cast("double")).alias(f"{s}__m"),
                            F.stddev_pop(F.col(s).cast("double")).alias(f"{s}__s"),
                        ]
                    )
                row = work.agg(*exprs).head()
                for c, s in zip(batch, sb):
                    params[c] = {
                        "mean": float(row[f"{s}__m"]) if row[f"{s}__m"] is not None else float("nan"),
                        "std": float(row[f"{s}__s"]) if row[f"{s}__s"] is not None else 0.0,
                    }
            elif self.scaler_type == ScalerType.ROBUST:
                exprs = []
                for s in sb:
                    dc = F.col(s).cast("double")
                    exprs.extend(
                        [
                            F.percentile_approx(dc, 0.5).alias(f"{s}__med"),
                            F.percentile_approx(dc, 0.25).alias(f"{s}__q25"),
                            F.percentile_approx(dc, 0.75).alias(f"{s}__q75"),
                        ]
                    )
                row = work.agg(*exprs).head()
                for c, s in zip(batch, sb):
                    med, q25, q75 = row[f"{s}__med"], row[f"{s}__q25"], row[f"{s}__q75"]
                    params[c] = {
                        "center": float(med) if med is not None else float("nan"),
                        "iqr": (float(q75) - float(q25)) if q75 is not None and q25 is not None else 0.0,
                    }
            elif self.scaler_type == ScalerType.MINMAX:
                exprs = []
                for s in sb:
                    dc = F.col(s).cast("double")
                    exprs.extend([F.min(dc).alias(f"{s}__min"), F.max(dc).alias(f"{s}__max")])
                row = work.agg(*exprs).head()
                for c, s in zip(batch, sb):
                    params[c] = {
                        "min": float(row[f"{s}__min"]) if row[f"{s}__min"] is not None else float("nan"),
                        "max": float(row[f"{s}__max"]) if row[f"{s}__max"] is not None else float("nan"),
                    }

        return params

    def _apply_spark(self, spark_df: Any) -> Any:
        """Apply scaling via a single native Spark .select() — no per-column __setitem__."""
        from pyspark.sql import functions as F  # noqa: N812

        offsets, scales = self._as_vectors()
        zero_mask = scales == 0
        safe_scales = np.array([1.0 if z else s for z, s in zip(zero_mask, scales)])

        all_spark_cols = set(spark_df.columns)
        exprs = []
        for i, col in enumerate(self._feature_names):
            if col not in all_spark_cols:
                continue
            if zero_mask[i]:
                exprs.append(F.lit(0.0).cast("double").alias(col))
            else:
                exprs.append(((F.col(col) - float(offsets[i])) / float(safe_scales[i])).alias(col))

        # Include any columns not in feature_names (passthrough)
        feature_set = set(self._feature_names)
        for col in spark_df.columns:
            if col not in feature_set:
                exprs.append(F.col(col))

        return spark_df.select(exprs)

    def _compute_params(self, X: DataFrame) -> Dict[str, Dict[str, float]]:
        params: Dict[str, Dict[str, float]] = {}

        if self.scaler_type == ScalerType.STANDARD:
            means = X.mean()
            stds = X.std(axis=0, ddof=0)
            for col in X.columns:
                params[col] = {"mean": float(means[col]), "std": float(stds[col])}

        elif self.scaler_type == ScalerType.ROBUST:
            medians = X.median()
            q25 = X.quantile(0.25)
            q75 = X.quantile(0.75)
            for col in X.columns:
                params[col] = {
                    "center": float(medians[col]),
                    "iqr": float(q75[col]) - float(q25[col]),
                }

        elif self.scaler_type == ScalerType.MINMAX:
            mins = X.min()
            maxs = X.max()
            for col in X.columns:
                params[col] = {"min": float(mins[col]), "max": float(maxs[col])}

        return params

    def _apply(self, X: DataFrame) -> DataFrame:
        offsets, scales = self._as_vectors()
        zero_mask = scales == 0
        safe_scales = np.array([1.0 if z else s for z, s in zip(zero_mask, scales)])
        result = X.copy()
        for i, col in enumerate(self._feature_names):
            if zero_mask[i]:
                result[col] = 0.0
            else:
                result[col] = (X[col] - offsets[i]) / safe_scales[i]
        return result

    def _as_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self._feature_names)
        offsets = np.empty(n)
        scales = np.empty(n)
        for i, c in enumerate(self._feature_names):
            p = self._params[c]
            if self.scaler_type == ScalerType.STANDARD:
                offsets[i], scales[i] = p["mean"], p["std"]
            elif self.scaler_type == ScalerType.ROBUST:
                offsets[i], scales[i] = p["center"], p["iqr"]
            elif self.scaler_type == ScalerType.MINMAX:
                offsets[i] = p["min"]
                scales[i] = p["max"] - p["min"]
        return offsets, scales

    def _extract_params(self) -> Dict[str, Any]:
        if not self._params:
            return {}

        if self.scaler_type == ScalerType.STANDARD:
            return {
                "mean": [self._params[c]["mean"] for c in self._feature_names],
                "scale": [self._params[c]["std"] for c in self._feature_names],
            }

        if self.scaler_type == ScalerType.ROBUST:
            return {
                "center": [self._params[c]["center"] for c in self._feature_names],
                "scale": [self._params[c]["iqr"] for c in self._feature_names],
            }

        if self.scaler_type == ScalerType.MINMAX:
            scales = []
            for c in self._feature_names:
                r = self._params[c]["max"] - self._params[c]["min"]
                scales.append(1.0 / r if r != 0 else 0.0)
            return {
                "data_min": [self._params[c]["min"] for c in self._feature_names],
                "data_max": [self._params[c]["max"] for c in self._feature_names],
                "scale": scales,
            }

        return {}
