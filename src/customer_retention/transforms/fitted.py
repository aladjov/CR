"""Stateful fit/transform classes.

These wrap sklearn transformers and integrate with :class:`ArtifactStore`
for persistence.  Transform application uses Series arithmetic, ensuring
the same source code works on both pandas and pyspark.pandas.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    PowerTransformer,
    StandardScaler,
)

from customer_retention.core.compat import DataFrame, _is_spark_pandas


class FittedScaler:
    """Wraps :class:`StandardScaler` / :class:`MinMaxScaler`."""

    def __init__(self, method: str = "standard"):
        self.method = method
        self._scaler = StandardScaler() if method == "standard" else MinMaxScaler()

    def fit_transform(self, df: DataFrame, column: str, artifact_store) -> DataFrame:
        if column not in df.columns:
            return df
        self._fit(df[column])
        df[column] = self._apply(df[column])
        artifact_store.register("scaler", column, self._scaler)
        return df

    def transform(self, df: DataFrame, column: str, artifact_store) -> DataFrame:
        if column not in df.columns:
            return df
        self._scaler = artifact_store.load(f"{column}_scaler")
        df[column] = self._apply(df[column])
        return df

    def _fit(self, series):
        if _is_spark_pandas(series):
            self._fit_from_stats(series)
        else:
            self._scaler.fit(series.to_numpy().reshape(-1, 1))

    def _fit_from_stats(self, series):
        mean, std = float(series.mean()), float(series.std(ddof=0))
        count = int(series.count())
        col_min, col_max = float(series.min()), float(series.max())
        self.fit_from_precomputed(mean, std, count, col_min, col_max)

    def fit_from_precomputed(self, mean: float, std: float, count: int, col_min: float = 0.0, col_max: float = 0.0):
        if isinstance(self._scaler, StandardScaler):
            self._scaler.mean_ = np.array([mean])
            self._scaler.scale_ = np.array([std if std > 0 else 1.0])
            self._scaler.var_ = np.array([std ** 2])
            self._scaler.n_features_in_ = 1
            self._scaler.n_samples_seen_ = count
        else:
            rng = col_max - col_min
            self._scaler.data_min_ = np.array([col_min])
            self._scaler.data_max_ = np.array([col_max])
            self._scaler.data_range_ = np.array([rng])
            if rng > 0:
                self._scaler.scale_ = np.array([1.0 / rng])
                self._scaler.min_ = np.array([-col_min / rng])
            else:
                self._scaler.scale_ = np.array([1.0])
                self._scaler.min_ = np.array([-col_min])
            self._scaler.n_features_in_ = 1
            self._scaler.n_samples_seen_ = count

    def _apply(self, series):
        if isinstance(self._scaler, StandardScaler):
            return (series - self._scaler.mean_[0]) / self._scaler.scale_[0]
        return series * self._scaler.scale_[0] + self._scaler.min_[0]


class FittedEncoder:
    """Wraps :class:`LabelEncoder` with unknown-class fallback."""

    def __init__(self):
        self._encoder = LabelEncoder()

    def fit_transform(self, df: DataFrame, column: str, artifact_store) -> DataFrame:
        if column not in df.columns:
            return df
        str_col = df[column].astype(str)
        classes = sorted(str_col.drop_duplicates().to_numpy().tolist())
        self._encoder.classes_ = np.array(classes)
        mapping = {v: i for i, v in enumerate(classes)}
        df[column] = str_col.map(mapping).fillna(0).astype(int)
        artifact_store.register("encoder", column, self._encoder)
        return df

    def transform(self, df: DataFrame, column: str, artifact_store) -> DataFrame:
        if column not in df.columns:
            return df
        encoder = artifact_store.load(f"{column}_encoder")
        mapping = {v: i for i, v in enumerate(encoder.classes_)}
        df[column] = df[column].astype(str).map(mapping).fillna(0).astype(int)
        return df


class FittedPowerTransform:
    """Wraps Yeo-Johnson :class:`PowerTransformer`."""

    _MAX_FIT_SAMPLE = 50_000

    def __init__(self):
        self._pt = PowerTransformer(method="yeo-johnson")

    def fit_transform(self, df: DataFrame, column: str, artifact_store) -> DataFrame:
        if column not in df.columns:
            return df
        series = df[column].fillna(0)
        self._fit(series)
        df[column] = self._apply_yj(series)
        artifact_store.register("power_transformer", column, self._pt)
        return df

    def _fit(self, series):
        if _is_spark_pandas(series):
            self._fit_from_sample(series)
        else:
            self._pt.fit(series.to_numpy().reshape(-1, 1))

    def _fit_from_sample(self, series):
        from customer_retention.core.compat import collect_for_sklearn, safe_sample
        n = len(series)
        if n > self._MAX_FIT_SAMPLE:
            sampled = safe_sample(series.to_frame(), self._MAX_FIT_SAMPLE).iloc[:, 0]
        else:
            sampled = series
        collected = collect_for_sklearn(sampled)
        self.fit_from_local(collected)

    def fit_from_local(self, pandas_series):
        self._pt.fit(pandas_series.to_numpy().reshape(-1, 1))

    def transform(self, df: DataFrame, column: str, artifact_store) -> DataFrame:
        if column not in df.columns:
            return df
        self._pt = artifact_store.load(f"{column}_power_transformer")
        df[column] = self._apply_yj(df[column].fillna(0))
        return df

    def _apply_yj(self, series):
        lmbda = self._pt.lambdas_[0]
        pos = series >= 0
        with np.errstate(invalid="ignore"):
            if abs(lmbda) < 1e-12:
                pos_vals = np.log1p(series)
            else:
                pos_vals = ((series + 1) ** lmbda - 1) / lmbda
            if abs(lmbda - 2) < 1e-12:
                neg_vals = -np.log1p(-series)
            else:
                neg_vals = -((-series + 1) ** (2 - lmbda) - 1) / (2 - lmbda)
        result = pos_vals.where(pos, neg_vals)
        if self._pt.standardize:
            result = (result - self._pt._scaler.mean_[0]) / self._pt._scaler.scale_[0]
        return result
