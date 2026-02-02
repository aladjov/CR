"""Stateful fit/transform classes.

These wrap sklearn transformers and integrate with :class:`ArtifactStore`
for persistence.  All classes use ``core.compat.to_pandas`` to convert
DataFrames to numpy-backed pandas before calling sklearn, ensuring the
same source code works on both pandas and pyspark.pandas.
"""

from __future__ import annotations

from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    PowerTransformer,
    StandardScaler,
)

from customer_retention.core.compat import DataFrame, ensure_pandas_series, to_pandas


class FittedScaler:
    """Wraps :class:`StandardScaler` / :class:`MinMaxScaler`."""

    def __init__(self, method: str = "standard"):
        self.method = method
        self._scaler = StandardScaler() if method == "standard" else MinMaxScaler()

    def fit_transform(self, df: DataFrame, column: str, artifact_store) -> DataFrame:
        if column not in df.columns:
            return df
        col_values = to_pandas(df[[column]])
        fitted = self._scaler.fit_transform(col_values)
        df[column] = fitted.ravel()
        artifact_store.register("scaler", column, self._scaler)
        return df

    def transform(self, df: DataFrame, column: str, artifact_store) -> DataFrame:
        if column not in df.columns:
            return df
        scaler = artifact_store.load(f"{column}_scaler")
        col_values = to_pandas(df[[column]])
        df[column] = scaler.transform(col_values).ravel()
        return df


class FittedEncoder:
    """Wraps :class:`LabelEncoder` with unknown-class fallback."""

    def __init__(self):
        self._encoder = LabelEncoder()

    def fit_transform(self, df: DataFrame, column: str, artifact_store) -> DataFrame:
        if column not in df.columns:
            return df
        series = ensure_pandas_series(df[column].astype(str))
        df[column] = self._encoder.fit_transform(series)
        artifact_store.register("encoder", column, self._encoder)
        return df

    def transform(self, df: DataFrame, column: str, artifact_store) -> DataFrame:
        if column not in df.columns:
            return df
        encoder = artifact_store.load(f"{column}_encoder")
        series = ensure_pandas_series(df[column].astype(str))
        df[column] = series.apply(
            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0
        )
        return df


class FittedPowerTransform:
    """Wraps Yeo-Johnson :class:`PowerTransformer`."""

    def __init__(self):
        self._pt = PowerTransformer(method="yeo-johnson")

    def fit_transform(self, df: DataFrame, column: str, artifact_store) -> DataFrame:
        if column not in df.columns:
            return df
        col_values = to_pandas(df[[column]].fillna(0))
        fitted = self._pt.fit_transform(col_values)
        df[column] = fitted.ravel()
        artifact_store.register("power_transformer", column, self._pt)
        return df

    def transform(self, df: DataFrame, column: str, artifact_store) -> DataFrame:
        if column not in df.columns:
            return df
        pt = artifact_store.load(f"{column}_power_transformer")
        col_values = to_pandas(df[[column]].fillna(0))
        df[column] = pt.transform(col_values).ravel()
        return df
