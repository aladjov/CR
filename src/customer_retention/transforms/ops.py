"""Stateless transformation functions.

Each function takes (df, column, **params) and returns df.
Uses core.compat for platform-agnostic DataFrame operations
(works on both pandas and pyspark.pandas).
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from customer_retention.core.compat import DataFrame, pd


def _requires_column(fn):
    @functools.wraps(fn)
    def wrapper(df: DataFrame, column: str, *args, **kwargs) -> DataFrame:
        if column not in df.columns:
            return df
        return fn(df, column, *args, **kwargs)
    return wrapper


@_requires_column
def apply_impute_null(df: DataFrame, column: str, *, value: Any = 0) -> DataFrame:
    if value == "median":
        df[column] = df[column].fillna(df[column].median())
    else:
        df[column] = df[column].fillna(value)
    return df


@_requires_column
def apply_cap_outlier(
    df: DataFrame, column: str, *, lower: float = 0, upper: float = 1_000_000
) -> DataFrame:
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def apply_type_cast(df: DataFrame, column: str, *, dtype: str = "float") -> DataFrame:
    if column not in df.columns:
        return df
    df[column] = df[column].astype(dtype)
    return df


def apply_drop_column(df: DataFrame, column: str) -> DataFrame:
    return df.drop(columns=[column], errors="ignore")


@_requires_column
def apply_winsorize(
    df: DataFrame, column: str, *, lower_bound: float = 0, upper_bound: float = 1_000_000
) -> DataFrame:
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df


def apply_segment_aware_cap(df: DataFrame, column: str, *, n_segments: int = 2) -> DataFrame:
    if column not in df.columns:
        return df
    from sklearn.cluster import KMeans

    valid = df[column].dropna()
    if len(valid) < n_segments:
        return df

    labels = KMeans(n_clusters=n_segments, random_state=42, n_init=10).fit_predict(
        valid.values.reshape(-1, 1)
    )
    df = df.copy()
    for seg in range(n_segments):
        mask = pd.Series(False, index=df.index)
        mask.iloc[valid.index[labels == seg]] = True
        seg_vals = df.loc[mask, column]
        q1, q3 = seg_vals.quantile(0.25), seg_vals.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df.loc[mask, column] = df.loc[mask, column].clip(lower=lower, upper=upper)
    return df


@_requires_column
def apply_log_transform(df: DataFrame, column: str) -> DataFrame:
    df[column] = np.log1p(df[column].clip(lower=0))
    return df


@_requires_column
def apply_sqrt_transform(df: DataFrame, column: str) -> DataFrame:
    df[column] = np.sqrt(df[column].clip(lower=0))
    return df


@_requires_column
def apply_zero_inflation_handling(df: DataFrame, column: str) -> DataFrame:
    df[f"{column}_is_zero"] = (df[column] == 0).astype(int)
    nonzero = df[column] != 0
    df.loc[nonzero, column] = np.log1p(df.loc[nonzero, column].clip(lower=0))
    return df


@_requires_column
def apply_cap_then_log(df: DataFrame, column: str) -> DataFrame:
    q99 = df[column].quantile(0.99)
    df[column] = np.log1p(df[column].clip(upper=q99).clip(lower=0))
    return df


@_requires_column
def apply_one_hot_encode(df: DataFrame, column: str) -> DataFrame:
    return pd.get_dummies(df, columns=[column], prefix=column)


def apply_feature_select(df: DataFrame, column: str) -> DataFrame:
    return df.drop(columns=[column], errors="ignore")


def apply_derived_ratio(
    df: DataFrame, column: str, *, numerator: str, denominator: str
) -> DataFrame:
    if numerator not in df.columns or denominator not in df.columns:
        return df
    df[column] = df[numerator] / df[denominator].replace(0, float("nan"))
    return df


def apply_derived_interaction(
    df: DataFrame, column: str, *, col_a: str, col_b: str
) -> DataFrame:
    if col_a not in df.columns or col_b not in df.columns:
        return df
    df[column] = df[col_a] * df[col_b]
    return df


def apply_derived_composite(
    df: DataFrame, column: str, *, columns: list[str]
) -> DataFrame:
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return df
    df[column] = df[valid].mean(axis=1)
    return df
