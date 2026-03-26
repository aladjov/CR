"""Stateless transformation functions.

Each function takes (df, column, **params) and returns df.
Uses core.compat for platform-agnostic DataFrame operations
(works on both pandas and pyspark.pandas).
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from customer_retention.core.compat import DataFrame, get_dummies


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

    valid_np = df[column].dropna().to_numpy()
    if len(valid_np) < n_segments:
        return df

    km = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
    km.fit(valid_np.reshape(-1, 1))
    centroids = km.cluster_centers_.ravel()
    labels = km.labels_
    seg_bounds = {}
    for seg in range(n_segments):
        seg_vals = valid_np[labels == seg]
        q1, q3 = np.percentile(seg_vals, 25), np.percentile(seg_vals, 75)
        iqr = q3 - q1
        seg_bounds[seg] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    sorted_idx = np.argsort(centroids)
    sorted_c = centroids[sorted_idx]
    mids = [(sorted_c[i] + sorted_c[i + 1]) / 2 for i in range(n_segments - 1)]
    boundaries = [-np.inf] + mids + [np.inf]
    df = df.copy()
    col = df[column]
    result = col
    for rank, orig_seg in enumerate(sorted_idx):
        in_seg = col.notna() & (col >= boundaries[rank]) & (col < boundaries[rank + 1])
        lower, upper = seg_bounds[orig_seg]
        result = col.clip(lower=lower, upper=upper).where(in_seg, result)
    df[column] = result
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
    df[f"{column}_log"] = df[column].where(df[column] == 0, np.log1p(df[column].clip(lower=0)))
    return df


@_requires_column
def apply_cap_then_log(df: DataFrame, column: str, *, _precomputed_q99: float | None = None) -> DataFrame:
    q99 = _precomputed_q99 if _precomputed_q99 is not None else df[column].quantile(0.99)
    df[column] = np.log1p(df[column].clip(upper=q99).clip(lower=0))
    return df


@_requires_column
def apply_one_hot_encode(df: DataFrame, column: str) -> DataFrame:
    return get_dummies(df, columns=[column], prefix=column)


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


# ── Batch ops (multi-column vectorized) ───────────────────────────


def apply_batch_log_transform(df: DataFrame, columns: list[str]) -> DataFrame:
    valid = [c for c in columns if c in df.columns]
    if valid:
        df[valid] = np.log1p(df[valid].clip(lower=0))
    return df


def apply_batch_sqrt_transform(df: DataFrame, columns: list[str]) -> DataFrame:
    valid = [c for c in columns if c in df.columns]
    if valid:
        df[valid] = np.sqrt(df[valid].clip(lower=0))
    return df


def apply_batch_zero_inflation(df: DataFrame, columns: list[str]) -> DataFrame:
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return df
    mask = df[valid] == 0
    log_values = df[valid].where(mask, np.log1p(df[valid].clip(lower=0)))
    for c in valid:
        df[f"{c}_is_zero"] = (df[c] == 0).astype(int)
        df[f"{c}_log"] = log_values[c]
    return df


def apply_batch_cap_then_log(df: DataFrame, columns_q99: list[tuple[str, float | None]]) -> DataFrame:
    valid = [(c, q) for c, q in columns_q99 if c in df.columns]
    if not valid:
        return df
    need_q99 = [c for c, q in valid if q is None]
    q99_map = {}
    if need_q99:
        q99s = df[need_q99].quantile(0.99)
        q99_map = {c: float(q99s[c]) for c in need_q99}
    for col, q99 in valid:
        q = q99 if q99 is not None else q99_map.get(col)
        if q is not None:
            df[col] = np.log1p(df[col].clip(upper=q).clip(lower=0))
    return df


def _apply_yj_column(series, lmbda: float, std_mean: float | None, std_scale: float | None, standardize: bool):
    filled = series.fillna(0)
    pos = filled >= 0
    with np.errstate(invalid="ignore"):
        pos_vals = np.log1p(filled) if abs(lmbda) < 1e-12 else ((filled + 1) ** lmbda - 1) / lmbda
        neg_vals = -np.log1p(-filled) if abs(lmbda - 2) < 1e-12 else -((-filled + 1) ** (2 - lmbda) - 1) / (2 - lmbda)
    result = pos_vals.where(pos, neg_vals)
    if standardize and std_mean is not None and std_scale is not None:
        result = (result - std_mean) / std_scale
    return result


def apply_batch_yeo_johnson(df: DataFrame, params_list: list[tuple[str, float, float | None, float | None, bool]]) -> DataFrame:
    for col, lmbda, std_mean, std_scale, standardize in params_list:
        if col in df.columns:
            df[col] = _apply_yj_column(df[col], lmbda, std_mean, std_scale, standardize)
    return df
