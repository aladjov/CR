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
from customer_retention.core.naming import sanitize_column_token
from customer_retention.parity import ApplyOpKind, apply_op


def _requires_column(fn):
    @functools.wraps(fn)
    def wrapper(df: DataFrame, column: str, *args, **kwargs) -> DataFrame:
        if column not in df.columns:
            return df
        return fn(df, column, *args, **kwargs)
    return wrapper


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"column", "value"})
@_requires_column
def apply_impute_null(df: DataFrame, column: str, *, value: Any = 0) -> DataFrame:
    if value == "median":
        df[column] = df[column].fillna(df[column].median())
    else:
        df[column] = df[column].fillna(value)
    return df


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"column", "lower", "upper"})
@_requires_column
def apply_cap_outlier(
    df: DataFrame, column: str, *, lower: float = 0, upper: float = 1_000_000
) -> DataFrame:
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"column", "dtype"})
def apply_type_cast(df: DataFrame, column: str, *, dtype: str = "float") -> DataFrame:
    if column not in df.columns:
        return df
    df[column] = df[column].astype(dtype)
    return df


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"column"})
def apply_drop_column(df: DataFrame, column: str) -> DataFrame:
    if column not in df.columns:
        return df
    return df.drop(columns=[column])


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"column", "lower_bound", "upper_bound"})
@_requires_column
def apply_winsorize(
    df: DataFrame, column: str, *, lower_bound: float = 0, upper_bound: float = 1_000_000
) -> DataFrame:
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"column", "n_segments"})
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


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"column"})
@_requires_column
def apply_log_transform(df: DataFrame, column: str) -> DataFrame:
    df[column] = np.log1p(df[column].clip(lower=0))
    return df


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"column"})
@_requires_column
def apply_sqrt_transform(df: DataFrame, column: str) -> DataFrame:
    df[column] = np.sqrt(df[column].clip(lower=0))
    return df


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"column"})
@_requires_column
def apply_zero_inflation_handling(df: DataFrame, column: str) -> DataFrame:
    df[f"{column}_is_zero"] = (df[column] == 0).astype(int)
    df[f"{column}_log"] = df[column].where(df[column] == 0, np.log1p(df[column].clip(lower=0)))
    return df


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"column"})
@_requires_column
def apply_cap_then_log(df: DataFrame, column: str, *, _precomputed_q99: float | None = None) -> DataFrame:
    q99 = _precomputed_q99 if _precomputed_q99 is not None else df[column].quantile(0.99)
    df[column] = np.log1p(df[column].clip(upper=q99).clip(lower=0))
    return df


@apply_op(kind=ApplyOpKind.GOLD_ENCODING, capture_kwargs={"column"})
@_requires_column
def apply_one_hot_encode(df: DataFrame, column: str) -> DataFrame:
    out = get_dummies(df, columns=[column], prefix=column)
    prefix = f"{column}_"
    # Build the rename map and detect collisions in one pass. Two raw
    # categories can sanitize to the same token (Salesforce strings often
    # carry whitespace/punctuation variants); without merging those, the
    # rename produces duplicate column names which pandas tolerates but
    # the Databricks/Delta downstream rejects with COLUMN_ALREADY_EXISTS.
    # Collisions are coalesced by summing the indicator columns and
    # clipping to {0,1}, preserving "row matches any of these raw values".
    grouped: dict[str, list] = {}
    for c in out.columns:
        if not (isinstance(c, str) and c.startswith(prefix)):
            continue
        token = c[len(prefix):]
        sanitized = sanitize_column_token(token)
        target = f"{prefix}{sanitized}"
        grouped.setdefault(target, []).append(c)
    drop_cols: list = []
    for target, sources in grouped.items():
        if len(sources) == 1:
            if sources[0] != target:
                out[target] = out[sources[0]]
                drop_cols.append(sources[0])
        else:
            out[target] = out[sources].sum(axis=1).clip(upper=1).astype(int)
            drop_cols.extend(s for s in sources if s != target)
    if drop_cols:
        out = out.drop(columns=list(set(drop_cols)))
    return out


@apply_op(kind=ApplyOpKind.GOLD_FEATURE_SPEC_GATE, capture_kwargs={"column"})
def apply_feature_select(df: DataFrame, column: str) -> DataFrame:
    if column not in df.columns:
        return df
    return df.drop(columns=[column])


@apply_op(kind=ApplyOpKind.SILVER_DERIVED_FEATURE, capture_kwargs={"column", "numerator", "denominator"})
def apply_derived_ratio(
    df: DataFrame, column: str, *, numerator: str, denominator: str
) -> DataFrame:
    if numerator not in df.columns or denominator not in df.columns:
        return df
    denom_series = df[denominator]
    if len(denom_series) > 0 and (denom_series.fillna(-1) == 0).all():
        raise ValueError(
            f"apply_derived_ratio: denominator {denominator!r} is all-zero "
            f"({len(denom_series)} rows). The resulting ratio {column!r} would "
            f"be NaN everywhere and median-imputed to a constant downstream. "
            f"Fix: remove this ratio from recommendations, or use a denominator "
            f"that has non-zero values."
        )
    df[column] = df[numerator] / denom_series.replace(0, float("nan"))
    return df


@apply_op(kind=ApplyOpKind.SILVER_DERIVED_FEATURE, capture_kwargs={"column", "col_a", "col_b"})
def apply_derived_interaction(
    df: DataFrame, column: str, *, col_a: str, col_b: str
) -> DataFrame:
    if col_a not in df.columns or col_b not in df.columns:
        return df
    df[column] = df[col_a] * df[col_b]
    return df


@apply_op(kind=ApplyOpKind.SILVER_DERIVED_FEATURE, capture_kwargs={"column", "columns"})
def apply_derived_composite(
    df: DataFrame, column: str, *, columns: list[str]
) -> DataFrame:
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return df
    df[column] = df[valid].mean(axis=1)
    return df


def _to_datetime_safe(series):
    """Coerce mixed-type / mixed-tz datetime series to tz-naive datetime64.

    Mirrors landing's `safe_to_datetime`: strips tz, falls back to NaT on
    parse errors so the difference math below produces NaN (not a crash).
    """
    from customer_retention.core.compat import safe_to_datetime
    return safe_to_datetime(series, errors="coerce")


def _diff_days(anchor_series, source_series):
    """Days between two datetime series via the compat dispatcher.

    Native pandas timestamp subtraction returns a timedelta (where
    `.dt.days` works); pyspark.pandas returns integer seconds. Routing
    through `timedelta_to_days` keeps this helper backend-agnostic per
    the Coding_Practices.md compat-layer contract.
    """
    from customer_retention.core.compat import timedelta_to_days
    return timedelta_to_days(anchor_series - source_series)


@apply_op(kind=ApplyOpKind.SILVER_DERIVED_FEATURE, capture_kwargs={"column", "source", "anchor_column"})
def apply_derived_recency(
    df: DataFrame, column: str, *, source: str, anchor_column: str = "as_of_date"
) -> DataFrame:
    """Days from ``source`` (a datetime column) to ``anchor_column``.

    Emits NaN when either side is NaT. Mirrors NB04's
    ``temporal_analyzer.recommend_features`` recency rec.
    """
    if source not in df.columns or anchor_column not in df.columns:
        return df
    df[column] = _diff_days(_to_datetime_safe(df[anchor_column]),
                            _to_datetime_safe(df[source]))
    return df


@apply_op(kind=ApplyOpKind.SILVER_DERIVED_FEATURE, capture_kwargs={"column", "col_a", "col_b"})
def apply_derived_duration(
    df: DataFrame, column: str, *, col_a: str, col_b: str
) -> DataFrame:
    """Days from ``col_a`` to ``col_b`` (signed: negative when ``col_b``
    precedes ``col_a``)."""
    if col_a not in df.columns or col_b not in df.columns:
        return df
    df[column] = _diff_days(_to_datetime_safe(df[col_b]),
                            _to_datetime_safe(df[col_a]))
    return df


@apply_op(kind=ApplyOpKind.SILVER_DERIVED_FEATURE, capture_kwargs={"column", "source"})
def apply_derived_cyclical(
    df: DataFrame, column: str, *, source: str
) -> DataFrame:
    """Emit ``<source>_month_sin`` and ``<source>_month_cos`` from the
    month component of ``source``. ``column`` is the umbrella
    ``<source>_month_sin_cos`` name carried by the recommendation; the
    actual persisted columns are the sin/cos pair."""
    if source not in df.columns:
        return df
    src_dt = _to_datetime_safe(df[source])
    df[f"{source}_month_sin"] = np.sin(2 * np.pi * src_dt.dt.month / 12)
    df[f"{source}_month_cos"] = np.cos(2 * np.pi * src_dt.dt.month / 12)
    return df


@apply_op(kind=ApplyOpKind.SILVER_DERIVED_FEATURE, capture_kwargs={"column", "source", "anchor_column"})
def apply_derived_tenure(
    df: DataFrame, column: str, *, source: str, anchor_column: str = "as_of_date"
) -> DataFrame:
    """Years from ``source`` to ``anchor_column`` (datediff / 365.25)."""
    if source not in df.columns or anchor_column not in df.columns:
        return df
    df[column] = _diff_days(_to_datetime_safe(df[anchor_column]),
                            _to_datetime_safe(df[source])) / 365.25
    return df


@apply_op(kind=ApplyOpKind.SILVER_DERIVED_FEATURE, capture_kwargs={"column", "source"})
def apply_derived_extraction_is_weekend(
    df: DataFrame, column: str, *, source: str
) -> DataFrame:
    """Emit ``column`` (the ``<src>_is_weekend`` umbrella name) as a 0/1
    flag from the day-of-week of ``source``. Passthrough when ``column``
    is already in df (landing's ``derive_datetime_features`` produces it
    upstream for every ``datetime_derivation_sources`` entry)."""
    if column in df.columns:
        return df
    if source not in df.columns:
        return df
    src_dt = _to_datetime_safe(df[source])
    df[column] = (src_dt.dt.dayofweek >= 5).astype("float64")
    return df


# ── Batch ops (multi-column vectorized) ───────────────────────────


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"columns"})
def apply_batch_log_transform(df: DataFrame, columns: list[str]) -> DataFrame:
    valid = [c for c in columns if c in df.columns]
    if valid:
        df[valid] = np.log1p(df[valid].clip(lower=0))
    return df


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"columns"})
def apply_batch_sqrt_transform(df: DataFrame, columns: list[str]) -> DataFrame:
    valid = [c for c in columns if c in df.columns]
    if valid:
        df[valid] = np.sqrt(df[valid].clip(lower=0))
    return df


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"columns"})
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


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"columns_q99"})
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


@apply_op(kind=ApplyOpKind.GOLD_TRANSFORMATION, capture_kwargs={"params_list"})
def apply_batch_yeo_johnson(df: DataFrame, params_list: list[tuple[str, float, float | None, float | None, bool]]) -> DataFrame:
    for col, lmbda, std_mean, std_scale, standardize in params_list:
        if col in df.columns:
            df[col] = _apply_yj_column(df[col], lmbda, std_mean, std_scale, standardize)
    return df
