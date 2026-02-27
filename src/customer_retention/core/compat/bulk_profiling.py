from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as _pandas

from customer_retention.core.compat import as_spark_df


@dataclass
class PerColumnStats:
    null_count: int
    distinct_count: int
    most_common_value: Optional[str] = None
    most_common_frequency: Optional[int] = None


@dataclass
class NumericColumnStats:
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    zero_count: int = 0
    negative_count: int = 0
    inf_count: int = 0
    outlier_count_iqr: int = 0
    outlier_count_zscore: int = 0
    non_null_count: int = 0
    histogram_bins: list = field(default_factory=list)


@dataclass
class BulkStats:
    total_count: int
    columns: dict[str, PerColumnStats] = field(default_factory=dict)
    numeric: dict[str, NumericColumnStats] = field(default_factory=dict)


def bulk_nunique(df: Any, columns: list[str] | None = None) -> dict[str, int]:
    if columns is None:
        columns = list(df.columns)
    if not columns:
        return {}
    if hasattr(df, "to_spark"):
        return _spark_bulk_nunique(df, columns)
    return {col: int(df[col].nunique()) for col in columns}


def _spark_bulk_nunique(df: Any, columns: list[str]) -> dict[str, int]:
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(df)
    exprs = [F.countDistinct(F.col(c)).alias(f"__dist__{c}") for c in columns]
    row = spark_df.agg(*exprs).collect()[0]
    return {c: int(row[f"__dist__{c}"]) for c in columns}


def compute_bulk_stats(df: Any) -> BulkStats:
    if hasattr(df, "to_spark"):
        return _spark_bulk_stats(df)
    return _pandas_bulk_stats(df)


def _compute_mode(series: _pandas.Series) -> tuple[Optional[str], Optional[int]]:
    vc = series.value_counts()
    if len(vc) == 0:
        return None, None
    return str(vc.index[0]), int(vc.iloc[0])


def _compute_histogram(series: _pandas.Series) -> list:
    finite = series.dropna()
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return []
    arr = finite.to_numpy()
    lo, hi = float(arr.min()), float(arr.max())
    if lo >= hi:
        return []
    histogram, bin_edges = np.histogram(arr, bins=10, range=(lo, hi))
    return [
        (round(float(bin_edges[i]), 4), round(float(bin_edges[i + 1]), 4), int(histogram[i]))
        for i in range(len(histogram))
    ]


def _pandas_bulk_stats(df: _pandas.DataFrame) -> BulkStats:
    total_count = len(df)
    if total_count == 0 or len(df.columns) == 0:
        return BulkStats(total_count=total_count)

    null_counts = df.isnull().sum()
    distinct_counts = df.nunique()

    columns: dict[str, PerColumnStats] = {}
    for col in df.columns:
        mode_val, mode_freq = _compute_mode(df[col])
        columns[col] = PerColumnStats(
            null_count=int(null_counts[col]),
            distinct_count=int(distinct_counts[col]),
            most_common_value=mode_val,
            most_common_frequency=mode_freq,
        )

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric: dict[str, NumericColumnStats] = {}

    if numeric_cols:
        numeric_df = df[numeric_cols]
        desc = numeric_df.describe(percentiles=[0.25, 0.5, 0.75])

        try:
            skew_vals = numeric_df.skew()
        except Exception:
            skew_vals = _pandas.Series(dtype=float)

        try:
            kurt_vals = numeric_df.kurtosis()
        except Exception:
            kurt_vals = _pandas.Series(dtype=float)

        for col in numeric_cols:
            null_count = int(null_counts[col])
            non_null_count = total_count - null_count
            clean = numeric_df[col].dropna()
            if len(clean) == 0:
                numeric[col] = NumericColumnStats()
                continue

            mean_val = float(desc.at["mean", col])
            std_val = float(desc.at["std", col])
            min_val = float(desc.at["min", col])
            max_val = float(desc.at["max", col])
            q1 = float(desc.at["25%", col])
            median_val = float(desc.at["50%", col])
            q3 = float(desc.at["75%", col])

            skew = float(skew_vals[col]) if col in skew_vals.index and not _pandas.isna(skew_vals[col]) else None
            kurt = float(kurt_vals[col]) if col in kurt_vals.index and not _pandas.isna(kurt_vals[col]) else None

            zero_count = int((clean == 0).sum())
            negative_count = int((clean < 0).sum())
            inf_count = int(((clean == float("inf")) | (clean == float("-inf"))).sum())

            iqr = q3 - q1
            outlier_iqr = int(((clean < (q1 - 1.5 * iqr)) | (clean > (q3 + 1.5 * iqr))).sum())

            if std_val > 0:
                z_scores = ((clean - mean_val) / std_val).abs()
                outlier_zscore = int((z_scores > 3).sum())
            else:
                outlier_zscore = 0

            histogram_bins = _compute_histogram(clean)

            numeric[col] = NumericColumnStats(
                mean=mean_val,
                std=std_val,
                min_val=min_val,
                max_val=max_val,
                q1=q1,
                median=median_val,
                q3=q3,
                skewness=skew,
                kurtosis=kurt,
                zero_count=zero_count,
                negative_count=negative_count,
                inf_count=inf_count,
                outlier_count_iqr=outlier_iqr,
                outlier_count_zscore=outlier_zscore,
                non_null_count=non_null_count,
                histogram_bins=histogram_bins,
            )

    return BulkStats(total_count=total_count, columns=columns, numeric=numeric)


def _spark_bulk_stats(df: Any) -> BulkStats:
    import pyspark.sql.functions as F  # noqa: N812
    from pyspark.sql.types import NumericType

    spark_df = as_spark_df(df)
    all_cols = [c for c in spark_df.columns]

    # --- Batch 1: count + null counts + distinct counts ---
    exprs: list[Any] = [F.count("*").alias("__total_count__")]
    for col in all_cols:
        exprs.append(F.sum(F.isnull(F.col(col)).cast("int")).alias(f"__null__{col}"))
        exprs.append(F.countDistinct(F.col(col)).alias(f"__dist__{col}"))

    row1 = spark_df.agg(*exprs).collect()[0]
    total_count = int(row1["__total_count__"])

    # --- Batch 1b: mode values for all columns ---
    mode_exprs: list[Any] = [F.mode(F.col(c)).alias(f"__mode__{c}") for c in all_cols]
    mode_row = spark_df.agg(*mode_exprs).collect()[0]

    # --- Batch 1c: mode counts ---
    count_exprs: list[Any] = []
    for col in all_cols:
        mode_val = mode_row[f"__mode__{col}"]
        if mode_val is not None:
            count_exprs.append(
                F.sum(F.when(F.col(col) == F.lit(mode_val), 1).otherwise(0)).alias(f"__mcount__{col}")
            )
        else:
            count_exprs.append(F.lit(0).alias(f"__mcount__{col}"))
    mode_count_row = spark_df.agg(*count_exprs).collect()[0]

    columns: dict[str, PerColumnStats] = {}
    for col in all_cols:
        mode_val = mode_row[f"__mode__{col}"]
        mode_freq = _safe_int(mode_count_row[f"__mcount__{col}"]) if mode_val is not None else None
        columns[col] = PerColumnStats(
            null_count=int(row1[f"__null__{col}"]),
            distinct_count=int(row1[f"__dist__{col}"]),
            most_common_value=str(mode_val) if mode_val is not None else None,
            most_common_frequency=mode_freq,
        )

    # --- Identify numeric columns from schema ---
    numeric_cols = [f.name for f in spark_df.schema.fields if isinstance(f.dataType, NumericType)]

    numeric: dict[str, NumericColumnStats] = {}
    if not numeric_cols:
        return BulkStats(total_count=total_count, columns=columns, numeric=numeric)

    # --- Batch 2: numeric descriptive stats ---
    exprs2: list[Any] = []
    for col in numeric_cols:
        c = F.col(col)
        exprs2.append(F.mean(c).alias(f"__mean__{col}"))
        exprs2.append(F.stddev(c).alias(f"__std__{col}"))
        exprs2.append(F.min(c).alias(f"__min__{col}"))
        exprs2.append(F.max(c).alias(f"__max__{col}"))
        exprs2.append(F.percentile_approx(c, 0.25).alias(f"__q1__{col}"))
        exprs2.append(F.percentile_approx(c, 0.5).alias(f"__med__{col}"))
        exprs2.append(F.percentile_approx(c, 0.75).alias(f"__q3__{col}"))
        exprs2.append(F.skewness(c).alias(f"__skew__{col}"))
        exprs2.append(F.kurtosis(c).alias(f"__kurt__{col}"))

    row2 = spark_df.agg(*exprs2).collect()[0]

    batch2_results: dict[str, dict[str, Any]] = {}
    for col in numeric_cols:
        mean_val = _safe_float(row2[f"__mean__{col}"])
        std_val = _safe_float(row2[f"__std__{col}"])
        q1 = _safe_float(row2[f"__q1__{col}"])
        q3 = _safe_float(row2[f"__q3__{col}"])
        batch2_results[col] = {
            "mean": mean_val,
            "std": std_val,
            "min_val": _safe_float(row2[f"__min__{col}"]),
            "max_val": _safe_float(row2[f"__max__{col}"]),
            "q1": q1,
            "median": _safe_float(row2[f"__med__{col}"]),
            "q3": q3,
            "skewness": _safe_float(row2[f"__skew__{col}"]),
            "kurtosis": _safe_float(row2[f"__kurt__{col}"]),
        }

    # --- Batch 3: counts (zero, negative, inf) + outliers ---
    exprs3: list[Any] = []
    for col in numeric_cols:
        c = F.col(col)
        exprs3.append(F.sum((c == 0).cast("int")).alias(f"__zero__{col}"))
        exprs3.append(F.sum((c < 0).cast("int")).alias(f"__neg__{col}"))
        exprs3.append(F.sum(((c == float("inf")) | (c == float("-inf"))).cast("int")).alias(f"__inf__{col}"))

        b2 = batch2_results[col]
        q1 = b2["q1"]
        q3 = b2["q3"]
        mean_val = b2["mean"]
        std_val = b2["std"]

        if q1 is not None and q3 is not None:
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            exprs3.append(F.sum(((c < lower) | (c > upper)).cast("int")).alias(f"__oiqr__{col}"))
        else:
            exprs3.append(F.lit(0).alias(f"__oiqr__{col}"))

        if std_val is not None and std_val > 0 and mean_val is not None:
            z_expr = F.abs((c - mean_val) / std_val)
            exprs3.append(F.sum((z_expr > 3).cast("int")).alias(f"__ozscore__{col}"))
        else:
            exprs3.append(F.lit(0).alias(f"__ozscore__{col}"))

    row3 = spark_df.agg(*exprs3).collect()[0]

    # --- Batch 4: histogram bin counts for all numeric columns ---
    hist_exprs: list[Any] = []
    hist_cols: list[str] = []
    for col in numeric_cols:
        b2 = batch2_results[col]
        min_v, max_v = b2["min_val"], b2["max_val"]
        if min_v is None or max_v is None or min_v >= max_v:
            continue
        hist_cols.append(col)
        c = F.col(col)
        finite = c.isNotNull() & (c != float("inf")) & (c != float("-inf"))
        bin_width = (max_v - min_v) / 10
        for i in range(10):
            lo = min_v + i * bin_width
            hi = min_v + (i + 1) * bin_width if i < 9 else max_v
            cond = finite & (c >= lo) & (c <= hi if i == 9 else c < hi)
            hist_exprs.append(F.sum(cond.cast("int")).alias(f"__hist_{i}__{col}"))

    hist_row = None
    if hist_exprs:
        hist_row = spark_df.agg(*hist_exprs).collect()[0]

    for col in numeric_cols:
        b2 = batch2_results[col]
        null_count = columns[col].null_count
        non_null_count = total_count - null_count

        histogram_bins: list = []
        if col in hist_cols and hist_row is not None:
            min_v, max_v = b2["min_val"], b2["max_val"]
            bin_width = (max_v - min_v) / 10
            histogram_bins = []
            for i in range(10):
                lo = min_v + i * bin_width
                hi = min_v + (i + 1) * bin_width if i < 9 else max_v
                count = _safe_int(hist_row[f"__hist_{i}__{col}"])
                histogram_bins.append((round(lo, 4), round(hi, 4), count))

        numeric[col] = NumericColumnStats(
            mean=b2["mean"],
            std=b2["std"],
            min_val=b2["min_val"],
            max_val=b2["max_val"],
            q1=b2["q1"],
            median=b2["median"],
            q3=b2["q3"],
            skewness=b2["skewness"],
            kurtosis=b2["kurtosis"],
            zero_count=_safe_int(row3[f"__zero__{col}"]),
            negative_count=_safe_int(row3[f"__neg__{col}"]),
            inf_count=_safe_int(row3[f"__inf__{col}"]),
            outlier_count_iqr=_safe_int(row3[f"__oiqr__{col}"]),
            outlier_count_zscore=_safe_int(row3[f"__ozscore__{col}"]),
            non_null_count=non_null_count,
            histogram_bins=histogram_bins,
        )

    return BulkStats(total_count=total_count, columns=columns, numeric=numeric)


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int:
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0
