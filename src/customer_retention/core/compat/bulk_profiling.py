from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as _pandas


@dataclass
class PerColumnStats:
    null_count: int
    distinct_count: int


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


@dataclass
class BulkStats:
    total_count: int
    columns: dict[str, PerColumnStats] = field(default_factory=dict)
    numeric: dict[str, NumericColumnStats] = field(default_factory=dict)


def compute_bulk_stats(df: Any) -> BulkStats:
    if hasattr(df, "to_spark"):
        return _spark_bulk_stats(df)
    return _pandas_bulk_stats(df)


def _pandas_bulk_stats(df: _pandas.DataFrame) -> BulkStats:
    total_count = len(df)
    null_counts = df.isnull().sum()
    distinct_counts = df.nunique()

    columns: dict[str, PerColumnStats] = {}
    for col in df.columns:
        columns[col] = PerColumnStats(
            null_count=int(null_counts[col]),
            distinct_count=int(distinct_counts[col]),
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
            )

    return BulkStats(total_count=total_count, columns=columns, numeric=numeric)


def _spark_bulk_stats(df: Any) -> BulkStats:
    import pyspark.sql.functions as F  # noqa: N812
    from pyspark.sql.types import NumericType

    spark_df = df.to_spark()
    all_cols = [c for c in spark_df.columns]

    # --- Batch 1: count + null counts + distinct counts ---
    exprs: list[Any] = [F.count("*").alias("__total_count__")]
    for col in all_cols:
        exprs.append(F.sum(F.isnull(F.col(col)).cast("int")).alias(f"__null__{col}"))
        exprs.append(F.countDistinct(F.col(col)).alias(f"__dist__{col}"))

    row1 = spark_df.agg(*exprs).collect()[0]
    total_count = int(row1["__total_count__"])

    columns: dict[str, PerColumnStats] = {}
    for col in all_cols:
        columns[col] = PerColumnStats(
            null_count=int(row1[f"__null__{col}"]),
            distinct_count=int(row1[f"__dist__{col}"]),
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

    for col in numeric_cols:
        b2 = batch2_results[col]
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
