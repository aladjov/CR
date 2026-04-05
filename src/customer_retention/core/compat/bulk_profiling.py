from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as _pandas

from customer_retention.core.compat import as_spark_df, as_tz_naive
from customer_retention.core.compat.timing import log_timing, timed

logger = logging.getLogger(__name__)


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
class DatetimeColumnStats:
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    date_range_days: Optional[int] = None
    future_date_count: int = 0
    placeholder_count: int = 0
    weekend_count: int = 0


@dataclass
class CategoricalColumnStats:
    cardinality: int = 0
    cardinality_ratio: float = 0.0
    top_categories: list = field(default_factory=list)
    value_counts: dict = field(default_factory=dict)
    rare_category_count: int = 0
    rare_category_percentage: float = 0.0
    case_variations: list = field(default_factory=list)


@dataclass
class IdentifierColumnStats:
    is_unique: bool = False
    duplicate_count: int = 0
    length_min: Optional[int] = None
    length_max: Optional[int] = None
    length_mode: Optional[int] = None


@dataclass
class BinaryColumnStats:
    true_count: int = 0
    false_count: int = 0
    true_percentage: float = 0.0
    balance_ratio: float = 0.0
    is_boolean: bool = False


@dataclass
class TextColumnStats:
    length_min: int = 0
    length_max: int = 0
    length_mean: float = 0.0
    length_median: float = 0.0
    empty_count: int = 0
    empty_percentage: float = 0.0
    word_count_mean: float = 0.0
    contains_digits_pct: float = 0.0
    contains_special_pct: float = 0.0
    pii_email_count: int = 0
    pii_phone_count: int = 0
    pii_ssn_count: int = 0
    pii_cc_count: int = 0


@dataclass
class TypedBulkStats:
    datetime: dict[str, DatetimeColumnStats] = field(default_factory=dict)
    categorical: dict[str, CategoricalColumnStats] = field(default_factory=dict)
    identifier: dict[str, IdentifierColumnStats] = field(default_factory=dict)
    binary: dict[str, BinaryColumnStats] = field(default_factory=dict)
    text: dict[str, TextColumnStats] = field(default_factory=dict)


@dataclass
class DatetimeDiscoveryCandidateStats:
    min_date: Any = None
    max_date: Any = None
    coverage: float = 0.0
    future_fraction: float = 0.0


@dataclass
class BulkStats:
    total_count: int
    columns: dict[str, PerColumnStats] = field(default_factory=dict)
    numeric: dict[str, NumericColumnStats] = field(default_factory=dict)


def bulk_datetime_discovery_stats(
    df: Any, columns: list[str],
) -> dict[str, DatetimeDiscoveryCandidateStats]:
    if not columns:
        return {}
    if hasattr(df, "to_spark"):
        return _spark_bulk_datetime_discovery(df, columns)
    return _pandas_bulk_datetime_discovery(df, columns)


def _pandas_bulk_datetime_discovery(
    df: Any, columns: list[str],
) -> dict[str, DatetimeDiscoveryCandidateStats]:
    now = _pandas.Timestamp.now()
    total = len(df)
    result: dict[str, DatetimeDiscoveryCandidateStats] = {}
    for col in columns:
        series = as_tz_naive(_pandas.to_datetime(df[col], errors="coerce"))
        non_null = series.notna().sum()
        coverage = float(non_null / total) if total > 0 else 0.0
        clean = series.dropna()
        future_frac = float((clean > now).mean()) if len(clean) > 0 else 0.0
        result[col] = DatetimeDiscoveryCandidateStats(
            min_date=clean.min() if len(clean) > 0 else None,
            max_date=clean.max() if len(clean) > 0 else None,
            coverage=coverage, future_fraction=future_frac,
        )
    return result


def _spark_bulk_datetime_discovery(
    df: Any, columns: list[str],
) -> dict[str, DatetimeDiscoveryCandidateStats]:
    import pyspark.sql.functions as F  # noqa: N812
    spark_df = as_spark_df(df)
    now = _pandas.Timestamp.now()
    total = spark_df.count()
    exprs: list[Any] = []
    for c in columns:
        col_expr = F.expr(f"try_to_timestamp(`{c}`)")
        exprs.extend([
            F.min(col_expr).alias(f"__min__{c}"),
            F.max(col_expr).alias(f"__max__{c}"),
            F.count(col_expr).alias(f"__cnt__{c}"),
            F.sum(F.when(col_expr > F.lit(now), 1).otherwise(0)).alias(f"__fut__{c}"),
        ])
    row = spark_df.agg(*exprs).collect()[0]
    result: dict[str, DatetimeDiscoveryCandidateStats] = {}
    for c in columns:
        cnt = int(row[f"__cnt__{c}"] or 0)
        fut = int(row[f"__fut__{c}"] or 0)
        result[c] = DatetimeDiscoveryCandidateStats(
            min_date=row[f"__min__{c}"], max_date=row[f"__max__{c}"],
            coverage=float(cnt / total) if total > 0 else 0.0,
            future_fraction=float(fut / cnt) if cnt > 0 else 0.0,
        )
    return result


def bulk_future_fractions(
    df: Any, reference_col: str, check_cols: list[str],
) -> dict[str, float]:
    if not check_cols:
        return {}
    valid = [c for c in check_cols if c in df.columns and c != reference_col]
    if not valid:
        return {}
    if reference_col not in df.columns:
        return {}
    if hasattr(df, "to_spark"):
        return _spark_bulk_future_fractions(df, reference_col, valid)
    return _pandas_bulk_future_fractions(df, reference_col, valid)


def _pandas_bulk_future_fractions(
    df: Any, reference_col: str, check_cols: list[str],
) -> dict[str, float]:
    ref = _pandas.to_datetime(df[reference_col], errors="coerce")
    total = len(df)
    result: dict[str, float] = {}
    for c in check_cols:
        parsed = _pandas.to_datetime(df[c], errors="coerce")
        if total == 0:
            result[c] = 0.0
        else:
            result[c] = float((parsed > ref).sum()) / total
    return result


def _spark_bulk_future_fractions(
    df: Any, reference_col: str, check_cols: list[str],
) -> dict[str, float]:
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(df)
    ref = F.expr(f"try_to_timestamp(`{reference_col}`)")
    exprs: list[Any] = [F.count(F.lit(1)).alias("__total__")]
    for c in check_cols:
        col_ts = F.expr(f"try_to_timestamp(`{c}`)")
        exprs.append(
            F.sum(F.when(col_ts > ref, 1).otherwise(0)).alias(f"__fut__{c}")
        )
    row = spark_df.agg(*exprs).collect()[0]
    total = int(row["__total__"])
    result: dict[str, float] = {}
    for c in check_cols:
        fut = int(row[f"__fut__{c}"] or 0)
        result[c] = float(fut / total) if total > 0 else 0.0
    return result


@dataclass
class HistogramData:
    bin_edges: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)

    @property
    def bin_centers(self) -> list[float]:
        return [
            (self.bin_edges[i] + self.bin_edges[i + 1]) / 2
            for i in range(len(self.counts))
        ]


def bulk_histogram(
    df: Any, column: str, nbins: int = 20,
) -> HistogramData:
    if column not in df.columns:
        return HistogramData()
    if hasattr(df, "to_spark"):
        return _spark_bulk_histogram(df, column, nbins)
    return _pandas_bulk_histogram(df, column, nbins)


def _pandas_bulk_histogram(
    df: Any, column: str, nbins: int,
) -> HistogramData:
    series = df[column].dropna()
    finite = series[np.isfinite(series)]
    if len(finite) == 0:
        return HistogramData()
    arr = finite.to_numpy()
    lo, hi = float(arr.min()), float(arr.max())
    if lo >= hi:
        return HistogramData()
    counts_arr, edges_arr = np.histogram(arr, bins=nbins, range=(lo, hi))
    return HistogramData(
        bin_edges=[round(float(e), 6) for e in edges_arr],
        counts=[int(c) for c in counts_arr],
    )


def _spark_bulk_histogram(
    df: Any, column: str, nbins: int,
) -> HistogramData:
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(df)
    c = F.col(column)
    finite = c.isNotNull() & (c != float("inf")) & (c != float("-inf"))
    bounds = spark_df.agg(
        F.min(F.when(finite, c)).alias("__lo__"),
        F.max(F.when(finite, c)).alias("__hi__"),
    ).collect()[0]
    lo, hi = bounds["__lo__"], bounds["__hi__"]
    if lo is None or hi is None or float(lo) >= float(hi):
        return HistogramData()
    lo, hi = float(lo), float(hi)
    bin_width = (hi - lo) / nbins
    edges = [lo + i * bin_width for i in range(nbins)] + [hi]
    exprs: list[Any] = []
    for i in range(nbins):
        b_lo = edges[i]
        b_hi = edges[i + 1]
        cond = finite & (c >= b_lo) & (c <= b_hi if i == nbins - 1 else c < b_hi)
        exprs.append(F.sum(cond.cast("int")).alias(f"__hbin_{i}__"))
    row = spark_df.agg(*exprs).collect()[0]
    counts = [_safe_int(row[f"__hbin_{i}__"]) for i in range(nbins)]
    return HistogramData(
        bin_edges=[round(e, 6) for e in edges],
        counts=counts,
    )


def bulk_histograms(
    df: Any, columns: list[str], nbins: int = 20,
) -> dict[str, HistogramData]:
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return {}
    if hasattr(df, "to_spark"):
        return _spark_bulk_histograms(df, valid, nbins)
    return _pandas_bulk_histograms(df, valid, nbins)


def _pandas_bulk_histograms(
    df: Any, columns: list[str], nbins: int,
) -> dict[str, HistogramData]:
    result: dict[str, HistogramData] = {}
    numeric = df[columns].select_dtypes(include=[np.number])
    for col in columns:
        if col not in numeric.columns:
            result[col] = HistogramData()
            continue
        arr = numeric[col].dropna().to_numpy()
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0 or float(finite.min()) >= float(finite.max()):
            result[col] = HistogramData()
            continue
        lo, hi = float(finite.min()), float(finite.max())
        counts_arr, edges_arr = np.histogram(finite, bins=nbins, range=(lo, hi))
        result[col] = HistogramData(
            bin_edges=[round(float(e), 6) for e in edges_arr],
            counts=[int(c) for c in counts_arr],
        )
    return result


def _spark_bulk_histograms(
    df: Any, columns: list[str], nbins: int,
) -> dict[str, HistogramData]:
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(df)

    # Batch 1: bounds for all columns in one Spark job
    bound_exprs: list[Any] = []
    for col in columns:
        c = F.col(col)
        fin = c.isNotNull() & (c != float("inf")) & (c != float("-inf"))
        bound_exprs.append(F.min(F.when(fin, c)).alias(f"__lo__{col}"))
        bound_exprs.append(F.max(F.when(fin, c)).alias(f"__hi__{col}"))
    bounds_row = spark_df.agg(*bound_exprs).collect()[0]

    # Build bin edges per column, skip degenerate
    col_edges: dict[str, list[float]] = {}
    result: dict[str, HistogramData] = {}
    for col in columns:
        lo_v, hi_v = bounds_row[f"__lo__{col}"], bounds_row[f"__hi__{col}"]
        if lo_v is None or hi_v is None or float(lo_v) >= float(hi_v):
            result[col] = HistogramData()
            continue
        lo, hi = float(lo_v), float(hi_v)
        bw = (hi - lo) / nbins
        col_edges[col] = [lo + i * bw for i in range(nbins)] + [hi]

    if not col_edges:
        return result

    # Batch 2: histogram counts for all remaining columns in one Spark job
    # Process in chunks to avoid excessively wide agg expressions
    chunk_size = max(1, 200 // nbins)
    edge_items = list(col_edges.items())
    for chunk_start in range(0, len(edge_items), chunk_size):
        chunk = edge_items[chunk_start:chunk_start + chunk_size]
        hist_exprs: list[Any] = []
        for col, edges in chunk:
            c = F.col(col)
            fin = c.isNotNull() & (c != float("inf")) & (c != float("-inf"))
            for i in range(nbins):
                b_lo, b_hi = edges[i], edges[i + 1]
                cond = fin & (c >= b_lo) & (c <= b_hi if i == nbins - 1 else c < b_hi)
                hist_exprs.append(F.sum(cond.cast("int")).alias(f"__hb_{col}_{i}__"))
        row = spark_df.agg(*hist_exprs).collect()[0]
        for col, edges in chunk:
            counts = [_safe_int(row[f"__hb_{col}_{i}__"]) for i in range(nbins)]
            result[col] = HistogramData(
                bin_edges=[round(e, 6) for e in edges],
                counts=counts,
            )
    return result


def bulk_monthly_counts(
    df: Any, column: str,
) -> list[tuple[str, int]]:
    if column not in df.columns:
        return []
    if hasattr(df, "to_spark"):
        return _spark_bulk_monthly_counts(df, column)
    return _pandas_bulk_monthly_counts(df, column)


def _pandas_bulk_monthly_counts(
    df: Any, column: str,
) -> list[tuple[str, int]]:
    from customer_retention.core.compat import safe_to_datetime

    dates = safe_to_datetime(df[column], errors="coerce").dropna()
    if len(dates) == 0:
        return []
    tz_free = dates.dt.tz_localize(None) if dates.dt.tz is not None else dates
    counts = tz_free.dt.strftime("%Y-%m").value_counts().sort_index()
    return [(str(k), int(v)) for k, v in zip(counts.index, counts.values)]


def _spark_bulk_monthly_counts(
    df: Any, column: str,
) -> list[tuple[str, int]]:
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(df)
    c = F.expr(f"try_to_timestamp(`{column}`)")
    month_col = F.date_format(c, "yyyy-MM")
    result = (
        spark_df
        .filter(c.isNotNull())
        .groupBy(month_col.alias("month"))
        .agg(F.count(F.lit(1)).alias("cnt"))
        .orderBy("month")
        .collect()
    )
    return [(str(row["month"]), int(row["cnt"])) for row in result]


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


def bulk_null_counts(
    df: Any, columns: list[str] | None = None, progress_fn: Any = None,
) -> dict[str, int]:
    valid = list(df.columns) if columns is None else [c for c in columns if c in df.columns]
    if not valid:
        return {}
    log = progress_fn or (lambda msg: None)
    if hasattr(df, "to_spark"):
        return _spark_bulk_null_counts(df, valid, log)
    result = {c: int(v) for c, v in df[valid].isnull().sum().items()}
    log(f"    null-count: {len(result)} columns (pandas)")
    return result


def _spark_bulk_null_counts(df: Any, columns: list[str], log: Any = None) -> dict[str, int]:
    import time as _time

    import pyspark.sql.functions as F  # noqa: N812

    log = log or (lambda msg: None)
    spark_df = as_spark_df(df)
    result: dict[str, int] = {}
    _BATCH = 200
    total = (len(columns) + _BATCH - 1) // _BATCH
    t0 = _time.monotonic()
    for batch_idx, start in enumerate(range(0, len(columns), _BATCH)):
        batch = columns[start : start + _BATCH]
        exprs = [F.coalesce(F.sum(F.isnull(F.col(c)).cast("int")), F.lit(0)).alias(f"__null__{c}") for c in batch]
        row = spark_df.agg(*exprs).collect()[0]
        for c in batch:
            result[c] = _safe_int(row[f"__null__{c}"])
        log(f"    null-count batch {batch_idx + 1}/{total} ({len(batch)} cols, {_time.monotonic() - t0:.0f}s)")
    return result


@timed(label="compute_bulk_stats")
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

    with log_timing("_pandas_bulk null_counts", logger):
        null_counts = df.isnull().sum()

    with log_timing("_pandas_bulk nunique", logger):
        distinct_counts = df.nunique()

    with log_timing("_pandas_bulk mode loop", logger, cols=len(df.columns)):
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
        with log_timing("_pandas_bulk describe+skew+kurt", logger):
            numeric_df = df[numeric_cols]
            desc = numeric_df.describe(percentiles=[0.25, 0.5, 0.75])

            try:
                skew_vals = numeric_df.skew()
            except (ValueError, TypeError) as exc:
                logger.debug("Skew computation failed: %s", exc)
                skew_vals = _pandas.Series(dtype=float)

            try:
                kurt_vals = numeric_df.kurtosis()
            except (ValueError, TypeError) as exc:
                logger.debug("Kurtosis computation failed: %s", exc)
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
    with log_timing("spark_bulk batch1 (count/null/distinct)", logger, cols=len(all_cols)):
        exprs: list[Any] = [F.count("*").alias("__total_count__")]
        for col in all_cols:
            exprs.append(F.coalesce(F.sum(F.isnull(F.col(col)).cast("int")), F.lit(0)).alias(f"__null__{col}"))
            exprs.append(F.coalesce(F.approx_count_distinct(F.col(col)), F.lit(0)).alias(f"__dist__{col}"))
        row1 = spark_df.agg(*exprs).collect()[0]
    total_count = int(row1["__total_count__"])

    # --- Batch 1b: mode values for all columns ---
    with log_timing("spark_bulk batch1b (mode)", logger, cols=len(all_cols)):
        mode_exprs: list[Any] = [F.mode(F.col(c)).alias(f"__mode__{c}") for c in all_cols]
        mode_row = spark_df.agg(*mode_exprs).collect()[0]

    # --- Batch 1c: mode counts ---
    with log_timing("spark_bulk batch1c (mode counts)", logger, cols=len(all_cols)):
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
            null_count=_safe_int(row1[f"__null__{col}"]),
            distinct_count=_safe_int(row1[f"__dist__{col}"]),
            most_common_value=str(mode_val) if mode_val is not None else None,
            most_common_frequency=mode_freq,
        )

    # --- Identify numeric columns from schema ---
    numeric_cols = [f.name for f in spark_df.schema.fields if isinstance(f.dataType, NumericType)]

    numeric: dict[str, NumericColumnStats] = {}
    if not numeric_cols:
        return BulkStats(total_count=total_count, columns=columns, numeric=numeric)

    # --- Batch 2: numeric descriptive stats ---
    with log_timing("spark_bulk batch2 (numeric stats)", logger, cols=len(numeric_cols)):
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
    with log_timing("spark_bulk batch3 (counts/outliers)", logger, cols=len(numeric_cols)):
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
    with log_timing("spark_bulk batch4 (histograms)", logger, cols=len(numeric_cols)):
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


@timed(label="compute_typed_bulk_stats")
def compute_typed_bulk_stats(
    df: Any,
    bulk: BulkStats,
    datetime_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    identifier_cols: list[str] | None = None,
    binary_cols: list[str] | None = None,
    text_cols: list[str] | None = None,
    cardinality_limit: int = 10000,
) -> TypedBulkStats:
    if hasattr(df, "to_spark"):
        return _spark_typed_bulk_stats(
            df, bulk,
            datetime_cols=datetime_cols or [],
            categorical_cols=categorical_cols or [],
            identifier_cols=identifier_cols or [],
            binary_cols=binary_cols or [],
            text_cols=text_cols or [],
            cardinality_limit=cardinality_limit,
        )
    return _pandas_typed_bulk_stats(
        df, bulk,
        datetime_cols=datetime_cols or [],
        categorical_cols=categorical_cols or [],
        identifier_cols=identifier_cols or [],
        binary_cols=binary_cols or [],
        text_cols=text_cols or [],
        cardinality_limit=cardinality_limit,
    )


_TRUE_VALUES = {1, 1.0, True, "1", "yes", "Yes", "YES", "true", "True", "TRUE", "y", "Y"}
_FALSE_VALUES = {0, 0.0, False, "0", "no", "No", "NO", "false", "False", "FALSE", "n", "N"}

_PLACEHOLDER_DATES = [
    _pandas.Timestamp("1970-01-01"),
    _pandas.Timestamp("1900-01-01"),
    _pandas.Timestamp("9999-12-31"),
]


def _pandas_typed_bulk_stats(
    df: _pandas.DataFrame,
    bulk: BulkStats,
    datetime_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    identifier_cols: list[str] | None = None,
    binary_cols: list[str] | None = None,
    text_cols: list[str] | None = None,
    cardinality_limit: int = 10000,
) -> TypedBulkStats:
    result = TypedBulkStats()

    with log_timing("pandas_typed datetime", logger, cols=len(datetime_cols or [])):
        for col in datetime_cols or []:
            result.datetime[col] = _pandas_datetime_stats(df[col])

    with log_timing("pandas_typed categorical", logger, cols=len(categorical_cols or [])):
        for col in categorical_cols or []:
            distinct = bulk.columns[col].distinct_count if col in bulk.columns else 0
            non_null = bulk.total_count - (bulk.columns[col].null_count if col in bulk.columns else 0)
            result.categorical[col] = _pandas_categorical_stats(
                df[col], distinct, non_null, cardinality_limit
            )

    with log_timing("pandas_typed identifier", logger, cols=len(identifier_cols or [])):
        for col in identifier_cols or []:
            distinct = bulk.columns[col].distinct_count if col in bulk.columns else 0
            non_null = bulk.total_count - (bulk.columns[col].null_count if col in bulk.columns else 0)
            result.identifier[col] = _pandas_identifier_stats(df[col], distinct, non_null)

    with log_timing("pandas_typed binary", logger, cols=len(binary_cols or [])):
        for col in binary_cols or []:
            col_stats = bulk.columns.get(col)
            result.binary[col] = _pandas_binary_stats(df[col], col_stats)

    with log_timing("pandas_typed text", logger, cols=len(text_cols or [])):
        for col in text_cols or []:
            result.text[col] = _pandas_text_stats(df[col], bulk.total_count)

    return result


def _pandas_datetime_stats(series: _pandas.Series) -> DatetimeColumnStats:
    clean = series.dropna()
    if len(clean) == 0:
        return DatetimeColumnStats()

    try:
        if not _pandas.api.types.is_datetime64_any_dtype(clean):
            clean = _pandas.to_datetime(clean, errors="coerce").dropna()
            if len(clean) == 0:
                return DatetimeColumnStats()
    except (ValueError, TypeError, OverflowError) as exc:
        logger.debug("Datetime conversion failed for column: %s", exc)
        return DatetimeColumnStats()

    clean = as_tz_naive(clean)
    min_date = clean.min()
    max_date = clean.max()
    date_range_days = (max_date - min_date).days

    now = _pandas.Timestamp.now()
    future_date_count = int((clean > now).sum())

    placeholder_count = 0
    for pd_date in _PLACEHOLDER_DATES:
        placeholder_count += int((clean == pd_date).sum())

    try:
        weekend_count = int(clean.dt.dayofweek.isin([5, 6]).sum())
    except Exception:
        weekend_count = 0

    return DatetimeColumnStats(
        min_date=str(min_date),
        max_date=str(max_date),
        date_range_days=date_range_days,
        future_date_count=future_date_count,
        placeholder_count=placeholder_count,
        weekend_count=weekend_count,
    )


def _pandas_categorical_stats(
    series: _pandas.Series,
    distinct_count: int,
    non_null_count: int,
    cardinality_limit: int,
) -> CategoricalColumnStats:
    clean = series.dropna()
    if len(clean) == 0:
        return CategoricalColumnStats()

    cardinality = distinct_count
    cardinality_ratio = round(cardinality / non_null_count, 4) if non_null_count > 0 else 0.0

    if cardinality > cardinality_limit:
        return CategoricalColumnStats(
            cardinality=cardinality,
            cardinality_ratio=cardinality_ratio,
        )

    vc = clean.value_counts()
    vc_dict = {str(k): int(v) for k, v in vc.to_dict().items()}
    top_categories = [(str(k), int(v)) for k, v in list(vc.to_dict().items())[:10]]

    rare_threshold = len(clean) * 0.01
    rare_count = sum(1 for v in vc_dict.values() if v < rare_threshold)
    rare_rows = sum(v for v in vc_dict.values() if v < rare_threshold)
    rare_pct = round(rare_rows / len(clean) * 100, 2) if len(clean) > 0 else 0.0

    str_values = clean.astype(str)
    lower_map: dict[str, list[str]] = {}
    for val in str_values.drop_duplicates():
        key = str(val).lower()
        lower_map.setdefault(key, []).append(str(val))
    case_variations = [
        f"{' vs '.join(sorted(variants))}"
        for variants in lower_map.values()
        if len(variants) > 1
    ][:10]

    return CategoricalColumnStats(
        cardinality=cardinality,
        cardinality_ratio=cardinality_ratio,
        top_categories=top_categories,
        value_counts=vc_dict,
        rare_category_count=rare_count,
        rare_category_percentage=rare_pct,
        case_variations=case_variations,
    )


def _pandas_identifier_stats(
    series: _pandas.Series, distinct_count: int, non_null_count: int
) -> IdentifierColumnStats:
    is_unique = distinct_count == non_null_count and non_null_count > 0

    duplicated = series[series.duplicated(keep=False)]
    duplicate_count = int(duplicated.nunique())

    str_series = series.dropna().astype(str)
    lengths = str_series.str.len()

    length_min = int(lengths.min()) if len(lengths) > 0 else None
    length_max = int(lengths.max()) if len(lengths) > 0 else None
    mode_result = lengths.mode()
    length_mode = int(mode_result.iloc[0]) if len(mode_result) > 0 else None

    return IdentifierColumnStats(
        is_unique=is_unique,
        duplicate_count=duplicate_count,
        length_min=length_min,
        length_max=length_max,
        length_mode=length_mode,
    )


def _pandas_binary_stats(
    series: _pandas.Series, col_stats: Optional[PerColumnStats]
) -> BinaryColumnStats:
    clean = series.dropna()
    if len(clean) == 0:
        return BinaryColumnStats()

    vc = clean.value_counts()
    vc_dict = vc.to_dict()
    values_found = list(vc_dict.keys())

    true_count = int(sum(vc_dict.get(v, 0) for v in values_found if v in _TRUE_VALUES))
    false_count = int(sum(vc_dict.get(v, 0) for v in values_found if v in _FALSE_VALUES))

    if true_count == 0 and false_count == 0:
        vc_values = list(vc_dict.values())
        true_count = int(vc_values[0]) if len(vc_values) > 0 else 0
        false_count = int(vc_values[1]) if len(vc_values) > 1 else 0

    total = true_count + false_count
    true_pct = round(true_count / total * 100, 2) if total > 0 else 0.0

    balance = (
        round(max(true_count, false_count) / min(true_count, false_count), 2)
        if min(true_count, false_count) > 0
        else float("inf")
    )

    is_boolean = _pandas.api.types.is_bool_dtype(series)

    return BinaryColumnStats(
        true_count=true_count,
        false_count=false_count,
        true_percentage=true_pct,
        balance_ratio=balance,
        is_boolean=is_boolean,
    )


_EMAIL_RE = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
_PHONE_RE = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
_SSN_RE = r"\b\d{3}-\d{2}-\d{4}\b"
_CC_RE = r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"


def _pandas_text_stats(series: _pandas.Series, total_count: int) -> TextColumnStats:
    clean = series.dropna()
    if len(clean) == 0:
        return TextColumnStats()

    str_series = clean.astype(str)
    lengths = str_series.str.len()

    length_min = int(lengths.min())
    length_max = int(lengths.max())
    length_mean = round(float(lengths.mean()), 2)
    length_median = round(float(lengths.median()), 2)

    empty_count = int((str_series == "").sum())
    empty_pct = round(empty_count / total_count * 100, 2) if total_count > 0 else 0.0

    word_counts = str_series.str.split().str.len()
    word_count_mean = round(float(word_counts.mean()), 2) if len(word_counts) > 0 else 0.0

    n = len(str_series)
    digits_count = int(str_series.str.contains(r"\d", regex=True, na=False).sum())
    contains_digits_pct = round(digits_count / n * 100, 2) if n > 0 else 0.0

    special_count = int(
        str_series.str.contains(r'[!@#$%^&*(),.?":{}|<>]', regex=True, na=False).sum()
    )
    contains_special_pct = round(special_count / n * 100, 2) if n > 0 else 0.0

    pii_email = int(str_series.str.contains(_EMAIL_RE, regex=True, na=False).sum())
    pii_phone = int(str_series.str.contains(_PHONE_RE, regex=True, na=False).sum())
    pii_ssn = int(str_series.str.contains(_SSN_RE, regex=True, na=False).sum())
    pii_cc = int(str_series.str.contains(_CC_RE, regex=True, na=False).sum())

    return TextColumnStats(
        length_min=length_min,
        length_max=length_max,
        length_mean=length_mean,
        length_median=length_median,
        empty_count=empty_count,
        empty_percentage=empty_pct,
        word_count_mean=word_count_mean,
        contains_digits_pct=contains_digits_pct,
        contains_special_pct=contains_special_pct,
        pii_email_count=pii_email,
        pii_phone_count=pii_phone,
        pii_ssn_count=pii_ssn,
        pii_cc_count=pii_cc,
    )


def _spark_typed_bulk_stats(
    df: Any,
    bulk: BulkStats,
    datetime_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    identifier_cols: list[str] | None = None,
    binary_cols: list[str] | None = None,
    text_cols: list[str] | None = None,
    cardinality_limit: int = 10000,
) -> TypedBulkStats:
    import pyspark.sql.functions as F  # noqa: N812
    from pyspark.sql.types import DateType, StringType, TimestampNTZType, TimestampType

    spark_df = as_spark_df(df)
    result = TypedBulkStats()

    datetime_cols = datetime_cols or []
    categorical_cols = categorical_cols or []
    identifier_cols = identifier_cols or []
    binary_cols = binary_cols or []
    text_cols = text_cols or []

    _string_fields = {f.name for f in spark_df.schema.fields if isinstance(f.dataType, StringType)}
    _timestamp_fields = {f.name for f in spark_df.schema.fields if isinstance(f.dataType, (TimestampType, TimestampNTZType, DateType))}
    text_cols = [c for c in text_cols if c in _string_fields]
    identifier_cols = [c for c in identifier_cols if c in _string_fields]
    datetime_cols = [c for c in datetime_cols if c in _timestamp_fields]

    # --- Batch 5: Datetime stats (1 Spark job) ---
    if datetime_cols:
        with log_timing("spark_typed batch5 (datetime)", logger, cols=len(datetime_cols)):
            dt_exprs: list[Any] = []
            now = _pandas.Timestamp.now()
            for col in datetime_cols:
                c = F.col(col)
                dt_exprs.append(F.min(c).alias(f"__dtmin__{col}"))
                dt_exprs.append(F.max(c).alias(f"__dtmax__{col}"))
                dt_exprs.append(
                    F.sum(F.when(c > F.lit(now), 1).otherwise(0)).alias(f"__dtfut__{col}")
                )
                for i, pd_date in enumerate(_PLACEHOLDER_DATES):
                    dt_exprs.append(
                        F.sum(F.when(c == F.lit(pd_date), 1).otherwise(0)).alias(
                            f"__dtph{i}__{col}"
                        )
                    )
                dt_exprs.append(
                    F.sum(
                        F.when(F.dayofweek(c).isin([1, 7]), 1).otherwise(0)
                    ).alias(f"__dtwknd__{col}")
                )
            dt_row = spark_df.agg(*dt_exprs).collect()[0]

            for col in datetime_cols:
                min_val = dt_row[f"__dtmin__{col}"]
                max_val = dt_row[f"__dtmax__{col}"]
                if min_val is not None and max_val is not None:
                    min_ts = _pandas.Timestamp(min_val)
                    max_ts = _pandas.Timestamp(max_val)
                    date_range_days = (max_ts - min_ts).days
                    min_str = str(min_ts)
                    max_str = str(max_ts)
                else:
                    date_range_days = None
                    min_str = None
                    max_str = None
                ph_count = sum(
                    _safe_int(dt_row[f"__dtph{i}__{col}"])
                    for i in range(len(_PLACEHOLDER_DATES))
                )
                result.datetime[col] = DatetimeColumnStats(
                    min_date=min_str,
                    max_date=max_str,
                    date_range_days=date_range_days,
                    future_date_count=_safe_int(dt_row[f"__dtfut__{col}"]),
                    placeholder_count=ph_count,
                    weekend_count=_safe_int(dt_row[f"__dtwknd__{col}"]),
                )

    # --- Batch 6: String/text length stats (1 Spark job) ---
    str_cols = list(set(identifier_cols + text_cols))
    if str_cols:
        with log_timing("spark_typed batch6 (string stats)", logger, cols=len(str_cols)):
            str_exprs: list[Any] = []
            for col in str_cols:
                c = F.col(col)
                ln = F.length(c)
                str_exprs.append(F.min(ln).alias(f"__smin__{col}"))
                str_exprs.append(F.max(ln).alias(f"__smax__{col}"))
                str_exprs.append(F.mean(ln.cast("double")).alias(f"__smean__{col}"))
                str_exprs.append(
                    F.percentile_approx(ln, 0.5).alias(f"__smed__{col}")
                )
            for col in text_cols:
                c = F.col(col)
                str_exprs.append(
                    F.sum(F.when(c == "", 1).otherwise(0)).alias(f"__tempty__{col}")
                )
                str_exprs.append(
                    F.mean(F.size(F.split(c, r"\s+")).cast("double")).alias(
                        f"__twcm__{col}"
                    )
                )
                str_exprs.append(
                    F.sum(F.when(c.rlike(r"\d"), 1).otherwise(0)).alias(
                        f"__tdig__{col}"
                    )
                )
                str_exprs.append(
                    F.sum(
                        F.when(c.rlike(r'[!@#$%^&*(),.?":{}|<>]'), 1).otherwise(0)
                    ).alias(f"__tspec__{col}")
                )
                str_exprs.append(
                    F.sum(
                        F.when(
                            c.rlike(
                                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
                            ),
                            1,
                        ).otherwise(0)
                    ).alias(f"__temail__{col}")
                )
                str_exprs.append(
                    F.sum(
                        F.when(c.rlike(r"\d{3}[-.]?\d{3}[-.]?\d{4}"), 1).otherwise(0)
                    ).alias(f"__tphone__{col}")
                )
                str_exprs.append(
                    F.sum(
                        F.when(c.rlike(r"\d{3}-\d{2}-\d{4}"), 1).otherwise(0)
                    ).alias(f"__tssn__{col}")
                )
                str_exprs.append(
                    F.sum(
                        F.when(
                            c.rlike(r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"), 1
                        ).otherwise(0)
                    ).alias(f"__tcc__{col}")
                )
            str_row = spark_df.agg(*str_exprs).collect()[0]

        with log_timing("spark_typed identifier groupBy loop", logger, cols=len(identifier_cols)):
            for col in identifier_cols:
                distinct = bulk.columns[col].distinct_count if col in bulk.columns else 0
                null_count = bulk.columns[col].null_count if col in bulk.columns else 0
                non_null = bulk.total_count - null_count
                is_unique = distinct == non_null and non_null > 0
                dup_count = 0
                if not is_unique:
                    dup_row = (
                        spark_df.groupBy(col)
                        .count()
                        .filter(F.col("count") > 1)
                        .count()
                    )
                    dup_count = int(dup_row)
                result.identifier[col] = IdentifierColumnStats(
                    is_unique=is_unique,
                    duplicate_count=dup_count,
                    length_min=_safe_int(str_row[f"__smin__{col}"]) or None,
                    length_max=_safe_int(str_row[f"__smax__{col}"]) or None,
                    length_mode=None,
                )

        for col in text_cols:
            non_null = bulk.total_count - (
                bulk.columns[col].null_count if col in bulk.columns else 0
            )
            n = max(non_null, 1)
            result.text[col] = TextColumnStats(
                length_min=_safe_int(str_row[f"__smin__{col}"]),
                length_max=_safe_int(str_row[f"__smax__{col}"]),
                length_mean=round(_safe_float(str_row[f"__smean__{col}"]) or 0.0, 2),
                length_median=round(float(_safe_int(str_row[f"__smed__{col}"])), 2),
                empty_count=_safe_int(str_row[f"__tempty__{col}"]),
                empty_percentage=round(
                    _safe_int(str_row[f"__tempty__{col}"]) / bulk.total_count * 100, 2
                )
                if bulk.total_count > 0
                else 0.0,
                word_count_mean=round(
                    _safe_float(str_row[f"__twcm__{col}"]) or 0.0, 2
                ),
                contains_digits_pct=round(
                    _safe_int(str_row[f"__tdig__{col}"]) / n * 100, 2
                ),
                contains_special_pct=round(
                    _safe_int(str_row[f"__tspec__{col}"]) / n * 100, 2
                ),
                pii_email_count=_safe_int(str_row[f"__temail__{col}"]),
                pii_phone_count=_safe_int(str_row[f"__tphone__{col}"]),
                pii_ssn_count=_safe_int(str_row[f"__tssn__{col}"]),
                pii_cc_count=_safe_int(str_row[f"__tcc__{col}"]),
            )

    # --- Batch 7: Categorical value counts (1-2 Spark jobs) ---
    eligible_cat_cols = [
        c for c in (categorical_cols or [])
        if c in bulk.columns and bulk.columns[c].distinct_count <= cardinality_limit
    ]
    high_card_cat_cols = [
        c for c in (categorical_cols or [])
        if c in bulk.columns and bulk.columns[c].distinct_count > cardinality_limit
    ]

    with log_timing("spark_typed batch7 (categorical)", logger, eligible=len(eligible_cat_cols), high_card=len(high_card_cat_cols)):
        for col in high_card_cat_cols:
            distinct = bulk.columns[col].distinct_count
            non_null = bulk.total_count - bulk.columns[col].null_count
            result.categorical[col] = CategoricalColumnStats(
                cardinality=distinct,
                cardinality_ratio=round(distinct / non_null, 4) if non_null > 0 else 0.0,
            )

        if eligible_cat_cols:
            stack_parts = []
            for col in eligible_cat_cols:
                stack_parts.append(
                    spark_df.select(
                        F.lit(col).alias("__col_label__"),
                        F.col(col).cast("string").alias("__value__"),
                    ).filter(F.col("__value__").isNotNull())
                )

            stacked = stack_parts[0]
            for part in stack_parts[1:]:
                stacked = stacked.unionAll(part)

            vc_rows = (
                stacked.groupBy("__col_label__", "__value__")
                .count()
                .collect()
            )

            col_vc: dict[str, dict[str, int]] = {c: {} for c in eligible_cat_cols}
            for row in vc_rows:
                label = row["__col_label__"]
                if label in col_vc:
                    col_vc[label][row["__value__"]] = int(row["count"])

            # Case variations batch
            case_rows = (
                stacked.select(
                    "__col_label__",
                    F.col("__value__").alias("__orig__"),
                    F.lower(F.col("__value__")).alias("__lower__"),
                )
                .groupBy("__col_label__", "__lower__")
                .agg(F.collect_set("__orig__").alias("__variants__"))
                .filter(F.size("__variants__") > 1)
                .collect()
            )

            col_case_vars: dict[str, list[str]] = {c: [] for c in eligible_cat_cols}
            for row in case_rows:
                label = row["__col_label__"]
                if label in col_case_vars:
                    variants = sorted(row["__variants__"])
                    col_case_vars[label].append(" vs ".join(variants))

            for col in eligible_cat_cols:
                vc = col_vc[col]
                distinct = bulk.columns[col].distinct_count
                non_null = bulk.total_count - bulk.columns[col].null_count
                sorted_vc = sorted(vc.items(), key=lambda x: x[1], reverse=True)
                top_categories = [(k, v) for k, v in sorted_vc[:10]]

                rare_threshold = non_null * 0.01
                rare_count = sum(1 for v in vc.values() if v < rare_threshold)
                rare_rows = sum(v for v in vc.values() if v < rare_threshold)
                rare_pct = round(rare_rows / non_null * 100, 2) if non_null > 0 else 0.0

                result.categorical[col] = CategoricalColumnStats(
                    cardinality=distinct,
                    cardinality_ratio=round(distinct / non_null, 4) if non_null > 0 else 0.0,
                    top_categories=top_categories,
                    value_counts=vc,
                    rare_category_count=rare_count,
                    rare_category_percentage=rare_pct,
                    case_variations=col_case_vars.get(col, [])[:10],
                )

    # --- Binary stats (per-column groupBy + collect) ---
    with log_timing("spark_typed binary loop", logger, cols=len(binary_cols)):
        for col in binary_cols:
            clean_vc = (
                spark_df.select(col)
                .filter(F.col(col).isNotNull())
                .groupBy(col)
                .count()
                .collect()
            )
            vc_dict = {row[col]: int(row["count"]) for row in clean_vc}
            values_found = list(vc_dict.keys())

            true_count = int(sum(vc_dict.get(v, 0) for v in values_found if v in _TRUE_VALUES))
            false_count = int(sum(vc_dict.get(v, 0) for v in values_found if v in _FALSE_VALUES))
            if true_count == 0 and false_count == 0:
                vc_values = list(vc_dict.values())
                true_count = int(vc_values[0]) if len(vc_values) > 0 else 0
                false_count = int(vc_values[1]) if len(vc_values) > 1 else 0

            total = true_count + false_count
            true_pct = round(true_count / total * 100, 2) if total > 0 else 0.0
            balance = (
                round(max(true_count, false_count) / min(true_count, false_count), 2)
                if min(true_count, false_count) > 0
                else float("inf")
            )

            field_schema = None
            for f in spark_df.schema.fields:
                if f.name == col:
                    field_schema = f
                    break
            from pyspark.sql.types import BooleanType

            is_boolean = isinstance(field_schema.dataType, BooleanType) if field_schema else False

            result.binary[col] = BinaryColumnStats(
                true_count=true_count,
                false_count=false_count,
                true_percentage=true_pct,
                balance_ratio=balance,
                is_boolean=is_boolean,
            )

    return result


def _chunked(columns: list[str], size: int) -> list[list[str]]:
    """Split a list of column names into chunks of the given size."""
    return [columns[i:i + size] for i in range(0, len(columns), size)]


# ---------------------------------------------------------------------------
# Bulk range validation
# ---------------------------------------------------------------------------

@dataclass
class RangeValidationBulkResult:
    non_null_count: int = 0
    invalid_count: int = 0
    actual_min: Optional[float] = None
    actual_max: Optional[float] = None


def bulk_validate_ranges(
    df: Any,
    rules: dict[str, dict[str, Any]],
    chunk_size: int = 200,
) -> dict[str, RangeValidationBulkResult]:
    """Validate value ranges for many columns in batched Spark agg calls."""
    if not rules:
        return {}
    if hasattr(df, "to_spark"):
        return _spark_bulk_validate_ranges(df, rules, chunk_size)
    return _pandas_bulk_validate_ranges(df, rules)


def _pandas_bulk_validate_ranges(
    df: Any,
    rules: dict[str, dict[str, Any]],
) -> dict[str, RangeValidationBulkResult]:
    result: dict[str, RangeValidationBulkResult] = {}
    for col, rule in rules.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        total = len(series)
        if total == 0:
            result[col] = RangeValidationBulkResult()
            continue
        try:
            series_numeric = series.astype(float)
        except (TypeError, ValueError):
            continue
        rule_type = rule.get("type", "range")
        invalid = _count_invalid_pandas(series_numeric, rule_type, rule)
        result[col] = RangeValidationBulkResult(
            non_null_count=total,
            invalid_count=int(invalid),
            actual_min=float(series_numeric.min()),
            actual_max=float(series_numeric.max()),
        )
    return result


def _count_invalid_pandas(series: Any, rule_type: str, rule: dict) -> int:
    import pandas as _pd
    min_val = rule.get("min")
    max_val = rule.get("max")
    if rule_type == "percentage":
        mn = min_val if min_val is not None else 0
        mx = max_val if max_val is not None else 100
        return int(((series < mn) | (series > mx)).sum())
    if rule_type == "binary":
        valid_values = rule.get("valid_values", [0, 1])
        return int((~series.isin(valid_values)).sum())
    if rule_type == "non_negative":
        return int((series < 0).sum())
    if rule_type == "rate":
        return int(((series < 0) | (series > 1)).sum())
    # general range
    mask = _pd.Series(False, index=series.index)
    if min_val is not None:
        mask = mask | (series < min_val)
    if max_val is not None:
        mask = mask | (series > max_val)
    return int(mask.sum())


def _spark_bulk_validate_ranges(
    df: Any,
    rules: dict[str, dict[str, Any]],
    chunk_size: int,
) -> dict[str, RangeValidationBulkResult]:
    import pyspark.sql.functions as F  # noqa: N812
    from pyspark.sql.types import NumericType

    spark_df = as_spark_df(df)
    numeric_cols = {f.name for f in spark_df.schema.fields if isinstance(f.dataType, NumericType)}
    valid_rules = {c: r for c, r in rules.items() if c in spark_df.columns and c in numeric_cols}
    if not valid_rules:
        return {}

    cols = list(valid_rules.keys())
    result: dict[str, RangeValidationBulkResult] = {}

    for chunk in _chunked(cols, chunk_size):
        exprs: list[Any] = []
        for col in chunk:
            c = F.col(f"`{col}`")
            rule = valid_rules[col]
            rule_type = rule.get("type", "range")
            invalid_expr = _spark_invalid_expr(c, rule_type, rule, F)
            exprs.extend([
                F.count(c).alias(f"__cnt__{col}"),
                F.min(c).alias(f"__min__{col}"),
                F.max(c).alias(f"__max__{col}"),
                F.sum(invalid_expr.cast("int")).alias(f"__inv__{col}"),
            ])
        with log_timing(f"bulk_validate_ranges chunk ({len(chunk)} cols)", logger):
            row = spark_df.agg(*exprs).collect()[0]
        for col in chunk:
            result[col] = RangeValidationBulkResult(
                non_null_count=_safe_int(row[f"__cnt__{col}"]),
                invalid_count=_safe_int(row[f"__inv__{col}"]),
                actual_min=_safe_float(row[f"__min__{col}"]),
                actual_max=_safe_float(row[f"__max__{col}"]),
            )
    return result


def _spark_invalid_expr(c: Any, rule_type: str, rule: dict, F: Any) -> Any:  # noqa: N803
    """Build a Spark Column expression that is True for invalid (non-null) values."""
    min_val = rule.get("min")
    max_val = rule.get("max")
    if rule_type == "percentage":
        mn = min_val if min_val is not None else 0
        mx = max_val if max_val is not None else 100
        return c.isNotNull() & ((c < mn) | (c > mx))
    if rule_type == "binary":
        valid_values = rule.get("valid_values", [0, 1])
        return c.isNotNull() & (~c.isin(*valid_values))
    if rule_type == "non_negative":
        return c.isNotNull() & (c < 0)
    if rule_type == "rate":
        return c.isNotNull() & ((c < 0) | (c > 1))
    # general range
    parts = []
    if min_val is not None:
        parts.append(c < min_val)
    if max_val is not None:
        parts.append(c > max_val)
    if not parts:
        return F.lit(False)
    combined = parts[0]
    for p in parts[1:]:
        combined = combined | p
    return c.isNotNull() & combined


# ---------------------------------------------------------------------------
# Bulk distribution stats
# ---------------------------------------------------------------------------

@dataclass
class DistributionBulkResult:
    non_null_count: int = 0
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
    outlier_count_iqr: int = 0
    percentiles: dict[str, float] = field(default_factory=dict)


def bulk_distribution_stats(
    df: Any,
    columns: list[str],
    chunk_size: int = 200,
) -> dict[str, DistributionBulkResult]:
    """Compute distribution statistics for many numeric columns in batched Spark agg calls."""
    if not columns:
        return {}
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return {}
    if hasattr(df, "to_spark"):
        return _spark_bulk_distribution_stats(df, valid, chunk_size)
    return _pandas_bulk_distribution_stats(df, valid)


def _pandas_bulk_distribution_stats(
    df: Any,
    columns: list[str],
) -> dict[str, DistributionBulkResult]:
    result: dict[str, DistributionBulkResult] = {}
    numeric_df = df[columns]
    desc = numeric_df.describe()
    try:
        extra_q = numeric_df.quantile([0.01, 0.05, 0.10, 0.90, 0.95, 0.99])
    except Exception:
        extra_q = _pandas.DataFrame(index=[0.01, 0.05, 0.10, 0.90, 0.95, 0.99], columns=columns)
    try:
        skew_vals = numeric_df.skew()
    except Exception:
        skew_vals = _pandas.Series(0.0, index=columns)
    try:
        kurt_vals = numeric_df.kurtosis()
    except Exception:
        kurt_vals = _pandas.Series(0.0, index=columns)
    zero_counts = (numeric_df == 0).sum()
    neg_counts = (numeric_df < 0).sum()

    q1_all = desc.loc["25%"]
    q3_all = desc.loc["75%"]
    iqr_all = q3_all - q1_all
    outlier_counts = ((numeric_df < (q1_all - 1.5 * iqr_all)) | (numeric_df > (q3_all + 1.5 * iqr_all))).sum()

    for col in columns:
        count = int(desc.loc["count", col])
        if count == 0:
            result[col] = DistributionBulkResult()
            continue
        q1 = float(q1_all[col])
        q3 = float(q3_all[col])

        pct = {"p25": q1, "p50": float(desc.loc["50%", col]), "p75": q3}
        for label, idx in [("p1", 0.01), ("p5", 0.05), ("p10", 0.10),
                           ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]:
            try:
                pct[label] = float(extra_q.loc[idx, col])
            except Exception:
                pct[label] = 0.0

        result[col] = DistributionBulkResult(
            non_null_count=count,
            mean=float(desc.loc["mean", col]),
            std=float(desc.loc["std", col]),
            min_val=float(desc.loc["min", col]),
            max_val=float(desc.loc["max", col]),
            q1=q1, median=pct["p50"], q3=q3,
            skewness=float(skew_vals[col]),
            kurtosis=float(kurt_vals[col]),
            zero_count=int(zero_counts[col]),
            negative_count=int(neg_counts[col]),
            outlier_count_iqr=int(outlier_counts[col]),
            percentiles=pct,
        )
    return result


def _spark_bulk_distribution_stats(
    df: Any,
    columns: list[str],
    chunk_size: int,
) -> dict[str, DistributionBulkResult]:
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(df)
    result: dict[str, DistributionBulkResult] = {}
    pctiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    # Pass 1: basic stats + percentiles per chunk
    pass1: dict[str, dict[str, Any]] = {}
    for chunk in _chunked(columns, chunk_size):
        exprs: list[Any] = []
        for col in chunk:
            c = F.col(f"`{col}`")
            exprs.extend([
                F.count(c).alias(f"__cnt__{col}"),
                F.mean(c).alias(f"__avg__{col}"),
                F.stddev(c).alias(f"__std__{col}"),
                F.min(c).alias(f"__min__{col}"),
                F.max(c).alias(f"__max__{col}"),
                F.percentile_approx(c, pctiles).alias(f"__pct__{col}"),
                F.skewness(c).alias(f"__skw__{col}"),
                F.kurtosis(c).alias(f"__krt__{col}"),
                F.sum((c == 0).cast("int")).alias(f"__zer__{col}"),
                F.sum((c < 0).cast("int")).alias(f"__neg__{col}"),
            ])
        with log_timing(f"bulk_distribution_stats pass1 ({len(chunk)} cols)", logger):
            row = spark_df.agg(*exprs).collect()[0]
        for col in chunk:
            pct_arr = row[f"__pct__{col}"] or [None] * len(pctiles)
            pass1[col] = {
                "cnt": _safe_int(row[f"__cnt__{col}"]),
                "avg": _safe_float(row[f"__avg__{col}"]),
                "std": _safe_float(row[f"__std__{col}"]),
                "min": _safe_float(row[f"__min__{col}"]),
                "max": _safe_float(row[f"__max__{col}"]),
                "skw": _safe_float(row[f"__skw__{col}"]),
                "krt": _safe_float(row[f"__krt__{col}"]),
                "zer": _safe_int(row[f"__zer__{col}"]),
                "neg": _safe_int(row[f"__neg__{col}"]),
                "pct": pct_arr,
            }

    # Pass 2: outlier counts using q1/q3 from pass 1
    cols_with_iqr = [c for c in columns if pass1.get(c, {}).get("cnt", 0) > 0]
    outlier_counts: dict[str, int] = {}
    for chunk in _chunked(cols_with_iqr, chunk_size):
        exprs = []
        for col in chunk:
            p = pass1[col]
            pct_arr = p["pct"]
            q1_val = _safe_float(pct_arr[3]) if pct_arr[3] is not None else None
            q3_val = _safe_float(pct_arr[5]) if pct_arr[5] is not None else None
            if q1_val is None or q3_val is None:
                outlier_counts[col] = 0
                continue
            iqr = q3_val - q1_val
            lower = q1_val - 1.5 * iqr
            upper = q3_val + 1.5 * iqr
            c = F.col(f"`{col}`")
            exprs.append(
                F.sum(((c < lower) | (c > upper)).cast("int")).alias(f"__out__{col}")
            )
        if exprs:
            with log_timing(f"bulk_distribution_stats pass2 ({len(chunk)} cols)", logger):
                row2 = spark_df.agg(*exprs).collect()[0]
            for col in chunk:
                alias = f"__out__{col}"
                if alias in row2.asDict():
                    outlier_counts[col] = _safe_int(row2[alias])
                elif col not in outlier_counts:
                    outlier_counts[col] = 0

    # Assemble results
    pct_labels = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
    for col in columns:
        p = pass1.get(col)
        if not p or p["cnt"] == 0:
            result[col] = DistributionBulkResult()
            continue
        pct_arr = p["pct"]
        pct_dict = {}
        for i, label in enumerate(pct_labels):
            pct_dict[label] = _safe_float(pct_arr[i]) if i < len(pct_arr) else None

        result[col] = DistributionBulkResult(
            non_null_count=p["cnt"],
            mean=p["avg"],
            std=p["std"],
            min_val=p["min"],
            max_val=p["max"],
            q1=pct_dict.get("p25"),
            median=pct_dict.get("p50"),
            q3=pct_dict.get("p75"),
            skewness=p["skw"],
            kurtosis=p["krt"],
            zero_count=p["zer"],
            negative_count=p["neg"],
            outlier_count_iqr=outlier_counts.get(col, 0),
            percentiles=pct_dict,
        )
    return result


# ---------------------------------------------------------------------------
# Bulk categorical distribution stats
# ---------------------------------------------------------------------------

@dataclass
class CategoricalDistributionBulkResult:
    total_count: int = 0
    category_count: int = 0
    value_counts: dict[str, int] = field(default_factory=dict)
    imbalance_ratio: float = 0.0
    entropy: float = 0.0
    normalized_entropy: float = 0.0
    top1_concentration: float = 0.0
    top3_concentration: float = 0.0
    rare_category_count: int = 0
    rare_category_names: list[str] = field(default_factory=list)


def bulk_categorical_distribution_stats(
    df: Any, columns: list[str], top_n: int = 20, rare_threshold: float = 0.01,
) -> dict[str, CategoricalDistributionBulkResult]:
    """Compute categorical distribution stats for many columns in batched Spark jobs."""
    if not columns:
        return {}
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return {}
    if hasattr(df, "to_spark"):
        return _spark_bulk_categorical_stats(df, valid, top_n, rare_threshold)
    return _pandas_bulk_categorical_stats(df, valid, top_n, rare_threshold)


def _pandas_bulk_categorical_stats(
    df: Any, columns: list[str], top_n: int, rare_threshold: float,
) -> dict[str, CategoricalDistributionBulkResult]:
    result: dict[str, CategoricalDistributionBulkResult] = {}
    for col in columns:
        series = df[col].dropna()
        total = len(series)
        if total == 0:
            result[col] = CategoricalDistributionBulkResult()
            continue
        vc = series.value_counts()
        result[col] = _build_categorical_result(col, total, vc.to_dict(), top_n, rare_threshold)
    return result


def _spark_bulk_categorical_stats(
    df: Any, columns: list[str], top_n: int, rare_threshold: float,
) -> dict[str, CategoricalDistributionBulkResult]:
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(df)
    # Pass 1: non-null counts per column (batched agg)
    count_exprs = [F.count(F.col(f"`{c}`")).alias(f"__cnt__{c}") for c in columns]
    with log_timing(f"bulk_categorical_stats counts ({len(columns)} cols)", logger):
        count_row = spark_df.agg(*count_exprs).collect()[0]
    col_counts = {c: _safe_int(count_row[f"__cnt__{c}"]) for c in columns}

    # Pass 2: stack all columns, single groupBy for value counts
    stack_parts = []
    for c in columns:
        if col_counts[c] == 0:
            continue
        stack_parts.append(
            spark_df.select(
                F.lit(c).alias("__col__"),
                F.col(f"`{c}`").cast("string").alias("__val__"),
            ).filter(F.col("__val__").isNotNull())
        )
    if not stack_parts:
        return {c: CategoricalDistributionBulkResult() for c in columns}

    stacked = stack_parts[0]
    for part in stack_parts[1:]:
        stacked = stacked.unionAll(part)

    with log_timing(f"bulk_categorical_stats value_counts ({len(columns)} cols)", logger):
        vc_rows = stacked.groupBy("__col__", "__val__").count().collect()

    # Organize value counts per column
    col_vc: dict[str, dict[str, int]] = {c: {} for c in columns}
    for row in vc_rows:
        label = row["__col__"]
        if label in col_vc:
            col_vc[label][row["__val__"]] = int(row["count"])

    result: dict[str, CategoricalDistributionBulkResult] = {}
    for c in columns:
        total = col_counts[c]
        if total == 0:
            result[c] = CategoricalDistributionBulkResult()
            continue
        result[c] = _build_categorical_result(c, total, col_vc[c], top_n, rare_threshold)
    return result


def _build_categorical_result(
    col: str, total: int, vc_dict: dict[str, int],
    top_n: int, rare_threshold: float,
) -> CategoricalDistributionBulkResult:
    """Compute derived metrics from a value-counts dict (shared by pandas/spark paths)."""
    if not vc_dict:
        return CategoricalDistributionBulkResult()

    sorted_vc = sorted(vc_dict.items(), key=lambda x: x[1], reverse=True)
    category_count = len(sorted_vc)
    max_count = sorted_vc[0][1]
    min_count = sorted_vc[-1][1]
    imbalance_ratio = float(max_count / min_count) if min_count > 0 else float("inf")

    # Entropy
    counts_arr = np.array([c for _, c in sorted_vc], dtype=float)
    probs = counts_arr / total
    entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
    max_entropy = np.log2(category_count) if category_count > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    top1 = float(max_count / total * 100)
    top3_sum = sum(c for _, c in sorted_vc[:3])
    top3 = float(top3_sum / total * 100)

    rare_cutoff = total * rare_threshold
    rare_items = [(name, cnt) for name, cnt in sorted_vc if cnt < rare_cutoff]
    rare_count = len(rare_items)
    rare_names = [name for name, _ in rare_items[:10]]

    top_vc = dict(sorted_vc[:top_n])

    return CategoricalDistributionBulkResult(
        total_count=total, category_count=category_count,
        value_counts=top_vc, imbalance_ratio=imbalance_ratio,
        entropy=entropy, normalized_entropy=normalized_entropy,
        top1_concentration=top1, top3_concentration=top3,
        rare_category_count=rare_count, rare_category_names=rare_names,
    )


# ---------------------------------------------------------------------------
# Bulk datetime analysis stats
# ---------------------------------------------------------------------------

@dataclass
class DatetimeAnalysisBulkResult:
    total_count: int = 0
    null_count: int = 0
    min_date: Any = None
    max_date: Any = None
    span_days: int = 0
    monthly_counts: list[tuple[str, int]] = field(default_factory=list)
    dow_counts: list[int] = field(default_factory=lambda: [0] * 7)
    placeholder_count: int = 0


def bulk_datetime_analysis_stats(
    df: Any, columns: list[str],
) -> dict[str, DatetimeAnalysisBulkResult]:
    """Compute datetime stats for multiple columns in batched Spark agg calls."""
    if not columns:
        return {}
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return {}
    if hasattr(df, "to_spark"):
        return _spark_bulk_datetime_analysis(df, valid)
    return _pandas_bulk_datetime_analysis(df, valid)


def _pandas_bulk_datetime_analysis(
    df: Any, columns: list[str],
) -> dict[str, DatetimeAnalysisBulkResult]:
    result: dict[str, DatetimeAnalysisBulkResult] = {}
    for col in columns:
        parsed = _pandas.to_datetime(df[col], errors="coerce")
        total = len(parsed)
        null_c = int(parsed.isna().sum())
        clean = parsed.dropna()
        if len(clean) == 0:
            result[col] = DatetimeAnalysisBulkResult(total_count=total, null_count=null_c)
            continue
        mn, mx = clean.min(), clean.max()
        span = (mx - mn).days
        monthly = clean.dt.strftime("%Y-%m").value_counts().sort_index()
        monthly_list = list(zip(monthly.index.tolist(), monthly.values.tolist()))
        dow = clean.dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
        plc = int((clean < _pandas.Timestamp("2000-01-01")).sum())
        result[col] = DatetimeAnalysisBulkResult(
            total_count=total, null_count=null_c, min_date=mn, max_date=mx,
            span_days=span, monthly_counts=monthly_list,
            dow_counts=dow.values.tolist(), placeholder_count=plc,
        )
    return result


def _spark_bulk_datetime_analysis(
    df: Any, columns: list[str],
) -> dict[str, DatetimeAnalysisBulkResult]:
    import pyspark.sql.functions as F  # noqa: N812

    spark_df = as_spark_df(df)
    total = spark_df.count()

    # Pass 1: min, max, null count per column (batched agg)
    # Use try_to_timestamp: returns NULL for unparseable strings (e.g. '2024Q4')
    # instead of raising CAST_INVALID_INPUT
    exprs: list[Any] = []
    for c in columns:
        col_expr = F.expr(f"try_to_timestamp(`{c}`)")
        exprs.extend([
            F.min(col_expr).alias(f"__min__{c}"),
            F.max(col_expr).alias(f"__max__{c}"),
            F.sum(F.when(col_expr.isNull(), 1).otherwise(0).cast("int")).alias(f"__null__{c}"),
            F.sum(F.when(col_expr < F.lit("2000-01-01").cast("timestamp"), 1).otherwise(0).cast("int")).alias(f"__plc__{c}"),
        ])
    with log_timing(f"bulk_datetime_analysis pass1 ({len(columns)} cols)", logger):
        row = spark_df.agg(*exprs).collect()[0]

    basic: dict[str, dict[str, Any]] = {}
    for c in columns:
        mn = row[f"__min__{c}"]
        mx = row[f"__max__{c}"]
        null_c = _safe_int(row[f"__null__{c}"])
        plc = _safe_int(row[f"__plc__{c}"])
        span = (mx - mn).days if mn and mx else 0
        basic[c] = {"min": mn, "max": mx, "null": null_c, "span": span, "placeholder": plc}

    # Pass 2: stack all columns, groupBy month + dow
    stack_parts = []
    for c in columns:
        if basic[c]["min"] is None:
            continue
        col_ts = F.expr(f"try_to_timestamp(`{c}`)")
        stack_parts.append(
            spark_df.select(
                F.lit(c).alias("__col__"),
                F.date_format(col_ts, "yyyy-MM").alias("__month__"),
                F.dayofweek(col_ts).alias("__dow__"),
            ).filter(col_ts.isNotNull())
        )

    monthly_data: dict[str, list[tuple[str, int]]] = {c: [] for c in columns}
    dow_data: dict[str, list[int]] = {c: [0] * 7 for c in columns}

    if stack_parts:
        stacked = stack_parts[0]
        for p in stack_parts[1:]:
            stacked = stacked.unionAll(p)

        with log_timing(f"bulk_datetime_analysis monthly ({len(columns)} cols)", logger):
            month_rows = stacked.groupBy("__col__", "__month__").count().collect()
        for r in month_rows:
            col_label = r["__col__"]
            if col_label in monthly_data:
                monthly_data[col_label].append((r["__month__"], int(r["count"])))
        for c in monthly_data:
            monthly_data[c].sort(key=lambda x: x[0])

        with log_timing(f"bulk_datetime_analysis dow ({len(columns)} cols)", logger):
            dow_rows = stacked.groupBy("__col__", "__dow__").count().collect()
        for r in dow_rows:
            col_label = r["__col__"]
            if col_label in dow_data:
                # Spark dayofweek: 1=Sun..7=Sat → convert to 0=Mon..6=Sun
                spark_dow = int(r["__dow__"])
                py_dow = (spark_dow - 2) % 7
                dow_data[col_label][py_dow] = int(r["count"])

    result: dict[str, DatetimeAnalysisBulkResult] = {}
    for c in columns:
        b = basic[c]
        result[c] = DatetimeAnalysisBulkResult(
            total_count=total, null_count=b["null"],
            min_date=b["min"], max_date=b["max"], span_days=b["span"],
            monthly_counts=monthly_data.get(c, []),
            dow_counts=dow_data.get(c, [0] * 7),
            placeholder_count=b["placeholder"],
        )
    return result


def batch_adversarial_diffs(
    gold_df: Any, score_df: Any, entity_column: str,
    target_column: str, holdout_column: str, tolerance: float = 1e-6,
) -> tuple[int, dict[str, tuple[float, float, int]]]:
    from customer_retention.core.compat import _is_native_spark_df, _is_spark_pandas

    if _is_spark_pandas(gold_df) or _is_native_spark_df(gold_df):
        return _spark_adversarial_diffs(gold_df, score_df, entity_column, target_column, holdout_column, tolerance)
    return _pandas_adversarial_diffs(gold_df, score_df, entity_column, target_column, holdout_column, tolerance)


def _pandas_adversarial_diffs(
    gold_df: Any, score_df: Any, entity_column: str,
    target_column: str, holdout_column: str, tolerance: float,
) -> tuple[int, dict[str, tuple[float, float, int]]]:
    is_holdout = gold_df[target_column].isna() & gold_df[holdout_column].notna()
    gold_holdout = gold_df[is_holdout]
    if gold_holdout.empty:
        return 0, {}
    exclude = {entity_column, target_column, holdout_column}
    feature_cols = [
        c for c in gold_holdout.columns
        if c not in exclude and not c.startswith("original_") and c in score_df.columns
    ]
    gold_keyed = gold_holdout.set_index(entity_column)[feature_cols]
    score_keyed = score_df.set_index(entity_column).reindex(columns=feature_cols)
    common = gold_keyed.index.intersection(score_keyed.index)
    if common.empty:
        return 0, {}
    ga, sa = gold_keyed.loc[common], score_keyed.loc[common]
    n = len(common)
    result: dict[str, tuple[float, float, int]] = {}
    for c in feature_cols:
        if c not in sa.columns:
            continue
        gc, sc = ga[c], sa[c]
        if _pandas.api.types.is_numeric_dtype(gc):
            diff = np.abs(gc.fillna(0).to_numpy(dtype=float) - sc.fillna(0).to_numpy(dtype=float))
            affected = int(np.sum(diff > tolerance))
            if affected > 0:
                result[c] = (float(np.max(diff)), float(np.mean(diff[diff > tolerance])), affected)
        else:
            affected = int((gc.astype(str) != sc.astype(str)).sum())
            if affected > 0:
                result[c] = (1.0, 1.0, affected)
    return n, result


def _spark_adversarial_diffs(
    gold_df: Any, score_df: Any, entity_column: str,
    target_column: str, holdout_column: str, tolerance: float,
) -> tuple[int, dict[str, tuple[float, float, int]]]:
    import pyspark.sql.functions as F  # noqa: N812
    from pyspark.sql.types import NumericType

    from customer_retention.core.compat import (
        _is_spark_pandas,
        normalize_timestamps,
        pandas_dtype_to_spark_schema,
    )
    from customer_retention.core.compat.detection import get_spark_session

    spark = get_spark_session()
    gold_spark = as_spark_df(gold_df).filter(
        F.col(target_column).isNull() & F.col(holdout_column).isNotNull()
    )
    exclude = {entity_column, target_column, holdout_column}
    feature_cols = [
        c for c in gold_spark.columns
        if c not in exclude and not c.startswith("original_") and c in score_df.columns
    ]
    if not feature_cols:
        return 0, {}

    if _is_spark_pandas(score_df):
        score_spark = as_spark_df(score_df)
    else:
        score_pd = normalize_timestamps(score_df)
        score_spark = spark.createDataFrame(score_pd, schema=pandas_dtype_to_spark_schema(score_pd))

    gold_schema = {f.name: f.dataType for f in gold_spark.schema.fields}
    numeric_cols = [c for c in feature_cols if isinstance(gold_schema.get(c), NumericType)]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    gold_sel = gold_spark.select(F.col(entity_column), *[F.col(c).alias(f"g_{c}") for c in feature_cols])
    score_sel = score_spark.select(F.col(entity_column), *[F.col(c).alias(f"s_{c}") for c in feature_cols])
    joined = gold_sel.join(F.broadcast(score_sel), entity_column, "inner")

    agg_exprs: list[Any] = [F.count("*").alias("_n")]
    for c in numeric_cols:
        diff = F.abs(F.col(f"g_{c}").cast("double") - F.col(f"s_{c}").cast("double"))
        agg_exprs.append(F.max(diff).alias(f"max_{c}"))
        agg_exprs.append(F.avg(F.when(diff > tolerance, diff)).alias(f"mean_{c}"))
        agg_exprs.append(F.sum(F.when(diff > tolerance, F.lit(1)).otherwise(F.lit(0))).alias(f"cnt_{c}"))
    for c in categorical_cols:
        mismatch = F.when(F.col(f"g_{c}").cast("string") != F.col(f"s_{c}").cast("string"), F.lit(1)).otherwise(F.lit(0))
        agg_exprs.append(F.sum(mismatch).alias(f"cat_{c}"))

    summary = joined.agg(*agg_exprs).collect()[0]
    n = _safe_int(summary["_n"])
    result: dict[str, tuple[float, float, int]] = {}
    for c in numeric_cols:
        max_val = _safe_float(summary[f"max_{c}"])
        if max_val is None or max_val <= tolerance:
            continue
        result[c] = (max_val, float(summary[f"mean_{c}"] or 0), _safe_int(summary[f"cnt_{c}"]))
    for c in categorical_cols:
        affected = _safe_int(summary[f"cat_{c}"])
        if affected > 0:
            result[c] = (1.0, 1.0, affected)
    return n, result


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
