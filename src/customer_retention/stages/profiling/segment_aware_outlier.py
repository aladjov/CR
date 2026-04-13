"""Segment-aware outlier analysis that considers natural data clusters."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from customer_retention.core.compat import (
    DataFrame,
    _is_spark_pandas,
    as_spark_df,
    native_pd,
    normalize_decimal_columns,
    resolve_time_slice,
    safe_len,
)
from customer_retention.stages.cleaning.outlier_handler import (
    OutlierDetectionMethod,
    OutlierHandler,
    OutlierResult,
    OutlierTreatmentStrategy,
)

from .segment_analyzer import SegmentAnalyzer, SegmentationResult

_EMPTY_SERIES: native_pd.Series = native_pd.Series([], dtype=float)
_STAT_BATCH = 100
_COUNT_BATCH = 70


def _safe_float(val: Any) -> float:
    return float(val) if val is not None else 0.0


@dataclass
class _BoundsInfo:
    """Lightweight outlier bounds -- scalars only, no data references."""
    lower: float
    upper: float
    outlier_count: int


@dataclass
class SegmentAwareOutlierResult:
    """Results from segment-aware outlier analysis."""
    n_segments: int
    global_analysis: Dict[str, OutlierResult]
    segment_analysis: Dict[Any, Dict[str, OutlierResult]]
    false_outliers: Dict[str, int]
    segmentation_recommended: bool
    recommendations: List[str]
    rationale: List[str]
    segment_labels: Optional[np.ndarray] = None
    segmentation_result: Optional[SegmentationResult] = None


class SegmentAwareOutlierAnalyzer:
    """Analyzes outliers considering natural data segments.

    Addresses the problem where global outliers may actually be valid data
    points from a different segment (e.g., enterprise vs retail customers).

    On Databricks with an explicit segment column the analysis stays fully
    distributed via batched Spark aggregation.  Without a segment column the
    auto-detection path delegates to SegmentAnalyzer which requires pandas.
    """

    FALSE_OUTLIER_THRESHOLD = 0.5
    MIN_SEGMENT_SIZE = 10

    def __init__(
        self,
        detection_method: OutlierDetectionMethod = OutlierDetectionMethod.IQR,
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        max_segments: int = 5
    ):
        self.detection_method = detection_method
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.max_segments = max_segments
        self._segment_analyzer = SegmentAnalyzer()

    def analyze(
        self,
        df: DataFrame,
        feature_cols: List[str],
        segment_col: Optional[str] = None,
        target_col: Optional[str] = None
    ) -> SegmentAwareOutlierResult:
        valid_cols = [c for c in feature_cols if c in df.columns]
        if safe_len(df) == 0 or not valid_cols:
            return self._empty_result(feature_cols)

        df = normalize_decimal_columns(df)

        if _is_spark_pandas(df):
            if segment_col and segment_col in df.columns:
                return self._analyze_spark_explicit(df, valid_cols, segment_col)
            return self._analyze_spark_auto(df, valid_cols, target_col)

        if all(df[col].isna().all() for col in valid_cols):
            return self._empty_result(feature_cols)

        return self._analyze_local(df, valid_cols, segment_col, target_col)

    # ---- local (pandas) path ----

    def _analyze_local(self, df, valid_cols, segment_col, target_col):
        global_bounds = self._compute_global_bounds(df, valid_cols)

        if segment_col and segment_col in df.columns:
            segment_labels, n_segments = self._use_explicit_segments(df, segment_col)
            segmentation_result = None
        else:
            segment_labels, n_segments, segmentation_result = self._detect_segments(
                df, valid_cols, target_col
            )

        segment_bounds = self._compute_segment_bounds(
            df, valid_cols, segment_labels, n_segments
        )

        false_outliers = self._count_false_outliers(
            df, valid_cols, global_bounds, segment_bounds, segment_labels
        )

        return self._build_result(
            valid_cols, global_bounds, segment_bounds, false_outliers,
            n_segments, segment_labels, segmentation_result,
        )

    # ---- distributed Spark path (auto-detect segments) ----

    _GRID_TIME_COLUMNS = ("as_of_date", "feature_timestamp")
    _MAX_COLLECT_FOR_SEGMENTATION = 50_000

    def _analyze_spark_auto(self, df, valid_cols, target_col):
        import pyspark.sql.functions as F  # noqa: N812

        spark_df = as_spark_df(df)
        n_rows = int(spark_df.count())
        global_bounds = self._spark_compute_bounds(spark_df, valid_cols)

        local_pdf = self._spark_iid_slice_for_segmentation(
            df, spark_df, valid_cols, target_col, n_rows,
        )
        segment_labels, n_segments, segmentation_result = self._detect_segments(
            local_pdf, valid_cols, target_col
        )
        del local_pdf

        if n_segments <= 1:
            return self._build_result(
                valid_cols, global_bounds, {}, {c: 0 for c in valid_cols},
                n_segments, np.zeros(n_rows, dtype=int), segmentation_result,
            )

        seg_col = "__segment__"
        spark_df = spark_df.withColumn("__mid__", F.monotonically_increasing_id())
        ids = [r["__mid__"] for r in spark_df.select("__mid__").collect()]
        label_pdf = native_pd.DataFrame({
            "__mid__": ids,
            seg_col: segment_labels.astype(int),
        })
        del ids
        label_sdf = spark_df.sparkSession.createDataFrame(label_pdf)
        spark_df = spark_df.join(label_sdf, "__mid__", "inner").drop("__mid__")
        del label_pdf

        valid_segs = list(range(n_segments))
        seg_map = {i: i for i in range(n_segments)}

        segment_bounds = self._spark_grouped_segment_bounds(
            spark_df, valid_cols, seg_col, valid_segs, seg_map,
        )

        false_outliers = self._spark_false_outlier_counts(
            spark_df, valid_cols, global_bounds, segment_bounds,
            seg_col, seg_map,
        )

        return self._build_result(
            valid_cols, global_bounds, segment_bounds, false_outliers,
            n_segments, segment_labels, segmentation_result,
        )

    def _spark_iid_slice_for_segmentation(
        self, df, spark_df, valid_cols, target_col, n_rows,
    ):
        """Extract an IID slice for sklearn segment detection.

        Grid datasets (entity × as_of_date) are reduced to the penultimate
        time slice — one row per entity — so clustering sees IID data instead
        of N_snapshots copies of each entity. Entity-level datasets without a
        time column fall back to bounded sampling.

        Returns a native pandas DataFrame (bounded by entity count or 50K).
        """
        time_col = next(
            (c for c in self._GRID_TIME_COLUMNS if c in df.columns), None,
        )
        slice_df_to_unpersist = None
        if time_col is not None:
            try:
                slice_df, _, slice_n = resolve_time_slice(
                    df, time_col, "penultimate",
                )
                slice_spark = as_spark_df(slice_df)
                slice_df_to_unpersist = slice_spark
            except Exception:
                slice_spark, slice_n = spark_df, n_rows
        else:
            slice_spark, slice_n = spark_df, n_rows

        select_cols = list(valid_cols)
        if target_col and target_col not in select_cols and target_col in spark_df.columns:
            select_cols.append(target_col)
        selected = slice_spark.select([f"`{c}`" for c in select_cols])
        if slice_n > self._MAX_COLLECT_FOR_SEGMENTATION:
            frac = min(1.0, self._MAX_COLLECT_FOR_SEGMENTATION * 1.5 / slice_n)
            selected = selected.sample(fraction=frac, seed=42).limit(
                self._MAX_COLLECT_FOR_SEGMENTATION
            )
        result = selected.toPandas()
        if slice_df_to_unpersist is not None:
            slice_df_to_unpersist.unpersist()
        return result

    # ---- distributed Spark path (explicit segment column) ----

    def _analyze_spark_explicit(self, df, valid_cols, segment_col):
        import pyspark.sql.functions as F  # noqa: N812

        spark_df = as_spark_df(df)

        seg_rows = (
            spark_df.groupBy(segment_col)
            .agg(F.count(F.lit(1)).alias("__cnt__"))
            .collect()
        )
        unique_segs = [r[segment_col] for r in seg_rows if r[segment_col] is not None]
        seg_counts = {r[segment_col]: int(r["__cnt__"]) for r in seg_rows if r[segment_col] is not None}
        seg_map = {v: i for i, v in enumerate(unique_segs)}
        n_segments = len(unique_segs)
        valid_segs = [v for v in unique_segs if seg_counts.get(v, 0) >= self.MIN_SEGMENT_SIZE]

        global_bounds = self._spark_compute_bounds(spark_df, valid_cols)

        segment_bounds = self._spark_grouped_segment_bounds(
            spark_df, valid_cols, segment_col, valid_segs, seg_map,
        )

        false_outliers = self._spark_false_outlier_counts(
            spark_df, valid_cols, global_bounds, segment_bounds,
            segment_col, seg_map,
        )

        segment_labels = self._use_explicit_segments(df, segment_col)[0]

        return self._build_result(
            valid_cols, global_bounds, segment_bounds, false_outliers,
            n_segments, segment_labels, None,
        )

    # ---- grouped segment bounds (single scan via groupBy per batch) ----

    def _spark_grouped_segment_bounds(
        self, spark_df, cols, segment_col, valid_segs, seg_map,
    ):
        if not valid_segs:
            return {}

        seg_lower_upper = self._spark_grouped_stats(
            spark_df, cols, segment_col, valid_segs,
        )
        return self._spark_grouped_outlier_counts(
            spark_df, cols, segment_col, valid_segs, seg_map,
            seg_lower_upper,
        )

    def _spark_grouped_stats(self, spark_df, cols, segment_col, valid_segs):
        import pyspark.sql.functions as F  # noqa: N812

        method = self.detection_method
        seg_lu: Dict[Any, Dict[str, tuple]] = {sv: {} for sv in valid_segs}
        valid_set = set(valid_segs)

        if method == OutlierDetectionMethod.IQR:
            self._grouped_iqr(spark_df, cols, segment_col, valid_set, seg_lu, F)
        elif method == OutlierDetectionMethod.ZSCORE:
            self._grouped_zscore(spark_df, cols, segment_col, valid_set, seg_lu, F)
        elif method == OutlierDetectionMethod.PERCENTILE:
            self._grouped_percentile(spark_df, cols, segment_col, valid_set, seg_lu, F)
        else:
            self._grouped_modified_zscore(spark_df, cols, segment_col, valid_set, seg_lu, F)
        return seg_lu

    def _grouped_iqr(self, spark_df, cols, seg_col, valid_set, seg_lu, F):
        for start in range(0, len(cols), _STAT_BATCH):
            batch = cols[start:start + _STAT_BATCH]
            exprs = []
            for c in batch:
                dc = F.col(f"`{c}`").cast("double")
                exprs.extend([
                    F.percentile_approx(dc, 0.25).alias(f"q1__{c}"),
                    F.percentile_approx(dc, 0.75).alias(f"q3__{c}"),
                ])
            rows = spark_df.groupBy(seg_col).agg(*exprs).collect()
            for row in rows:
                sv = row[seg_col]
                if sv not in valid_set:
                    continue
                for c in batch:
                    q1, q3 = _safe_float(row[f"q1__{c}"]), _safe_float(row[f"q3__{c}"])
                    iqr = q3 - q1
                    seg_lu[sv][c] = (q1 - self.iqr_multiplier * iqr, q3 + self.iqr_multiplier * iqr)

    def _grouped_zscore(self, spark_df, cols, seg_col, valid_set, seg_lu, F):
        for start in range(0, len(cols), _STAT_BATCH):
            batch = cols[start:start + _STAT_BATCH]
            exprs = []
            for c in batch:
                dc = F.col(f"`{c}`").cast("double")
                exprs.extend([F.mean(dc).alias(f"m__{c}"), F.stddev_samp(dc).alias(f"s__{c}")])
            rows = spark_df.groupBy(seg_col).agg(*exprs).collect()
            for row in rows:
                sv = row[seg_col]
                if sv not in valid_set:
                    continue
                for c in batch:
                    m, s = _safe_float(row[f"m__{c}"]), _safe_float(row[f"s__{c}"])
                    seg_lu[sv][c] = (m - self.zscore_threshold * s, m + self.zscore_threshold * s)

    def _grouped_percentile(self, spark_df, cols, seg_col, valid_set, seg_lu, F):
        for start in range(0, len(cols), _STAT_BATCH):
            batch = cols[start:start + _STAT_BATCH]
            exprs = []
            for c in batch:
                dc = F.col(f"`{c}`").cast("double")
                exprs.extend([
                    F.percentile_approx(dc, 0.01).alias(f"lo__{c}"),
                    F.percentile_approx(dc, 0.99).alias(f"hi__{c}"),
                ])
            rows = spark_df.groupBy(seg_col).agg(*exprs).collect()
            for row in rows:
                sv = row[seg_col]
                if sv not in valid_set:
                    continue
                for c in batch:
                    seg_lu[sv][c] = (_safe_float(row[f"lo__{c}"]), _safe_float(row[f"hi__{c}"]))

    def _grouped_modified_zscore(self, spark_df, cols, seg_col, valid_set, seg_lu, F):
        medians: Dict[Any, Dict[str, float]] = {sv: {} for sv in valid_set}
        for start in range(0, len(cols), _STAT_BATCH):
            batch = cols[start:start + _STAT_BATCH]
            exprs = [
                F.percentile_approx(F.col(f"`{c}`").cast("double"), 0.5).alias(c)
                for c in batch
            ]
            rows = spark_df.groupBy(seg_col).agg(*exprs).collect()
            for row in rows:
                sv = row[seg_col]
                if sv not in valid_set:
                    continue
                for c in batch:
                    medians[sv][c] = _safe_float(row[c])

        k = 1.4826
        for start in range(0, len(cols), _STAT_BATCH):
            batch = cols[start:start + _STAT_BATCH]
            exprs = [
                F.percentile_approx(
                    F.abs(F.col(f"`{c}`").cast("double") - F.lit(medians.get(sv, {}).get(c, 0.0))),
                    0.5,
                ).alias(f"mad_{c}")
                for c in batch
                for sv in valid_set
            ]
            if not exprs:
                continue
            # Modified zscore MAD needs per-segment computation -- fall back to groupBy
            mad_exprs = []
            for c in batch:
                dc = F.col(f"`{c}`").cast("double")
                for sv in valid_set:
                    med = medians.get(sv, {}).get(c, 0.0)
                    seg_cond = F.col(f"`{seg_col}`") == F.lit(sv)
                    mad_exprs.append(
                        F.percentile_approx(
                            F.when(seg_cond, F.abs(dc - F.lit(med))),
                            0.5,
                        ).alias(f"mad_{seg_col}_{sv}_{c}")
                    )
            if not mad_exprs:
                continue
            # Batch MAD expressions to keep plan size manageable
            seg_batch = max(5, _STAT_BATCH // max(len(valid_set), 1))
            for ms in range(0, len(batch), seg_batch):
                mb = batch[ms:ms + seg_batch]
                mb_exprs = []
                for c in mb:
                    dc = F.col(f"`{c}`").cast("double")
                    for sv in valid_set:
                        med = medians.get(sv, {}).get(c, 0.0)
                        seg_cond = F.col(f"`{seg_col}`") == F.lit(sv)
                        mb_exprs.append(
                            F.percentile_approx(
                                F.when(seg_cond, F.abs(dc - F.lit(med))),
                                0.5,
                            ).alias(f"mad_{id(sv)}_{c}")
                        )
                if not mb_exprs:
                    continue
                row = spark_df.agg(*mb_exprs).head()
                for c in mb:
                    for sv in valid_set:
                        med = medians.get(sv, {}).get(c, 0.0)
                        mad = _safe_float(row[f"mad_{id(sv)}_{c}"])
                        seg_lu[sv][c] = (med - 3.5 * k * mad, med + 3.5 * k * mad)

    def _spark_grouped_outlier_counts(
        self, spark_df, cols, segment_col, valid_segs, seg_map, seg_lower_upper,
    ):
        import pyspark.sql.functions as F  # noqa: N812

        segment_bounds: Dict[int, Dict[str, _BoundsInfo]] = {}
        for sv in valid_segs:
            segment_bounds[seg_map[sv]] = {}

        n_segs = max(len(valid_segs), 1)
        batch_size = max(10, _COUNT_BATCH // n_segs)

        for start in range(0, len(cols), batch_size):
            batch = cols[start:start + batch_size]
            exprs = []
            expr_keys: List[tuple] = []
            for c in batch:
                for sv in valid_segs:
                    lu = seg_lower_upper[sv].get(c)
                    if lu is None:
                        continue
                    lo, hi = lu
                    dc = F.col(f"`{c}`").cast("double")
                    seg_cond = F.col(f"`{segment_col}`") == F.lit(sv)
                    is_outlier = (dc < F.lit(lo)) | (dc > F.lit(hi))
                    alias = f"o_{seg_map[sv]}_{len(expr_keys)}"
                    exprs.append(
                        F.coalesce(
                            F.sum(F.when(seg_cond, is_outlier.cast("int")).otherwise(0)),
                            F.lit(0),
                        ).alias(alias)
                    )
                    expr_keys.append((seg_map[sv], c, lo, hi, alias))

            if not exprs:
                continue
            row = spark_df.agg(*exprs).head()
            for seg_id, c, lo, hi, alias in expr_keys:
                cnt = int(row[alias]) if row[alias] is not None else 0
                segment_bounds[seg_id][c] = _BoundsInfo(lo, hi, cnt)

        return segment_bounds

    # ---- batched Spark bounds computation ----

    def _spark_compute_bounds(self, spark_df, cols):
        lower_upper = self._spark_bulk_stats(spark_df, cols)
        return self._spark_bulk_outlier_counts(spark_df, cols, lower_upper)

    def _spark_bulk_stats(self, spark_df, cols):
        import pyspark.sql.functions as F  # noqa: N812

        method = self.detection_method
        results: Dict[str, tuple] = {}

        if method == OutlierDetectionMethod.IQR:
            self._spark_batch_iqr(spark_df, cols, results, F)
        elif method == OutlierDetectionMethod.ZSCORE:
            self._spark_batch_zscore(spark_df, cols, results, F)
        elif method == OutlierDetectionMethod.PERCENTILE:
            self._spark_batch_percentile(spark_df, cols, results, F)
        else:
            self._spark_batch_modified_zscore(spark_df, cols, results, F)
        return results

    def _spark_batch_iqr(self, spark_df, cols, results, F):
        for start in range(0, len(cols), _STAT_BATCH):
            batch = cols[start:start + _STAT_BATCH]
            exprs = []
            for c in batch:
                dc = F.col(f"`{c}`").cast("double")
                exprs.extend([
                    F.percentile_approx(dc, 0.25).alias(f"q1__{c}"),
                    F.percentile_approx(dc, 0.75).alias(f"q3__{c}"),
                ])
            row = spark_df.agg(*exprs).head()
            for c in batch:
                q1, q3 = _safe_float(row[f"q1__{c}"]), _safe_float(row[f"q3__{c}"])
                iqr = q3 - q1
                results[c] = (q1 - self.iqr_multiplier * iqr, q3 + self.iqr_multiplier * iqr)

    def _spark_batch_zscore(self, spark_df, cols, results, F):
        for start in range(0, len(cols), _STAT_BATCH):
            batch = cols[start:start + _STAT_BATCH]
            exprs = []
            for c in batch:
                dc = F.col(f"`{c}`").cast("double")
                exprs.extend([F.mean(dc).alias(f"m__{c}"), F.stddev_samp(dc).alias(f"s__{c}")])
            row = spark_df.agg(*exprs).head()
            for c in batch:
                m, s = _safe_float(row[f"m__{c}"]), _safe_float(row[f"s__{c}"])
                results[c] = (m - self.zscore_threshold * s, m + self.zscore_threshold * s)

    def _spark_batch_percentile(self, spark_df, cols, results, F):
        for start in range(0, len(cols), _STAT_BATCH):
            batch = cols[start:start + _STAT_BATCH]
            exprs = []
            for c in batch:
                dc = F.col(f"`{c}`").cast("double")
                exprs.extend([
                    F.percentile_approx(dc, 0.01).alias(f"lo__{c}"),
                    F.percentile_approx(dc, 0.99).alias(f"hi__{c}"),
                ])
            row = spark_df.agg(*exprs).head()
            for c in batch:
                results[c] = (_safe_float(row[f"lo__{c}"]), _safe_float(row[f"hi__{c}"]))

    def _spark_batch_modified_zscore(self, spark_df, cols, results, F):
        medians: Dict[str, float] = {}
        for start in range(0, len(cols), _STAT_BATCH):
            batch = cols[start:start + _STAT_BATCH]
            exprs = [
                F.percentile_approx(F.col(f"`{c}`").cast("double"), 0.5).alias(c)
                for c in batch
            ]
            row = spark_df.agg(*exprs).head()
            for c in batch:
                medians[c] = _safe_float(row[c])

        k = 1.4826
        for start in range(0, len(cols), _STAT_BATCH):
            batch = cols[start:start + _STAT_BATCH]
            exprs = [
                F.percentile_approx(
                    F.abs(F.col(f"`{c}`").cast("double") - F.lit(medians[c])), 0.5
                ).alias(c)
                for c in batch
            ]
            row = spark_df.agg(*exprs).head()
            for c in batch:
                mad = _safe_float(row[c])
                results[c] = (medians[c] - 3.5 * k * mad, medians[c] + 3.5 * k * mad)

    def _spark_bulk_outlier_counts(self, spark_df, cols, lower_upper):
        import pyspark.sql.functions as F  # noqa: N812

        bounds: Dict[str, _BoundsInfo] = {}
        for start in range(0, len(cols), _COUNT_BATCH):
            batch = cols[start:start + _COUNT_BATCH]
            exprs = []
            for c in batch:
                lo, hi = lower_upper[c]
                dc = F.col(f"`{c}`").cast("double")
                exprs.append(
                    F.coalesce(
                        F.sum(((dc < F.lit(lo)) | (dc > F.lit(hi))).cast("int")),
                        F.lit(0),
                    ).alias(c)
                )
            row = spark_df.agg(*exprs).head()
            for c in batch:
                lo, hi = lower_upper[c]
                bounds[c] = _BoundsInfo(lo, hi, int(row[c]) if row[c] is not None else 0)
        return bounds

    def _spark_false_outlier_counts(
        self, spark_df, valid_cols, global_bounds, segment_bounds,
        segment_col, seg_map,
    ):
        import pyspark.sql.functions as F  # noqa: N812

        if not segment_bounds:
            return {c: 0 for c in valid_cols}

        id_to_val = {v: k for k, v in seg_map.items()}
        false_outliers: Dict[str, int] = {}

        for start in range(0, len(valid_cols), _COUNT_BATCH):
            batch = valid_cols[start:start + _COUNT_BATCH]
            exprs = []
            batch_with_expr: List[str] = []

            for c in batch:
                g = global_bounds[c]
                if g.outlier_count == 0:
                    false_outliers[c] = 0
                    continue

                dc = F.col(f"`{c}`").cast("double")
                sc = F.col(f"`{segment_col}`")
                is_global_outlier = (dc < F.lit(g.lower)) | (dc > F.lit(g.upper))

                seg_normal_cases = []
                for seg_id, col_bounds in segment_bounds.items():
                    sb = col_bounds.get(c)
                    if sb is None:
                        continue
                    seg_val = id_to_val[seg_id]
                    seg_normal_cases.append(
                        (sc == F.lit(seg_val))
                        & (dc >= F.lit(sb.lower))
                        & (dc <= F.lit(sb.upper))
                    )

                if not seg_normal_cases:
                    false_outliers[c] = 0
                    continue

                any_seg_normal = seg_normal_cases[0]
                for cond in seg_normal_cases[1:]:
                    any_seg_normal = any_seg_normal | cond

                exprs.append(
                    F.coalesce(
                        F.sum((is_global_outlier & any_seg_normal).cast("int")),
                        F.lit(0),
                    ).alias(c)
                )
                batch_with_expr.append(c)

            if exprs:
                row = spark_df.agg(*exprs).head()
                for c in batch_with_expr:
                    false_outliers[c] = int(row[c]) if row[c] is not None else 0

        return false_outliers

    # ---- shared result builder ----

    def _build_result(
        self, valid_cols, global_bounds, segment_bounds, false_outliers,
        n_segments, segment_labels, segmentation_result,
    ):
        global_analysis = {col: self._bounds_to_result(b) for col, b in global_bounds.items()}
        segment_analysis = {
            seg_id: {col: self._bounds_to_result(b) for col, b in cb.items()}
            for seg_id, cb in segment_bounds.items()
        }

        segmentation_recommended, recommendations, rationale = self._make_recommendations(
            global_analysis, segment_analysis, false_outliers, n_segments
        )

        return SegmentAwareOutlierResult(
            n_segments=n_segments,
            global_analysis=global_analysis,
            segment_analysis=segment_analysis,
            false_outliers=false_outliers,
            segmentation_recommended=segmentation_recommended,
            recommendations=recommendations,
            rationale=rationale,
            segment_labels=segment_labels,
            segmentation_result=segmentation_result,
        )

    # ---- local bounds computation (scalars only) ----

    def _compute_global_bounds(
        self, df: DataFrame, feature_cols: List[str]
    ) -> Dict[str, _BoundsInfo]:
        handler = OutlierHandler(
            detection_method=self.detection_method,
            iqr_multiplier=self.iqr_multiplier,
            zscore_threshold=self.zscore_threshold,
        )
        bounds: Dict[str, _BoundsInfo] = {}
        for col in feature_cols:
            series = df[col]
            clean = series.dropna()
            lower, upper = handler._compute_bounds(clean)
            count = int(((series < lower) | (series > upper)).fillna(False).sum())
            bounds[col] = _BoundsInfo(lower, upper, count)
        return bounds

    def _compute_segment_bounds(
        self, df: DataFrame, feature_cols: List[str],
        segment_labels: np.ndarray, n_segments: int,
    ) -> Dict[Any, Dict[str, _BoundsInfo]]:
        handler = OutlierHandler(
            detection_method=self.detection_method,
            iqr_multiplier=self.iqr_multiplier,
            zscore_threshold=self.zscore_threshold,
        )
        result: Dict[Any, Dict[str, _BoundsInfo]] = {}
        for seg_id in range(n_segments):
            mask = segment_labels == seg_id
            if mask.sum() < self.MIN_SEGMENT_SIZE:
                continue
            seg_df = df.loc[mask]
            col_bounds: Dict[str, _BoundsInfo] = {}
            for col in feature_cols:
                series = seg_df[col]
                clean = series.dropna()
                if len(clean) == 0:
                    col_bounds[col] = _BoundsInfo(0.0, 0.0, 0)
                    continue
                lower, upper = handler._compute_bounds(clean)
                count = int(((series < lower) | (series > upper)).fillna(False).sum())
                col_bounds[col] = _BoundsInfo(lower, upper, count)
            result[seg_id] = col_bounds
        return result

    # ---- local false outlier counting (bounds-only, one column at a time) ----

    def _count_false_outliers(
        self, df: DataFrame, feature_cols: List[str],
        global_bounds: Dict[str, _BoundsInfo],
        segment_bounds: Dict[Any, Dict[str, _BoundsInfo]],
        segment_labels: np.ndarray,
    ) -> Dict[str, int]:
        false_outliers: Dict[str, int] = {}
        for col in feature_cols:
            g = global_bounds[col]
            if g.outlier_count == 0:
                false_outliers[col] = 0
                continue
            count = 0
            for seg_id, col_bounds in segment_bounds.items():
                sb = col_bounds.get(col)
                if sb is None:
                    continue
                seg_mask = segment_labels == seg_id
                seg_series = df.loc[seg_mask, col]
                is_global_outlier = (seg_series < g.lower) | (seg_series > g.upper)
                is_seg_normal = (seg_series >= sb.lower) & (seg_series <= sb.upper)
                count += int((is_global_outlier & is_seg_normal).fillna(False).sum())
            false_outliers[col] = count
        return false_outliers

    # ---- lightweight OutlierResult construction ----

    def _bounds_to_result(self, b: _BoundsInfo) -> OutlierResult:
        return OutlierResult(
            series=_EMPTY_SERIES,
            method_used=self.detection_method,
            strategy_used=OutlierTreatmentStrategy.NONE,
            outliers_detected=b.outlier_count,
            outliers_treated=0,
            lower_bound=b.lower,
            upper_bound=b.upper,
        )

    # ---- segmentation ----

    def _use_explicit_segments(self, df: DataFrame, segment_col: str) -> tuple:
        unique_vals = df[segment_col].dropna().unique()
        if hasattr(unique_vals, 'to_numpy'):
            unique_vals = unique_vals.to_numpy()
        label_map = {v: i for i, v in enumerate(unique_vals)}
        raw = df[segment_col].to_numpy()
        labels = np.array([label_map.get(v, -1) for v in raw], dtype=int)
        return labels, len(unique_vals)

    def _detect_segments(
        self, df: DataFrame, feature_cols: List[str], target_col: Optional[str]
    ) -> tuple:
        n_rows = len(df)
        if n_rows < self.MIN_SEGMENT_SIZE * 2:
            return np.zeros(n_rows, dtype=int), 1, None

        try:
            result = self._segment_analyzer.analyze(
                df,
                target_col=target_col,
                feature_cols=feature_cols,
                max_segments=self.max_segments
            )
            if len(result.labels) != n_rows:
                return np.zeros(n_rows, dtype=int), 1, None
            return result.labels, result.n_segments, result
        except Exception:
            return np.zeros(n_rows, dtype=int), 1, None

    # ---- recommendations ----

    def _make_recommendations(
        self,
        global_analysis: Dict[str, OutlierResult],
        segment_analysis: Dict[Any, Dict[str, OutlierResult]],
        false_outliers: Dict[str, int],
        n_segments: int
    ) -> tuple:
        recommendations: List[str] = []
        rationale: List[str] = []
        segmentation_recommended = False

        for col, false_count in false_outliers.items():
            global_count = global_analysis[col].outliers_detected
            if global_count == 0:
                continue

            false_ratio = false_count / global_count

            if false_ratio >= self.FALSE_OUTLIER_THRESHOLD:
                segmentation_recommended = True
                rationale.append(
                    f"{col}: {false_count}/{global_count} ({false_ratio:.0%}) global outliers "
                    f"are normal within their segment"
                )
                recommendations.append(
                    f"Consider segment-specific outlier treatment for '{col}' - "
                    f"global outliers may be valid data from different customer segments"
                )
            elif false_ratio > 0.2:
                rationale.append(
                    f"{col}: {false_count}/{global_count} ({false_ratio:.0%}) false outliers detected"
                )

        if n_segments > 1 and not segmentation_recommended:
            total_global = sum(r.outliers_detected for r in global_analysis.values())
            total_segment = sum(
                sum(r.outliers_detected for r in seg.values())
                for seg in segment_analysis.values()
            )

            if total_global > 0:
                reduction = (total_global - total_segment) / total_global
                if reduction > 0.3:
                    rationale.append(
                        f"Segment-aware analysis reduces outliers by {reduction:.0%} "
                        f"({total_global} global -> {total_segment} segment-level)"
                    )

        if not segmentation_recommended and n_segments <= 1:
            rationale.append("Data appears homogeneous - global outlier treatment is appropriate")
            recommendations.append("Use standard global outlier detection methods")

        return segmentation_recommended, recommendations, rationale

    def _empty_result(self, feature_cols: List[str]) -> SegmentAwareOutlierResult:
        empty = OutlierResult(
            series=_EMPTY_SERIES,
            method_used=self.detection_method,
            strategy_used=OutlierTreatmentStrategy.NONE,
            outliers_detected=0,
            outliers_treated=0,
            lower_bound=None,
            upper_bound=None,
        )

        return SegmentAwareOutlierResult(
            n_segments=0,
            global_analysis={col: empty for col in feature_cols},
            segment_analysis={},
            false_outliers={col: 0 for col in feature_cols},
            segmentation_recommended=False,
            recommendations=["Insufficient data for outlier analysis"],
            rationale=["Empty or all-null dataset"]
        )
