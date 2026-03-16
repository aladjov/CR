"""Segment-aware outlier analysis that considers natural data clusters."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from customer_retention.core.compat import DataFrame, normalize_decimal_columns, pd
from customer_retention.stages.cleaning.outlier_handler import OutlierDetectionMethod, OutlierHandler, OutlierResult

from .segment_analyzer import SegmentAnalyzer, SegmentationResult


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
    """

    FALSE_OUTLIER_THRESHOLD = 0.5  # If >50% of global outliers are segment-normal
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
        if len(df) == 0 or all(df[col].isna().all() for col in feature_cols if col in df.columns):
            return self._empty_result(feature_cols)

        valid_cols = [c for c in feature_cols if c in df.columns]
        if not valid_cols:
            return self._empty_result(feature_cols)

        df = normalize_decimal_columns(df)

        global_analysis = self._analyze_global(df, valid_cols)

        if segment_col and segment_col in df.columns:
            segment_labels, n_segments = self._use_explicit_segments(df, segment_col)
            segmentation_result = None
        else:
            segment_labels, n_segments, segmentation_result = self._detect_segments(
                df, valid_cols, target_col
            )

        segment_analysis = self._analyze_by_segment(df, valid_cols, segment_labels, n_segments)
        false_outliers = self._identify_false_outliers(
            df, valid_cols, global_analysis, segment_analysis, segment_labels
        )

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
            segmentation_result=segmentation_result
        )

    def _analyze_global(self, df: DataFrame, feature_cols: List[str]) -> Dict[str, OutlierResult]:
        handler = OutlierHandler(
            detection_method=self.detection_method,
            iqr_multiplier=self.iqr_multiplier,
            zscore_threshold=self.zscore_threshold
        )
        return {col: handler.detect(df[col]) for col in feature_cols}

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

    def _analyze_by_segment(
        self,
        df: DataFrame,
        feature_cols: List[str],
        segment_labels: np.ndarray,
        n_segments: int
    ) -> Dict[Any, Dict[str, OutlierResult]]:
        segment_analysis = {}
        handler = OutlierHandler(
            detection_method=self.detection_method,
            iqr_multiplier=self.iqr_multiplier,
            zscore_threshold=self.zscore_threshold
        )

        for seg_id in range(n_segments):
            mask = segment_labels == seg_id
            if mask.sum() < self.MIN_SEGMENT_SIZE:
                continue

            segment_df = df.loc[mask]
            segment_analysis[seg_id] = {
                col: handler.detect(segment_df[col]) for col in feature_cols
            }

        return segment_analysis

    def _identify_false_outliers(
        self,
        df: DataFrame,
        feature_cols: List[str],
        global_analysis: Dict[str, OutlierResult],
        segment_analysis: Dict[Any, Dict[str, OutlierResult]],
        segment_labels: np.ndarray,
    ) -> Dict[str, int]:
        segment_normal = self._build_segment_normal_masks(
            feature_cols, segment_analysis, segment_labels,
        )
        false_outliers = {}
        for col in feature_cols:
            global_result = global_analysis[col]
            if global_result.outlier_mask is None:
                false_outliers[col] = 0
                continue
            global_mask = np.asarray(global_result.outlier_mask, dtype=bool)
            false_outliers[col] = int(np.sum(global_mask & segment_normal.get(col, np.zeros(len(segment_labels), dtype=bool))))
        return false_outliers

    @staticmethod
    def _build_segment_normal_masks(
        feature_cols: List[str],
        segment_analysis: Dict[Any, Dict[str, OutlierResult]],
        segment_labels: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        n = len(segment_labels)
        seg_index_map: Dict[Any, np.ndarray] = {}
        for seg_id in segment_analysis:
            seg_index_map[seg_id] = np.flatnonzero(segment_labels == seg_id)

        result: Dict[str, np.ndarray] = {}
        for col in feature_cols:
            normal_mask = np.zeros(n, dtype=bool)
            for seg_id, seg_cols in segment_analysis.items():
                seg_result = seg_cols.get(col)
                if seg_result is None or seg_result.outlier_mask is None:
                    continue
                seg_indices = seg_index_map[seg_id]
                seg_outlier_arr = np.asarray(seg_result.outlier_mask, dtype=bool)
                count = min(len(seg_indices), len(seg_outlier_arr))
                normal_mask[seg_indices[:count]] = ~seg_outlier_arr[:count]
            result[col] = normal_mask
        return result

    def _make_recommendations(
        self,
        global_analysis: Dict[str, OutlierResult],
        segment_analysis: Dict[Any, Dict[str, OutlierResult]],
        false_outliers: Dict[str, int],
        n_segments: int
    ) -> tuple:
        recommendations = []
        rationale = []
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
                        f"({total_global} global → {total_segment} segment-level)"
                    )

        if not segmentation_recommended and n_segments <= 1:
            rationale.append("Data appears homogeneous - global outlier treatment is appropriate")
            recommendations.append("Use standard global outlier detection methods")

        return segmentation_recommended, recommendations, rationale

    def _empty_result(self, feature_cols: List[str]) -> SegmentAwareOutlierResult:
        empty_handler = OutlierHandler()
        empty_series = pd.Series([], dtype=float)
        empty_result = empty_handler.detect(empty_series)

        return SegmentAwareOutlierResult(
            n_segments=0,
            global_analysis={col: empty_result for col in feature_cols},
            segment_analysis={},
            false_outliers={col: 0 for col in feature_cols},
            segmentation_recommended=False,
            recommendations=["Insufficient data for outlier analysis"],
            rationale=["Empty or all-null dataset"]
        )
