from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from customer_retention.core.compat import DataFrame, native_pd, to_pandas

from .segment_analyzer import (
    DimensionReductionMethod,
    FullSegmentationResult,
    SegmentAnalyzer,
    SegmentationDecisionMetrics,
    SegmentationMethod,
    SegmentationResult,
)


def _stratified_sample(
    df: DataFrame, target_col: Optional[str], max_size: int,
) -> DataFrame:
    if len(df) <= max_size:
        return df

    if target_col and target_col in df.columns:
        groups = df.groupby(target_col)
        fraction = max_size / len(df)
        samples = []
        for _name, group in groups:
            n = max(1, int(len(group) * fraction))
            samples.append(group.sample(n=min(n, len(group)), random_state=42))
        result = native_pd.concat(samples, ignore_index=False)
        if len(result) > max_size:
            result = result.sample(n=max_size, random_state=42)
        return result

    return df.sample(n=max_size, random_state=42)


def _propagate_labels(
    full_df: DataFrame,
    sample_df: DataFrame,
    sample_labels: np.ndarray,
    feature_cols: List[str],
    scaler: StandardScaler,
) -> np.ndarray:
    full_labels = np.full(len(full_df), -1)

    valid_mask = ~full_df[feature_cols].isna().any(axis=1)
    valid_indices = full_df[feature_cols][valid_mask].index.to_numpy()

    non_noise = sample_labels >= 0
    if non_noise.sum() == 0:
        return full_labels

    sample_index_values = sample_df.index.to_numpy()
    idx_to_sample_pos = {
        int(idx): pos for pos, idx in enumerate(sample_index_values)
    }

    non_noise_labels = sample_labels[non_noise]
    unique_labels = sorted(set(int(lbl) for lbl in non_noise_labels))

    sample_non_noise_features = sample_df.loc[
        sample_index_values[non_noise], feature_cols
    ].to_numpy()
    sample_non_noise_scaled = scaler.transform(sample_non_noise_features)

    centroids = np.zeros((len(unique_labels), len(feature_cols)))
    for ci, lbl in enumerate(unique_labels):
        lbl_mask = non_noise_labels == lbl
        centroids[ci] = sample_non_noise_scaled[lbl_mask].mean(axis=0)

    full_features_scaled = scaler.transform(
        full_df.loc[valid_indices, feature_cols].to_numpy(),
    )

    sample_set = set(idx_to_sample_pos.keys())
    for i, idx in enumerate(valid_indices):
        idx_int = int(idx)
        if idx_int in sample_set:
            pos = idx_to_sample_pos[idx_int]
            full_labels[idx_int] = int(sample_labels[pos])
        else:
            dists = np.linalg.norm(centroids - full_features_scaled[i], axis=1)
            full_labels[idx_int] = unique_labels[int(np.argmin(dists))]

    return full_labels


class SparkSegmentAnalyzer(SegmentAnalyzer):

    def __init__(
        self,
        default_method: SegmentationMethod = SegmentationMethod.KMEANS,
        max_sample_size: int = 50_000,
    ):
        super().__init__(default_method=default_method)
        self.max_sample_size = max_sample_size

    def analyze(
        self,
        df: DataFrame,
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        max_segments: int = 5,
        method: Optional[SegmentationMethod] = None,
        max_sample_size: Optional[int] = None,
    ) -> SegmentationResult:
        df = self._to_latest_snapshot(df)
        df = to_pandas(df)
        effective_max = max_sample_size if max_sample_size is not None else self.max_sample_size

        method = method or self.default_method
        feature_cols = self._select_features(df, feature_cols, target_col)

        if len(feature_cols) == 0:
            return self._empty_result(df, method)

        features_df = df[feature_cols].dropna()
        if len(features_df) < 10:
            return self._single_segment_result(df, method, target_col)

        if len(df) <= effective_max:
            return super().analyze(
                df,
                target_col=target_col,
                feature_cols=feature_cols,
                max_segments=max_segments,
                method=method,
            )

        sample_df = _stratified_sample(df, target_col, effective_max)
        sample_reset = sample_df.reset_index(drop=True)

        sample_result = super().analyze(
            sample_reset,
            target_col=target_col,
            feature_cols=feature_cols,
            max_segments=max_segments,
            method=method,
        )

        if sample_result.n_segments <= 1:
            labels = np.zeros(len(df), dtype=int)
            return SegmentationResult(
                n_segments=sample_result.n_segments,
                method=sample_result.method,
                quality_score=sample_result.quality_score,
                profiles=self.profile_segments(df, labels, feature_cols, target_col),
                target_variance_ratio=sample_result.target_variance_ratio,
                recommendation=sample_result.recommendation,
                confidence=sample_result.confidence,
                rationale=sample_result.rationale,
                labels=labels,
            )

        return self._rescale_result(
            sample_result, df, sample_df, target_col, feature_cols,
        )

    def _rescale_result(
        self,
        sample_result: SegmentationResult,
        full_df: DataFrame,
        sample_df: DataFrame,
        target_col: Optional[str],
        feature_cols: List[str],
    ) -> SegmentationResult:
        full_labels = _propagate_labels(
            full_df, sample_df, sample_result.labels, feature_cols, self._scaler,
        )

        profiles = self.profile_segments(full_df, full_labels, feature_cols, target_col)
        target_variance = self._calculate_target_variance(
            full_df, full_labels, target_col,
        )
        recommendation, confidence, rationale = self._make_recommendation(
            sample_result.quality_score, target_variance,
            sample_result.n_segments, profiles,
        )

        return SegmentationResult(
            n_segments=sample_result.n_segments,
            method=sample_result.method,
            quality_score=sample_result.quality_score,
            profiles=profiles,
            target_variance_ratio=target_variance,
            recommendation=recommendation,
            confidence=confidence,
            rationale=rationale,
            labels=full_labels,
        )

    def run_full_analysis(
        self,
        df: DataFrame,
        feature_cols: List[str],
        target_col: Optional[str] = None,
        max_segments: int = 5,
        method: Optional[SegmentationMethod] = None,
        dim_reduction: DimensionReductionMethod = DimensionReductionMethod.PCA,
        max_sample_size: Optional[int] = None,
    ) -> FullSegmentationResult:
        seg_result = self.analyze(
            df,
            target_col=target_col,
            feature_cols=feature_cols,
            max_segments=max_segments,
            method=method,
            max_sample_size=max_sample_size,
        )

        metrics = SegmentationDecisionMetrics.from_segmentation_result(seg_result)

        total = sum(p.size for p in seg_result.profiles)
        min_size = min(p.size for p in seg_result.profiles)
        max_size = max(p.size for p in seg_result.profiles)
        balance_ratio = min_size / max_size if max_size > 0 else 0

        size_distribution = {
            "total": total,
            "min_size": min_size,
            "max_size": max_size,
            "min_pct": min_size / total * 100 if total > 0 else 0,
            "max_pct": max_size / total * 100 if total > 0 else 0,
            "balance_ratio": balance_ratio,
        }

        visualization = None
        valid_features = [c for c in feature_cols if c in df.columns]
        if seg_result.n_segments > 1 and len(valid_features) >= 2:
            visualization = self.get_cluster_visualization(
                to_pandas(df),
                labels=seg_result.labels,
                feature_cols=valid_features,
                method=dim_reduction,
            )

        return FullSegmentationResult(
            metrics=metrics,
            profiles=seg_result.profiles,
            size_distribution=size_distribution,
            visualization=visualization,
            segmentation_result=seg_result,
        )
