from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from customer_retention.core.compat import DataFrame, to_pandas


class SegmentationMethod(Enum):
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"


class DimensionReductionMethod(Enum):
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"


@dataclass
class SegmentProfile:
    segment_id: int
    size: int
    size_pct: float
    target_rate: Optional[float]
    defining_features: Dict[str, Any]


@dataclass
class SegmentationResult:
    n_segments: int
    method: SegmentationMethod
    quality_score: float
    profiles: List[SegmentProfile]
    target_variance_ratio: Optional[float]
    recommendation: str
    confidence: float
    rationale: List[str]
    labels: np.ndarray = field(repr=False)


@dataclass
class ClusterVisualizationResult:
    x: np.ndarray
    y: np.ndarray
    labels: np.ndarray
    method: DimensionReductionMethod
    explained_variance_ratio: Optional[float] = None


@dataclass
class SegmentationDecisionMetrics:
    """Key metrics for segmentation decision-making."""
    silhouette_score: float
    silhouette_interpretation: str
    target_variance_ratio: Optional[float]
    target_variance_interpretation: str
    n_segments: int
    segments_interpretation: str
    confidence: float
    confidence_interpretation: str
    recommendation: str
    rationale: List[str]

    @classmethod
    def from_segmentation_result(cls, result: "SegmentationResult") -> "SegmentationDecisionMetrics":
        """Create metrics from a SegmentationResult."""
        # Convert normalized quality score back to silhouette
        silhouette = result.quality_score * 2 - 1

        # Silhouette interpretation
        if silhouette > 0.5:
            sil_interp = "Strong structure"
        elif silhouette > 0.25:
            sil_interp = "Reasonable"
        elif silhouette > 0:
            sil_interp = "Weak structure"
        else:
            sil_interp = "No structure"

        # Target variance interpretation
        if result.target_variance_ratio is not None:
            tv = result.target_variance_ratio
            if tv > 0.3:
                tv_interp = "High separation"
            elif tv > 0.15:
                tv_interp = "Moderate"
            else:
                tv_interp = "Low separation"
        else:
            tv_interp = "N/A"

        # Segments interpretation
        seg_interp = "Manageable" if result.n_segments <= 4 else "Complex"

        # Confidence interpretation
        if result.confidence > 0.6:
            conf_interp = "High"
        elif result.confidence > 0.3:
            conf_interp = "Medium"
        else:
            conf_interp = "Low"

        return cls(
            silhouette_score=silhouette,
            silhouette_interpretation=sil_interp,
            target_variance_ratio=result.target_variance_ratio,
            target_variance_interpretation=tv_interp,
            n_segments=result.n_segments,
            segments_interpretation=seg_interp,
            confidence=result.confidence,
            confidence_interpretation=conf_interp,
            recommendation=result.recommendation,
            rationale=result.rationale,
        )


@dataclass
class FullSegmentationResult:
    """Complete segmentation analysis result for dashboard display."""
    metrics: SegmentationDecisionMetrics
    profiles: List[SegmentProfile]
    size_distribution: Dict[str, Any]
    visualization: Optional[ClusterVisualizationResult]
    segmentation_result: Optional[SegmentationResult]

    @property
    def has_visualization(self) -> bool:
        """Check if visualization is available."""
        return self.visualization is not None

    def get_decision_summary(self) -> str:
        """Get a human-readable decision summary."""
        if self.metrics.recommendation == "strong_segmentation":
            return (
                "STRONG EVIDENCE FOR SEGMENTATION\n\n"
                "The data shows clear cluster structure with meaningful target rate "
                "differences across segments. Consider building separate models per "
                "segment if EPV requirements are met."
            )
        elif self.metrics.recommendation == "consider_segmentation":
            return (
                "MODERATE EVIDENCE FOR SEGMENTATION\n\n"
                "Some cluster structure exists but may not be strong enough to justify "
                "separate models. Consider:\n"
                "- Using segments as a feature in a single model\n"
                "- Segment-specific preprocessing but unified modeling"
            )
        else:
            return (
                "SINGLE MODEL RECOMMENDED\n\n"
                "The data does not show sufficient cluster structure or target rate "
                "variation to justify segmentation. A single unified model is likely "
                "the best approach."
            )


class SegmentAnalyzer:
    def __init__(self, default_method: SegmentationMethod = SegmentationMethod.KMEANS):
        self.default_method = default_method
        self._scaler = StandardScaler()

    def analyze(
        self,
        df: DataFrame,
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        max_segments: int = 5,
        method: Optional[SegmentationMethod] = None,
    ) -> SegmentationResult:
        df = to_pandas(df)
        method = method or self.default_method

        feature_cols = self._select_features(df, feature_cols, target_col)
        if len(feature_cols) == 0:
            return self._empty_result(df, method)

        features_df = df[feature_cols].copy()
        features_df = features_df.dropna()
        valid_indices = features_df.index

        if len(features_df) < 10:
            return self._single_segment_result(df, method, target_col)

        n_segments = self.find_optimal_segments(
            df.loc[valid_indices], feature_cols, max_k=max_segments
        )

        scaled_features = self._scaler.fit_transform(features_df)
        labels = self._fit_clusters(scaled_features, n_segments, method)

        full_labels = np.full(len(df), -1)
        full_labels[valid_indices] = labels

        quality_score = self._calculate_quality(scaled_features, labels)
        profiles = self.profile_segments(df, full_labels, feature_cols, target_col)
        target_variance = self._calculate_target_variance(df, full_labels, target_col)
        recommendation, confidence, rationale = self._make_recommendation(
            quality_score, target_variance, n_segments, profiles
        )

        return SegmentationResult(
            n_segments=n_segments,
            method=method,
            quality_score=quality_score,
            profiles=profiles,
            target_variance_ratio=target_variance,
            recommendation=recommendation,
            confidence=confidence,
            rationale=rationale,
            labels=full_labels,
        )

    def find_optimal_segments(
        self,
        df: DataFrame,
        feature_cols: List[str],
        max_k: int = 10,
    ) -> int:
        df = to_pandas(df)
        features_df = df[feature_cols].dropna()

        if len(features_df) < 10:
            return 1

        max_k = min(max_k, len(features_df) // 3, 10)
        if max_k < 2:
            return 1

        scaled = self._scaler.fit_transform(features_df)

        silhouette_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled)
            if len(set(labels)) > 1:
                score = silhouette_score(scaled, labels)
                silhouette_scores.append((k, score))

        if not silhouette_scores:
            return 1

        best_k = max(silhouette_scores, key=lambda x: x[1])[0]
        return best_k

    def profile_segments(
        self,
        df: DataFrame,
        labels: np.ndarray,
        feature_cols: List[str],
        target_col: Optional[str] = None,
    ) -> List[SegmentProfile]:
        df = to_pandas(df)
        profiles = []
        unique_labels = sorted(set(labels[labels >= 0]))
        total_valid = sum(labels >= 0)

        for seg_id in unique_labels:
            mask = labels == seg_id
            segment_df = df.loc[mask]
            size = len(segment_df)

            target_rate = None
            if target_col and target_col in df.columns:
                target_series = segment_df[target_col]
                if target_series.dtype in [np.int64, np.float64, int, float]:
                    unique_vals = target_series.dropna().unique()
                    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                        target_rate = float(target_series.mean())

            defining_features = {}
            for col in feature_cols:
                if col in segment_df.columns:
                    col_data = segment_df[col].dropna()
                    if len(col_data) > 0 and np.issubdtype(col_data.dtype, np.number):
                        defining_features[col] = {
                            "mean": float(col_data.mean()),
                            "std": float(col_data.std()),
                            "min": float(col_data.min()),
                            "max": float(col_data.max()),
                        }

            profiles.append(SegmentProfile(
                segment_id=int(seg_id),
                size=size,
                size_pct=round(size / total_valid * 100, 1) if total_valid > 0 else 0,
                target_rate=target_rate,
                defining_features=defining_features,
            ))

        return profiles

    def _select_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]],
        target_col: Optional[str],
    ) -> List[str]:
        if feature_cols:
            return [c for c in feature_cols if c in df.columns]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        return numeric_cols

    def _fit_clusters(
        self,
        scaled_features: np.ndarray,
        n_clusters: int,
        method: SegmentationMethod,
    ) -> np.ndarray:
        if method == SegmentationMethod.KMEANS:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == SegmentationMethod.HIERARCHICAL:
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == SegmentationMethod.DBSCAN:
            model = DBSCAN(eps=0.5, min_samples=5)
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        return model.fit_predict(scaled_features)

    def _calculate_quality(self, scaled_features: np.ndarray, labels: np.ndarray) -> float:
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            return 0.0
        try:
            score = silhouette_score(scaled_features, labels)
            return float(max(0, (score + 1) / 2))
        except Exception:
            return 0.0

    def _calculate_target_variance(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        target_col: Optional[str],
    ) -> Optional[float]:
        if not target_col or target_col not in df.columns:
            return None

        target = df[target_col]
        if not np.issubdtype(target.dtype, np.number):
            return None

        unique_vals = target.dropna().unique()
        if not (len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})):
            return None

        segment_rates = []
        for seg_id in set(labels[labels >= 0]):
            mask = labels == seg_id
            seg_rate = target[mask].mean()
            if not np.isnan(seg_rate):
                segment_rates.append(seg_rate)

        if len(segment_rates) < 2:
            return 0.0

        variance = np.var(segment_rates)
        max_possible_variance = 0.25
        return float(min(1.0, variance / max_possible_variance))

    def _make_recommendation(
        self,
        quality_score: float,
        target_variance: Optional[float],
        n_segments: int,
        profiles: List[SegmentProfile],
    ) -> tuple:
        rationale = []
        score = 0.0

        if quality_score > 0.7:
            rationale.append(f"High cluster quality (silhouette: {quality_score:.2f})")
            score += 0.3
        elif quality_score > 0.5:
            rationale.append(f"Moderate cluster quality (silhouette: {quality_score:.2f})")
            score += 0.15

        if target_variance is not None:
            if target_variance > 0.3:
                rationale.append(f"High target rate variation across segments ({target_variance:.2f})")
                score += 0.4
            elif target_variance > 0.15:
                rationale.append(f"Moderate target rate variation ({target_variance:.2f})")
                score += 0.2
            else:
                rationale.append(f"Low target rate variation ({target_variance:.2f})")

        min_segment_pct = min(p.size_pct for p in profiles) if profiles else 0
        if min_segment_pct >= 10:
            rationale.append(f"All segments have sufficient size (min: {min_segment_pct:.1f}%)")
            score += 0.2
        elif min_segment_pct >= 5:
            rationale.append(f"Some segments are small (min: {min_segment_pct:.1f}%)")
            score += 0.1
        else:
            rationale.append(f"Very small segments detected (min: {min_segment_pct:.1f}%)")

        if n_segments > 5:
            rationale.append(f"Many segments ({n_segments}) may complicate maintenance")
            score -= 0.1

        confidence = min(1.0, max(0.0, score))

        if score >= 0.6:
            recommendation = "strong_segmentation"
        elif score >= 0.3:
            recommendation = "consider_segmentation"
        else:
            recommendation = "single_model"

        return recommendation, confidence, rationale

    def _empty_result(self, df: pd.DataFrame, method: SegmentationMethod) -> SegmentationResult:
        return SegmentationResult(
            n_segments=1,
            method=method,
            quality_score=0.0,
            profiles=[SegmentProfile(
                segment_id=0,
                size=len(df),
                size_pct=100.0,
                target_rate=None,
                defining_features={},
            )],
            target_variance_ratio=None,
            recommendation="single_model",
            confidence=0.0,
            rationale=["No numeric features available for segmentation"],
            labels=np.zeros(len(df), dtype=int),
        )

    def _single_segment_result(
        self,
        df: pd.DataFrame,
        method: SegmentationMethod,
        target_col: Optional[str],
    ) -> SegmentationResult:
        target_rate = None
        if target_col and target_col in df.columns:
            target = df[target_col]
            if np.issubdtype(target.dtype, np.number):
                target_rate = float(target.mean())

        return SegmentationResult(
            n_segments=1,
            method=method,
            quality_score=0.0,
            profiles=[SegmentProfile(
                segment_id=0,
                size=len(df),
                size_pct=100.0,
                target_rate=target_rate,
                defining_features={},
            )],
            target_variance_ratio=0.0,
            recommendation="single_model",
            confidence=0.5,
            rationale=["Insufficient data for meaningful segmentation"],
            labels=np.zeros(len(df), dtype=int),
        )

    def get_cluster_visualization(
        self,
        df: DataFrame,
        labels: np.ndarray,
        feature_cols: List[str],
        method: DimensionReductionMethod = DimensionReductionMethod.PCA,
    ) -> ClusterVisualizationResult:
        """Reduce features to 2D for cluster visualization.

        Args:
            df: DataFrame with features.
            labels: Cluster labels from analyze().
            feature_cols: Feature columns to use for dimensionality reduction.
            method: Dimensionality reduction method (PCA, TSNE, UMAP).

        Returns:
            ClusterVisualizationResult with 2D coordinates and labels.
        """
        df = to_pandas(df)
        feature_cols = [c for c in feature_cols if c in df.columns]

        # Initialize output arrays with NaN
        x = np.full(len(df), np.nan)
        y = np.full(len(df), np.nan)

        # Get valid rows (non-NaN features)
        features_df = df[feature_cols].copy()
        valid_mask = ~features_df.isna().any(axis=1)
        valid_indices = features_df[valid_mask].index

        if len(valid_indices) < 2:
            return ClusterVisualizationResult(
                x=x, y=y, labels=labels,
                method=method, explained_variance_ratio=None
            )

        # Scale features
        scaled = self._scaler.fit_transform(features_df.loc[valid_indices])

        # Apply dimensionality reduction
        explained_variance = None

        if method == DimensionReductionMethod.PCA:
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(scaled)
            explained_variance = float(sum(reducer.explained_variance_ratio_))

        elif method == DimensionReductionMethod.TSNE:
            perplexity = min(30, len(valid_indices) - 1)
            reducer = TSNE(
                n_components=2,
                perplexity=max(5, perplexity),
                random_state=42,
            )
            coords = reducer.fit_transform(scaled)

        elif method == DimensionReductionMethod.UMAP:
            try:
                import umap
                n_neighbors = min(15, len(valid_indices) - 1)
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=max(2, n_neighbors),
                    random_state=42,
                )
                coords = reducer.fit_transform(scaled)
            except ImportError:
                # Fall back to PCA if UMAP not installed
                reducer = PCA(n_components=2, random_state=42)
                coords = reducer.fit_transform(scaled)
                explained_variance = float(sum(reducer.explained_variance_ratio_))
                method = DimensionReductionMethod.PCA

        else:
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(scaled)
            explained_variance = float(sum(reducer.explained_variance_ratio_))

        # Fill in valid coordinates
        x[valid_indices] = coords[:, 0]
        y[valid_indices] = coords[:, 1]

        return ClusterVisualizationResult(
            x=x,
            y=y,
            labels=labels,
            method=method,
            explained_variance_ratio=explained_variance,
        )

    def run_full_analysis(
        self,
        df: DataFrame,
        feature_cols: List[str],
        target_col: Optional[str] = None,
        max_segments: int = 5,
        method: Optional[SegmentationMethod] = None,
        dim_reduction: DimensionReductionMethod = DimensionReductionMethod.PCA,
    ) -> FullSegmentationResult:
        """Run complete segmentation analysis for dashboard display.

        Args:
            df: DataFrame with features.
            feature_cols: Feature columns for clustering.
            target_col: Optional target column for variance analysis.
            max_segments: Maximum segments to consider.
            method: Clustering method.
            dim_reduction: Dimensionality reduction for visualization.

        Returns:
            FullSegmentationResult with metrics, profiles, and visualization.
        """
        df = to_pandas(df)

        # Run segmentation
        seg_result = self.analyze(
            df,
            target_col=target_col,
            feature_cols=feature_cols,
            max_segments=max_segments,
            method=method,
        )

        # Create decision metrics
        metrics = SegmentationDecisionMetrics.from_segmentation_result(seg_result)

        # Calculate size distribution
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

        # Get visualization if multiple segments and enough features
        visualization = None
        valid_features = [c for c in feature_cols if c in df.columns]
        if seg_result.n_segments > 1 and len(valid_features) >= 2:
            visualization = self.get_cluster_visualization(
                df,
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
