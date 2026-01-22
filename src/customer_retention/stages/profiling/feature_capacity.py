"""FeatureCapacityAnalyzer - estimates favorable feature-to-data ratios for modeling.

Key Concepts:
    Events Per Variable (EPV): Minimum minority class samples per feature
        - Conservative (EPV=20): Very stable, recommended for regulatory/high-stakes
        - Moderate (EPV=10): Standard practice, widely used in literature
        - Aggressive (EPV=5): With strong regularization, acceptable for exploration

    Effective Features: Features that contribute independent information
        - Highly correlated features (r > 0.8) count as ~1 effective feature
        - Use eigenvalue analysis to estimate true dimensionality

Guidelines Based on Common Statistical Practice:
    - Harrell (2015): EPV >= 10-20 for logistic regression
    - Peduzzi et al. (1996): EPV >= 10 to avoid coefficient bias
    - Tree models: More flexible, but still benefit from adequate data
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from customer_retention.core.compat import pd


@dataclass
class EffectiveFeaturesResult:
    """Results from effective feature analysis."""
    total_count: int
    effective_count: float
    redundant_features: List[str]
    representative_features: List[str]
    feature_clusters: List[List[str]]
    correlation_matrix: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_count": self.total_count,
            "effective_count": self.effective_count,
            "redundant_features": self.redundant_features,
            "representative_features": self.representative_features,
            "n_clusters": len(self.feature_clusters),
        }


@dataclass
class ModelComplexityGuidance:
    """Guidance on model complexity given data constraints."""
    max_features_linear: int
    max_features_tree: int
    max_features_regularized: int
    recommended_model_type: str
    model_recommendations: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_features_linear": self.max_features_linear,
            "max_features_tree": self.max_features_tree,
            "max_features_regularized": self.max_features_regularized,
            "recommended_model_type": self.recommended_model_type,
            "model_recommendations": self.model_recommendations,
            "recommendations": self.recommendations,
        }


@dataclass
class FeatureCapacityResult:
    """Results from feature capacity analysis."""
    total_samples: int
    minority_class_samples: int
    total_features: int
    effective_features: float
    recommended_features_conservative: int
    recommended_features_moderate: int
    recommended_features_aggressive: int
    events_per_variable: float
    samples_per_feature: float
    capacity_status: str  # "adequate", "limited", "inadequate"
    recommendations: List[str]
    effective_features_result: Optional[EffectiveFeaturesResult] = None
    complexity_guidance: Optional[ModelComplexityGuidance] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "minority_class_samples": self.minority_class_samples,
            "total_features": self.total_features,
            "effective_features": self.effective_features,
            "recommended_features_conservative": self.recommended_features_conservative,
            "recommended_features_moderate": self.recommended_features_moderate,
            "recommended_features_aggressive": self.recommended_features_aggressive,
            "events_per_variable": self.events_per_variable,
            "samples_per_feature": self.samples_per_feature,
            "capacity_status": self.capacity_status,
            "recommendations": self.recommendations,
        }


@dataclass
class SegmentCapacityResult:
    """Results from segment-level capacity analysis."""
    segment_capacities: Dict[str, FeatureCapacityResult]
    recommended_strategy: str  # "single_model", "segment_models", "hybrid"
    strategy_reason: str
    viable_segments: List[str]
    insufficient_segments: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_strategy": self.recommended_strategy,
            "strategy_reason": self.strategy_reason,
            "viable_segments": self.viable_segments,
            "insufficient_segments": self.insufficient_segments,
            "recommendations": self.recommendations,
            "segment_details": {k: v.to_dict() for k, v in self.segment_capacities.items()},
        }


class FeatureCapacityAnalyzer:
    """Analyzes feature capacity relative to available data.

    Determines how many features can be reliably used given:
    - Total sample size
    - Minority class events (for classification)
    - Feature correlation structure
    - Model type assumptions

    Key Assumptions:
        1. EPV (Events Per Variable) of 10-20 needed for stable logistic regression
        2. Tree models are more flexible but still benefit from adequate data
        3. Highly correlated features provide redundant information
        4. Regularization allows using more features with less data
    """

    EPV_CONSERVATIVE = 20  # Very stable, regulatory-grade
    EPV_MODERATE = 10      # Standard practice
    EPV_AGGRESSIVE = 5     # With strong regularization
    CORRELATION_THRESHOLD = 0.8  # Features above this are considered redundant
    MIN_SEGMENT_EVENTS = 50  # Minimum events for viable segment model

    def analyze(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> FeatureCapacityResult:
        """Analyze feature capacity for a dataset."""
        n_samples = len(df)
        n_features = len(feature_cols)

        # Calculate minority class
        target = df[target_col]
        class_counts = target.value_counts()
        minority_samples = int(class_counts.min())

        # Calculate EPV
        epv = minority_samples / n_features if n_features > 0 else 0
        samples_per_feature = n_samples / n_features if n_features > 0 else 0

        # Calculate effective features
        eff_result = self.calculate_effective_features(df, feature_cols)
        effective_features = eff_result.effective_count

        # Recommended feature counts
        rec_conservative = int(minority_samples / self.EPV_CONSERVATIVE)
        rec_moderate = int(minority_samples / self.EPV_MODERATE)
        rec_aggressive = int(minority_samples / self.EPV_AGGRESSIVE)

        # Determine capacity status
        capacity_status = self._determine_capacity_status(epv, n_features, effective_features)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            epv, n_features, effective_features, minority_samples, capacity_status
        )

        # Get complexity guidance
        complexity_guidance = self.get_complexity_guidance(n_samples, minority_samples, n_features)

        return FeatureCapacityResult(
            total_samples=n_samples,
            minority_class_samples=minority_samples,
            total_features=n_features,
            effective_features=effective_features,
            recommended_features_conservative=rec_conservative,
            recommended_features_moderate=rec_moderate,
            recommended_features_aggressive=rec_aggressive,
            events_per_variable=epv,
            samples_per_feature=samples_per_feature,
            capacity_status=capacity_status,
            recommendations=recommendations,
            effective_features_result=eff_result,
            complexity_guidance=complexity_guidance,
        )

    def calculate_effective_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> EffectiveFeaturesResult:
        """Calculate effective number of independent features.

        Uses correlation analysis and eigenvalue decomposition to estimate
        the true dimensionality of the feature space.
        """
        valid_cols = [c for c in feature_cols if c in df.columns]
        if len(valid_cols) < 2:
            return EffectiveFeaturesResult(
                total_count=len(valid_cols),
                effective_count=float(len(valid_cols)),
                redundant_features=[],
                representative_features=valid_cols,
                feature_clusters=[valid_cols] if valid_cols else [],
            )

        # Calculate correlation matrix
        corr_matrix = df[valid_cols].corr()

        # Find redundant features (highly correlated pairs)
        redundant = set()
        feature_clusters = []
        processed = set()

        for i, col1 in enumerate(valid_cols):
            if col1 in processed:
                continue

            cluster = [col1]
            for col2 in valid_cols[i + 1:]:
                if col2 in processed:
                    continue
                corr_val = abs(corr_matrix.loc[col1, col2])
                if corr_val >= self.CORRELATION_THRESHOLD:
                    cluster.append(col2)
                    redundant.add(col2)

            if len(cluster) > 1:
                feature_clusters.append(cluster)
                processed.update(cluster)

        # Representative features: one from each cluster + unclustered
        representative = []
        clustered = set()
        for cluster in feature_clusters:
            representative.append(cluster[0])  # First feature represents cluster
            clustered.update(cluster)

        # Add unclustered features
        for col in valid_cols:
            if col not in clustered:
                representative.append(col)

        # Estimate effective features using eigenvalue analysis
        try:
            eigenvalues = np.linalg.eigvalsh(corr_matrix.values)
            eigenvalues = np.sort(eigenvalues)[::-1]
            # Effective dimensionality: count eigenvalues > 1 (Kaiser criterion)
            # or use cumulative variance explained
            total_var = eigenvalues.sum()
            cumsum = np.cumsum(eigenvalues)
            # Count eigenvalues needed for 95% variance
            effective_count = float(np.searchsorted(cumsum, 0.95 * total_var) + 1)
            effective_count = min(effective_count, len(valid_cols))
        except Exception:
            effective_count = float(len(representative))

        return EffectiveFeaturesResult(
            total_count=len(valid_cols),
            effective_count=effective_count,
            redundant_features=list(redundant),
            representative_features=representative,
            feature_clusters=feature_clusters,
            correlation_matrix=corr_matrix,
        )

    def analyze_segment_capacity(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        segment_col: str,
    ) -> SegmentCapacityResult:
        """Analyze feature capacity for each segment to guide segmented modeling."""
        segment_capacities = {}
        viable_segments = []
        insufficient_segments = []

        for segment_value in df[segment_col].unique():
            segment_df = df[df[segment_col] == segment_value]
            capacity = self.analyze(segment_df, feature_cols, target_col)
            segment_capacities[str(segment_value)] = capacity

            if capacity.capacity_status == "adequate":
                viable_segments.append(str(segment_value))
            else:
                insufficient_segments.append(str(segment_value))

        # Determine recommended strategy
        strategy, reason = self._determine_segment_strategy(
            segment_capacities, viable_segments, insufficient_segments
        )

        # Generate recommendations
        recommendations = self._generate_segment_recommendations(
            segment_capacities, viable_segments, insufficient_segments, strategy
        )

        return SegmentCapacityResult(
            segment_capacities=segment_capacities,
            recommended_strategy=strategy,
            strategy_reason=reason,
            viable_segments=viable_segments,
            insufficient_segments=insufficient_segments,
            recommendations=recommendations,
        )

    def get_complexity_guidance(
        self,
        n_samples: int,
        n_minority: int,
        n_features: int,
    ) -> ModelComplexityGuidance:
        """Provide model complexity guidance based on data constraints."""
        # Linear models: strict EPV requirements
        max_linear = int(n_minority / self.EPV_MODERATE)

        # Tree models: more flexible, use samples_per_feature
        # Rule of thumb: at least 20-30 samples per leaf, which loosely translates
        max_tree = int(n_samples / 30)

        # Regularized models: can use EPV=5 with strong regularization
        max_regularized = int(n_minority / self.EPV_AGGRESSIVE)

        # Determine recommended model type
        if n_minority < 50:
            recommended = "simple_linear"
            model_recs = [
                "Very limited events - use simple logistic regression with 1-3 features",
                "Consider exact logistic regression for very small samples",
                "Cross-validation may be unreliable - use bootstrap or leave-one-out",
            ]
        elif n_features <= max_linear:
            recommended = "linear"
            model_recs = [
                "Adequate data for standard logistic regression",
                "Can use all features without regularization",
                "Consider tree models for comparison",
            ]
        elif n_features <= max_regularized:
            recommended = "regularized_linear"
            model_recs = [
                "Use L1/L2 regularization (Lasso, Ridge, Elastic Net)",
                "Cross-validate regularization strength",
                "Tree-based models are also well-suited",
            ]
        else:
            recommended = "tree_ensemble"
            model_recs = [
                "Feature count exceeds linear model capacity",
                "Use Random Forest, XGBoost, or LightGBM",
                "Consider feature selection before linear models",
            ]

        # General recommendations
        recommendations = []
        epv = n_minority / n_features if n_features > 0 else float("inf")

        if epv < 5:
            recommendations.append(
                f"Critical: EPV={epv:.1f} is very low. Reduce features to {max_linear} or fewer."
            )
        elif epv < 10:
            recommendations.append(
                f"Limited: EPV={epv:.1f}. Use regularization or reduce to {max_linear} features."
            )
        elif epv < 20:
            recommendations.append(
                f"Moderate: EPV={epv:.1f}. Adequate for regularized models."
            )
        else:
            recommendations.append(
                f"Adequate: EPV={epv:.1f}. Sufficient data for robust modeling."
            )

        return ModelComplexityGuidance(
            max_features_linear=max_linear,
            max_features_tree=max_tree,
            max_features_regularized=max_regularized,
            recommended_model_type=recommended,
            model_recommendations=model_recs,
            recommendations=recommendations,
        )

    def _determine_capacity_status(
        self,
        epv: float,
        n_features: int,
        effective_features: float,
    ) -> str:
        """Determine overall capacity status."""
        if epv >= self.EPV_MODERATE:
            return "adequate"
        elif epv >= self.EPV_AGGRESSIVE:
            return "limited"
        else:
            return "inadequate"

    def _generate_recommendations(
        self,
        epv: float,
        n_features: int,
        effective_features: float,
        minority_samples: int,
        capacity_status: str,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # EPV-based recommendations
        if epv < self.EPV_AGGRESSIVE:
            max_features = int(minority_samples / self.EPV_MODERATE)
            recommendations.append(
                f"⚠️ EPV={epv:.1f} is below minimum. Reduce to {max_features} features or collect more data."
            )
        elif epv < self.EPV_MODERATE:
            recommendations.append(
                f"EPV={epv:.1f} is limited. Use strong regularization (L1/Lasso)."
            )

        # Effective features recommendation
        if effective_features < n_features * 0.7:
            redundant = n_features - int(effective_features)
            recommendations.append(
                f"~{redundant} features are redundant due to high correlation. Consider removing."
            )

        # Model selection guidance
        if capacity_status == "inadequate":
            recommendations.append(
                "Consider: (1) Feature selection, (2) PCA for dimensionality reduction, "
                "(3) Collecting more data, (4) Simple 2-3 feature model."
            )
        elif capacity_status == "limited":
            recommendations.append(
                "Use regularized models (Lasso, Ridge) or tree ensembles."
            )

        return recommendations

    def _determine_segment_strategy(
        self,
        capacities: Dict[str, FeatureCapacityResult],
        viable: List[str],
        insufficient: List[str],
    ) -> Tuple[str, str]:
        """Determine recommended segmentation strategy."""
        n_viable = len(viable)
        n_insufficient = len(insufficient)
        n_total = n_viable + n_insufficient

        if n_insufficient == 0:
            return "segment_models", "All segments have adequate data for separate models."
        elif n_viable == 0:
            return "single_model", "No segments have adequate data for separate models."
        elif n_viable >= n_total * 0.5:
            return "hybrid", f"{n_viable}/{n_total} segments viable. Use segment models for large segments, pooled model for small."
        else:
            return "single_model", f"Only {n_viable}/{n_total} segments have adequate data. Single model recommended."

    def _generate_segment_recommendations(
        self,
        capacities: Dict[str, FeatureCapacityResult],
        viable: List[str],
        insufficient: List[str],
        strategy: str,
    ) -> List[str]:
        """Generate segment-specific recommendations."""
        recommendations = []

        if strategy == "segment_models":
            recommendations.append(
                f"✅ All {len(viable)} segments have sufficient data for independent models."
            )
            recommendations.append(
                "Consider: Separate models may capture segment-specific patterns better."
            )
        elif strategy == "hybrid":
            recommendations.append(
                f"Build separate models for: {', '.join(viable)}"
            )
            recommendations.append(
                f"Pool small segments into single model: {', '.join(insufficient)}"
            )
        else:
            if insufficient:
                events = [capacities[s].minority_class_samples for s in insufficient]
                recommendations.append(
                    f"Small segments ({', '.join(insufficient)}) have {sum(events)} total events."
                )
            recommendations.append(
                "Use a single model with segment as a feature for stratification."
            )

        return recommendations
