"""RelationshipRecommender - generates actionable recommendations from relationship analysis."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from customer_retention.core.compat import (
    batched_corr_matrix,
    bulk_corr_with_target,
    groupby_multi_agg,
    pd,
)


class RecommendationCategory(Enum):
    FEATURE_SELECTION = "feature_selection"
    FEATURE_ENGINEERING = "feature_engineering"
    STRATIFICATION = "stratification"
    MODEL_SELECTION = "model_selection"


@dataclass
class RelationshipRecommendation:
    category: RecommendationCategory
    title: str
    description: str
    action: str
    priority: str  # "high", "medium", "low"
    affected_features: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "action": self.action,
            "priority": self.priority,
            "affected_features": self.affected_features,
            "evidence": self.evidence,
        }


@dataclass
class RelationshipAnalysisSummary:
    recommendations: List[RelationshipRecommendation]
    correlation_matrix: Optional[pd.DataFrame] = None
    strong_predictors: List[Dict[str, Any]] = field(default_factory=list)
    weak_predictors: List[str] = field(default_factory=list)
    multicollinear_pairs: List[Dict[str, Any]] = field(default_factory=list)
    high_risk_segments: List[Dict[str, Any]] = field(default_factory=list)
    categorical_associations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def recommendations_by_category(self) -> Dict[RecommendationCategory, List[RelationshipRecommendation]]:
        grouped = {cat: [] for cat in RecommendationCategory}
        for rec in self.recommendations:
            grouped[rec.category].append(rec)
        return grouped

    @property
    def high_priority_actions(self) -> List[RelationshipRecommendation]:
        return [r for r in self.recommendations if r.priority == "high"]


class RelationshipRecommender:
    HIGH_CORRELATION_THRESHOLD = 0.7
    STRONG_PREDICTOR_THRESHOLD = 0.3
    WEAK_PREDICTOR_THRESHOLD = 0.1
    HIGH_RISK_LIFT_THRESHOLD = 0.85
    MIN_CATEGORY_SIZE = 10

    def analyze(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        effect_sizes: Optional[Dict[str, float]] = None,
        categorical_results: Optional[Dict[str, Any]] = None,
    ) -> RelationshipAnalysisSummary:
        numeric_cols = numeric_cols or []
        categorical_cols = categorical_cols or []
        recommendations = []

        result_correlation_matrix = None
        strong_predictors = []
        weak_predictors = []
        multicollinear_pairs = []
        high_risk_segments = []
        categorical_associations = []

        # Analyze numeric features
        if numeric_cols:
            corr_results = self._analyze_numeric_correlations(
                df, numeric_cols, target_col, correlation_matrix=correlation_matrix,
            )
            result_correlation_matrix = corr_results["correlation_matrix"]
            multicollinear_pairs = corr_results["multicollinear_pairs"]
            recommendations.extend(corr_results["recommendations"])

            if target_col:
                predictor_results = self._analyze_predictive_power(
                    df, numeric_cols, target_col,
                    correlation_matrix=correlation_matrix,
                    effect_sizes=effect_sizes,
                )
                strong_predictors = predictor_results["strong"]
                weak_predictors = predictor_results["weak"]
                recommendations.extend(predictor_results["recommendations"])

        # Analyze categorical features
        if categorical_cols and target_col:
            cat_results = self._analyze_categorical_relationships(
                df, categorical_cols, target_col, precomputed=categorical_results,
            )
            high_risk_segments = cat_results["high_risk_segments"]
            categorical_associations = cat_results["associations"]
            recommendations.extend(cat_results["recommendations"])

        # Model selection recommendations
        model_recs = self._generate_model_recommendations(
            multicollinear_pairs, strong_predictors, categorical_associations
        )
        recommendations.extend(model_recs)

        # Feature engineering recommendations
        eng_recs = self._generate_engineering_recommendations(
            df, numeric_cols, target_col, correlation_matrix=correlation_matrix,
        )
        recommendations.extend(eng_recs)

        return RelationshipAnalysisSummary(
            recommendations=recommendations,
            correlation_matrix=result_correlation_matrix,
            strong_predictors=strong_predictors,
            weak_predictors=weak_predictors,
            multicollinear_pairs=multicollinear_pairs,
            high_risk_segments=high_risk_segments,
            categorical_associations=categorical_associations,
        )

    def _analyze_numeric_correlations(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        target_col: Optional[str],
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        recommendations = []
        multicollinear_pairs = []

        cols_to_analyze = [c for c in numeric_cols if c in df.columns]
        if len(cols_to_analyze) < 2:
            return {
                "correlation_matrix": None,
                "multicollinear_pairs": [],
                "recommendations": [],
            }

        if correlation_matrix is not None:
            available = [c for c in cols_to_analyze if c in correlation_matrix.index]
            if len(available) < 2:
                raise ValueError(
                    f"Precomputed correlation_matrix covers {len(available)} of "
                    f"{len(cols_to_analyze)} requested columns — cannot analyze"
                )
            correlation_matrix = correlation_matrix.loc[available, available]
            cols_to_analyze = available
        else:
            correlation_matrix = batched_corr_matrix(df, cols_to_analyze)

        # Find multicollinear pairs
        for i, col1 in enumerate(cols_to_analyze):
            for col2 in cols_to_analyze[i + 1:]:
                if col1 == target_col or col2 == target_col:
                    continue
                corr_val = correlation_matrix.loc[col1, col2]
                if abs(corr_val) >= self.HIGH_CORRELATION_THRESHOLD:
                    multicollinear_pairs.append({
                        "feature1": col1,
                        "feature2": col2,
                        "correlation": float(corr_val),
                    })

        # Generate recommendations for multicollinearity
        if multicollinear_pairs:
            for pair in multicollinear_pairs:
                recommendations.append(RelationshipRecommendation(
                    category=RecommendationCategory.FEATURE_SELECTION,
                    title="Remove multicollinear feature",
                    description=f"{pair['feature1']} and {pair['feature2']} are highly correlated (r={pair['correlation']:.2f})",
                    action="Consider dropping one of these features. Keep the one with stronger business meaning or higher target correlation.",
                    priority="high" if abs(pair["correlation"]) >= 0.85 else "medium",
                    affected_features=[pair["feature1"], pair["feature2"]],
                    evidence={"correlation": pair["correlation"]},
                ))

        return {
            "correlation_matrix": correlation_matrix,
            "multicollinear_pairs": multicollinear_pairs,
            "recommendations": recommendations,
        }

    def _analyze_predictive_power(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        target_col: str,
        correlation_matrix: Optional[pd.DataFrame] = None,
        effect_sizes: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        recommendations = []
        strong = []
        weak = []

        if target_col not in df.columns:
            return {"strong": [], "weak": [], "recommendations": []}

        feature_cols = [c for c in numeric_cols if c != target_col and c in df.columns]
        if not feature_cols:
            return {"strong": [], "weak": [], "recommendations": []}

        if correlation_matrix is not None and target_col in correlation_matrix.index:
            available = [c for c in feature_cols if c in correlation_matrix.index]
            target_corrs = {c: float(correlation_matrix.loc[c, target_col]) for c in available}
        else:
            target_corrs = bulk_corr_with_target(df, feature_cols, target_col)
            available = list(target_corrs.keys())

        for col in available:
            corr = target_corrs.get(col, 0.0)
            effect_size = effect_sizes.get(col, 0.0) if effect_sizes else 0.0

            predictor_info = {
                "feature": col,
                "correlation": float(corr),
                "effect_size": effect_size,
            }

            if abs(effect_size) >= 0.5 or abs(corr) >= self.STRONG_PREDICTOR_THRESHOLD:
                strong.append(predictor_info)
            elif abs(effect_size) < 0.2 and abs(corr) < self.WEAK_PREDICTOR_THRESHOLD:
                weak.append(col)

        # Recommendations for strong predictors
        if strong:
            top_predictors = sorted(strong, key=lambda x: abs(x["effect_size"]), reverse=True)[:3]
            features_list = [p["feature"] for p in top_predictors]
            recommendations.append(RelationshipRecommendation(
                category=RecommendationCategory.FEATURE_SELECTION,
                title="Prioritize strong predictors",
                description=f"Top predictive features: {', '.join(features_list)}",
                action="Ensure these features are included in your model and check for data quality issues.",
                priority="high",
                affected_features=features_list,
                evidence={"predictors": top_predictors},
            ))

        # Recommendations for weak predictors
        if weak:
            recommendations.append(RelationshipRecommendation(
                category=RecommendationCategory.FEATURE_SELECTION,
                title="Consider removing weak predictors",
                description=f"Features with low predictive power: {', '.join(weak[:5])}",
                action="These features may add noise. Consider removing or combining with other features.",
                priority="low",
                affected_features=weak[:5],
                evidence={"weak_features": weak},
            ))

        return {"strong": strong, "weak": weak, "recommendations": recommendations}

    def _analyze_categorical_relationships(
        self, df: pd.DataFrame, categorical_cols: List[str], target_col: str,
        precomputed: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        recommendations = []
        high_risk_segments = []
        associations = []

        if not precomputed and target_col not in df.columns:
            return {"high_risk_segments": [], "associations": [], "recommendations": []}

        overall_rate = None
        n_rows = None

        for col in categorical_cols:
            if precomputed and col in precomputed:
                r = precomputed[col]
                cramers_v = r.cramers_v
                cs = r.category_stats
                counts_arr = cs["total_count"].to_numpy()
                rates_arr = cs["retention_rate"].to_numpy()
                lifts_arr = cs["lift"].to_numpy()
                cat_names = cs["category"].to_numpy()
            elif col in df.columns:
                if overall_rate is None:
                    overall_rate = float(df[target_col].mean())
                    n_rows = len(df)
                cs = groupby_multi_agg(df, col, target_col, ["sum", "count", "mean"])
                cs.columns = [col, "retained_count", "count", "retention_rate"]
                cs["lift"] = cs["retention_rate"] / overall_rate
                cramers_v = self._cramers_v_from_stats(cs, "retained_count", "count", n_rows)
                counts_arr = cs["count"].to_numpy()
                rates_arr = cs["retention_rate"].to_numpy()
                lifts_arr = cs["lift"].to_numpy()
                cat_names = cs[col].to_numpy()
            else:
                continue

            associations.append({"feature": col, "cramers_v": cramers_v})

            for idx in range(len(counts_arr)):
                if counts_arr[idx] >= self.MIN_CATEGORY_SIZE and lifts_arr[idx] < self.HIGH_RISK_LIFT_THRESHOLD:
                    high_risk_segments.append({
                        "feature": col, "segment": cat_names[idx],
                        "retention_rate": float(rates_arr[idx]),
                        "lift": float(lifts_arr[idx]), "count": int(counts_arr[idx]),
                    })

            min_count = float(counts_arr.min()) if len(counts_arr) > 0 else 0
            size_ratio = float(counts_arr.max()) / min_count if min_count > 0 else float("inf")
            rate_spread = float(rates_arr.max()) - float(rates_arr.min()) if len(rates_arr) > 0 else 0.0

            if rate_spread > 0.15 or size_ratio > 10:
                recommendations.append(RelationshipRecommendation(
                    category=RecommendationCategory.STRATIFICATION,
                    title=f"Stratify by {col}",
                    description=f"Significant variation in retention rates across {col} categories (spread: {rate_spread:.1%})",
                    action=f"Use stratified sampling by {col} in train/test split to ensure all segments are represented.",
                    priority="high" if rate_spread > 0.25 else "medium",
                    affected_features=[col],
                    evidence={"rate_spread": rate_spread, "size_ratio": size_ratio, "cramers_v": cramers_v},
                ))

        if high_risk_segments:
            segment_names = list(set(s["segment"] for s in high_risk_segments[:3]))
            recommendations.append(RelationshipRecommendation(
                category=RecommendationCategory.STRATIFICATION,
                title="Monitor high-risk segments",
                description=f"Segments with below-average retention: {', '.join(str(s) for s in segment_names)}",
                action="Target these segments for intervention campaigns and ensure adequate representation in training data.",
                priority="high",
                affected_features=[s["feature"] for s in high_risk_segments[:3]],
                evidence={"high_risk_segments": high_risk_segments[:5]},
            ))

        return {
            "high_risk_segments": high_risk_segments,
            "associations": associations,
            "recommendations": recommendations,
        }

    @staticmethod
    def _cramers_v_from_stats(
        cat_stats: pd.DataFrame, retained_col: str, count_col: str, n: int,
    ) -> float:
        try:
            from scipy.stats import chi2_contingency
            counts = cat_stats[count_col].to_numpy()
            retained = cat_stats[retained_col].to_numpy()
            churned = counts - retained
            contingency = np.column_stack([churned, retained])
            if contingency.shape[0] < 2 or contingency.min() < 0:
                return 0.0
            chi2, _, _, _ = chi2_contingency(contingency)
            min_dim = min(contingency.shape) - 1
            return float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0
        except Exception:
            return 0.0

    def _generate_model_recommendations(
        self,
        multicollinear_pairs: List[Dict],
        strong_predictors: List[Dict],
        categorical_associations: List[Dict],
    ) -> List[RelationshipRecommendation]:
        recommendations = []

        # Based on multicollinearity
        if multicollinear_pairs:
            recommendations.append(RelationshipRecommendation(
                category=RecommendationCategory.MODEL_SELECTION,
                title="Consider tree-based models for multicollinearity",
                description=f"Found {len(multicollinear_pairs)} highly correlated feature pairs",
                action="Tree-based models (Random Forest, XGBoost) are robust to multicollinearity. For linear models, remove redundant features first.",
                priority="medium",
                affected_features=[],
                evidence={"n_multicollinear_pairs": len(multicollinear_pairs)},
            ))

        # Based on predictor strength
        if strong_predictors:
            avg_effect = np.mean([abs(p["effect_size"]) for p in strong_predictors])
            if avg_effect >= 0.5:
                recommendations.append(RelationshipRecommendation(
                    category=RecommendationCategory.MODEL_SELECTION,
                    title="Linear models may perform well",
                    description=f"Strong linear relationships detected (avg effect size: {avg_effect:.2f})",
                    action="Start with Logistic Regression as baseline. Clear feature-target relationships suggest interpretable models may work well.",
                    priority="medium",
                    affected_features=[p["feature"] for p in strong_predictors],
                    evidence={"avg_effect_size": avg_effect},
                ))
            else:
                recommendations.append(RelationshipRecommendation(
                    category=RecommendationCategory.MODEL_SELECTION,
                    title="Non-linear models may improve performance",
                    description="Moderate effect sizes suggest potential non-linear relationships",
                    action="Try ensemble methods (Random Forest, Gradient Boosting) to capture non-linear patterns and interactions.",
                    priority="medium",
                    affected_features=[],
                    evidence={"avg_effect_size": avg_effect},
                ))
        else:
            recommendations.append(RelationshipRecommendation(
                category=RecommendationCategory.MODEL_SELECTION,
                title="Explore ensemble methods",
                description="No strong linear predictors identified",
                action="Use tree-based ensembles to discover non-linear patterns and feature interactions.",
                priority="medium",
                affected_features=[],
                evidence={},
            ))

        # Based on categorical strength
        strong_cats = [a for a in categorical_associations if a.get("cramers_v", 0) >= 0.2]
        if strong_cats:
            features = [a["feature"] for a in strong_cats]
            recommendations.append(RelationshipRecommendation(
                category=RecommendationCategory.MODEL_SELECTION,
                title="Categorical features are predictive",
                description=f"Strong categorical associations: {', '.join(features)}",
                action="Use target encoding for tree-based models or one-hot encoding for linear models. Consider CatBoost for native categorical handling.",
                priority="medium",
                affected_features=features,
                evidence={"strong_categorical": strong_cats},
            ))

        return recommendations

    def _generate_engineering_recommendations(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        target_col: Optional[str],
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> List[RelationshipRecommendation]:
        recommendations = []

        if not numeric_cols or len(numeric_cols) < 2:
            return recommendations

        # Suggest ratio features for correlated pairs
        cols_in_df = [c for c in numeric_cols if c in df.columns and c != target_col]
        if len(cols_in_df) >= 2:
            if correlation_matrix is not None:
                available = [c for c in cols_in_df if c in correlation_matrix.index]
                if len(available) < 2:
                    raise ValueError(
                        f"Precomputed correlation_matrix covers {len(available)} of "
                        f"{len(cols_in_df)} requested columns — cannot analyze"
                    )
                corr_matrix = correlation_matrix.loc[available, available]
                cols_in_df = available
            else:
                corr_matrix = batched_corr_matrix(df, cols_in_df)
            moderate_pairs = []

            for i, col1 in enumerate(cols_in_df):
                for col2 in cols_in_df[i + 1:]:
                    corr = corr_matrix.loc[col1, col2]
                    if 0.3 <= abs(corr) < 0.7:
                        moderate_pairs.append((col1, col2, corr))

            if moderate_pairs:
                pair_strs = [f"{p[0]}/{p[1]}" for p in moderate_pairs[:3]]
                recommendations.append(RelationshipRecommendation(
                    category=RecommendationCategory.FEATURE_ENGINEERING,
                    title="Consider ratio features",
                    description=f"Moderately correlated pairs may benefit from ratio features: {', '.join(pair_strs)}",
                    action="Create ratio features (e.g., feature_a / feature_b) to capture relative relationships.",
                    priority="low",
                    affected_features=[p[0] for p in moderate_pairs[:3]] + [p[1] for p in moderate_pairs[:3]],
                    evidence={"moderate_pairs": moderate_pairs[:3]},
                ))

        # General interaction recommendation
        if len(cols_in_df) >= 2:
            recommendations.append(RelationshipRecommendation(
                category=RecommendationCategory.FEATURE_ENGINEERING,
                title="Test feature interactions",
                description="Interaction terms may capture non-linear relationships",
                action="Use PolynomialFeatures(interaction_only=True) or tree-based models which automatically discover interactions.",
                priority="low",
                affected_features=cols_in_df[:4],
                evidence={},
            ))

        return recommendations
