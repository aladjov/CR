"""RelationshipRecommender - generates actionable recommendations from relationship analysis."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import numpy as np

from customer_retention.core.compat import pd


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
    ) -> RelationshipAnalysisSummary:
        numeric_cols = numeric_cols or []
        categorical_cols = categorical_cols or []
        recommendations = []

        correlation_matrix = None
        strong_predictors = []
        weak_predictors = []
        multicollinear_pairs = []
        high_risk_segments = []
        categorical_associations = []

        # Analyze numeric features
        if numeric_cols:
            corr_results = self._analyze_numeric_correlations(df, numeric_cols, target_col)
            correlation_matrix = corr_results["correlation_matrix"]
            multicollinear_pairs = corr_results["multicollinear_pairs"]
            recommendations.extend(corr_results["recommendations"])

            if target_col:
                predictor_results = self._analyze_predictive_power(df, numeric_cols, target_col)
                strong_predictors = predictor_results["strong"]
                weak_predictors = predictor_results["weak"]
                recommendations.extend(predictor_results["recommendations"])

        # Analyze categorical features
        if categorical_cols and target_col:
            cat_results = self._analyze_categorical_relationships(df, categorical_cols, target_col)
            high_risk_segments = cat_results["high_risk_segments"]
            categorical_associations = cat_results["associations"]
            recommendations.extend(cat_results["recommendations"])

        # Model selection recommendations
        model_recs = self._generate_model_recommendations(
            multicollinear_pairs, strong_predictors, categorical_associations
        )
        recommendations.extend(model_recs)

        # Feature engineering recommendations
        eng_recs = self._generate_engineering_recommendations(df, numeric_cols, target_col)
        recommendations.extend(eng_recs)

        return RelationshipAnalysisSummary(
            recommendations=recommendations,
            correlation_matrix=correlation_matrix,
            strong_predictors=strong_predictors,
            weak_predictors=weak_predictors,
            multicollinear_pairs=multicollinear_pairs,
            high_risk_segments=high_risk_segments,
            categorical_associations=categorical_associations,
        )

    def _analyze_numeric_correlations(
        self, df: pd.DataFrame, numeric_cols: List[str], target_col: Optional[str]
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

        correlation_matrix = df[cols_to_analyze].corr()

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
                    action=f"Consider dropping one of these features. Keep the one with stronger business meaning or higher target correlation.",
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
        self, df: pd.DataFrame, numeric_cols: List[str], target_col: str
    ) -> Dict[str, Any]:
        recommendations = []
        strong = []
        weak = []

        if target_col not in df.columns:
            return {"strong": [], "weak": [], "recommendations": []}

        target = df[target_col]

        for col in numeric_cols:
            if col == target_col or col not in df.columns:
                continue

            corr = df[[col, target_col]].corr().iloc[0, 1]
            effect_size = self._calculate_effect_size(df, col, target_col)

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

    def _calculate_effect_size(self, df: pd.DataFrame, col: str, target_col: str) -> float:
        """Calculate Cohen's d effect size."""
        group0 = df[df[target_col] == 0][col].dropna()
        group1 = df[df[target_col] == 1][col].dropna()

        if len(group0) < 2 or len(group1) < 2:
            return 0.0

        pooled_std = np.sqrt(
            ((len(group0) - 1) * group0.std() ** 2 + (len(group1) - 1) * group1.std() ** 2)
            / (len(group0) + len(group1) - 2)
        )

        if pooled_std == 0:
            return 0.0

        return float((group1.mean() - group0.mean()) / pooled_std)

    def _analyze_categorical_relationships(
        self, df: pd.DataFrame, categorical_cols: List[str], target_col: str
    ) -> Dict[str, Any]:
        recommendations = []
        high_risk_segments = []
        associations = []

        if target_col not in df.columns:
            return {"high_risk_segments": [], "associations": [], "recommendations": []}

        overall_rate = df[target_col].mean()

        for col in categorical_cols:
            if col not in df.columns:
                continue

            # Calculate retention rates by category
            cat_stats = df.groupby(col)[target_col].agg(["mean", "count"]).reset_index()
            cat_stats.columns = [col, "retention_rate", "count"]
            cat_stats["lift"] = cat_stats["retention_rate"] / overall_rate

            # Calculate Cramér's V
            cramers_v = self._calculate_cramers_v(df, col, target_col)
            associations.append({"feature": col, "cramers_v": cramers_v})

            # Identify high-risk segments
            for _, row in cat_stats.iterrows():
                if row["count"] >= self.MIN_CATEGORY_SIZE and row["lift"] < self.HIGH_RISK_LIFT_THRESHOLD:
                    high_risk_segments.append({
                        "feature": col,
                        "segment": row[col],
                        "retention_rate": float(row["retention_rate"]),
                        "lift": float(row["lift"]),
                        "count": int(row["count"]),
                    })

            # Check if category sizes are imbalanced
            size_ratio = cat_stats["count"].max() / cat_stats["count"].min() if cat_stats["count"].min() > 0 else float("inf")
            rate_spread = cat_stats["retention_rate"].max() - cat_stats["retention_rate"].min()

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

        # High risk segment recommendations
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

    def _calculate_cramers_v(self, df: pd.DataFrame, col: str, target_col: str) -> float:
        """Calculate Cramér's V for categorical association."""
        try:
            from scipy.stats import chi2_contingency
            contingency = pd.crosstab(df[col], df[target_col])
            chi2, _, _, _ = chi2_contingency(contingency)
            n = len(df)
            min_dim = min(contingency.shape) - 1
            if min_dim == 0:
                return 0.0
            return float(np.sqrt(chi2 / (n * min_dim)))
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
    ) -> List[RelationshipRecommendation]:
        recommendations = []

        if not numeric_cols or len(numeric_cols) < 2:
            return recommendations

        # Suggest ratio features for correlated pairs
        cols_in_df = [c for c in numeric_cols if c in df.columns and c != target_col]
        if len(cols_in_df) >= 2:
            # Check for potential ratio/interaction features
            corr_matrix = df[cols_in_df].corr()
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
