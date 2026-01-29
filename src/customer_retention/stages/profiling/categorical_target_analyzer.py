from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from customer_retention.core.compat import DataFrame, to_pandas

CARDINALITY_THRESHOLD = 0.5
MIN_CATEGORIES_FOR_ANALYSIS = 2
MAX_CATEGORIES_FOR_ANALYSIS = 50


@dataclass
class CategoricalFeatureInsight:
    feature_name: str
    cramers_v: float
    effect_strength: str
    p_value: float
    n_categories: int
    high_risk_categories: List[str]
    low_risk_categories: List[str]
    interpretation: str
    category_stats: pd.DataFrame


@dataclass
class CategoricalAnalysisResult:
    feature_insights: List[CategoricalFeatureInsight]
    filtered_columns: List[str]
    filter_reasons: Dict[str, str]
    overall_target_rate: float
    recommendations: List[Dict]
    key_findings: List[str] = field(default_factory=list)


def _validate_categorical_column(col: str, df: DataFrame, entity_column: str, target_column: str, n_entities: int, cardinality_threshold: float) -> tuple:
    if col in [entity_column, target_column]:
        return False, "entity or target column"
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return False, "datetime column"
    n_unique = df[col].nunique()
    ratio = n_unique / n_entities
    if ratio > cardinality_threshold:
        return False, f"high cardinality identifier ({n_unique} unique values, {ratio:.0%} of entities)"
    if n_unique < MIN_CATEGORIES_FOR_ANALYSIS:
        return False, f"too few categories ({n_unique})"
    if n_unique > MAX_CATEGORIES_FOR_ANALYSIS:
        return False, f"too many categories ({n_unique})"
    return True, None


def filter_categorical_columns(df: DataFrame, entity_column: str, target_column: str, cardinality_threshold: float = CARDINALITY_THRESHOLD) -> List[str]:
    n_entities = df[entity_column].nunique() if entity_column in df.columns else len(df)
    return [
        col for col in df.select_dtypes(include=["object", "category"]).columns
        if _validate_categorical_column(col, df, entity_column, target_column, n_entities, cardinality_threshold)[0]
    ]


def _get_filter_reasons(df: DataFrame, entity_column: str, target_column: str, cardinality_threshold: float = CARDINALITY_THRESHOLD) -> Dict[str, str]:
    n_entities = df[entity_column].nunique() if entity_column in df.columns else len(df)
    reasons = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        is_valid, reason = _validate_categorical_column(
            col, df, entity_column, target_column, n_entities, cardinality_threshold
        )
        if not is_valid and reason:
            reasons[col] = reason
    return reasons


def _generate_interpretation(result: "CategoricalTargetResult") -> str:
    if result.cramers_v >= 0.3:
        strength_desc = "strongly associated"
    elif result.cramers_v >= 0.1:
        strength_desc = "moderately associated"
    else:
        strength_desc = "weakly associated"
    parts = [f"{result.categorical_col} is {strength_desc} with target (V={result.cramers_v:.2f})"]
    if result.high_risk_categories:
        parts.append(f"High-risk: {', '.join(result.high_risk_categories[:3])}")
    if result.low_risk_categories:
        parts.append(f"Low-risk: {', '.join(result.low_risk_categories[:3])}")
    return ". ".join(parts)


def _generate_categorical_recommendations(insights: List[CategoricalFeatureInsight]) -> List[Dict]:
    recommendations = []
    strong = [i for i in insights if i.cramers_v >= 0.3]
    moderate = [i for i in insights if 0.1 <= i.cramers_v < 0.3]
    if strong:
        recommendations.append({
            "action": "add_categorical_features", "priority": "high",
            "reason": f"Strong predictors: {', '.join(i.feature_name for i in strong[:3])}",
            "features": [i.feature_name for i in strong]
        })
    if moderate:
        recommendations.append({
            "action": "consider_categorical_features", "priority": "medium",
            "reason": f"Moderate predictors: {', '.join(i.feature_name for i in moderate[:3])}",
            "features": [i.feature_name for i in moderate]
        })
    with_high_risk = [i for i in insights if i.high_risk_categories]
    if with_high_risk:
        recommendations.append({
            "action": "create_risk_flags", "priority": "medium",
            "reason": f"Features with high-risk segments: {', '.join(i.feature_name for i in with_high_risk[:3])}",
            "features": [f"{i.feature_name}_is_high_risk" for i in with_high_risk[:3]]
        })
    return recommendations[:3]


def analyze_categorical_features(df: DataFrame, entity_column: str, target_column: str, cardinality_threshold: float = CARDINALITY_THRESHOLD) -> CategoricalAnalysisResult:
    df = to_pandas(df)
    valid_cols = filter_categorical_columns(df, entity_column, target_column, cardinality_threshold)
    filter_reasons = _get_filter_reasons(df, entity_column, target_column, cardinality_threshold)
    filtered_cols = [c for c in filter_reasons if c not in valid_cols and c not in [entity_column, target_column]]
    overall_rate = float(df[target_column].mean()) if target_column in df.columns else 0.0
    analyzer = CategoricalTargetAnalyzer()
    insights = []
    for col in valid_cols:
        result = analyzer.analyze(df, col, target_column)
        interpretation = _generate_interpretation(result)
        insights.append(CategoricalFeatureInsight(
            feature_name=col, cramers_v=result.cramers_v, effect_strength=result.effect_strength,
            p_value=result.p_value, n_categories=result.n_categories,
            high_risk_categories=result.high_risk_categories, low_risk_categories=result.low_risk_categories,
            interpretation=interpretation, category_stats=result.category_stats
        ))
    insights.sort(key=lambda x: x.cramers_v, reverse=True)
    recommendations = _generate_categorical_recommendations(insights)
    key_findings = []
    if filtered_cols:
        key_findings.append(f"Filtered {len(filtered_cols)} columns: {', '.join(filtered_cols[:3])}{'...' if len(filtered_cols) > 3 else ''}")
    strong_count = sum(1 for i in insights if i.cramers_v >= 0.3)
    if strong_count > 0:
        key_findings.append(f"{strong_count} feature(s) strongly predict target")
    elif insights:
        key_findings.append("No categorical features strongly predict target")
    return CategoricalAnalysisResult(
        feature_insights=insights, filtered_columns=filtered_cols, filter_reasons=filter_reasons,
        overall_target_rate=overall_rate, recommendations=recommendations, key_findings=key_findings
    )


@dataclass
class CategoricalTargetResult:
    categorical_col: str
    target_col: str
    n_categories: int
    cramers_v: float
    chi2_statistic: float
    p_value: float
    effect_strength: str
    category_stats: pd.DataFrame
    high_risk_categories: List[str]
    low_risk_categories: List[str]
    overall_rate: float


class CategoricalTargetAnalyzer:
    EFFECT_THRESHOLDS = {
        'weak': 0.1,
        'moderate': 0.3,
        'strong': 0.5
    }

    HIGH_RISK_LIFT_THRESHOLD = 0.9
    LOW_RISK_LIFT_THRESHOLD = 1.1

    def __init__(self, min_samples_per_category: int = 10):
        self.min_samples_per_category = min_samples_per_category

    def analyze(self, df: DataFrame, categorical_col: str, target_col: str) -> CategoricalTargetResult:
        df = to_pandas(df)
        if len(df) == 0 or categorical_col not in df.columns or target_col not in df.columns:
            return self._empty_result(categorical_col, target_col)
        clean_df = df[[categorical_col, target_col]].dropna()
        if len(clean_df) == 0:
            return self._empty_result(categorical_col, target_col)
        overall_rate = clean_df[target_col].mean()
        category_stats = self._calculate_category_stats(clean_df, categorical_col, target_col, overall_rate)
        cramers_v, chi2_stat, p_value = self._calculate_cramers_v(clean_df, categorical_col, target_col)
        effect_strength = self._determine_effect_strength(cramers_v)
        high_risk = category_stats[category_stats['lift'] < self.HIGH_RISK_LIFT_THRESHOLD]['category'].tolist()
        low_risk = category_stats[category_stats['lift'] > self.LOW_RISK_LIFT_THRESHOLD]['category'].tolist()

        return CategoricalTargetResult(
            categorical_col=categorical_col,
            target_col=target_col,
            n_categories=len(category_stats),
            cramers_v=cramers_v,
            chi2_statistic=chi2_stat,
            p_value=p_value,
            effect_strength=effect_strength,
            category_stats=category_stats,
            high_risk_categories=high_risk,
            low_risk_categories=low_risk,
            overall_rate=overall_rate
        )

    def _calculate_category_stats(self, df: pd.DataFrame, categorical_col: str, target_col: str, overall_rate: float) -> pd.DataFrame:
        stats = df.groupby(categorical_col)[target_col].agg(['sum', 'count', 'mean']).reset_index()
        stats.columns = ['category', 'retained_count', 'total_count', 'retention_rate']
        stats['churned_count'] = stats['total_count'] - stats['retained_count']
        stats['lift'] = stats['retention_rate'] / overall_rate if overall_rate > 0 else 0
        stats['pct_of_total'] = stats['total_count'] / len(df)
        stats = stats[stats['total_count'] >= self.min_samples_per_category]
        return stats.sort_values('retention_rate', ascending=False).reset_index(drop=True)

    def _calculate_cramers_v(self, df: pd.DataFrame, categorical_col: str, target_col: str) -> tuple:
        contingency = pd.crosstab(df[categorical_col], df[target_col])

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return 0.0, 0.0, 1.0

        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1

            if min_dim == 0 or n == 0:
                return 0.0, chi2, p_value

            cramers_v = np.sqrt(chi2 / (n * min_dim))
            return float(cramers_v), float(chi2), float(p_value)
        except (ValueError, ZeroDivisionError):
            return 0.0, 0.0, 1.0

    def _determine_effect_strength(self, cramers_v: float) -> str:
        if cramers_v >= self.EFFECT_THRESHOLDS['strong']:
            return 'strong'
        elif cramers_v >= self.EFFECT_THRESHOLDS['moderate']:
            return 'moderate'
        elif cramers_v >= self.EFFECT_THRESHOLDS['weak']:
            return 'weak'
        else:
            return 'negligible'

    def _empty_result(self, categorical_col: str, target_col: str) -> CategoricalTargetResult:
        return CategoricalTargetResult(
            categorical_col=categorical_col,
            target_col=target_col,
            n_categories=0,
            cramers_v=0.0,
            chi2_statistic=0.0,
            p_value=1.0,
            effect_strength='negligible',
            category_stats=pd.DataFrame(columns=[
                'category', 'retained_count', 'total_count', 'retention_rate',
                'churned_count', 'lift', 'pct_of_total'
            ]),
            high_risk_categories=[],
            low_risk_categories=[],
            overall_rate=0.0
        )

    def analyze_multiple(self, df: DataFrame, categorical_cols: List[str], target_col: str) -> pd.DataFrame:
        results = []
        for col in categorical_cols:
            result = self.analyze(df, col, target_col)
            results.append({
                'feature': col,
                'n_categories': result.n_categories,
                'cramers_v': result.cramers_v,
                'p_value': result.p_value,
                'effect_strength': result.effect_strength,
                'high_risk_count': len(result.high_risk_categories),
                'low_risk_count': len(result.low_risk_categories)
            })

        return pd.DataFrame(results).sort_values('cramers_v', ascending=False).reset_index(drop=True)
